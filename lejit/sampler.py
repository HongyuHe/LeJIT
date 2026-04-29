from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from lejit.codecs import decode_atomic_token, encode_atomic_token, parse_numeric_value
from lejit.config import DecodeConfig
from lejit.constraints import ConstraintProgram
from lejit.schema import EncodingKind, SchemaArtifact
from lejit.tokenizer import SPECIAL_TOKENS, RowSerializer, TokenVocabulary, field_token


@dataclass(slots=True)
class DecoderState:
    assignment: dict[str, Any] = field(default_factory=dict)
    field_index: int = 0
    phase: str = "field_token"
    value_buffer: str = ""
    pending_value: Any | None = None


class StepwiseConstrainedSampler:
    def __init__(
        self,
        model: Any,
        schema: SchemaArtifact,
        vocab: TokenVocabulary,
        serializer: RowSerializer,
        constraints: ConstraintProgram,
        decode_config: DecodeConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.schema = schema
        self.vocab = vocab
        self.serializer = serializer
        self.constraints = constraints
        self.decode_config = decode_config
        self.device = torch.device(device)
        self.model.eval()

    def generate_rows(self, n_rows: int) -> pd.DataFrame:
        rows = [self._generate_one({}, []) for _ in range(n_rows)]
        return pd.DataFrame(rows, columns=self.schema.field_order)

    def complete_rows(
        self,
        prompts: pd.DataFrame,
        samples_per_prompt: int = 1,
    ) -> pd.DataFrame:
        rows = []
        prompt_columns = prompts.columns.tolist()
        for _, prompt in prompts.iterrows():
            prompt_dict = prompt.to_dict()
            for _ in range(samples_per_prompt):
                rows.append(self._generate_one(prompt_dict, prompt_columns))
        return pd.DataFrame(rows, columns=self.schema.field_order)

    def _generate_one(self, prompt: dict[str, Any], prompt_columns: list[str]) -> dict[str, Any]:
        tokens = self.serializer.serialize_prompt(prompt, prompt_columns)
        state = self._prime_state(prompt)
        while state.phase != "done":
            allowed_ids = self._allowed_token_ids(state)
            if not allowed_ids:
                raise RuntimeError(
                    f"No feasible tokens remain for field {self._current_field_name(state)}."
                )
            if len(allowed_ids) == 1:
                token_id = allowed_ids[0]
            else:
                token_id = self._sample_token(tokens, allowed_ids)
            token = self.vocab.id_to_token[token_id]
            tokens.append(token)
            self._advance_state(state, token)
        return self.serializer.decode_row_tokens(tokens)

    def _prime_state(self, prompt: dict[str, Any]) -> DecoderState:
        state = DecoderState()
        for field_name in prompt:
            value = prompt[field_name]
            if not self.constraints.is_value_feasible(state.assignment, field_name, value):
                raise RuntimeError(f"Prompt value is infeasible for {field_name}: {value}")
            state.assignment[field_name] = value
            state.field_index += 1
        state.phase = "eos" if state.field_index == len(self.schema.field_order) else "field_token"
        return state

    def _allowed_token_ids(self, state: DecoderState) -> list[int]:
        if state.phase == "done":
            return []
        if state.phase == "eos":
            return [self.vocab.token_to_id[SPECIAL_TOKENS["eos"]]]
        if state.phase == "field_token":
            return [self.vocab.token_to_id[field_token(self._current_field_name(state))]]
        if state.phase == "end_field":
            return [self.vocab.token_to_id[SPECIAL_TOKENS["end_field"]]]
        field_name = self._current_field_name(state)
        field = self.schema.field(field_name)
        if field.encoding_kind == EncodingKind.CATEGORICAL:
            return self._allowed_categorical_ids(state, field_name, field.domain or [])
        return self._allowed_numeric_ids(state, field_name)

    def _allowed_categorical_ids(
        self,
        state: DecoderState,
        field_name: str,
        domain: list[Any],
    ) -> list[int]:
        token_ids = []
        for value in domain:
            if self.constraints.is_value_feasible(state.assignment, field_name, value):
                token_ids.append(self.vocab.token_to_id[encode_atomic_token(field_name, value)])
        return token_ids

    def _allowed_numeric_ids(self, state: DecoderState, field_name: str) -> list[int]:
        field = self.schema.field(field_name)
        codec = field.numeric_codec
        assert codec is not None
        candidates = []
        next_index = len(state.value_buffer)
        for token in self._numeric_candidates(codec, next_index):
            proposal = state.value_buffer + token
            if len(proposal) == codec.rendered_length:
                value = parse_numeric_value(proposal, codec)
                if self.constraints.is_value_feasible(state.assignment, field_name, value):
                    candidates.append(self.vocab.token_to_id[token])
            elif self.constraints.is_numeric_prefix_feasible(
                state.assignment,
                field_name,
                proposal,
            ):
                candidates.append(self.vocab.token_to_id[token])
        return candidates

    def _numeric_candidates(self, codec: Any, index: int) -> list[str]:
        if codec.signed and index == 0:
            return ["+", "-"]
        decimal_index = codec.integer_digits + int(codec.signed)
        if codec.fraction_digits > 0 and index == decimal_index:
            return ["."]
        return [str(value) for value in range(10)]

    def _sample_token(self, tokens: list[str], allowed_ids: list[int]) -> int:
        input_ids = torch.tensor([self.vocab.encode(tokens)], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        filtered = torch.full_like(logits, -torch.inf)
        filtered[:, allowed_ids] = logits[:, allowed_ids]
        filtered = filtered / max(self.decode_config.temperature, 1e-5)
        if self.decode_config.top_k is not None and self.decode_config.top_k > 0:
            values, _ = torch.topk(filtered, min(self.decode_config.top_k, filtered.shape[-1]))
            threshold = values[:, -1].unsqueeze(-1)
            filtered = torch.where(
                filtered < threshold,
                torch.full_like(filtered, -torch.inf),
                filtered,
            )
        if self.decode_config.top_p is not None and 0.0 < self.decode_config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > self.decode_config.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            filtered = torch.full_like(filtered, -torch.inf)
            filtered.scatter_(1, sorted_indices, sorted_logits)
        probs = torch.softmax(filtered, dim=-1)
        if self.decode_config.do_sample:
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(probs, dim=-1).item())

    def _advance_state(self, state: DecoderState, token: str) -> None:
        if state.phase == "field_token":
            state.phase = "value"
            return
        if state.phase == "value":
            field_name = self._current_field_name(state)
            field = self.schema.field(field_name)
            if field.encoding_kind == EncodingKind.CATEGORICAL:
                state.pending_value = decode_atomic_token(token)
                state.phase = "end_field"
                return
            state.value_buffer += token
            codec = field.numeric_codec
            assert codec is not None
            if len(state.value_buffer) == codec.rendered_length:
                state.pending_value = parse_numeric_value(state.value_buffer, codec)
                state.phase = "end_field"
            return
        if state.phase == "end_field":
            field_name = self._current_field_name(state)
            assert state.pending_value is not None
            state.assignment[field_name] = state.pending_value
            state.field_index += 1
            state.value_buffer = ""
            state.pending_value = None
            if state.field_index == len(self.schema.field_order):
                state.phase = "eos"
            else:
                state.phase = "field_token"
            return
        if state.phase == "eos":
            state.phase = "done"

    def _current_field_name(self, state: DecoderState) -> str:
        return self.schema.field_order[state.field_index]
