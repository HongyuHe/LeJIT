from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lejit.codecs import (
    decode_atomic_token,
    encode_atomic_token,
    format_numeric_value,
    parse_numeric_value,
)
from lejit.exceptions import UnsupportedPromptError
from lejit.schema import EncodingKind, SchemaArtifact

SPECIAL_TOKENS = {
    "bos": "<BOS>",
    "eos": "<EOS>",
    "end_field": "<END_FIELD>",
}


@dataclass(slots=True)
class TokenVocabulary:
    id_to_token: list[str]

    @classmethod
    def from_schema(cls, schema: SchemaArtifact) -> TokenVocabulary:
        tokens = [
            SPECIAL_TOKENS["bos"],
            SPECIAL_TOKENS["eos"],
            SPECIAL_TOKENS["end_field"],
        ]
        tokens.extend(field_token(name) for name in schema.field_order)
        for field_name in schema.field_order:
            field = schema.field(field_name)
            if field.encoding_kind == EncodingKind.CATEGORICAL:
                for value in field.domain or []:
                    tokens.append(encode_atomic_token(field_name, value))
            else:
                tokens.extend(str(value) for value in range(10))
                if field.numeric_codec and field.numeric_codec.fraction_digits > 0:
                    tokens.append(".")
                if field.numeric_codec and field.numeric_codec.signed:
                    tokens.extend(["+", "-"])
        deduped = list(dict.fromkeys(tokens))
        return cls(id_to_token=deduped)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.id_to_token)}

    def encode(self, tokens: list[str]) -> list[int]:
        table = self.token_to_id
        return [table[token] for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[index] for index in ids]


def field_token(field_name: str) -> str:
    return f"<FIELD::{field_name}>"


@dataclass(slots=True)
class RowSerializer:
    schema: SchemaArtifact
    vocab: TokenVocabulary

    def serialize_row(self, row: Mapping[str, Any]) -> list[str]:
        tokens = [SPECIAL_TOKENS["bos"]]
        for field_name in self.schema.field_order:
            tokens.append(field_token(field_name))
            tokens.extend(self.encode_field_value(field_name, row[field_name]))
            tokens.append(SPECIAL_TOKENS["end_field"])
        tokens.append(SPECIAL_TOKENS["eos"])
        return tokens

    def serialize_prompt(self, row: Mapping[str, Any], prompt_columns: list[str]) -> list[str]:
        expected = self.schema.field_order[: len(prompt_columns)]
        if prompt_columns != expected:
            raise UnsupportedPromptError(
                "Prompt columns must be a strict schema prefix. "
                f"Expected {expected}, got {prompt_columns}."
            )
        tokens = [SPECIAL_TOKENS["bos"]]
        for field_name in prompt_columns:
            tokens.append(field_token(field_name))
            tokens.extend(self.encode_field_value(field_name, row[field_name]))
            tokens.append(SPECIAL_TOKENS["end_field"])
        return tokens

    def encode_field_value(self, field_name: str, value: Any) -> list[str]:
        field = self.schema.field(field_name)
        if field.encoding_kind == EncodingKind.CATEGORICAL:
            return [encode_atomic_token(field_name, value)]
        codec = field.numeric_codec
        assert codec is not None
        return list(format_numeric_value(value, codec))

    def decode_row_tokens(self, tokens: list[str]) -> dict[str, Any]:
        cursor = 0
        if tokens[cursor] != SPECIAL_TOKENS["bos"]:
            raise ValueError("Sequence does not start with <BOS>.")
        cursor += 1
        row: dict[str, Any] = {}
        for field_name in self.schema.field_order:
            expected_field_token = field_token(field_name)
            if tokens[cursor] != expected_field_token:
                raise ValueError(f"Expected {expected_field_token}, got {tokens[cursor]}.")
            cursor += 1
            field = self.schema.field(field_name)
            if field.encoding_kind == EncodingKind.CATEGORICAL:
                row[field_name] = decode_atomic_token(tokens[cursor])
                cursor += 1
            else:
                codec = field.numeric_codec
                assert codec is not None
                value_tokens = tokens[cursor: cursor + codec.rendered_length]
                row[field_name] = parse_numeric_value("".join(value_tokens), codec)
                cursor += codec.rendered_length
            if tokens[cursor] != SPECIAL_TOKENS["end_field"]:
                raise ValueError(f"Expected <END_FIELD>, got {tokens[cursor]}.")
            cursor += 1
        if tokens[cursor] != SPECIAL_TOKENS["eos"]:
            raise ValueError("Sequence does not terminate with <EOS>.")
        return row
