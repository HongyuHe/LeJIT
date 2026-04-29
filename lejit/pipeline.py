from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lejit.artifacts import load_bundle, save_bundle
from lejit.config import LeJITConfig
from lejit.constraints import ConstraintProgram
from lejit.modeling import build_model
from lejit.netnomos_adapter import NetNomosArtifacts, load_artifacts
from lejit.sampler import StepwiseConstrainedSampler
from lejit.tokenizer import RowSerializer, TokenVocabulary
from lejit.training import train_model
from lejit.utils import set_random_seed


@dataclass(slots=True)
class LeJITPipeline:
    config: LeJITConfig
    artifacts: NetNomosArtifacts
    vocab: TokenVocabulary
    serializer: RowSerializer
    model: Any

    @classmethod
    def build_from_config(
        cls,
        config: LeJITConfig,
        base_dir: str | Path | None = None,
    ) -> LeJITPipeline:
        config = materialize_dataset_paths(config, base_dir)
        artifacts = load_artifacts(config.dataset, config.serialization, base_dir=base_dir)
        vocab = TokenVocabulary.from_schema(artifacts.schema)
        serializer = RowSerializer(schema=artifacts.schema, vocab=vocab)
        model = build_model(config.model, len(vocab.id_to_token))
        model.config.bos_token_id = vocab.token_to_id["<BOS>"]
        model.config.eos_token_id = vocab.token_to_id["<EOS>"]
        model.config.pad_token_id = vocab.token_to_id["<EOS>"]
        return cls(
            config=config,
            artifacts=artifacts,
            vocab=vocab,
            serializer=serializer,
            model=model,
        )

    @classmethod
    def load(cls, bundle_dir: str | Path, device: str = "cpu") -> LeJITPipeline:
        saved = load_bundle(bundle_dir, device=device)
        config = saved.config
        artifacts = load_artifacts(config.dataset, config.serialization, base_dir=saved.root)
        serializer = RowSerializer(schema=saved.schema, vocab=saved.vocab)
        return cls(
            config=config,
            artifacts=artifacts,
            vocab=saved.vocab,
            serializer=serializer,
            model=saved.model,
        )

    def train(self, output_dir: str | Path) -> Path:
        set_random_seed(self.config.training.seed)
        records = self.artifacts.prepared.dataframe[self.artifacts.schema.field_order].to_dict(
            orient="records"
        )
        sequences = [
            self.vocab.encode(self.serializer.serialize_row(row))
            for row in records
        ]
        train_model(
            model=self.model,
            sequences=sequences,
            pad_token_id=self.vocab.token_to_id["<EOS>"],
            config=self.config.training,
            output_dir=str(Path(output_dir) / "trainer"),
        )
        return save_bundle(
            output_dir=output_dir,
            config=self.config,
            artifacts=self.artifacts,
            vocab=self.vocab,
            model=self.model,
        )

    def generate(
        self,
        n_samples: int | None = None,
        device: str = "cpu",
    ) -> pd.DataFrame:
        constraints = ConstraintProgram(
            schema=self.artifacts.schema,
            prepared=self.artifacts.prepared,
            rules=self.artifacts.rules,
        )
        sampler = StepwiseConstrainedSampler(
            model=self.model,
            schema=self.artifacts.schema,
            vocab=self.vocab,
            serializer=self.serializer,
            constraints=constraints,
            decode_config=self.config.decoding,
            device=device,
        )
        return sampler.generate_rows(n_samples or self.config.run.n_samples)

    def complete(
        self,
        prompts: pd.DataFrame,
        samples_per_prompt: int | None = None,
        device: str = "cpu",
    ) -> pd.DataFrame:
        constraints = ConstraintProgram(
            schema=self.artifacts.schema,
            prepared=self.artifacts.prepared,
            rules=self.artifacts.rules,
        )
        sampler = StepwiseConstrainedSampler(
            model=self.model,
            schema=self.artifacts.schema,
            vocab=self.vocab,
            serializer=self.serializer,
            constraints=constraints,
            decode_config=self.config.decoding,
            device=device,
        )
        return sampler.complete_rows(
            prompts,
            samples_per_prompt=samples_per_prompt or self.config.run.samples_per_prompt,
        )


def materialize_dataset_paths(
    config: LeJITConfig,
    base_dir: str | Path | None,
) -> LeJITConfig:
    if base_dir is None:
        return config
    payload = config.model_dump(mode="json")
    base = Path(base_dir)
    for key in ["dataset_spec", "rules_path", "input_path"]:
        value = payload["dataset"].get(key)
        if value:
            payload["dataset"][key] = str((base / value).resolve())
    return LeJITConfig.model_validate(payload)
