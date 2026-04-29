from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    dataset_spec: str
    input_path: str | None = None
    rules_path: str
    limit: int | None = None


class ModelSourceConfig(StrictModel):
    mode: Literal["config", "pretrained"] = "config"
    architecture: str = "gpt2"
    name_or_path: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_name_or_path(self) -> ModelSourceConfig:
        if self.mode == "pretrained" and not self.name_or_path:
            raise ValueError("`name_or_path` is required when model.mode='pretrained'.")
        return self


class SerializationConfig(StrictModel):
    field_order: list[str] | None = None
    max_categorical_domain: int | None = None
    force_string_fields: list[str] = Field(default_factory=list)
    numeric_precision: int = 6


class TrainingConfig(StrictModel):
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 200


class DecodeConfig(StrictModel):
    max_new_tokens: int | None = None
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = True
    backtrack_limit: int = 50
    num_return_sequences: int = 1


class RunConfig(StrictModel):
    n_samples: int = 10
    batch_size: int = 1
    samples_per_prompt: int = 1
    prompt_columns: list[str] | None = None


class LeJITConfig(StrictModel):
    dataset: DatasetConfig
    model: ModelSourceConfig = Field(default_factory=ModelSourceConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    decoding: DecodeConfig = Field(default_factory=DecodeConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> LeJITConfig:
        import tomllib

        return cls.model_validate(tomllib.loads(Path(path).read_text()))

    def to_toml_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
