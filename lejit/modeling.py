from __future__ import annotations

from pathlib import Path

import torch
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM, PreTrainedModel

from lejit.config import ModelSourceConfig


def build_model(config: ModelSourceConfig, vocab_size: int) -> PreTrainedModel:
    if config.mode == "pretrained":
        assert config.name_or_path is not None
        model = AutoModelForCausalLM.from_pretrained(config.name_or_path)
    else:
        if config.name_or_path:
            model_config = AutoConfig.from_pretrained(config.name_or_path)
        else:
            model_config = CONFIG_MAPPING[config.architecture]()
        for key, value in config.config_overrides.items():
            setattr(model_config, key, value)
        model = AutoModelForCausalLM.from_config(model_config)
    model.resize_token_embeddings(vocab_size)
    model.config.vocab_size = vocab_size
    return model


def save_model(model: PreTrainedModel, path: str | Path) -> None:
    model.save_pretrained(Path(path))


def load_model(path: str | Path, device: str | torch.device = "cpu") -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(Path(path))
    model.to(device)
    model.eval()
    return model
