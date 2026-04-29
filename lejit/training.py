from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from lejit.config import TrainingConfig


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self.sequences[index]
        return {"input_ids": sample, "labels": sample}


@dataclass(slots=True)
class SequenceCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        labels = []
        attention_mask = []
        for feature in features:
            sample = feature["input_ids"]
            pad = [self.pad_token_id] * (max_len - len(sample))
            input_ids.append(sample + pad)
            attention_mask.append([1] * len(sample) + [0] * len(pad))
            labels.append(sample + [-100] * len(pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_training_arguments(output_dir: str, config: TrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
        optim="adamw_torch",
        use_cpu=not torch.cuda.is_available(),
    )


def train_model(
    model: Any,
    sequences: list[list[int]],
    pad_token_id: int,
    config: TrainingConfig,
    output_dir: str,
) -> Trainer:
    dataset = SequenceDataset(sequences)
    collator = SequenceCollator(pad_token_id=pad_token_id)
    trainer = Trainer(
        model=model,
        args=build_training_arguments(output_dir, config),
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    return trainer
