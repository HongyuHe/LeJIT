from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lejit.config import LeJITConfig
from lejit.constraints import ConstraintProgram
from lejit.exceptions import UnsupportedPromptError
from lejit.pipeline import LeJITPipeline


def write_toy_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    csv_path = tmp_path / "toy.csv"
    spec_path = tmp_path / "toy_dataset.json"
    rules_path = tmp_path / "rules.json"

    pd.DataFrame(
        [
            {"cat": "alpha", "x": 1, "y": 1},
            {"cat": "beta", "x": 2, "y": 2},
        ]
    ).to_csv(csv_path, index=False)

    spec = {
        "name": "toy",
        "source": {
            "type": "csv",
            "path": str(csv_path),
        },
        "fields": [
            {"name": "cat", "value_type": "categorical"},
            {"name": "x", "value_type": "integer"},
            {"name": "y", "value_type": "integer"},
        ],
    }
    rules = [
        {
            "rule_id": "r0",
            "formula": {
                "kind": "implies",
                "left": {
                    "kind": "compare",
                    "op": "=",
                    "left": {"kind": "symbol", "name": "cat"},
                    "right": {"kind": "constant", "value": "alpha"},
                },
                "right": {
                    "kind": "compare",
                    "op": "=",
                    "left": {"kind": "symbol", "name": "x"},
                    "right": {"kind": "constant", "value": 1},
                },
            },
            "display": "(cat = alpha) -> (x = 1)",
            "support": 1.0,
            "source": {},
        },
        {
            "rule_id": "r1",
            "formula": {
                "kind": "implies",
                "left": {
                    "kind": "compare",
                    "op": "=",
                    "left": {"kind": "symbol", "name": "cat"},
                    "right": {"kind": "constant", "value": "beta"},
                },
                "right": {
                    "kind": "compare",
                    "op": "=",
                    "left": {"kind": "symbol", "name": "x"},
                    "right": {"kind": "constant", "value": 2},
                },
            },
            "display": "(cat = beta) -> (x = 2)",
            "support": 1.0,
            "source": {},
        },
        {
            "rule_id": "r2",
            "formula": {
                "kind": "compare",
                "op": "<=",
                "left": {"kind": "symbol", "name": "x"},
                "right": {"kind": "symbol", "name": "y"},
            },
            "display": "x <= y",
            "support": 1.0,
            "source": {},
        },
    ]

    spec_path.write_text(json.dumps(spec, indent=2))
    rules_path.write_text(json.dumps(rules, indent=2))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"name": "toy", "rule_count": len(rules)}, indent=2)
    )
    return csv_path, spec_path, rules_path


def build_toy_config(spec_path: Path, rules_path: Path) -> LeJITConfig:
    return LeJITConfig.model_validate(
        {
            "dataset": {
                "dataset_spec": str(spec_path),
                "input_path": str(spec_path.parent / "toy.csv"),
                "rules_path": str(rules_path),
            },
            "model": {
                "mode": "config",
                "architecture": "gpt2",
                "config_overrides": {
                    "n_positions": 64,
                    "n_ctx": 64,
                    "n_embd": 32,
                    "n_layer": 1,
                    "n_head": 1,
                },
            },
            "serialization": {
                "numeric_precision": 2,
            },
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "max_steps": 1,
                "logging_steps": 1,
                "save_steps": 1,
            },
            "decoding": {
                "do_sample": False,
                "temperature": 1.0,
            },
            "run": {
                "n_samples": 2,
                "samples_per_prompt": 1,
            },
        }
    )


def test_constrained_generation_respects_rules_and_bounds(tmp_path: Path) -> None:
    _, spec_path, rules_path = write_toy_dataset(tmp_path)
    config = build_toy_config(spec_path, rules_path)
    pipeline = LeJITPipeline.build_from_config(config)

    generated = pipeline.generate(n_samples=4)
    constraints = ConstraintProgram(
        schema=pipeline.artifacts.schema,
        prepared=pipeline.artifacts.prepared,
        rules=pipeline.artifacts.rules,
    )

    assert list(generated.columns) == ["cat", "x", "y"]
    assert not generated.empty
    assert set(generated["cat"]).issubset({"alpha", "beta"})
    assert generated["x"].between(1, 2).all()
    assert generated["y"].between(1, 2).all()
    assert (generated["x"] <= generated["y"]).all()
    assert all(constraints.row_satisfies(row) for row in generated.to_dict(orient="records"))


def test_train_save_reload_and_complete_with_embedded_artifacts(tmp_path: Path) -> None:
    _, spec_path, rules_path = write_toy_dataset(tmp_path)
    config = build_toy_config(spec_path, rules_path)
    pipeline = LeJITPipeline.build_from_config(config)

    bundle_dir = tmp_path / "bundle"
    pipeline.train(bundle_dir)

    spec_path.unlink()
    rules_path.unlink()
    (tmp_path / "metadata.json").unlink()

    loaded = LeJITPipeline.load(bundle_dir)
    prompts = pd.DataFrame([{"cat": "alpha"}])
    completed = loaded.complete(prompts, samples_per_prompt=1)

    assert completed.iloc[0]["cat"] == "alpha"
    assert completed.iloc[0]["x"] == 1
    assert completed.iloc[0]["y"] in {1, 2}


def test_non_prefix_prompts_are_rejected(tmp_path: Path) -> None:
    _, spec_path, rules_path = write_toy_dataset(tmp_path)
    config = build_toy_config(spec_path, rules_path)
    pipeline = LeJITPipeline.build_from_config(config)

    with pytest.raises(UnsupportedPromptError):
        pipeline.complete(pd.DataFrame([{"x": 1}]), samples_per_prompt=1)
