from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lejit.config import LeJITConfig
from lejit.modeling import load_model, save_model
from lejit.netnomos_adapter import NetNomosArtifacts
from lejit.schema import SchemaArtifact
from lejit.tokenizer import TokenVocabulary
from lejit.utils import ensure_dir, write_json


@dataclass(slots=True)
class SavedBundle:
    root: Path
    config: LeJITConfig
    schema: SchemaArtifact
    vocab: TokenVocabulary
    model: Any


def save_bundle(
    output_dir: str | Path,
    config: LeJITConfig,
    artifacts: NetNomosArtifacts,
    vocab: TokenVocabulary,
    model: Any,
) -> Path:
    root = ensure_dir(output_dir)
    save_model(model, root / "model")
    write_json(root / "dataset_spec.json", artifacts.dataset_spec.model_dump(mode="json"))
    write_json(root / "rules.json", [rule.to_dict() for rule in artifacts.rules.rules])
    if artifacts.rules.metadata:
        write_json(root / "metadata.json", artifacts.rules.metadata)
    config_payload = config.model_dump(mode="json")
    config_payload["dataset"]["dataset_spec"] = "dataset_spec.json"
    config_payload["dataset"]["rules_path"] = "rules.json"
    write_json(root / "config.json", config_payload)
    write_json(root / "schema.json", artifacts.schema.model_dump(mode="json"))
    write_json(root / "vocab.json", {"id_to_token": vocab.id_to_token})
    write_json(
        root / "manifest.json",
        {
            "dataset_name": artifacts.schema.dataset_name,
            "dataset_spec": config.dataset.dataset_spec,
            "input_path": config.dataset.input_path,
            "rules_path": config.dataset.rules_path,
            "rule_count": len(artifacts.rules.rules),
            "field_order": artifacts.schema.field_order,
            "excluded_fields": artifacts.schema.auto_excluded_fields,
        },
    )
    return root


def load_bundle(path: str | Path, device: str = "cpu") -> SavedBundle:
    root = Path(path).resolve()
    config = LeJITConfig.model_validate_json((root / "config.json").read_text())
    schema = SchemaArtifact.model_validate_json((root / "schema.json").read_text())
    vocab_payload = TokenVocabulary(**json_load(root / "vocab.json"))
    model = load_model(root / "model", device=device)
    return SavedBundle(root=root, config=config, schema=schema, vocab=vocab_payload, model=model)


def json_load(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text())
