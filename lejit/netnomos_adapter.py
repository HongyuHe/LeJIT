from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from netnomos.dataset import PreparedDataset, prepare_dataset
from netnomos.specs import DatasetSpec, load_dataset_spec

from lejit.config import DatasetConfig, SerializationConfig
from lejit.rules import RuleBundle, collect_formula_references, load_rule_bundle
from lejit.schema import SchemaArtifact


@dataclass(slots=True)
class NetNomosArtifacts:
    dataset_spec: DatasetSpec
    prepared: PreparedDataset
    schema: SchemaArtifact
    rules: RuleBundle


def load_artifacts(
    dataset: DatasetConfig,
    serialization: SerializationConfig,
    base_dir: str | Path | None = None,
) -> NetNomosArtifacts:
    dataset_spec_path = resolve_relative(dataset.dataset_spec, base_dir)
    dataset_spec = load_dataset_spec(dataset_spec_path)
    input_path = resolve_relative(dataset.input_path, base_dir) if dataset.input_path else None
    prepared = prepare_dataset(dataset_spec, input_path=input_path, limit=dataset.limit)
    schema = SchemaArtifact.from_prepared(prepared, serialization)
    rules = load_rule_bundle(resolve_relative(dataset.rules_path, base_dir))
    validate_rule_references(schema, rules)
    return NetNomosArtifacts(
        dataset_spec=dataset_spec,
        prepared=prepared,
        schema=schema,
        rules=rules,
    )


def validate_rule_references(schema: SchemaArtifact, rules: RuleBundle) -> None:
    missing: dict[str, set[str]] = {}
    for rule in rules.rules:
        refs = collect_formula_references(rule.formula)
        unknown = {name for name in refs if not reference_exists(schema, name)}
        if unknown:
            missing[rule.rule_id] = unknown
    if missing:
        details = ", ".join(
            f"{rule_id}: {sorted(values)}"
            for rule_id, values in sorted(missing.items())
        )
        raise ValueError(f"Rule bundle references unavailable prepared fields: {details}")


def reference_exists(schema: SchemaArtifact, reference: str) -> bool:
    if "[" in reference and reference.endswith("]"):
        base, _, suffix = reference[:-1].partition("[")
        family = schema.context_families.get(base)
        if family is None:
            return False
        if suffix.isdigit():
            return int(suffix) < len(family)
        return True
    return reference in schema.fields


def resolve_relative(path: str | Path, base_dir: str | Path | None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    if base_dir is None:
        return value.resolve()
    return (Path(base_dir) / value).resolve()
