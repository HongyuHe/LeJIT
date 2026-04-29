from __future__ import annotations

from enum import Enum
from typing import Any

from netnomos.dataset import PreparedDataset
from netnomos.specs import FieldSpec, ValueType
from pydantic import Field

from lejit.config import SerializationConfig, StrictModel


class EncodingKind(str, Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


class NumericCodecSpec(StrictModel):
    integer_digits: int
    fraction_digits: int = 0
    signed: bool = False

    @property
    def rendered_length(self) -> int:
        width = self.integer_digits
        if self.fraction_digits > 0:
            width += 1 + self.fraction_digits
        if self.signed:
            width += 1
        return width


class FieldArtifact(StrictModel):
    name: str
    value_type: str
    roles: list[str] = Field(default_factory=list)
    encoding_kind: EncodingKind
    domain: list[Any] | None = None
    lower_bound: int | float | None = None
    upper_bound: int | float | None = None
    enum_labels: dict[str, str] = Field(default_factory=dict)
    numeric_codec: NumericCodecSpec | None = None
    context_family: str | None = None
    context_index: int | None = None

    @property
    def rendered_length(self) -> int:
        if self.encoding_kind == EncodingKind.CATEGORICAL:
            return 1
        assert self.numeric_codec is not None
        return self.numeric_codec.rendered_length


class SchemaArtifact(StrictModel):
    dataset_name: str
    field_order: list[str]
    fields: dict[str, FieldArtifact]
    context_families: dict[str, list[str]]
    configured_exclude_fields: list[str] = Field(default_factory=list)
    auto_excluded_fields: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedDataset,
        serialization: SerializationConfig,
    ) -> SchemaArtifact:
        field_order = serialization.field_order or prepared.dataframe.columns.tolist()
        missing = [name for name in field_order if name not in prepared.field_specs]
        if missing:
            raise ValueError(f"Field order references unavailable prepared fields: {missing}")
        artifacts = {
            name: build_field_artifact(
                field=prepared.field_specs[name],
                values=prepared.dataframe[name].dropna().tolist(),
                serialization=serialization,
            )
            for name in field_order
        }
        return cls(
            dataset_name=prepared.spec.name,
            field_order=field_order,
            fields=artifacts,
            context_families=prepared.context_families,
            configured_exclude_fields=prepared.configured_exclude_fields,
            auto_excluded_fields=prepared.excluded_fields,
        )

    def field(self, name: str) -> FieldArtifact:
        return self.fields[name]


def build_field_artifact(
    field: FieldSpec,
    values: list[Any],
    serialization: SerializationConfig,
) -> FieldArtifact:
    value_type = field.value_type or infer_value_type(values)
    if value_type in {ValueType.CATEGORICAL, ValueType.STRING, ValueType.BOOLEAN}:
        domain = list(field.domain or sorted(set(values)))
        if (
            serialization.max_categorical_domain is not None
            and len(domain) > serialization.max_categorical_domain
        ):
            raise ValueError(
                f"Categorical field '{field.name}' has domain size {len(domain)}, which exceeds "
                f"serialization.max_categorical_domain={serialization.max_categorical_domain}."
            )
        return FieldArtifact(
            name=field.name,
            value_type=value_type.value,
            roles=[role.value if hasattr(role, "value") else str(role) for role in field.roles],
            encoding_kind=EncodingKind.CATEGORICAL,
            domain=domain,
            enum_labels=field.enum_labels,
            context_family=field.context_family,
            context_index=field.context_index,
        )
    numeric_codec = build_numeric_codec_spec(value_type, values, serialization.numeric_precision)
    lower_bound, upper_bound = resolve_numeric_bounds(field, values)
    return FieldArtifact(
        name=field.name,
        value_type=value_type.value,
        roles=[role.value if hasattr(role, "value") else str(role) for role in field.roles],
        encoding_kind=EncodingKind.NUMERIC,
        domain=list(field.domain or []),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        enum_labels=field.enum_labels,
        numeric_codec=numeric_codec,
        context_family=field.context_family,
        context_index=field.context_index,
    )


def infer_value_type(values: list[Any]) -> ValueType:
    if not values:
        return ValueType.STRING
    sample = next((value for value in values if value is not None), None)
    if isinstance(sample, bool):
        return ValueType.BOOLEAN
    if isinstance(sample, int) and not isinstance(sample, bool):
        return ValueType.INTEGER
    if isinstance(sample, float):
        return ValueType.REAL
    return ValueType.STRING


def build_numeric_codec_spec(
    value_type: ValueType,
    values: list[Any],
    numeric_precision: int,
) -> NumericCodecSpec:
    if not values:
        return NumericCodecSpec(integer_digits=1)
    signed = any(float(value) < 0 for value in values)
    if value_type == ValueType.INTEGER:
        integer_digits = max(len(str(abs(int(value)))) for value in values)
        return NumericCodecSpec(integer_digits=max(1, integer_digits), signed=signed)
    integer_digits = 1
    fraction_digits = 0
    for value in values:
        text = f"{float(value):.{numeric_precision}f}"
        if "." in text:
            integer_part, fraction_part = text.split(".", maxsplit=1)
            fraction_part = fraction_part.rstrip("0")
        else:
            integer_part, fraction_part = text, ""
        integer_digits = max(integer_digits, len(integer_part.lstrip("-")))
        fraction_digits = max(fraction_digits, len(fraction_part))
    return NumericCodecSpec(
        integer_digits=integer_digits,
        fraction_digits=fraction_digits,
        signed=signed,
    )


def resolve_numeric_bounds(
    field: FieldSpec,
    values: list[Any],
) -> tuple[int | float | None, int | float | None]:
    if field.bounds is not None:
        return field.bounds.lower, field.bounds.upper
    if not values:
        return None, None
    return min(values), max(values)
