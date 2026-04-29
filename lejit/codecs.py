from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from lejit.schema import EncodingKind, FieldArtifact, NumericCodecSpec, SchemaArtifact


@dataclass(frozen=True, slots=True)
class NumericInterval:
    lower: int | float
    upper: int | float


def encode_atomic_token(field: str, value: Any) -> str:
    return f"VAL::{field}::{json.dumps(value, separators=(',', ':'), sort_keys=True, default=str)}"


def decode_atomic_token(token: str) -> Any:
    _, _, raw = token.partition("::")
    _, _, value = raw.partition("::")
    return json.loads(value)


def format_numeric_value(value: Any, codec: NumericCodecSpec) -> str:
    decimal_value = Decimal(str(value))
    sign = ""
    if codec.signed:
        sign = "+" if decimal_value >= 0 else "-"
    magnitude = abs(decimal_value)
    if codec.fraction_digits == 0:
        rendered = str(int(magnitude)).zfill(codec.integer_digits)
    else:
        quant = Decimal(1).scaleb(-codec.fraction_digits)
        rendered = f"{magnitude.quantize(quant):f}"
        integer_part, _, fraction_part = rendered.partition(".")
        integer_part = integer_part.zfill(codec.integer_digits)
        fraction_part = fraction_part.ljust(codec.fraction_digits, "0")
        rendered = f"{integer_part}.{fraction_part}"
    return f"{sign}{rendered}"


def parse_numeric_value(text: str, codec: NumericCodecSpec) -> int | float:
    if codec.fraction_digits == 0:
        return int(text)
    return float(text)


def numeric_prefix_interval(prefix: str, codec: NumericCodecSpec) -> NumericInterval:
    total = codec.rendered_length
    if len(prefix) > total:
        raise ValueError(f"Prefix '{prefix}' exceeds fixed numeric width {total}.")
    lower_fill = "0"
    upper_fill = "9"
    if codec.signed and prefix.startswith("-"):
        lower_fill = "9"
        upper_fill = "0"
    lower_suffix: list[str] = []
    upper_suffix: list[str] = []
    for index in range(len(prefix), total):
        if codec.signed and index == 0:
            lower_suffix.append("-")
            upper_suffix.append("+")
        elif codec.fraction_digits > 0 and index == codec.integer_digits + int(codec.signed):
            lower_suffix.append(".")
            upper_suffix.append(".")
        else:
            lower_suffix.append(lower_fill)
            upper_suffix.append(upper_fill)
    lower_text = prefix + "".join(lower_suffix)
    upper_text = prefix + "".join(upper_suffix)
    return NumericInterval(
        lower=parse_numeric_value(lower_text, codec),
        upper=parse_numeric_value(upper_text, codec),
    )


def canonicalize_value(value: Any, field: FieldArtifact) -> Any:
    if field.encoding_kind == EncodingKind.NUMERIC:
        codec = field.numeric_codec
        assert codec is not None
        return parse_numeric_value(format_numeric_value(value, codec), codec)
    return value


def validate_prefix_shape(prefix: str, codec: NumericCodecSpec) -> bool:
    if len(prefix) > codec.rendered_length:
        return False
    if codec.signed:
        if prefix and prefix[0] not in {"+", "-"}:
            return False
    if codec.fraction_digits == 0:
        return all(
            char.isdigit() or (codec.signed and index == 0 and char in {"+", "-"})
            for index, char in enumerate(prefix)
        )
    decimal_index = codec.integer_digits + int(codec.signed)
    for index, char in enumerate(prefix):
        if codec.signed and index == 0:
            if char not in {"+", "-"}:
                return False
            continue
        if index == decimal_index:
            if char != ".":
                return False
            continue
        if not char.isdigit():
            return False
    return True


def sequence_field_lengths(schema: SchemaArtifact) -> dict[str, int]:
    return {name: schema.field(name).rendered_length for name in schema.field_order}
