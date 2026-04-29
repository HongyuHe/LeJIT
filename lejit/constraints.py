from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import z3
from netnomos.dataset import PreparedDataset
from netnomos.theory import lower_formula, symbol_for_field

from lejit.codecs import canonicalize_value, numeric_prefix_interval, validate_prefix_shape
from lejit.rules import RuleBundle
from lejit.schema import EncodingKind, FieldArtifact, SchemaArtifact


@dataclass(slots=True)
class ConstraintProgram:
    schema: SchemaArtifact
    prepared: PreparedDataset
    rules: RuleBundle
    formulas: list[z3.ExprRef] = field(init=False)
    cache: dict[tuple[Any, ...], bool] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.formulas = [
            lower_formula(
                rule.formula,
                self.prepared.field_specs,
                self.prepared.context_families,
            )
            for rule in self.rules.rules
        ]

    def is_value_feasible(self, assignment: Mapping[str, Any], field_name: str, value: Any) -> bool:
        field = self.schema.field(field_name)
        normalized = canonicalize_value(value, field)
        if not self._value_within_domain(field, normalized):
            return False
        key = (
            "value",
            field_name,
            self._freeze_assignment(assignment),
            self._freeze_value(normalized),
        )
        return self._check(key, assignment, field_name, normalized, None)

    def is_numeric_prefix_feasible(
        self,
        assignment: Mapping[str, Any],
        field_name: str,
        prefix: str,
    ) -> bool:
        field = self.schema.field(field_name)
        codec = field.numeric_codec
        assert codec is not None
        if not validate_prefix_shape(prefix, codec):
            return False
        interval = numeric_prefix_interval(prefix, codec)
        if not self._interval_within_bounds(field, interval.lower, interval.upper):
            return False
        key = ("prefix", field_name, self._freeze_assignment(assignment), prefix)
        return self._check(key, assignment, field_name, None, interval)

    def row_satisfies(self, row: Mapping[str, Any]) -> bool:
        solver = self._base_solver()
        for field_name, value in row.items():
            field = self.prepared.field_specs[field_name]
            solver.add(symbol_for_field(field_name, field) == self._z3_value(field, value))
        return solver.check() == z3.sat

    def _check(
        self,
        key: tuple[Any, ...],
        assignment: Mapping[str, Any],
        field_name: str,
        value: Any | None,
        interval: Any | None,
    ) -> bool:
        if key in self.cache:
            return self.cache[key]
        solver = self._base_solver()
        for assigned_field, assigned_value in assignment.items():
            field_spec = self.prepared.field_specs[assigned_field]
            solver.add(
                symbol_for_field(assigned_field, field_spec)
                == self._z3_value(field_spec, assigned_value)
            )
        field_spec = self.prepared.field_specs[field_name]
        symbol = symbol_for_field(field_name, field_spec)
        if value is not None:
            solver.add(symbol == self._z3_value(field_spec, value))
        if interval is not None:
            solver.add(symbol >= self._z3_value(field_spec, interval.lower))
            solver.add(symbol <= self._z3_value(field_spec, interval.upper))
        result = solver.check() == z3.sat
        self.cache[key] = result
        return result

    def _base_solver(self) -> z3.Solver:
        solver = z3.Solver()
        solver.add(self.formulas)
        return solver

    def _freeze_assignment(self, assignment: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        frozen = []
        for name, value in sorted(assignment.items()):
            field = self.schema.field(name)
            frozen.append((name, self._freeze_value(canonicalize_value(value, field))))
        return tuple(frozen)

    def _freeze_value(self, value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 12)
        return value

    def _z3_value(self, field: FieldArtifact | Any, value: Any) -> z3.ExprRef:
        if isinstance(value, bool):
            return z3.BoolVal(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return z3.IntVal(value)
        if isinstance(value, float):
            return z3.RealVal(str(value))
        return z3.StringVal(str(value))

    def _value_within_domain(self, field: FieldArtifact, value: Any) -> bool:
        if field.encoding_kind == EncodingKind.CATEGORICAL:
            if field.domain:
                return value in field.domain
            return True
        if field.lower_bound is not None and value < field.lower_bound:
            return False
        if field.upper_bound is not None and value > field.upper_bound:
            return False
        return True

    def _interval_within_bounds(
        self,
        field: FieldArtifact,
        lower: int | float,
        upper: int | float,
    ) -> bool:
        bounded_lower = field.lower_bound if field.lower_bound is not None else lower
        bounded_upper = field.upper_bound if field.upper_bound is not None else upper
        return not (upper < bounded_lower or lower > bounded_upper)
