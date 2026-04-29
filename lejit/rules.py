from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from netnomos.ast import (
    BinaryTerm,
    BoolAnd,
    BoolNot,
    BoolOr,
    Compare,
    Constant,
    Exists,
    ForAll,
    Formula,
    FuncCall,
    Implies,
    IndexedRef,
    SymbolRef,
    Term,
    formula_from_dict,
)
from netnomos.learners import LearnedRule


@dataclass(slots=True)
class RuleBundle:
    path: Path
    rules: list[LearnedRule]
    metadata: dict[str, Any]


def load_rule_bundle(path: str | Path) -> RuleBundle:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text())
    rules = [
        LearnedRule(
            rule_id=item["rule_id"],
            formula=formula_from_dict(item["formula"]),
            display=item.get("display", ""),
            support=float(item.get("support", 0.0)),
            source=item.get("source", {}),
        )
        for item in payload
    ]
    metadata_path = resolved.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return RuleBundle(path=resolved, rules=rules, metadata=metadata)


def collect_formula_references(formula: Formula) -> set[str]:
    refs: set[str] = set()
    _collect_formula_references(formula, refs)
    return refs


def _collect_formula_references(formula: Formula, refs: set[str]) -> None:
    if isinstance(formula, Compare):
        _collect_term_references(formula.left, refs)
        _collect_term_references(formula.right, refs)
        return
    if isinstance(formula, BoolNot):
        _collect_formula_references(formula.value, refs)
        return
    if isinstance(formula, (BoolAnd, BoolOr)):
        for value in formula.values:
            _collect_formula_references(value, refs)
        return
    if isinstance(formula, Implies):
        _collect_formula_references(formula.left, refs)
        _collect_formula_references(formula.right, refs)
        return
    if isinstance(formula, (ForAll, Exists)):
        _collect_formula_references(formula.body, refs)
        return


def _collect_term_references(term: Term, refs: set[str]) -> None:
    if isinstance(term, SymbolRef):
        refs.add(term.name)
        return
    if isinstance(term, IndexedRef):
        refs.add(f"{term.base}[{term.index}]")
        return
    if isinstance(term, BinaryTerm):
        _collect_term_references(term.left, refs)
        _collect_term_references(term.right, refs)
        return
    if isinstance(term, FuncCall):
        for arg in term.args:
            _collect_term_references(arg, refs)
        return
    if isinstance(term, Constant):
        return
