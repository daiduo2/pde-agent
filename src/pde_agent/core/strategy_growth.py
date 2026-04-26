from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROPOSAL_STATE_HINT = "STRATEGY_PROPOSAL_RECORDED"
VALID_SOURCE_TYPES = {"local_library", "external_retrieval", "agent_hypothesis"}


class StrategyGrowthError(ValueError):
    """Raised when a strategy-growth artifact violates the protocol."""


def record_strategy_proposal(
    *,
    strategy_id: str,
    title: str,
    source_type: str,
    problem_context: str,
    hypothesis: str,
    mechanism: str,
    implementation_plan: dict[str, Any],
    expected_observables: list[str],
    known_risks: list[str],
    output_path: Path,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _require_nonempty("strategy_id", strategy_id)
    _require_nonempty("title", title)
    _require_nonempty("problem_context", problem_context)
    _require_nonempty("hypothesis", hypothesis)
    _require_nonempty("mechanism", mechanism)
    if source_type not in VALID_SOURCE_TYPES:
        raise StrategyGrowthError(f"unsupported source_type: {source_type}")
    if not isinstance(implementation_plan, dict) or not implementation_plan:
        raise StrategyGrowthError("implementation_plan must be a non-empty object")
    _require_nonempty_list("expected_observables", expected_observables)
    _require_nonempty_list("known_risks", known_risks)
    if output_file.exists():
        raise StrategyGrowthError(f"output file already exists: {output_file}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": PROPOSAL_STATE_HINT,
        "strategy_id": strategy_id,
        "title": title,
        "source_type": source_type,
        "problem_context": problem_context,
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "implementation_plan": implementation_plan,
        "expected_observables": expected_observables,
        "known_risks": known_risks,
        "strategy_proposal_path": str(output_file),
        "next_command": "build-strategy-rubric",
    }
    _write_json(output_file, report)
    return report


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StrategyGrowthError(f"{name} must be a non-empty string")


def _require_nonempty_list(name: str, value: list[str]) -> None:
    if not isinstance(value, list) or not value:
        raise StrategyGrowthError(f"{name} must be a non-empty list")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise StrategyGrowthError(f"{name} must contain only non-empty strings")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
