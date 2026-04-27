from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "EXTERNAL_STRATEGY_CANDIDATE_FORMALIZED"
DECISION_STATE_HINT = "STRATEGY_NEXT_ACTION_RECOMMENDED"


class ExternalStrategyCandidateError(ValueError):
    """Raised when external strategy candidate formalization is invalid."""


def formalize_external_strategy_candidate(
    *,
    strategy_decision_path: Path,
    output_path: Path,
    strategy_id: str,
    title: str,
    source_title: str,
    source_url: str,
    scientific_claim: str,
    mechanism: str,
    applicable_pde: str,
    targeted_failure_modes: list[str],
    implementation_plan: dict[str, Any],
    expected_observables: list[str],
    known_risks: list[str],
) -> dict[str, Any]:
    decision_file = Path(strategy_decision_path)
    output_file = Path(output_path)
    if output_file.exists():
        raise ExternalStrategyCandidateError(f"output file already exists: {output_file}")

    decision = _read_json(decision_file)
    if decision.get("state_hint") != DECISION_STATE_HINT:
        raise ExternalStrategyCandidateError("strategy decision has invalid state_hint")
    if decision.get("decision") != "retrieve_external_strategy":
        raise ExternalStrategyCandidateError(
            "strategy decision must be retrieve_external_strategy"
        )
    if decision.get("recommended_source_type") != "external_retrieval":
        raise ExternalStrategyCandidateError(
            "strategy decision must recommend external_retrieval"
        )

    proposal_fields = {
        "strategy_id": _nonempty_text(strategy_id, "strategy_id"),
        "title": _nonempty_text(title, "title"),
        "source_type": "external_retrieval",
        "problem_context": _problem_context(decision),
        "hypothesis": _nonempty_text(scientific_claim, "scientific_claim"),
        "mechanism": _nonempty_text(mechanism, "mechanism"),
        "implementation_plan": _nonempty_object(
            implementation_plan,
            "implementation_plan",
        ),
        "expected_observables": _nonempty_text_list(
            expected_observables,
            "expected_observables",
        ),
        "known_risks": _nonempty_text_list(known_risks, "known_risks"),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "source_type": "external_retrieval",
        "strategy_decision_path": str(decision_file),
        "retrieved_strategy_candidate_path": str(output_file),
        "retrieval_source": {
            "title": _nonempty_text(source_title, "source_title"),
            "url": _nonempty_text(source_url, "source_url"),
        },
        "scientific_claim": proposal_fields["hypothesis"],
        "mechanism": proposal_fields["mechanism"],
        "applicable_pde": _nonempty_text(applicable_pde, "applicable_pde"),
        "targeted_failure_modes": _nonempty_text_list(
            targeted_failure_modes,
            "targeted_failure_modes",
        ),
        "record_strategy_proposal_fields": proposal_fields,
        "next_command": "record-strategy-proposal",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _problem_context(decision: dict[str, Any]) -> str:
    model_family = _nonempty_text(decision.get("model_family"), "model_family")
    best_strategy_id = decision.get("best_strategy_id")
    best_clause = (
        f" after local best strategy {best_strategy_id}"
        if isinstance(best_strategy_id, str) and best_strategy_id.strip()
        else ""
    )
    reasons = decision.get("reason_codes", [])
    if isinstance(reasons, list) and reasons:
        reason_clause = ", ".join(str(item) for item in reasons)
    else:
        reason_clause = "failed local strategy evidence"
    return f"{model_family} strategy retrieval triggered by {reason_clause}{best_clause}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExternalStrategyCandidateError(f"JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExternalStrategyCandidateError(f"JSON artifact must be an object: {path}")
    return payload


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalStrategyCandidateError(f"{label} must be a non-empty string")
    return value.strip()


def _nonempty_text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ExternalStrategyCandidateError(f"{label} must be a non-empty list")
    return [_nonempty_text(item, label) for item in value]


def _nonempty_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ExternalStrategyCandidateError(f"{label} must be a non-empty object")
    return value
