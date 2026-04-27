from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


STATE_HINT = "STRATEGY_NEXT_ACTION_RECOMMENDED"
EVIDENCE_STATE_HINT = "STRATEGY_EVIDENCE_SUMMARY_BUILT"


class StrategyDecisionError(ValueError):
    """Raised when strategy evidence cannot be routed to a next action."""


def recommend_strategy_next_action(
    *,
    evidence_summary_path: Path,
    output_path: Path,
    min_evaluation_windows: int,
) -> dict[str, Any]:
    evidence_file = Path(evidence_summary_path)
    output_file = Path(output_path)
    if min_evaluation_windows < 2:
        raise StrategyDecisionError("min_evaluation_windows must be at least 2")
    if output_file.exists():
        raise StrategyDecisionError(f"output file already exists: {output_file}")

    evidence = _read_json(evidence_file)
    if evidence.get("state_hint") != EVIDENCE_STATE_HINT:
        raise StrategyDecisionError("evidence summary has invalid state_hint")

    model_family = _nonempty_text(evidence.get("model_family"), "model_family")
    best_strategy_id = _optional_text(evidence.get("best_strategy_id"))
    evaluation_window_count = _integer(evidence, "evaluation_window_count")
    mean_score_delta = _number(evidence, "mean_score_delta")
    long_window_score_delta = _number(evidence, "long_window_score_delta")
    failure_rate = _number(evidence, "failure_rate")
    ranking = evidence.get("strategy_ranking", [])
    if not isinstance(ranking, list):
        raise StrategyDecisionError("strategy_ranking must be a list when present")

    decision, reason_codes, next_command, source_type, suggested_actions = (
        _route_decision(
            model_family=model_family,
            evaluation_window_count=evaluation_window_count,
            mean_score_delta=mean_score_delta,
            long_window_score_delta=long_window_score_delta,
            failure_rate=failure_rate,
            ranking=ranking,
            min_evaluation_windows=min_evaluation_windows,
        )
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "strategy_decision_path": str(output_file),
        "evidence_summary_path": str(evidence_file),
        "model_family": model_family,
        "best_strategy_id": best_strategy_id,
        "decision": decision,
        "reason_codes": reason_codes,
        "recommended_source_type": source_type,
        "suggested_actions": suggested_actions,
        "metrics": {
            "evaluation_window_count": evaluation_window_count,
            "mean_score_delta": mean_score_delta,
            "long_window_score_delta": long_window_score_delta,
            "failure_rate": failure_rate,
            "strategy_ranking_count": len(ranking),
        },
        "next_command": next_command,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _route_decision(
    *,
    model_family: str,
    evaluation_window_count: int,
    mean_score_delta: float,
    long_window_score_delta: float,
    failure_rate: float,
    ranking: list[Any],
    min_evaluation_windows: int,
) -> tuple[str, list[str], str, str | None, list[str]]:
    if evaluation_window_count < min_evaluation_windows:
        return (
            "expand_evidence",
            ["insufficient_evaluation_windows"],
            _evidence_sweep_command(model_family),
            None,
            [
                "add evaluation windows before promotion",
                "keep the current best strategy as a control arm",
            ],
        )

    failed_reasons = _failed_sweep_reasons(
        mean_score_delta=mean_score_delta,
        long_window_score_delta=long_window_score_delta,
        failure_rate=failure_rate,
    )
    if failed_reasons:
        return (
            "retrieve_external_strategy",
            failed_reasons,
            "record-strategy-proposal",
            "external_retrieval",
            [
                "search for a new PDE-operator stabilization strategy",
                "formalize the retrieved idea as a bounded experiment plan",
                "compare against the current best strategy and baseline",
            ],
        )

    if not ranking:
        return (
            "expand_evidence",
            ["missing_strategy_ranking"],
            _evidence_sweep_command(model_family),
            None,
            [
                "rebuild operator evidence with normalized strategy ranking",
                "include all candidate records in the evidence summary",
            ],
        )

    return (
        "evaluate_promotion",
        ["strong_multi_window_evidence"],
        "evaluate-strategy-promotion",
        "local_library",
        [
            "evaluate promotion against the rubric",
            "append the outcome to the strategy ledger",
        ],
    )


def _failed_sweep_reasons(
    *,
    mean_score_delta: float,
    long_window_score_delta: float,
    failure_rate: float,
) -> list[str]:
    reasons = []
    if mean_score_delta < 0:
        reasons.append("negative_mean_score_delta")
    if long_window_score_delta < 0:
        reasons.append("negative_long_window_score_delta")
    if failure_rate > 0.5:
        reasons.append("high_failure_rate")
    return reasons


def _evidence_sweep_command(model_family: str) -> str:
    normalized = model_family.lower()
    if normalized == "fno":
        return "run-fno-evidence-sweep"
    if normalized == "deeponet":
        return "run-deeponet-strategy-sweep"
    return "run-operator-evidence-sweep"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StrategyDecisionError(f"JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StrategyDecisionError(f"JSON artifact must be an object: {path}")
    return payload


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyDecisionError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, "best_strategy_id")


def _integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrategyDecisionError(f"{key} must be an integer")
    return value


def _number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyDecisionError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise StrategyDecisionError(f"{key} must be finite")
    return number
