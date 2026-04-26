from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROPOSAL_STATE_HINT = "STRATEGY_PROPOSAL_RECORDED"
RUBRIC_STATE_HINT = "STRATEGY_RUBRIC_BUILT"
PROMOTION_STATE_HINT = "STRATEGY_PROMOTION_EVALUATED"
VALID_SOURCE_TYPES = {"local_library", "external_retrieval", "agent_hypothesis"}
REQUIRED_PROMOTION_METRICS = [
    "evaluation_window_count",
    "mean_score_delta",
    "score_delta_std",
    "long_window_score_delta",
    "physics_residual_delta",
    "inference_time_regression",
    "failure_rate",
]


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


def build_strategy_rubric(
    *,
    proposal_path: Path,
    output_path: Path,
    min_evaluation_windows: int,
    max_inference_time_regression: float,
) -> dict[str, Any]:
    proposal_file = Path(proposal_path)
    output_file = Path(output_path)
    proposal = _read_json(proposal_file)
    if proposal.get("state_hint") != PROPOSAL_STATE_HINT:
        raise StrategyGrowthError("proposal artifact has invalid state_hint")
    if min_evaluation_windows < 2:
        raise StrategyGrowthError("min_evaluation_windows must be at least 2")
    if max_inference_time_regression < 0:
        raise StrategyGrowthError("max_inference_time_regression must be non-negative")
    if output_file.exists():
        raise StrategyGrowthError(f"output file already exists: {output_file}")

    rubric: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": RUBRIC_STATE_HINT,
        "strategy_id": proposal["strategy_id"],
        "strategy_proposal_path": str(proposal_file),
        "experiment_rubric_path": str(output_file),
        "required_metrics": REQUIRED_PROMOTION_METRICS,
        "promotion_thresholds": {
            "min_evaluation_windows": min_evaluation_windows,
            "min_mean_score_delta": 0.0,
            "min_long_window_score_delta": 0.0,
            "max_score_delta_std": 2.0,
            "max_physics_residual_delta": 0.0,
            "max_inference_time_regression": max_inference_time_regression,
            "max_failure_rate": 0.0,
        },
        "next_command": "evaluate-strategy-promotion",
    }
    _write_json(output_file, rubric)
    return rubric


def evaluate_strategy_promotion(
    *,
    proposal_path: Path,
    rubric_path: Path,
    evidence_summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    proposal_file = Path(proposal_path)
    rubric_file = Path(rubric_path)
    evidence_file = Path(evidence_summary_path)
    output_file = Path(output_path)
    proposal = _read_json(proposal_file)
    rubric = _read_json(rubric_file)
    evidence = _read_json(evidence_file)
    if proposal.get("state_hint") != PROPOSAL_STATE_HINT:
        raise StrategyGrowthError("proposal artifact has invalid state_hint")
    if rubric.get("state_hint") != RUBRIC_STATE_HINT:
        raise StrategyGrowthError("rubric artifact has invalid state_hint")
    if proposal.get("strategy_id") != rubric.get("strategy_id"):
        raise StrategyGrowthError("proposal and rubric strategy_id mismatch")
    if output_file.exists():
        raise StrategyGrowthError(f"output file already exists: {output_file}")

    thresholds = rubric["promotion_thresholds"]
    reason_codes: list[str] = []
    missing = [
        metric for metric in rubric["required_metrics"] if metric not in evidence
    ]
    for metric in missing:
        reason_codes.append(f"missing_metric:{metric}")

    if not missing:
        if evidence["evaluation_window_count"] < thresholds["min_evaluation_windows"]:
            reason_codes.append("insufficient_evaluation_windows")
        if evidence["mean_score_delta"] < thresholds["min_mean_score_delta"]:
            reason_codes.append("negative_mean_score_delta")
        if evidence["long_window_score_delta"] < thresholds["min_long_window_score_delta"]:
            reason_codes.append("negative_long_window_score_delta")
        if evidence["failure_rate"] > thresholds["max_failure_rate"]:
            reason_codes.append("nonzero_failure_rate")
        if (
            evidence["inference_time_regression"]
            > thresholds["max_inference_time_regression"]
        ):
            reason_codes.append("inference_time_regression_too_high")
        if evidence["physics_residual_delta"] > thresholds["max_physics_residual_delta"]:
            reason_codes.append("physics_residual_regressed")
        if evidence["score_delta_std"] > thresholds["max_score_delta_std"]:
            reason_codes.append("score_delta_variance_too_high")

    hard_reject_reasons = {
        code
        for code in reason_codes
        if code.startswith("missing_metric:")
        or code
        in {
            "insufficient_evaluation_windows",
            "negative_mean_score_delta",
            "negative_long_window_score_delta",
            "nonzero_failure_rate",
            "inference_time_regression_too_high",
        }
    }
    if hard_reject_reasons:
        decision = "reject"
        promote = False
    elif reason_codes:
        decision = "revise"
        promote = False
    else:
        decision = "promote"
        promote = True
        reason_codes = ["all_thresholds_passed"]

    report: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": PROMOTION_STATE_HINT,
        "strategy_id": proposal["strategy_id"],
        "strategy_proposal_path": str(proposal_file),
        "experiment_rubric_path": str(rubric_file),
        "evidence_summary_path": str(evidence_file),
        "promotion_report_path": str(output_file),
        "decision": decision,
        "promote_to_skill_candidate": promote,
        "reason_codes": reason_codes,
        "metric_results": {
            metric: evidence.get(metric) for metric in rubric["required_metrics"]
        },
        "next_command": "append-strategy-ledger",
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StrategyGrowthError(f"JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StrategyGrowthError(f"JSON artifact must be an object: {path}")
    return payload
