"""Diagnose Hermes experiment artifacts into convergence bottlenecks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HERMES_RESULT_DIAGNOSED"


class HermesResultDiagnosisError(RuntimeError):
    pass


def diagnose_hermes_result(
    *,
    artifact_path: str | Path,
    output_path: str | Path | None,
    max_inference_seconds: float = 120.0,
    min_score_delta: float = 1.0,
) -> dict[str, Any]:
    artifact_file = Path(artifact_path)
    artifact = _read_json_object(artifact_file)
    diagnosis_core = _diagnose(
        artifact=artifact,
        max_inference_seconds=max_inference_seconds,
        min_score_delta=min_score_delta,
    )
    diagnosis = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "source_artifact_path": str(artifact_file),
        "source_state_hint": artifact.get("state_hint"),
        "thresholds": {
            "max_inference_seconds": float(max_inference_seconds),
            "min_score_delta": float(min_score_delta),
        },
        **diagnosis_core,
        "next_command": "plan-hermes-next-action",
    }
    if output_path is not None:
        output_file = Path(output_path)
        diagnosis["diagnosis_path"] = str(output_file)
        _write_json(output_file, diagnosis)
    return diagnosis


def _diagnose(
    *,
    artifact: dict[str, Any],
    max_inference_seconds: float,
    min_score_delta: float,
) -> dict[str, Any]:
    state_hint = artifact.get("state_hint")
    operator_family = _operator_family(artifact)
    failure_reason = _string_or_none(artifact.get("failure_reason"))
    inference_time = _number_or_none(artifact.get("inference_time_seconds"))
    if inference_time is None:
        inference_time = _number_or_none(artifact.get("best_projected_full_inference_seconds"))
    if _looks_like_timeout(failure_reason) or (
        inference_time is not None and inference_time > max_inference_seconds
    ):
        return _diagnosis(
            primary_bottleneck="inference_timeout",
            action_family="runtime_tuning",
            operator_family=operator_family,
            confidence=0.9,
            reason_codes=["inference_over_budget"],
            evidence={
                "failure_reason": failure_reason,
                "inference_time_seconds": inference_time,
            },
        )

    if state_hint == "HERMES_OPERATOR_TUNING_LOOP_RAN":
        decision = artifact.get("decision")
        action = decision.get("action") if isinstance(decision, dict) else None
        if action == "accept_execution_config":
            return _diagnosis(
                primary_bottleneck="none_runtime_feasible",
                action_family="submission",
                operator_family=operator_family,
                confidence=0.85,
                reason_codes=["operator_runtime_config_accepted"],
                evidence={
                    "selected_config": artifact.get("selected_config"),
                    "best_projected_full_inference_seconds": artifact.get(
                        "best_projected_full_inference_seconds"
                    ),
                    "next_command": artifact.get("next_command"),
                },
            )

    score_delta = _score_delta(artifact)
    if score_delta is not None and score_delta < min_score_delta:
        return _diagnosis(
            primary_bottleneck="low_accuracy",
            action_family="operator_strategy",
            operator_family=operator_family,
            confidence=0.8,
            reason_codes=["score_delta_below_threshold"],
            evidence={"score_delta": score_delta},
        )

    if state_hint == "LOCAL_STRATEGY_LEADERBOARD_UPDATED":
        decision = artifact.get("decision")
        decision_name = decision.get("decision") if isinstance(decision, dict) else None
        if decision_name == "continue_scheduled_experiments":
            return _diagnosis(
                primary_bottleneck="pending_strategy_queue",
                action_family="continue_local_schedule",
                operator_family=operator_family,
                confidence=0.8,
                reason_codes=["pending_local_strategy_experiments"],
                evidence={"pending_experiment_ids": artifact.get("pending_experiment_ids")},
            )
        if decision_name == "local_strategy_queue_exhausted":
            return _diagnosis(
                primary_bottleneck="local_strategy_exhausted",
                action_family="research_log",
                operator_family=operator_family,
                confidence=0.7,
                reason_codes=["no_accepted_local_strategy"],
                evidence={"result_count": artifact.get("result_count")},
            )

    return _diagnosis(
        primary_bottleneck="no_blocker_detected",
        action_family="submission",
        operator_family=operator_family,
        confidence=0.6,
        reason_codes=["artifact_does_not_indicate_failure"],
        evidence={},
    )


def _diagnosis(
    *,
    primary_bottleneck: str,
    action_family: str,
    operator_family: str | None,
    confidence: float,
    reason_codes: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "primary_bottleneck": primary_bottleneck,
        "recommended_action_family": action_family,
        "operator_family": operator_family,
        "confidence": float(confidence),
        "reason_codes": reason_codes,
        "evidence": evidence,
    }


def _operator_family(artifact: dict[str, Any]) -> str | None:
    for key in ("operator_family", "model_family"):
        value = _string_or_none(artifact.get(key))
        if value is not None:
            return value
    source = artifact.get("source")
    if isinstance(source, dict):
        for key in ("operator_family", "model_family"):
            value = _string_or_none(source.get(key))
            if value is not None:
                return value
    candidate = artifact.get("candidate")
    if isinstance(candidate, dict):
        return _string_or_none(candidate.get("model_family"))
    return None


def _score_delta(artifact: dict[str, Any]) -> float | None:
    value = _number_or_none(artifact.get("score_delta"))
    if value is not None:
        return value
    score = _number_or_none(artifact.get("score"))
    baseline = _number_or_none(artifact.get("baseline_score"))
    if score is not None and baseline is not None:
        return score - baseline
    return None


def _looks_like_timeout(value: str | None) -> bool:
    return value is not None and "timeout" in value.lower()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HermesResultDiagnosisError(f"missing artifact path: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesResultDiagnosisError(f"invalid artifact JSON: {path}") from exc
    if not isinstance(data, dict):
        raise HermesResultDiagnosisError("artifact JSON must be an object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
