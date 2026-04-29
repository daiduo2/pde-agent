"""Diagnose Hermes validation scores into model-iteration actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HERMES_MODEL_QUALITY_DIAGNOSED"


class HermesModelQualityDiagnosisError(RuntimeError):
    pass


def diagnose_hermes_model_quality(
    *,
    validation_report_path: str | Path,
    output_path: str | Path | None,
    min_segment_score: float = 60.0,
    min_task_prediction_score: float = 45.0,
    max_late_rmse: float = 0.25,
    max_inference_seconds: float = 120.0,
) -> dict[str, Any]:
    validation_file = Path(validation_report_path)
    validation = _read_json_object(validation_file)
    if validation.get("state_hint") != "HERMES_VALIDATION_SCORED":
        raise HermesModelQualityDiagnosisError("validation report has invalid state_hint")
    score = validation.get("score")
    if not isinstance(score, dict):
        raise HermesModelQualityDiagnosisError("validation report missing score")

    diagnosis = _diagnose(
        validation=validation,
        score=score,
        min_segment_score=float(min_segment_score),
        min_task_prediction_score=float(min_task_prediction_score),
        max_late_rmse=float(max_late_rmse),
        max_inference_seconds=float(max_inference_seconds),
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "source_validation_report_path": str(validation_file),
        "operator_family": validation.get("operator_family"),
        "thresholds": {
            "min_segment_score": float(min_segment_score),
            "min_task_prediction_score": float(min_task_prediction_score),
            "max_late_rmse": float(max_late_rmse),
            "max_inference_seconds": float(max_inference_seconds),
        },
        **diagnosis,
        "next_command": "run-hermes-model-iteration-loop",
    }
    if output_path is not None:
        output_file = Path(output_path)
        report["diagnosis_path"] = str(output_file)
        _write_json(output_file, report)
    return report


def _diagnose(
    *,
    validation: dict[str, Any],
    score: dict[str, Any],
    min_segment_score: float,
    min_task_prediction_score: float,
    max_late_rmse: float,
    max_inference_seconds: float,
) -> dict[str, Any]:
    inference_seconds = _inference_seconds(validation)
    if inference_seconds is not None and inference_seconds > max_inference_seconds:
        return _diagnosis(
            primary_bottleneck="inference_over_budget",
            strategy_family="runtime_tuning",
            confidence=0.9,
            reason_codes=["inference_time_above_budget"],
            evidence={"inference_seconds": inference_seconds},
        )

    segments = score.get("segments")
    if not isinstance(segments, dict):
        raise HermesModelQualityDiagnosisError("score missing segments")
    early = _segment(segments, "early")
    middle = _segment(segments, "middle")
    late = _segment(segments, "late")
    early_score = _number(early.get("score"), "early.score")
    middle_score = _number(middle.get("score"), "middle.score")
    late_score = _number(late.get("score"), "late.score")
    late_rmse = _number(late.get("rmse"), "late.rmse")
    task_score = _optional_number(score.get("task_prediction_score"))

    if early_score < min_segment_score:
        return _diagnosis(
            primary_bottleneck="short_window_accuracy",
            strategy_family="residual_calibration",
            confidence=0.82,
            reason_codes=["early_score_below_threshold"],
            evidence=_score_evidence(early_score, middle_score, late_score, late_rmse),
        )
    if middle_score < min_segment_score:
        return _diagnosis(
            primary_bottleneck="mid_window_error_growth",
            strategy_family="spectral_correction",
            confidence=0.78,
            reason_codes=["middle_score_below_threshold"],
            evidence=_score_evidence(early_score, middle_score, late_score, late_rmse),
        )
    if late_score < min_segment_score or late_rmse > max_late_rmse:
        reasons = []
        if late_score < min_segment_score:
            reasons.append("late_score_below_threshold")
        if late_rmse > max_late_rmse:
            reasons.append("late_rmse_above_threshold")
        return _diagnosis(
            primary_bottleneck="late_window_drift",
            strategy_family="rollout_stabilization",
            confidence=0.85,
            reason_codes=reasons,
            evidence=_score_evidence(early_score, middle_score, late_score, late_rmse),
        )
    if task_score is not None and task_score < min_task_prediction_score:
        return _diagnosis(
            primary_bottleneck="low_competition_score",
            strategy_family="residual_calibration",
            confidence=0.75,
            reason_codes=["task_prediction_score_below_threshold"],
            evidence={
                **_score_evidence(early_score, middle_score, late_score, late_rmse),
                "task_prediction_score": task_score,
            },
        )
    return _diagnosis(
        primary_bottleneck="quality_acceptable",
        strategy_family="promote_candidate",
        confidence=0.88,
        reason_codes=["all_quality_thresholds_met"],
        evidence={
            **_score_evidence(early_score, middle_score, late_score, late_rmse),
            "task_prediction_score": task_score,
        },
    )


def _diagnosis(
    *,
    primary_bottleneck: str,
    strategy_family: str,
    confidence: float,
    reason_codes: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "primary_bottleneck": primary_bottleneck,
        "recommended_strategy_family": strategy_family,
        "confidence": confidence,
        "reason_codes": reason_codes,
        "evidence": evidence,
    }


def _score_evidence(
    early_score: float,
    middle_score: float,
    late_score: float,
    late_rmse: float,
) -> dict[str, float]:
    return {
        "early_score": early_score,
        "middle_score": middle_score,
        "late_score": late_score,
        "late_rmse": late_rmse,
    }


def _segment(segments: dict[str, Any], name: str) -> dict[str, Any]:
    value = segments.get(name)
    if not isinstance(value, dict):
        raise HermesModelQualityDiagnosisError(f"score missing {name} segment")
    return value


def _inference_seconds(validation: dict[str, Any]) -> float | None:
    prediction = validation.get("prediction")
    if isinstance(prediction, dict):
        inference = prediction.get("inference")
        if isinstance(inference, dict):
            return _optional_number(inference.get("elapsed_seconds"))
    materialized = validation.get("materialized")
    if isinstance(materialized, dict):
        submission = materialized.get("submission")
        if isinstance(submission, dict):
            metrics = submission.get("metrics")
            if isinstance(metrics, dict):
                return _optional_number(metrics.get("inference_time"))
    return None


def _number(value: object, label: str) -> float:
    result = _optional_number(value)
    if result is None:
        raise HermesModelQualityDiagnosisError(f"missing numeric {label}")
    return result


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HermesModelQualityDiagnosisError(f"missing validation report: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesModelQualityDiagnosisError(f"invalid validation report JSON: {path}") from exc
    if not isinstance(data, dict):
        raise HermesModelQualityDiagnosisError("validation report JSON must be an object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
