"""Convert official operator comparison reports into strategy evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


STATE_HINT = "STRATEGY_EVIDENCE_SUMMARY_BUILT"
SOURCE_STATE_HINT = "OFFICIAL_OPERATOR_COMPARISON_COMPLETED"
MODEL_FAMILY = "OfficialOperatorComparison"


class OfficialComparisonEvidenceError(ValueError):
    pass


def convert_official_comparison_to_evidence(
    *,
    comparison_report_path: str | Path,
    output_path: str | Path,
    long_window_threshold: int = 31,
    force: bool = False,
) -> dict[str, Any]:
    comparison_file = Path(comparison_report_path)
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise OfficialComparisonEvidenceError(f"output file already exists: {output_file}")
    if long_window_threshold <= 0:
        raise OfficialComparisonEvidenceError("long_window_threshold must be positive")

    report = _read_json(comparison_file)
    if report.get("state_hint") != SOURCE_STATE_HINT:
        raise OfficialComparisonEvidenceError("comparison report has invalid state_hint")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise OfficialComparisonEvidenceError("comparison report inputs must be an object")
    evaluation_window_count = _positive_integer(inputs, "sample_limit")
    forecast_steps = _positive_integer(inputs, "forecast_steps")
    records = _candidate_records(report.get("model_records"))
    if not records:
        raise OfficialComparisonEvidenceError(
            "comparison report must contain non-baseline model records"
        )

    deltas = [record["score_delta"] for record in records]
    mean_delta = float(mean(deltas))
    provenance_notes: list[str] = []
    if forecast_steps >= long_window_threshold:
        long_window_score_delta = mean_delta
    else:
        long_window_score_delta = 0.0
        provenance_notes.append("forecast_steps_below_long_window_threshold")

    ranking = _strategy_ranking(records)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "model_family": MODEL_FAMILY,
        "source_state_hint": SOURCE_STATE_HINT,
        "source_evidence_summary_path": str(comparison_file),
        "source_official_comparison_report_path": str(comparison_file),
        "evidence_summary_path": str(output_file),
        "evaluation_window_count": evaluation_window_count,
        "mean_score_delta": mean_delta,
        "score_delta_std": float(pstdev(deltas)) if len(deltas) > 1 else 0.0,
        "long_window_score_delta": long_window_score_delta,
        "physics_residual_delta": 0.0,
        "inference_time_regression": 0.0,
        "failure_rate": sum(delta < 0.0 for delta in deltas) / len(deltas),
        "candidate_records": records,
        "strategy_ranking": ranking,
        "physics_metric_summary": {
            "record_count": 0,
            "mean_residual_rmse": None,
            "max_high_frequency_energy_ratio": None,
            "max_mass_drift": None,
            "max_energy_growth_ratio": None,
        },
        "best_strategy_id": ranking[0]["strategy_id"] if ranking else None,
        "provenance_notes": provenance_notes,
        "source_metrics": {
            "forecast_steps": forecast_steps,
            "comparison_model_count": len(report.get("model_records", [])),
            "candidate_record_count": len(records),
            "best_model_id": report.get("best_model_id"),
            "best_model_family": report.get("best_model_family"),
        },
        "next_command": "recommend-strategy-next-action",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _candidate_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise OfficialComparisonEvidenceError("model_records must be a list")
    normalized = []
    for record in records:
        if not isinstance(record, dict):
            raise OfficialComparisonEvidenceError("model_records entries must be objects")
        if record.get("source_type") == "baseline":
            continue
        normalized.append(_candidate_record(record))
    return normalized


def _candidate_record(record: dict[str, Any]) -> dict[str, Any]:
    model_id = _nonempty_text(record.get("model_id"), "record.model_id")
    return {
        "model_family": _nonempty_text(record.get("model_family"), "record.model_family"),
        "strategy_id": model_id,
        "training_strategy": model_id,
        "experiment_id": f"{model_id}_official_comparison",
        "start_index": 0,
        "train_steps": 0,
        "score": _numeric(record, "score"),
        "score_delta": _numeric(record, "score_delta_vs_persistence"),
        "rel_mse": _numeric(record, "rel_mse"),
        "rmse": _numeric(record, "rmse"),
        "dominant_error_pattern": _nonempty_text(
            record.get("dominant_error_pattern"),
            "record.dominant_error_pattern",
        ),
        "physics_issue": _nonempty_text(
            record.get("physics_issue"),
            "record.physics_issue",
        ),
        "hyperparameters": {
            "adapter_type": _nonempty_text(
                record.get("adapter_type"),
                "record.adapter_type",
            ),
            "source_type": _nonempty_text(
                record.get("source_type"),
                "record.source_type",
            ),
        },
    }


def _strategy_ranking(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda record: (
            -record["score_delta"],
            -record["score"],
            record["strategy_id"],
        ),
    )
    rows = []
    for rank, record in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank,
                "strategy_id": record["strategy_id"],
                "training_strategy": record["training_strategy"],
                "model_family": record["model_family"],
                "record_count": 1,
                "win_count": 1 if rank == 1 else 0,
                "win_rate": 1.0 if rank == 1 else 0.0,
                "mean_score": record["score"],
                "mean_score_delta": record["score_delta"],
                "best_score": record["score"],
                "failure_rate": 1.0 if record["score_delta"] < 0.0 else 0.0,
                "dominant_error_patterns": [record["dominant_error_pattern"]],
                "physics_issues": [record["physics_issue"]],
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OfficialComparisonEvidenceError(f"comparison report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OfficialComparisonEvidenceError("comparison report must be an object")
    return payload


def _positive_integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OfficialComparisonEvidenceError(f"{key} must be a positive integer")
    return value


def _numeric(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialComparisonEvidenceError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise OfficialComparisonEvidenceError(f"{key} must be finite")
    return number


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OfficialComparisonEvidenceError(f"{label} must be a non-empty string")
    return value.strip()
