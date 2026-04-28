from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


class UNetEvidenceSweepError(RuntimeError):
    pass


STATE_HINT = "UNET_EVIDENCE_SWEEP_COMPLETED"
EVALUATION_STATE_HINT = "UNET_CANDIDATE_EVALUATED"

_CSV_FIELDS = [
    "start_index",
    "train_steps",
    "experiment_id",
    "training_strategy",
    "baseline_score",
    "score",
    "score_delta",
    "rel_mse",
    "rmse",
    "dominant_error_pattern",
    "physics_issue",
    "residual_rmse",
    "mass_drift",
    "energy_growth_ratio",
    "high_frequency_energy_ratio",
]


def build_unet_evidence_sweep(
    *,
    evaluation_report_paths: list[str | Path],
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    if not evaluation_report_paths:
        raise UNetEvidenceSweepError("at least one UNet evaluation report is required")
    output_root = Path(output_dir)
    summary_path = output_root / "evidence_summary.json"
    csv_path = output_root / "evidence_summary.csv"
    if (summary_path.exists() or csv_path.exists()) and not force:
        raise UNetEvidenceSweepError(f"UNet evidence output already exists: {output_root}")

    records = [_load_evaluation_report(Path(path)) for path in evaluation_report_paths]
    _reject_duplicate_experiment_ids(records)
    comparison_winners = _comparison_winners(records)
    forecast_steps = _common_forecast_steps(records)

    summary = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "model_family": "UNet",
        "inputs": {
            "evaluation_report_paths": [
                str(Path(path)) for path in evaluation_report_paths
            ],
            "forecast_steps": forecast_steps,
        },
        "artifacts": {
            "summary_json_path": str(summary_path),
            "summary_csv_path": str(csv_path),
        },
        "records": records,
        "record_count": len(records),
        "comparison_winners": comparison_winners,
        "comparison_count": len(comparison_winners),
        "strategy_statistics": _strategy_statistics(
            records=records,
            winners=comparison_winners,
        ),
        "next_command": "convert-operator-evidence-summary",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, records)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_evaluation_report(path: Path) -> dict[str, Any]:
    report_file = _require_file(path, "UNet evaluation report")
    report = _load_json_object(report_file, "UNet evaluation report")
    if report.get("state_hint") != EVALUATION_STATE_HINT:
        raise UNetEvidenceSweepError(
            "UNet evaluation report must include state_hint "
            f"{EVALUATION_STATE_HINT}"
        )
    candidate = _require_object(report, "candidate")
    score = _require_object(report, "score")
    baseline = _require_object(report, "baseline_comparison")
    error = _require_object(report, "error_diagnosis")
    physics = _require_object(report, "physics")
    inputs = report.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise UNetEvidenceSweepError("inputs must be an object when present")

    record = {
        "start_index": _require_int(candidate, "start_index", "candidate"),
        "train_steps": _require_int(candidate, "train_steps", "candidate"),
        "experiment_id": _require_string(candidate, "experiment_id", "candidate"),
        "training_strategy": _require_string(
            candidate,
            "training_strategy",
            "candidate",
        ),
        "forecast_steps": _optional_positive_int(inputs, "forecast_steps"),
        "baseline_score": _require_number(
            baseline,
            "baseline_score",
            "baseline_comparison",
        ),
        "score": _require_number(score, "competition_like_score_short", "score"),
        "score_delta": _require_number(
            baseline,
            "score_delta",
            "baseline_comparison",
        ),
        "rel_mse": _require_number(score, "rel_mse", "score"),
        "rmse": _require_number(score, "rmse", "score"),
        "dominant_error_pattern": _require_string(
            error,
            "dominant_error_pattern",
            "error_diagnosis",
        ),
        "physics_issue": _require_string(physics, "physics_issue", "physics"),
        "evaluation_report_path": str(report_file),
    }
    physics_residual = report.get("physics_residual")
    if isinstance(physics_residual, dict):
        record.update(_physics_residual_metrics(physics_residual))
    return record


def _comparison_winners(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["start_index"], record["train_steps"]), []).append(
            record
        )

    winners = []
    for (start_index, train_steps), group in sorted(grouped.items()):
        winner = sorted(
            group,
            key=lambda record: (
                -record["score"],
                record["rel_mse"],
                record["rmse"],
                record["experiment_id"],
            ),
        )[0]
        winners.append(
            {
                "start_index": start_index,
                "train_steps": train_steps,
                "comparison_group_key": (
                    f"start_{start_index:04d}_steps_{train_steps:04d}"
                ),
                "winner_experiment_id": winner["experiment_id"],
                "winner_training_strategy": winner["training_strategy"],
                "winner_score": winner["score"],
                "winner_score_delta": winner["score_delta"],
            }
        )
    return winners


def _strategy_statistics(
    *,
    records: list[dict[str, Any]],
    winners: list[dict[str, Any]],
) -> dict[str, Any]:
    strategies = sorted({record["training_strategy"] for record in records})
    return {
        strategy: {
            "record_count": sum(
                record["training_strategy"] == strategy for record in records
            ),
            "win_count": sum(
                winner["winner_training_strategy"] == strategy for winner in winners
            ),
            "mean_score_delta": mean(
                [
                    record["score_delta"]
                    for record in records
                    if record["training_strategy"] == strategy
                ]
            ),
        }
        for strategy in strategies
    }


def _common_forecast_steps(records: list[dict[str, Any]]) -> int:
    values = {record["forecast_steps"] for record in records}
    if len(values) != 1:
        raise UNetEvidenceSweepError(
            "all UNet evaluation reports must share forecast_steps"
        )
    return values.pop()


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in _CSV_FIELDS})


def _reject_duplicate_experiment_ids(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        experiment_id = record["experiment_id"]
        if experiment_id in seen:
            raise UNetEvidenceSweepError(
                f"duplicate experiment_id in UNet reports: {experiment_id}"
            )
        seen.add(experiment_id)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UNetEvidenceSweepError(f"failed to parse {label} JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise UNetEvidenceSweepError(f"{label} JSON must be an object")
    return data


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise UNetEvidenceSweepError(f"missing {label} path: {path}")
    if not path.is_file():
        raise UNetEvidenceSweepError(f"{label} path is not a file: {path}")
    return path


def _require_object(report: dict[str, Any], key: str) -> dict[str, Any]:
    value = report.get(key)
    if not isinstance(value, dict):
        raise UNetEvidenceSweepError(f"candidate report must include {key}")
    return value


def _require_string(report: dict[str, Any], key: str, label: str) -> str:
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise UNetEvidenceSweepError(f"{label}.{key} must be a non-empty string")
    return value


def _require_int(report: dict[str, Any], key: str, label: str) -> int:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise UNetEvidenceSweepError(f"{label}.{key} must be an integer")
    if value < 0:
        raise UNetEvidenceSweepError(f"{label}.{key} must be non-negative")
    return value


def _optional_positive_int(report: dict[str, Any], key: str) -> int:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise UNetEvidenceSweepError(f"inputs.{key} must be an integer")
    if value <= 0:
        raise UNetEvidenceSweepError(f"inputs.{key} must be positive")
    return value


def _require_number(report: dict[str, Any], key: str, label: str) -> float:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UNetEvidenceSweepError(f"{label}.{key} must be numeric")
    return float(value)


def _physics_residual_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {
        "residual_rmse": _require_number(
            report,
            "residual_rmse",
            "physics_residual",
        ),
        "mass_drift": _require_number(report, "mass_drift", "physics_residual"),
        "energy_growth_ratio": _require_number(
            report,
            "energy_growth_ratio",
            "physics_residual",
        ),
        "high_frequency_energy_ratio": _require_number(
            report,
            "high_frequency_energy_ratio",
            "physics_residual",
        ),
    }
