from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


STATE_HINT = "STRATEGY_EVIDENCE_SUMMARY_BUILT"

_NORMALIZED_RECORD_SCHEMA = {
    "schema_version": 1,
    "record_key": "model_family:strategy_id:experiment_id:start_index:train_steps",
    "ranking_order": [
        "win_count_desc",
        "mean_score_delta_desc",
        "mean_score_desc",
        "strategy_id_asc",
    ],
}

_CANONICAL_RECORD_KEYS = {
    "model_family",
    "strategy_id",
    "training_strategy",
    "experiment_id",
    "start_index",
    "train_steps",
    "score",
    "score_delta",
    "rel_mse",
    "rmse",
    "dominant_error_pattern",
    "physics_issue",
}

_RECORD_KEYS_TO_EXCLUDE_FROM_HYPERPARAMETERS = _CANONICAL_RECORD_KEYS | {
    "baseline_score",
    "candidate_execution_report_path",
    "evaluation_report_path",
    "final_loss",
    "initial_loss",
    "strategy_label",
}


class OperatorStrategyEvidenceError(ValueError):
    """Raised when an operator sweep summary cannot become strategy-growth evidence."""


def convert_operator_evidence_summary(
    *,
    sweep_summary_path: Path,
    output_path: Path,
    model_family: str,
    source_state_hint: str,
    long_window_threshold: int,
    force: bool = False,
) -> dict[str, Any]:
    sweep_file = Path(sweep_summary_path)
    output_file = Path(output_path)
    model_family = _nonempty_text(model_family, "model_family")
    source_state_hint = _nonempty_text(source_state_hint, "source_state_hint")
    if output_file.exists() and not force:
        raise OperatorStrategyEvidenceError(f"output file already exists: {output_file}")
    if long_window_threshold <= 0:
        raise OperatorStrategyEvidenceError("long_window_threshold must be positive")

    sweep = _read_json(sweep_file)
    if sweep.get("state_hint") != source_state_hint:
        raise OperatorStrategyEvidenceError(
            f"{model_family} sweep summary has invalid state_hint"
        )
    winners = sweep.get("comparison_winners")
    if not isinstance(winners, list) or not winners:
        raise OperatorStrategyEvidenceError(
            "comparison_winners must be a non-empty list"
        )

    deltas = [_numeric(winner, "winner_score_delta") for winner in winners]
    start_indices = {
        winner["start_index"]
        for winner in winners
        if isinstance(winner, dict) and "start_index" in winner
    }
    if not start_indices:
        raise OperatorStrategyEvidenceError(
            "comparison_winners must include start_index"
        )
    candidate_records = _candidate_records(
        records=sweep.get("records"),
        model_family=model_family,
    )
    strategy_ranking = _strategy_ranking(
        records=candidate_records,
        winners=winners,
    )

    provenance_notes: list[str] = []
    forecast_steps = int((sweep.get("inputs") or {}).get("forecast_steps", 0))
    mean_delta = float(mean(deltas))
    if forecast_steps >= long_window_threshold:
        long_window_score_delta = mean_delta
    else:
        long_window_score_delta = 0.0
        provenance_notes.append("forecast_steps_below_long_window_threshold")

    physics_residual_delta = _optional_numeric(
        sweep,
        "physics_residual_delta",
        provenance_notes,
        f"physics_residual_delta_missing_from_{_model_slug(model_family)}_sweep",
    )
    inference_time_regression = _optional_numeric(
        sweep,
        "inference_time_regression",
        provenance_notes,
        f"inference_time_regression_missing_from_{_model_slug(model_family)}_sweep",
    )
    failure_rate = sum(delta < 0 for delta in deltas) / len(deltas)

    source_model_key = f"source_{_model_slug(model_family)}_evidence_summary_path"
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "model_family": model_family,
        "source_state_hint": source_state_hint,
        "source_evidence_summary_path": str(sweep_file),
        source_model_key: str(sweep_file),
        "evidence_summary_path": str(output_file),
        "evaluation_window_count": len(start_indices),
        "mean_score_delta": mean_delta,
        "score_delta_std": float(pstdev(deltas)) if len(deltas) > 1 else 0.0,
        "long_window_score_delta": long_window_score_delta,
        "physics_residual_delta": physics_residual_delta,
        "inference_time_regression": inference_time_regression,
        "failure_rate": failure_rate,
        "operator_evidence_schema": _NORMALIZED_RECORD_SCHEMA,
        "candidate_records": candidate_records,
        "strategy_ranking": strategy_ranking,
        "best_strategy_id": (
            strategy_ranking[0]["strategy_id"] if strategy_ranking else None
        ),
        "provenance_notes": provenance_notes,
        "source_metrics": {
            "forecast_steps": forecast_steps,
            "comparison_count": sweep.get("comparison_count"),
            "record_count": sweep.get("record_count"),
        },
        "next_command": "evaluate-strategy-promotion",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OperatorStrategyEvidenceError(f"summary file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OperatorStrategyEvidenceError(f"summary must be a JSON object: {path}")
    return payload


def _numeric(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorStrategyEvidenceError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise OperatorStrategyEvidenceError(f"{key} must be finite")
    return number


def _candidate_records(
    *,
    records: Any,
    model_family: str,
) -> list[dict[str, Any]]:
    if records is None:
        return []
    if not isinstance(records, list):
        raise OperatorStrategyEvidenceError("records must be a list when present")
    return [
        _candidate_record(record=record, model_family=model_family)
        for record in records
    ]


def _candidate_record(
    *,
    record: Any,
    model_family: str,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise OperatorStrategyEvidenceError("records entries must be objects")
    training_strategy = _nonempty_text(
        record.get("training_strategy"),
        "record.training_strategy",
    )
    strategy_id = _strategy_id(record=record, training_strategy=training_strategy)
    normalized = {
        "model_family": model_family,
        "strategy_id": strategy_id,
        "training_strategy": training_strategy,
        "experiment_id": _nonempty_text(
            record.get("experiment_id"),
            "record.experiment_id",
        ),
        "start_index": _integer(record, "start_index"),
        "train_steps": _integer(record, "train_steps"),
        "score": _numeric(record, "score"),
        "score_delta": _numeric(record, "score_delta"),
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
        "hyperparameters": _hyperparameters(record),
    }
    return normalized


def _strategy_id(
    *,
    record: dict[str, Any],
    training_strategy: str,
) -> str:
    value = record.get("strategy_label")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return training_strategy


def _hyperparameters(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(record.items()):
        if key in _RECORD_KEYS_TO_EXCLUDE_FROM_HYPERPARAMETERS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _strategy_ranking(
    *,
    records: list[dict[str, Any]],
    winners: list[Any],
) -> list[dict[str, Any]]:
    if not records:
        return []
    winner_ids = {
        _nonempty_text(winner.get("winner_experiment_id"), "winner_experiment_id")
        for winner in winners
        if isinstance(winner, dict)
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["strategy_id"], []).append(record)

    ranking = []
    for strategy_id, items in groups.items():
        win_count = sum(item["experiment_id"] in winner_ids for item in items)
        training_strategies = sorted({item["training_strategy"] for item in items})
        ranking.append(
            {
                "strategy_id": strategy_id,
                "training_strategy": (
                    training_strategies[0]
                    if len(training_strategies) == 1
                    else "mixed"
                ),
                "record_count": len(items),
                "win_count": win_count,
                "win_rate": win_count / len(items),
                "mean_score": float(mean(item["score"] for item in items)),
                "mean_score_delta": float(
                    mean(item["score_delta"] for item in items)
                ),
                "best_score": max(item["score"] for item in items),
                "failure_rate": sum(item["score_delta"] < 0 for item in items)
                / len(items),
                "dominant_error_patterns": sorted(
                    {item["dominant_error_pattern"] for item in items}
                ),
                "physics_issues": sorted({item["physics_issue"] for item in items}),
            }
        )
    ranking.sort(
        key=lambda item: (
            -item["win_count"],
            -item["mean_score_delta"],
            -item["mean_score"],
            item["strategy_id"],
        )
    )
    return [
        {
            "rank": rank,
            **item,
        }
        for rank, item in enumerate(ranking, start=1)
    ]


def _integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OperatorStrategyEvidenceError(f"{key} must be an integer")
    return value


def _optional_numeric(
    mapping: dict[str, Any],
    key: str,
    provenance_notes: list[str],
    missing_note: str,
) -> float:
    value = mapping.get(key)
    if value is None:
        provenance_notes.append(missing_note)
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorStrategyEvidenceError(f"{key} must be numeric when present")
    number = float(value)
    if not math.isfinite(number):
        raise OperatorStrategyEvidenceError(f"{key} must be finite")
    return number


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OperatorStrategyEvidenceError(f"{name} must be a non-empty string")
    return value.strip()


def _model_slug(model_family: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_family.lower()).strip("_")
    if not slug:
        raise OperatorStrategyEvidenceError("model_family must contain alphanumeric text")
    return slug
