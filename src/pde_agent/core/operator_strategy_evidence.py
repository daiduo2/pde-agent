from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


STATE_HINT = "STRATEGY_EVIDENCE_SUMMARY_BUILT"


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
