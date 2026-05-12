from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HypothesisLabError(RuntimeError):
    pass


# State hints
SANDBOX_STATE_HINT = "HYPOTHESIS_LAB_SANDBOX_VERIFIED"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HypothesisLabError(f"JSON artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HypothesisLabError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise HypothesisLabError(f"expected JSON object: {path}")
    return value


def _require_state(payload: dict[str, Any], state_hint: str) -> None:
    if payload.get("state_hint") != state_hint:
        raise HypothesisLabError(f"expected state_hint {state_hint}")


def _guard_output(path: Path | None, *, force: bool) -> None:
    if path is not None and path.exists() and not force:
        raise HypothesisLabError(f"artifact already exists: {path}")


def _write_optional_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is not None:
        _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return int(max(lower, min(upper, int(value))))


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return round(max(lower, min(upper, float(value))), 6)


def _clamp_task1_retrieval_k(value: int) -> int:
    return _clamp_int(value, 4, 99)


def _string_list_from_any(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


# Constants used by retrieval refinement parameter normalization
EARLY_REFINEMENT_ANCHORS = {
    "retrieval",
    "persistence",
    "linear_velocity",
    "damped_linear_velocity",
    "scaled_damped_velocity",
    "retrieval_derivative_match",
    "phase_aligned_retrieval",
    "mean_trend_bias",
    "second_order_velocity",
    "affine_context_calibration",
    "phase_derivative_hybrid",
}
EARLY_HANDOFF_CURVES = {"linear", "cosine", "quadratic"}


def _normalize_retrieval_refinement_params(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "k": int(row["k"]),
        "mode": str(row["mode"]),
        "early_blend_steps": int(row["early_blend_steps"]),
        "persistence_weight_start": float(row["persistence_weight_start"]),
        "spectral_strength": float(row.get("spectral_strength", 0.0)),
    }
    if "early_anchor" in row:
        early_anchor = str(row["early_anchor"])
        if early_anchor not in EARLY_REFINEMENT_ANCHORS:
            raise HypothesisLabError(
                "retrieval refinement early_anchor must be one of retrieval, "
                "persistence, linear_velocity, damped_linear_velocity, "
                "scaled_damped_velocity, or "
                "retrieval_derivative_match, phase_aligned_retrieval, "
                "mean_trend_bias, second_order_velocity, "
                "affine_context_calibration, or phase_derivative_hybrid"
            )
        normalized["early_anchor"] = early_anchor
    if "handoff_curve" in row:
        handoff_curve = str(row["handoff_curve"])
        if handoff_curve not in EARLY_HANDOFF_CURVES:
            raise HypothesisLabError(
                "retrieval refinement handoff_curve must be one of linear, "
                "cosine, or quadratic"
            )
        normalized["handoff_curve"] = handoff_curve
    if "phase_shift_scale" in row:
        normalized["phase_shift_scale"] = _clamp_float(
            float(row["phase_shift_scale"]),
            0.0,
            2.0,
        )
    if "velocity_scale" in row:
        normalized["velocity_scale"] = _clamp_float(
            float(row["velocity_scale"]),
            0.0,
            1.5,
        )
    elif normalized.get("early_anchor") == "scaled_damped_velocity":
        normalized["velocity_scale"] = 0.5
    return normalized
