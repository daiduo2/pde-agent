"""Controlled short-window dynamics source for Task 1 Burgers candidates."""

from __future__ import annotations

from typing import Any

import numpy as np

from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)


class Task1ShortWindowCorrectorError(RuntimeError):
    pass


CORRECTOR_MODES = {"context_pod_ridge", "delta_pod_ridge"}


def short_window_corrector_params_from_any(row: dict[str, Any]) -> dict[str, Any]:
    corrector_mode = str(row.get("corrector_mode", "context_pod_ridge"))
    if corrector_mode not in CORRECTOR_MODES:
        raise Task1ShortWindowCorrectorError(
            "corrector_mode must be context_pod_ridge or delta_pod_ridge"
        )
    start_step = int(row.get("start_step", 20))
    stop_step = int(row.get("stop_step", 35))
    if start_step < CONTEXT_STEPS:
        raise Task1ShortWindowCorrectorError("start_step must be >= context length")
    if stop_step <= start_step:
        raise Task1ShortWindowCorrectorError("stop_step must be greater than start_step")
    if stop_step > PREDICTION_STEPS:
        raise Task1ShortWindowCorrectorError("stop_step must be <= prediction length")
    return {
        "start_step": start_step,
        "stop_step": stop_step,
        "feature_count": _clamp_int(row.get("feature_count", 24), 2, 96),
        "ridge_alpha": max(0.0, float(row.get("ridge_alpha", 1e-4))),
        "correction_weight": _clamp_float(row.get("correction_weight", 1.0), 0.0, 1.5),
        "reference_time_stride": _clamp_int(row.get("reference_time_stride", 1), 1, 10),
        "corrector_mode": corrector_mode,
    }


def fit_task1_short_window_corrector(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    reference: np.ndarray,
    start_step: int = 20,
    stop_step: int = 35,
    feature_count: int = 24,
    ridge_alpha: float = 1e-4,
    correction_weight: float = 1.0,
    reference_time_stride: int = 1,
    corrector_mode: str = "context_pod_ridge",
) -> dict[str, Any]:
    params = short_window_corrector_params_from_any(
        {
            "start_step": start_step,
            "stop_step": stop_step,
            "feature_count": feature_count,
            "ridge_alpha": ridge_alpha,
            "correction_weight": correction_weight,
            "reference_time_stride": reference_time_stride,
            "corrector_mode": corrector_mode,
        }
    )
    context = _as_context(context)
    base_prediction = _as_prediction(base_prediction, "base_prediction")
    reference = _as_reference(reference)
    reference = _time_stride_reference(
        reference,
        stride=int(params["reference_time_stride"]),
    )
    if context.shape[0] != base_prediction.shape[0]:
        raise Task1ShortWindowCorrectorError(
            "context and base_prediction sample counts must match"
        )
    start = int(params["start_step"])
    stop = int(params["stop_step"])
    if start < CONTEXT_STEPS:
        raise Task1ShortWindowCorrectorError("start_step must be >= context length")
    if stop <= start:
        raise Task1ShortWindowCorrectorError("stop_step must be greater than start_step")
    if stop > reference.shape[1]:
        raise Task1ShortWindowCorrectorError(
            "reference must cover the configured short window"
        )
    prediction = np.array(base_prediction, dtype=np.float32, copy=True)
    short_window = _predict_short_window(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        start=start,
        stop=stop,
        feature_count=int(params["feature_count"]),
        ridge_alpha=float(params["ridge_alpha"]),
        corrector_mode=str(params["corrector_mode"]),
    )
    weight = float(params["correction_weight"])
    prediction[:, start:stop, :] = (
        (1.0 - weight) * prediction[:, start:stop, :] + weight * short_window
    )
    prediction[:, :CONTEXT_STEPS, :] = context
    metadata = {
        "candidate_family": "task1_short_window_corrector",
        "start_step": start,
        "stop_step": stop,
        "window_steps": stop - start,
        "feature_count": int(params["feature_count"]),
        "ridge_alpha": float(params["ridge_alpha"]),
        "correction_weight": weight,
        "reference_time_stride": int(params["reference_time_stride"]),
        "reference_time_steps": int(reference.shape[1]),
        "reference_count": int(reference.shape[0]),
        "corrector_mode": str(params["corrector_mode"]),
        "uses_target_as_reference": False,
    }
    return {"prediction": prediction.astype(np.float32, copy=False), "metadata": metadata}


def _predict_short_window(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    reference: np.ndarray,
    start: int,
    stop: int,
    feature_count: int,
    ridge_alpha: float,
    corrector_mode: str,
) -> np.ndarray:
    basis = _context_basis(reference[:, :CONTEXT_STEPS, :], feature_count=feature_count)
    x_ref = _project_context(reference[:, :CONTEXT_STEPS, :], basis)
    x_query = _project_context(context, basis)
    y_ref = reference[:, start:stop, :].reshape(reference.shape[0], -1).astype(np.float64)
    if corrector_mode == "delta_pod_ridge":
        reference_base = np.repeat(
            reference[:, CONTEXT_STEPS - 1 : CONTEXT_STEPS, :],
            stop - start,
            axis=1,
        )
        y_ref = y_ref - reference_base.reshape(reference.shape[0], -1)
    weights = _ridge_fit(x_ref, y_ref, ridge_alpha=ridge_alpha)
    prediction = _with_bias(x_query) @ weights
    if corrector_mode == "delta_pod_ridge":
        query_base = np.repeat(
            context[:, CONTEXT_STEPS - 1 : CONTEXT_STEPS, :],
            stop - start,
            axis=1,
        )
        prediction = prediction + query_base.reshape(context.shape[0], -1)
    return prediction.reshape(context.shape[0], stop - start, SPACE_POINTS).astype(
        np.float32,
        copy=False,
    )


def _context_basis(reference_context: np.ndarray, *, feature_count: int) -> np.ndarray:
    flattened = reference_context.reshape(reference_context.shape[0], -1).astype(np.float64)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2 or not np.any(centered):
        return np.eye(flattened.shape[1], min(feature_count, flattened.shape[1]), dtype=np.float64)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(feature_count, vt.shape[0])
    return vt[:count].T


def _project_context(context: np.ndarray, basis: np.ndarray) -> np.ndarray:
    flattened = context.reshape(context.shape[0], -1).astype(np.float64)
    return flattened @ basis


def _ridge_fit(x: np.ndarray, y: np.ndarray, *, ridge_alpha: float) -> np.ndarray:
    design = _with_bias(x)
    gram = design.T @ design
    regularizer = float(ridge_alpha) * np.eye(gram.shape[0], dtype=np.float64)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(gram + regularizer, design.T @ y)


def _with_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)


def _as_context(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (CONTEXT_STEPS, SPACE_POINTS):
        raise Task1ShortWindowCorrectorError(
            f"context must have shape (N, {CONTEXT_STEPS}, {SPACE_POINTS})"
        )
    _require_finite(array, "context")
    return array


def _as_prediction(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise Task1ShortWindowCorrectorError(
            f"{name} must have shape (N, {PREDICTION_STEPS}, {SPACE_POINTS})"
        )
    _require_finite(array, name)
    return array


def _as_reference(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] < CONTEXT_STEPS + 1 or array.shape[2] != SPACE_POINTS:
        raise Task1ShortWindowCorrectorError(
            f"reference must have shape (N, T>=11, {SPACE_POINTS})"
        )
    _require_finite(array, "reference")
    return array


def _time_stride_reference(reference: np.ndarray, *, stride: int) -> np.ndarray:
    if stride <= 1:
        return reference
    reduced = reference[:, ::stride, :]
    if reduced.shape[1] < CONTEXT_STEPS + 1:
        raise Task1ShortWindowCorrectorError(
            "reference_time_stride leaves too few time steps"
        )
    return reduced


def _require_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise Task1ShortWindowCorrectorError(f"{name} must contain only finite values")


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
