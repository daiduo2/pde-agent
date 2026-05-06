"""Local short-window transition expert for Task 1 Burgers rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)


class Task1EarlyTransitionExpertError(RuntimeError):
    pass


TRANSITION_MODES = {"velocity_ridge", "delta_mean", "hybrid"}
FEATURE_MODES = {"last_velocity", "last"}


def early_transition_expert_params_from_any(row: dict[str, Any]) -> dict[str, Any]:
    mode = str(row.get("transition_mode", "velocity_ridge"))
    if mode not in TRANSITION_MODES:
        raise Task1EarlyTransitionExpertError(
            "transition_mode must be one of velocity_ridge, delta_mean, or hybrid"
        )
    feature_mode = str(row.get("feature_mode", "last_velocity"))
    if feature_mode not in FEATURE_MODES:
        raise Task1EarlyTransitionExpertError(
            "feature_mode must be one of last_velocity or last"
        )
    return {
        "k": _clamp_int(row.get("k", 48), 1, 512),
        "horizon_steps": _clamp_int(row.get("horizon_steps", 47), 1, 190),
        "ridge_alpha": max(0.0, float(row.get("ridge_alpha", 1e-3))),
        "correction_weight": _clamp_float(row.get("correction_weight", 1.0), 0.0, 1.5),
        "transition_mode": mode,
        "feature_mode": feature_mode,
        "distance_temperature": max(
            1.0e-12,
            float(row.get("distance_temperature", 1.0)),
        ),
        "spectral_strength": max(0.0, float(row.get("spectral_strength", 0.0))),
    }


def fit_task1_early_transition_expert(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    reference: np.ndarray,
    k: int = 48,
    horizon_steps: int = 47,
    ridge_alpha: float = 1e-3,
    correction_weight: float = 1.0,
    transition_mode: str = "velocity_ridge",
    feature_mode: str = "last_velocity",
    distance_temperature: float = 1.0,
    spectral_strength: float = 0.0,
) -> dict[str, Any]:
    params = early_transition_expert_params_from_any(
        {
            "k": k,
            "horizon_steps": horizon_steps,
            "ridge_alpha": ridge_alpha,
            "correction_weight": correction_weight,
            "transition_mode": transition_mode,
            "feature_mode": feature_mode,
            "distance_temperature": distance_temperature,
            "spectral_strength": spectral_strength,
        }
    )
    context = _as_context(context)
    base_prediction = _as_prediction(base_prediction, "base_prediction")
    reference = _as_reference(reference)
    if base_prediction.shape[0] != context.shape[0]:
        raise Task1EarlyTransitionExpertError(
            "base_prediction and context sample counts must match"
        )
    prediction = np.array(base_prediction, copy=True, dtype=np.float32)
    stop = min(
        PREDICTION_STEPS,
        reference.shape[1],
        CONTEXT_STEPS + int(params["horizon_steps"]),
    )
    horizon = stop - CONTEXT_STEPS
    if horizon < 1:
        raise Task1EarlyTransitionExpertError(
            "reference must contain at least one step beyond the context window"
        )
    k_eff = min(int(params["k"]), reference.shape[0])
    leave_one_out = _is_leave_one_out_reference(context=context, reference=reference)

    reference_features = _transition_features(reference[:, :CONTEXT_STEPS, :], feature_mode=params["feature_mode"])
    neighbor_distances = []
    expert_future = np.empty((context.shape[0], horizon, SPACE_POINTS), dtype=np.float32)
    for sample_index in range(context.shape[0]):
        query_feature = _transition_features(
            context[sample_index : sample_index + 1],
            feature_mode=params["feature_mode"],
        )[0]
        distances = np.mean((reference_features - query_feature) ** 2, axis=1)
        if leave_one_out and sample_index < distances.shape[0]:
            distances[sample_index] = np.inf
        indices = np.argpartition(distances, kth=k_eff - 1)[:k_eff]
        local_distances = distances[indices]
        weights = _distance_weights(
            distances=local_distances,
            temperature=float(params["distance_temperature"]),
        )
        neighbor_distances.append(float(np.mean(local_distances[np.isfinite(local_distances)])))
        expert_future[sample_index] = _predict_sample_future(
            query_context=context[sample_index],
            reference_neighbors=reference[indices],
            weights=weights,
            horizon=horizon,
            ridge_alpha=float(params["ridge_alpha"]),
            transition_mode=str(params["transition_mode"]),
        )

    weight = float(params["correction_weight"])
    prediction[:, CONTEXT_STEPS:stop, :] = (
        (1.0 - weight) * prediction[:, CONTEXT_STEPS:stop, :]
        + weight * expert_future
    )
    strength = float(params["spectral_strength"])
    if strength > 0.0:
        prediction[:, CONTEXT_STEPS:stop, :] = _spectral_smooth_window(
            prediction[:, CONTEXT_STEPS:stop, :],
            strength=strength,
        )
    prediction[:, :CONTEXT_STEPS, :] = context
    metadata = {
        "transition_mode": params["transition_mode"],
        "feature_mode": params["feature_mode"],
        "k": k_eff,
        "horizon_steps": horizon,
        "requested_horizon_steps": int(params["horizon_steps"]),
        "reference_time_steps": int(reference.shape[1]),
        "ridge_alpha": float(params["ridge_alpha"]),
        "correction_weight": weight,
        "spectral_strength": strength,
        "reference_count": int(reference.shape[0]),
        "leave_one_out": leave_one_out,
        "mean_neighbor_distance": float(np.nanmean(neighbor_distances))
        if neighbor_distances
        else None,
    }
    return {"prediction": prediction.astype(np.float32, copy=False), "metadata": metadata}


def _predict_sample_future(
    *,
    query_context: np.ndarray,
    reference_neighbors: np.ndarray,
    weights: np.ndarray,
    horizon: int,
    ridge_alpha: float,
    transition_mode: str,
) -> np.ndarray:
    query_last = query_context[CONTEXT_STEPS - 1].astype(np.float64)
    query_velocity = (
        query_context[CONTEXT_STEPS - 1] - query_context[CONTEXT_STEPS - 2]
    ).astype(np.float64)
    reference_last = reference_neighbors[:, CONTEXT_STEPS - 1, :].astype(np.float64)
    reference_velocity = (
        reference_neighbors[:, CONTEXT_STEPS - 1, :]
        - reference_neighbors[:, CONTEXT_STEPS - 2, :]
    ).astype(np.float64)
    reference_delta = (
        reference_neighbors[:, CONTEXT_STEPS : CONTEXT_STEPS + horizon, :].astype(
            np.float64
        )
        - reference_last[:, None, :]
    )
    weight_view = weights.astype(np.float64)[:, None, None]
    delta_mean = np.sum(weight_view * reference_delta, axis=0)
    if transition_mode == "delta_mean":
        return (query_last[None, :] + delta_mean).astype(np.float32)

    velocity_view = reference_velocity[:, None, :]
    numerator = np.sum(weight_view * reference_delta * velocity_view, axis=0)
    denominator = np.sum(weight_view * velocity_view * velocity_view, axis=0)
    coefficient = numerator / (denominator + float(ridge_alpha))
    velocity_ridge = query_last[None, :] + query_velocity[None, :] * coefficient
    if transition_mode == "velocity_ridge":
        return velocity_ridge.astype(np.float32)
    return (0.5 * velocity_ridge + 0.5 * (query_last[None, :] + delta_mean)).astype(
        np.float32
    )


def _transition_features(context: np.ndarray, *, feature_mode: str) -> np.ndarray:
    last = context[:, CONTEXT_STEPS - 1, :].astype(np.float32)
    if feature_mode == "last":
        return _row_normalize(last)
    velocity = (
        context[:, CONTEXT_STEPS - 1, :] - context[:, CONTEXT_STEPS - 2, :]
    ).astype(np.float32)
    return _row_normalize(np.concatenate([last, velocity], axis=1))


def _distance_weights(*, distances: np.ndarray, temperature: float) -> np.ndarray:
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return np.full(distances.shape, 1.0 / float(distances.size), dtype=np.float32)
    scale = max(float(np.median(finite)) * float(temperature), 1.0e-12)
    safe = np.where(np.isfinite(distances), distances, np.max(finite) + scale)
    shifted = safe - float(np.min(safe))
    weights = np.exp(-shifted / scale)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(distances.shape, 1.0 / float(distances.size), dtype=np.float32)
    return (weights / total).astype(np.float32)


def _spectral_smooth_window(window: np.ndarray, *, strength: float) -> np.ndarray:
    transformed = np.fft.rfft(window.astype(np.float64), axis=2)
    freq = np.fft.rfftfreq(SPACE_POINTS).reshape(1, 1, -1)
    damping = np.exp(-float(strength) * (freq / max(freq.max(), 1.0e-12)) ** 2)
    smoothed = np.fft.irfft(transformed * damping, n=SPACE_POINTS, axis=2)
    return smoothed.astype(np.float32)


def _row_normalize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return centered / np.maximum(scale, 1.0e-6)


def _is_leave_one_out_reference(*, context: np.ndarray, reference: np.ndarray) -> bool:
    return bool(
        context.shape[0] == reference.shape[0]
        and np.allclose(context, reference[:, :CONTEXT_STEPS, :], atol=1e-6)
    )


def _as_context(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (CONTEXT_STEPS, SPACE_POINTS):
        raise Task1EarlyTransitionExpertError(
            f"context must have shape [N, {CONTEXT_STEPS}, {SPACE_POINTS}]"
        )
    return array


def _as_prediction(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise Task1EarlyTransitionExpertError(
            f"{name} must have shape [N, {PREDICTION_STEPS}, {SPACE_POINTS}]"
        )
    return array


def _as_reference(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] <= CONTEXT_STEPS or array.shape[2] != SPACE_POINTS:
        raise Task1EarlyTransitionExpertError(
            f"reference must have shape [N, > {CONTEXT_STEPS}, {SPACE_POINTS}]"
        )
    return array[:, :PREDICTION_STEPS, :]


def _clamp_int(value: Any, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _clamp_float(value: Any, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
