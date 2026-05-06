"""Controlled front-25 dynamics source for Task 1 Burgers early windows."""

from __future__ import annotations

from typing import Any

import numpy as np

from pde_agent.core.burgers_pseudospectral import pseudospectral_burgers_rollout
from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)


class Task1Front25DynamicsError(RuntimeError):
    pass


DYNAMICS_MODES = {
    "delta_mean",
    "velocity_feature_ridge",
    "burgers_feature_ridge",
    "context_pod_ridge",
    "pseudospectral",
    "hybrid",
}
FEATURE_MODES = {"last_velocity", "last_velocity_gradient"}


def front25_dynamics_params_from_any(row: dict[str, Any]) -> dict[str, Any]:
    dynamics_mode = str(row.get("dynamics_mode", "hybrid"))
    if dynamics_mode not in DYNAMICS_MODES:
        raise Task1Front25DynamicsError(
            "dynamics_mode must be delta_mean, velocity_feature_ridge, "
            "burgers_feature_ridge, or hybrid"
        )
    feature_mode = str(row.get("feature_mode", "last_velocity_gradient"))
    if feature_mode not in FEATURE_MODES:
        raise Task1Front25DynamicsError(
            "feature_mode must be last_velocity or last_velocity_gradient"
        )
    return {
        "k": _clamp_int(row.get("k", 64), 1, 512),
        "horizon_steps": _clamp_int(row.get("horizon_steps", 25), 1, 25),
        "ridge_alpha": max(0.0, float(row.get("ridge_alpha", 1e-4))),
        "correction_weight": _clamp_float(row.get("correction_weight", 1.0), 0.0, 1.5),
        "dynamics_mode": dynamics_mode,
        "feature_mode": feature_mode,
        "distance_temperature": max(1.0e-12, float(row.get("distance_temperature", 1.0))),
        "spectral_strength": max(0.0, float(row.get("spectral_strength", 0.0))),
        "viscosity": max(0.0, float(row.get("viscosity", 0.001))),
        "feature_count": _clamp_int(row.get("feature_count", 16), 2, 64),
        "reference_time_stride": _clamp_int(row.get("reference_time_stride", 1), 1, 10),
        "pseudospectral_dt": max(1.0e-12, float(row.get("pseudospectral_dt", 0.05))),
        "pseudospectral_substeps": _clamp_int(row.get("pseudospectral_substeps", 4), 1, 32),
        "pseudospectral_filter_strength": max(
            0.0,
            float(row.get("pseudospectral_filter_strength", 0.0)),
        ),
        "pseudospectral_dealias_fraction": _clamp_float(
            row.get("pseudospectral_dealias_fraction", 2.0 / 3.0),
            0.0,
            1.0,
        ),
    }


def fit_task1_front25_dynamics(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    reference: np.ndarray,
    k: int = 64,
    horizon_steps: int = 25,
    ridge_alpha: float = 1e-4,
    correction_weight: float = 1.0,
    dynamics_mode: str = "hybrid",
    feature_mode: str = "last_velocity_gradient",
    distance_temperature: float = 1.0,
    spectral_strength: float = 0.0,
    viscosity: float = 0.001,
    feature_count: int = 16,
    reference_time_stride: int = 1,
    pseudospectral_dt: float = 0.05,
    pseudospectral_substeps: int = 4,
    pseudospectral_filter_strength: float = 0.0,
    pseudospectral_dealias_fraction: float = 2.0 / 3.0,
) -> dict[str, Any]:
    params = front25_dynamics_params_from_any(
        {
            "k": k,
            "horizon_steps": horizon_steps,
            "ridge_alpha": ridge_alpha,
            "correction_weight": correction_weight,
            "dynamics_mode": dynamics_mode,
            "feature_mode": feature_mode,
            "distance_temperature": distance_temperature,
            "spectral_strength": spectral_strength,
            "viscosity": viscosity,
            "feature_count": feature_count,
            "reference_time_stride": reference_time_stride,
            "pseudospectral_dt": pseudospectral_dt,
            "pseudospectral_substeps": pseudospectral_substeps,
            "pseudospectral_filter_strength": pseudospectral_filter_strength,
            "pseudospectral_dealias_fraction": pseudospectral_dealias_fraction,
        }
    )
    context = _as_context(context)
    base_prediction = _as_prediction(base_prediction, "base_prediction")
    reference = _as_reference(reference)
    reference = _time_stride_reference(
        reference,
        stride=int(params["reference_time_stride"]),
        horizon_steps=int(params["horizon_steps"]),
    )
    if base_prediction.shape[0] != context.shape[0]:
        raise Task1Front25DynamicsError(
            "base_prediction and context sample counts must match"
        )
    prediction = np.array(base_prediction, copy=True, dtype=np.float32)
    horizon = min(
        int(params["horizon_steps"]),
        PREDICTION_STEPS - CONTEXT_STEPS,
        reference.shape[1] - CONTEXT_STEPS,
    )
    if horizon < 1:
        raise Task1Front25DynamicsError(
            "reference must contain at least one step beyond the context window"
        )
    stop = CONTEXT_STEPS + horizon
    k_eff = min(int(params["k"]), reference.shape[0])
    leave_one_out = _is_leave_one_out_reference(context=context, reference=reference)
    neighbor_distances = []
    if params["dynamics_mode"] == "pseudospectral":
        short_future = pseudospectral_burgers_rollout(
            context=context,
            viscosity=float(params["viscosity"]),
            dt=float(params["pseudospectral_dt"]),
            substeps=int(params["pseudospectral_substeps"]),
            filter_strength=float(params["pseudospectral_filter_strength"]),
            dealias_fraction=float(params["pseudospectral_dealias_fraction"]),
            prediction_steps=stop,
        )[:, CONTEXT_STEPS:stop, :]
    elif params["dynamics_mode"] == "context_pod_ridge":
        short_future = _predict_context_pod_ridge_front25(
            context=context,
            reference=reference,
            horizon=horizon,
            ridge_alpha=float(params["ridge_alpha"]),
            feature_count=int(params["feature_count"]),
            viscosity=float(params["viscosity"]),
        )
    else:
        reference_context = reference[:, :CONTEXT_STEPS, :]
        reference_features = _context_features(
            reference_context,
            feature_mode=str(params["feature_mode"]),
            viscosity=float(params["viscosity"]),
        )
        short_future = np.empty(
            (context.shape[0], horizon, SPACE_POINTS),
            dtype=np.float32,
        )
        for sample_index in range(context.shape[0]):
            query_feature = _context_features(
                context[sample_index : sample_index + 1],
                feature_mode=str(params["feature_mode"]),
                viscosity=float(params["viscosity"]),
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
            finite = local_distances[np.isfinite(local_distances)]
            neighbor_distances.append(
                float(np.mean(finite)) if finite.size else float("nan")
            )
            short_future[sample_index] = _predict_front25_sample(
                query_context=context[sample_index],
                reference_neighbors=reference[indices],
                weights=weights,
                horizon=horizon,
                ridge_alpha=float(params["ridge_alpha"]),
                dynamics_mode=str(params["dynamics_mode"]),
                viscosity=float(params["viscosity"]),
            )
    weight = float(params["correction_weight"])
    prediction[:, CONTEXT_STEPS:stop, :] = (
        (1.0 - weight) * prediction[:, CONTEXT_STEPS:stop, :]
        + weight * short_future
    )
    strength = float(params["spectral_strength"])
    if strength > 0.0:
        prediction[:, CONTEXT_STEPS:stop, :] = _spectral_smooth_window(
            prediction[:, CONTEXT_STEPS:stop, :],
            strength=strength,
        )
    prediction[:, :CONTEXT_STEPS, :] = context
    metadata = {
        "dynamics_mode": params["dynamics_mode"],
        "feature_mode": params["feature_mode"],
        "k": k_eff,
        "horizon_steps": horizon,
        "ridge_alpha": float(params["ridge_alpha"]),
        "correction_weight": weight,
        "spectral_strength": strength,
        "viscosity": float(params["viscosity"]),
        "feature_count": int(params["feature_count"]),
        "reference_time_stride": int(params["reference_time_stride"]),
        "reference_time_steps": int(reference.shape[1]),
        "pseudospectral_dt": float(params["pseudospectral_dt"]),
        "pseudospectral_substeps": int(params["pseudospectral_substeps"]),
        "pseudospectral_filter_strength": float(
            params["pseudospectral_filter_strength"]
        ),
        "pseudospectral_dealias_fraction": float(
            params["pseudospectral_dealias_fraction"]
        ),
        "reference_count": int(reference.shape[0]),
        "leave_one_out": leave_one_out,
        "mean_neighbor_distance": float(np.nanmean(neighbor_distances))
        if neighbor_distances
        else None,
    }
    return {"prediction": prediction.astype(np.float32, copy=False), "metadata": metadata}


def _predict_front25_sample(
    *,
    query_context: np.ndarray,
    reference_neighbors: np.ndarray,
    weights: np.ndarray,
    horizon: int,
    ridge_alpha: float,
    dynamics_mode: str,
    viscosity: float,
) -> np.ndarray:
    query_last = query_context[CONTEXT_STEPS - 1].astype(np.float64)
    query_velocity = (
        query_context[CONTEXT_STEPS - 1] - query_context[CONTEXT_STEPS - 2]
    ).astype(np.float64)
    query_rhs = _burgers_rhs(query_context[CONTEXT_STEPS - 1], viscosity=viscosity)
    reference_last = reference_neighbors[:, CONTEXT_STEPS - 1, :].astype(np.float64)
    reference_velocity = (
        reference_neighbors[:, CONTEXT_STEPS - 1, :]
        - reference_neighbors[:, CONTEXT_STEPS - 2, :]
    ).astype(np.float64)
    reference_rhs = np.stack(
        [
            _burgers_rhs(row[CONTEXT_STEPS - 1], viscosity=viscosity)
            for row in reference_neighbors
        ],
        axis=0,
    )
    reference_delta = (
        reference_neighbors[:, CONTEXT_STEPS : CONTEXT_STEPS + horizon, :].astype(
            np.float64
        )
        - reference_last[:, None, :]
    )
    weight_view = weights.astype(np.float64)[:, None, None]
    delta_mean = query_last[None, :] + np.sum(weight_view * reference_delta, axis=0)
    velocity_ridge = _feature_ridge_future(
        query_last=query_last,
        query_feature=query_velocity,
        reference_feature=reference_velocity,
        reference_delta=reference_delta,
        weights=weights,
        ridge_alpha=ridge_alpha,
    )
    burgers_ridge = _feature_ridge_future(
        query_last=query_last,
        query_feature=query_rhs,
        reference_feature=reference_rhs,
        reference_delta=reference_delta,
        weights=weights,
        ridge_alpha=ridge_alpha,
    )
    if dynamics_mode == "delta_mean":
        return delta_mean.astype(np.float32)
    if dynamics_mode == "velocity_feature_ridge":
        return velocity_ridge.astype(np.float32)
    if dynamics_mode == "burgers_feature_ridge":
        return burgers_ridge.astype(np.float32)
    return (0.4 * velocity_ridge + 0.4 * burgers_ridge + 0.2 * delta_mean).astype(
        np.float32
    )


def _feature_ridge_future(
    *,
    query_last: np.ndarray,
    query_feature: np.ndarray,
    reference_feature: np.ndarray,
    reference_delta: np.ndarray,
    weights: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    weight_view = weights.astype(np.float64)[:, None, None]
    feature_view = reference_feature[:, None, :]
    numerator = np.sum(weight_view * reference_delta * feature_view, axis=0)
    denominator = np.sum(weight_view * feature_view * feature_view, axis=0)
    coefficient = numerator / (denominator + float(ridge_alpha))
    return query_last[None, :] + query_feature[None, :] * coefficient


def _predict_context_pod_ridge_front25(
    *,
    context: np.ndarray,
    reference: np.ndarray,
    horizon: int,
    ridge_alpha: float,
    feature_count: int,
    viscosity: float,
) -> np.ndarray:
    reference_context = reference[:, :CONTEXT_STEPS, :]
    x_ref = _global_context_features(
        reference_context,
        feature_count=feature_count,
        viscosity=viscosity,
    )
    x_query = _global_context_features(
        context,
        feature_count=feature_count,
        viscosity=viscosity,
    )
    x_mean = np.mean(x_ref, axis=0, keepdims=True)
    x_scale = np.std(x_ref, axis=0, keepdims=True)
    x_ref = (x_ref - x_mean) / np.maximum(x_scale, 1.0e-6)
    x_query = (x_query - x_mean) / np.maximum(x_scale, 1.0e-6)
    x_ref = np.concatenate(
        [np.ones((x_ref.shape[0], 1), dtype=np.float64), x_ref],
        axis=1,
    )
    x_query = np.concatenate(
        [np.ones((x_query.shape[0], 1), dtype=np.float64), x_query],
        axis=1,
    )
    y = (
        reference[:, CONTEXT_STEPS : CONTEXT_STEPS + horizon, :]
        - reference[:, CONTEXT_STEPS - 1 : CONTEXT_STEPS, :]
    ).reshape(reference.shape[0], -1)
    lhs = x_ref.T @ x_ref
    lhs += float(ridge_alpha) * np.eye(lhs.shape[0], dtype=np.float64)
    lhs[0, 0] -= float(ridge_alpha)
    weights = np.linalg.solve(lhs, x_ref.T @ y.astype(np.float64))
    delta = x_query @ weights
    return (
        context[:, CONTEXT_STEPS - 1 : CONTEXT_STEPS, :]
        + delta.reshape(context.shape[0], horizon, SPACE_POINTS)
    ).astype(np.float32)


def _global_context_features(
    context: np.ndarray,
    *,
    feature_count: int,
    viscosity: float,
) -> np.ndarray:
    last = context[:, CONTEXT_STEPS - 1, :].astype(np.float64)
    velocity = (
        context[:, CONTEXT_STEPS - 1, :] - context[:, CONTEXT_STEPS - 2, :]
    ).astype(np.float64)
    rhs = np.stack([_burgers_rhs(row, viscosity=viscosity) for row in last], axis=0)
    chunks = []
    for values in [last, velocity, rhs]:
        spectrum = np.fft.rfft(values, axis=1)[:, :feature_count]
        chunks.extend([np.real(spectrum), np.imag(spectrum)])
        chunks.append(np.mean(values, axis=1, keepdims=True))
        chunks.append(np.std(values, axis=1, keepdims=True))
    return np.concatenate(chunks, axis=1).astype(np.float64)


def _context_features(
    context: np.ndarray,
    *,
    feature_mode: str,
    viscosity: float,
) -> np.ndarray:
    last = context[:, CONTEXT_STEPS - 1, :].astype(np.float32)
    velocity = (
        context[:, CONTEXT_STEPS - 1, :] - context[:, CONTEXT_STEPS - 2, :]
    ).astype(np.float32)
    if feature_mode == "last_velocity":
        return _row_normalize(np.concatenate([last, velocity], axis=1))
    gradients = np.stack([_gradient(row) for row in last], axis=0).astype(np.float32)
    rhs = np.stack([_burgers_rhs(row, viscosity=viscosity) for row in last], axis=0)
    return _row_normalize(np.concatenate([last, velocity, gradients, rhs], axis=1))


def _burgers_rhs(values: np.ndarray, *, viscosity: float) -> np.ndarray:
    values64 = values.astype(np.float64)
    grad = _gradient(values64)
    lap = np.roll(values64, -1) - 2.0 * values64 + np.roll(values64, 1)
    return (-values64 * grad + float(viscosity) * lap).astype(np.float64)


def _gradient(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64)
    return 0.5 * (np.roll(values64, -1) - np.roll(values64, 1))


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
        raise Task1Front25DynamicsError(
            f"context must have shape [N, {CONTEXT_STEPS}, {SPACE_POINTS}]"
        )
    return array


def _as_prediction(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise Task1Front25DynamicsError(
            f"{name} must have shape [N, {PREDICTION_STEPS}, {SPACE_POINTS}]"
        )
    return array


def _as_reference(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] <= CONTEXT_STEPS or array.shape[2] != SPACE_POINTS:
        raise Task1Front25DynamicsError(
            f"reference must have shape [N, > {CONTEXT_STEPS}, {SPACE_POINTS}]"
        )
    return array[:, :PREDICTION_STEPS, :]


def _time_stride_reference(
    reference: np.ndarray,
    *,
    stride: int,
    horizon_steps: int,
) -> np.ndarray:
    if stride <= 1:
        return reference
    reduced = reference[:, ::stride, :]
    required_steps = CONTEXT_STEPS + int(horizon_steps)
    if reduced.shape[1] < required_steps:
        raise Task1Front25DynamicsError(
            "time-strided reference is shorter than context plus horizon: "
            f"{reduced.shape[1]} < {required_steps}"
        )
    return reduced


def _clamp_int(value: Any, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _clamp_float(value: Any, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
