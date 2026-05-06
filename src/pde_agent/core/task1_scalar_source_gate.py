"""Scalar feature gates for selecting Task 1 candidate dynamics sources."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)
from pde_agent.core.manual_operator_tuning import score_prediction_arrays


class Task1ScalarSourceGateError(RuntimeError):
    pass


DEFAULT_WINDOWS = [
    (10, 20),
    (20, 35),
    (35, 57),
    (10, 35),
    (10, 57),
    (57, 105),
    (105, 200),
]
DEFAULT_FEATURES = [
    "context_velocity_energy",
    "context_gradient_energy",
    "context_last_std",
    "source_base_disagreement",
    "source_base_energy_ratio",
    "source_jump_start",
    "base_jump_start",
]
DEFAULT_QUANTILES = [0.05, 0.10, 0.18, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
DEFAULT_DIRECTIONS = ["gt", "lt"]
DEFAULT_BLEND_WEIGHTS = [1.0, 0.75, 0.5, 0.25]


def fit_task1_scalar_source_gate(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    source_predictions: Sequence[tuple[str, np.ndarray]],
    target: np.ndarray,
    max_steps: int = 8,
    windows: Sequence[tuple[int, int]] | None = None,
    feature_names: Sequence[str] | None = None,
    quantiles: Sequence[float] | None = None,
    directions: Sequence[str] | None = None,
    blend_weights: Sequence[float] | None = None,
    min_selected: int = 2,
    min_segment_improvement: float = 1.0e-9,
) -> dict[str, Any]:
    context = _as_context(context)
    current = _as_prediction(base_prediction, "base_prediction")
    target = _as_prediction(target, "target")
    if current.shape != target.shape:
        raise Task1ScalarSourceGateError("base_prediction and target shapes must match")
    if current.shape[0] != context.shape[0]:
        raise Task1ScalarSourceGateError("context and base_prediction sample counts must match")
    sources = [(label, _as_prediction(prediction, label)) for label, prediction in source_predictions]
    for _label, prediction in sources:
        if prediction.shape != current.shape:
            raise Task1ScalarSourceGateError("source prediction shapes must match base_prediction")
    if max_steps < 0:
        raise Task1ScalarSourceGateError("max_steps must be non-negative")

    gate_windows = list(windows or DEFAULT_WINDOWS)
    gate_features = list(feature_names or DEFAULT_FEATURES)
    gate_quantiles = [float(item) for item in (quantiles or DEFAULT_QUANTILES)]
    gate_directions = list(directions or DEFAULT_DIRECTIONS)
    weights = [float(item) for item in (blend_weights or DEFAULT_BLEND_WEIGHTS)]
    _validate_search_space(
        windows=gate_windows,
        quantiles=gate_quantiles,
        directions=gate_directions,
        blend_weights=weights,
    )

    selected_specs: list[dict[str, Any]] = []
    target_space_sse = np.sum(target.astype(np.float64) ** 2, axis=2)
    current_error_sse = _prediction_error_sse(prediction=current, target=target)
    score = _score_from_error_sse(
        error_sse=current_error_sse,
        target_space_sse=target_space_sse,
    )
    current_segment_score = float(score["segment_score"])
    for _step in range(max_steps):
        best_spec: dict[str, Any] | None = None
        best_prediction: np.ndarray | None = None
        best_score: dict[str, Any] | None = None
        for source_label, source_prediction in sources:
            for window_start, window_stop in gate_windows:
                for feature_name in gate_features:
                    values = _feature_values(
                        feature_name=feature_name,
                        context=context,
                        current_prediction=current,
                        source_prediction=source_prediction,
                        window_start=window_start,
                        window_stop=window_stop,
                    )
                    for direction in gate_directions:
                        for quantile in gate_quantiles:
                            threshold = float(np.quantile(values, quantile))
                            mask = values > threshold if direction == "gt" else values < threshold
                            selected_count = int(np.sum(mask))
                            if selected_count < min_selected or selected_count == mask.shape[0]:
                                continue
                            for blend_weight in weights:
                                candidate_error_sse = _patched_error_sse(
                                    current_error_sse=current_error_sse,
                                    current_prediction=current,
                                    source=source_prediction,
                                    target=target,
                                    mask=mask,
                                    window_start=window_start,
                                    window_stop=window_stop,
                                    blend_weight=blend_weight,
                                )
                                candidate_score = _score_from_error_sse(
                                    error_sse=candidate_error_sse,
                                    target_space_sse=target_space_sse,
                                )
                                candidate_segment_score = float(
                                    candidate_score["segment_score"]
                                )
                                if (
                                    candidate_segment_score
                                    > current_segment_score + min_segment_improvement
                                    and (
                                        best_score is None
                                        or candidate_segment_score
                                        > float(best_score["segment_score"])
                                    )
                                ):
                                    best_prediction = _apply_source_window(
                                        base=current,
                                        source=source_prediction,
                                        mask=mask,
                                        window_start=window_start,
                                        window_stop=window_stop,
                                        blend_weight=blend_weight,
                                    )
                                    best_score = candidate_score
                                    best_error_sse = candidate_error_sse
                                    best_spec = {
                                        "source_label": source_label,
                                        "window_start": int(window_start),
                                        "window_stop": int(window_stop),
                                        "feature_name": feature_name,
                                        "direction": direction,
                                        "threshold": threshold,
                                        "quantile": quantile,
                                        "blend_weight": float(blend_weight),
                                        "selected_count": selected_count,
                                        "segment_score": candidate_segment_score,
                                        "segment_score_delta": candidate_segment_score
                                        - current_segment_score,
                                    }
        if best_spec is None or best_prediction is None or best_score is None:
            break
        current = best_prediction
        current_error_sse = best_error_sse
        score = best_score
        current_segment_score = float(best_score["segment_score"])
        selected_specs.append(best_spec)
    current[:, :CONTEXT_STEPS, :] = context
    return {
        "prediction": current.astype(np.float32, copy=False),
        "selected_specs": selected_specs,
        "score": score,
        "source_count": len(sources),
        "search_space": {
            "windows": [[int(start), int(stop)] for start, stop in gate_windows],
            "feature_names": gate_features,
            "quantiles": gate_quantiles,
            "directions": gate_directions,
            "blend_weights": weights,
            "max_steps": int(max_steps),
            "min_selected": int(min_selected),
        },
    }


def _apply_source_window(
    *,
    base: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
    window_start: int,
    window_stop: int,
    blend_weight: float,
) -> np.ndarray:
    candidate = np.array(base, copy=True)
    if not np.any(mask):
        return candidate
    candidate[mask, window_start:window_stop, :] = (
        (1.0 - blend_weight) * candidate[mask, window_start:window_stop, :]
        + blend_weight * source[mask, window_start:window_stop, :]
    )
    return candidate.astype(np.float32, copy=False)


def _patched_error_sse(
    *,
    current_error_sse: np.ndarray,
    current_prediction: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    window_start: int,
    window_stop: int,
    blend_weight: float,
) -> np.ndarray:
    patched = np.array(current_error_sse, copy=True)
    if not np.any(mask):
        return patched
    values = (
        (1.0 - blend_weight) * current_prediction[mask, window_start:window_stop, :]
        + blend_weight * source[mask, window_start:window_stop, :]
    )
    patched[mask, window_start:window_stop] = np.sum(
        (
            values.astype(np.float64)
            - target[mask, window_start:window_stop, :].astype(np.float64)
        )
        ** 2,
        axis=2,
    )
    return patched


def _prediction_error_sse(*, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sum(
        (prediction.astype(np.float64) - target.astype(np.float64)) ** 2,
        axis=2,
    )


def _score_from_error_sse(
    *,
    error_sse: np.ndarray,
    target_space_sse: np.ndarray,
) -> dict[str, Any]:
    early = _rel_segment_from_sse(
        error_sse=error_sse,
        target_space_sse=target_space_sse,
        future_start=0,
        future_stop=47,
    )
    middle = _rel_segment_from_sse(
        error_sse=error_sse,
        target_space_sse=target_space_sse,
        future_start=47,
        future_stop=95,
    )
    late = _late_segment_from_sse(
        error_sse=error_sse,
        future_start=95,
        future_stop=190,
    )
    segment_score = float(
        0.25 * early["score"] + 0.25 * middle["score"] + 0.5 * late["score"]
    )
    return {
        "segment_score": segment_score,
        "task1_prediction_score": float(segment_score * 0.75),
        "segments": {"early": early, "middle": middle, "late": late},
    }


def _rel_segment_from_sse(
    *,
    error_sse: np.ndarray,
    target_space_sse: np.ndarray,
    future_start: int,
    future_stop: int,
) -> dict[str, Any]:
    start = CONTEXT_STEPS + future_start
    stop = CONTEXT_STEPS + future_stop
    rel_by_sample_time = error_sse[:, start:stop] / (
        target_space_sse[:, start:stop] + 1.0e-12
    )
    rel_by_sample = np.mean(rel_by_sample_time, axis=1)
    rel_mse = float(np.mean(np.minimum(rel_by_sample, 5.0)))
    score = float(100.0 * np.exp(-_score_decay(future_start) * rel_mse))
    return {
        "time_start": start,
        "time_stop": stop,
        "future_start": future_start,
        "future_stop": future_stop,
        "rel_mse": rel_mse,
        "score": score,
    }


def _late_segment_from_sse(
    *,
    error_sse: np.ndarray,
    future_start: int,
    future_stop: int,
) -> dict[str, Any]:
    start = CONTEXT_STEPS + future_start
    stop = CONTEXT_STEPS + future_stop
    rmse = float(np.sqrt(np.mean(error_sse[:, start:stop]) / float(SPACE_POINTS)))
    score = float(100.0 / (1.0 + 10.0 * rmse))
    return {
        "time_start": start,
        "time_stop": stop,
        "future_start": future_start,
        "future_stop": future_stop,
        "rmse": rmse,
        "lorentzian_score": score,
        "frechet_score": None,
        "score": score,
    }


def _score_decay(future_start: int) -> float:
    return 20.0 if future_start == 0 else 10.0


def _feature_values(
    *,
    feature_name: str,
    context: np.ndarray,
    current_prediction: np.ndarray,
    source_prediction: np.ndarray,
    window_start: int,
    window_stop: int,
) -> np.ndarray:
    last = context[:, CONTEXT_STEPS - 1, :].astype(np.float64)
    velocity = (
        context[:, CONTEXT_STEPS - 1, :] - context[:, CONTEXT_STEPS - 2, :]
    ).astype(np.float64)
    if feature_name == "context_velocity_energy":
        return np.mean(velocity**2, axis=1)
    if feature_name == "context_gradient_energy":
        gradient = 0.5 * (np.roll(last, -1, axis=1) - np.roll(last, 1, axis=1))
        return np.mean(gradient**2, axis=1)
    if feature_name == "context_last_std":
        return np.std(last, axis=1)
    if feature_name == "source_base_disagreement":
        return np.mean(
            (
                source_prediction[:, window_start:window_stop, :]
                - current_prediction[:, window_start:window_stop, :]
            )
            ** 2,
            axis=(1, 2),
        )
    if feature_name == "source_base_energy_ratio":
        source_energy = np.mean(
            source_prediction[:, window_start:window_stop, :] ** 2,
            axis=(1, 2),
        )
        base_energy = np.mean(
            current_prediction[:, window_start:window_stop, :] ** 2,
            axis=(1, 2),
        )
        return source_energy / np.maximum(base_energy, 1.0e-12)
    if feature_name == "source_jump_start":
        prior = max(CONTEXT_STEPS - 1, int(window_start) - 1)
        return np.mean(
            (source_prediction[:, window_start, :] - source_prediction[:, prior, :]) ** 2,
            axis=1,
        )
    if feature_name == "base_jump_start":
        prior = max(CONTEXT_STEPS - 1, int(window_start) - 1)
        return np.mean(
            (
                current_prediction[:, window_start, :]
                - current_prediction[:, prior, :]
            )
            ** 2,
            axis=1,
        )
    raise Task1ScalarSourceGateError(f"unsupported scalar gate feature: {feature_name}")


def _validate_search_space(
    *,
    windows: Sequence[tuple[int, int]],
    quantiles: Sequence[float],
    directions: Sequence[str],
    blend_weights: Sequence[float],
) -> None:
    for window_start, window_stop in windows:
        if window_start < CONTEXT_STEPS or window_stop > PREDICTION_STEPS:
            raise Task1ScalarSourceGateError("windows must stay inside prediction future")
        if window_stop <= window_start:
            raise Task1ScalarSourceGateError("window_stop must be greater than window_start")
    for quantile in quantiles:
        if quantile < 0.0 or quantile > 1.0:
            raise Task1ScalarSourceGateError("quantiles must be in [0, 1]")
    for direction in directions:
        if direction not in {"gt", "lt"}:
            raise Task1ScalarSourceGateError("directions must be gt or lt")
    for blend_weight in blend_weights:
        if blend_weight < 0.0 or blend_weight > 1.0:
            raise Task1ScalarSourceGateError("blend weights must be in [0, 1]")


def _as_context(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (CONTEXT_STEPS, SPACE_POINTS):
        raise Task1ScalarSourceGateError(
            f"context must have shape [N, {CONTEXT_STEPS}, {SPACE_POINTS}]"
        )
    return array


def _as_prediction(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise Task1ScalarSourceGateError(
            f"{label} must have shape [N, {PREDICTION_STEPS}, {SPACE_POINTS}]"
        )
    if not np.isfinite(array).all():
        raise Task1ScalarSourceGateError(f"{label} contains NaN or Inf values")
    return array
