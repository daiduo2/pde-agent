from __future__ import annotations

import numpy as np
from typing import Any

from pde_agent.core.hypothesis_lab_utils import _clamp_int


def _max_retrieval_k(context: np.ndarray, reference: np.ndarray) -> int:
    leave_one_out = bool(
        context.shape[0] == reference.shape[0]
        and np.allclose(context, reference[:, : context.shape[1], :], atol=1e-6)
    )
    return max(1, reference.shape[0] - 1 if leave_one_out else reference.shape[0])


def _linear_velocity_anchor_prediction(context: np.ndarray, steps: int) -> np.ndarray:
    prediction = np.empty((context.shape[0], steps, context.shape[2]), dtype=np.float32)
    prediction[:, : context.shape[1], :] = context
    velocity = context[:, -1:, :] - context[:, -2:-1, :]
    horizon = np.arange(
        1,
        steps - context.shape[1] + 1,
        dtype=np.float32,
    ).reshape(1, -1, 1)
    prediction[:, context.shape[1] :, :] = context[:, -1:, :] + horizon * velocity
    return prediction


def _damped_linear_velocity_anchor_prediction(
    *,
    context: np.ndarray,
    steps: int,
    decay: float,
    velocity_scale: float = 1.0,
) -> np.ndarray:
    prediction = np.empty((context.shape[0], steps, context.shape[2]), dtype=np.float32)
    prediction[:, : context.shape[1], :] = context
    velocity = context[:, -1:, :] - context[:, -2:-1, :]
    horizon = np.arange(
        1,
        steps - context.shape[1] + 1,
        dtype=np.float32,
    ).reshape(1, -1, 1)
    damping = np.exp(-(horizon - 1.0) / float(decay))
    prediction[:, context.shape[1] :, :] = (
        context[:, -1:, :] + float(velocity_scale) * horizon * damping * velocity
    )
    return prediction


def _retrieval_derivative_match_anchor_prediction(
    *,
    context: np.ndarray,
    retrieval: np.ndarray,
    decay: float,
) -> np.ndarray:
    prediction = retrieval.copy()
    velocity = context[:, -1:, :] - context[:, -2:-1, :]
    desired_first = context[:, -1:, :] + velocity
    first_future = retrieval[:, context.shape[1] : context.shape[1] + 1, :]
    correction = desired_first - first_future
    horizon = np.arange(
        0,
        retrieval.shape[1] - context.shape[1],
        dtype=np.float32,
    ).reshape(1, -1, 1)
    damping = np.exp(-horizon / float(decay))
    prediction[:, context.shape[1] :, :] = (
        prediction[:, context.shape[1] :, :] + damping * correction
    )
    prediction[:, : context.shape[1], :] = context
    return prediction


def _phase_aligned_retrieval_anchor_prediction(
    *,
    context: np.ndarray,
    retrieval: np.ndarray,
    scale: float,
) -> np.ndarray:
    prediction = retrieval.copy()
    shifts = _context_integer_phase_shifts(context)
    forecast_steps = retrieval.shape[1] - context.shape[1]
    for sample_index, per_step_shift in enumerate(shifts):
        for horizon_index in range(forecast_steps):
            shift = _clamp_int(
                round(float(per_step_shift) * float(scale) * float(horizon_index + 1)),
                -8,
                8,
            )
            if shift == 0:
                continue
            frame_index = context.shape[1] + horizon_index
            prediction[sample_index, frame_index, :] = np.roll(
                prediction[sample_index, frame_index, :],
                shift=shift,
            )
    prediction[:, : context.shape[1], :] = context
    return prediction


def _context_integer_phase_shifts(context: np.ndarray) -> np.ndarray:
    previous = context[:, -2, :]
    current = context[:, -1, :]
    shifts = np.zeros((context.shape[0],), dtype=np.int32)
    for sample_index in range(context.shape[0]):
        best_shift = 0
        best_mse = float("inf")
        for shift in range(-3, 4):
            mse = float(
                np.mean(
                    (
                        np.roll(previous[sample_index], shift)
                        - current[sample_index]
                    )
                    ** 2
                )
            )
            if mse < best_mse:
                best_mse = mse
                best_shift = shift
        shifts[sample_index] = best_shift
    return shifts


def _mean_trend_bias_anchor_prediction(
    *,
    context: np.ndarray,
    retrieval: np.ndarray,
    decay: float,
) -> np.ndarray:
    prediction = retrieval.copy()
    forecast_steps = retrieval.shape[1] - context.shape[1]
    horizon = np.arange(1, forecast_steps + 1, dtype=np.float32).reshape(1, -1, 1)
    mean_velocity = np.mean(
        context[:, -1:, :] - context[:, -2:-1, :],
        axis=2,
        keepdims=True,
    )
    last_mean = np.mean(context[:, -1:, :], axis=2, keepdims=True)
    desired_mean = last_mean + horizon * mean_velocity
    retrieval_mean = np.mean(
        retrieval[:, context.shape[1] :, :],
        axis=2,
        keepdims=True,
    )
    damping = np.exp(-(horizon - 1.0) / float(decay))
    prediction[:, context.shape[1] :, :] = (
        prediction[:, context.shape[1] :, :]
        + damping * (desired_mean - retrieval_mean)
    )
    prediction[:, : context.shape[1], :] = context
    return prediction


def _second_order_velocity_anchor_prediction(
    *,
    context: np.ndarray,
    steps: int,
    decay: float,
) -> np.ndarray:
    prediction = np.empty((context.shape[0], steps, context.shape[2]), dtype=np.float32)
    prediction[:, : context.shape[1], :] = context
    velocity = context[:, -1:, :] - context[:, -2:-1, :]
    acceleration = (
        context[:, -1:, :]
        - 2.0 * context[:, -2:-1, :]
        + context[:, -3:-2, :]
    )
    horizon = np.arange(
        1,
        steps - context.shape[1] + 1,
        dtype=np.float32,
    ).reshape(1, -1, 1)
    damping = np.exp(-(horizon - 1.0) / float(decay))
    prediction[:, context.shape[1] :, :] = (
        context[:, -1:, :]
        + horizon * damping * velocity
        + 0.5 * horizon * (horizon - 1.0) * damping * acceleration
    )
    return prediction


def _affine_context_calibration_anchor_prediction(
    *,
    context: np.ndarray,
    retrieval: np.ndarray,
    decay: float,
) -> np.ndarray:
    prediction = retrieval.copy()
    forecast_steps = retrieval.shape[1] - context.shape[1]
    horizon = np.arange(1, forecast_steps + 1, dtype=np.float32).reshape(1, -1, 1)
    last_mean = np.mean(context[:, -1:, :], axis=2, keepdims=True)
    prev_mean = np.mean(context[:, -2:-1, :], axis=2, keepdims=True)
    mean_velocity = last_mean - prev_mean
    last_std = np.std(context[:, -1:, :], axis=2, keepdims=True)
    prev_std = np.std(context[:, -2:-1, :], axis=2, keepdims=True)
    std_velocity = last_std - prev_std
    desired_mean = last_mean + horizon * mean_velocity
    desired_std = np.maximum(
        np.float32(1e-6),
        last_std + horizon * std_velocity,
    )
    future = retrieval[:, context.shape[1] :, :]
    future_mean = np.mean(future, axis=2, keepdims=True)
    future_std = np.maximum(
        np.float32(1e-6),
        np.std(future, axis=2, keepdims=True),
    )
    calibrated = (future - future_mean) * (desired_std / future_std) + desired_mean
    damping = np.exp(-(horizon - 1.0) / float(decay))
    prediction[:, context.shape[1] :, :] = future + damping * (calibrated - future)
    prediction[:, : context.shape[1], :] = context
    return prediction


def _phase_derivative_hybrid_anchor_prediction(
    *,
    context: np.ndarray,
    retrieval: np.ndarray,
    decay: float,
    scale: float,
) -> np.ndarray:
    phase_aligned = _phase_aligned_retrieval_anchor_prediction(
        context=context,
        retrieval=retrieval,
        scale=scale,
    )
    return _retrieval_derivative_match_anchor_prediction(
        context=context,
        retrieval=phase_aligned,
        decay=decay,
    )


def _early_handoff_weights(
    *,
    forecast_steps: int,
    early_steps: int,
    weight_start: float,
    curve: str,
) -> np.ndarray:
    horizon = np.arange(forecast_steps, dtype=np.float32)
    progress = np.clip(1.0 - horizon / float(early_steps), 0.0, 1.0)
    if curve == "cosine":
        progress = 0.5 * (1.0 - np.cos(np.pi * progress))
    elif curve == "quadratic":
        progress = progress**2
    return (float(weight_start) * progress).reshape(1, forecast_steps, 1)
