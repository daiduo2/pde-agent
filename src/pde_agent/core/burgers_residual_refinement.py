"""Burgers residual scoring and bounded physics-guided refinement."""

from __future__ import annotations

from typing import Any

import numpy as np

from pde_agent.core.competition_submission import CONTEXT_STEPS


class BurgersResidualRefinementError(RuntimeError):
    pass


def burgers_residual_summary(
    trajectory: Any,
    *,
    viscosity: float = 0.001,
    dt: float = 0.05,
    dx: float = 1.0 / 256.0,
) -> dict[str, Any]:
    array = _as_trajectory(trajectory)
    residual = burgers_residual_array(
        array,
        viscosity=viscosity,
        dt=dt,
        dx=dx,
    )
    future = residual[:, max(0, CONTEXT_STEPS - 1) :, :]
    early = future[:, :47, :]
    middle = future[:, 47:95, :]
    late = future[:, 95:190, :]
    return {
        "schema_version": 1,
        "state_hint": "BURGERS_RESIDUAL_SUMMARY_BUILT",
        "residual_rmse": _rmse(future),
        "early_residual_rmse": _rmse(early),
        "middle_residual_rmse": _rmse(middle),
        "late_residual_rmse": _rmse(late),
        "residual_mean_abs": float(np.mean(np.abs(future))),
        "viscosity": float(viscosity),
        "dt": float(dt),
        "dx": float(dx),
    }


def residual_guided_refinement(
    *,
    context: Any,
    base_prediction: Any,
    viscosity: float = 0.001,
    dt: float = 0.05,
    dx: float = 1.0 / 256.0,
    correction_weight: float = 0.05,
    stop_step: int = 105,
    spectral_strength: float = 0.0,
) -> np.ndarray:
    context_array = _as_context(context)
    prediction = _as_trajectory(base_prediction).copy()
    if prediction.shape[0] != context_array.shape[0]:
        raise BurgersResidualRefinementError(
            "base_prediction sample count must match context"
        )
    if prediction.shape[2] != context_array.shape[2]:
        raise BurgersResidualRefinementError(
            "base_prediction spatial dimension must match context"
        )
    if prediction.shape[1] < context_array.shape[1] + 1:
        raise BurgersResidualRefinementError(
            "base_prediction must include forecast steps beyond context"
        )
    _validate_parameters(
        viscosity=viscosity,
        dt=dt,
        dx=dx,
        correction_weight=correction_weight,
        spectral_strength=spectral_strength,
    )
    stop = max(context_array.shape[1], min(int(stop_step), prediction.shape[1]))
    original_prediction = prediction.copy()
    prediction[:, : context_array.shape[1], :] = context_array
    lower = float(np.min(context_array)) - 1.0
    upper = float(np.max(context_array)) + 1.0
    for frame_index in range(context_array.shape[1], stop):
        previous = prediction[:, frame_index - 1, :].astype(np.float64, copy=False)
        physics_next = _explicit_burgers_next(
            previous,
            viscosity=float(viscosity),
            dt=float(dt),
            dx=float(dx),
        )
        base_next = prediction[:, frame_index, :].astype(np.float64, copy=False)
        refined_next = (
            (1.0 - float(correction_weight)) * base_next
            + float(correction_weight) * physics_next
        )
        prediction[:, frame_index, :] = np.clip(refined_next, lower, upper)
    if stop < prediction.shape[1]:
        offset = prediction[:, stop - 1 : stop, :] - original_prediction[
            :,
            stop - 1 : stop,
            :,
        ]
        horizon = np.arange(
            1,
            prediction.shape[1] - stop + 1,
            dtype=np.float32,
        ).reshape(1, -1, 1)
        handoff = np.exp(-(horizon - 1.0) / 8.0)
        prediction[:, stop:, :] = original_prediction[:, stop:, :] + offset * handoff
    if spectral_strength > 0.0 and stop > context_array.shape[1]:
        smoothed = _spectral_smooth_future(
            prediction=prediction,
            start=context_array.shape[1],
            stop=stop,
            strength=float(spectral_strength),
        )
        prediction[:, context_array.shape[1] : stop, :] = smoothed[
            :, context_array.shape[1] : stop, :
        ]
    prediction[:, : context_array.shape[1], :] = context_array
    if not np.all(np.isfinite(prediction)):
        raise BurgersResidualRefinementError("refined prediction contains non-finite values")
    return prediction.astype(np.float32, copy=False)


def physics_residual_refinement_params_from_any(params: dict[str, Any]) -> dict[str, Any]:
    base_family = str(params.get("base_family", "task1_retrieval_refinement"))
    if base_family not in {
        "linear_trend_spectral",
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
    }:
        raise BurgersResidualRefinementError(
            "base_family must be linear_trend_spectral, task1_retrieval_prior, "
            "or task1_retrieval_refinement"
        )
    return {
        **params,
        "base_family": base_family,
        "viscosity": float(params.get("viscosity", 0.001)),
        "dt": float(params.get("dt", 0.05)),
        "dx": float(params.get("dx", 1.0 / 256.0)),
        "correction_weight": float(params.get("correction_weight", 0.05)),
        "stop_step": int(params.get("stop_step", 105)),
        "residual_spectral_strength": float(
            params.get("residual_spectral_strength", 0.0)
        ),
    }


def burgers_residual_array(
    trajectory: Any,
    *,
    viscosity: float = 0.001,
    dt: float = 0.05,
    dx: float = 1.0 / 256.0,
) -> np.ndarray:
    array = _as_trajectory(trajectory).astype(np.float64, copy=False)
    _validate_parameters(
        viscosity=viscosity,
        dt=dt,
        dx=dx,
        correction_weight=0.0,
        spectral_strength=0.0,
    )
    u = array[:, :-1, :]
    u_next = array[:, 1:, :]
    u_t = (u_next - u) / float(dt)
    u_x = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2.0 * float(dx))
    u_xx = (
        np.roll(u, -1, axis=2) - (2.0 * u) + np.roll(u, 1, axis=2)
    ) / (float(dx) * float(dx))
    residual = u_t + (u * u_x) - (float(viscosity) * u_xx)
    if not np.all(np.isfinite(residual)):
        raise BurgersResidualRefinementError("Burgers residual contains non-finite values")
    return residual


def _explicit_burgers_next(
    state: np.ndarray,
    *,
    viscosity: float,
    dt: float,
    dx: float,
) -> np.ndarray:
    u_x = (np.roll(state, -1, axis=1) - np.roll(state, 1, axis=1)) / (2.0 * dx)
    u_xx = (np.roll(state, -1, axis=1) - (2.0 * state) + np.roll(state, 1, axis=1)) / (
        dx * dx
    )
    return state + dt * (-(state * u_x) + viscosity * u_xx)


def _spectral_smooth_future(
    *,
    prediction: np.ndarray,
    start: int,
    stop: int,
    strength: float,
) -> np.ndarray:
    smoothed = prediction.copy()
    future = smoothed[:, start:stop, :]
    modes = np.abs(np.fft.rfftfreq(future.shape[2]))
    if modes.max() <= 0.0:
        return smoothed
    damping = np.exp(-36.0 * strength * (modes / modes.max()) ** 8)
    transformed = np.fft.rfft(future, axis=2)
    smoothed[:, start:stop, :] = np.fft.irfft(
        transformed * damping.reshape(1, 1, -1),
        n=future.shape[2],
        axis=2,
    )
    return smoothed


def _as_context(context: Any) -> np.ndarray:
    array = np.asarray(context, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] < CONTEXT_STEPS or array.shape[2] < 4:
        raise BurgersResidualRefinementError("context must have shape (N, T>=10, X>=4)")
    if not np.all(np.isfinite(array)):
        raise BurgersResidualRefinementError("context contains non-finite values")
    return array[:, :CONTEXT_STEPS, :]


def _as_trajectory(trajectory: Any) -> np.ndarray:
    array = np.asarray(trajectory, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] < CONTEXT_STEPS + 1 or array.shape[2] < 4:
        raise BurgersResidualRefinementError(
            "trajectory must have shape (N, T>=11, X>=4)"
        )
    if not np.all(np.isfinite(array)):
        raise BurgersResidualRefinementError("trajectory contains non-finite values")
    return array


def _validate_parameters(
    *,
    viscosity: float,
    dt: float,
    dx: float,
    correction_weight: float,
    spectral_strength: float,
) -> None:
    values = {
        "viscosity": viscosity,
        "dt": dt,
        "dx": dx,
        "correction_weight": correction_weight,
        "spectral_strength": spectral_strength,
    }
    for name, value in values.items():
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise BurgersResidualRefinementError(f"{name} must be numeric")
        if not np.isfinite(float(value)):
            raise BurgersResidualRefinementError(f"{name} must be finite")
    if viscosity < 0.0:
        raise BurgersResidualRefinementError("viscosity must be non-negative")
    if dt <= 0.0 or dx <= 0.0:
        raise BurgersResidualRefinementError("dt and dx must be positive")
    if correction_weight < 0.0 or correction_weight > 1.0:
        raise BurgersResidualRefinementError("correction_weight must be in [0, 1]")
    if spectral_strength < 0.0:
        raise BurgersResidualRefinementError("spectral_strength must be non-negative")


def _rmse(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2)))
