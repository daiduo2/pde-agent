"""Lightweight spectral-element style Burgers rollout candidates."""

from __future__ import annotations

from typing import Any

import numpy as np

from pde_agent.core.competition_submission import CONTEXT_STEPS, PREDICTION_STEPS


class BurgersSpectralElementError(RuntimeError):
    pass


def sem_burgers_rollout(
    *,
    context: np.ndarray,
    viscosity: float = 0.001,
    dt: float = 0.05,
    element_count: int = 16,
    polynomial_order: int = 8,
    substeps: int = 4,
    filter_strength: float = 0.0,
    interface_penalty: float = 0.02,
) -> np.ndarray:
    """Roll out 1D Burgers from the last context frame.

    The executor uses element-local spectral collocation derivatives on the 256
    point competition grid. It is intentionally small and deterministic so it
    can be swept inside Hypothesis Lab before a heavier SEM implementation is
    justified.
    """

    context = _as_context_array(context)
    if viscosity < 0.0:
        raise BurgersSpectralElementError("viscosity must be non-negative")
    if dt <= 0.0:
        raise BurgersSpectralElementError("dt must be positive")
    if element_count < 1:
        raise BurgersSpectralElementError("element_count must be positive")
    if context.shape[2] % element_count != 0:
        raise BurgersSpectralElementError(
            "element_count must divide the spatial point count"
        )
    if polynomial_order < 1:
        raise BurgersSpectralElementError("polynomial_order must be positive")
    if substeps < 1:
        raise BurgersSpectralElementError("substeps must be positive")

    prediction = np.empty(
        (context.shape[0], PREDICTION_STEPS, context.shape[2]),
        dtype=np.float32,
    )
    prediction[:, :CONTEXT_STEPS, :] = context[:, :CONTEXT_STEPS, :]
    state = context[:, CONTEXT_STEPS - 1, :].astype(np.float64, copy=True)
    max_speed = max(float(np.max(np.abs(state))), 1e-6)
    cfl_substeps = int(np.ceil(float(dt) / (0.2 / (context.shape[2] * max_speed))))
    effective_substeps = max(int(substeps), cfl_substeps)
    step_dt = float(dt) / effective_substeps
    for output_index in range(CONTEXT_STEPS, PREDICTION_STEPS):
        for _ in range(effective_substeps):
            state = _rk4_step(
                state=state,
                dt=step_dt,
                viscosity=float(viscosity),
                element_count=int(element_count),
            )
            state = _interface_blend(
                state=state,
                element_count=int(element_count),
                strength=float(interface_penalty),
            )
        if filter_strength > 0.0:
            state = _spectral_filter(state=state, strength=float(filter_strength))
        prediction[:, output_index, :] = state.astype(np.float32, copy=False)
    prediction[:, :CONTEXT_STEPS, :] = context[:, :CONTEXT_STEPS, :]
    return prediction


def sem_params_from_any(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "viscosity": float(params.get("viscosity", 0.001)),
        "dt": float(params.get("dt", 0.05)),
        "element_count": int(params.get("element_count", 16)),
        "polynomial_order": int(params.get("polynomial_order", 8)),
        "substeps": int(params.get("substeps", 4)),
        "filter_strength": float(params.get("filter_strength", 0.0)),
        "interface_penalty": float(params.get("interface_penalty", 0.02)),
    }


def _rk4_step(
    *,
    state: np.ndarray,
    dt: float,
    viscosity: float,
    element_count: int,
) -> np.ndarray:
    k1 = _burgers_rhs(state=state, viscosity=viscosity, element_count=element_count)
    k2 = _burgers_rhs(
        state=state + 0.5 * dt * k1,
        viscosity=viscosity,
        element_count=element_count,
    )
    k3 = _burgers_rhs(
        state=state + 0.5 * dt * k2,
        viscosity=viscosity,
        element_count=element_count,
    )
    k4 = _burgers_rhs(
        state=state + dt * k3,
        viscosity=viscosity,
        element_count=element_count,
    )
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(
        np.nan_to_num(next_state, nan=0.0, posinf=10.0, neginf=-10.0),
        -10.0,
        10.0,
    )


def _burgers_rhs(
    *,
    state: np.ndarray,
    viscosity: float,
    element_count: int,
) -> np.ndarray:
    ux, uxx = _element_local_spectral_derivatives(
        state=state,
        element_count=element_count,
    )
    return -state * ux + viscosity * uxx


def _element_local_spectral_derivatives(
    *,
    state: np.ndarray,
    element_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count, point_count = state.shape
    points_per_element = point_count // element_count
    elements = state.reshape(sample_count, element_count, points_per_element)
    dx = 1.0 / float(point_count)
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(points_per_element, d=dx)
    transformed = np.fft.fft(elements, axis=-1)
    ux = np.fft.ifft(1j * wave_numbers * transformed, axis=-1).real
    uxx = np.fft.ifft(-(wave_numbers**2) * transformed, axis=-1).real
    return (
        ux.reshape(sample_count, point_count),
        uxx.reshape(sample_count, point_count),
    )


def _interface_blend(
    *,
    state: np.ndarray,
    element_count: int,
    strength: float,
) -> np.ndarray:
    if element_count <= 1 or strength <= 0.0:
        return state
    sample_count, point_count = state.shape
    points_per_element = point_count // element_count
    elements = state.reshape(sample_count, element_count, points_per_element).copy()
    left = elements[:, :, 0]
    right = elements[:, :, -1]
    next_left = np.roll(left, shift=-1, axis=1)
    interface_average = 0.5 * (right + next_left)
    elements[:, :, -1] = (1.0 - strength) * right + strength * interface_average
    elements[:, :, 0] = (1.0 - strength) * left + strength * np.roll(
        interface_average,
        shift=1,
        axis=1,
    )
    return elements.reshape(sample_count, point_count)


def _spectral_filter(*, state: np.ndarray, strength: float) -> np.ndarray:
    point_count = state.shape[1]
    modes = np.abs(np.fft.fftfreq(point_count))
    if modes.max() == 0.0:
        return state
    scaled = modes / modes.max()
    damping = np.exp(-36.0 * float(strength) * scaled**8)
    transformed = np.fft.fft(state, axis=1)
    return np.fft.ifft(transformed * damping.reshape(1, -1), axis=1).real


def _as_context_array(context: np.ndarray) -> np.ndarray:
    array = np.asarray(context, dtype=np.float32)
    if array.ndim != 3:
        raise BurgersSpectralElementError("context must have shape (N, T, X)")
    if array.shape[1] < CONTEXT_STEPS:
        raise BurgersSpectralElementError("context must include at least 10 steps")
    if array.shape[2] <= 0:
        raise BurgersSpectralElementError("context spatial dimension must be positive")
    return array[:, :CONTEXT_STEPS, :]
