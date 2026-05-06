"""Global Fourier pseudo-spectral Burgers rollout candidates."""

from __future__ import annotations

from typing import Any

import numpy as np

from pde_agent.core.competition_submission import CONTEXT_STEPS, PREDICTION_STEPS


class BurgersPseudospectralError(RuntimeError):
    pass


def pseudospectral_burgers_rollout(
    *,
    context: np.ndarray,
    viscosity: float = 0.001,
    dt: float = 0.05,
    substeps: int = 4,
    filter_strength: float = 0.0,
    dealias_fraction: float = 2.0 / 3.0,
    prediction_steps: int = PREDICTION_STEPS,
) -> np.ndarray:
    context_array = _as_context_array(context)
    output_steps = int(prediction_steps)
    if output_steps < CONTEXT_STEPS:
        raise BurgersPseudospectralError("prediction_steps must include context steps")
    if output_steps > PREDICTION_STEPS:
        raise BurgersPseudospectralError(
            f"prediction_steps must be <= {PREDICTION_STEPS}"
        )
    if viscosity < 0.0:
        raise BurgersPseudospectralError("viscosity must be non-negative")
    if dt <= 0.0:
        raise BurgersPseudospectralError("dt must be positive")
    if substeps < 1:
        raise BurgersPseudospectralError("substeps must be positive")
    prediction = np.empty(
        (context_array.shape[0], output_steps, context_array.shape[2]),
        dtype=np.float32,
    )
    prediction[:, :CONTEXT_STEPS, :] = context_array
    state = context_array[:, CONTEXT_STEPS - 1, :].astype(np.float64, copy=True)
    step_dt = float(dt) / int(substeps)
    wave_numbers = _wave_numbers(context_array.shape[2])
    dealias_mask = _dealias_mask(context_array.shape[2], fraction=float(dealias_fraction))
    lower = float(np.min(context_array)) - 1.0
    upper = float(np.max(context_array)) + 1.0
    for output_index in range(CONTEXT_STEPS, output_steps):
        for _ in range(int(substeps)):
            state = _bounded_state(
                state=_rk4_step(
                    state=state,
                    dt=step_dt,
                    viscosity=float(viscosity),
                    wave_numbers=wave_numbers,
                    dealias_mask=dealias_mask,
                ),
                lower=lower,
                upper=upper,
            )
        if filter_strength > 0.0:
            state = _spectral_filter(state=state, strength=float(filter_strength))
        state = _bounded_state(state=state, lower=lower, upper=upper)
        prediction[:, output_index, :] = state.astype(np.float32, copy=False)
    prediction[:, :CONTEXT_STEPS, :] = context_array
    return prediction


def pseudospectral_params_from_any(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "viscosity": float(params.get("viscosity", 0.001)),
        "dt": float(params.get("dt", 0.05)),
        "substeps": int(params.get("substeps", 4)),
        "filter_strength": float(params.get("filter_strength", 0.0)),
        "dealias_fraction": float(params.get("dealias_fraction", 2.0 / 3.0)),
    }


def _rk4_step(
    *,
    state: np.ndarray,
    dt: float,
    viscosity: float,
    wave_numbers: np.ndarray,
    dealias_mask: np.ndarray,
) -> np.ndarray:
    k1 = _burgers_rhs(
        state=state,
        viscosity=viscosity,
        wave_numbers=wave_numbers,
        dealias_mask=dealias_mask,
    )
    k2 = _burgers_rhs(
        state=state + 0.5 * dt * k1,
        viscosity=viscosity,
        wave_numbers=wave_numbers,
        dealias_mask=dealias_mask,
    )
    k3 = _burgers_rhs(
        state=state + 0.5 * dt * k2,
        viscosity=viscosity,
        wave_numbers=wave_numbers,
        dealias_mask=dealias_mask,
    )
    k4 = _burgers_rhs(
        state=state + dt * k3,
        viscosity=viscosity,
        wave_numbers=wave_numbers,
        dealias_mask=dealias_mask,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _burgers_rhs(
    *,
    state: np.ndarray,
    viscosity: float,
    wave_numbers: np.ndarray,
    dealias_mask: np.ndarray,
) -> np.ndarray:
    transformed = np.fft.fft(state, axis=1)
    ux = np.fft.ifft(1j * wave_numbers.reshape(1, -1) * transformed, axis=1).real
    uxx = np.fft.ifft(
        -(wave_numbers.reshape(1, -1) ** 2) * transformed,
        axis=1,
    ).real
    nonlinear = state * ux
    nonlinear_hat = np.fft.fft(nonlinear, axis=1) * dealias_mask.reshape(1, -1)
    nonlinear_filtered = np.fft.ifft(nonlinear_hat, axis=1).real
    return -nonlinear_filtered + float(viscosity) * uxx


def _wave_numbers(point_count: int) -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(point_count, d=1.0 / float(point_count))


def _dealias_mask(point_count: int, *, fraction: float) -> np.ndarray:
    fraction = min(1.0, max(0.0, float(fraction)))
    modes = np.abs(np.fft.fftfreq(point_count))
    cutoff = fraction * float(np.max(modes))
    return (modes <= cutoff).astype(np.float64)


def _spectral_filter(*, state: np.ndarray, strength: float) -> np.ndarray:
    point_count = state.shape[1]
    modes = np.abs(np.fft.fftfreq(point_count))
    if modes.max() == 0.0:
        return state
    damping = np.exp(-36.0 * float(strength) * (modes / modes.max()) ** 8)
    transformed = np.fft.fft(state, axis=1)
    return np.fft.ifft(transformed * damping.reshape(1, -1), axis=1).real


def _bounded_state(*, state: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(
        np.nan_to_num(state, nan=0.0, posinf=upper, neginf=lower),
        lower,
        upper,
    )


def _as_context_array(context: np.ndarray) -> np.ndarray:
    array = np.asarray(context, dtype=np.float32)
    if array.ndim != 3:
        raise BurgersPseudospectralError("context must have shape (N, T, X)")
    if array.shape[1] < CONTEXT_STEPS:
        raise BurgersPseudospectralError("context must include at least 10 steps")
    if array.shape[2] <= 0:
        raise BurgersPseudospectralError("context spatial dimension must be positive")
    return array[:, :CONTEXT_STEPS, :]
