"""Numpy PI-DeepONet-style correction head for bounded Task 1 sandboxing."""


from typing import Any

import numpy as np

from pde_agent.core.burgers_residual_refinement import (
    burgers_residual_summary,
    residual_guided_refinement,
)
from pde_agent.core.competition_submission import CONTEXT_STEPS


class PIDeepONetCorrectionHeadError(RuntimeError):
    pass


def fit_pi_deeponet_correction_head(
    *,
    context: Any,
    base_prediction: Any,
    target: Any,
    ridge_alpha: float = 1e-4,
    correction_weight: float = 0.25,
    physics_weight: float = 0.0,
    viscosity: float = 0.001,
    dt: float = 0.05,
    dx: float = 1.0 / 256.0,
) -> dict[str, Any]:
    context_array = _as_context(context)
    base_array = _as_prediction(base_prediction, context_array=context_array)
    target_array = _as_prediction(target, context_array=context_array)
    alpha = _non_negative_float(ridge_alpha, "ridge_alpha")
    correction_w = _bounded_float(correction_weight, "correction_weight")
    physics_w = _bounded_float(physics_weight, "physics_weight")
    pde_nu = _non_negative_float(viscosity, "viscosity")
    pde_dt = _positive_float(dt, "dt")
    pde_dx = _positive_float(dx, "dx")

    branch = _branch_features(context_array)
    trunk = _trunk_features(
        forecast_steps=base_array.shape[1] - context_array.shape[1],
        space_points=base_array.shape[2],
    )
    target_correction = (
        target_array[:, context_array.shape[1] :, :]
        - base_array[:, context_array.shape[1] :, :]
    ).reshape(context_array.shape[0], -1)
    weights = _fit_branch_trunk_ridge(
        branch=branch,
        trunk=trunk,
        target=target_correction,
        ridge_alpha=alpha,
    )
    learned_correction = (branch @ weights @ trunk.T).reshape(
        context_array.shape[0],
        base_array.shape[1] - context_array.shape[1],
        base_array.shape[2],
    )
    prediction = base_array.copy()
    prediction[:, context_array.shape[1] :, :] = (
        prediction[:, context_array.shape[1] :, :]
        + correction_w * learned_correction.astype(np.float32, copy=False)
    )
    if physics_w > 0.0:
        physics_prediction = residual_guided_refinement(
            context=context_array,
            base_prediction=prediction,
            viscosity=pde_nu,
            dt=pde_dt,
            dx=pde_dx,
            correction_weight=min(physics_w, 0.1),
            stop_step=min(prediction.shape[1], 57),
            spectral_strength=0.0,
        )
        prediction[:, context_array.shape[1] :, :] = (
            (1.0 - physics_w) * prediction[:, context_array.shape[1] :, :]
            + physics_w * physics_prediction[:, context_array.shape[1] :, :]
        )
    prediction[:, : context_array.shape[1], :] = context_array
    if not np.all(np.isfinite(prediction)):
        raise PIDeepONetCorrectionHeadError("prediction contains non-finite values")

    base_residual = burgers_residual_summary(
        base_array,
        viscosity=pde_nu,
        dt=pde_dt,
        dx=pde_dx,
    )
    candidate_residual = burgers_residual_summary(
        prediction,
        viscosity=pde_nu,
        dt=pde_dt,
        dx=pde_dx,
    )
    return {
        "prediction": prediction.astype(np.float32, copy=False),
        "metadata": {
            "adapter_type": "numpy_pi_deeponet_correction_head",
            "train_mode": "ridge_branch_trunk_correction",
            "branch_feature_count": int(branch.shape[1]),
            "trunk_feature_count": int(trunk.shape[1]),
            "ridge_alpha": alpha,
            "correction_weight": correction_w,
            "physics_weight": physics_w,
            "learned_correction_rmse": float(
                np.sqrt(np.mean(learned_correction.astype(np.float64) ** 2))
            ),
            "viscosity": pde_nu,
            "dt": pde_dt,
            "dx": pde_dx,
            "base_residual_rmse": base_residual["residual_rmse"],
            "candidate_residual_rmse": candidate_residual["residual_rmse"],
            "residual_rmse_delta": (
                float(candidate_residual["residual_rmse"])
                - float(base_residual["residual_rmse"])
            ),
        },
    }


def pi_deeponet_params_from_any(params: dict[str, Any]) -> dict[str, Any]:
    base_family = str(params.get("base_family", "task1_retrieval_refinement"))
    if base_family not in {
        "linear_trend_spectral",
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
    }:
        raise PIDeepONetCorrectionHeadError(
            "base_family must be linear_trend_spectral, task1_retrieval_prior, "
            "or task1_retrieval_refinement"
        )
    return {
        **params,
        "base_family": base_family,
        "ridge_alpha": float(params.get("ridge_alpha", 1e-4)),
        "correction_weight": float(params.get("correction_weight", 0.25)),
        "physics_weight": float(params.get("physics_weight", 0.0)),
        "viscosity": float(params.get("viscosity", 0.001)),
        "dt": float(params.get("dt", 0.05)),
        "dx": float(params.get("dx", 1.0 / 256.0)),
    }


def _fit_branch_trunk_ridge(
    *,
    branch: np.ndarray,
    trunk: np.ndarray,
    target: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    gram_branch = branch.T @ branch
    gram_trunk = trunk.T @ trunk
    lhs = np.kron(gram_trunk, gram_branch)
    rhs = (branch.T @ target @ trunk).reshape(-1, order="F")
    lhs = lhs + float(ridge_alpha) * np.eye(lhs.shape[0], dtype=np.float64)
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return solution.reshape(branch.shape[1], trunk.shape[1], order="F")


def _branch_features(context: np.ndarray) -> np.ndarray:
    last = context[:, -1, :]
    previous = context[:, -2, :] if context.shape[1] >= 2 else last
    velocity = last - previous
    features = np.stack(
        [
            np.ones(context.shape[0], dtype=np.float64),
            np.mean(last, axis=1),
            np.std(last, axis=1),
            np.mean(velocity, axis=1),
            np.std(velocity, axis=1),
            np.min(last, axis=1),
            np.max(last, axis=1),
        ],
        axis=1,
    )
    return _normalize_columns(features)


def _trunk_features(*, forecast_steps: int, space_points: int) -> np.ndarray:
    if forecast_steps < 1:
        raise PIDeepONetCorrectionHeadError("forecast_steps must be positive")
    t = np.linspace(0.0, 1.0, forecast_steps, dtype=np.float64)
    x = np.linspace(0.0, 1.0, space_points, endpoint=False, dtype=np.float64)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    features = np.stack(
        [
            np.ones_like(tt),
            tt,
            tt * tt,
            np.sin(2.0 * np.pi * xx),
            np.cos(2.0 * np.pi * xx),
            tt * np.sin(2.0 * np.pi * xx),
            tt * np.cos(2.0 * np.pi * xx),
        ],
        axis=2,
    ).reshape(forecast_steps * space_points, -1)
    return _normalize_columns(features)


def _normalize_columns(features: np.ndarray) -> np.ndarray:
    normalized = np.asarray(features, dtype=np.float64).copy()
    for index in range(1, normalized.shape[1]):
        scale = float(np.std(normalized[:, index]))
        if scale > 1e-12:
            normalized[:, index] = (
                normalized[:, index] - float(np.mean(normalized[:, index]))
            ) / scale
    return normalized


def _as_context(context: Any) -> np.ndarray:
    array = np.asarray(context, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] < CONTEXT_STEPS or array.shape[2] < 4:
        raise PIDeepONetCorrectionHeadError("context must have shape (N, T>=10, X>=4)")
    if not np.all(np.isfinite(array)):
        raise PIDeepONetCorrectionHeadError("context contains non-finite values")
    return array[:, :CONTEXT_STEPS, :]


def _as_prediction(prediction: Any, *, context_array: np.ndarray) -> np.ndarray:
    array = np.asarray(prediction, dtype=np.float32)
    if array.ndim != 3:
        raise PIDeepONetCorrectionHeadError("prediction must have rank 3")
    if array.shape[0] != context_array.shape[0] or array.shape[2] != context_array.shape[2]:
        raise PIDeepONetCorrectionHeadError(
            "prediction must share sample and space dimensions with context"
        )
    if array.shape[1] <= context_array.shape[1]:
        raise PIDeepONetCorrectionHeadError("prediction must include forecast steps")
    if not np.all(np.isfinite(array)):
        raise PIDeepONetCorrectionHeadError("prediction contains non-finite values")
    return array


def _positive_float(value: float, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise PIDeepONetCorrectionHeadError(f"{label} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise PIDeepONetCorrectionHeadError(f"{label} must be positive")
    return value


def _non_negative_float(value: float, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise PIDeepONetCorrectionHeadError(f"{label} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise PIDeepONetCorrectionHeadError(f"{label} must be non-negative")
    return value


def _bounded_float(value: float, label: str) -> float:
    value = _non_negative_float(value, label)
    if value > 1.0:
        raise PIDeepONetCorrectionHeadError(f"{label} must be <= 1")
    return value
