"""Local Burgers-equation physics checks for rank-3 rollout arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class BurgersPhysicsError(RuntimeError):
    pass


def check_burgers_physics_npz(
    *,
    prediction_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
) -> dict[str, Any]:
    """Check local Burgers residual and conservation proxies from NPZ files."""

    prediction_file = Path(prediction_path)
    target_file = Path(target_path)
    prediction = _load_npz_array(prediction_file, "prediction")
    target = _load_npz_array(target_file, "target")

    result = check_burgers_physics_arrays(
        prediction=prediction,
        target=target,
        nu=nu,
        dt=dt,
        dx=dx,
    )
    result["prediction_path"] = str(prediction_file)
    result["target_path"] = str(target_file)

    if output_path is not None:
        physics_file = Path(output_path)
        result["physics_check_path"] = str(physics_file)
        physics_file.parent.mkdir(parents=True, exist_ok=True)
        physics_file.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return result


def check_burgers_physics_arrays(
    *,
    prediction: Any,
    target: Any,
    nu: float,
    dt: float,
    dx: float,
) -> dict[str, Any]:
    """Compute finite Burgers residual and rollout-shape physics metrics."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _validate_arrays(pred, truth)
    _validate_parameters(nu=nu, dt=dt, dx=dx)

    pred_residual = _burgers_residual(pred, nu=nu, dt=dt, dx=dx)
    target_residual = _burgers_residual(truth, nu=nu, dt=dt, dx=dx)
    error = pred - truth

    prediction_residual_rmse = _rmse(pred_residual)
    target_residual_rmse = _rmse(target_residual)
    residual_ratio = prediction_residual_rmse / (target_residual_rmse + 1e-12)
    mass_drift = _mass_drift(pred)
    energy_decay_ratio = float(
        np.mean(pred[:, -1, :] ** 2) / (np.mean(truth[:, -1, :] ** 2) + 1e-12)
    )
    total_variation_ratio = float(
        _total_variation(pred) / (_total_variation(truth) + 1e-12)
    )
    gradient_error_ratio = _gradient_error_ratio(pred, truth)
    shock_region_error_ratio = _shock_region_error_ratio(error, truth)

    metrics = [
        prediction_residual_rmse,
        target_residual_rmse,
        residual_ratio,
        mass_drift,
        energy_decay_ratio,
        total_variation_ratio,
        gradient_error_ratio,
        shock_region_error_ratio,
    ]
    _raise_if_non_finite(metrics, "derived Burgers physics metrics")

    physics_issue = _classify_physics_issue(
        total_variation_ratio=total_variation_ratio,
        energy_decay_ratio=energy_decay_ratio,
        gradient_error_ratio=gradient_error_ratio,
        residual_ratio=residual_ratio,
        mass_drift=mass_drift,
    )

    return {
        "sample_count": int(pred.shape[0]),
        "time_steps": int(pred.shape[1]),
        "space_points": int(pred.shape[2]),
        "nu": float(nu),
        "dt": float(dt),
        "dx": float(dx),
        "prediction_residual_rmse": prediction_residual_rmse,
        "target_residual_rmse": target_residual_rmse,
        "residual_ratio": residual_ratio,
        "mass_drift": mass_drift,
        "energy_decay_ratio": energy_decay_ratio,
        "total_variation_ratio": total_variation_ratio,
        "gradient_error_ratio": gradient_error_ratio,
        "shock_region_error_ratio": shock_region_error_ratio,
        "physics_issue": physics_issue,
        "notes": _build_notes(physics_issue),
    }


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise BurgersPhysicsError(f"missing path: {path}")
    if not path.is_file():
        raise BurgersPhysicsError(f"path is not a file: {path}")

    try:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise BurgersPhysicsError(f"{path} missing key '{key}'")
            return np.asarray(data[key])
    except BurgersPhysicsError:
        raise
    except OSError as exc:
        raise BurgersPhysicsError(f"failed to load NPZ file {path}: {exc}") from exc


def _validate_arrays(prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise BurgersPhysicsError(
            "shape mismatch: "
            f"prediction shape {prediction.shape} != target shape {target.shape}"
        )
    if prediction.ndim != 3:
        raise BurgersPhysicsError(
            f"expected rank 3 arrays shaped [N, T, X], got rank {prediction.ndim}"
        )
    if any(dimension <= 0 for dimension in prediction.shape):
        raise BurgersPhysicsError(
            f"expected non-empty [N, T, X] dimensions, got shape {prediction.shape}"
        )
    if prediction.shape[1] < 2:
        raise BurgersPhysicsError("expected at least 2 time steps")
    if prediction.shape[2] < 3:
        raise BurgersPhysicsError("expected at least 3 space points")
    if not np.all(np.isfinite(prediction)):
        raise BurgersPhysicsError("prediction contains non-finite values")
    if not np.all(np.isfinite(target)):
        raise BurgersPhysicsError("target contains non-finite values")


def _validate_parameters(*, nu: float, dt: float, dx: float) -> None:
    values = {"nu": nu, "dt": dt, "dx": dx}
    for name, value in values.items():
        if not isinstance(value, int | float) or not np.isfinite(value):
            raise BurgersPhysicsError(f"{name} must be finite")
    if nu < 0:
        raise BurgersPhysicsError("nu must be >= 0")
    if dt <= 0:
        raise BurgersPhysicsError("dt must be > 0")
    if dx <= 0:
        raise BurgersPhysicsError("dx must be > 0")


def _burgers_residual(array: np.ndarray, *, nu: float, dt: float, dx: float) -> np.ndarray:
    u = array[:, :-1, :]
    u_next = array[:, 1:, :]
    u_t = (u_next - u) / dt
    u_x = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2.0 * dx)
    u_xx = (
        np.roll(u, -1, axis=2) - (2.0 * u) + np.roll(u, 1, axis=2)
    ) / (dx * dx)
    residual = u_t + (u * u_x) - (nu * u_xx)
    _raise_if_non_finite(residual, "Burgers residual")
    return residual


def _rmse(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(array**2)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 1.0 if abs(numerator) <= 1e-12 else float(numerator / 1e-12)
    return float(numerator / denominator)


def _mass_drift(prediction: np.ndarray) -> float:
    initial_mass = np.mean(prediction[:, 0, :], axis=1)
    final_mass = np.mean(prediction[:, -1, :], axis=1)
    return float(np.mean(np.abs(final_mass - initial_mass)))


def _total_variation(array: np.ndarray) -> float:
    return float(np.mean(np.sum(np.abs(np.diff(array, axis=2)), axis=2)))


def _gradient_error_ratio(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_gradient = np.diff(prediction, axis=2)
    target_gradient = np.diff(target, axis=2)
    gradient_error = _rmse(pred_gradient - target_gradient)
    target_gradient_rmse = _rmse(target_gradient)
    return float(gradient_error / (target_gradient_rmse + 1e-12))


def _shock_region_error_ratio(error: np.ndarray, target: np.ndarray) -> float:
    abs_error = np.abs(error)
    gradient = np.abs(np.diff(target, axis=2, prepend=target[:, :, :1]))
    threshold = float(np.quantile(gradient, 0.8))
    shock_mask = gradient >= threshold
    if not np.any(shock_mask):
        return 1.0

    local_error = float(np.mean(abs_error[shock_mask]))
    global_error = float(np.mean(abs_error) + 1e-12)
    return local_error / global_error


def _classify_physics_issue(
    *,
    total_variation_ratio: float,
    energy_decay_ratio: float,
    gradient_error_ratio: float,
    residual_ratio: float,
    mass_drift: float,
) -> str:
    if total_variation_ratio < 0.5 or energy_decay_ratio < 0.5:
        return "overdissipation"
    if gradient_error_ratio > 2.0 and total_variation_ratio > 1.5:
        return "nonphysical_oscillation"
    if residual_ratio > 2.0:
        return "residual_mismatch"
    if mass_drift > 0.1:
        return "mass_drift"
    if residual_ratio < 1.25 and gradient_error_ratio < 0.25:
        return "stable_low_error"
    return "inconclusive"


def _build_notes(physics_issue: str) -> list[str]:
    return [
        "Finite-difference diagnostics are relative proxies on downsampled local windows.",
        f"classified_as={physics_issue}",
    ]


def _raise_if_non_finite(value: Any, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise BurgersPhysicsError(f"{label} contains non-finite values")
