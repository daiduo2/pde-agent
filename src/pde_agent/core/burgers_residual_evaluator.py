"""Target-free Burgers residual diagnostics for PI/PINN-style evidence."""


import json
from pathlib import Path
from typing import Any

import numpy as np


STATE_HINT = "BURGERS_RESIDUAL_EVALUATED"


class BurgersResidualEvaluationError(RuntimeError):
    pass


def evaluate_burgers_residual_file(
    *,
    trajectory_path: str | Path,
    output_path: str | Path,
    array_key: str,
    nu: float,
    dt: float,
    dx: float,
) -> dict[str, Any]:
    trajectory_file = Path(trajectory_path)
    output_file = Path(output_path)
    trajectory, source_format = _load_trajectory(trajectory_file, array_key)
    result = evaluate_burgers_residual_arrays(
        trajectory=trajectory,
        nu=nu,
        dt=dt,
        dx=dx,
    )
    result.update(
        {
            "trajectory_path": str(trajectory_file),
            "array_key": array_key,
            "source_format": source_format,
            "burgers_residual_path": str(output_file),
        }
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def evaluate_burgers_residual_arrays(
    *,
    trajectory: Any,
    nu: float,
    dt: float,
    dx: float,
) -> dict[str, Any]:
    array = np.asarray(trajectory, dtype=np.float64)
    _validate_trajectory(array)
    _validate_parameters(nu=nu, dt=dt, dx=dx)

    residual = _burgers_residual(array, nu=nu, dt=dt, dx=dx)
    energy = np.mean(array**2, axis=2)
    total_variation = np.sum(np.abs(np.diff(array, axis=2)), axis=2)
    residual_rmse = _rmse(residual)
    residual_mean_abs = float(np.mean(np.abs(residual)))
    mass_drift = _mass_drift(array)
    initial_energy = float(np.mean(energy[:, 0]))
    final_energy = float(np.mean(energy[:, -1]))
    energy_growth_ratio = _safe_ratio(final_energy, initial_energy)
    tv_growth_ratio = _safe_ratio(
        float(np.mean(total_variation[:, -1])),
        float(np.mean(total_variation[:, 0])),
    )
    high_frequency_energy_ratio = _high_frequency_energy_ratio(array)

    metrics = [
        residual_rmse,
        residual_mean_abs,
        mass_drift,
        initial_energy,
        final_energy,
        energy_growth_ratio,
        tv_growth_ratio,
        high_frequency_energy_ratio,
    ]
    if not np.all(np.isfinite(metrics)):
        raise BurgersResidualEvaluationError("derived metrics contain non-finite values")

    physics_issue = _classify_issue(
        residual_rmse=residual_rmse,
        mass_drift=mass_drift,
        energy_growth_ratio=energy_growth_ratio,
        total_variation_growth_ratio=tv_growth_ratio,
        high_frequency_energy_ratio=high_frequency_energy_ratio,
    )
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "sample_count": int(array.shape[0]),
        "time_steps": int(array.shape[1]),
        "space_points": int(array.shape[2]),
        "nu": float(nu),
        "dt": float(dt),
        "dx": float(dx),
        "residual_rmse": residual_rmse,
        "residual_mean_abs": residual_mean_abs,
        "mass_drift": mass_drift,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_growth_ratio": energy_growth_ratio,
        "total_variation_growth_ratio": tv_growth_ratio,
        "high_frequency_energy_ratio": high_frequency_energy_ratio,
        "physics_issue": physics_issue,
        "next_command": "select-pde-hypothesis",
    }


def _load_trajectory(path: Path, array_key: str) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise BurgersResidualEvaluationError(f"trajectory file not found: {path}")
    if not isinstance(array_key, str) or not array_key.strip():
        raise BurgersResidualEvaluationError("array_key must be a non-empty string")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if array_key not in archive.files:
                raise BurgersResidualEvaluationError(
                    f"NPZ trajectory missing key: {array_key}"
                )
            return np.asarray(archive[array_key]), "npz"
    if suffix in {".h5", ".hdf5"}:
        h5py = _import_h5py()
        with h5py.File(path, "r") as handle:
            if array_key not in handle:
                raise BurgersResidualEvaluationError(
                    f"HDF5 trajectory missing key: {array_key}"
                )
            return np.asarray(handle[array_key]), "hdf5"
    raise BurgersResidualEvaluationError(
        f"unsupported trajectory file suffix: {path.suffix}"
    )


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise BurgersResidualEvaluationError(
            "h5py is required for HDF5 residual evaluation"
        ) from exc
    return h5py


def _validate_trajectory(array: np.ndarray) -> None:
    if array.ndim != 3:
        raise BurgersResidualEvaluationError(
            f"expected trajectory shape [N, T, X], got rank {array.ndim}"
        )
    if array.shape[0] <= 0 or array.shape[1] < 2 or array.shape[2] < 4:
        raise BurgersResidualEvaluationError(
            f"expected non-empty [N, T>=2, X>=4], got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise BurgersResidualEvaluationError("trajectory contains non-finite values")


def _validate_parameters(*, nu: float, dt: float, dx: float) -> None:
    for name, value in {"nu": nu, "dt": dt, "dx": dx}.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise BurgersResidualEvaluationError(f"{name} must be numeric")
        if not np.isfinite(value):
            raise BurgersResidualEvaluationError(f"{name} must be finite")
    if nu < 0:
        raise BurgersResidualEvaluationError("nu must be >= 0")
    if dt <= 0:
        raise BurgersResidualEvaluationError("dt must be > 0")
    if dx <= 0:
        raise BurgersResidualEvaluationError("dx must be > 0")


def _burgers_residual(array: np.ndarray, *, nu: float, dt: float, dx: float) -> np.ndarray:
    u = array[:, :-1, :]
    u_next = array[:, 1:, :]
    u_t = (u_next - u) / dt
    u_x = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2.0 * dx)
    u_xx = (
        np.roll(u, -1, axis=2) - (2.0 * u) + np.roll(u, 1, axis=2)
    ) / (dx * dx)
    residual = u_t + (u * u_x) - (nu * u_xx)
    if not np.all(np.isfinite(residual)):
        raise BurgersResidualEvaluationError("Burgers residual contains non-finite values")
    return residual


def _rmse(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(array**2)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 1.0 if abs(numerator) <= 1e-12 else float(numerator / 1e-12)
    return float(numerator / denominator)


def _mass_drift(array: np.ndarray) -> float:
    initial_mass = np.mean(array[:, 0, :], axis=1)
    final_mass = np.mean(array[:, -1, :], axis=1)
    return float(np.mean(np.abs(final_mass - initial_mass)))


def _high_frequency_energy_ratio(array: np.ndarray) -> float:
    spectrum = np.fft.rfft(array, axis=2)
    energy = np.abs(spectrum) ** 2
    if energy.shape[2] <= 1:
        return 0.0
    cutoff = max(1, int(np.ceil(energy.shape[2] * 0.6)))
    high_energy = float(np.sum(energy[:, :, cutoff:]))
    total_energy = float(np.sum(energy[:, :, 1:]))
    if total_energy <= 1e-12:
        return 0.0
    return float(high_energy / total_energy)


def _classify_issue(
    *,
    residual_rmse: float,
    mass_drift: float,
    energy_growth_ratio: float,
    total_variation_growth_ratio: float,
    high_frequency_energy_ratio: float,
) -> str:
    if high_frequency_energy_ratio > 0.75:
        return "high_frequency_instability"
    if energy_growth_ratio > 1.5 or total_variation_growth_ratio > 2.0:
        return "blowup_like_behavior"
    if mass_drift > 0.1:
        return "mass_drift"
    if residual_rmse < 1e-8:
        return "stable_low_residual"
    return "residual_mismatch"
