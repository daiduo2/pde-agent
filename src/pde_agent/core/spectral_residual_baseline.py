from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_TYPE = "spectral_residual_ridge"


class SpectralResidualBaselineError(RuntimeError):
    pass


def train_spectral_residual_baseline(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    modes: int = 16,
    ridge: float = 1e-3,
    force: bool = False,
) -> dict[str, Any]:
    windows = Path(windows_path)
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists() and not force:
        raise SpectralResidualBaselineError(
            f"checkpoint path already exists: {checkpoint}"
        )

    context, target = _load_windows(windows)
    sample_count, context_steps, space_points = context.shape
    forecast_steps = target.shape[1]
    mode_count = _validate_modes(modes, space_points)
    ridge_value = _validate_ridge(ridge)

    features = _build_features(context, mode_count)
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
    normalized = (features - feature_mean) / feature_scale
    design = np.concatenate(
        [np.ones((sample_count, 1), dtype=np.float64), normalized],
        axis=1,
    )

    persistence = np.repeat(context[:, -1:, :], repeats=forecast_steps, axis=1)
    residual = target - persistence
    residual_coeffs = np.fft.rfft(residual, axis=2)[:, :, :mode_count]
    target_real = residual_coeffs.real.reshape(sample_count, -1)
    target_imag = residual_coeffs.imag.reshape(sample_count, -1)

    weights_real = _fit_ridge(design, target_real, ridge_value)
    weights_imag = _fit_ridge(design, target_imag, ridge_value)
    metadata = {
        "model_type": MODEL_TYPE,
        "modes": mode_count,
        "ridge": ridge_value,
        "context_steps": int(context_steps),
        "forecast_steps": int(forecast_steps),
        "space_points": int(space_points),
        "feature_dim": int(features.shape[1]),
        "train_sample_count": int(sample_count),
        "windows_path": str(windows),
    }

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as handle:
        np.savez(
            handle,
            weights_real=weights_real.astype(np.float64, copy=False),
            weights_imag=weights_imag.astype(np.float64, copy=False),
            feature_mean=feature_mean.astype(np.float64, copy=False),
            feature_scale=feature_scale.astype(np.float64, copy=False),
            metadata_json=json.dumps(
                metadata,
                allow_nan=False,
                sort_keys=True,
            ),
        )

    return {
        "checkpoint_path": str(checkpoint),
        "train_sample_count": int(sample_count),
        "context_shape": [int(axis) for axis in context.shape],
        "target_shape": [int(axis) for axis in target.shape],
        **metadata,
    }


def predict_spectral_residual_baseline(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    windows = Path(windows_path)
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    if output.exists() and not force:
        raise SpectralResidualBaselineError(f"output path already exists: {output}")

    context, target = _load_windows(windows)
    checkpoint_data = _load_checkpoint(checkpoint)
    metadata = checkpoint_data["metadata"]
    _validate_checkpoint_compatible(metadata, context, target)

    mode_count = int(metadata["modes"])
    forecast_steps = int(metadata["forecast_steps"])
    space_points = int(metadata["space_points"])
    features = _build_features(context, mode_count)
    feature_mean = checkpoint_data["feature_mean"]
    feature_scale = checkpoint_data["feature_scale"]
    if features.shape[1] != feature_mean.shape[0] or feature_mean.shape != feature_scale.shape:
        raise SpectralResidualBaselineError("checkpoint is incompatible with features")

    normalized = (features - feature_mean) / feature_scale
    design = np.concatenate(
        [np.ones((features.shape[0], 1), dtype=np.float64), normalized],
        axis=1,
    )
    weights_real = checkpoint_data["weights_real"]
    weights_imag = checkpoint_data["weights_imag"]
    if weights_real.shape != weights_imag.shape or weights_real.shape[0] != design.shape[1]:
        raise SpectralResidualBaselineError("checkpoint is incompatible with features")

    flat_real = design @ weights_real
    flat_imag = design @ weights_imag
    expected_width = forecast_steps * mode_count
    if flat_real.shape[1] != expected_width:
        raise SpectralResidualBaselineError("checkpoint is incompatible with horizon")

    low_coeffs = (flat_real + 1j * flat_imag).reshape(
        context.shape[0],
        forecast_steps,
        mode_count,
    )
    full_coeffs = np.zeros(
        (context.shape[0], forecast_steps, space_points // 2 + 1),
        dtype=np.complex128,
    )
    full_coeffs[:, :, :mode_count] = low_coeffs
    residual = np.fft.irfft(full_coeffs, n=space_points, axis=2)
    persistence = np.repeat(context[:, -1:, :], repeats=forecast_steps, axis=1)
    prediction = (persistence + residual).astype(np.float32)
    output_metadata = {
        "model_type": MODEL_TYPE,
        "windows_path": str(windows),
        "checkpoint_path": str(checkpoint),
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction,
            metadata_json=json.dumps(
                output_metadata,
                allow_nan=False,
                sort_keys=True,
            ),
        )

    return {
        "prediction_path": str(output),
        "prediction_shape": [int(axis) for axis in prediction.shape],
        **output_metadata,
    }


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SpectralResidualBaselineError(f"missing windows path: {path}")
    if not path.is_file():
        raise SpectralResidualBaselineError(f"windows path is not a file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in ("context", "target"):
                if key not in archive.files:
                    raise SpectralResidualBaselineError(f"{path} missing key '{key}'")
            context = np.asarray(archive["context"], dtype=np.float64)
            target = np.asarray(archive["target"], dtype=np.float64)
    except SpectralResidualBaselineError:
        raise
    except (OSError, ValueError) as exc:
        raise SpectralResidualBaselineError(
            f"failed to load windows NPZ {path}: {exc}"
        ) from exc

    _validate_windows(context, target)
    return context, target


def _validate_windows(context: np.ndarray, target: np.ndarray) -> None:
    for name, array in (("context", context), ("target", target)):
        if array.ndim != 3:
            raise SpectralResidualBaselineError(
                f"{name} must be rank 3 shaped [N, T, X]"
            )
        if any(axis <= 0 for axis in array.shape):
            raise SpectralResidualBaselineError(f"{name} dimensions must be non-empty")
        if not np.all(np.isfinite(array)):
            raise SpectralResidualBaselineError(f"{name} contains non-finite values")
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise SpectralResidualBaselineError(
            f"context/target shape mismatch: {context.shape} vs {target.shape}"
        )
    if context.shape[1] < 2:
        raise SpectralResidualBaselineError("context must contain at least 2 steps")


def _validate_modes(modes: int, space_points: int) -> int:
    if not isinstance(modes, int) or isinstance(modes, bool):
        raise SpectralResidualBaselineError("modes must be an integer")
    max_modes = space_points // 2 + 1
    if modes < 1 or modes > max_modes:
        raise SpectralResidualBaselineError(
            f"modes must be in [1, {max_modes}], got {modes}"
        )
    return modes


def _validate_ridge(ridge: float) -> float:
    value = float(ridge)
    if not np.isfinite(value) or value < 0.0:
        raise SpectralResidualBaselineError("ridge must be finite and non-negative")
    return value


def _build_features(context: np.ndarray, modes: int) -> np.ndarray:
    final = context[:, -1, :]
    delta = context[:, -1, :] - context[:, -2, :]
    final_coeffs = np.fft.rfft(final, axis=1)[:, :modes]
    delta_coeffs = np.fft.rfft(delta, axis=1)[:, :modes]
    return np.concatenate(
        [
            np.mean(final, axis=1, keepdims=True),
            np.mean(delta, axis=1, keepdims=True),
            final_coeffs.real,
            final_coeffs.imag,
            delta_coeffs.real,
            delta_coeffs.imag,
        ],
        axis=1,
    ).astype(np.float64, copy=False)


def _fit_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SpectralResidualBaselineError(f"missing checkpoint path: {path}")
    if not path.is_file():
        raise SpectralResidualBaselineError(f"checkpoint path is not a file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "weights_real",
                "weights_imag",
                "feature_mean",
                "feature_scale",
                "metadata_json",
            }
            missing = required.difference(archive.files)
            if missing:
                raise SpectralResidualBaselineError(
                    f"checkpoint missing keys: {sorted(missing)}"
                )
            metadata = json.loads(str(archive["metadata_json"]))
            return {
                "weights_real": np.asarray(archive["weights_real"], dtype=np.float64),
                "weights_imag": np.asarray(archive["weights_imag"], dtype=np.float64),
                "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
                "feature_scale": np.asarray(archive["feature_scale"], dtype=np.float64),
                "metadata": metadata,
            }
    except SpectralResidualBaselineError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SpectralResidualBaselineError(
            f"failed to load checkpoint NPZ {path}: {exc}"
        ) from exc


def _validate_checkpoint_compatible(
    metadata: dict[str, Any],
    context: np.ndarray,
    target: np.ndarray,
) -> None:
    expected = {
        "model_type": MODEL_TYPE,
        "context_steps": int(context.shape[1]),
        "forecast_steps": int(target.shape[1]),
        "space_points": int(context.shape[2]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SpectralResidualBaselineError(
                f"incompatible checkpoint/windows for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    _validate_modes(int(metadata.get("modes", 0)), int(context.shape[2]))
