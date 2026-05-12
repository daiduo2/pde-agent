
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

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
    metadata_values = _validate_checkpoint_compatible(metadata, context, target)

    mode_count = metadata_values["modes"]
    forecast_steps = metadata_values["forecast_steps"]
    space_points = metadata_values["space_points"]
    features = _build_features(context, mode_count)
    feature_mean = checkpoint_data["feature_mean"]
    feature_scale = checkpoint_data["feature_scale"]
    expected_width = forecast_steps * mode_count
    _validate_checkpoint_arrays(
        weights_real=checkpoint_data["weights_real"],
        weights_imag=checkpoint_data["weights_imag"],
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        feature_dim=int(features.shape[1]),
        output_width=expected_width,
    )

    normalized = (features - feature_mean) / feature_scale
    design = np.concatenate(
        [np.ones((features.shape[0], 1), dtype=np.float64), normalized],
        axis=1,
    )
    weights_real = checkpoint_data["weights_real"]
    weights_imag = checkpoint_data["weights_imag"]

    flat_real = design @ weights_real
    flat_imag = design @ weights_imag

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
    except (OSError, ValueError, EOFError, BadZipFile) as exc:
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
    try:
        value = float(ridge)
    except (TypeError, ValueError) as exc:
        raise SpectralResidualBaselineError(
            f"ridge must be finite and non-negative, got {ridge!r}"
        ) from exc
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
            if not isinstance(metadata, dict):
                raise SpectralResidualBaselineError(
                    "checkpoint metadata_json must decode to an object"
                )
            return {
                "weights_real": np.asarray(archive["weights_real"], dtype=np.float64),
                "weights_imag": np.asarray(archive["weights_imag"], dtype=np.float64),
                "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
                "feature_scale": np.asarray(archive["feature_scale"], dtype=np.float64),
                "metadata": metadata,
            }
    except SpectralResidualBaselineError:
        raise
    except (OSError, ValueError, EOFError, BadZipFile, json.JSONDecodeError) as exc:
        raise SpectralResidualBaselineError(
            f"failed to load checkpoint NPZ {path}: {exc}"
        ) from exc


def _validate_checkpoint_arrays(
    *,
    weights_real: np.ndarray,
    weights_imag: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    feature_dim: int,
    output_width: int,
) -> None:
    _validate_checkpoint_vector(
        name="feature_mean",
        value=feature_mean,
        length=feature_dim,
    )
    _validate_checkpoint_vector(
        name="feature_scale",
        value=feature_scale,
        length=feature_dim,
    )
    if np.any(feature_scale <= 1e-12):
        raise SpectralResidualBaselineError(
            "checkpoint feature_scale must be greater than 1e-12"
        )
    expected_weight_shape = (feature_dim + 1, output_width)
    _validate_checkpoint_matrix(
        name="weights_real",
        value=weights_real,
        shape=expected_weight_shape,
    )
    _validate_checkpoint_matrix(
        name="weights_imag",
        value=weights_imag,
        shape=expected_weight_shape,
    )


def _validate_checkpoint_vector(
    *,
    name: str,
    value: np.ndarray,
    length: int,
) -> None:
    if value.ndim != 1:
        raise SpectralResidualBaselineError(
            f"checkpoint {name} must be rank 1, got rank {value.ndim}"
        )
    if value.shape[0] != length:
        raise SpectralResidualBaselineError(
            f"checkpoint {name} length {value.shape[0]} != expected {length}"
        )
    if not np.all(np.isfinite(value)):
        raise SpectralResidualBaselineError(
            f"checkpoint {name} contains non-finite values"
        )


def _validate_checkpoint_matrix(
    *,
    name: str,
    value: np.ndarray,
    shape: tuple[int, int],
) -> None:
    if value.ndim != 2:
        raise SpectralResidualBaselineError(
            f"checkpoint {name} must be rank 2, got rank {value.ndim}"
        )
    if value.shape != shape:
        raise SpectralResidualBaselineError(
            f"checkpoint {name} shape {value.shape} != expected {shape}"
        )
    if not np.all(np.isfinite(value)):
        raise SpectralResidualBaselineError(
            f"checkpoint {name} contains non-finite values"
        )


def _validate_checkpoint_compatible(
    metadata: dict[str, Any],
    context: np.ndarray,
    target: np.ndarray,
) -> dict[str, int]:
    if metadata.get("model_type") != MODEL_TYPE:
        raise SpectralResidualBaselineError(
            "incompatible checkpoint/windows for model_type: "
            f"{metadata.get('model_type')!r} != {MODEL_TYPE!r}"
        )

    modes = _metadata_int(metadata, "modes")
    context_steps = _metadata_int(metadata, "context_steps")
    forecast_steps = _metadata_int(metadata, "forecast_steps")
    space_points = _metadata_int(metadata, "space_points")
    expected = {
        "context_steps": int(context.shape[1]),
        "forecast_steps": int(target.shape[1]),
        "space_points": int(context.shape[2]),
    }
    actual = {
        "context_steps": context_steps,
        "forecast_steps": forecast_steps,
        "space_points": space_points,
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise SpectralResidualBaselineError(
                f"incompatible checkpoint/windows for {key}: "
                f"{value!r} != {expected[key]!r}"
            )
    try:
        _validate_modes(modes, int(context.shape[2]))
    except SpectralResidualBaselineError as exc:
        raise SpectralResidualBaselineError(
            f"checkpoint metadata incompatible for modes: {exc}"
        ) from exc
    return {
        "modes": modes,
        "forecast_steps": forecast_steps,
        "space_points": space_points,
    }


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    if key not in metadata:
        raise SpectralResidualBaselineError(
            f"checkpoint metadata missing required key: {key}"
        )
    value = metadata[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpectralResidualBaselineError(
            f"checkpoint metadata invalid for {key}: expected integer, got {value!r}"
        )
    return value
