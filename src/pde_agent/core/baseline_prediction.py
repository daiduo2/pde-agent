from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class BaselinePredictionError(RuntimeError):
    pass


def generate_baseline_prediction(
    *,
    windows_path: str | Path,
    output_path: str | Path,
    method: str = "persistence",
    force: bool = False,
) -> dict[str, Any]:
    windows = Path(windows_path)
    output = Path(output_path)
    if method != "persistence":
        raise BaselinePredictionError(f"unknown baseline method: {method}")
    if not windows.exists():
        raise BaselinePredictionError(f"missing windows path: {windows}")
    if output.exists() and not force:
        raise BaselinePredictionError(f"output path already exists: {output}")

    context, target = _load_windows(windows)
    prediction = np.repeat(context[:, -1:, :], repeats=target.shape[1], axis=1)
    prediction = prediction.astype(np.float32, copy=False)
    metadata = {
        "method": method,
        "windows_path": str(windows),
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
    return {
        "prediction_path": str(output),
        "prediction_shape": [int(axis) for axis in prediction.shape],
        **metadata,
    }


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in ("context", "target"):
                if key not in archive.files:
                    raise BaselinePredictionError(f"{path} missing key '{key}'")
            context = np.asarray(archive["context"], dtype=np.float64)
            target = np.asarray(archive["target"], dtype=np.float64)
    except BaselinePredictionError:
        raise
    except OSError as exc:
        raise BaselinePredictionError(
            f"failed to load windows NPZ {path}: {exc}"
        ) from exc
    _validate_rank3("context", context)
    _validate_rank3("target", target)
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise BaselinePredictionError(
            f"context/target shape mismatch: {context.shape} vs {target.shape}"
        )
    if not np.all(np.isfinite(context)) or not np.all(np.isfinite(target)):
        raise BaselinePredictionError("windows arrays contain non-finite values")
    return context, target


def _validate_rank3(name: str, array: np.ndarray) -> None:
    if array.ndim != 3:
        raise BaselinePredictionError(f"{name} must be rank 3 [N, T, X]")
    if any(axis <= 0 for axis in array.shape):
        raise BaselinePredictionError(f"{name} dimensions must be non-empty")
