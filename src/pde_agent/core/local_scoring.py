"""Short-horizon local scoring utilities for PDE rollout predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class LocalScoreError(RuntimeError):
    pass


def score_prediction_npz(
    *,
    prediction_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    """Score prediction and target arrays loaded from NPZ files."""

    prediction_file = Path(prediction_path)
    target_file = Path(target_path)
    prediction = _load_npz_array(prediction_file, "prediction")
    target = _load_npz_array(target_file, "target")

    result = score_prediction_arrays(prediction, target)
    result["prediction_path"] = str(prediction_file)
    result["target_path"] = str(target_file)

    if output_path is not None:
        score_file = Path(output_path)
        result["score_path"] = str(score_file)
        score_file.parent.mkdir(parents=True, exist_ok=True)
        score_file.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return result


def score_prediction_arrays(prediction: Any, target: Any) -> dict[str, Any]:
    """Compute a competition-like short-horizon score for arrays shaped [N, T, X]."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _validate_arrays(pred, truth)

    error = pred - truth
    _raise_if_non_finite(error, "derived error")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        squared_error = error**2
        _raise_if_non_finite(squared_error, "derived squared_error")
        truth_squared = truth**2
        _raise_if_non_finite(truth_squared, "derived target squared")
        rmse = float(np.sqrt(np.mean(squared_error)))
        target_rms = float(np.sqrt(np.mean(truth_squared)))
        nrmse = float(rmse / (target_rms + 1e-12))
        _raise_if_non_finite([rmse, target_rms, nrmse], "derived global metrics")

        rel_by_time = np.sum(squared_error, axis=2) / (
            np.sum(truth_squared, axis=2) + 1e-12
        )
        _raise_if_non_finite(rel_by_time, "derived relative MSE by time")
        rel_capped = np.minimum(rel_by_time, 5.0)
        rel_mse = float(np.mean(rel_by_time))
        rel_mse_capped = float(np.mean(rel_capped))
        _raise_if_non_finite([rel_mse, rel_mse_capped], "derived relative metrics")

        early_slice, middle_slice, late_slice = _segment_slices(pred.shape[1])
        early_rel = float(np.mean(rel_capped[:, early_slice]))
        middle_rel = float(np.mean(rel_capped[:, middle_slice]))
        late_rmse = float(np.sqrt(np.mean(squared_error[:, late_slice, :])))
        _raise_if_non_finite(
            [early_rel, middle_rel, late_rmse],
            "derived segment metrics",
        )

        early_score = float(100.0 * np.exp(-20.0 * early_rel))
        middle_score = float(100.0 * np.exp(-10.0 * middle_rel))
        late_score = float(100.0 / (1.0 + 10.0 * late_rmse))
        total_score = float(
            0.25 * early_score + 0.25 * middle_score + 0.5 * late_score
        )
        _raise_if_non_finite(
            [early_score, middle_score, late_score, total_score],
            "derived scores",
        )

    return {
        "sample_count": int(pred.shape[0]),
        "time_steps": int(pred.shape[1]),
        "space_points": int(pred.shape[2]),
        "rmse": rmse,
        "nrmse": nrmse,
        "rel_mse": rel_mse,
        "rel_mse_capped": rel_mse_capped,
        "competition_like_score_short": total_score,
        "segments": {
            "early": {
                "time_start": int(early_slice.start),
                "time_stop": int(early_slice.stop),
                "rel_mse_capped": early_rel,
                "score": early_score,
            },
            "middle": {
                "time_start": int(middle_slice.start),
                "time_stop": int(middle_slice.stop),
                "rel_mse_capped": middle_rel,
                "score": middle_score,
            },
            "late": {
                "time_start": int(late_slice.start),
                "time_stop": int(late_slice.stop),
                "rmse": late_rmse,
                "score": late_score,
            },
        },
        "notes": (
            "Segments are proportional to the available local horizon: early first "
            "25%, middle next 25%, and late remaining 50%. The late segment uses a "
            "Lorentzian RMSE proxy; Frechet scoring is reserved for the hidden "
            "long-horizon evaluator."
        ),
    }


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise LocalScoreError(f"missing path: {path}")
    if not path.is_file():
        raise LocalScoreError(f"path is not a file: {path}")

    try:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise LocalScoreError(f"{path} missing key '{key}'")
            return np.asarray(data[key])
    except LocalScoreError:
        raise
    except OSError as exc:
        raise LocalScoreError(f"failed to load NPZ file {path}: {exc}") from exc


def _validate_arrays(prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise LocalScoreError(
            f"shape mismatch: prediction shape {prediction.shape} != target shape {target.shape}"
        )
    if prediction.ndim != 3:
        raise LocalScoreError(
            f"expected rank 3 arrays shaped [N, T, X], got rank {prediction.ndim}"
        )
    if any(dimension <= 0 for dimension in prediction.shape):
        raise LocalScoreError(
            f"expected non-empty [N, T, X] dimensions, got shape {prediction.shape}"
        )
    if not np.all(np.isfinite(prediction)):
        raise LocalScoreError("prediction contains non-finite values")
    if not np.all(np.isfinite(target)):
        raise LocalScoreError("target contains non-finite values")


def _raise_if_non_finite(value: Any, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise LocalScoreError(f"{label} contains non-finite values")


def _segment_slices(time_steps: int) -> tuple[slice, slice, slice]:
    if time_steps == 1:
        return slice(0, 1), slice(0, 1), slice(0, 1)
    if time_steps == 2:
        return slice(0, 1), slice(0, 1), slice(1, 2)

    early_stop = max(1, time_steps // 4)
    middle_stop = max(early_stop + 1, time_steps // 2)
    return (
        slice(0, early_stop),
        slice(early_stop, middle_stop),
        slice(middle_stop, time_steps),
    )
