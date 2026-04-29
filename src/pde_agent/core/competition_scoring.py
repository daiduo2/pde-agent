"""Competition-shaped scoring utilities for Burgers submission predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)


STATE_HINT = "COMPETITION_PREDICTION_SCORED"
FUTURE_STEPS = PREDICTION_STEPS - CONTEXT_STEPS
EARLY_FUTURE_STOP = 47
MIDDLE_FUTURE_STOP = 95


class CompetitionScoringError(RuntimeError):
    pass


def score_competition_prediction(
    *,
    task_id: int,
    prediction_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None,
    prediction_dataset_key: str = "tensor",
    target_dataset_key: str = "tensor",
) -> dict[str, Any]:
    if task_id not in {1, 2}:
        raise CompetitionScoringError("task_id must be 1 or 2")
    prediction_file = Path(prediction_path)
    target_file = Path(target_path)
    prediction = _load_competition_hdf5(
        path=prediction_file,
        dataset_key=prediction_dataset_key,
    )
    target = _load_competition_hdf5(
        path=target_file,
        dataset_key=target_dataset_key,
    )
    _validate_pair(prediction=prediction, target=target)

    squared_error = (prediction - target) ** 2
    target_squared = target**2
    early = _rel_mse_segment(
        squared_error=squared_error,
        target_squared=target_squared,
        future_start=0,
        future_stop=EARLY_FUTURE_STOP,
    )
    middle = _rel_mse_segment(
        squared_error=squared_error,
        target_squared=target_squared,
        future_start=EARLY_FUTURE_STOP,
        future_stop=MIDDLE_FUTURE_STOP,
    )
    late = _late_segment(
        squared_error=squared_error,
        future_start=MIDDLE_FUTURE_STOP,
        future_stop=FUTURE_STEPS,
    )
    segment_score = float(
        0.25 * early["score"] + 0.25 * middle["score"] + 0.5 * late["score"]
    )
    task_prediction_score = (
        float(segment_score * 0.75) if task_id == 1 else float(segment_score * 1.5)
    )

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "task_id": task_id,
        "prediction_path": str(prediction_file),
        "target_path": str(target_file),
        "sample_count": int(prediction.shape[0]),
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "segment_score": segment_score,
        "task_prediction_score": task_prediction_score,
        "segments": {
            "early": early,
            "middle": middle,
            "late": late,
        },
        "notes": (
            "This local scorer implements the stated 190-step segmented Rel-MSE "
            "and Lorentzian long-window formula. The public rules mention a "
            "Frechet alternative for the final segment but do not define FD^2; "
            "therefore the long-window score is a Lorentzian-only lower-bound "
            "proxy for strategy selection."
        ),
    }
    if output_path is not None:
        score_file = Path(output_path)
        score_file.parent.mkdir(parents=True, exist_ok=True)
        report["score_path"] = str(score_file)
        score_file.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _load_competition_hdf5(*, path: Path, dataset_key: str) -> np.ndarray:
    if not path.exists():
        raise CompetitionScoringError(f"HDF5 file not found: {path}")
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise CompetitionScoringError(
                f"dataset {dataset_key!r} not found in {path}"
            )
        data = np.asarray(handle[dataset_key][:], dtype=np.float64)
    if data.ndim != 3:
        raise CompetitionScoringError(
            f"expected rank-3 tensor in {path}, got shape {data.shape}"
        )
    if data.shape[1] == 1000:
        data = data[:, ::5, :]
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    if data.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise CompetitionScoringError(
            f"expected shape [N, 200, 256] after reduction, got {data.shape}"
        )
    if not np.isfinite(data).all():
        raise CompetitionScoringError(f"{path} contains NaN or Inf values")
    return data


def _validate_pair(*, prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise CompetitionScoringError(
            f"shape mismatch: prediction {prediction.shape} != target {target.shape}"
        )
    if prediction.shape[0] <= 0:
        raise CompetitionScoringError("prediction batch is empty")


def _rel_mse_segment(
    *,
    squared_error: np.ndarray,
    target_squared: np.ndarray,
    future_start: int,
    future_stop: int,
) -> dict[str, Any]:
    absolute_start = CONTEXT_STEPS + future_start
    absolute_stop = CONTEXT_STEPS + future_stop
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_by_sample_time = (
            np.sum(squared_error[:, absolute_start:absolute_stop, :], axis=2)
            / (
                np.sum(target_squared[:, absolute_start:absolute_stop, :], axis=2)
                + 1e-12
            )
        )
    _raise_if_non_finite(rel_by_sample_time, "segment relative MSE")
    rel_by_sample = np.mean(rel_by_sample_time, axis=1)
    rel_mse = float(np.mean(np.minimum(rel_by_sample, 5.0)))
    score = float(100.0 * np.exp(-_score_decay(future_start) * rel_mse))
    _raise_if_non_finite([rel_mse, score], "segment score")
    return {
        "time_start": absolute_start,
        "time_stop": absolute_stop,
        "future_start": future_start,
        "future_stop": future_stop,
        "rel_mse": rel_mse,
        "score": score,
    }


def _late_segment(
    *,
    squared_error: np.ndarray,
    future_start: int,
    future_stop: int,
) -> dict[str, Any]:
    absolute_start = CONTEXT_STEPS + future_start
    absolute_stop = CONTEXT_STEPS + future_stop
    rmse = float(np.sqrt(np.mean(squared_error[:, absolute_start:absolute_stop, :])))
    lorentzian_score = float(100.0 / (1.0 + 10.0 * rmse))
    _raise_if_non_finite([rmse, lorentzian_score], "late segment score")
    return {
        "time_start": absolute_start,
        "time_stop": absolute_stop,
        "future_start": future_start,
        "future_stop": future_stop,
        "rmse": rmse,
        "lorentzian_score": lorentzian_score,
        "frechet_score": None,
        "score": lorentzian_score,
    }


def _score_decay(future_start: int) -> float:
    if future_start == 0:
        return 20.0
    return 10.0


def _raise_if_non_finite(value: Any, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise CompetitionScoringError(f"{label} contains non-finite values")
