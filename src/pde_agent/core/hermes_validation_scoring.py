"""Hermes validation-set scoring loop for competition-shaped Burgers models."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.competition_candidate_materialization import (
    materialize_competition_candidate,
)
from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)
from pde_agent.core.unet_competition_prediction import predict_unet_competition


STATE_HINT = "HERMES_VALIDATION_SCORED"
Predictor = Callable[..., dict[str, Any]]
Materializer = Callable[..., dict[str, Any]]


class HermesValidationScoringError(RuntimeError):
    pass


def run_hermes_validation_scoring(
    *,
    output_dir: str | Path,
    operator_family: str,
    task_id: int,
    validation_path: str | Path,
    checkpoint_path: str | Path,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    submission_id: str = "team_hermes_validation",
    inference_batch_size: int | None = None,
    torch_num_threads: int | None = None,
    sample_limit: int | None = None,
    predictor: Predictor | None = None,
    materializer: Materializer | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if task_id not in {1, 2}:
        raise HermesValidationScoringError("task_id must be 1 or 2")
    if operator_family != "UNet":
        raise HermesValidationScoringError(
            "validation scoring currently supports UNet only"
        )
    root = Path(output_dir)
    report_path = root / "hermes_validation_scoring.json"
    if report_path.exists() and not force:
        raise HermesValidationScoringError(
            f"validation scoring report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)

    validation = _load_validation_tensor(
        path=Path(validation_path),
        dataset_key=dataset_key,
        sample_limit=sample_limit,
    )
    context_path = root / "validation_context.hdf5"
    target_path = root / "validation_target.hdf5"
    _write_hdf5(context_path, validation[:, :CONTEXT_STEPS, :], dataset_key)
    _write_hdf5(target_path, validation, dataset_key)

    run_predictor = predictor or predict_unet_competition
    run_materializer = materializer or materialize_competition_candidate
    prediction = run_predictor(
        task_id=task_id,
        test_path=context_path,
        checkpoint_path=Path(checkpoint_path),
        output_dir=root / "prediction",
        dataset_key=dataset_key,
        inference_batch_size=inference_batch_size,
        torch_num_threads=torch_num_threads,
        force=True,
    )
    candidate_report_path = _candidate_report_path(prediction)
    materialized = run_materializer(
        task_id=task_id,
        candidate_report_path=candidate_report_path,
        test_path=context_path,
        output_dir=root / "materialized",
        submission_id=submission_id,
        target_path=target_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        source_prediction_key="prediction",
        train_time=None,
        force=True,
    )
    score = materialized.get("score")
    if not isinstance(score, dict):
        raise HermesValidationScoringError("materializer did not return score")

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "validation_report_path": str(report_path),
        "operator_family": operator_family,
        "task_id": task_id,
        "validation_path": str(validation_path),
        "checkpoint_path": str(checkpoint_path),
        "selected_config": {
            "inference_batch_size": inference_batch_size,
            "torch_num_threads": torch_num_threads,
        },
        "prediction": prediction,
        "materialized": materialized,
        "score": score,
        "artifacts": {
            "validation_report_path": str(report_path),
            "validation_context_path": str(context_path),
            "validation_target_path": str(target_path),
            "candidate_execution_report_path": str(candidate_report_path),
            "score_json_path": materialized["artifacts"]["score_json_path"],
            "submission_dir": materialized["artifacts"]["submission_dir"],
        },
        "next_command": "diagnose-hermes-model-quality",
    }
    _write_json(report_path, report)
    return report


def _load_validation_tensor(
    *,
    path: Path,
    dataset_key: str,
    sample_limit: int | None,
) -> np.ndarray:
    if not path.exists():
        raise HermesValidationScoringError(f"validation HDF5 not found: {path}")
    if sample_limit is not None and sample_limit <= 0:
        raise HermesValidationScoringError("sample_limit must be positive")
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise HermesValidationScoringError(
                f"dataset {dataset_key!r} not found in {path}"
            )
        data = np.asarray(handle[dataset_key][:], dtype=np.float32)
    if data.ndim != 3:
        raise HermesValidationScoringError(
            f"expected rank-3 validation tensor, got shape {data.shape}"
        )
    if sample_limit is not None:
        data = data[:sample_limit]
    if data.shape[1] >= 1000:
        data = data[:, ::5, :]
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    if data.shape[1] < PREDICTION_STEPS:
        raise HermesValidationScoringError(
            f"expected at least {PREDICTION_STEPS} validation time steps, got {data.shape[1]}"
        )
    if data.shape[2] != SPACE_POINTS:
        raise HermesValidationScoringError(
            f"expected validation spatial width 256 or 1024, got {data.shape[2]}"
        )
    data = np.asarray(data[:, :PREDICTION_STEPS, :], dtype=np.float32)
    if data.shape[0] <= 0:
        raise HermesValidationScoringError("validation tensor has no samples")
    if not np.isfinite(data).all():
        raise HermesValidationScoringError("validation tensor contains NaN or Inf")
    return data


def _candidate_report_path(prediction: dict[str, Any]) -> Path:
    artifacts = prediction.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HermesValidationScoringError("prediction report missing artifacts")
    value = artifacts.get("candidate_execution_report_path")
    if not isinstance(value, str) or not value:
        raise HermesValidationScoringError(
            "prediction report missing artifacts.candidate_execution_report_path"
        )
    return Path(value)


def _write_hdf5(path: Path, data: np.ndarray, dataset_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset_key, data=data)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
