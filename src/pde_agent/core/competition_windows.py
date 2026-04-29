"""Prepare competition test HDF5 files for existing local rollout backends."""

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


STATE_HINT = "COMPETITION_WINDOWS_READY"
FORECAST_STEPS = PREDICTION_STEPS - CONTEXT_STEPS


class CompetitionWindowsError(RuntimeError):
    pass


def prepare_competition_windows(
    *,
    task_id: int,
    test_path: str | Path,
    output_path: str | Path,
    dataset_key: str = "tensor",
    force: bool = False,
) -> dict[str, Any]:
    if task_id not in {1, 2}:
        raise CompetitionWindowsError("task_id must be 1 or 2")
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise CompetitionWindowsError(f"competition windows already exist: {output_file}")
    context = _load_context(path=Path(test_path), dataset_key=dataset_key)
    target = np.zeros(
        (context.shape[0], FORECAST_STEPS, SPACE_POINTS),
        dtype=np.float32,
    )
    metadata = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "task_id": task_id,
        "source_test_path": str(test_path),
        "dataset_key": dataset_key,
        "context_steps": CONTEXT_STEPS,
        "forecast_steps": FORECAST_STEPS,
        "space_points": SPACE_POINTS,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as handle:
        np.savez(
            handle,
            context=context,
            target=target,
            metadata_json=json.dumps(metadata, allow_nan=False, sort_keys=True),
        )
    return {
        **metadata,
        "windows_path": str(output_file),
        "sample_count": int(context.shape[0]),
        "context_shape": list(context.shape),
        "target_shape": list(target.shape),
    }


def _load_context(*, path: Path, dataset_key: str) -> np.ndarray:
    if not path.exists():
        raise CompetitionWindowsError(f"test HDF5 not found: {path}")
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise CompetitionWindowsError(f"dataset {dataset_key!r} not found")
        data = np.asarray(handle[dataset_key][:])
    if data.ndim != 3:
        raise CompetitionWindowsError(f"expected rank-3 tensor, got shape {data.shape}")
    if data.shape[1] < CONTEXT_STEPS:
        raise CompetitionWindowsError(
            f"expected at least {CONTEXT_STEPS} context steps, got {data.shape[1]}"
        )
    if data.shape[2] == SPACE_POINTS:
        context = data[:, :CONTEXT_STEPS, :]
    elif data.shape[2] == 1024:
        context = data[:, :CONTEXT_STEPS, ::4]
    else:
        raise CompetitionWindowsError(
            f"expected spatial width 256 or 1024, got {data.shape[2]}"
        )
    context = np.asarray(context, dtype=np.float32)
    if not np.isfinite(context).all():
        raise CompetitionWindowsError("context contains NaN or Inf values")
    return context
