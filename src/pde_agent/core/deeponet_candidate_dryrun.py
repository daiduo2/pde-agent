"""Dry-run DeepONet candidate executor for closing the local evidence loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DeepONetCandidateDryrunError(RuntimeError):
    pass


def execute_deeponet_candidate_dryrun(
    *,
    windows_path: str | Path,
    output_dir: str | Path,
    start_index: int,
    train_steps: int,
    training_strategy: str,
    force: bool = False,
) -> dict[str, Any]:
    windows_file = _require_file(Path(windows_path), "windows")
    context, target = _load_windows(windows_file)
    start = _non_negative_int(start_index, "start_index")
    steps = _positive_int(train_steps, "train_steps")
    strategy = _training_strategy(training_strategy)

    output_root = Path(output_dir)
    paths = {
        "candidate_prediction_path": output_root / "candidate_prediction.npz",
        "candidate_execution_report_path": output_root
        / "candidate_execution_report.json",
    }
    _prepare_output_paths(paths=paths, force=force)

    prediction = _predict(context=context, forecast_steps=target.shape[1], strategy=strategy)
    experiment_id = f"deeponet_{strategy}_s{start:04d}_steps{steps:04d}"
    with paths["candidate_prediction_path"].open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction.astype(np.float32),
            metadata_json=json.dumps(
                {
                    "model_type": "deeponet",
                    "adapter_type": "dryrun_branch_trunk_proxy",
                    "experiment_id": experiment_id,
                    "training_strategy": strategy,
                    "source_windows_path": str(windows_file),
                },
                allow_nan=False,
                sort_keys=True,
            ),
        )

    report = {
        "schema_version": 1,
        "state_hint": "DEEPONET_CANDIDATE_EXECUTED",
        "windows_path": str(windows_file),
        "inputs": {
            "context_steps": int(context.shape[1]),
            "forecast_steps": int(target.shape[1]),
            "space_points": int(target.shape[2]),
        },
        "candidate": {
            "experiment_id": experiment_id,
            "start_index": start,
            "train_steps": steps,
            "training_strategy": strategy,
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
        "train": {
            "mode": "dryrun",
            "loss_trace": [],
            "note": "No neural training is performed by this dry-run executor.",
        },
        "next_command": "evaluate-deeponet-candidate",
    }
    paths["candidate_execution_report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.load(path)
        context = data["context"].astype(np.float32)
        target = data["target"].astype(np.float32)
    except KeyError as exc:
        raise DeepONetCandidateDryrunError(
            "windows npz must include context and target arrays"
        ) from exc
    if context.ndim != 3 or target.ndim != 3:
        raise DeepONetCandidateDryrunError(
            "context and target must have shape (sample, time, space)"
        )
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise DeepONetCandidateDryrunError(
            "context and target must share sample and space dimensions"
        )
    if context.shape[1] < 1 or target.shape[1] < 1:
        raise DeepONetCandidateDryrunError("context and target must have time steps")
    return context, target


def _predict(*, context: np.ndarray, forecast_steps: int, strategy: str) -> np.ndarray:
    last = context[:, -1, :]
    if strategy == "persistence":
        return np.repeat(last[:, None, :], forecast_steps, axis=1)
    if strategy == "branch_trunk_rollout":
        if context.shape[1] >= 2:
            trend = context[:, -1, :] - context[:, -2, :]
        else:
            trend = np.zeros_like(last)
        steps = np.arange(1, forecast_steps + 1, dtype=np.float32)[None, :, None]
        return last[:, None, :] + steps * trend[:, None, :]
    raise DeepONetCandidateDryrunError(f"unsupported training_strategy: {strategy}")


def _training_strategy(value: str) -> str:
    if value not in {"persistence", "branch_trunk_rollout"}:
        raise DeepONetCandidateDryrunError(
            "training_strategy must be one of: persistence, branch_trunk_rollout"
        )
    return value


def _prepare_output_paths(*, paths: dict[str, Path], force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise DeepONetCandidateDryrunError(
            f"DeepONet candidate dryrun artifact already exists: {names}"
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _non_negative_int(value: int, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise DeepONetCandidateDryrunError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: int, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise DeepONetCandidateDryrunError(f"{label} must be a positive integer")
    return value


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise DeepONetCandidateDryrunError(f"missing {label} path: {path}")
    if not path.is_file():
        raise DeepONetCandidateDryrunError(f"{label} path is not a file: {path}")
    return path
