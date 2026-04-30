"""Unified Hermes sandbox for official checkpoint family candidates."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.fno_competition_prediction import predict_fno_competition
from pde_agent.core.fno_finetune_candidate import execute_fno_finetune_candidate
from pde_agent.core.hermes_local_judge import judge_hermes_submission
from pde_agent.core.unet_competition_prediction import predict_unet_competition


STATE_HINT = "CHECKPOINT_FAMILY_SANDBOX_RAN"
CheckpointRunner = Callable[..., dict[str, Any]]


class CheckpointFamilySandboxError(RuntimeError):
    pass


def run_checkpoint_family_sandbox(
    *,
    checkpoint_paths: list[str | Path],
    task_id: int,
    test_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    submission_id: str = "team_hermes",
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    calibration_weights: list[float] | None = None,
    fno_finetune_steps: int = 0,
    fno_runner: CheckpointRunner | None = None,
    unet_runner: CheckpointRunner | None = None,
    fno_finetune_runner: CheckpointRunner | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not checkpoint_paths:
        raise CheckpointFamilySandboxError("checkpoint_paths must be non-empty")
    if task_id not in {1, 2}:
        raise CheckpointFamilySandboxError("task_id must be 1 or 2")
    root = Path(output_dir)
    report_path = root / "checkpoint_family_sandbox_report.json"
    if report_path.exists() and not force:
        raise CheckpointFamilySandboxError(f"report already exists: {report_path}")
    root.mkdir(parents=True, exist_ok=True)

    weights = _calibration_weights(calibration_weights)
    run_fno = fno_runner or predict_fno_competition
    run_unet = unet_runner or predict_unet_competition
    run_finetune = fno_finetune_runner or _execute_fno_finetune_from_checkpoint

    rows: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoint_paths):
        checkpoint_file = _require_file(Path(checkpoint), "checkpoint")
        family = _checkpoint_family(checkpoint_file)
        raw_report = _run_raw_checkpoint(
            family=family,
            runner=run_fno if family == "FNO" else run_unet,
            checkpoint_path=checkpoint_file,
            task_id=task_id,
            test_path=Path(test_path),
            output_dir=root / f"checkpoint_{index:02d}_{family.lower()}_raw",
            dataset_key=dataset_key,
            force=force,
        )
        rows.extend(
            _materialize_calibrated_rows(
                source="official_checkpoint",
                family=family,
                checkpoint_path=checkpoint_file,
                candidate_report=raw_report,
                task_id=task_id,
                test_path=Path(test_path),
                target_path=Path(target_path),
                output_dir=root / f"checkpoint_{index:02d}_{family.lower()}_calibrated",
                submission_id=submission_id,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                weights=weights,
                force=force,
            )
        )
        if family == "FNO" and fno_finetune_steps > 0:
            finetune_output_dir = root / f"checkpoint_{index:02d}_fno_finetuned"
            windows_path = finetune_output_dir / "finetune_windows.npz"
            _write_supervised_windows(
                test_path=Path(test_path),
                target_path=Path(target_path),
                output_path=windows_path,
                dataset_key=dataset_key,
            )
            finetuned = run_finetune(
                checkpoint_path=checkpoint_file,
                task_id=task_id,
                test_path=Path(test_path),
                target_path=Path(target_path),
                output_dir=finetune_output_dir,
                windows_path=windows_path,
                max_train_steps=fno_finetune_steps,
                dataset_key=dataset_key,
                force=force,
            )
            rows.extend(
                _materialize_calibrated_rows(
                    source="fno_finetuned",
                    family="FNO",
                    checkpoint_path=checkpoint_file,
                    candidate_report=finetuned,
                    task_id=task_id,
                    test_path=Path(test_path),
                    target_path=Path(target_path),
                    output_dir=root
                    / f"checkpoint_{index:02d}_fno_finetuned_calibrated",
                    submission_id=submission_id,
                    dataset_key=dataset_key,
                    prediction_dataset_key=prediction_dataset_key,
                    weights=weights,
                    force=force,
                )
            )

    if not rows:
        raise CheckpointFamilySandboxError("checkpoint sandbox produced no candidates")
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    best = rows[0]
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "checkpoint_family_report_path": str(report_path),
        "task_id": task_id,
        "candidate_count": len(rows),
        "best_candidate": best,
        "top_candidates": rows[:20],
        "artifacts": {
            "checkpoint_family_report_path": str(report_path),
            "best_submission_dir": best["submission_dir"],
            "best_judge_report_path": best["judge_report_path"],
        },
        "next_command": "record-strategy-proposal"
        if best["rules_passed"]
        else "revise-hypothesis",
    }
    _write_json(report_path, report)
    return report


def _run_raw_checkpoint(
    *,
    family: str,
    runner: CheckpointRunner,
    checkpoint_path: Path,
    task_id: int,
    test_path: Path,
    output_dir: Path,
    dataset_key: str,
    force: bool,
) -> dict[str, Any]:
    return runner(
        task_id=task_id,
        test_path=test_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        dataset_key=dataset_key,
        force=force,
    )


def _materialize_calibrated_rows(
    *,
    source: str,
    family: str,
    checkpoint_path: Path,
    candidate_report: dict[str, Any],
    task_id: int,
    test_path: Path,
    target_path: Path,
    output_dir: Path,
    submission_id: str,
    dataset_key: str,
    prediction_dataset_key: str,
    weights: list[float],
    force: bool,
) -> list[dict[str, Any]]:
    candidate_prediction = _load_candidate_prediction(candidate_report)
    rows = []
    for weight in weights:
        checkpoint_id = _safe_id(checkpoint_path.stem)
        candidate_id = f"{checkpoint_id}_{family.lower()}_{source}_calibrated_w{weight:g}"
        start = time.perf_counter()
        prediction_path = output_dir / candidate_id / "calibrated_prediction.npz"
        prediction = _blend_with_persistence(
            test_path=test_path,
            dataset_key=dataset_key,
            candidate_future=candidate_prediction,
            candidate_weight=weight,
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, prediction=prediction)
        inference_seconds = _candidate_inference_seconds(candidate_report) + (
            time.perf_counter() - start
        )
        train_seconds = _candidate_train_seconds(candidate_report)
        submission = build_competition_submission(
            task_id=task_id,
            test_path=test_path,
            output_dir=output_dir / candidate_id / "submission_package",
            submission_id=submission_id,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            method="external_prediction",
            source_prediction_path=prediction_path,
            source_prediction_key="prediction",
            train_time=train_seconds,
            inference_time=inference_seconds,
            force=force,
        )
        judge_path = output_dir / candidate_id / "judge_report.json"
        judge = judge_hermes_submission(
            submission_dir=submission["artifacts"]["submission_dir"],
            output_path=judge_path,
            task1_test_path=test_path if task_id == 1 else None,
            task1_target_path=target_path if task_id == 1 else None,
            task2_test_path=test_path if task_id == 2 else None,
            task2_target_path=target_path if task_id == 2 else None,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
        )
        task = judge["tasks"][str(task_id)]
        rows.append(
            {
                "candidate_id": candidate_id,
                "source": source,
                "model_family": family,
                "checkpoint_path": str(checkpoint_path),
                "calibration_weight": weight,
                "score": float(judge["total_score_proxy"]),
                "segment_score": task["score"]["segment_score"],
                "rules_passed": bool(judge["rules_passed"]),
                "submission_dir": submission["artifacts"]["submission_dir"],
                "prediction_path": submission["artifacts"]["prediction_path"],
                "judge_report_path": str(judge_path),
                "candidate_execution_report_path": _candidate_report_path(
                    candidate_report
                ),
                "train_time": train_seconds,
                "inference_time": inference_seconds,
            }
        )
    return rows


def _execute_fno_finetune_from_checkpoint(
    *,
    checkpoint_path: Path,
    task_id: int,
    test_path: Path,
    target_path: Path,
    output_dir: Path,
    windows_path: Path,
    max_train_steps: int,
    dataset_key: str,
    force: bool,
) -> dict[str, Any]:
    _ = test_path
    _ = target_path
    _ = dataset_key
    plan_path = output_dir / "candidate_execution_plan.json"
    artifacts_dir = output_dir / "artifacts"
    logs_dir = output_dir / "logs"
    plan = {
        "schema_version": 1,
        "state_hint": "CANDIDATE_EXECUTION_PREPARED",
        "candidate": {
            "experiment_id": f"fno_checkpoint_finetune_task{task_id}",
            "intervention": "use_scheduled_sampling_rollout_finetune",
            "parameters": {
                "rollout_steps": 20,
                "teacher_forcing_decay": 0.5,
                "training_strategy": "late_weighted_scheduled_sampling",
            },
        },
        "workspace": {
            "workspace_dir": str(output_dir),
            "artifacts_dir": str(artifacts_dir),
            "logs_dir": str(logs_dir),
            "candidate_execution_plan_path": str(plan_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(plan_path, plan)
    start = time.perf_counter()
    report = execute_fno_finetune_candidate(
        candidate_plan_path=plan_path,
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        max_train_steps=max_train_steps,
        force=force,
    )
    train = report.setdefault("train", {})
    if isinstance(train, dict):
        train["elapsed_seconds"] = time.perf_counter() - start
    report["inference"] = {"elapsed_seconds": 0.0}
    return report


def _write_supervised_windows(
    *,
    test_path: Path,
    target_path: Path,
    output_path: Path,
    dataset_key: str,
) -> None:
    context = _load_context(test_path, dataset_key=dataset_key)
    target = _load_target_future(target_path, dataset_key=dataset_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez(handle, context=context, target=target)


def _blend_with_persistence(
    *,
    test_path: Path,
    dataset_key: str,
    candidate_future: np.ndarray,
    candidate_weight: float,
) -> np.ndarray:
    context = _load_context(test_path, dataset_key=dataset_key)
    future = _future_prediction(candidate_future, context_steps=context.shape[1])
    persistence = np.repeat(context[:, -1:, :], future.shape[1], axis=1)
    blended_future = (1.0 - candidate_weight) * persistence + candidate_weight * future
    prediction = np.empty(
        (context.shape[0], context.shape[1] + future.shape[1], context.shape[2]),
        dtype=np.float32,
    )
    prediction[:, : context.shape[1], :] = context
    prediction[:, context.shape[1] :, :] = blended_future.astype(np.float32)
    return prediction


def _future_prediction(candidate: np.ndarray, *, context_steps: int) -> np.ndarray:
    if candidate.ndim != 3:
        raise CheckpointFamilySandboxError("candidate prediction must be rank 3")
    if candidate.shape[1] == 190:
        return np.asarray(candidate, dtype=np.float32)
    if candidate.shape[1] == 200:
        return np.asarray(candidate[:, context_steps:, :], dtype=np.float32)
    raise CheckpointFamilySandboxError(
        f"candidate prediction must have 190 or 200 time steps, got {candidate.shape[1]}"
    )


def _load_candidate_prediction(report: dict[str, Any]) -> np.ndarray:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CheckpointFamilySandboxError("candidate report missing artifacts")
    path_value = artifacts.get("candidate_prediction_path")
    if not isinstance(path_value, str):
        raise CheckpointFamilySandboxError(
            "candidate report missing artifacts.candidate_prediction_path"
        )
    path = Path(path_value)
    if not path.exists():
        raise CheckpointFamilySandboxError(f"candidate prediction not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "prediction" not in archive:
            raise CheckpointFamilySandboxError("candidate prediction missing key prediction")
        return np.asarray(archive["prediction"], dtype=np.float32)


def _load_context(path: Path, *, dataset_key: str) -> np.ndarray:
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle[dataset_key][:])
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    return np.asarray(data[:, :10, :], dtype=np.float32)


def _load_target_future(path: Path, *, dataset_key: str) -> np.ndarray:
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle[dataset_key][:])
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    return np.asarray(data[:, 10:, :], dtype=np.float32)


def _candidate_inference_seconds(report: dict[str, Any]) -> float:
    inference = report.get("inference")
    if isinstance(inference, dict) and isinstance(
        inference.get("elapsed_seconds"), int | float
    ):
        return max(0.0, float(inference["elapsed_seconds"]))
    return 0.0


def _candidate_train_seconds(report: dict[str, Any]) -> float:
    train = report.get("train")
    if isinstance(train, dict):
        value = train.get("elapsed_seconds")
        if isinstance(value, int | float):
            return max(0.0, float(value))
        steps = train.get("train_steps")
        if isinstance(steps, int | float):
            return max(0.0, float(steps))
    return 0.0


def _candidate_report_path(report: dict[str, Any]) -> str | None:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get("candidate_execution_report_path")
    return str(value) if isinstance(value, str) else None


def _calibration_weights(values: list[float] | None) -> list[float]:
    raw = values if values is not None else [1.0]
    if not raw:
        raise CheckpointFamilySandboxError("calibration_weights must be non-empty")
    weights = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CheckpointFamilySandboxError("calibration weights must be numeric")
        weights.append(float(min(1.0, max(0.0, value))))
    return sorted(set(weights))


def _checkpoint_family(path: Path) -> str:
    name = path.name.lower()
    if "unet" in name:
        return "UNet"
    if "fno" in name:
        return "FNO"
    raise CheckpointFamilySandboxError(
        f"cannot infer checkpoint family from filename: {path.name}"
    )


def _safe_id(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    return "_".join("".join(chars).split("_")).strip("_") or "checkpoint"


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise CheckpointFamilySandboxError(f"missing {label} path: {path}")
    if not path.is_file():
        raise CheckpointFamilySandboxError(f"{label} path is not a file: {path}")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
