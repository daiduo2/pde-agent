"""Hermes loop that converts validation quality into model-iteration actions."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_model_quality_diagnosis import (
    diagnose_hermes_model_quality,
)
from pde_agent.core.hermes_validation_scoring import run_hermes_validation_scoring


STATE_HINT = "HERMES_MODEL_ITERATION_LOOP_RAN"
ValidationScorer = Callable[..., dict[str, Any]]
QualityDiagnoser = Callable[..., dict[str, Any]]


class HermesModelIterationLoopError(RuntimeError):
    pass


def run_hermes_model_iteration_loop(
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
    min_segment_score: float = 60.0,
    min_task_prediction_score: float = 45.0,
    max_late_rmse: float = 0.25,
    max_inference_seconds: float = 120.0,
    validation_scorer: ValidationScorer | None = None,
    quality_diagnoser: QualityDiagnoser | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_path = root / "hermes_model_iteration_loop.json"
    if report_path.exists() and not force:
        raise HermesModelIterationLoopError(
            f"model iteration loop report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    score_validation = validation_scorer or run_hermes_validation_scoring
    diagnose_quality = quality_diagnoser or diagnose_hermes_model_quality

    validation = score_validation(
        output_dir=root / "validation",
        operator_family=operator_family,
        task_id=task_id,
        validation_path=Path(validation_path),
        checkpoint_path=Path(checkpoint_path),
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        submission_id=submission_id,
        inference_batch_size=inference_batch_size,
        torch_num_threads=torch_num_threads,
        sample_limit=sample_limit,
        force=True,
    )
    validation_report_path = _path_from_report(
        validation,
        direct_key="validation_report_path",
        artifact_key="validation_report_path",
    )
    diagnosis = diagnose_quality(
        validation_report_path=validation_report_path,
        output_path=root / "diagnosis" / "hermes_model_quality_diagnosis.json",
        min_segment_score=min_segment_score,
        min_task_prediction_score=min_task_prediction_score,
        max_late_rmse=max_late_rmse,
        max_inference_seconds=max_inference_seconds,
    )
    decision = _decision(
        operator_family=operator_family,
        diagnosis=diagnosis,
        inference_batch_size=inference_batch_size,
        torch_num_threads=torch_num_threads,
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "iteration_report_path": str(report_path),
        "operator_family": operator_family,
        "task_id": task_id,
        "validation": validation,
        "diagnosis": diagnosis,
        "decision": decision,
        "artifacts": {
            "iteration_report_path": str(report_path),
            "validation_report_path": str(validation_report_path),
            "diagnosis_path": diagnosis.get("diagnosis_path"),
        },
        "next_command": decision["next_command"],
    }
    _write_json(report_path, report)
    return report


def _decision(
    *,
    operator_family: str,
    diagnosis: dict[str, Any],
    inference_batch_size: int | None,
    torch_num_threads: int | None,
) -> dict[str, Any]:
    strategy_family = diagnosis.get("recommended_strategy_family")
    if strategy_family == "runtime_tuning":
        return {
            "action": "retune_execution_path",
            "strategy_family": strategy_family,
            "reason_codes": diagnosis.get("reason_codes", []),
            "next_command": "run-hermes-operator-tuning-loop",
        }
    if strategy_family == "promote_candidate":
        return {
            "action": "promote_validated_candidate",
            "strategy_family": strategy_family,
            "selected_config": {
                "inference_batch_size": inference_batch_size,
                "torch_num_threads": torch_num_threads,
            },
            "reason_codes": diagnosis.get("reason_codes", []),
            "next_command": "run-hermes-submission-candidate",
        }
    if operator_family == "UNet":
        return {
            "action": "schedule_model_strategy_sweep",
            "strategy_family": strategy_family,
            "primary_bottleneck": diagnosis.get("primary_bottleneck"),
            "reason_codes": diagnosis.get("reason_codes", []),
            "next_command": "run-unet-strategy-sweep",
        }
    return {
        "action": "record_model_iteration_proposal",
        "strategy_family": strategy_family,
        "primary_bottleneck": diagnosis.get("primary_bottleneck"),
        "reason_codes": diagnosis.get("reason_codes", []),
        "next_command": "record-strategy-proposal",
    }


def _path_from_report(
    report: dict[str, Any],
    *,
    direct_key: str,
    artifact_key: str,
) -> Path:
    value = report.get(direct_key)
    if not isinstance(value, str):
        artifacts = report.get("artifacts")
        if isinstance(artifacts, dict):
            value = artifacts.get(artifact_key)
    if not isinstance(value, str) or not value:
        raise HermesModelIterationLoopError(
            f"report missing {direct_key} or artifacts.{artifact_key}"
        )
    return Path(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
