from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_model_iteration_loop import (
    STATE_HINT,
    run_hermes_model_iteration_loop,
)
from pde_agent.core.json_output import success_payload


def run_hermes_model_iteration_loop_command(
    *,
    output_dir: str,
    operator_family: str,
    task_id: int,
    validation_path: str,
    checkpoint_path: str,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    inference_batch_size: int | None,
    torch_num_threads: int | None,
    sample_limit: int | None,
    min_segment_score: float,
    min_task_prediction_score: float,
    max_late_rmse: float,
    max_inference_seconds: float,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_model_iteration_loop(
        output_dir=Path(output_dir),
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
        min_segment_score=min_segment_score,
        min_task_prediction_score=min_task_prediction_score,
        max_late_rmse=max_late_rmse,
        max_inference_seconds=max_inference_seconds,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-model-iteration-loop",
        state_hint=STATE_HINT,
        artifacts={"iteration_report_path": report["iteration_report_path"]},
        metrics={
            "operator_family": report["operator_family"],
            "task_id": report["task_id"],
            "segment_score": report["validation"]["score"]["segment_score"],
            "primary_bottleneck": report["diagnosis"]["primary_bottleneck"],
            "recommended_strategy_family": report["diagnosis"][
                "recommended_strategy_family"
            ],
            "decision": report["decision"]["action"],
        },
        next_hint=report["next_command"],
    )
