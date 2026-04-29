from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_validation_scoring import (
    STATE_HINT,
    run_hermes_validation_scoring,
)
from pde_agent.core.json_output import success_payload


def run_hermes_validation_scoring_command(
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
    force: bool,
) -> dict[str, object]:
    report = run_hermes_validation_scoring(
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
        force=force,
    )
    score = report["score"]
    return success_payload(
        run_id="local",
        command="run-hermes-validation-scoring",
        state_hint=STATE_HINT,
        artifacts={
            "validation_report_path": report["validation_report_path"],
            "score_json_path": report["artifacts"]["score_json_path"],
        },
        metrics={
            "operator_family": report["operator_family"],
            "task_id": report["task_id"],
            "sample_count": score["sample_count"],
            "segment_score": score["segment_score"],
            "task_prediction_score": score["task_prediction_score"],
        },
        next_hint=report["next_command"],
    )
