from __future__ import annotations

from pathlib import Path

from pde_agent.core.competition_submission import (
    STATE_HINT,
    build_competition_submission,
)
from pde_agent.core.json_output import success_payload


def build_competition_submission_command(
    *,
    task_id: int,
    test_path: str,
    output_dir: str,
    submission_id: str,
    dataset_key: str,
    prediction_dataset_key: str,
    method: str,
    source_prediction_path: str | None,
    source_prediction_key: str,
    train_time: float,
    inference_time: float | None,
    force: bool,
) -> dict[str, object]:
    report = build_competition_submission(
        task_id=task_id,
        test_path=Path(test_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method=method,
        source_prediction_path=source_prediction_path,
        source_prediction_key=source_prediction_key,
        train_time=train_time,
        inference_time=inference_time,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-competition-submission",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics=report["metrics"],
        next_hint=str(report["next_command"]),
    )
