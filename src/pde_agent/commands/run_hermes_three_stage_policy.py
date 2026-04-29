from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_three_stage_policy import (
    STATE_HINT,
    run_hermes_three_stage_policy,
)
from pde_agent.core.json_output import success_payload


def run_hermes_three_stage_policy_command(
    *,
    output_dir: str,
    initial_submission_dir: str,
    task1_test_path: str | None,
    task1_target_path: str | None,
    task2_test_path: str | None,
    task2_target_path: str | None,
    unet_checkpoint_path: str | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    stage_deadline_seconds: float,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_three_stage_policy(
        output_dir=Path(output_dir),
        initial_submission_dir=Path(initial_submission_dir),
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
        task2_target_path=Path(task2_target_path)
        if task2_target_path is not None
        else None,
        unet_checkpoint_path=Path(unet_checkpoint_path)
        if unet_checkpoint_path is not None
        else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        submission_id=submission_id,
        stage_deadline_seconds=stage_deadline_seconds,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-three-stage-policy",
        state_hint=STATE_HINT,
        artifacts={"policy_report_path": report["policy_report_path"]},
        metrics={
            "stage_count": report["stage_count"],
            "best_candidate_id": report["best_candidate"].get("candidate_id"),
            "best_score": report["best_score"],
            "decision": report["decision"],
        },
        next_hint=report["next_command"],
    )
