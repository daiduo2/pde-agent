from __future__ import annotations

from pathlib import Path

from pde_agent.core.competition_task_pipeline import (
    STATE_HINT,
    run_competition_task_pipeline,
)
from pde_agent.core.json_output import success_payload


def run_competition_task_pipeline_command(
    *,
    task_id: int,
    model_family: str,
    test_path: str,
    checkpoint_path: str,
    output_dir: str,
    submission_id: str,
    target_path: str | None,
    dataset_key: str,
    force: bool,
) -> dict[str, object]:
    report = run_competition_task_pipeline(
        task_id=task_id,
        model_family=model_family,
        test_path=Path(test_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        target_path=Path(target_path) if target_path is not None else None,
        dataset_key=dataset_key,
        force=force,
    )
    materialized = report["materialized"]
    score = materialized.get("score")
    return success_payload(
        run_id="local",
        command="run-competition-task-pipeline",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "task_id": report["task_id"],
            "model_family": report["model_family"],
            "audit_decision": report["audit"]["decision"],
            "audit_issue_count": report["audit"]["issue_count"],
            "has_score": score is not None,
            "segment_score": score["segment_score"] if score is not None else None,
        },
        next_hint=str(report["next_command"]),
    )
