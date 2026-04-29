from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_local_judge import (
    STATE_HINT,
    judge_hermes_submission,
)
from pde_agent.core.json_output import success_payload


def judge_hermes_submission_command(
    *,
    submission_dir: str,
    output_path: str,
    task1_test_path: str | None,
    task1_target_path: str | None,
    task2_test_path: str | None,
    task2_target_path: str | None,
    dataset_key: str,
    prediction_dataset_key: str,
) -> dict[str, object]:
    report = judge_hermes_submission(
        submission_dir=Path(submission_dir),
        output_path=Path(output_path),
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
        task2_target_path=Path(task2_target_path)
        if task2_target_path is not None
        else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
    )
    return success_payload(
        run_id="local",
        command="judge-hermes-submission",
        state_hint=STATE_HINT,
        artifacts={"judge_report_path": report["judge_report_path"]},
        metrics={
            "rules_passed": report["rules_passed"],
            "total_score_proxy": report["total_score_proxy"],
            "decision": report["decision"],
            "task_count": len(report["tasks"]),
        },
        next_hint=report["decision"],
    )
