from __future__ import annotations

from pathlib import Path

from pde_agent.core.competition_submission_merge import (
    STATE_HINT,
    merge_competition_submissions,
)
from pde_agent.core.json_output import success_payload


def merge_competition_submissions_command(
    *,
    output_dir: str,
    submission_id: str,
    task_submission_dirs: list[str],
    force: bool,
) -> dict[str, object]:
    report = merge_competition_submissions(
        output_dir=Path(output_dir),
        submission_id=submission_id,
        task_submission_dirs=[Path(path) for path in task_submission_dirs],
        force=force,
    )
    return success_payload(
        run_id="local",
        command="merge-competition-submissions",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "task_ids": report["task_ids"],
            "task_count": len(report["task_ids"]),
        },
        next_hint=str(report["next_command"]),
    )
