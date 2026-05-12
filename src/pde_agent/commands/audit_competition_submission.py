
from pathlib import Path

from pde_agent.core.competition_submission_audit import (
    STATE_HINT,
    audit_competition_submission,
)
from pde_agent.core.json_output import success_payload


def audit_competition_submission_command(
    *,
    task_id: int,
    submission_dir: str,
    test_path: str,
    output_path: str,
    dataset_key: str,
) -> dict[str, object]:
    audit = audit_competition_submission(
        task_id=task_id,
        submission_dir=Path(submission_dir),
        test_path=Path(test_path),
        output_path=Path(output_path),
        dataset_key=dataset_key,
    )
    return success_payload(
        run_id="local",
        command="audit-competition-submission",
        state_hint=STATE_HINT,
        artifacts={"audit_path": audit["audit_path"]},
        metrics={
            "task_id": audit["task_id"],
            "decision": audit["decision"],
            "issue_count": audit["issue_count"],
        },
        next_hint="submit-if-passed-or-repair-submission",
    )
