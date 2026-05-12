
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.task2_source_policy import STATE_HINT, audit_task2_source_policy


def audit_task2_source_policy_command(
    *,
    train_data_paths: list[str],
    test_data_paths: list[str],
    checkpoint_paths: list[str],
    candidate_source_paths: list[str],
    output_path: str,
) -> dict[str, object]:
    report = audit_task2_source_policy(
        train_data_paths=[Path(path) for path in train_data_paths],
        test_data_paths=[Path(path) for path in test_data_paths],
        checkpoint_paths=[Path(path) for path in checkpoint_paths],
        candidate_source_paths=[Path(path) for path in candidate_source_paths],
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="audit-task2-source-policy",
        state_hint=STATE_HINT,
        artifacts={"audit_path": report["audit_path"]},
        metrics={
            "decision": report["decision"],
            "issue_count": report["issue_count"],
            "issue_codes": report["issue_codes"],
        },
        next_hint=report["next_command"],
    )
