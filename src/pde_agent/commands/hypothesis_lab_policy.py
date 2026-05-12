from pathlib import Path

from pde_agent.core.hypothesis_lab_policy import (
    STATE_HINT,
    build_hypothesis_lab_next_policy,
)
from pde_agent.core.json_output import success_payload


def build_hypothesis_lab_next_policy_command(
    *,
    output_path: str,
    failure_memory_path: str | None,
    hypothesis_lab_report_path: str | None,
    force: bool,
) -> dict[str, object]:
    report = build_hypothesis_lab_next_policy(
        output_path=Path(output_path),
        failure_memory_path=Path(failure_memory_path)
        if failure_memory_path is not None
        else None,
        hypothesis_lab_report_path=Path(hypothesis_lab_report_path)
        if hypothesis_lab_report_path is not None
        else None,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-hypothesis-lab-next-policy",
        state_hint=STATE_HINT,
        artifacts={
            "policy_path": output_path,
            "hypothesis_lab_next_policy_json": output_path,
        },
        metrics={
            "next_action": report["policy"]["next_action"],
            "candidate_generator": report["policy"]["candidate_generator"],
            "learning_objective": report["policy"]["learning_objective"],
        },
        next_hint=report["next_command"],
    )
