from __future__ import annotations

from pathlib import Path

from pde_agent.core.candidate_execution_prep import (
    prepare_candidate_execution_from_audit,
)
from pde_agent.core.json_output import success_payload


def prepare_candidate_execution(
    *,
    audit_path: str,
    output_dir: str,
    candidate_index: int,
    force: bool,
) -> dict[str, object]:
    plan = prepare_candidate_execution_from_audit(
        audit_path=Path(audit_path),
        output_dir=Path(output_dir),
        candidate_index=candidate_index,
        force=force,
    )
    candidate = plan["candidate"]
    workspace = plan["workspace"]
    return success_payload(
        run_id="local",
        command="prepare-candidate-execution",
        state_hint="CANDIDATE_EXECUTION_PREPARED",
        artifacts={
            "candidate_execution_plan_path": workspace[
                "candidate_execution_plan_path"
            ],
            "workspace_dir": workspace["workspace_dir"],
        },
        metrics={
            "candidate_index": plan["candidate_index"],
            "experiment_id": candidate["experiment_id"],
            "intervention": candidate["intervention"],
            "requires_training": candidate["requires_training"],
            "execution_status": plan["execution_status"],
        },
        next_hint="execute-candidate-experiment",
    )
