
from pathlib import Path

from pde_agent.core.experiment_config_audit import audit_experiment_config_from_file
from pde_agent.core.json_output import success_payload


def audit_experiment_config(
    *,
    config_path: str,
    output_path: str,
    plan_path: str | None,
) -> dict[str, object]:
    audit = audit_experiment_config_from_file(
        config_path=Path(config_path),
        output_path=Path(output_path),
        plan_path=Path(plan_path) if plan_path is not None else None,
    )
    return success_payload(
        run_id="local",
        command="audit-experiment-config",
        state_hint="EXPERIMENT_CONFIG_AUDITED",
        artifacts={"experiment_config_audit_path": audit["audit_path"]},
        metrics={
            "decision": audit["decision"],
            "issue_count": audit["issue_count"],
            "selected_intervention": audit["summary"]["selected_intervention"],
            "candidate_count": audit["summary"]["candidate_count"],
            "requires_training": audit["summary"]["requires_training"],
        },
        next_hint=(
            "execute-candidate-experiment"
            if audit["decision"] == "approved_for_execution"
            else "inspect-experiment-config-audit"
        ),
    )
