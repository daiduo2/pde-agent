from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.next_experiment_plan import plan_next_experiment_from_files


def plan_next_experiment(
    *,
    score_path: str,
    error_diagnosis_path: str,
    physics_check_path: str,
    hypothesis_path: str,
    output_path: str,
) -> dict[str, object]:
    plan = plan_next_experiment_from_files(
        score_path=Path(score_path),
        error_diagnosis_path=Path(error_diagnosis_path),
        physics_check_path=Path(physics_check_path),
        hypothesis_path=Path(hypothesis_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="plan-next-experiment",
        state_hint="NEXT_EXPERIMENT_PLANNED",
        artifacts={"next_experiment_plan_path": plan["next_experiment_plan_path"]},
        metrics={
            "selected_intervention": plan["selected_intervention"],
            "change_scope": plan["change_scope"],
            "max_experiment_count": plan["budget"]["max_experiment_count"],
            "next_command": plan["next_command"],
        },
        next_hint="generate-experiment-config",
    )
