from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_variance_intervention import (
    STATE_HINT,
    plan_spectral_residual_variance_intervention,
)


def plan_spectral_residual_variance_intervention_command(
    *,
    grid_summary_path: str,
    promotion_report_path: str,
    output_path: str,
) -> dict[str, object]:
    plan = plan_spectral_residual_variance_intervention(
        grid_summary_path=Path(grid_summary_path),
        promotion_report_path=Path(promotion_report_path),
        output_path=Path(output_path),
    )
    parameters = plan["intervention_plan"]["candidate_parameters"]
    candidate_count = len(parameters["aggregation_policy"]) * len(
        parameters["window_grouping"]
    )
    return success_payload(
        run_id="local",
        command="plan-spectral-residual-variance-intervention",
        state_hint=STATE_HINT,
        artifacts={"next_experiment_plan_path": plan["next_experiment_plan_path"]},
        metrics={
            "selected_intervention": plan["selected_intervention"],
            "source_strategy_id": plan["source_strategy_id"],
            "primary_failure_mode": plan["variance_attribution"][
                "primary_failure_mode"
            ],
            "candidate_count": candidate_count,
            "may_execute_without_human_review": plan["automation_policy"][
                "may_execute_without_human_review"
            ],
        },
        next_hint=str(plan["next_command"]),
    )
