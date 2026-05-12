
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_next_experiment import (
    STATE_HINT,
    plan_next_spectral_residual_experiment,
)


def plan_next_spectral_residual_experiment_command(
    *,
    evidence_summary_path: str,
    promotion_report_path: str,
    ledger_path: str,
    output_path: str,
) -> dict[str, object]:
    plan = plan_next_spectral_residual_experiment(
        evidence_summary_path=Path(evidence_summary_path),
        promotion_report_path=Path(promotion_report_path),
        ledger_path=Path(ledger_path),
        output_path=Path(output_path),
    )
    parameters = plan["intervention_plan"]["candidate_parameters"]
    candidate_count = len(parameters["modes"]) * len(parameters["ridge"])
    return success_payload(
        run_id="local",
        command="plan-next-spectral-residual-experiment",
        state_hint=STATE_HINT,
        artifacts={"next_experiment_plan_path": plan["next_experiment_plan_path"]},
        metrics={
            "selected_intervention": plan["selected_intervention"],
            "source_strategy_id": plan["source_strategy_id"],
            "candidate_count": candidate_count,
            "reason_count": len(plan["failure_mode"]["reason_codes"]),
            "may_execute_without_human_review": plan["automation_policy"][
                "may_execute_without_human_review"
            ],
        },
        next_hint=str(plan["next_command"]),
    )
