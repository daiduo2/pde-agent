
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.pde_hypothesis import select_pde_hypothesis_from_fno_summary


def select_pde_hypothesis_from_fno_summary_command(
    *,
    summary_path: str,
    output_path: str,
) -> dict[str, object]:
    result = select_pde_hypothesis_from_fno_summary(
        summary_path=Path(summary_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="select-pde-hypothesis-from-fno-summary",
        state_hint="PDE_HYPOTHESIS_SELECTED",
        artifacts={"selected_pde_hypothesis_path": result["hypothesis_path"]},
        metrics={
            "selected_hypothesis": result["selected_hypothesis"],
            "next_experiment_scope": result["next_experiment_scope"],
        },
        next_hint="plan-next-experiment",
    )
