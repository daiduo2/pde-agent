from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.pde_hypothesis import select_pde_hypothesis_from_files


def select_pde_hypothesis(
    *,
    score_path: str,
    error_diagnosis_path: str,
    physics_check_path: str,
    output_path: str,
) -> dict[str, object]:
    result = select_pde_hypothesis_from_files(
        score_path=Path(score_path),
        error_diagnosis_path=Path(error_diagnosis_path),
        physics_check_path=Path(physics_check_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="select-pde-hypothesis",
        state_hint="PDE_HYPOTHESIS_SELECTED",
        artifacts={"selected_pde_hypothesis_path": result["hypothesis_path"]},
        metrics={
            "selected_hypothesis": result["selected_hypothesis"],
            "next_experiment_scope": result["next_experiment_scope"],
        },
        next_hint="Use the selected PDE hypothesis to define the next local experiment.",
    )
