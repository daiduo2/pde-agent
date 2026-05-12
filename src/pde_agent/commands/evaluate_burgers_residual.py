
from pathlib import Path

from pde_agent.core.burgers_residual_evaluator import (
    STATE_HINT,
    evaluate_burgers_residual_file,
)
from pde_agent.core.json_output import success_payload


def evaluate_burgers_residual_command(
    *,
    trajectory_path: str,
    output_path: str,
    array_key: str,
    nu: float,
    dt: float,
    dx: float,
) -> dict[str, object]:
    result = evaluate_burgers_residual_file(
        trajectory_path=Path(trajectory_path),
        output_path=Path(output_path),
        array_key=array_key,
        nu=nu,
        dt=dt,
        dx=dx,
    )
    return success_payload(
        run_id="local",
        command="evaluate-burgers-residual",
        state_hint=STATE_HINT,
        artifacts={"burgers_residual_path": result["burgers_residual_path"]},
        metrics={
            "physics_issue": result["physics_issue"],
            "residual_rmse": result["residual_rmse"],
            "mass_drift": result["mass_drift"],
            "energy_growth_ratio": result["energy_growth_ratio"],
            "high_frequency_energy_ratio": result["high_frequency_energy_ratio"],
        },
        next_hint=str(result["next_command"]),
    )
