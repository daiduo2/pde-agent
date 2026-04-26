from __future__ import annotations

from pathlib import Path

from pde_agent.core.burgers_physics_local import check_burgers_physics_npz
from pde_agent.core.json_output import success_payload


def check_burgers_physics_local(
    *,
    prediction_path: str,
    target_path: str,
    output_path: str,
    nu: float,
    dt: float,
    dx: float,
) -> dict[str, object]:
    result = check_burgers_physics_npz(
        prediction_path=Path(prediction_path),
        target_path=Path(target_path),
        output_path=Path(output_path),
        nu=nu,
        dt=dt,
        dx=dx,
    )
    return success_payload(
        run_id="local",
        command="check-burgers-physics-local",
        state_hint="BURGERS_PHYSICS_CHECKED",
        artifacts={"burgers_physics_check_path": result["physics_check_path"]},
        metrics={
            "physics_issue": result["physics_issue"],
            "residual_ratio": result["residual_ratio"],
            "total_variation_ratio": result["total_variation_ratio"],
        },
        next_hint="select-pde-hypothesis",
    )
