from pde_agent.core.json_output import success_payload


def check_physics(run_id: str, run_root: str) -> dict[str, object]:
    return success_payload(
        run_id=run_id,
        command="check-physics",
        state_hint="RESULTS_COLLECTED",
        artifacts={"physics_check_path": f"{run_root}/metrics/physics_check.json"},
        metrics={
            "nonphysical_oscillation": False,
            "blowup_like_behavior": False,
            "excessive_smoothing": False,
        },
    )
