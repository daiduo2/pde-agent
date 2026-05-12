
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_convergence_grid import (
    STATE_HINT,
    run_spectral_residual_convergence_grid,
)


def run_spectral_residual_convergence_grid_command(
    *,
    audit_path: str,
    output_dir: str,
    source_path: str,
    dataset_key: str,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    summary = run_spectral_residual_convergence_grid(
        audit_path=Path(audit_path),
        output_dir=Path(output_dir),
        source_path=Path(source_path),
        dataset_key=dataset_key,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    best = summary["best_candidate"]
    return success_payload(
        run_id="local",
        command="run-spectral-residual-convergence-grid",
        state_hint=STATE_HINT,
        artifacts={
            "convergence_grid_summary_path": summary[
                "convergence_grid_summary_path"
            ],
        },
        metrics={
            "candidate_count": summary["candidate_count"],
            "best_experiment_id": best["experiment_id"],
            "best_strategy_id": best["best_strategy_id"],
            "best_mean_score_delta": best["mean_score_delta"],
            "best_score_delta_std": best["score_delta_std"],
            "best_failure_rate": best["failure_rate"],
        },
        next_hint=str(summary["next_command"]),
    )
