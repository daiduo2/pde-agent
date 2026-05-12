
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_window_stratification import (
    STATE_HINT,
    run_spectral_residual_window_stratification_grid,
)


def run_spectral_residual_window_stratification_command(
    *,
    audit_path: str,
    output_dir: str,
    force: bool,
) -> dict[str, object]:
    summary = run_spectral_residual_window_stratification_grid(
        audit_path=Path(audit_path),
        output_dir=Path(output_dir),
        force=force,
    )
    best = summary["best_candidate"]
    return success_payload(
        run_id="local",
        command="run-spectral-residual-window-stratification",
        state_hint=STATE_HINT,
        artifacts={
            "window_stratification_summary_path": summary[
                "window_stratification_summary_path"
            ],
            "best_evidence_summary_path": best["evidence_summary_path"],
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
