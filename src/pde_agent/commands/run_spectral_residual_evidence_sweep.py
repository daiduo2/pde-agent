from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_evidence_sweep import (
    STATE_HINT,
    run_spectral_residual_evidence_sweep,
)


def run_spectral_residual_evidence_sweep_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    start_indices: list[int],
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    spectral_modes: int,
    spectral_ridge: float,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    evidence = run_spectral_residual_evidence_sweep(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        start_indices=start_indices,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        spectral_modes=spectral_modes,
        spectral_ridge=spectral_ridge,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-spectral-residual-evidence-sweep",
        state_hint=STATE_HINT,
        artifacts={
            "strategy_evidence_summary_path": evidence["evidence_summary_path"],
        },
        metrics={
            "model_family": evidence["model_family"],
            "evaluation_window_count": evidence["evaluation_window_count"],
            "mean_score_delta": evidence["mean_score_delta"],
            "long_window_score_delta": evidence["long_window_score_delta"],
            "failure_rate": evidence["failure_rate"],
            "candidate_record_count": len(evidence["candidate_records"]),
            "strategy_ranking_count": len(evidence["strategy_ranking"]),
            "best_strategy_id": evidence["best_strategy_id"],
        },
        next_hint=str(evidence["next_command"]),
    )
