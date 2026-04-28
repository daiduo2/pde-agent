from __future__ import annotations

from pathlib import Path

from pde_agent.core.burgers_spectral_viscosity_benchmark import (
    STATE_HINT,
    run_burgers_spectral_viscosity_benchmark,
)
from pde_agent.core.json_output import success_payload


def run_burgers_spectral_viscosity_benchmark_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    sample_limit: int,
    min_evaluation_windows: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: list[int],
    hidden_widths: list[int],
    latent_dims: list[int],
    unet_base_channels: list[int],
    unet_depths: list[int],
    learning_rates: list[float],
    seeds: list[int],
    spectral_weights: list[float],
    physics_weights: list[float],
    proposal_id: str,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    report = run_burgers_spectral_viscosity_benchmark(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        sample_limit=sample_limit,
        min_evaluation_windows=min_evaluation_windows,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        train_steps=train_steps,
        hidden_widths=hidden_widths,
        latent_dims=latent_dims,
        unet_base_channels=unet_base_channels,
        unet_depths=unet_depths,
        learning_rates=learning_rates,
        seeds=seeds,
        spectral_weights=spectral_weights,
        physics_weights=physics_weights,
        proposal_id=proposal_id,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-burgers-spectral-viscosity-benchmark",
        state_hint=STATE_HINT,
        artifacts={
            "benchmark_report_path": report["artifacts"]["benchmark_report_path"],
            "strategy_decision_path": report["artifacts"]["strategy_decision_path"],
            "strategy_evidence_summary_path": report["artifacts"][
                "strategy_evidence_summary_path"
            ],
        },
        metrics={
            "proposal_id": report["proposal_id"],
            "evaluation_window_count": report["evaluation_window_count"],
            "best_strategy_id": report["metrics"]["best_strategy_id"],
            "best_model_family": report["metrics"]["best_model_family"],
            "mean_score_delta": report["metrics"]["mean_score_delta"],
            "long_window_score_delta": report["metrics"]["long_window_score_delta"],
            "failure_rate": report["metrics"]["failure_rate"],
            "decision": report["decision"]["decision"],
        },
        next_hint=report["decision"]["next_command"],
    )
