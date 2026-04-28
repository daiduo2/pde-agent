from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.official_operator_comparison import (
    STATE_HINT,
    run_official_operator_comparison,
)


def run_official_operator_comparison_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    fno_checkpoint_path: str,
    unet_checkpoint_path: str,
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
    report = run_official_operator_comparison(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        fno_checkpoint_path=Path(fno_checkpoint_path),
        unet_checkpoint_path=Path(unet_checkpoint_path),
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
        command="run-official-operator-comparison",
        state_hint=STATE_HINT,
        artifacts={
            "comparison_report_path": report["artifacts"][
                "comparison_report_path"
            ],
        },
        metrics={
            "best_model_id": report["best_model_id"],
            "best_model_family": report["best_model_family"],
            "best_score": report["best_score"],
            "best_score_delta_vs_persistence": report[
                "best_score_delta_vs_persistence"
            ],
            "model_count": report["model_count"],
            "official_model_count": report["official_model_count"],
            "recommendation": report["recommendation"]["next_action"],
        },
        next_hint=report["recommendation"]["next_action"],
    )
