
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.real_hdf5_baseline_smoke import (
    STATE_HINT,
    run_real_hdf5_baseline_smoke,
)


def run_real_hdf5_baseline_smoke_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    start_index: int,
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
    summary = run_real_hdf5_baseline_smoke(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        start_index=start_index,
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
    return success_payload(
        run_id="local",
        command="run-real-hdf5-baseline-smoke",
        state_hint=STATE_HINT,
        artifacts={"summary_path": summary["artifacts"]["summary_path"]},
        metrics={
            "sample_count": summary["metrics"]["sample_count"],
            "forecast_steps": summary["metrics"]["forecast_steps"],
            "competition_like_score_short": summary["metrics"][
                "competition_like_score_short"
            ],
            "rel_mse": summary["metrics"]["rel_mse"],
            "residual_rmse": summary["metrics"]["residual_rmse"],
            "dominant_error_pattern": summary["metrics"]["dominant_error_pattern"],
            "physics_issue": summary["metrics"]["physics_issue"],
            "residual_issue": summary["metrics"]["residual_issue"],
        },
        next_hint=str(summary["next_command"]),
    )
