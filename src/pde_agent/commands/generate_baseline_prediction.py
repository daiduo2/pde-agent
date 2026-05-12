
from pathlib import Path

from pde_agent.core.baseline_prediction import (
    generate_baseline_prediction as generate_prediction,
)
from pde_agent.core.json_output import success_payload


def generate_baseline_prediction(
    *,
    windows_path: str,
    output_path: str,
    method: str,
    force: bool,
) -> dict[str, object]:
    summary = generate_prediction(
        windows_path=Path(windows_path),
        output_path=Path(output_path),
        method=method,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="generate-baseline-prediction",
        state_hint="BASELINE_PREDICTION_READY",
        artifacts={"prediction_npz_path": summary["prediction_path"]},
        metrics={
            "method": summary["method"],
            "sample_count": summary["sample_count"],
            "forecast_steps": summary["forecast_steps"],
            "space_points": summary["space_points"],
        },
        next_hint="score-local-rollout",
    )
