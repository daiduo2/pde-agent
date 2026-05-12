
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_baseline import (
    predict_spectral_residual_baseline,
)


def predict_local_model(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    summary = predict_spectral_residual_baseline(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="predict-local-model",
        state_hint="LOCAL_MODEL_PREDICTED",
        artifacts={"prediction_npz_path": summary["prediction_path"]},
        metrics={
            "model_type": summary["model_type"],
            "sample_count": summary["sample_count"],
            "forecast_steps": summary["forecast_steps"],
            "space_points": summary["space_points"],
        },
        next_hint="score-local-rollout",
    )
