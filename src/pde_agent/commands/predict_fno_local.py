from __future__ import annotations

from pathlib import Path

from pde_agent.core.fno_backend import predict_fno_local as predict_core
from pde_agent.core.json_output import success_payload


def predict_fno_local(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_path: str,
    metadata_path: str | None,
    force: bool,
) -> dict[str, object]:
    summary = predict_core(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        metadata_path=Path(metadata_path) if metadata_path is not None else None,
        force=force,
    )
    metrics = {
        "model_type": summary["model_type"],
        "sample_count": summary["sample_count"],
        "forecast_steps": summary["forecast_steps"],
        "space_points": summary["space_points"],
        "checkpoint_format": summary["checkpoint_format"],
    }
    for key in ("adapter_type", "history_steps", "modes", "width"):
        if key in summary:
            metrics[key] = summary[key]
    return success_payload(
        run_id="local",
        command="predict-fno-local",
        state_hint="FNO_PREDICTION_READY",
        artifacts={"prediction_npz_path": summary["prediction_path"]},
        metrics=metrics,
        next_hint="score-local-rollout",
    )
