
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.unet_backend import predict_unet_local as predict_core


def predict_unet_local(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_path: str,
    metadata_path: str | None,
    inference_batch_size: int | None,
    torch_num_threads: int | None,
    force: bool,
) -> dict[str, object]:
    summary = predict_core(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        metadata_path=Path(metadata_path) if metadata_path is not None else None,
        inference_batch_size=inference_batch_size,
        torch_num_threads=torch_num_threads,
        force=force,
    )
    metrics = {
        "model_type": summary["model_type"],
        "sample_count": summary["sample_count"],
        "forecast_steps": summary["forecast_steps"],
        "space_points": summary["space_points"],
        "checkpoint_format": summary["checkpoint_format"],
        "inference_batch_size": summary["inference_batch_size"],
        "torch_num_threads": summary["torch_num_threads"],
        "rollout_chunk_count": summary["rollout_chunk_count"],
    }
    for key in ("adapter_type", "history_steps", "base_channels", "depth"):
        if key in summary:
            metrics[key] = summary[key]
    return success_payload(
        run_id="local",
        command="predict-unet-local",
        state_hint="UNET_PREDICTION_READY",
        artifacts={"prediction_npz_path": summary["prediction_path"]},
        metrics=metrics,
        next_hint="score-local-rollout",
    )
