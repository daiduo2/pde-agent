"""Run official-style UNet checkpoints on competition test inputs."""


import json
import time
from pathlib import Path
from typing import Any

from pde_agent.core.competition_windows import prepare_competition_windows
from pde_agent.core.unet_backend import predict_unet_local


STATE_HINT = "UNET_COMPETITION_PREDICTION_READY"


class UNetCompetitionPredictionError(RuntimeError):
    pass


def predict_unet_competition(
    *,
    task_id: int,
    test_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    dataset_key: str = "tensor",
    force: bool = False,
    inference_batch_size: int | None = None,
    torch_num_threads: int | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    report_path = output_root / "candidate_execution_report.json"
    if report_path.exists() and not force:
        raise UNetCompetitionPredictionError(
            f"UNet competition report already exists: {report_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    windows = prepare_competition_windows(
        task_id=task_id,
        test_path=Path(test_path),
        output_path=output_root / "competition_windows.npz",
        dataset_key=dataset_key,
        force=force,
    )
    start = time.perf_counter()
    prediction = predict_unet_local(
        windows_path=Path(windows["windows_path"]),
        checkpoint_path=Path(checkpoint_path),
        output_path=output_root / "candidate_prediction.npz",
        metadata_path=None,
        force=force,
        inference_batch_size=inference_batch_size,
        torch_num_threads=torch_num_threads,
    )
    inference_time = time.perf_counter() - start
    experiment_id = f"unet_competition_task{task_id}"
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "test_path": str(test_path),
        "checkpoint_path": str(checkpoint_path),
        "inputs": {
            "context_steps": windows["context_steps"],
            "forecast_steps": windows["forecast_steps"],
            "space_points": windows["space_points"],
            "sample_count": windows["sample_count"],
        },
        "candidate": {
            "experiment_id": experiment_id,
            "model_family": "UNet",
            "task_id": task_id,
        },
        "artifacts": {
            "competition_windows_path": windows["windows_path"],
            "candidate_prediction_path": prediction["prediction_path"],
            "candidate_execution_report_path": str(report_path),
        },
        "train": {
            "mode": "official_checkpoint_inference",
            "elapsed_seconds": 0.0,
        },
        "inference": {
            "elapsed_seconds": inference_time,
            "backend": prediction["adapter_type"],
            "checkpoint_format": prediction["checkpoint_format"],
            "inference_batch_size": prediction["inference_batch_size"],
            "torch_num_threads": prediction["torch_num_threads"],
            "rollout_chunk_count": prediction["rollout_chunk_count"],
            "phase_timings": prediction["phase_timings"],
        },
        "next_command": "materialize-competition-candidate",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
