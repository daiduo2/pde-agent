"""Run official-style FNO checkpoints on competition test inputs."""


import json
import time
from pathlib import Path
from typing import Any

from pde_agent.core.competition_windows import prepare_competition_windows
from pde_agent.core.fno_backend import predict_fno_local


STATE_HINT = "FNO_COMPETITION_PREDICTION_READY"


class FNOCompetitionPredictionError(RuntimeError):
    pass


def predict_fno_competition(
    *,
    task_id: int,
    test_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    dataset_key: str = "tensor",
    force: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    report_path = output_root / "candidate_execution_report.json"
    if report_path.exists() and not force:
        raise FNOCompetitionPredictionError(
            f"FNO competition report already exists: {report_path}"
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
    prediction = predict_fno_local(
        windows_path=Path(windows["windows_path"]),
        checkpoint_path=Path(checkpoint_path),
        output_path=output_root / "candidate_prediction.npz",
        metadata_path=None,
        force=force,
    )
    inference_time = time.perf_counter() - start
    experiment_id = f"fno_competition_task{task_id}"
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
            "model_family": "FNO",
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
        },
        "next_command": "materialize-competition-candidate",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
