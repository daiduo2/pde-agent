import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.synthetic_pde_fixture import (
    SYNTHETIC_INFERENCE_TIME_SECONDS,
    SYNTHETIC_TRAIN_TIME_SECONDS,
    synthetic_timing_payload,
)


def record_timing(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    timing_path = root / "metrics" / "timing.json"
    if (root / "config" / "request.json").exists():
        payload = synthetic_timing_payload(run_id=run_id, run_root=root)
        metrics = {
            "train_time_seconds": SYNTHETIC_TRAIN_TIME_SECONDS,
            "inference_time_seconds": SYNTHETIC_INFERENCE_TIME_SECONDS,
        }
    else:
        payload = {
            "run_id": run_id,
            "train_time_seconds": 0.0,
            "inference_time_seconds": 0.0,
            "time_csv_path": str(root / "submission" / "time.csv"),
        }
        metrics = {"train_time_seconds": 0.0, "inference_time_seconds": 0.0}
    timing_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="record-timing",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "timing_metrics_path": str(timing_path),
            "time_csv_path": payload["time_csv_path"],
        },
        metrics=metrics,
    )
