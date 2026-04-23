import json
from pathlib import Path

from pde_agent.core.json_output import success_payload


def record_timing(run_id: str, run_root: str) -> dict[str, object]:
    timing_path = Path(run_root) / "metrics" / "timing.json"
    payload = {
        "run_id": run_id,
        "train_time_seconds": 0.0,
        "inference_time_seconds": 0.0,
        "time_csv_path": str(Path(run_root) / "submission" / "time.csv"),
    }
    timing_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="record-timing",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "timing_metrics_path": str(timing_path),
            "time_csv_path": payload["time_csv_path"],
        },
        metrics={"train_time_seconds": 0.0, "inference_time_seconds": 0.0},
    )
