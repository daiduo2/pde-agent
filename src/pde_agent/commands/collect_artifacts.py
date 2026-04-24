import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.manifest_builder import build_manifest


def collect_artifacts(run_id: str, run_root: str, config_hash: str = "unknown") -> dict[str, object]:
    root = Path(run_root)
    rollout_path = root / "metrics" / "rollout.json"
    timing_path = root / "metrics" / "timing.json"
    rollout_payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    prediction_hdf5_path = rollout_payload.get("prediction", {}).get("path")
    error_curve_path = rollout_payload.get("curve", {}).get("path")
    time_csv_path = timing_payload.get("time_csv_path")
    manifest = build_manifest(
        run_id=run_id,
        config={
            "resolved_config_path": str(root / "config" / "resolved.json"),
            "config_hash": config_hash,
        },
        artifacts={
            "rollout_metrics_path": str(rollout_path),
            "timing_metrics_path": str(timing_path),
            "prediction_hdf5_path": prediction_hdf5_path,
            "error_curve_path": error_curve_path,
            "time_csv_path": time_csv_path,
        },
        summary={
            "one_step_nrmse": rollout_payload.get("one_step", {}).get("nrmse", 0.0),
            "long_horizon_nrmse": rollout_payload.get("long_horizon", {}).get("nrmse", 0.0),
            "train_time_seconds": timing_payload.get("train_time_seconds", 0.0),
            "inference_time_seconds": timing_payload.get("inference_time_seconds", 0.0),
        },
        completeness={
            "has_rollout": True,
            "has_spectrum": False,
            "has_physics_check": False,
            "has_timing": True,
        },
    )
    manifest_path = root / "artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="collect-artifacts",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "artifact_manifest_path": str(manifest_path),
            "prediction_hdf5_path": prediction_hdf5_path,
            "error_curve_path": error_curve_path,
            "time_csv_path": time_csv_path,
        },
        metrics={"artifact_count": 6},
    )
