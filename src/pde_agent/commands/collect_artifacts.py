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
    prediction_hdf5_path = rollout_payload["prediction"]["path"]
    error_curve_path = rollout_payload["curve"]["path"]
    time_csv_path = timing_payload["time_csv_path"]
    artifacts = {
        "rollout_metrics_path": str(rollout_path),
        "timing_metrics_path": str(timing_path),
        "prediction_hdf5_path": prediction_hdf5_path,
        "error_curve_path": error_curve_path,
        "time_csv_path": time_csv_path,
    }
    manifest = build_manifest(
        run_id=run_id,
        config={
            "resolved_config_path": str(root / "config" / "resolved.json"),
            "config_hash": config_hash,
        },
        artifacts=artifacts,
        summary={
            "one_step_nrmse": rollout_payload["one_step"]["nrmse"],
            "long_horizon_nrmse": rollout_payload["long_horizon"]["nrmse"],
            "train_time_seconds": timing_payload["train_time_seconds"],
            "inference_time_seconds": timing_payload["inference_time_seconds"],
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
        metrics={"artifact_count": len(artifacts)},
    )
