import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.manifest_builder import build_manifest


def collect_artifacts(run_id: str, run_root: str, config_hash: str = "unknown") -> dict[str, object]:
    root = Path(run_root)
    manifest = build_manifest(
        run_id=run_id,
        config={
            "resolved_config_path": str(root / "config" / "resolved.json"),
            "config_hash": config_hash,
        },
        artifacts={
            "rollout_metrics_path": str(root / "metrics" / "rollout.json"),
            "timing_metrics_path": str(root / "metrics" / "timing.json"),
        },
        summary={
            "one_step_nrmse": 0.0,
            "long_horizon_nrmse": 0.0,
            "train_time_seconds": 0.0,
            "inference_time_seconds": 0.0,
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
        state_hint="ERROR_ATTRIBUTED",
        artifacts={"artifact_manifest_path": str(manifest_path)},
        metrics={"artifact_count": 3},
    )
