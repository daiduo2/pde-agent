import json
from pathlib import Path

from pde_agent.core.data_config import load_data_config
from pde_agent.core.json_output import success_payload
from pde_agent.core.pdebench_adapter import load_array_bundle
from pde_agent.core.real_metrics import rollout_metrics
from pde_agent.core.synthetic_pde_fixture import (
    SYNTHETIC_LONG_HORIZON_NRMSE,
    SYNTHETIC_ONE_STEP_NRMSE,
    synthetic_rollout_payload,
)


def eval_rollout(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    resolved_config_path = root / "config" / "resolved.json"
    if resolved_config_path.exists():
        data_config = load_data_config(resolved_config_path)
        if data_config.mode != "synthetic":
            return _eval_real_rollout(run_id=run_id, root=root)

    rollout_path = root / "metrics" / "rollout.json"
    rollout_payload = synthetic_rollout_payload(run_id=run_id, run_root=root)
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(json.dumps(rollout_payload, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="eval-rollout",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "rollout_metrics_path": str(rollout_path),
            "error_curve_path": rollout_payload["curve"]["path"],
            "prediction_hdf5_path": rollout_payload["prediction"]["path"],
        },
        metrics={
            "one_step_nrmse": SYNTHETIC_ONE_STEP_NRMSE,
            "long_horizon_nrmse": SYNTHETIC_LONG_HORIZON_NRMSE,
        },
    )


def _eval_real_rollout(run_id: str, root: Path) -> dict[str, object]:
    data_config = load_data_config(root / "config" / "resolved.json")
    payload = rollout_metrics(load_array_bundle(data_config))
    rollout_path = root / "metrics" / "rollout.json"
    error_curve_path = root / "metrics" / "error_curve.json"
    source_path = str(data_config.source_path)

    payload["run_id"] = run_id
    curve = payload["curve"]
    prediction = payload["prediction"]
    if not isinstance(curve, dict) or not isinstance(prediction, dict):
        raise KeyError("rollout curve and prediction must be objects")
    curve["path"] = str(error_curve_path)
    prediction["path"] = source_path

    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    error_curve_path.write_text(json.dumps(curve, indent=2), encoding="utf-8")

    return success_payload(
        run_id=run_id,
        command="eval-rollout",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "rollout_metrics_path": str(rollout_path),
            "error_curve_path": str(error_curve_path),
            "prediction_hdf5_path": source_path,
        },
        metrics={
            "source_mode": payload["source_mode"],
            "one_step_nrmse": payload["one_step"]["nrmse"],
            "long_horizon_nrmse": payload["long_horizon"]["nrmse"],
        },
    )
