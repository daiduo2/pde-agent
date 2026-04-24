import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.synthetic_pde_fixture import (
    SYNTHETIC_LONG_HORIZON_NRMSE,
    SYNTHETIC_ONE_STEP_NRMSE,
    synthetic_rollout_payload,
)


def eval_rollout(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    rollout_path = root / "metrics" / "rollout.json"
    rollout_payload = synthetic_rollout_payload(run_id=run_id, run_root=root)
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
