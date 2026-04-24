from pathlib import Path

from pde_agent.core.diagnostic_synthesis import physics_payload, read_json, write_json
from pde_agent.core.data_config import load_data_config
from pde_agent.core.json_output import success_payload
from pde_agent.core.pdebench_adapter import load_array_bundle
from pde_agent.core.real_metrics import physics_metrics


def check_physics(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    resolved_config_path = root / "config" / "resolved.json"
    if resolved_config_path.exists():
        data_config = load_data_config(resolved_config_path)
        if data_config.mode != "synthetic":
            payload = physics_metrics(load_array_bundle(data_config))
            physics_path = root / "metrics" / "physics_check.json"
            write_json(physics_path, payload)
            return success_payload(
                run_id=run_id,
                command="check-physics",
                state_hint="PHYSICS_CHECKED",
                artifacts={"physics_check_path": str(physics_path)},
                metrics={
                    "source_mode": payload["source_mode"],
                    "mass_drift": payload["mass_drift"],
                    "energy_decay_ratio": payload["energy_decay_ratio"],
                    "total_variation_ratio": payload["total_variation_ratio"],
                    "physics_issue": payload["physics_issue"],
                },
            )

    rollout = read_json(root / "metrics" / "rollout.json")
    payload = physics_payload(run_id=run_id, rollout=rollout)
    physics_path = root / "metrics" / "physics_check.json"
    write_json(physics_path, payload)
    return success_payload(
        run_id=run_id,
        command="check-physics",
        state_hint="PHYSICS_CHECKED",
        artifacts={"physics_check_path": str(physics_path)},
        metrics={
            "mass_drift": payload["mass_drift"],
            "energy_decay_ratio": payload["energy_decay_ratio"],
            "total_variation_ratio": payload["total_variation_ratio"],
            "physics_issue": payload["physics_issue"],
        },
    )
