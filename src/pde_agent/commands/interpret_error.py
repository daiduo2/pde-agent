from pathlib import Path

from pde_agent.core.diagnostic_synthesis import (
    interpretation_payload,
    read_json,
    write_json,
)
from pde_agent.core.json_output import success_payload


def interpret_error(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    rollout = read_json(root / "metrics" / "rollout.json")
    spectrum = read_json(root / "metrics" / "spectrum.json")
    physics = read_json(root / "metrics" / "physics_check.json")
    payload = interpretation_payload(
        run_id=run_id,
        rollout=rollout,
        spectrum=spectrum,
        physics=physics,
    )
    interpretation_path = root / "diagnostics" / "error_interpretation.json"
    write_json(interpretation_path, payload)
    return success_payload(
        run_id=run_id,
        command="interpret-error",
        state_hint="ERROR_INTERPRETED",
        artifacts={"error_interpretation_path": str(interpretation_path)},
        metrics={
            "dominant_failure_mode": payload["dominant_failure_mode"],
            "confidence": payload["confidence"],
        },
    )
