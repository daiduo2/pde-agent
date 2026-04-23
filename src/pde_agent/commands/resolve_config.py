import hashlib
import json
from pathlib import Path

from pde_agent.core.config_resolver import resolve_config_files
from pde_agent.core.json_output import success_payload


def resolve_config(
    run_id: str,
    base_config_file: str,
    override_file: str,
    task_spec_file: str,
    run_root: str,
) -> dict[str, object]:
    resolved = resolve_config_files(
        Path(base_config_file),
        Path(override_file),
        Path(task_spec_file),
    )
    resolved_path = Path(run_root) / "config" / "resolved.json"
    resolved_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    config_hash = hashlib.sha256(
        json.dumps(resolved, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return success_payload(
        run_id=run_id,
        command="resolve-config",
        state_hint="BASELINE_READY",
        artifacts={
            "resolved_config_path": str(resolved_path),
            "config_hash": config_hash,
        },
        metrics={
            "input_window": resolved.get("input_window"),
            "prediction_mode": resolved.get("prediction_mode"),
        },
    )
