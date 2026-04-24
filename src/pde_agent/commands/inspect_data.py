import json
from pathlib import Path

from pde_agent.core.data_config import load_data_config
from pde_agent.core.json_output import success_payload
from pde_agent.core.pdebench_adapter import inspect_data_source


def inspect_data(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    resolved_config_path = root / "config" / "resolved.json"
    data_config = load_data_config(resolved_config_path)
    inspection = inspect_data_source(data_config)

    inspection_path = root / "metrics" / "data_inspection.json"
    inspection_path.parent.mkdir(parents=True, exist_ok=True)
    inspection_path.write_text(json.dumps(inspection, indent=2), encoding="utf-8")

    return success_payload(
        run_id=run_id,
        command="inspect-data",
        state_hint="DATA_INSPECTED",
        artifacts={"data_inspection_path": str(inspection_path)},
        metrics={
            "source_mode": inspection["source_mode"],
            "is_compatible": inspection["is_compatible"],
            "sample_count": inspection["sample_count"],
        },
    )
