import json
from pathlib import Path

from pde_agent.core.config_resolver import resolve_config_files


def test_resolve_config_files_merges_base_task_and_override(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    task = tmp_path / "task.json"
    override = tmp_path / "override.json"

    base.write_text(json.dumps({"model_family": "fno", "epochs": 20}), encoding="utf-8")
    task.write_text(
        json.dumps({"input_window": 10, "prediction_mode": "one_step"}),
        encoding="utf-8",
    )
    override.write_text(json.dumps({"epochs": 50}), encoding="utf-8")

    resolved = resolve_config_files(base, override, task)

    assert resolved["model_family"] == "fno"
    assert resolved["epochs"] == 50
    assert resolved["input_window"] == 10
    assert resolved["prediction_mode"] == "one_step"
