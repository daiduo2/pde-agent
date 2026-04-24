import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pde_agent.cli import app
from pde_agent.commands.inspect_data import inspect_data


def test_inspect_data_writes_synthetic_inspection_by_default(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run-001"
    config_path = run_root / "config" / "resolved.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"data": {}}), encoding="utf-8")

    result = inspect_data(run_id="run-001", run_root=str(run_root))

    inspection_path = run_root / "metrics" / "data_inspection.json"
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    assert inspection["source_mode"] == "synthetic"
    assert inspection["is_compatible"] is True
    assert inspection["sample_count"] is None
    assert result["command"] == "inspect-data"
    assert result["state_hint"] == "DATA_INSPECTED"
    assert result["artifacts"] == {"data_inspection_path": str(inspection_path)}
    assert result["metrics"] == {
        "source_mode": "synthetic",
        "is_compatible": True,
        "sample_count": None,
    }


def test_inspect_data_writes_npz_fixture_inspection(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.npz"
    np.savez(
        source_path,
        context=np.ones((2, 3, 5), dtype=np.float32),
        prediction=np.ones((2, 4, 5), dtype=np.float64),
        target=np.ones((2, 4, 5), dtype=np.float64),
    )
    run_root = tmp_path / "runs" / "run-002"
    config_path = run_root / "config" / "resolved.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "mode": "npz_fixture",
                    "source_path": str(source_path),
                    "context_steps": 3,
                    "forecast_steps": 4,
                    "time_reduction": 1,
                    "space_reduction": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    result = inspect_data(run_id="run-002", run_root=str(run_root))

    inspection_path = run_root / "metrics" / "data_inspection.json"
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    assert inspection["source_mode"] == "npz_fixture"
    assert inspection["source_path"] == str(source_path)
    assert inspection["sample_count"] == 2
    assert inspection["contract_shape"] == {
        "context": [2, 3, 5],
        "prediction": [2, 4, 5],
        "target": [2, 4, 5],
    }
    assert result["metrics"] == {
        "source_mode": "npz_fixture",
        "is_compatible": True,
        "sample_count": 2,
    }


def test_inspect_data_cli_smoke(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run-003"
    config_path = run_root / "config" / "resolved.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"data": {}}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "inspect-data",
            "--run-id",
            "run-003",
            "--run-root",
            str(run_root),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "inspect-data"
    assert payload["state_hint"] == "DATA_INSPECTED"
