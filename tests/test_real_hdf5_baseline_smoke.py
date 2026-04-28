import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pde_agent.cli import app


def _write_hdf5(path: Path) -> Path:
    h5py = pytest.importorskip("h5py")
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
    tensor = np.zeros((3, 30, 16), dtype=np.float32)
    for sample in range(tensor.shape[0]):
        for time in range(tensor.shape[1]):
            tensor[sample, time, :] = np.sin(x + 0.1 * sample + 0.03 * time)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)
    return path


def test_run_real_hdf5_baseline_smoke_writes_prediction_and_metrics(
    tmp_path: Path,
) -> None:
    from pde_agent.core.real_hdf5_baseline_smoke import run_real_hdf5_baseline_smoke

    source_path = _write_hdf5(tmp_path / "burgers.hdf5")
    output_dir = tmp_path / "smoke"

    summary = run_real_hdf5_baseline_smoke(
        source_path=source_path,
        output_dir=output_dir,
        dataset_key="tensor",
        start_index=0,
        sample_limit=2,
        context_steps=4,
        forecast_steps=8,
        time_reduction=2,
        space_reduction=2,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert summary["schema_version"] == 1
    assert summary["state_hint"] == "REAL_HDF5_BASELINE_SMOKE_COMPLETED"
    assert summary["inputs"]["sample_limit"] == 2
    assert summary["inputs"]["forecast_steps"] == 8
    assert summary["metrics"]["competition_like_score_short"] > 0.0
    assert summary["metrics"]["residual_rmse"] >= 0.0
    assert summary["metrics"]["dominant_error_pattern"]
    assert summary["metrics"]["physics_issue"]
    assert summary["artifacts"]["windows_npz_path"].endswith("windows_f008.npz")
    for path in summary["artifacts"].values():
        assert Path(path).exists()
    written = json.loads((output_dir / "real_hdf5_baseline_smoke_summary.json").read_text())
    assert written == summary


def test_run_real_hdf5_baseline_smoke_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    source_path = _write_hdf5(tmp_path / "burgers.hdf5")
    output_dir = tmp_path / "smoke"

    result = CliRunner().invoke(
        app,
        [
            "run-real-hdf5-baseline-smoke",
            "--source-path",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--dataset-key",
            "tensor",
            "--start-index",
            "0",
            "--sample-limit",
            "2",
            "--context-steps",
            "4",
            "--forecast-steps",
            "8",
            "--time-reduction",
            "2",
            "--space-reduction",
            "2",
            "--nu",
            "0.001",
            "--dt",
            "1.0",
            "--dx",
            "1.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-real-hdf5-baseline-smoke"
    assert payload["state_hint"] == "REAL_HDF5_BASELINE_SMOKE_COMPLETED"
    assert payload["artifacts"]["summary_path"] == str(
        output_dir / "real_hdf5_baseline_smoke_summary.json"
    )
    assert payload["metrics"]["sample_count"] == 2
    assert payload["metrics"]["forecast_steps"] == 8
    assert payload["next_hint"] == "inspect-real-baseline-smoke-summary"
