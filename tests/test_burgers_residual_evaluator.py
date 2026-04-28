import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pde_agent.cli import app


def test_evaluate_burgers_residual_arrays_reports_low_residual_for_constant_solution() -> None:
    from pde_agent.core.burgers_residual_evaluator import (
        evaluate_burgers_residual_arrays,
    )

    trajectory = np.ones((2, 6, 16), dtype=np.float32) * 0.25

    result = evaluate_burgers_residual_arrays(
        trajectory=trajectory,
        nu=0.001,
        dt=0.2,
        dx=0.1,
    )

    assert result["state_hint"] == "BURGERS_RESIDUAL_EVALUATED"
    assert result["sample_count"] == 2
    assert result["time_steps"] == 6
    assert result["space_points"] == 16
    assert result["residual_rmse"] == pytest.approx(0.0)
    assert result["mass_drift"] == pytest.approx(0.0)
    assert result["energy_growth_ratio"] == pytest.approx(1.0)
    assert result["physics_issue"] == "stable_low_residual"


def test_evaluate_burgers_residual_arrays_detects_high_frequency_instability() -> None:
    from pde_agent.core.burgers_residual_evaluator import (
        evaluate_burgers_residual_arrays,
    )

    pattern = np.array([1.0, -1.0] * 16, dtype=np.float64)
    trajectory = np.stack(
        [(1.0 + 0.2 * step) * pattern for step in range(6)],
        axis=0,
    )[None, :, :]

    result = evaluate_burgers_residual_arrays(
        trajectory=trajectory,
        nu=0.001,
        dt=0.2,
        dx=0.1,
    )

    assert result["residual_rmse"] > 0.0
    assert result["high_frequency_energy_ratio"] > 0.9
    assert result["physics_issue"] == "high_frequency_instability"


def test_evaluate_burgers_residual_npz_writes_artifact(tmp_path: Path) -> None:
    from pde_agent.core.burgers_residual_evaluator import evaluate_burgers_residual_file

    trajectory_path = tmp_path / "trajectory.npz"
    output_path = tmp_path / "burgers_residual.json"
    np.savez(
        trajectory_path,
        trajectory=np.zeros((1, 5, 8), dtype=np.float32),
    )

    result = evaluate_burgers_residual_file(
        trajectory_path=trajectory_path,
        output_path=output_path,
        array_key="trajectory",
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert output_path.exists()
    assert result == json.loads(output_path.read_text(encoding="utf-8"))
    assert result["trajectory_path"] == str(trajectory_path)
    assert result["array_key"] == "trajectory"
    assert result["physics_issue"] == "stable_low_residual"
    assert result["next_command"] == "select-pde-hypothesis"


def test_evaluate_burgers_residual_hdf5_reads_tensor_key(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    from pde_agent.core.burgers_residual_evaluator import evaluate_burgers_residual_file

    trajectory_path = tmp_path / "trajectory.hdf5"
    output_path = tmp_path / "burgers_residual.json"
    with h5py.File(trajectory_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((1, 5, 8), dtype=np.float32))

    result = evaluate_burgers_residual_file(
        trajectory_path=trajectory_path,
        output_path=output_path,
        array_key="tensor",
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["source_format"] == "hdf5"
    assert result["array_key"] == "tensor"
    assert result["residual_rmse"] == pytest.approx(0.0)


def test_evaluate_burgers_residual_cli_returns_protocol_payload(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.npz"
    output_path = tmp_path / "burgers_residual.json"
    np.savez(
        trajectory_path,
        trajectory=np.zeros((1, 5, 8), dtype=np.float32),
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-burgers-residual",
            "--trajectory-path",
            str(trajectory_path),
            "--output-path",
            str(output_path),
            "--array-key",
            "trajectory",
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
    assert payload["command"] == "evaluate-burgers-residual"
    assert payload["state_hint"] == "BURGERS_RESIDUAL_EVALUATED"
    assert payload["artifacts"]["burgers_residual_path"] == str(output_path)
    assert payload["metrics"]["physics_issue"] == "stable_low_residual"
    assert payload["metrics"]["residual_rmse"] == pytest.approx(0.0)
    assert payload["next_hint"] == "select-pde-hypothesis"
