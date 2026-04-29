import json
from pathlib import Path

import numpy as np


def _write_hdf5(path: Path, data: np.ndarray) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


def test_prepare_competition_windows_from_reduced_test_hdf5(tmp_path: Path) -> None:
    from pde_agent.core.competition_windows import prepare_competition_windows

    data = np.arange(3 * 10 * 256, dtype=np.float32).reshape(3, 10, 256)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", data)
    output_path = tmp_path / "competition_windows.npz"

    report = prepare_competition_windows(
        task_id=1,
        test_path=test_path,
        output_path=output_path,
        dataset_key="tensor",
        force=False,
    )

    assert report["state_hint"] == "COMPETITION_WINDOWS_READY"
    assert report["context_shape"] == [3, 10, 256]
    assert report["target_shape"] == [3, 190, 256]
    with np.load(output_path, allow_pickle=False) as archive:
        context = archive["context"]
        target = archive["target"]
        metadata = json.loads(str(archive["metadata_json"]))
    np.testing.assert_allclose(context, data)
    assert target.shape == (3, 190, 256)
    assert np.all(target == 0.0)
    assert metadata["task_id"] == 1
    assert metadata["forecast_steps"] == 190


def test_prepare_competition_windows_reduces_raw_space(tmp_path: Path) -> None:
    from pde_agent.core.competition_windows import prepare_competition_windows

    data = np.ones((2, 12, 1024), dtype=np.float32)
    test_path = _write_hdf5(tmp_path / "task2_test.hdf5", data)
    output_path = tmp_path / "competition_windows.npz"

    report = prepare_competition_windows(
        task_id=2,
        test_path=test_path,
        output_path=output_path,
        dataset_key="tensor",
        force=False,
    )

    assert report["context_shape"] == [2, 10, 256]
    with np.load(output_path, allow_pickle=False) as archive:
        assert archive["context"].shape == (2, 10, 256)


def test_prepare_competition_windows_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    data = np.ones((2, 10, 256), dtype=np.float32)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", data)
    output_path = tmp_path / "competition_windows.npz"

    result = CliRunner().invoke(
        app,
        [
            "prepare-competition-windows",
            "--task-id",
            "1",
            "--test-path",
            str(test_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "prepare-competition-windows"
    assert payload["state_hint"] == "COMPETITION_WINDOWS_READY"
    assert payload["metrics"]["forecast_steps"] == 190
