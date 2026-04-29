import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _official_fno_state_dict(
    *,
    torch: object,
    history_steps: int = 3,
    width: int = 4,
    modes: int = 2,
    fc_hidden: int = 5,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(31)
    state: dict[str, object] = {
        "fc0.weight": 0.001 * torch.randn(width, history_steps + 1, generator=generator),
        "fc0.bias": torch.zeros(width),
        "fc1.weight": 0.001 * torch.randn(fc_hidden, width, generator=generator),
        "fc1.bias": torch.zeros(fc_hidden),
        "fc2.weight": 0.001 * torch.randn(1, fc_hidden, generator=generator),
        "fc2.bias": torch.zeros(1),
    }
    for index in range(4):
        state[f"conv{index}.weights1"] = 0.001 * torch.randn(
            width,
            width,
            modes,
            dtype=torch.complex64,
            generator=generator,
        )
        state[f"w{index}.weight"] = 0.001 * torch.randn(
            width,
            width,
            1,
            generator=generator,
        )
        state[f"w{index}.bias"] = torch.zeros(width)
    return state


def _write_hdf5(path: Path, data: np.ndarray) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO competition test requires torch")
def test_predict_fno_competition_writes_candidate_report(tmp_path: Path) -> None:
    import torch

    from pde_agent.core.fno_competition_prediction import predict_fno_competition

    test_path = _write_hdf5(
        tmp_path / "task1_test.hdf5",
        np.ones((1, 10, 256), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "fno.pt"
    torch.save(
        {"model_state_dict": _official_fno_state_dict(torch=torch)},
        checkpoint_path,
    )

    report = predict_fno_competition(
        task_id=1,
        test_path=test_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "fno_competition",
        dataset_key="tensor",
        force=False,
    )

    assert report["state_hint"] == "FNO_COMPETITION_PREDICTION_READY"
    assert report["candidate"]["experiment_id"] == "fno_competition_task1"
    assert Path(report["artifacts"]["candidate_execution_report_path"]).exists()
    with np.load(report["artifacts"]["candidate_prediction_path"], allow_pickle=False) as archive:
        assert archive["prediction"].shape == (1, 190, 256)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO competition test requires torch")
def test_predict_fno_competition_cli_returns_payload(tmp_path: Path) -> None:
    import torch
    from typer.testing import CliRunner

    from pde_agent.cli import app

    test_path = _write_hdf5(
        tmp_path / "task1_test.hdf5",
        np.ones((1, 10, 256), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "fno.pt"
    torch.save(
        {"model_state_dict": _official_fno_state_dict(torch=torch)},
        checkpoint_path,
    )

    result = CliRunner().invoke(
        app,
        [
            "predict-fno-competition",
            "--task-id",
            "1",
            "--test-path",
            str(test_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(tmp_path / "fno_competition"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "predict-fno-competition"
    assert payload["state_hint"] == "FNO_COMPETITION_PREDICTION_READY"
    assert payload["metrics"]["forecast_steps"] == 190
