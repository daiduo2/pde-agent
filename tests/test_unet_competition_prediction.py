import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _official_unet_state_dict(
    *,
    torch: object,
    history_steps: int = 3,
    base_channels: int = 1,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(23)
    channels = [
        base_channels,
        base_channels * 2,
        base_channels * 4,
        base_channels * 8,
        base_channels * 16,
    ]

    def conv_block(prefix: str, in_channels: int, out_channels: int) -> dict[str, object]:
        return {
            f"{prefix}conv1.weight": 0.001
            * torch.randn(out_channels, in_channels, 3, generator=generator),
            f"{prefix}norm1.weight": torch.ones(out_channels),
            f"{prefix}norm1.bias": torch.zeros(out_channels),
            f"{prefix}norm1.running_mean": torch.zeros(out_channels),
            f"{prefix}norm1.running_var": torch.ones(out_channels),
            f"{prefix}norm1.num_batches_tracked": torch.tensor(0, dtype=torch.long),
            f"{prefix}conv2.weight": 0.001
            * torch.randn(out_channels, out_channels, 3, generator=generator),
            f"{prefix}norm2.weight": torch.ones(out_channels),
            f"{prefix}norm2.bias": torch.zeros(out_channels),
            f"{prefix}norm2.running_mean": torch.zeros(out_channels),
            f"{prefix}norm2.running_var": torch.ones(out_channels),
            f"{prefix}norm2.num_batches_tracked": torch.tensor(0, dtype=torch.long),
        }

    state: dict[str, object] = {}
    state.update(
        {
            f"encoder1.{key}": value
            for key, value in conv_block("enc1", history_steps, channels[0]).items()
        }
    )
    for index, prefix in enumerate(("enc2", "enc3", "enc4"), start=1):
        state.update(
            {
                f"encoder{index + 1}.{key}": value
                for key, value in conv_block(prefix, channels[index - 1], channels[index]).items()
            }
        )
    state.update(
        {
            f"bottleneck.{key}": value
            for key, value in conv_block("bottleneck", channels[3], channels[4]).items()
        }
    )
    for level, out_index in ((4, 3), (3, 2), (2, 1), (1, 0)):
        state[f"upconv{level}.weight"] = 0.001 * torch.randn(
            channels[out_index + 1],
            channels[out_index],
            2,
            generator=generator,
        )
        state[f"upconv{level}.bias"] = torch.zeros(channels[out_index])
        state.update(
            {
                f"decoder{level}.{key}": value
                for key, value in conv_block(
                    f"dec{level}", channels[out_index + 1], channels[out_index]
                ).items()
            }
        )
    state["conv.weight"] = 0.001 * torch.randn(1, channels[0], 1, generator=generator)
    state["conv.bias"] = torch.zeros(1)
    return state


def _write_hdf5(path: Path, data: np.ndarray) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="UNet competition test requires torch")
def test_predict_unet_competition_writes_candidate_report(tmp_path: Path) -> None:
    import torch

    from pde_agent.core.unet_competition_prediction import predict_unet_competition

    test_path = _write_hdf5(
        tmp_path / "task1_test.hdf5",
        np.ones((1, 10, 256), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "unet.pt"
    torch.save(
        {"model_state_dict": _official_unet_state_dict(torch=torch)},
        checkpoint_path,
    )

    report = predict_unet_competition(
        task_id=1,
        test_path=test_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "unet_competition",
        dataset_key="tensor",
        force=False,
    )

    assert report["state_hint"] == "UNET_COMPETITION_PREDICTION_READY"
    assert report["candidate"]["experiment_id"] == "unet_competition_task1"
    assert Path(report["artifacts"]["candidate_execution_report_path"]).exists()
    with np.load(report["artifacts"]["candidate_prediction_path"], allow_pickle=False) as archive:
        assert archive["prediction"].shape == (1, 190, 256)
    candidate_report = json.loads(
        Path(report["artifacts"]["candidate_execution_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert candidate_report["artifacts"]["candidate_prediction_path"] == report[
        "artifacts"
    ]["candidate_prediction_path"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="UNet competition test requires torch")
def test_predict_unet_competition_cli_returns_payload(tmp_path: Path) -> None:
    import torch
    from typer.testing import CliRunner

    from pde_agent.cli import app

    test_path = _write_hdf5(
        tmp_path / "task1_test.hdf5",
        np.ones((1, 10, 256), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "unet.pt"
    torch.save(
        {"model_state_dict": _official_unet_state_dict(torch=torch)},
        checkpoint_path,
    )

    result = CliRunner().invoke(
        app,
        [
            "predict-unet-competition",
            "--task-id",
            "1",
            "--test-path",
            str(test_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(tmp_path / "unet_competition"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "predict-unet-competition"
    assert payload["state_hint"] == "UNET_COMPETITION_PREDICTION_READY"
    assert payload["metrics"]["forecast_steps"] == 190
