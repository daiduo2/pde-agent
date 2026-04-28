import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

from pde_agent.core.unet_backend import UNetBackendError, inspect_unet_checkpoint

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _official_unet_state_dict(
    *,
    torch: object,
    history_steps: int = 3,
    base_channels: int = 4,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(11)
    channels = [
        base_channels,
        base_channels * 2,
        base_channels * 4,
        base_channels * 8,
        base_channels * 16,
    ]

    def conv_block(prefix: str, in_channels: int, out_channels: int) -> dict[str, object]:
        return {
            f"{prefix}conv1.weight": 0.01
            * torch.randn(
                out_channels,
                in_channels,
                3,
                generator=generator,
            ),
            f"{prefix}norm1.weight": torch.ones(out_channels),
            f"{prefix}norm1.bias": torch.zeros(out_channels),
            f"{prefix}norm1.running_mean": torch.zeros(out_channels),
            f"{prefix}norm1.running_var": torch.ones(out_channels),
            f"{prefix}norm1.num_batches_tracked": torch.tensor(0, dtype=torch.long),
            f"{prefix}conv2.weight": 0.01
            * torch.randn(
                out_channels,
                out_channels,
                3,
                generator=generator,
            ),
            f"{prefix}norm2.weight": torch.ones(out_channels),
            f"{prefix}norm2.bias": torch.zeros(out_channels),
            f"{prefix}norm2.running_mean": torch.zeros(out_channels),
            f"{prefix}norm2.running_var": torch.ones(out_channels),
            f"{prefix}norm2.num_batches_tracked": torch.tensor(0, dtype=torch.long),
        }

    state: dict[str, object] = {}
    state.update(
        {f"encoder1.{key}": value for key, value in conv_block("enc1", history_steps, channels[0]).items()}
    )
    state.update(
        {f"encoder2.{key}": value for key, value in conv_block("enc2", channels[0], channels[1]).items()}
    )
    state.update(
        {f"encoder3.{key}": value for key, value in conv_block("enc3", channels[1], channels[2]).items()}
    )
    state.update(
        {f"encoder4.{key}": value for key, value in conv_block("enc4", channels[2], channels[3]).items()}
    )
    state.update(
        {
            f"bottleneck.{key}": value
            for key, value in conv_block("bottleneck", channels[3], channels[4]).items()
        }
    )
    state["upconv4.weight"] = 0.01 * torch.randn(
        channels[4],
        channels[3],
        2,
        generator=generator,
    )
    state["upconv4.bias"] = torch.zeros(channels[3])
    state.update(
        {f"decoder4.{key}": value for key, value in conv_block("dec4", channels[4], channels[3]).items()}
    )
    state["upconv3.weight"] = 0.01 * torch.randn(
        channels[3],
        channels[2],
        2,
        generator=generator,
    )
    state["upconv3.bias"] = torch.zeros(channels[2])
    state.update(
        {f"decoder3.{key}": value for key, value in conv_block("dec3", channels[3], channels[2]).items()}
    )
    state["upconv2.weight"] = 0.01 * torch.randn(
        channels[2],
        channels[1],
        2,
        generator=generator,
    )
    state["upconv2.bias"] = torch.zeros(channels[1])
    state.update(
        {f"decoder2.{key}": value for key, value in conv_block("dec2", channels[2], channels[1]).items()}
    )
    state["upconv1.weight"] = 0.01 * torch.randn(
        channels[1],
        channels[0],
        2,
        generator=generator,
    )
    state["upconv1.bias"] = torch.zeros(channels[0])
    state.update(
        {f"decoder1.{key}": value for key, value in conv_block("dec1", channels[1], channels[0]).items()}
    )
    state["conv.weight"] = 0.01 * torch.randn(1, channels[0], 1, generator=generator)
    state["conv.bias"] = torch.zeros(1)
    return state


def _write_windows(path: Path) -> tuple[object, object]:
    np = pytest.importorskip("numpy")
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    context = np.zeros((2, 4, 32), dtype=np.float32)
    target = np.zeros((2, 3, 32), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.1)
        delta = 0.04 * np.cos(x) + 0.01 * sample
        for time_index in range(4):
            context[sample, time_index, :] = base + time_index * delta
        for time_index in range(3):
            target[sample, time_index, :] = base + (4 + time_index) * delta
    with path.open("wb") as handle:
        np.savez(handle, context=context, target=target)
    return context, target


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="UNet backend tests require torch extra")
def test_infer_pdebench_unet1d_config_from_state_dict() -> None:
    import torch

    from pde_agent.core.unet_backend import _infer_pdebench_unet1d_config

    state = _official_unet_state_dict(
        torch=torch,
        history_steps=3,
        base_channels=4,
    )

    config = _infer_pdebench_unet1d_config(state, torch=torch)

    assert config.history_steps == 3
    assert config.base_channels == 4
    assert config.depth == 4


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="UNet backend tests require torch extra")
def test_pdebench_unet1d_model_loads_official_state_dict_strictly() -> None:
    import torch

    from pde_agent.core.unet_backend import (
        _infer_pdebench_unet1d_config,
        _make_pdebench_unet1d_class,
    )

    state = _official_unet_state_dict(torch=torch)
    config = _infer_pdebench_unet1d_config(state, torch=torch)
    model_class = _make_pdebench_unet1d_class(torch)
    model = model_class(config)

    missing, unexpected = model.load_state_dict(state, strict=True)

    assert missing == []
    assert unexpected == []


def test_infer_pdebench_unet1d_config_rejects_non_tensor_values() -> None:
    from pde_agent.core.unet_backend import (
        _infer_pdebench_unet1d_config,
        _pdebench_unet1d_required_keys,
    )

    state = {key: object() for key in _pdebench_unet1d_required_keys()}

    with pytest.raises(UNetBackendError, match="tensor"):
        _infer_pdebench_unet1d_config(state)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="UNet backend tests require torch extra")
def test_inspect_unet_checkpoint_tar_selects_first_pt_member(tmp_path: Path) -> None:
    import torch

    first_checkpoint = tmp_path / "1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt"
    second_checkpoint = tmp_path / "1D_Burgers_Sols_Nu0.01_Unet-PF-20.pt"
    archive_path = tmp_path / "burgers_Unet.tar"
    output_path = tmp_path / "metadata.json"
    torch.save({"model_state_dict": _official_unet_state_dict(torch=torch)}, first_checkpoint)
    torch.save({"model_state_dict": _official_unet_state_dict(torch=torch)}, second_checkpoint)

    with tarfile.open(archive_path, "w") as archive:
        archive.add(second_checkpoint, arcname=second_checkpoint.name)
        archive.add(first_checkpoint, arcname=first_checkpoint.name)

    metadata = inspect_unet_checkpoint(
        checkpoint_path=archive_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert metadata["state_hint"] == "UNET_CHECKPOINT_INSPECTED"
    assert metadata["archive_member"] == first_checkpoint.name
    assert metadata["official_unet1d_detected"] is True
    assert metadata["history_steps"] == 3
    assert metadata["base_channels"] == 4
    assert any("multiple" in warning for warning in metadata["load_warnings"])
    assert json.loads(output_path.read_text(encoding="utf-8")) == metadata


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="UNet backend tests require torch extra")
def test_predict_unet_local_official_like_state(tmp_path: Path) -> None:
    import torch

    from pde_agent.core.unet_backend import predict_unet_local

    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unet.pt"
    output_path = tmp_path / "prediction.npz"
    _write_windows(windows_path)
    torch.save({"model_state_dict": _official_unet_state_dict(torch=torch)}, checkpoint_path)

    summary = predict_unet_local(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        force=False,
    )

    assert output_path.exists()
    assert summary["model_type"] == "unet"
    assert summary["adapter_type"] == "pdebench_unet1d"
    assert summary["history_steps"] == 3
    assert summary["prediction_shape"] == [2, 3, 32]
    np = pytest.importorskip("numpy")
    with np.load(output_path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
        metadata = json.loads(str(archive["metadata_json"]))
    assert prediction.shape == (2, 3, 32)
    assert np.all(np.isfinite(prediction))
    assert metadata["adapter_type"] == "pdebench_unet1d"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="UNet backend tests require torch extra")
def test_predict_unet_local_cli_writes_payload(tmp_path: Path) -> None:
    import torch
    from typer.testing import CliRunner

    from pde_agent.cli import app

    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unet.pt"
    output_path = tmp_path / "prediction.npz"
    _write_windows(windows_path)
    torch.save({"model_state_dict": _official_unet_state_dict(torch=torch)}, checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "predict-unet-local",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "predict-unet-local"
    assert payload["state_hint"] == "UNET_PREDICTION_READY"
    assert payload["metrics"]["adapter_type"] == "pdebench_unet1d"
