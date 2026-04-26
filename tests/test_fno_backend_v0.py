import importlib.util
import json
from pathlib import Path

import pytest

from pde_agent.core.fno_backend import (
    FNOBackendError,
    _collect_tensors,
    inspect_fno_checkpoint,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _write_fixture_checkpoint(path: Path) -> None:
    import torch

    torch.save(
        {
            "model_type": "fno_v0_fixture",
            "prediction_rule": "delta_extrapolation",
            "config": {
                "modes": 8,
                "width": 16,
            },
            "model_state_dict": {
                "weight": torch.ones(2, 3),
                "bias": torch.zeros(3),
            },
        },
        path,
    )


def _write_windows(path: Path) -> tuple[object, object]:
    np = pytest.importorskip("numpy")
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    context = np.zeros((2, 4, 8), dtype=np.float32)
    target = np.zeros((2, 3, 8), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.1)
        delta = 0.05 * np.cos(x) + 0.01 * sample
        for t in range(4):
            context[sample, t, :] = base + t * delta
        for t in range(3):
            target[sample, t, :] = base + (4 + t) * delta
    with path.open("wb") as handle:
        np.savez(
            handle,
            context=context,
            target=target,
            metadata_json=json.dumps({"fixture": True}),
        )
    return context, target


class FakeTorch:
    @staticmethod
    def is_tensor(value: object) -> bool:
        return False


def test_collect_tensors_ignores_cyclic_dict() -> None:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle

    assert _collect_tensors(cycle, torch=FakeTorch()) == []


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_inspect_fno_checkpoint_fixture(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_path = tmp_path / "metadata.json"
    _write_fixture_checkpoint(checkpoint_path)

    metadata = inspect_fno_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert metadata["state_hint"] == "FNO_CHECKPOINT_INSPECTED"
    assert metadata["checkpoint_path"] == str(checkpoint_path)
    assert metadata["checkpoint_format"] == "torch_serialized"
    assert metadata["has_model_state"] is True
    assert metadata["has_config"] is True
    assert metadata["tensor_summary"]["tensor_key_count"] == 2
    assert metadata["tensor_summary"]["parameter_count"] == 9
    assert "model_state_dict" in metadata["top_level_keys"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == metadata
    assert issubclass(FNOBackendError, RuntimeError)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_predict_fno_local_fixture_writes_prediction(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_path = tmp_path / "prediction"
    context, target = _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    from pde_agent.core.fno_backend import predict_fno_local

    summary = predict_fno_local(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        metadata_path=None,
        force=False,
    )

    np = pytest.importorskip("numpy")
    assert output_path.exists()
    assert not output_path.with_suffix(".npz").exists()
    with np.load(output_path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
        metadata = json.loads(str(archive["metadata_json"]))
    assert prediction.shape == target.shape
    assert prediction.dtype == np.float32
    assert metadata["model_type"] == "fno"
    assert metadata["prediction_rule"] == "delta_extrapolation"
    assert summary["prediction_shape"] == list(target.shape)
    assert summary["model_type"] == "fno"
    assert context.shape[0] == prediction.shape[0]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_predict_fno_local_rejects_unsupported_checkpoint(tmp_path: Path) -> None:
    import torch

    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unsupported.pt"
    _write_windows(windows_path)
    torch.save({"model_type": "unknown"}, checkpoint_path)

    from pde_agent.core.fno_backend import predict_fno_local

    with pytest.raises(FNOBackendError, match="unsupported"):
        predict_fno_local(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "prediction.npz",
            metadata_path=None,
            force=False,
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_inspect_fno_checkpoint_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_path = tmp_path / "metadata.json"
    _write_fixture_checkpoint(checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "inspect-fno-checkpoint",
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "inspect-fno-checkpoint"
    assert payload["state_hint"] == "FNO_CHECKPOINT_INSPECTED"
    assert payload["artifacts"]["fno_checkpoint_metadata_path"] == str(output_path)
    assert payload["metrics"]["checkpoint_format"] == "torch_serialized"
    assert payload["next_hint"] == "predict-fno-local"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_predict_fno_local_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_path = tmp_path / "prediction.npz"
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "predict-fno-local",
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
    assert payload["command"] == "predict-fno-local"
    assert payload["state_hint"] == "FNO_PREDICTION_READY"
    assert payload["artifacts"]["prediction_npz_path"] == str(output_path)
    assert payload["metrics"]["model_type"] == "fno"
    assert payload["next_hint"] == "score-local-rollout"
    assert output_path.exists()
