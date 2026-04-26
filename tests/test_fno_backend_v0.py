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
