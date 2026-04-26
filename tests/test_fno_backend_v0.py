import json
from pathlib import Path

import pytest

from pde_agent.core.fno_backend import (
    FNOBackendError,
    inspect_fno_checkpoint,
)

torch = pytest.importorskip("torch", reason="FNO backend tests require torch extra")


def _write_fixture_checkpoint(path: Path) -> None:
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
