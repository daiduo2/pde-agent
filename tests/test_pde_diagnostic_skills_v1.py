import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pde_agent.cli import app
from pde_agent.core.baseline_prediction import (
    BaselinePredictionError,
    generate_baseline_prediction,
)


def _write_windows(path: Path, context: np.ndarray, target: np.ndarray) -> None:
    full_reference = np.concatenate([context, target], axis=1)
    np.savez(
        path,
        context=context.astype(np.float32),
        target=target.astype(np.float32),
        full_reference=full_reference.astype(np.float32),
        metadata_json=json.dumps({"fixture": True}),
    )


def _write_prediction(path: Path, prediction: np.ndarray) -> None:
    np.savez(path, prediction=prediction.astype(np.float32))


def test_generate_baseline_prediction_repeats_last_context_frame(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    context = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    target = np.zeros((2, 5, 4), dtype=np.float32)
    _write_windows(windows_path, context, target)

    summary = generate_baseline_prediction(
        windows_path=windows_path,
        output_path=output_path,
        method="persistence",
        force=False,
    )

    expected = np.repeat(context[:, -1:, :], repeats=5, axis=1)
    with np.load(output_path, allow_pickle=False) as archive:
        assert np.array_equal(archive["prediction"], expected)
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["method"] == "persistence"
    assert summary["prediction_shape"] == [2, 5, 4]
    assert summary["method"] == "persistence"


def test_generate_baseline_prediction_refuses_overwrite_without_force(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    _write_windows(
        windows_path,
        np.ones((1, 2, 3), dtype=np.float32),
        np.ones((1, 4, 3), dtype=np.float32),
    )
    output_path.write_text("exists", encoding="utf-8")

    with pytest.raises(BaselinePredictionError, match="already exists"):
        generate_baseline_prediction(
            windows_path=windows_path,
            output_path=output_path,
            method="persistence",
            force=False,
        )


def test_generate_baseline_prediction_cli_writes_payload(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    _write_windows(
        windows_path,
        np.ones((2, 3, 4), dtype=np.float32),
        np.zeros((2, 5, 4), dtype=np.float32),
    )

    result = CliRunner().invoke(
        app,
        [
            "generate-baseline-prediction",
            "--windows-path",
            str(windows_path),
            "--output-path",
            str(output_path),
            "--method",
            "persistence",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "generate-baseline-prediction"
    assert payload["state_hint"] == "BASELINE_PREDICTION_READY"
    assert payload["artifacts"]["prediction_npz_path"] == str(output_path)
    assert payload["metrics"] == {
        "method": "persistence",
        "sample_count": 2,
        "forecast_steps": 5,
        "space_points": 4,
    }
    assert output_path.exists()
