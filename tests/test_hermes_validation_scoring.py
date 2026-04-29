import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def test_run_hermes_validation_scoring_builds_context_target_and_scores(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_validation_scoring import (
        run_hermes_validation_scoring,
    )

    target = np.ones((2, 200, 256), dtype=np.float32)
    target[:, :10, :] = np.arange(2 * 10 * 256, dtype=np.float32).reshape(
        2, 10, 256
    )
    validation_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    calls: dict[str, Any] = {}

    def fake_predictor(**kwargs: Any) -> dict[str, Any]:
        calls["predictor"] = kwargs
        output_dir = Path(kwargs["output_dir"])
        prediction_path = output_dir / "candidate_prediction.npz"
        report_path = output_dir / "candidate_execution_report.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(prediction_path, prediction=target[:, 10:, :])
        report = {
            "state_hint": "UNET_COMPETITION_PREDICTION_READY",
            "candidate": {"model_family": "UNet", "task_id": kwargs["task_id"]},
            "artifacts": {"candidate_prediction_path": str(prediction_path)},
            "train": {"elapsed_seconds": 0.0},
            "inference": {"elapsed_seconds": 3.25},
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return {
            **report,
            "artifacts": {
                **report["artifacts"],
                "candidate_execution_report_path": str(report_path),
            },
        }

    report = run_hermes_validation_scoring(
        output_dir=tmp_path / "validation",
        operator_family="UNet",
        task_id=1,
        validation_path=validation_path,
        checkpoint_path="burgers_Unet.tar",
        inference_batch_size=8,
        torch_num_threads=1,
        predictor=fake_predictor,
        force=False,
    )

    assert report["state_hint"] == "HERMES_VALIDATION_SCORED"
    assert calls["predictor"]["test_path"] == Path(
        report["artifacts"]["validation_context_path"]
    )
    assert calls["predictor"]["inference_batch_size"] == 8
    assert report["score"]["segment_score"] == pytest.approx(100.0)
    assert report["score"]["task_prediction_score"] == pytest.approx(75.0)
    assert Path(report["artifacts"]["validation_context_path"]).exists()
    assert Path(report["artifacts"]["validation_target_path"]).exists()
    assert report["next_command"] == "diagnose-hermes-model-quality"


def test_hermes_validation_scoring_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_validation_scoring as command_module
    from typer.testing import CliRunner

    def fake_runner(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "validation.json"
        score_path = tmp_path / "score.json"
        report_path.write_text("{}", encoding="utf-8")
        score_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_VALIDATION_SCORED",
            "validation_report_path": str(report_path),
            "operator_family": "UNet",
            "task_id": 1,
            "score": {
                "segment_score": 87.5,
                "task_prediction_score": 65.625,
                "sample_count": 100,
            },
            "artifacts": {
                "validation_report_path": str(report_path),
                "score_json_path": str(score_path),
            },
            "next_command": "diagnose-hermes-model-quality",
        }

    monkeypatch.setattr(command_module, "run_hermes_validation_scoring", fake_runner)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-validation-scoring",
            "--output-dir",
            str(tmp_path / "validation"),
            "--operator-family",
            "UNet",
            "--task-id",
            "1",
            "--validation-path",
            str(tmp_path / "task1_val.hdf5"),
            "--checkpoint-path",
            "burgers_Unet.tar",
            "--inference-batch-size",
            "8",
            "--torch-num-threads",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_VALIDATION_SCORED"
    assert payload["metrics"]["segment_score"] == 87.5
    assert payload["next_hint"] == "diagnose-hermes-model-quality"
