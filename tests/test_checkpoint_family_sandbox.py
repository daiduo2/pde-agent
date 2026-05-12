from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def _fake_candidate_report(
    *,
    output_dir: Path,
    family: str,
    value: float,
    inference_seconds: float = 1.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "candidate_prediction.npz"
    prediction = np.full((1, 190, 256), value, dtype=np.float32)
    np.savez_compressed(prediction_path, prediction=prediction)
    report_path = output_dir / "candidate_execution_report.json"
    report = {
        "state_hint": f"{family.upper()}_COMPETITION_PREDICTION_READY",
        "candidate": {
            "experiment_id": f"{family.lower()}_raw",
            "model_family": family,
            "task_id": 1,
        },
        "artifacts": {
            "candidate_prediction_path": str(prediction_path),
            "candidate_execution_report_path": str(report_path),
        },
        "train": {"mode": "official_checkpoint_inference", "elapsed_seconds": 0.0},
        "inference": {"elapsed_seconds": inference_seconds},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report


def test_checkpoint_family_sandbox_runs_checkpoint_candidates_and_calibration(
    tmp_path: Path,
) -> None:
    from pde_agent.core.checkpoint_family_sandbox import run_checkpoint_family_sandbox

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    target[:, 10:, :] = 0.2
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_target.hdf5", target)
    fno_checkpoint = tmp_path / "burgers_FNO.pt"
    unet_checkpoint = tmp_path / "burgers_Unet.pt"
    fno_checkpoint.write_bytes(b"fno")
    unet_checkpoint.write_bytes(b"unet")
    calls: list[tuple[str, Path]] = []

    def fake_fno_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(("FNO", Path(kwargs["checkpoint_path"])))
        return _fake_candidate_report(
            output_dir=Path(kwargs["output_dir"]),
            family="FNO",
            value=0.2,
            inference_seconds=2.0,
        )

    def fake_unet_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(("UNet", Path(kwargs["checkpoint_path"])))
        return _fake_candidate_report(
            output_dir=Path(kwargs["output_dir"]),
            family="UNet",
            value=0.05,
            inference_seconds=3.0,
        )

    report = run_checkpoint_family_sandbox(
        checkpoint_paths=[fno_checkpoint, unet_checkpoint],
        task_id=1,
        test_path=test_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_default",
        calibration_weights=[0.0, 0.5, 1.0],
        fno_runner=fake_fno_runner,
        unet_runner=fake_unet_runner,
        force=False,
    )

    assert report["state_hint"] == "CHECKPOINT_FAMILY_SANDBOX_RAN"
    assert calls == [("FNO", fno_checkpoint), ("UNet", unet_checkpoint)]
    assert report["best_candidate"]["model_family"] == "FNO"
    assert report["best_candidate"]["calibration_weight"] == 1.0
    assert report["best_candidate"]["score"] > 100.0
    assert Path(report["artifacts"]["best_submission_dir"]).exists()
    assert report["next_command"] == "record-strategy-proposal"


def test_checkpoint_family_sandbox_can_execute_fno_finetune_branch(
    tmp_path: Path,
) -> None:
    from pde_agent.core.checkpoint_family_sandbox import run_checkpoint_family_sandbox

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    target[:, 10:, :] = 0.3
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_target.hdf5", target)
    checkpoint = tmp_path / "burgers_FNO.pt"
    checkpoint.write_bytes(b"fno")
    finetune_calls: list[dict[str, Any]] = []

    def fake_fno_runner(**kwargs: Any) -> dict[str, Any]:
        return _fake_candidate_report(
            output_dir=Path(kwargs["output_dir"]),
            family="FNO",
            value=0.05,
            inference_seconds=1.0,
        )

    def fake_finetune_runner(**kwargs: Any) -> dict[str, Any]:
        finetune_calls.append(kwargs)
        return _fake_candidate_report(
            output_dir=Path(kwargs["output_dir"]),
            family="FNO",
            value=0.3,
            inference_seconds=4.0,
        )

    report = run_checkpoint_family_sandbox(
        checkpoint_paths=[checkpoint],
        task_id=1,
        test_path=test_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_default",
        calibration_weights=[1.0],
        fno_finetune_steps=1,
        fno_runner=fake_fno_runner,
        fno_finetune_runner=fake_finetune_runner,
        force=False,
    )

    assert finetune_calls
    assert Path(finetune_calls[0]["windows_path"]).exists()
    assert report["best_candidate"]["source"] == "fno_finetuned"
    assert report["best_candidate"]["score"] > 100.0


def test_checkpoint_family_sandbox_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    import pde_agent.commands.run_checkpoint_family_sandbox as command_module

    def fake_run(**_: Any) -> dict[str, Any]:
        return {
            "state_hint": "CHECKPOINT_FAMILY_SANDBOX_RAN",
            "artifacts": {
                "checkpoint_family_report_path": str(tmp_path / "report.json"),
                "best_submission_dir": str(tmp_path / "submission"),
            },
            "best_candidate": {
                "candidate_id": "fno_calibrated_w1",
                "score": 123.0,
                "model_family": "FNO",
            },
            "candidate_count": 2,
            "next_command": "record-strategy-proposal",
        }

    monkeypatch.setattr(command_module, "run_checkpoint_family_sandbox", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "run-checkpoint-family-sandbox",
            "--checkpoint-path",
            str(tmp_path / "fno.pt"),
            "--task-id",
            "1",
            "--test-path",
            str(tmp_path / "task1_test.hdf5"),
            "--target-path",
            str(tmp_path / "task1_target.hdf5"),
            "--output-dir",
            str(tmp_path / "sandbox"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "CHECKPOINT_FAMILY_SANDBOX_RAN"
    assert payload["metrics"]["best_score"] == 123.0
