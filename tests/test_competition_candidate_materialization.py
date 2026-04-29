import json
from pathlib import Path

import numpy as np
import pytest


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def _write_candidate_report(root: Path, prediction_path: Path) -> Path:
    report = {
        "schema_version": 1,
        "state_hint": "UNET_CANDIDATE_EXECUTED",
        "candidate": {"experiment_id": "unet_competition_ready"},
        "artifacts": {"candidate_prediction_path": str(prediction_path)},
        "train": {"elapsed_seconds": 12.5},
    }
    path = root / "candidate_execution_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_materialize_competition_candidate_builds_submission_and_score(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_candidate_materialization import (
        materialize_competition_candidate,
    )

    context = np.arange(2 * 10 * 256, dtype=np.float32).reshape(2, 10, 256)
    future = np.full((2, 190, 256), 3.0, dtype=np.float32)
    prediction_path = tmp_path / "candidate_prediction.npz"
    np.savez(prediction_path, prediction=future)
    candidate_report_path = _write_candidate_report(tmp_path, prediction_path)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target = np.concatenate([context, future], axis=1)
    target_path = _write_hdf5(tmp_path / "task1_gt.hdf5", target)

    report = materialize_competition_candidate(
        task_id=1,
        candidate_report_path=candidate_report_path,
        test_path=test_path,
        output_dir=tmp_path / "materialized",
        submission_id="team_hermes",
        target_path=target_path,
        force=True,
    )

    assert report["state_hint"] == "COMPETITION_CANDIDATE_MATERIALIZED"
    assert report["candidate"]["experiment_id"] == "unet_competition_ready"
    assert Path(report["artifacts"]["submission_zip_path"]).exists()
    assert Path(report["artifacts"]["score_json_path"]).exists()
    assert report["score"]["segment_score"] == pytest.approx(100.0)
    assert report["submission"]["metrics"]["train_time"] == pytest.approx(12.5)
    assert report["next_command"] == "review-or-submit-competition-package"


def test_materialize_competition_candidate_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    context = np.ones((2, 10, 256), dtype=np.float32)
    future = np.ones((2, 190, 256), dtype=np.float32)
    prediction_path = tmp_path / "candidate_prediction.npz"
    np.savez(prediction_path, prediction=future)
    candidate_report_path = _write_candidate_report(tmp_path, prediction_path)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)

    result = CliRunner().invoke(
        app,
        [
            "materialize-competition-candidate",
            "--task-id",
            "1",
            "--candidate-report-path",
            str(candidate_report_path),
            "--test-path",
            str(test_path),
            "--output-dir",
            str(tmp_path / "materialized"),
            "--submission-id",
            "team_hermes",
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "materialize-competition-candidate"
    assert payload["state_hint"] == "COMPETITION_CANDIDATE_MATERIALIZED"
    assert payload["metrics"]["task_id"] == 1
    assert payload["metrics"]["has_score"] is False
