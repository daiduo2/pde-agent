import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _write_test_hdf5(path: Path) -> Path:
    h5py = __import__("h5py")
    data = np.arange(3 * 10 * 256, dtype=np.float32).reshape(3, 10, 256)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


def test_build_competition_submission_writes_task_files(tmp_path: Path) -> None:
    from pde_agent.core.competition_submission import build_competition_submission

    test_path = _write_test_hdf5(tmp_path / "task1_test.hdf5")

    report = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_out",
        submission_id="team_hermes",
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        method="persistence",
        force=True,
    )

    assert report["state_hint"] == "COMPETITION_SUBMISSION_READY"
    submission_dir = Path(report["artifacts"]["submission_dir"])
    pred_path = submission_dir / "task1_pred.hdf5"
    time_path = submission_dir / "task1_time.csv"
    log_path = submission_dir / "task1_logs.log"
    submission_json = submission_dir / "submission.json"
    code_dir = submission_dir / "code"
    assert pred_path.exists()
    assert time_path.exists()
    assert log_path.exists()
    assert submission_json.exists()
    assert (code_dir / "model.py").exists()
    assert (code_dir / "train.py").exists()

    h5py = __import__("h5py")
    with h5py.File(test_path, "r") as handle:
        source = handle["tensor"][:]
    with h5py.File(pred_path, "r") as handle:
        prediction = handle["tensor"][:]
    assert prediction.shape == (3, 200, 256)
    np.testing.assert_allclose(prediction[:, :10, :], source, atol=1e-6)
    expected_future = np.repeat(source[:, -1:, :], repeats=190, axis=1)
    np.testing.assert_allclose(prediction[:, 10:, :], expected_future, atol=1e-6)

    rows = list(csv.DictReader(time_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert set(rows[0]) == {"train_time", "inference_time"}
    assert float(rows[0]["train_time"]) == 0.0
    assert float(rows[0]["inference_time"]) >= 0.0
    metadata = json.loads(submission_json.read_text(encoding="utf-8"))
    assert metadata == {
        "submission_id": "team_hermes",
        "problem_id": "PDE_Burgers",
        "code_path": "code",
    }
    assert "Experiment Tracking" in log_path.read_text(encoding="utf-8")

    zip_path = Path(report["artifacts"]["submission_zip_path"])
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "submission/task1_pred.hdf5" in names
    assert "submission/task1_time.csv" in names
    assert "submission/task1_logs.log" in names
    assert "submission/submission.json" in names
    assert "submission/code/model.py" in names


def test_build_competition_submission_reduces_raw_space_to_256(tmp_path: Path) -> None:
    from pde_agent.core.competition_submission import build_competition_submission

    h5py = __import__("h5py")
    data = np.ones((2, 10, 1024), dtype=np.float32)
    test_path = tmp_path / "task2_test.hdf5"
    with h5py.File(test_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    report = build_competition_submission(
        task_id=2,
        test_path=test_path,
        output_dir=tmp_path / "submission_out",
        submission_id="team_hermes",
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        method="persistence",
        force=True,
    )

    with h5py.File(report["artifacts"]["prediction_path"], "r") as handle:
        prediction = handle["tensor"][:]
    assert prediction.shape == (2, 200, 256)


def test_build_competition_submission_wraps_external_future_prediction(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission

    h5py = __import__("h5py")
    context = np.arange(2 * 10 * 256, dtype=np.float32).reshape(2, 10, 256)
    test_path = tmp_path / "task1_test.hdf5"
    with h5py.File(test_path, "w") as handle:
        handle.create_dataset("tensor", data=context)
    future = np.full((2, 190, 256), 7.0, dtype=np.float32)
    source_prediction_path = tmp_path / "candidate_prediction.npz"
    np.savez(source_prediction_path, prediction=future)

    report = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_out",
        submission_id="team_hermes",
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        method="external_prediction",
        source_prediction_path=source_prediction_path,
        source_prediction_key="prediction",
        train_time=12.5,
        force=True,
    )

    with h5py.File(report["artifacts"]["prediction_path"], "r") as handle:
        prediction = handle["tensor"][:]
    assert prediction.shape == (2, 200, 256)
    np.testing.assert_allclose(prediction[:, :10, :], context, atol=1e-6)
    np.testing.assert_allclose(prediction[:, 10:, :], future, atol=1e-6)
    rows = list(
        csv.DictReader(Path(report["artifacts"]["time_csv_path"]).open(encoding="utf-8"))
    )
    assert float(rows[0]["train_time"]) == pytest.approx(12.5)
    log_text = Path(report["artifacts"]["log_path"]).read_text(encoding="utf-8")
    assert "external_prediction" in log_text
    assert str(source_prediction_path) in log_text


def test_build_competition_submission_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    test_path = _write_test_hdf5(tmp_path / "task1_test.hdf5")
    result = CliRunner().invoke(
        app,
        [
            "build-competition-submission",
            "--task-id",
            "1",
            "--test-path",
            str(test_path),
            "--output-dir",
            str(tmp_path / "submission_out"),
            "--submission-id",
            "team_hermes",
            "--dataset-key",
            "tensor",
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "build-competition-submission"
    assert payload["state_hint"] == "COMPETITION_SUBMISSION_READY"
    assert payload["metrics"]["task_id"] == 1
    assert payload["next_hint"] == "submit-zip-to-competition"


def test_build_competition_submission_cli_accepts_external_prediction(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    test_path = _write_test_hdf5(tmp_path / "task1_test.hdf5")
    future = np.ones((3, 190, 256), dtype=np.float32)
    source_prediction_path = tmp_path / "candidate_prediction.npz"
    np.savez(source_prediction_path, prediction=future)

    result = CliRunner().invoke(
        app,
        [
            "build-competition-submission",
            "--task-id",
            "1",
            "--test-path",
            str(test_path),
            "--output-dir",
            str(tmp_path / "submission_out"),
            "--submission-id",
            "team_hermes",
            "--method",
            "external_prediction",
            "--source-prediction-path",
            str(source_prediction_path),
            "--train-time",
            "12.5",
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metrics"]["method"] == "external_prediction"
    assert payload["metrics"]["train_time"] == pytest.approx(12.5)
