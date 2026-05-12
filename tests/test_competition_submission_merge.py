import json
import zipfile
from pathlib import Path

import numpy as np


def _write_hdf5(path: Path, data: np.ndarray) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


def test_merge_competition_submissions_builds_dual_task_package(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.competition_submission_merge import (
        merge_competition_submissions,
    )
    from pde_agent.core.submission_judge import judge_submission

    task1_data = np.ones((2, 200, 256), dtype=np.float32)
    task2_data = np.full((2, 200, 256), 2.0, dtype=np.float32)
    task1_path = _write_hdf5(tmp_path / "task1_val.hdf5", task1_data)
    task2_path = _write_hdf5(tmp_path / "task2_val.h5", task2_data)
    task1 = build_competition_submission(
        task_id=1,
        test_path=task1_path,
        output_dir=tmp_path / "task1_package",
        submission_id="old_task1",
    )
    task2 = build_competition_submission(
        task_id=2,
        test_path=task2_path,
        output_dir=tmp_path / "task2_package",
        submission_id="old_task2",
    )

    report = merge_competition_submissions(
        output_dir=tmp_path / "merged",
        submission_id="team_default_combined",
        task_submission_dirs=[
            Path(task1["artifacts"]["submission_dir"]),
            Path(task2["artifacts"]["submission_dir"]),
        ],
        force=False,
    )

    submission_dir = Path(report["artifacts"]["submission_dir"])
    assert report["state_hint"] == "COMPETITION_SUBMISSIONS_MERGED"
    assert report["task_ids"] == [1, 2]
    assert (submission_dir / "task1_pred.hdf5").exists()
    assert (submission_dir / "task1_time.csv").exists()
    assert (submission_dir / "task1_logs.log").exists()
    assert (submission_dir / "task2_pred.hdf5").exists()
    assert (submission_dir / "task2_time.csv").exists()
    assert (submission_dir / "task2_logs.log").exists()
    assert json.loads((submission_dir / "submission.json").read_text()) == {
        "code_path": "code",
        "problem_id": "PDE_Burgers",
        "submission_id": "team_default_combined",
    }
    assert (submission_dir / "code" / "train.py").exists()
    assert (submission_dir / "code" / "model.py").exists()

    zip_path = Path(report["artifacts"]["submission_zip_path"])
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "submission/task1_pred.hdf5" in names
    assert "submission/task2_pred.hdf5" in names

    judge = judge_submission(
        submission_dir=submission_dir,
        output_path=None,
        task1_test_path=task1_path,
        task1_target_path=task1_path,
        task2_test_path=task2_path,
        task2_target_path=task2_path,
    )
    assert judge["rules_passed"] is True
    assert set(judge["tasks"]) == {"1", "2"}
    assert judge["total_score_proxy"] > 0.0


def test_merge_competition_submissions_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from typer.testing import CliRunner

    from pde_agent.cli import app

    task1_data = np.ones((1, 200, 256), dtype=np.float32)
    task2_data = np.full((1, 200, 256), 2.0, dtype=np.float32)
    task1_path = _write_hdf5(tmp_path / "task1_val.hdf5", task1_data)
    task2_path = _write_hdf5(tmp_path / "task2_val.h5", task2_data)
    task1 = build_competition_submission(
        task_id=1,
        test_path=task1_path,
        output_dir=tmp_path / "task1_package",
        submission_id="old_task1",
    )
    task2 = build_competition_submission(
        task_id=2,
        test_path=task2_path,
        output_dir=tmp_path / "task2_package",
        submission_id="old_task2",
    )

    result = CliRunner().invoke(
        app,
        [
            "merge-competition-submissions",
            "--output-dir",
            str(tmp_path / "merged"),
            "--submission-id",
            "team_default_combined",
            "--task-submission-dir",
            str(task1["artifacts"]["submission_dir"]),
            "--task-submission-dir",
            str(task2["artifacts"]["submission_dir"]),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "merge-competition-submissions"
    assert payload["state_hint"] == "COMPETITION_SUBMISSIONS_MERGED"
    assert payload["metrics"]["task_ids"] == [1, 2]
    assert Path(payload["artifacts"]["submission_zip_path"]).exists()
