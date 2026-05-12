import json
from pathlib import Path

import numpy as np


def _write_test_hdf5(path: Path) -> Path:
    h5py = __import__("h5py")
    data = np.arange(2 * 10 * 256, dtype=np.float32).reshape(2, 10, 256)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


def test_audit_competition_submission_passes_valid_package(tmp_path: Path) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.competition_submission_audit import audit_competition_submission

    test_path = _write_test_hdf5(tmp_path / "task1_test.hdf5")
    submission = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_out",
        submission_id="team_default",
        force=True,
    )

    audit = audit_competition_submission(
        task_id=1,
        submission_dir=Path(submission["artifacts"]["submission_dir"]),
        test_path=test_path,
        output_path=tmp_path / "audit.json",
        dataset_key="tensor",
    )

    assert audit["state_hint"] == "COMPETITION_SUBMISSION_AUDITED"
    assert audit["decision"] == "passed"
    assert audit["issue_count"] == 0
    assert Path(audit["audit_path"]).exists()


def test_audit_competition_submission_reports_missing_log(tmp_path: Path) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.competition_submission_audit import audit_competition_submission

    test_path = _write_test_hdf5(tmp_path / "task1_test.hdf5")
    submission = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_out",
        submission_id="team_default",
        force=True,
    )
    Path(submission["artifacts"]["log_path"]).unlink()

    audit = audit_competition_submission(
        task_id=1,
        submission_dir=Path(submission["artifacts"]["submission_dir"]),
        test_path=test_path,
        output_path=None,
        dataset_key="tensor",
    )

    assert audit["decision"] == "failed"
    assert "missing_task_log" in audit["issue_codes"]


def test_audit_competition_submission_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from pde_agent.core.competition_submission import build_competition_submission
    from typer.testing import CliRunner

    test_path = _write_test_hdf5(tmp_path / "task1_test.hdf5")
    submission = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_out",
        submission_id="team_default",
        force=True,
    )
    output_path = tmp_path / "audit.json"

    result = CliRunner().invoke(
        app,
        [
            "audit-competition-submission",
            "--task-id",
            "1",
            "--submission-dir",
            submission["artifacts"]["submission_dir"],
            "--test-path",
            str(test_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "audit-competition-submission"
    assert payload["state_hint"] == "COMPETITION_SUBMISSION_AUDITED"
    assert payload["metrics"]["decision"] == "passed"
