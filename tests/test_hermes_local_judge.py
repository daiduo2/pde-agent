import csv
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


def _write_time_csv(path: Path, *, train_time: float, inference_time: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow(
            {
                "train_time": f"{train_time:.6f}",
                "inference_time": f"{inference_time:.6f}",
            }
        )


def test_hermes_local_judge_scores_perfect_task1_submission_at_150(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_local_judge import judge_hermes_submission

    context = np.ones((2, 10, 256), dtype=np.float32)
    future = np.ones((2, 190, 256), dtype=np.float32)
    target = np.concatenate([context, future], axis=1)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    submission = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "out",
        submission_id="team_hermes",
        method="external_prediction",
        source_prediction_path=_write_npz(tmp_path / "future.npz", future),
        train_time=0.0,
        inference_time=0.0,
        force=True,
    )

    report = judge_hermes_submission(
        submission_dir=submission["artifacts"]["submission_dir"],
        output_path=tmp_path / "judge.json",
        task1_test_path=test_path,
        task1_target_path=target_path,
    )

    task1 = report["tasks"]["1"]
    assert report["state_hint"] == "HERMES_LOCAL_JUDGE_SCORED"
    assert report["rules_passed"] is True
    assert task1["score"]["task_total_score"] == pytest.approx(150.0)
    assert task1["score"]["prediction_accuracy_score"] == pytest.approx(75.0)
    assert task1["score"]["train_time_score"] == pytest.approx(35.0)
    assert task1["score"]["inference_time_score"] == pytest.approx(40.0)
    assert report["total_score_proxy"] == pytest.approx(150.0)
    assert report["decision"] == "submit_candidate"
    assert Path(report["judge_report_path"]).exists()


def test_hermes_local_judge_zeroes_task2_when_inference_exceeds_two_minutes(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_local_judge import judge_hermes_submission

    context = np.ones((1, 10, 256), dtype=np.float32)
    future = np.ones((1, 190, 256), dtype=np.float32)
    target = np.concatenate([context, future], axis=1)
    test_path = _write_hdf5(tmp_path / "task2_test.h5", context)
    target_path = _write_hdf5(tmp_path / "task2_val.h5", target)
    submission = build_competition_submission(
        task_id=2,
        test_path=test_path,
        output_dir=tmp_path / "out",
        submission_id="team_hermes",
        method="external_prediction",
        source_prediction_path=_write_npz(tmp_path / "future.npz", future),
        train_time=1.0,
        inference_time=121.0,
        force=True,
    )

    report = judge_hermes_submission(
        submission_dir=submission["artifacts"]["submission_dir"],
        output_path=None,
        task2_test_path=test_path,
        task2_target_path=target_path,
    )

    task2 = report["tasks"]["2"]
    assert task2["score"]["timeout_zeroed"] is True
    assert task2["score"]["task_total_score"] == 0.0
    assert report["decision"] == "iterate_before_submit"


def test_hermes_local_judge_fails_rule_gate_when_trace_is_missing(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_local_judge import judge_hermes_submission

    context = np.ones((1, 10, 256), dtype=np.float32)
    future = np.ones((1, 190, 256), dtype=np.float32)
    target = np.concatenate([context, future], axis=1)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    submission = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "out",
        submission_id="team_hermes",
        method="external_prediction",
        source_prediction_path=_write_npz(tmp_path / "future.npz", future),
        force=True,
    )
    log_path = Path(submission["artifacts"]["log_path"])
    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace("Trace:", "Notes:"),
        encoding="utf-8",
    )

    report = judge_hermes_submission(
        submission_dir=submission["artifacts"]["submission_dir"],
        output_path=None,
        task1_test_path=test_path,
        task1_target_path=target_path,
    )

    task1 = report["tasks"]["1"]
    assert report["rules_passed"] is False
    assert "missing_trace" in task1["log_review"]["issue_codes"]
    assert task1["score"]["task_total_score"] == 0.0
    assert report["decision"] == "hold_rule_gate_failed"


def test_hermes_local_judge_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.judge_hermes_submission as command_module
    from typer.testing import CliRunner

    def fake_judge(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "judge.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": 123.5,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": 123.5}}},
        }

    monkeypatch.setattr(command_module, "judge_hermes_submission", fake_judge)

    result = CliRunner().invoke(
        app,
        [
            "judge-hermes-submission",
            "--submission-dir",
            str(tmp_path / "submission"),
            "--output-path",
            str(tmp_path / "judge.json"),
            "--task1-test-path",
            "task1_test.hdf5",
            "--task1-target-path",
            "task1_val.hdf5",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_LOCAL_JUDGE_SCORED"
    assert payload["metrics"]["total_score_proxy"] == 123.5
    assert payload["next_hint"] == "submit_candidate"


def _write_npz(path: Path, prediction: np.ndarray) -> Path:
    np.savez(path, prediction=prediction)
    return path
