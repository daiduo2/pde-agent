import json
from pathlib import Path
from typing import Any


def test_hermes_batch_optimization_loop_tracks_best_and_stops_on_patience(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    scores = {
        "initial": 80.0,
        "candidate_1": 91.0,
        "candidate_2": 91.2,
        "candidate_3": 96.0,
    }
    candidate_calls: list[int] = []

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any]:
        round_index = int(kwargs["round_index"])
        candidate_calls.append(round_index)
        submission_dir = tmp_path / f"candidate_{round_index}"
        submission_dir.mkdir()
        return {
            "candidate_id": f"candidate_{round_index}",
            "submission_dir": str(submission_dir),
            "strategy_id": "residual_calibration_v0",
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        name = Path(kwargs["submission_dir"]).name
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": scores[name],
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": scores[name]}}},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=3,
        patience=1,
        min_score_improvement=1.0,
        submit_score_threshold=95.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        force=False,
    )

    assert candidate_calls == [1, 2]
    assert report["state_hint"] == "HERMES_BATCH_OPTIMIZATION_LOOP_RAN"
    assert report["round_count"] == 3
    assert report["best_candidate"]["candidate_id"] == "candidate_1"
    assert report["best_candidate"]["score"] == 91.0
    assert report["stop_reason"] == "patience_exhausted"
    assert report["decision"] == "iterate_more"
    assert report["next_command"] == "run-hermes-batch-optimization-loop"
    assert Path(report["loop_report_path"]).exists()


def test_hermes_batch_optimization_loop_recommends_submit_when_threshold_met(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any]:
        submission_dir = tmp_path / "candidate"
        submission_dir.mkdir()
        return {
            "candidate_id": "candidate",
            "submission_dir": str(submission_dir),
            "strategy_id": "rollout_stabilization_v0",
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 88.0 if Path(kwargs["submission_dir"]).name == "initial" else 97.5
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=3,
        patience=2,
        min_score_improvement=0.5,
        submit_score_threshold=95.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        force=False,
    )

    assert report["best_candidate"]["score"] == 97.5
    assert report["stop_reason"] == "submit_threshold_met"
    assert report["decision"] == "submit_best_candidate"
    assert report["next_command"] == "submit-best-candidate"


def test_hermes_batch_optimization_loop_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_batch_optimization_loop as command_module
    from typer.testing import CliRunner

    def fake_loop(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "loop.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 2,
            "best_candidate": {
                "candidate_id": "candidate_1",
                "score": 96.0,
                "submission_dir": str(tmp_path / "candidate_1"),
            },
            "stop_reason": "submit_threshold_met",
            "decision": "submit_best_candidate",
            "next_command": "submit-best-candidate",
        }

    monkeypatch.setattr(command_module, "run_hermes_batch_optimization_loop", fake_loop)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-batch-optimization-loop",
            "--output-dir",
            str(tmp_path / "loop"),
            "--initial-submission-dir",
            str(tmp_path / "initial"),
            "--task1-test-path",
            "task1_test.hdf5",
            "--task1-target-path",
            "task1_val.hdf5",
            "--max-rounds",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_BATCH_OPTIMIZATION_LOOP_RAN"
    assert payload["metrics"]["best_score"] == 96.0
    assert payload["next_hint"] == "submit-best-candidate"
