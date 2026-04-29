from pathlib import Path
from typing import Any


def test_hermes_three_stage_policy_runs_hybrid_accuracy_and_score(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_three_stage_policy import run_hermes_three_stage_policy

    calls: list[dict[str, Any]] = []
    submission_dirs = {
        "hybrid": tmp_path / "hybrid_best",
        "accuracy": tmp_path / "accuracy_best",
        "score": tmp_path / "score_best",
    }
    for path in submission_dirs.values():
        path.mkdir()

    def fake_heartbeat(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        stage = Path(kwargs["output_dir"]).name
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        if stage == "01_hybrid_gate":
            return {
                "heartbeat_report_path": str(report_path),
                "best_candidate": {
                    "candidate_id": "fast_baseline_persistence",
                    "score": 85.0,
                    "submission_dir": str(submission_dirs["hybrid"]),
                },
                "best_score": 85.0,
                "best_objective": "accuracy",
                "best_objective_score": 10.0,
                "stop_reason": "max_batches_exhausted",
                "decision": "continue_next_heartbeat",
            }
        return {
            "heartbeat_report_path": str(report_path),
            "best_candidate": {
                "candidate_id": "postprocess_spectral_smooth_w0.125",
                "score": 86.0,
                "submission_dir": str(submission_dirs["accuracy"]),
            },
            "best_score": 86.0,
            "best_objective": "accuracy",
            "best_objective_score": 11.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
        }

    def fake_batch(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "loop_report_path": str(report_path),
            "best_candidate": {
                "candidate_id": "postprocess_spectral_smooth_w0.125",
                "score": 86.0,
                "submission_dir": str(submission_dirs["score"]),
            },
            "round_count": 3,
            "stop_reason": "patience_exhausted",
            "decision": "iterate_more",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_three_stage_policy(
        output_dir=tmp_path / "policy",
        initial_submission_dir=initial_dir,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        unet_checkpoint_path="checkpoint.pt",
        heartbeat_runner=fake_heartbeat,
        batch_runner=fake_batch,
        force=False,
    )

    assert [Path(call["output_dir"]).name for call in calls] == [
        "01_hybrid_gate",
        "02_accuracy_refine",
        "03_score_arbitration",
    ]
    assert calls[0]["candidate_generator"] == "hybrid_gate"
    assert calls[0]["learning_objective"] == "auto"
    assert calls[1]["candidate_generator"] == "postprocess"
    assert calls[1]["learning_objective"] == "accuracy"
    assert calls[1]["initial_submission_dir"] == submission_dirs["hybrid"]
    assert calls[2]["candidate_generator"] == "none"
    assert calls[2]["learning_objective"] == "score"
    assert calls[2]["initial_submission_dir"] == submission_dirs["accuracy"]
    assert report["best_candidate"]["submission_dir"] == str(submission_dirs["score"])
    assert report["stage_count"] == 3
    assert Path(report["policy_report_path"]).exists()


def test_hermes_three_stage_policy_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_three_stage_policy as command_module
    from typer.testing import CliRunner

    def fake_policy(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "policy.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_THREE_STAGE_POLICY_RAN",
            "policy_report_path": str(report_path),
            "stage_count": 3,
            "best_candidate": {"candidate_id": "best", "score": 86.0},
            "best_score": 86.0,
            "decision": "review_policy_result",
            "next_command": "run-hermes-three-stage-policy",
        }

    monkeypatch.setattr(command_module, "run_hermes_three_stage_policy", fake_policy)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-three-stage-policy",
            "--output-dir",
            str(tmp_path / "policy"),
            "--initial-submission-dir",
            str(tmp_path / "initial"),
            "--task1-test-path",
            "task1_test.hdf5",
            "--task1-target-path",
            "task1_val.hdf5",
            "--unet-checkpoint-path",
            "checkpoint.pt",
        ],
    )

    assert result.exit_code == 0
    payload = __import__("json").loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_THREE_STAGE_POLICY_RAN"
    assert payload["metrics"]["best_score"] == 86.0
