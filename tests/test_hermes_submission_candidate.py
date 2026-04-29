import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_hermes_submission_candidate_uses_selected_unet_config(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_submission_candidate import (
        run_hermes_submission_candidate,
    )

    tuning_path = _write_json(
        tmp_path / "operator_tuning.json",
        {
            "state_hint": "HERMES_OPERATOR_TUNING_LOOP_RAN",
            "operator_family": "UNet",
            "selected_config": {
                "inference_batch_size": 8,
                "torch_num_threads": 1,
            },
            "best_projected_full_inference_seconds": 99.0,
        },
    )
    calls: dict[str, Any] = {}

    def fake_predictor(**kwargs: Any) -> dict[str, Any]:
        calls["predictor"] = kwargs
        report_path = Path(kwargs["output_dir"]) / "candidate_execution_report.json"
        prediction_path = Path(kwargs["output_dir"]) / "candidate_prediction.npz"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        prediction_path.write_text("", encoding="utf-8")
        return {
            "state_hint": "UNET_COMPETITION_PREDICTION_READY",
            "candidate": {"task_id": kwargs["task_id"]},
            "inputs": {"sample_count": 2},
            "artifacts": {
                "candidate_execution_report_path": str(report_path),
                "candidate_prediction_path": str(prediction_path),
            },
            "inference": {
                "elapsed_seconds": 98.0,
                "inference_batch_size": kwargs["inference_batch_size"],
                "torch_num_threads": kwargs["torch_num_threads"],
            },
        }

    def fake_materializer(**kwargs: Any) -> dict[str, Any]:
        calls["materializer"] = kwargs
        submission_dir = Path(kwargs["output_dir"]) / "submission_package" / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        (submission_dir / "task1_logs.log").write_text(
            "Experiment Tracking\n",
            encoding="utf-8",
        )
        return {
            "state_hint": "COMPETITION_CANDIDATE_MATERIALIZED",
            "artifacts": {
                "submission_dir": str(submission_dir),
                "submission_zip_path": str(submission_dir.parent / "submission.zip"),
                "prediction_path": str(submission_dir / "task1_pred.hdf5"),
            },
            "submission": {
                "metrics": {
                    "prediction_shape": [2, 200, 256],
                    "inference_time": 98.0,
                }
            },
            "score": None,
        }

    def fake_chronicle(**kwargs: Any) -> dict[str, Any]:
        calls["chronicle"] = kwargs
        chronicle_path = Path(kwargs["output_path"])
        chronicle_path.parent.mkdir(parents=True, exist_ok=True)
        chronicle_path.write_text("Hermes Research Chronicle\n", encoding="utf-8")
        return {
            "state_hint": "RESEARCH_CHRONICLE_BUILT",
            "artifacts": {"chronicle_path": str(chronicle_path)},
        }

    def fake_auditor(**kwargs: Any) -> dict[str, Any]:
        calls["auditor"] = kwargs
        audit_path = Path(kwargs["output_path"])
        audit_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "COMPETITION_SUBMISSION_AUDITED",
            "audit_path": str(audit_path),
            "decision": "passed",
            "issue_count": 0,
        }

    report = run_hermes_submission_candidate(
        tuning_report_path=tuning_path,
        output_dir=tmp_path / "candidate",
        task_id=1,
        test_path="task1_test.hdf5",
        checkpoint_path="burgers_Unet.tar",
        submission_id="team_hermes",
        target_path=None,
        force=False,
        predictor=fake_predictor,
        materializer=fake_materializer,
        chronicle_builder=fake_chronicle,
        auditor=fake_auditor,
    )

    assert calls["predictor"]["inference_batch_size"] == 8
    assert calls["predictor"]["torch_num_threads"] == 1
    assert calls["materializer"]["candidate_report_path"] == Path(
        report["prediction"]["artifacts"]["candidate_execution_report_path"]
    )
    assert calls["chronicle"]["artifact_paths"][0] == str(tuning_path)
    assert calls["auditor"]["submission_dir"] == report["materialized"]["artifacts"][
        "submission_dir"
    ]
    assert report["state_hint"] == "HERMES_SUBMISSION_CANDIDATE_READY"
    assert report["audit"]["decision"] == "passed"
    assert report["selected_config"]["inference_batch_size"] == 8
    assert report["next_command"] == "submit-if-passed-or-review"


def test_convergence_loop_executor_runs_submission_candidate_from_terminal_plan(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_convergence_loop import run_hermes_convergence_loop
    from pde_agent.core.hermes_convergence_executor import (
        hermes_controlled_action_executor,
    )

    tuning_path = _write_json(
        tmp_path / "operator_tuning.json",
        {
            "state_hint": "HERMES_OPERATOR_TUNING_LOOP_RAN",
            "operator_family": "UNet",
            "selected_config": {
                "inference_batch_size": 8,
                "torch_num_threads": 1,
            },
            "best_projected_full_inference_seconds": 99.0,
            "decision": {"action": "accept_execution_config"},
        },
    )
    calls: dict[str, Any] = {}

    def fake_submission_candidate(**kwargs: Any) -> dict[str, Any]:
        calls["candidate"] = kwargs
        report_path = Path(kwargs["output_dir"]) / "hermes_submission_candidate.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_SUBMISSION_CANDIDATE_READY",
            "candidate_report_path": str(report_path),
            "audit": {"decision": "passed"},
            "next_command": "submit-if-passed-or-review",
        }

    executor = hermes_controlled_action_executor(
        task_id=1,
        test_path="task1_test.hdf5",
        checkpoint_path="burgers_Unet.tar",
        submission_id="team_hermes",
        submission_candidate_runner=fake_submission_candidate,
    )

    report = run_hermes_convergence_loop(
        initial_artifact_path=tuning_path,
        output_dir=tmp_path / "loop",
        max_steps=2,
        action_executor=executor,
        force=False,
    )

    assert calls["candidate"]["tuning_report_path"] == tuning_path
    assert report["iteration_count"] == 1
    assert report["iterations"][0]["executed_artifact_path"].endswith(
        "hermes_submission_candidate.json"
    )
    assert report["next_command"] == "submit-if-passed-or-review"


def test_hermes_submission_candidate_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_submission_candidate as command_module
    from typer.testing import CliRunner

    tuning_path = _write_json(
        tmp_path / "operator_tuning.json",
        {"state_hint": "HERMES_OPERATOR_TUNING_LOOP_RAN"},
    )

    def fake_runner(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "candidate.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_SUBMISSION_CANDIDATE_READY",
            "candidate_report_path": str(report_path),
            "task_id": 1,
            "selected_config": {
                "inference_batch_size": 8,
                "torch_num_threads": 1,
            },
            "materialized": {
                "artifacts": {
                    "submission_dir": str(tmp_path / "submission"),
                    "submission_zip_path": str(tmp_path / "submission.zip"),
                }
            },
            "chronicle": {
                "artifacts": {"chronicle_path": str(tmp_path / "chronicle.md")}
            },
            "audit": {
                "audit_path": str(tmp_path / "audit.json"),
                "decision": "passed",
                "issue_count": 0,
            },
            "next_command": "submit-if-passed-or-review",
        }

    monkeypatch.setattr(command_module, "run_hermes_submission_candidate", fake_runner)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-submission-candidate",
            "--tuning-report-path",
            str(tuning_path),
            "--output-dir",
            str(tmp_path / "candidate"),
            "--task-id",
            "1",
            "--test-path",
            "task1_test.hdf5",
            "--checkpoint-path",
            "burgers_Unet.tar",
            "--submission-id",
            "team_hermes",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_SUBMISSION_CANDIDATE_READY"
    assert payload["metrics"]["audit_decision"] == "passed"
    assert payload["metrics"]["inference_batch_size"] == 8
    assert payload["next_hint"] == "submit-if-passed-or-review"
