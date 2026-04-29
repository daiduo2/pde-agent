import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_diagnose_hermes_result_routes_timeout_to_runtime_tuning(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_result_diagnosis import diagnose_hermes_result

    artifact_path = _write_json(
        tmp_path / "timeout_result.json",
        {
            "state_hint": "LOCAL_EXPERIMENT_RESULT",
            "result_status": "failed",
            "failure_reason": "unet_inference_timeout",
            "inference_time_seconds": 125.0,
            "source": {"operator_family": "UNet"},
        },
    )

    diagnosis = diagnose_hermes_result(
        artifact_path=artifact_path,
        output_path=tmp_path / "diagnosis.json",
        max_inference_seconds=120.0,
        min_score_delta=1.0,
    )

    assert diagnosis["state_hint"] == "HERMES_RESULT_DIAGNOSED"
    assert diagnosis["primary_bottleneck"] == "inference_timeout"
    assert diagnosis["recommended_action_family"] == "runtime_tuning"
    assert diagnosis["operator_family"] == "UNet"
    assert diagnosis["confidence"] >= 0.8
    assert "inference_over_budget" in diagnosis["reason_codes"]
    assert Path(diagnosis["diagnosis_path"]).exists()


def test_diagnose_hermes_result_routes_low_score_to_operator_strategy(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_result_diagnosis import diagnose_hermes_result

    artifact_path = _write_json(
        tmp_path / "low_score.json",
        {
            "state_hint": "LOCAL_EXPERIMENT_RESULT",
            "result_status": "completed",
            "score": 70.0,
            "baseline_score": 72.0,
            "score_delta": -2.0,
            "inference_time_seconds": 90.0,
            "source": {"operator_family": "FNO"},
        },
    )

    diagnosis = diagnose_hermes_result(
        artifact_path=artifact_path,
        output_path=tmp_path / "diagnosis.json",
        max_inference_seconds=120.0,
        min_score_delta=1.0,
    )

    assert diagnosis["primary_bottleneck"] == "low_accuracy"
    assert diagnosis["recommended_action_family"] == "operator_strategy"
    assert diagnosis["operator_family"] == "FNO"
    assert "score_delta_below_threshold" in diagnosis["reason_codes"]


def test_plan_hermes_next_action_maps_diagnosis_to_commands(tmp_path: Path) -> None:
    from pde_agent.core.hermes_next_action_policy import plan_hermes_next_action

    diagnosis_path = _write_json(
        tmp_path / "diagnosis.json",
        {
            "state_hint": "HERMES_RESULT_DIAGNOSED",
            "primary_bottleneck": "inference_timeout",
            "recommended_action_family": "runtime_tuning",
            "operator_family": "UNet",
            "source_artifact_path": "timeout_result.json",
        },
    )

    plan = plan_hermes_next_action(
        diagnosis_path=diagnosis_path,
        output_path=tmp_path / "plan.json",
    )

    assert plan["state_hint"] == "HERMES_NEXT_ACTION_PLANNED"
    assert plan["action"] == "run_operator_runtime_tuning"
    assert plan["command"] == "run-hermes-operator-tuning-loop"
    assert plan["arguments"]["operator_family"] == "UNet"
    assert plan["requires_human_review"] is False


def test_run_hermes_convergence_loop_replans_from_results(tmp_path: Path) -> None:
    from pde_agent.core.hermes_convergence_loop import run_hermes_convergence_loop

    initial_artifact = _write_json(
        tmp_path / "initial_result.json",
        {
            "state_hint": "LOCAL_EXPERIMENT_RESULT",
            "result_status": "failed",
            "failure_reason": "unet_inference_timeout",
            "inference_time_seconds": 130.0,
            "source": {"operator_family": "UNet"},
        },
    )

    def fake_executor(plan: dict[str, Any], workspace_dir: Path) -> Path | None:
        assert plan["command"] == "run-hermes-operator-tuning-loop"
        return _write_json(
            workspace_dir / "operator_tuning.json",
            {
                "state_hint": "HERMES_OPERATOR_TUNING_LOOP_RAN",
                "operator_family": "UNet",
                "selected_config": {
                    "inference_batch_size": 8,
                    "torch_num_threads": 1,
                },
                "best_projected_full_inference_seconds": 99.0,
                "decision": {"action": "accept_execution_config"},
                "next_command": "predict-unet-competition",
            },
        )

    report = run_hermes_convergence_loop(
        initial_artifact_path=initial_artifact,
        output_dir=tmp_path / "loop",
        max_steps=3,
        action_executor=fake_executor,
        max_inference_seconds=120.0,
        min_score_delta=1.0,
        force=False,
    )

    assert report["state_hint"] == "HERMES_CONVERGENCE_LOOP_RAN"
    assert report["iteration_count"] == 2
    assert report["iterations"][0]["planned_command"] == (
        "run-hermes-operator-tuning-loop"
    )
    assert report["iterations"][1]["planned_command"] == "predict-unet-competition"
    assert report["final_decision"]["action"] == "ready_for_submission_prediction"
    assert report["next_command"] == "predict-unet-competition"


def test_hermes_convergence_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_convergence_loop as command_module
    from typer.testing import CliRunner

    artifact = _write_json(
        tmp_path / "artifact.json",
        {"state_hint": "LOCAL_EXPERIMENT_RESULT", "result_status": "completed"},
    )

    def fake_loop(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "report.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_CONVERGENCE_LOOP_RAN",
            "loop_report_path": str(report_path),
            "iteration_count": 1,
            "final_decision": {
                "action": "ready_for_submission_prediction",
                "command": "predict-unet-competition",
            },
            "next_command": "predict-unet-competition",
        }

    monkeypatch.setattr(command_module, "run_hermes_convergence_loop", fake_loop)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-convergence-loop",
            "--initial-artifact-path",
            str(artifact),
            "--output-dir",
            str(tmp_path / "loop"),
            "--max-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_CONVERGENCE_LOOP_RAN"
    assert payload["metrics"]["iteration_count"] == 1
    assert payload["next_hint"] == "predict-unet-competition"
