import json
from pathlib import Path
from typing import Any


def test_run_hermes_model_iteration_loop_routes_validation_diagnosis_to_action(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_model_iteration_loop import (
        run_hermes_model_iteration_loop,
    )

    calls: dict[str, Any] = {}

    def fake_scorer(**kwargs: Any) -> dict[str, Any]:
        calls["scorer"] = kwargs
        report_path = Path(kwargs["output_dir"]) / "hermes_validation_scoring.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_VALIDATION_SCORED",
            "validation_report_path": str(report_path),
            "operator_family": "UNet",
            "task_id": 1,
            "score": {"segment_score": 51.0},
            "prediction": {"inference": {"elapsed_seconds": 44.0}},
            "artifacts": {"validation_report_path": str(report_path)},
        }

    def fake_diagnoser(**kwargs: Any) -> dict[str, Any]:
        calls["diagnoser"] = kwargs
        diagnosis_path = Path(kwargs["output_path"])
        diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
        diagnosis_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_MODEL_QUALITY_DIAGNOSED",
            "diagnosis_path": str(diagnosis_path),
            "operator_family": "UNet",
            "primary_bottleneck": "late_window_drift",
            "recommended_strategy_family": "rollout_stabilization",
            "confidence": 0.86,
            "reason_codes": ["late_score_below_threshold"],
            "next_command": "run-hermes-model-iteration-loop",
        }

    report = run_hermes_model_iteration_loop(
        output_dir=tmp_path / "iteration",
        operator_family="UNet",
        task_id=1,
        validation_path="task1_val.hdf5",
        checkpoint_path="burgers_Unet.tar",
        inference_batch_size=8,
        torch_num_threads=1,
        force=False,
        validation_scorer=fake_scorer,
        quality_diagnoser=fake_diagnoser,
    )

    assert calls["scorer"]["output_dir"] == tmp_path / "iteration" / "validation"
    assert calls["diagnoser"]["validation_report_path"] == Path(
        report["validation"]["validation_report_path"]
    )
    assert report["state_hint"] == "HERMES_MODEL_ITERATION_LOOP_RAN"
    assert report["decision"]["action"] == "schedule_model_strategy_sweep"
    assert report["decision"]["strategy_family"] == "rollout_stabilization"
    assert report["next_command"] == "run-unet-strategy-sweep"
    assert Path(report["iteration_report_path"]).exists()


def test_run_hermes_model_iteration_loop_promotes_acceptable_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_model_iteration_loop import (
        run_hermes_model_iteration_loop,
    )

    def fake_scorer(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_dir"]) / "hermes_validation_scoring.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_VALIDATION_SCORED",
            "validation_report_path": str(report_path),
            "operator_family": "UNet",
            "task_id": 1,
            "score": {"segment_score": 92.0},
            "artifacts": {"validation_report_path": str(report_path)},
        }

    def fake_diagnoser(**kwargs: Any) -> dict[str, Any]:
        diagnosis_path = Path(kwargs["output_path"])
        diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
        diagnosis_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_MODEL_QUALITY_DIAGNOSED",
            "diagnosis_path": str(diagnosis_path),
            "operator_family": "UNet",
            "primary_bottleneck": "quality_acceptable",
            "recommended_strategy_family": "promote_candidate",
            "confidence": 0.9,
            "reason_codes": ["all_quality_thresholds_met"],
            "next_command": "run-hermes-model-iteration-loop",
        }

    report = run_hermes_model_iteration_loop(
        output_dir=tmp_path / "iteration",
        operator_family="UNet",
        task_id=1,
        validation_path="task1_val.hdf5",
        checkpoint_path="burgers_Unet.tar",
        force=True,
        validation_scorer=fake_scorer,
        quality_diagnoser=fake_diagnoser,
    )

    assert report["decision"]["action"] == "promote_validated_candidate"
    assert report["next_command"] == "run-hermes-submission-candidate"


def test_hermes_model_iteration_loop_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_model_iteration_loop as command_module
    from typer.testing import CliRunner

    def fake_loop(**_: object) -> dict[str, object]:
        report_path = tmp_path / "iteration.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_MODEL_ITERATION_LOOP_RAN",
            "iteration_report_path": str(report_path),
            "operator_family": "UNet",
            "task_id": 1,
            "validation": {"score": {"segment_score": 52.0}},
            "diagnosis": {
                "primary_bottleneck": "late_window_drift",
                "recommended_strategy_family": "rollout_stabilization",
            },
            "decision": {"action": "schedule_model_strategy_sweep"},
            "next_command": "run-unet-strategy-sweep",
        }

    monkeypatch.setattr(command_module, "run_hermes_model_iteration_loop", fake_loop)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-model-iteration-loop",
            "--output-dir",
            str(tmp_path / "iteration"),
            "--operator-family",
            "UNet",
            "--task-id",
            "1",
            "--validation-path",
            "task1_val.hdf5",
            "--checkpoint-path",
            "burgers_Unet.tar",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["metrics"]["decision"] == "schedule_model_strategy_sweep"
    assert payload["next_hint"] == "run-unet-strategy-sweep"
