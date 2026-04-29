import json
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_diagnose_hermes_model_quality_detects_late_rollout_drift(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_model_quality_diagnosis import (
        diagnose_hermes_model_quality,
    )

    validation_path = _write_json(
        tmp_path / "validation.json",
        {
            "state_hint": "HERMES_VALIDATION_SCORED",
            "operator_family": "UNet",
            "score": {
                "segment_score": 52.0,
                "task_prediction_score": 39.0,
                "segments": {
                    "early": {"score": 86.0, "rel_mse": 0.006},
                    "middle": {"score": 75.0, "rel_mse": 0.012},
                    "late": {"score": 18.0, "rmse": 0.46},
                },
            },
            "prediction": {"inference": {"elapsed_seconds": 40.0}},
        },
    )

    report = diagnose_hermes_model_quality(
        validation_report_path=validation_path,
        output_path=tmp_path / "diagnosis.json",
        min_segment_score=60.0,
        max_late_rmse=0.25,
        max_inference_seconds=120.0,
    )

    assert report["state_hint"] == "HERMES_MODEL_QUALITY_DIAGNOSED"
    assert report["primary_bottleneck"] == "late_window_drift"
    assert report["recommended_strategy_family"] == "rollout_stabilization"
    assert "late_score_below_threshold" in report["reason_codes"]
    assert report["next_command"] == "run-hermes-model-iteration-loop"
    assert Path(report["diagnosis_path"]).exists()


def test_diagnose_hermes_model_quality_prefers_runtime_when_over_budget(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_model_quality_diagnosis import (
        diagnose_hermes_model_quality,
    )

    validation_path = _write_json(
        tmp_path / "validation.json",
        {
            "state_hint": "HERMES_VALIDATION_SCORED",
            "operator_family": "UNet",
            "score": {
                "segment_score": 90.0,
                "segments": {
                    "early": {"score": 95.0, "rel_mse": 0.001},
                    "middle": {"score": 92.0, "rel_mse": 0.002},
                    "late": {"score": 85.0, "rmse": 0.03},
                },
            },
            "prediction": {"inference": {"elapsed_seconds": 140.0}},
        },
    )

    report = diagnose_hermes_model_quality(
        validation_report_path=validation_path,
        output_path=None,
        max_inference_seconds=120.0,
    )

    assert report["primary_bottleneck"] == "inference_over_budget"
    assert report["recommended_strategy_family"] == "runtime_tuning"


def test_diagnose_hermes_model_quality_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.diagnose_hermes_model_quality as command_module
    from typer.testing import CliRunner

    validation_path = _write_json(
        tmp_path / "validation.json",
        {"state_hint": "HERMES_VALIDATION_SCORED"},
    )

    def fake_diagnoser(**_: object) -> dict[str, object]:
        diagnosis_path = tmp_path / "diagnosis.json"
        diagnosis_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_MODEL_QUALITY_DIAGNOSED",
            "diagnosis_path": str(diagnosis_path),
            "operator_family": "UNet",
            "primary_bottleneck": "late_window_drift",
            "recommended_strategy_family": "rollout_stabilization",
            "confidence": 0.85,
            "next_command": "run-hermes-model-iteration-loop",
        }

    monkeypatch.setattr(command_module, "diagnose_hermes_model_quality", fake_diagnoser)

    result = CliRunner().invoke(
        app,
        [
            "diagnose-hermes-model-quality",
            "--validation-report-path",
            str(validation_path),
            "--output-path",
            str(tmp_path / "diagnosis.json"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["metrics"]["primary_bottleneck"] == "late_window_drift"
    assert payload["next_hint"] == "run-hermes-model-iteration-loop"
