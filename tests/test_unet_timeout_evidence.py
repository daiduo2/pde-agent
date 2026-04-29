import json
from pathlib import Path


def test_record_unet_timeout_evidence_writes_failed_local_result_with_optimization_plan(
    tmp_path: Path,
) -> None:
    from pde_agent.core.unet_timeout_evidence import record_unet_timeout_evidence

    result = record_unet_timeout_evidence(
        output_path=tmp_path / "unet_timeout_result.json",
        elapsed_seconds=305.0,
        timeout_seconds=120.0,
        attempted_command="predict-unet-competition",
        checkpoint_path="data/pdebench/checkpoints/burgers_Unet.tar",
    )

    assert result["state_hint"] == "LOCAL_EXPERIMENT_RESULT"
    assert result["result_status"] == "failed"
    assert result["failure_reason"] == "unet_inference_timeout"
    assert result["inference_time_seconds"] == 305.0
    assert result["source"]["optimization_plan"]["strategy_id"] == (
        "unet_timeout_optimization_v0"
    )
    assert "reduce rollout sample batch size" in result["source"]["optimization_plan"][
        "candidate_actions"
    ]
    assert Path(result["result_path"]).exists()


def test_record_unet_timeout_evidence_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    output_path = tmp_path / "unet_timeout_result.json"
    result = CliRunner().invoke(
        app,
        [
            "record-unet-timeout-evidence",
            "--output-path",
            str(output_path),
            "--elapsed-seconds",
            "305",
            "--timeout-seconds",
            "120",
            "--attempted-command",
            "predict-unet-competition",
            "--checkpoint-path",
            "data/pdebench/checkpoints/burgers_Unet.tar",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "LOCAL_EXPERIMENT_RESULT"
    assert payload["metrics"]["failure_reason"] == "unet_inference_timeout"
    assert payload["next_hint"] == "record-local-strategy-result"
