import json
from pathlib import Path


def _write_operator_sweep_summary(
    path: Path,
    *,
    state_hint: str = "DEEPONET_EVIDENCE_SWEEP_COMPLETED",
    forecast_steps: int = 31,
    winner_deltas: list[float] | None = None,
) -> Path:
    deltas = winner_deltas or [1.5, 0.5, 2.5]
    comparison_winners = [
        {
            "start_index": index,
            "winner_experiment_id": f"deeponet_candidate_{index:03d}",
            "winner_training_strategy": "branch_trunk_rollout",
            "winner_score": 30.0 + delta,
            "winner_score_delta": delta,
        }
        for index, delta in enumerate(deltas)
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": state_hint,
                "inputs": {"forecast_steps": forecast_steps},
                "comparison_winners": comparison_winners,
                "comparison_count": len(comparison_winners),
                "record_count": len(comparison_winners) * 2,
                "physics_residual_delta": -0.1,
                "inference_time_regression": 0.05,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_convert_operator_evidence_summary_supports_deeponet_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.operator_strategy_evidence import (
        convert_operator_evidence_summary,
    )

    sweep_path = _write_operator_sweep_summary(
        tmp_path / "deeponet_evidence_summary.json"
    )
    output_path = tmp_path / "strategy_evidence_summary.json"

    evidence = convert_operator_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=output_path,
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=31,
    )

    assert output_path.exists()
    assert evidence == json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == 1
    assert evidence["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert evidence["model_family"] == "DeepONet"
    assert evidence["source_state_hint"] == "DEEPONET_EVIDENCE_SWEEP_COMPLETED"
    assert evidence["source_evidence_summary_path"] == str(sweep_path)
    assert evidence["source_deeponet_evidence_summary_path"] == str(sweep_path)
    assert evidence["evaluation_window_count"] == 3
    assert evidence["mean_score_delta"] == 1.5
    assert evidence["long_window_score_delta"] == 1.5
    assert evidence["physics_residual_delta"] == -0.1
    assert evidence["inference_time_regression"] == 0.05
    assert evidence["failure_rate"] == 0.0
    assert evidence["next_command"] == "evaluate-strategy-promotion"


def test_convert_operator_evidence_summary_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    sweep_path = _write_operator_sweep_summary(
        tmp_path / "deeponet_evidence_summary.json"
    )
    output_path = tmp_path / "strategy_evidence_summary.json"

    result = CliRunner().invoke(
        app,
        [
            "convert-operator-evidence-summary",
            "--sweep-summary-path",
            str(sweep_path),
            "--output-path",
            str(output_path),
            "--model-family",
            "DeepONet",
            "--source-state-hint",
            "DEEPONET_EVIDENCE_SWEEP_COMPLETED",
            "--long-window-threshold",
            "31",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "convert-operator-evidence-summary"
    assert payload["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert payload["artifacts"]["strategy_evidence_summary_path"] == str(output_path)
    assert payload["artifacts"]["source_evidence_summary_path"] == str(sweep_path)
    assert payload["metrics"]["model_family"] == "DeepONet"
    assert payload["metrics"]["evaluation_window_count"] == 3
    assert payload["metrics"]["mean_score_delta"] == 1.5
    assert payload["metrics"]["failure_rate"] == 0.0
    assert payload["next_hint"] == "evaluate-strategy-promotion"
