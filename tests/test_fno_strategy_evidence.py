import json
from pathlib import Path


def _write_sweep_summary(
    path: Path,
    *,
    forecast_steps: int = 31,
    winner_deltas: list[float] | None = None,
) -> Path:
    deltas = winner_deltas or [2.0, 1.0, 3.0]
    comparison_winners = [
        {
            "start_index": index,
            "train_steps": 1,
            "winner_experiment_id": f"candidate_{index:03d}",
            "winner_training_strategy": "scheduled_sampling",
            "winner_score": 40.0 + delta,
            "winner_score_delta": delta,
        }
        for index, delta in enumerate(deltas)
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "FNO_EVIDENCE_SWEEP_COMPLETED",
                "inputs": {"forecast_steps": forecast_steps},
                "artifacts": {"summary_json_path": str(path)},
                "comparison_winners": comparison_winners,
                "comparison_count": len(comparison_winners),
                "record_count": len(comparison_winners) * 2,
                "strategy_statistics": {
                    "scheduled_sampling": {
                        "count": len(comparison_winners),
                        "mean_score_delta": sum(deltas) / len(deltas),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_convert_fno_evidence_summary_writes_strategy_growth_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.fno_strategy_evidence import convert_fno_evidence_summary

    sweep_path = _write_sweep_summary(tmp_path / "fno_evidence_summary.json")
    output_path = tmp_path / "strategy_evidence_summary.json"

    evidence = convert_fno_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=output_path,
        long_window_threshold=31,
    )

    assert output_path.exists()
    assert evidence == json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == 1
    assert evidence["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert evidence["model_family"] == "FNO"
    assert evidence["source_state_hint"] == "FNO_EVIDENCE_SWEEP_COMPLETED"
    assert evidence["source_evidence_summary_path"] == str(sweep_path)
    assert evidence["evidence_summary_path"] == str(output_path)
    assert evidence["source_fno_evidence_summary_path"] == str(sweep_path)
    assert evidence["evaluation_window_count"] == 3
    assert evidence["mean_score_delta"] == 2.0
    assert evidence["score_delta_std"] == 0.816496580927726
    assert evidence["long_window_score_delta"] == 2.0
    assert evidence["physics_residual_delta"] == 0.0
    assert evidence["inference_time_regression"] == 0.0
    assert evidence["failure_rate"] == 0.0
    assert "physics_residual_delta_missing_from_fno_sweep" in evidence["provenance_notes"]
    assert evidence["next_command"] == "evaluate-strategy-promotion"


def test_convert_fno_evidence_summary_preserves_single_window_count(
    tmp_path: Path,
) -> None:
    from pde_agent.core.fno_strategy_evidence import convert_fno_evidence_summary

    sweep_path = _write_sweep_summary(
        tmp_path / "fno_evidence_summary.json",
        forecast_steps=10,
        winner_deltas=[5.0],
    )

    evidence = convert_fno_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=tmp_path / "strategy_evidence_summary.json",
        long_window_threshold=31,
    )

    assert evidence["evaluation_window_count"] == 1
    assert evidence["mean_score_delta"] == 5.0
    assert evidence["score_delta_std"] == 0.0
    assert evidence["long_window_score_delta"] == 0.0
    assert "forecast_steps_below_long_window_threshold" in evidence["provenance_notes"]


def test_convert_fno_evidence_summary_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    sweep_path = _write_sweep_summary(tmp_path / "fno_evidence_summary.json")
    output_path = tmp_path / "strategy_evidence_summary.json"

    result = CliRunner().invoke(
        app,
        [
            "convert-fno-evidence-summary",
            "--sweep-summary-path",
            str(sweep_path),
            "--output-path",
            str(output_path),
            "--long-window-threshold",
            "31",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "convert-fno-evidence-summary"
    assert payload["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert payload["artifacts"]["strategy_evidence_summary_path"] == str(output_path)
    assert payload["artifacts"]["source_fno_evidence_summary_path"] == str(sweep_path)
    assert payload["metrics"]["evaluation_window_count"] == 3
    assert payload["metrics"]["mean_score_delta"] == 2.0
    assert payload["metrics"]["failure_rate"] == 0.0
    assert payload["next_hint"] == "evaluate-strategy-promotion"
