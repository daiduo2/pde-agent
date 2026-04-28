import json
from pathlib import Path

import pytest


def _write_comparison_report(
    path: Path,
    *,
    sample_limit: int = 8,
    forecast_steps: int = 31,
) -> Path:
    records = [
        {
            "model_id": "persistence",
            "model_family": "Persistence",
            "source_type": "baseline",
            "adapter_type": "persistence",
            "score": 17.0,
            "score_delta_vs_persistence": 0.0,
            "rel_mse": 0.25,
            "rmse": 0.22,
            "dominant_error_pattern": "overdamped_rollout",
            "physics_issue": "nonphysical_oscillation",
        },
        {
            "model_id": "spectral_residual",
            "model_family": "SpectralResidual",
            "source_type": "local_baseline",
            "adapter_type": "spectral_residual_ridge",
            "score": 74.0,
            "score_delta_vs_persistence": 57.0,
            "rel_mse": 0.01,
            "rmse": 0.05,
            "dominant_error_pattern": "shock_region_error",
            "physics_issue": "nonphysical_oscillation",
        },
        {
            "model_id": "official_fno",
            "model_family": "FNO",
            "source_type": "official_checkpoint",
            "adapter_type": "pdebench_fno1d",
            "score": 40.0,
            "score_delta_vs_persistence": 23.0,
            "rel_mse": 0.14,
            "rmse": 0.16,
            "dominant_error_pattern": "long_horizon_drift",
            "physics_issue": "mass_drift",
        },
        {
            "model_id": "official_unet",
            "model_family": "UNet",
            "source_type": "official_checkpoint",
            "adapter_type": "pdebench_unet1d",
            "score": 7.0,
            "score_delta_vs_persistence": -10.0,
            "rel_mse": 1.05,
            "rmse": 0.63,
            "dominant_error_pattern": "short_horizon_bias",
            "physics_issue": "overdissipation",
        },
    ]
    payload = {
        "schema_version": 1,
        "state_hint": "OFFICIAL_OPERATOR_COMPARISON_COMPLETED",
        "inputs": {
            "sample_limit": sample_limit,
            "forecast_steps": forecast_steps,
        },
        "model_records": records,
        "leaderboard": [
            {"rank": 1, "model_id": "spectral_residual"},
            {"rank": 2, "model_id": "official_fno"},
            {"rank": 3, "model_id": "persistence"},
            {"rank": 4, "model_id": "official_unet"},
        ],
        "best_model_id": "spectral_residual",
        "best_model_family": "SpectralResidual",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_convert_official_comparison_to_strategy_evidence(tmp_path: Path) -> None:
    from pde_agent.core.official_comparison_evidence import (
        convert_official_comparison_to_evidence,
    )

    report_path = _write_comparison_report(tmp_path / "comparison.json")
    output_path = tmp_path / "strategy_evidence_summary.json"

    evidence = convert_official_comparison_to_evidence(
        comparison_report_path=report_path,
        output_path=output_path,
        long_window_threshold=31,
        force=False,
    )

    assert output_path.exists()
    assert evidence == json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert evidence["model_family"] == "OfficialOperatorComparison"
    assert evidence["source_state_hint"] == "OFFICIAL_OPERATOR_COMPARISON_COMPLETED"
    assert evidence["source_official_comparison_report_path"] == str(report_path)
    assert evidence["evaluation_window_count"] == 8
    assert evidence["mean_score_delta"] == pytest.approx((57.0 + 23.0 - 10.0) / 3.0)
    assert evidence["long_window_score_delta"] == evidence["mean_score_delta"]
    assert evidence["failure_rate"] == pytest.approx(1.0 / 3.0)
    assert evidence["best_strategy_id"] == "spectral_residual"
    assert [row["strategy_id"] for row in evidence["strategy_ranking"]] == [
        "spectral_residual",
        "official_fno",
        "official_unet",
    ]
    assert evidence["candidate_records"][1]["model_family"] == "FNO"
    assert evidence["candidate_records"][1]["hyperparameters"] == {
        "adapter_type": "pdebench_fno1d",
        "source_type": "official_checkpoint",
    }
    assert evidence["next_command"] == "recommend-strategy-next-action"


def test_convert_official_comparison_to_strategy_evidence_marks_short_horizon(
    tmp_path: Path,
) -> None:
    from pde_agent.core.official_comparison_evidence import (
        convert_official_comparison_to_evidence,
    )

    report_path = _write_comparison_report(
        tmp_path / "comparison.json",
        forecast_steps=8,
    )

    evidence = convert_official_comparison_to_evidence(
        comparison_report_path=report_path,
        output_path=tmp_path / "strategy_evidence_summary.json",
        long_window_threshold=31,
        force=False,
    )

    assert evidence["long_window_score_delta"] == 0.0
    assert "forecast_steps_below_long_window_threshold" in evidence["provenance_notes"]


def test_convert_official_comparison_evidence_rejects_bad_state_hint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.official_comparison_evidence import (
        OfficialComparisonEvidenceError,
        convert_official_comparison_to_evidence,
    )

    report_path = _write_comparison_report(tmp_path / "comparison.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["state_hint"] = "WRONG"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OfficialComparisonEvidenceError, match="invalid state_hint"):
        convert_official_comparison_to_evidence(
            comparison_report_path=report_path,
            output_path=tmp_path / "strategy_evidence_summary.json",
            long_window_threshold=31,
            force=False,
        )


def test_convert_official_comparison_evidence_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    report_path = _write_comparison_report(tmp_path / "comparison.json")
    output_path = tmp_path / "strategy_evidence_summary.json"

    result = CliRunner().invoke(
        app,
        [
            "convert-official-comparison-evidence",
            "--comparison-report-path",
            str(report_path),
            "--output-path",
            str(output_path),
            "--long-window-threshold",
            "31",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "convert-official-comparison-evidence"
    assert payload["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert payload["artifacts"]["strategy_evidence_summary_path"] == str(output_path)
    assert payload["metrics"]["model_family"] == "OfficialOperatorComparison"
    assert payload["metrics"]["candidate_record_count"] == 3
    assert payload["metrics"]["best_strategy_id"] == "spectral_residual"
    assert payload["next_hint"] == "recommend-strategy-next-action"
