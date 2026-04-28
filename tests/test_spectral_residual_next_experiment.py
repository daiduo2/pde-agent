import json
from pathlib import Path

import pytest


def _write_evidence(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
                "model_family": "SpectralResidual",
                "best_strategy_id": "spectral_residual_m16_ridge1e-03",
                "evaluation_window_count": 4,
                "mean_score_delta": 50.09,
                "score_delta_std": 5.33,
                "long_window_score_delta": 50.09,
                "physics_residual_delta": 0.0,
                "inference_time_regression": 0.0,
                "failure_rate": 0.0,
                "source_metrics": {
                    "forecast_steps": 31,
                    "sample_limit": 8,
                    "spectral_modes": 16,
                    "spectral_ridge": 0.001,
                    "start_indices": [0, 8, 16, 24],
                },
                "strategy_ranking": [
                    {
                        "strategy_id": "spectral_residual_m16_ridge1e-03",
                        "mean_score_delta": 50.09,
                        "failure_rate": 0.0,
                        "record_count": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_promotion(path: Path, *, decision: str = "revise") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_PROMOTION_EVALUATED",
                "strategy_id": "spectral_residual_m16_ridge1e-03",
                "decision": decision,
                "promote_to_skill_candidate": False,
                "reason_codes": ["score_delta_variance_too_high"],
                "metric_results": {
                    "evaluation_window_count": 4,
                    "failure_rate": 0.0,
                    "mean_score_delta": 50.09,
                    "score_delta_std": 5.33,
                    "long_window_score_delta": 50.09,
                    "physics_residual_delta": 0.0,
                    "inference_time_regression": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_ledger(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_LEDGER_APPENDED",
                "strategy_id": "spectral_residual_m16_ridge1e-03",
                "decision": "revise",
                "reason_codes": ["score_delta_variance_too_high"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_plan_next_spectral_residual_experiment_from_revise_artifacts(
    tmp_path: Path,
) -> None:
    from pde_agent.core.spectral_residual_next_experiment import (
        plan_next_spectral_residual_experiment,
    )

    evidence_path = _write_evidence(tmp_path / "evidence.json")
    promotion_path = _write_promotion(tmp_path / "promotion.json")
    ledger_path = _write_ledger(tmp_path / "ledger.jsonl")
    output_path = tmp_path / "next_plan.json"

    plan = plan_next_spectral_residual_experiment(
        evidence_summary_path=evidence_path,
        promotion_report_path=promotion_path,
        ledger_path=ledger_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert plan == json.loads(output_path.read_text(encoding="utf-8"))
    assert plan["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert plan["selected_intervention"] == "stabilize_spectral_residual_evidence"
    assert plan["failure_mode"]["reason_codes"] == ["score_delta_variance_too_high"]
    assert plan["intervention_plan"]["candidate_parameters"] == {
        "modes": [8, 12, 16],
        "ridge": [0.001, 0.01, 0.1],
    }
    assert plan["intervention_plan"]["fixed_parameters"] == {
        "start_indices": [0, 4, 8, 12, 16, 20, 24, 28],
    }
    assert plan["ranking_policy"] == [
        "failure_rate == 0",
        "score_delta_std <= 2.0",
        "mean_score_delta desc",
    ]
    assert (
        "run-spectral-residual-evidence-sweep"
        in plan["recommended_command"]["command"]
    )
    assert "--start-index 28" in plan["recommended_command"]["command"]
    assert plan["next_command"] == "generate-experiment-config"


def test_plan_next_spectral_residual_experiment_rejects_promoted_artifact(
    tmp_path: Path,
) -> None:
    from pde_agent.core.spectral_residual_next_experiment import (
        SpectralResidualNextExperimentError,
        plan_next_spectral_residual_experiment,
    )

    evidence_path = _write_evidence(tmp_path / "evidence.json")
    promotion_path = _write_promotion(tmp_path / "promotion.json", decision="promote")
    ledger_path = _write_ledger(tmp_path / "ledger.jsonl")

    with pytest.raises(SpectralResidualNextExperimentError, match="revise or reject"):
        plan_next_spectral_residual_experiment(
            evidence_summary_path=evidence_path,
            promotion_report_path=promotion_path,
            ledger_path=ledger_path,
            output_path=tmp_path / "next_plan.json",
        )


def test_plan_next_spectral_residual_experiment_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    evidence_path = _write_evidence(tmp_path / "evidence.json")
    promotion_path = _write_promotion(tmp_path / "promotion.json")
    ledger_path = _write_ledger(tmp_path / "ledger.jsonl")
    output_path = tmp_path / "next_plan.json"

    result = CliRunner().invoke(
        app,
        [
            "plan-next-spectral-residual-experiment",
            "--evidence-summary-path",
            str(evidence_path),
            "--promotion-report-path",
            str(promotion_path),
            "--ledger-path",
            str(ledger_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "plan-next-spectral-residual-experiment"
    assert payload["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert payload["artifacts"]["next_experiment_plan_path"] == str(output_path)
    assert payload["metrics"]["selected_intervention"] == (
        "stabilize_spectral_residual_evidence"
    )
    assert payload["metrics"]["candidate_count"] == 9
    assert payload["next_hint"] == "generate-experiment-config"


def test_plan_next_spectral_residual_experiment_config_audits(
    tmp_path: Path,
) -> None:
    from pde_agent.core.experiment_config import generate_experiment_config_from_file
    from pde_agent.core.experiment_config_audit import audit_experiment_config_from_file
    from pde_agent.core.spectral_residual_next_experiment import (
        plan_next_spectral_residual_experiment,
    )

    evidence_path = _write_evidence(tmp_path / "evidence.json")
    promotion_path = _write_promotion(tmp_path / "promotion.json")
    ledger_path = _write_ledger(tmp_path / "ledger.jsonl")
    plan_path = tmp_path / "next_plan.json"
    config_path = tmp_path / "experiment_config.json"
    audit_path = tmp_path / "experiment_config_audit.json"

    plan_next_spectral_residual_experiment(
        evidence_summary_path=evidence_path,
        promotion_report_path=promotion_path,
        ledger_path=ledger_path,
        output_path=plan_path,
    )
    config = generate_experiment_config_from_file(
        plan_path=plan_path,
        output_path=config_path,
    )
    audit = audit_experiment_config_from_file(
        config_path=config_path,
        output_path=audit_path,
        plan_path=plan_path,
    )

    assert len(config["candidate_experiments"]) == 9
    assert config["candidate_experiments"][0]["parameters"] == {
        "modes": 8,
        "ridge": 0.001,
    }
    assert config["candidate_experiments"][0]["fixed_parameters"] == {
        "start_indices": [0, 4, 8, 12, 16, 20, 24, 28],
    }
    assert audit["decision"] == "approved_for_execution"
