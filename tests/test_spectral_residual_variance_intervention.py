import json
from pathlib import Path


def _write_grid_summary(path: Path) -> Path:
    reports = []
    for modes, mean_delta, std in [
        (8, 28.0, 5.98),
        (12, 40.0, 4.94),
        (16, 49.2, 4.32),
    ]:
        for ridge in [0.001, 0.01, 0.1]:
            reports.append(
                {
                    "experiment_id": f"candidate_m{modes}_r{ridge:g}",
                    "parameters": {"modes": modes, "ridge": ridge},
                    "fixed_parameters": {
                        "start_indices": [0, 4, 8, 12, 16, 20, 24, 28]
                    },
                    "evidence_summary_path": str(
                        path.parent / f"evidence_m{modes}_r{ridge:g}.json"
                    ),
                    "best_strategy_id": f"spectral_residual_m{modes:02d}_ridge{ridge:.0e}",
                    "evaluation_window_count": 8,
                    "mean_score_delta": mean_delta,
                    "score_delta_std": std,
                    "long_window_score_delta": mean_delta,
                    "failure_rate": 0.0,
                    "candidate_record_count": 8,
                    "strategy_ranking_count": 1,
                }
            )
    best = next(
        report for report in reports if report["parameters"] == {"modes": 16, "ridge": 0.001}
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "SPECTRAL_RESIDUAL_CONVERGENCE_GRID_COMPLETED",
                "candidate_count": len(reports),
                "candidate_reports": reports,
                "best_candidate": best,
                "ranking_policy": [
                    "failure_rate == 0",
                    "score_delta_std <= 2.0",
                    "mean_score_delta desc",
                ],
                "convergence_grid_summary_path": str(path),
                "next_command": "recommend-strategy-next-action",
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
                    "evaluation_window_count": 8,
                    "failure_rate": 0.0,
                    "mean_score_delta": 49.2,
                    "score_delta_std": 4.32,
                    "long_window_score_delta": 49.2,
                    "physics_residual_delta": 0.0,
                    "inference_time_regression": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_plan_spectral_residual_variance_intervention_from_revise_grid(
    tmp_path: Path,
) -> None:
    from pde_agent.core.spectral_residual_variance_intervention import (
        plan_spectral_residual_variance_intervention,
    )

    grid_path = _write_grid_summary(tmp_path / "grid.json")
    promotion_path = _write_promotion(tmp_path / "promotion.json")
    output_path = tmp_path / "variance_plan.json"

    plan = plan_spectral_residual_variance_intervention(
        grid_summary_path=grid_path,
        promotion_report_path=promotion_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert plan == json.loads(output_path.read_text(encoding="utf-8"))
    assert plan["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert plan["selected_intervention"] == (
        "stratify_spectral_residual_window_evidence"
    )
    assert plan["variance_attribution"]["primary_failure_mode"] == (
        "start_window_variance_dominates_parameter_sweep"
    )
    assert plan["variance_attribution"]["best_candidate_still_violates_std_gate"] is True
    assert plan["variance_attribution"]["parameter_sensitivity"]["ridge"][
        "score_delta_std_range"
    ] == 0.0
    assert plan["intervention_plan"]["candidate_parameters"] == {
        "aggregation_policy": ["median_group_score_delta", "min_group_score_delta"],
        "window_grouping": [
            "contiguous_pairs",
            "interleaved_even_odd",
            "leave_one_start_out",
        ],
    }
    assert plan["intervention_plan"]["fixed_parameters"] == {
        "modes": 16,
        "ridge": 0.001,
        "start_indices": [0, 4, 8, 12, 16, 20, 24, 28],
    }
    assert plan["budget"]["max_experiment_count"] == 6
    assert plan["next_command"] == "generate-experiment-config"


def test_plan_spectral_residual_variance_intervention_rejects_promoted_report(
    tmp_path: Path,
) -> None:
    from pde_agent.core.spectral_residual_variance_intervention import (
        SpectralResidualVarianceInterventionError,
        plan_spectral_residual_variance_intervention,
    )

    grid_path = _write_grid_summary(tmp_path / "grid.json")
    promotion_path = _write_promotion(tmp_path / "promotion.json", decision="promote")

    try:
        plan_spectral_residual_variance_intervention(
            grid_summary_path=grid_path,
            promotion_report_path=promotion_path,
            output_path=tmp_path / "variance_plan.json",
        )
    except SpectralResidualVarianceInterventionError as exc:
        assert "revise or reject" in str(exc)
    else:
        raise AssertionError("expected revise/reject validation error")


def test_plan_spectral_residual_variance_intervention_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    grid_path = _write_grid_summary(tmp_path / "grid.json")
    promotion_path = _write_promotion(tmp_path / "promotion.json")
    output_path = tmp_path / "variance_plan.json"

    result = CliRunner().invoke(
        app,
        [
            "plan-spectral-residual-variance-intervention",
            "--grid-summary-path",
            str(grid_path),
            "--promotion-report-path",
            str(promotion_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "plan-spectral-residual-variance-intervention"
    assert payload["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert payload["metrics"]["candidate_count"] == 6
    assert payload["metrics"]["primary_failure_mode"] == (
        "start_window_variance_dominates_parameter_sweep"
    )
    assert payload["next_hint"] == "generate-experiment-config"
