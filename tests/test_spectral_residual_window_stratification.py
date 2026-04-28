import json
from pathlib import Path


def _write_plan(path: Path, *, grid_path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "NEXT_EXPERIMENT_PLANNED",
                "selected_intervention": "stratify_spectral_residual_window_evidence",
                "source_grid_summary_path": str(grid_path),
                "source_strategy_id": "spectral_residual_m16_ridge1e-03",
                "change_scope": "evidence_window_stratification",
                "intervention_plan": {
                    "candidate_parameters": {
                        "aggregation_policy": ["median_group_score_delta"],
                        "window_grouping": ["contiguous_pairs", "leave_one_start_out"],
                    },
                    "fixed_parameters": {
                        "modes": 16,
                        "ridge": 0.001,
                        "start_indices": [0, 4, 8, 12],
                    },
                    "forbidden_actions": ["relax promotion rubric thresholds"],
                },
                "budget": {"max_experiment_count": 2, "requires_training": False},
                "controls": {"baseline_required": True},
                "acceptance_criteria": ["group variance drops"],
                "rejection_criteria": ["group variance stays high"],
                "required_next_artifacts": ["window_stratified_evidence_summary.json"],
                "next_command": "generate-experiment-config",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_evidence(path: Path) -> Path:
    records = []
    for start_index, delta in [(0, 10.0), (4, 12.0), (8, 14.0), (12, 16.0)]:
        records.append(
            {
                "model_family": "SpectralResidual",
                "strategy_id": "spectral_residual_m16_ridge1e-03",
                "training_strategy": "spectral_residual_ridge",
                "experiment_id": f"spectral_residual_m16_ridge1e-03_s{start_index:04d}",
                "start_index": start_index,
                "score": 20.0 + delta,
                "score_delta": delta,
                "baseline_score": 20.0,
                "rel_mse": 0.01,
                "rmse": 0.05,
                "dominant_error_pattern": "shock_region_error",
                "physics_issue": "nonphysical_oscillation",
                "hyperparameters": {"modes": 16, "ridge": 0.001},
                "physics_metrics": {
                    "residual_ratio": 0.8,
                    "total_variation_ratio": 2.0,
                    "gradient_error_ratio": 1.5,
                },
                "artifacts": {"score_path": str(path.parent / f"score_{start_index}.json")},
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
                "model_family": "SpectralResidual",
                "source_state_hint": "SPECTRAL_RESIDUAL_EVIDENCE_SWEEP_COMPLETED",
                "evidence_summary_path": str(path),
                "evaluation_window_count": 4,
                "mean_score_delta": 13.0,
                "score_delta_std": 2.2360679775,
                "long_window_score_delta": 13.0,
                "physics_residual_delta": 0.0,
                "inference_time_regression": 0.0,
                "failure_rate": 0.0,
                "candidate_records": records,
                "strategy_ranking": [
                    {
                        "rank": 1,
                        "strategy_id": "spectral_residual_m16_ridge1e-03",
                        "record_count": 4,
                        "mean_score_delta": 13.0,
                        "failure_rate": 0.0,
                    }
                ],
                "best_strategy_id": "spectral_residual_m16_ridge1e-03",
                "source_metrics": {
                    "forecast_steps": 31,
                    "start_indices": [0, 4, 8, 12],
                    "sample_limit": 8,
                    "spectral_modes": 16,
                    "spectral_ridge": 0.001,
                    "record_count": 4,
                },
                "next_command": "recommend-strategy-next-action",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_grid(path: Path, *, evidence_path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "SPECTRAL_RESIDUAL_CONVERGENCE_GRID_COMPLETED",
                "candidate_count": 1,
                "candidate_reports": [
                    {
                        "experiment_id": "best_grid_candidate",
                        "parameters": {"modes": 16, "ridge": 0.001},
                        "fixed_parameters": {"start_indices": [0, 4, 8, 12]},
                        "evidence_summary_path": str(evidence_path),
                        "best_strategy_id": "spectral_residual_m16_ridge1e-03",
                        "mean_score_delta": 13.0,
                        "score_delta_std": 2.2360679775,
                        "failure_rate": 0.0,
                    }
                ],
                "best_candidate": {
                    "experiment_id": "best_grid_candidate",
                    "parameters": {"modes": 16, "ridge": 0.001},
                    "fixed_parameters": {"start_indices": [0, 4, 8, 12]},
                    "evidence_summary_path": str(evidence_path),
                    "best_strategy_id": "spectral_residual_m16_ridge1e-03",
                    "mean_score_delta": 13.0,
                    "score_delta_std": 2.2360679775,
                    "failure_rate": 0.0,
                },
                "next_command": "recommend-strategy-next-action",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_config(path: Path, *, plan_path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXPERIMENT_CONFIG_GENERATED",
                "source_plan_path": str(plan_path),
                "selected_intervention": "stratify_spectral_residual_window_evidence",
                "change_scope": "evidence_window_stratification",
                "requires_training": False,
                "budget": {"max_experiment_count": 2, "requires_training": False},
                "candidate_experiments": [
                    {
                        "experiment_id": "stratify_spectral_residual_window_evidence_001",
                        "intervention": "stratify_spectral_residual_window_evidence",
                        "change_scope": "evidence_window_stratification",
                        "requires_training": False,
                        "parameters": {
                            "aggregation_policy": "median_group_score_delta",
                            "window_grouping": "contiguous_pairs",
                        },
                        "fixed_parameters": {
                            "modes": 16,
                            "ridge": 0.001,
                            "start_indices": [0, 4, 8, 12],
                        },
                        "expected_artifacts": ["window_stratified_evidence_summary.json"],
                        "forbidden_actions": ["relax promotion rubric thresholds"],
                    },
                    {
                        "experiment_id": "stratify_spectral_residual_window_evidence_002",
                        "intervention": "stratify_spectral_residual_window_evidence",
                        "change_scope": "evidence_window_stratification",
                        "requires_training": False,
                        "parameters": {
                            "aggregation_policy": "median_group_score_delta",
                            "window_grouping": "leave_one_start_out",
                        },
                        "fixed_parameters": {
                            "modes": 16,
                            "ridge": 0.001,
                            "start_indices": [0, 4, 8, 12],
                        },
                        "expected_artifacts": ["window_stratified_evidence_summary.json"],
                        "forbidden_actions": ["relax promotion rubric thresholds"],
                    },
                ],
                "controls": {"baseline_required": True},
                "acceptance_criteria": ["group variance drops"],
                "rejection_criteria": ["group variance stays high"],
                "required_artifacts": ["window_stratified_evidence_summary.json"],
                "next_command": "execute-candidate-experiment",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_audit(path: Path, *, config_path: Path, plan_path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXPERIMENT_CONFIG_AUDITED",
                "decision": "approved_for_execution",
                "issue_count": 0,
                "issues": [],
                "config_path": str(config_path),
                "plan_path": str(plan_path),
                "summary": {
                    "selected_intervention": "stratify_spectral_residual_window_evidence",
                    "candidate_count": 2,
                    "requires_training": False,
                    "next_command": "execute-candidate-experiment",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_run_spectral_residual_window_stratification_grid_builds_group_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.spectral_residual_window_stratification import (
        run_spectral_residual_window_stratification_grid,
    )

    evidence_path = _write_evidence(tmp_path / "best_evidence.json")
    grid_path = _write_grid(tmp_path / "grid.json", evidence_path=evidence_path)
    plan_path = _write_plan(tmp_path / "plan.json", grid_path=grid_path)
    config_path = _write_config(tmp_path / "config.json", plan_path=plan_path)
    audit_path = _write_audit(
        tmp_path / "audit.json", config_path=config_path, plan_path=plan_path
    )

    summary = run_spectral_residual_window_stratification_grid(
        audit_path=audit_path,
        output_dir=tmp_path / "stratified",
        force=True,
    )

    assert summary["state_hint"] == (
        "SPECTRAL_RESIDUAL_WINDOW_STRATIFICATION_GRID_COMPLETED"
    )
    assert summary["candidate_count"] == 2
    assert summary["best_candidate"]["experiment_id"] == (
        "stratify_spectral_residual_window_evidence_002"
    )
    assert summary["best_candidate"]["score_delta_std"] <= 1.0
    best_evidence = json.loads(
        Path(summary["best_candidate"]["evidence_summary_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert best_evidence["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert best_evidence["evaluation_window_count"] == 4
    assert best_evidence["source_state_hint"] == (
        "SPECTRAL_RESIDUAL_WINDOW_STRATIFICATION_COMPLETED"
    )
    assert best_evidence["next_command"] == "recommend-strategy-next-action"


def test_run_spectral_residual_window_stratification_cli_returns_payload(
    tmp_path: Path, monkeypatch
) -> None:
    import pde_agent.commands.run_spectral_residual_window_stratification as command_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    def fake_run(**kwargs):
        return {
            "state_hint": "SPECTRAL_RESIDUAL_WINDOW_STRATIFICATION_GRID_COMPLETED",
            "window_stratification_summary_path": str(tmp_path / "summary.json"),
            "candidate_count": 2,
            "best_candidate": {
                "experiment_id": "stratify_spectral_residual_window_evidence_002",
                "best_strategy_id": "spectral_residual_m16_ridge1e-03",
                "mean_score_delta": 13.0,
                "score_delta_std": 0.5,
                "failure_rate": 0.0,
                "evidence_summary_path": str(tmp_path / "evidence.json"),
            },
            "next_command": "recommend-strategy-next-action",
        }

    monkeypatch.setattr(
        command_module, "run_spectral_residual_window_stratification_grid", fake_run
    )

    result = CliRunner().invoke(
        app,
        [
            "run-spectral-residual-window-stratification",
            "--audit-path",
            str(audit_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-spectral-residual-window-stratification"
    assert payload["state_hint"] == (
        "SPECTRAL_RESIDUAL_WINDOW_STRATIFICATION_GRID_COMPLETED"
    )
    assert payload["metrics"]["candidate_count"] == 2
    assert payload["metrics"]["best_score_delta_std"] == 0.5
    assert payload["next_hint"] == "recommend-strategy-next-action"
