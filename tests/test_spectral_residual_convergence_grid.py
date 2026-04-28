import json
from pathlib import Path


def _write_config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXPERIMENT_CONFIG_GENERATED",
                "source_plan_path": str(path.parent / "plan.json"),
                "selected_intervention": "stabilize_spectral_residual_evidence",
                "change_scope": "evidence_sweep_parameter_grid",
                "requires_training": False,
                "budget": {
                    "max_experiment_count": 2,
                    "max_train_time_seconds": 0,
                    "max_inference_time_seconds": 600,
                    "requires_training": False,
                },
                "candidate_experiments": [
                    {
                        "experiment_id": "stabilize_spectral_residual_evidence_001",
                        "intervention": "stabilize_spectral_residual_evidence",
                        "change_scope": "evidence_sweep_parameter_grid",
                        "requires_training": False,
                        "parameters": {"modes": 8, "ridge": 0.001},
                        "fixed_parameters": {"start_indices": [0, 4]},
                        "expected_artifacts": ["spectral_residual_evidence_summary.json"],
                        "forbidden_actions": ["relax promotion rubric thresholds"],
                    },
                    {
                        "experiment_id": "stabilize_spectral_residual_evidence_002",
                        "intervention": "stabilize_spectral_residual_evidence",
                        "change_scope": "evidence_sweep_parameter_grid",
                        "requires_training": False,
                        "parameters": {"modes": 12, "ridge": 0.01},
                        "fixed_parameters": {"start_indices": [0, 4]},
                        "expected_artifacts": ["spectral_residual_evidence_summary.json"],
                        "forbidden_actions": ["relax promotion rubric thresholds"],
                    },
                ],
                "controls": {"baseline_required": True},
                "acceptance_criteria": ["failure_rate remains 0.0"],
                "rejection_criteria": ["failure_rate becomes nonzero"],
                "required_artifacts": ["spectral_residual_evidence_summary.json"],
                "next_command": "execute-candidate-experiment",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_audit(path: Path, *, config_path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXPERIMENT_CONFIG_AUDITED",
                "decision": "approved_for_execution",
                "issue_count": 0,
                "issues": [],
                "config_path": str(config_path),
                "plan_path": str(path.parent / "plan.json"),
                "summary": {
                    "selected_intervention": "stabilize_spectral_residual_evidence",
                    "candidate_count": 2,
                    "requires_training": False,
                    "next_command": "execute-candidate-experiment",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_run_spectral_residual_convergence_grid_ranks_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    import pde_agent.core.spectral_residual_convergence_grid as grid_module
    from pde_agent.core.spectral_residual_convergence_grid import (
        run_spectral_residual_convergence_grid,
    )

    config_path = _write_config(tmp_path / "config.json")
    audit_path = _write_audit(tmp_path / "audit.json", config_path=config_path)
    calls = []

    def fake_sweep(**kwargs):
        calls.append(kwargs)
        modes = kwargs["spectral_modes"]
        score_delta = 40.0 if modes == 12 else 45.0
        score_std = 1.5 if modes == 12 else 4.0
        evidence_path = Path(kwargs["output_dir"]) / "spectral_residual_evidence_summary.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
            "model_family": "SpectralResidual",
            "evidence_summary_path": str(evidence_path),
            "best_strategy_id": f"spectral_residual_m{modes:02d}",
            "evaluation_window_count": 2,
            "mean_score_delta": score_delta,
            "score_delta_std": score_std,
            "long_window_score_delta": score_delta,
            "failure_rate": 0.0,
            "candidate_records": [{}, {}],
            "strategy_ranking": [{"strategy_id": f"spectral_residual_m{modes:02d}"}],
        }

    monkeypatch.setattr(grid_module, "run_spectral_residual_evidence_sweep", fake_sweep)

    summary = run_spectral_residual_convergence_grid(
        audit_path=audit_path,
        output_dir=tmp_path / "grid",
        source_path=tmp_path / "burgers.hdf5",
        dataset_key="tensor",
        sample_limit=8,
        context_steps=10,
        forecast_steps=31,
        time_reduction=5,
        space_reduction=4,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=True,
    )

    assert [call["spectral_modes"] for call in calls] == [8, 12]
    assert calls[0]["start_indices"] == [0, 4]
    assert summary["state_hint"] == "SPECTRAL_RESIDUAL_CONVERGENCE_GRID_COMPLETED"
    assert summary["candidate_count"] == 2
    assert summary["best_candidate"]["experiment_id"] == (
        "stabilize_spectral_residual_evidence_002"
    )
    assert summary["best_candidate"]["score_delta_std"] == 1.5
    assert Path(summary["convergence_grid_summary_path"]).exists()
    assert summary["next_command"] == "recommend-strategy-next-action"


def test_run_spectral_residual_convergence_grid_cli_returns_payload(
    tmp_path: Path, monkeypatch
) -> None:
    import pde_agent.commands.run_spectral_residual_convergence_grid as command_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    def fake_run(**kwargs):
        return {
            "state_hint": "SPECTRAL_RESIDUAL_CONVERGENCE_GRID_COMPLETED",
            "convergence_grid_summary_path": str(tmp_path / "summary.json"),
            "candidate_count": 2,
            "best_candidate": {
                "experiment_id": "stabilize_spectral_residual_evidence_002",
                "best_strategy_id": "spectral_residual_m12",
                "mean_score_delta": 40.0,
                "score_delta_std": 1.5,
                "failure_rate": 0.0,
            },
            "next_command": "recommend-strategy-next-action",
        }

    monkeypatch.setattr(command_module, "run_spectral_residual_convergence_grid", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "run-spectral-residual-convergence-grid",
            "--audit-path",
            str(audit_path),
            "--output-dir",
            str(tmp_path / "grid"),
            "--source-path",
            str(tmp_path / "burgers.hdf5"),
            "--dataset-key",
            "tensor",
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-spectral-residual-convergence-grid"
    assert payload["state_hint"] == "SPECTRAL_RESIDUAL_CONVERGENCE_GRID_COMPLETED"
    assert payload["metrics"]["candidate_count"] == 2
    assert payload["metrics"]["best_experiment_id"] == (
        "stabilize_spectral_residual_evidence_002"
    )
    assert payload["next_hint"] == "recommend-strategy-next-action"
