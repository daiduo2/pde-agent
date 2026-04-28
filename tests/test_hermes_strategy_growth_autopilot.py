import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_strategy_artifacts(root: Path) -> tuple[Path, Path]:
    proposal = _write_json(
        root / "strategy_proposal.json",
        {
            "schema_version": 1,
            "state_hint": "STRATEGY_PROPOSAL_RECORDED",
            "strategy_id": "spectral_residual_m16_ridge1e-03",
            "title": "Spectral residual ridge multi-start Burgers baseline",
            "source_type": "local_library",
            "problem_context": "Burgers rollout evidence needs stable promotion.",
            "hypothesis": "Window-stratified evidence stabilizes promotion.",
            "mechanism": "Fit spectral residual ridge over persistence.",
            "implementation_plan": {
                "allowed_commands": ["run-spectral-residual-evidence-sweep"],
                "expected_artifacts": ["spectral_residual_evidence_summary.json"],
            },
            "expected_observables": ["positive score delta"],
            "known_risks": ["same-window fitting can overestimate generalization"],
            "strategy_proposal_path": str(root / "strategy_proposal.json"),
            "next_command": "build-strategy-rubric",
        },
    )
    rubric = _write_json(
        root / "experiment_rubric.json",
        {
            "schema_version": 1,
            "state_hint": "STRATEGY_RUBRIC_BUILT",
            "strategy_id": "spectral_residual_m16_ridge1e-03",
            "strategy_proposal_path": str(proposal),
            "experiment_rubric_path": str(root / "experiment_rubric.json"),
            "required_metrics": [
                "evaluation_window_count",
                "mean_score_delta",
                "score_delta_std",
                "long_window_score_delta",
                "physics_residual_delta",
                "inference_time_regression",
                "failure_rate",
            ],
            "promotion_thresholds": {
                "min_evaluation_windows": 3,
                "min_mean_score_delta": 0.0,
                "min_long_window_score_delta": 0.0,
                "max_score_delta_std": 2.0,
                "max_physics_residual_delta": 0.0,
                "max_inference_time_regression": 0.1,
                "max_failure_rate": 0.0,
            },
            "next_command": "evaluate-strategy-promotion",
        },
    )
    return proposal, rubric


def _write_window_plan(root: Path) -> Path:
    source_evidence = _write_json(
        root / "best_evidence.json",
        {
            "schema_version": 1,
            "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
            "model_family": "SpectralResidual",
            "source_state_hint": "SPECTRAL_RESIDUAL_EVIDENCE_SWEEP_COMPLETED",
            "evidence_summary_path": str(root / "best_evidence.json"),
            "evaluation_window_count": 4,
            "mean_score_delta": 13.0,
            "score_delta_std": 2.23,
            "long_window_score_delta": 13.0,
            "physics_residual_delta": 0.0,
            "inference_time_regression": 0.0,
            "failure_rate": 0.0,
            "candidate_records": [
                {
                    "model_family": "SpectralResidual",
                    "strategy_id": "spectral_residual_m16_ridge1e-03",
                    "training_strategy": "spectral_residual_ridge",
                    "experiment_id": f"spectral_residual_m16_ridge1e-03_s{start:04d}",
                    "start_index": start,
                    "score": 20.0 + delta,
                    "score_delta": delta,
                    "baseline_score": 20.0,
                    "dominant_error_pattern": "shock_region_error",
                    "physics_issue": "nonphysical_oscillation",
                }
                for start, delta in [(0, 10.0), (4, 12.0), (8, 14.0), (12, 16.0)]
            ],
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
                "start_indices": [0, 4, 8, 12],
                "spectral_modes": 16,
                "spectral_ridge": 0.001,
            },
            "next_command": "recommend-strategy-next-action",
        },
    )
    grid = _write_json(
        root / "convergence_grid.json",
        {
            "schema_version": 1,
            "state_hint": "SPECTRAL_RESIDUAL_CONVERGENCE_GRID_COMPLETED",
            "candidate_count": 1,
            "candidate_reports": [],
            "best_candidate": {
                "experiment_id": "best_grid_candidate",
                "parameters": {"modes": 16, "ridge": 0.001},
                "fixed_parameters": {"start_indices": [0, 4, 8, 12]},
                "evidence_summary_path": str(source_evidence),
                "best_strategy_id": "spectral_residual_m16_ridge1e-03",
                "mean_score_delta": 13.0,
                "score_delta_std": 2.23,
                "failure_rate": 0.0,
            },
            "next_command": "recommend-strategy-next-action",
        },
    )
    return _write_json(
        root / "variance_plan.json",
        {
            "schema_version": 1,
            "state_hint": "NEXT_EXPERIMENT_PLANNED",
            "selected_intervention": "stratify_spectral_residual_window_evidence",
            "source_grid_summary_path": str(grid),
            "source_strategy_id": "spectral_residual_m16_ridge1e-03",
            "change_scope": "evidence_window_stratification",
            "intervention_plan": {
                "candidate_parameters": {
                    "aggregation_policy": ["min_group_score_delta"],
                    "window_grouping": ["leave_one_start_out"],
                },
                "fixed_parameters": {
                    "modes": 16,
                    "ridge": 0.001,
                    "start_indices": [0, 4, 8, 12],
                },
                "forbidden_actions": ["relax promotion rubric thresholds"],
            },
            "budget": {"max_experiment_count": 1, "requires_training": False},
            "controls": {"baseline_required": True},
            "acceptance_criteria": ["group variance drops"],
            "rejection_criteria": ["group variance stays high"],
            "required_next_artifacts": ["window_stratified_evidence_summary.json"],
            "next_command": "generate-experiment-config",
        },
    )


def test_hermes_strategy_growth_autopilot_reaches_registered_skill(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_strategy_growth_autopilot import (
        run_hermes_strategy_growth_autopilot,
    )

    proposal, rubric = _write_strategy_artifacts(tmp_path)
    seed_plan = _write_window_plan(tmp_path)

    report = run_hermes_strategy_growth_autopilot(
        seed_artifact_path=seed_plan,
        workspace_dir=tmp_path / "autopilot",
        proposal_path=proposal,
        rubric_path=rubric,
        skill_name="spectral-residual-window-stratified-burgers",
        install_root=tmp_path / "installed_skills",
        registry_path=tmp_path / "strategy_registry.json",
        reviewer_id="hermes-autopilot",
        max_steps=16,
        force=True,
    )

    assert report["state_hint"] == "HERMES_STRATEGY_GROWTH_AUTOPILOT_COMPLETED"
    assert report["final_state_hint"] == "STRATEGY_SKILL_REGISTERED"
    assert report["human_review_required"] is False
    assert report["step_count"] >= 10
    assert [step["command"] for step in report["steps"]] == [
        "generate-experiment-config",
        "audit-experiment-config",
        "run-spectral-residual-window-stratification",
        "recommend-strategy-next-action",
        "evaluate-strategy-promotion",
        "append-strategy-ledger",
        "create-skill-candidate",
        "review-skill-candidate",
        "materialize-skill-draft",
        "review-skill-draft",
        "install-skill-draft",
        "register-strategy-skill",
    ]
    registry = json.loads((tmp_path / "strategy_registry.json").read_text())
    assert registry["state_hint"] == "STRATEGY_SKILL_REGISTRY_UPDATED"
    assert registry["entries"][0]["strategy_id"] == "spectral_residual_m16_ridge1e-03"
    assert registry["entries"][0]["skill_name"] == (
        "spectral-residual-window-stratified-burgers"
    )


def test_hermes_strategy_growth_autopilot_cli_returns_payload(
    tmp_path: Path, monkeypatch
) -> None:
    import pde_agent.commands.hermes_strategy_growth_autopilot as command_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    def fake_run(**kwargs):
        return {
            "state_hint": "HERMES_STRATEGY_GROWTH_AUTOPILOT_COMPLETED",
            "autopilot_report_path": str(tmp_path / "report.json"),
            "final_state_hint": "STRATEGY_SKILL_REGISTERED",
            "step_count": 12,
            "human_review_required": False,
            "registry_path": str(tmp_path / "registry.json"),
            "installed_skill_manifest_path": str(tmp_path / "install.json"),
            "steps": [],
        }

    monkeypatch.setattr(command_module, "run_hermes_strategy_growth_autopilot", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "hermes-strategy-growth-autopilot",
            "--seed-artifact-path",
            str(tmp_path / "seed.json"),
            "--workspace-dir",
            str(tmp_path / "workspace"),
            "--proposal-path",
            str(tmp_path / "proposal.json"),
            "--rubric-path",
            str(tmp_path / "rubric.json"),
            "--skill-name",
            "spectral-residual-window-stratified-burgers",
            "--install-root",
            str(tmp_path / "installed"),
            "--registry-path",
            str(tmp_path / "registry.json"),
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "hermes-strategy-growth-autopilot"
    assert payload["state_hint"] == "HERMES_STRATEGY_GROWTH_AUTOPILOT_COMPLETED"
    assert payload["metrics"]["final_state_hint"] == "STRATEGY_SKILL_REGISTERED"
    assert payload["next_hint"] == "strategy-growth-complete"
