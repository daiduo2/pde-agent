import json
from pathlib import Path


def test_create_skill_candidate_includes_window_stratification_steps(
    tmp_path: Path,
) -> None:
    from pde_agent.core.strategy_growth import create_skill_candidate

    proposal_path = tmp_path / "proposal.json"
    rubric_path = tmp_path / "rubric.json"
    evidence_path = tmp_path / "evidence.json"
    promotion_path = tmp_path / "promotion.json"
    candidate_path = tmp_path / "skill_candidate.json"
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_PROPOSAL_RECORDED",
                "strategy_id": "spectral_residual_m16_ridge1e-03",
                "title": "Spectral residual ridge multi-start Burgers baseline",
                "source_type": "local_library",
                "problem_context": "Burgers rollout variance remained high.",
                "hypothesis": "Window-stratified evidence stabilizes promotion.",
                "mechanism": "Fit spectral residual ridge over persistence.",
                "implementation_plan": {
                    "allowed_commands": ["run-spectral-residual-evidence-sweep"],
                    "expected_artifacts": ["spectral_residual_evidence_summary.json"],
                },
                "expected_observables": ["positive score delta"],
                "known_risks": ["same-window fitting can overestimate generalization"],
                "strategy_proposal_path": str(proposal_path),
                "next_command": "build-strategy-rubric",
            }
        ),
        encoding="utf-8",
    )
    rubric_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_RUBRIC_BUILT",
                "strategy_id": "spectral_residual_m16_ridge1e-03",
                "strategy_proposal_path": str(proposal_path),
                "experiment_rubric_path": str(rubric_path),
                "required_metrics": [
                    "evaluation_window_count",
                    "mean_score_delta",
                    "score_delta_std",
                    "long_window_score_delta",
                    "physics_residual_delta",
                    "inference_time_regression",
                    "failure_rate",
                ],
                "promotion_thresholds": {"max_score_delta_std": 2.0},
                "next_command": "evaluate-strategy-promotion",
            }
        ),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
                "model_family": "SpectralResidual",
                "source_state_hint": "SPECTRAL_RESIDUAL_WINDOW_STRATIFICATION_COMPLETED",
                "best_strategy_id": "spectral_residual_m16_ridge1e-03",
                "evaluation_window_count": 8,
                "mean_score_delta": 42.95,
                "score_delta_std": 0.66,
                "long_window_score_delta": 42.95,
                "physics_residual_delta": 0.0,
                "inference_time_regression": 0.0,
                "failure_rate": 0.0,
                "source_metrics": {
                    "window_grouping": "leave_one_start_out",
                    "aggregation_policy": "min_group_score_delta",
                    "spectral_modes": 16,
                    "spectral_ridge": 0.001,
                },
                "strategy_ranking": [],
                "evidence_summary_path": str(evidence_path),
                "next_command": "recommend-strategy-next-action",
            }
        ),
        encoding="utf-8",
    )
    promotion_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_PROMOTION_EVALUATED",
                "strategy_id": "spectral_residual_m16_ridge1e-03",
                "decision": "promote",
                "promote_to_skill_candidate": True,
                "reason_codes": ["all_thresholds_passed"],
                "metric_results": {
                    "evaluation_window_count": 8,
                    "mean_score_delta": 42.95,
                    "score_delta_std": 0.66,
                    "long_window_score_delta": 42.95,
                    "physics_residual_delta": 0.0,
                    "inference_time_regression": 0.0,
                    "failure_rate": 0.0,
                },
                "next_command": "append-strategy-ledger",
            }
        ),
        encoding="utf-8",
    )

    candidate = create_skill_candidate(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        promotion_report_path=promotion_path,
        output_path=candidate_path,
        skill_name="spectral-residual-window-stratified-burgers",
        skill_dir="spectral-residual-window-stratified-burgers",
    )

    procedure = "\n".join(candidate["procedure_steps"])
    assert (
        "Run only the approved commands: run-spectral-residual-evidence-sweep, "
        "run-spectral-residual-window-stratification"
    ) in procedure
    assert "run-spectral-residual-window-stratification" in procedure
    assert "window_grouping=leave_one_start_out" in procedure
    assert "aggregation_policy=min_group_score_delta" in procedure
