"""Tests for FNO training closed loop modules."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pde_agent.core.fno_training_agent_interface import (
    FnoTrainingAgentInterfaceError,
    build_fno_training_closed_loop_contract,
    build_fno_training_iteration_report,
)
from pde_agent.core.fno_training_checkpoint import (
    FnoTrainingCheckpointError,
    checkpoint_training_loop,
    list_training_checkpoints,
    resume_training_loop,
)
from pde_agent.core.fno_training_convergence import (
    FnoTrainingConvergenceError,
    check_training_convergence,
    check_training_doom_loop,
    detect_overfitting,
)
from pde_agent.core.fno_training_memory import (
    FnoTrainingMemoryError,
    build_training_iteration_memory,
    load_training_iteration_memory,
    merge_training_memory_with_hypothesis_feedback,
)


def _make_opt_report(
    candidate_records: list[dict[str, Any]],
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
        "candidate_records": candidate_records,
        "selected_candidate": selected or candidate_records[0] if candidate_records else {},
    }


def _make_candidate(
    candidate_id: str,
    score: float,
    model_family: str = "fno_lite",
    loss_type: str = "mse",
    effective_improvement: bool | None = None,
    rules_passed: bool | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "score": score,
        "model_family": model_family,
        "loss_type": loss_type,
        "effective_improvement": effective_improvement,
        "rules_passed": rules_passed,
        "config": config or {"modes": 8, "width": 32},
    }


class TestMemoryBuilder:
    def test_memory_structure(self) -> None:
        report = _make_opt_report(
            candidate_records=[
                _make_candidate("c1", 0.5, effective_improvement=True),
            ]
        )
        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )
        assert memory["state_hint"] == "FNO_TRAINING_MEMORY_BUILT"
        assert memory["iteration_index"] == 0
        assert "trajectory_summary" in memory
        assert "best_so_far" in memory
        assert "avoid_repeating" in memory
        assert "family_outcomes" in memory
        assert "positive_signals" in memory
        assert "search_guidance" in memory
        assert "convergence_history" in memory
        assert "next_hints" in memory

    def test_memory_uses_config_fallback_for_training_fields(self) -> None:
        report = _make_opt_report(
            candidate_records=[
                _make_candidate(
                    "c1",
                    0.5,
                    model_family="fno_lite",
                    config={
                        "model_family": "fno_lite",
                        "loss_type": "early_weighted_mse",
                        "forecast_steps": 12,
                        "train_steps": 4,
                        "modes": 8,
                        "width": 32,
                    },
                    effective_improvement=True,
                ),
            ]
        )
        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )

        assert memory["best_so_far"]["loss_type"] == "early_weighted_mse"
        assert memory["best_so_far"]["forecast_steps"] == 12
        assert memory["best_so_far"]["train_steps"] == 4
        trajectory = memory["trajectory_summary"][0]
        assert trajectory["loss_type"] == "early_weighted_mse"
        assert trajectory["forecast_steps"] == 12
        assert trajectory["train_steps"] == 4
        signal = memory["positive_signals"][0]
        assert signal["loss_type"] == "early_weighted_mse"

    def test_memory_preserves_global_best_across_regression(self) -> None:
        previous_memory = {
            "state_hint": "FNO_TRAINING_MEMORY_BUILT",
            "iteration_index": 0,
            "best_so_far": {
                "candidate_id": "prev_best",
                "score": 59.7,
                "config": {
                    "model_family": "fno_ar1",
                    "loss_type": "early_weighted_mse",
                    "forecast_steps": 12,
                    "train_steps": 4,
                    "modes": 8,
                    "width": 40,
                    "depth": 1,
                },
                "model_family": "fno_ar1",
                "loss_type": "early_weighted_mse",
                "forecast_steps": 12,
                "train_steps": 4,
            },
        }
        report = _make_opt_report(
            candidate_records=[
                _make_candidate(
                    "current_best",
                    55.0,
                    model_family="fno_lite",
                    config={
                        "model_family": "fno_lite",
                        "loss_type": "mse",
                        "forecast_steps": 20,
                        "train_steps": 2,
                        "modes": 4,
                        "width": 32,
                        "depth": 1,
                    },
                )
            ]
        )

        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=1,
            previous_memory=previous_memory,
        )

        assert memory["current_best"]["candidate_id"] == "current_best"
        assert memory["best_so_far"]["candidate_id"] == "prev_best"
        assert memory["best_so_far"]["score"] == 59.7

    def test_memory_promotes_new_global_best_to_training_anchor(self) -> None:
        previous_memory = {
            "state_hint": "FNO_TRAINING_MEMORY_BUILT",
            "iteration_index": 0,
            "best_so_far": {"score": 56.0},
        }
        config = {
            "model_family": "fno_ar1",
            "loss_type": "early_weighted_mse",
            "forecast_steps": 12,
            "train_steps": 4,
            "modes": 8,
            "width": 40,
            "depth": 1,
            "normalization": "none",
        }
        report = _make_opt_report(
            candidate_records=[
                _make_candidate(
                    "new_best",
                    59.7,
                    model_family="fno_ar1",
                    loss_type="early_weighted_mse",
                    config=config,
                )
            ]
        )

        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=1,
            previous_memory=previous_memory,
        )

        anchors = memory["training_memory_anchors"]
        assert anchors[0]["anchor_type"] == "operator_training_skill_candidate"
        assert anchors[0]["model_family"] == "fno_ar1"
        assert anchors[0]["config"] == config
        assert anchors[0]["status"] == "active"
        assert "reuse_as_training_anchor" in anchors[0]["next_actions"]

    def test_training_anchor_lifecycle_challenges_repeated_regression(self) -> None:
        config = {
            "model_family": "fno_ar1",
            "loss_type": "early_weighted_mse",
            "forecast_steps": 12,
            "train_steps": 4,
            "modes": 8,
            "width": 40,
            "depth": 1,
            "normalization": "none",
        }
        previous_memory = {
            "state_hint": "FNO_TRAINING_MEMORY_BUILT",
            "best_so_far": {"score": 59.7, "config": config},
            "training_memory_anchors": [
                {
                    "anchor_id": "training_anchor_fno_ar1",
                    "anchor_type": "operator_training_skill_candidate",
                    "status": "active",
                    "model_family": "fno_ar1",
                    "loss_type": "early_weighted_mse",
                    "score": 59.7,
                    "best_score": 59.7,
                    "reuse_count": 1,
                    "confirmation_count": 1,
                    "regression_count": 1,
                    "config": config,
                }
            ],
        }
        report = _make_opt_report(
            candidate_records=[
                _make_candidate(
                    "anchor_reuse_regressed",
                    57.0,
                    model_family="fno_ar1",
                    loss_type="early_weighted_mse",
                    config=config,
                )
            ]
        )

        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=2,
            previous_memory=previous_memory,
        )

        anchor = memory["training_memory_anchors"][0]
        assert anchor["status"] == "challenged"
        assert anchor["reuse_count"] == 2
        assert anchor["regression_count"] == 2

    def test_memory_includes_operator_training_decision_policy(self) -> None:
        report = _make_opt_report(
            candidate_records=[
                {
                    "candidate_id": "c1",
                    "score": 20.0,
                    "model_family": "fno_lite",
                    "early_rel_mse": 1.8,
                    "loss_curve": {"train": [2.4, 2.0], "val": [2.5, 2.1]},
                    "config": {
                        "model_family": "fno_lite",
                        "loss_type": "early_weighted_mse",
                        "forecast_steps": 12,
                        "train_steps": 4,
                        "modes": 8,
                        "width": 32,
                        "depth": 1,
                        "normalization": "none",
                    },
                }
            ]
        )
        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )

        decision = memory["operator_training_decision"]
        assert decision["state_hint"] == "OPERATOR_TRAINING_DECISION_BUILT"
        assert {item["code"] for item in decision["diagnosis"]} >= {"undertrained"}
        assert memory["decision_policy"]["next_action"] == "increase_training_budget"
        assert memory["decision_policy"]["budget_adjustments"]["train_steps_delta"] > 0
        assert any(
            item["guidance_type"] == "operator_training_policy"
            for item in memory["search_guidance"]
        )

    def test_memory_builder_avoid_list(self) -> None:
        report = _make_opt_report(
            candidate_records=[
                _make_candidate("c1", 0.5, effective_improvement=True),
                _make_candidate("c2", 0.3, effective_improvement=False),
                _make_candidate("c3", 0.2, rules_passed=False),
            ]
        )
        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )
        avoid = memory["avoid_repeating"]
        assert len(avoid) == 2
        assert avoid[0]["candidate_id"] == "c2"
        assert avoid[0]["reason"] == "below_min_effective_improvement"
        assert avoid[1]["candidate_id"] == "c3"
        assert avoid[1]["reason"] == "rule_gate_failed"

    def test_memory_builder_family_outcomes(self) -> None:
        report = _make_opt_report(
            candidate_records=[
                _make_candidate("c1", 0.5, model_family="fno_lite", effective_improvement=True),
                _make_candidate("c2", 0.3, model_family="conv1d_residual", effective_improvement=False),
                _make_candidate("c3", 0.1, model_family="conv1d_residual", effective_improvement=False),
            ]
        )
        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )
        outcomes = memory["family_outcomes"]
        assert outcomes["fno_lite"]["status"] == "positive_signal"
        assert outcomes["conv1d_residual"]["status"] == "weak_or_failed"
        assert outcomes["conv1d_residual"]["attempt_count"] == 2

    def test_memory_invalid_state_hint(self) -> None:
        with pytest.raises(FnoTrainingMemoryError):
            build_training_iteration_memory(
                optimization_report={"state_hint": "INVALID"},
                iteration_index=0,
            )

    def test_merge_hypothesis_feedback(self) -> None:
        report = _make_opt_report(
            candidate_records=[_make_candidate("c1", 0.5, effective_improvement=True)]
        )
        memory = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )
        hypothesis_report = {
            "state_hint": "HYPOTHESIS_LAB_STRATEGY_FEEDBACK_INTEGRATED",
            "feedback": {
                "blocked_families": ["conv1d_residual"],
                "search_guidance": [
                    {"guidance_type": "try_spectral", "reason": "test"}
                ],
            },
        }
        merged = merge_training_memory_with_hypothesis_feedback(
            memory=memory,
            hypothesis_report=hypothesis_report,
        )
        assert merged["family_outcomes"]["conv1d_residual"]["status"] == "blocked"
        assert any(
            g["guidance_type"] == "try_spectral"
            for g in merged.get("search_guidance", [])
        )

    def test_load_memory_missing_file(self) -> None:
        result = load_training_iteration_memory("/nonexistent/path.json")
        assert result is None

    def test_load_memory_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        path.write_text(json.dumps({"state_hint": "FNO_TRAINING_MEMORY_BUILT", "iteration_index": 5}))
        result = load_training_iteration_memory(path)
        assert result is not None
        assert result["iteration_index"] == 5


class TestConstraintPropagation:
    def test_blocked_families_filtered_from_plan(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        out_path = tmp_path / "plan.json"
        plan = plan_fno_training_candidates(
            evidence_paths=[],
            output_path=str(out_path),
            max_candidates=10,
            modes=[4, 8],
            widths=[16, 32],
            depths=[1],
            loss_types=["mse"],
            forecast_steps_values=[12],
            train_steps_values=[2],
            blocked_families=["fno_lite"],
            force=True,
        )
        families = {c.get("model_family") for c in plan.get("candidate_configs", [])}
        assert "fno_lite" not in families
        assert plan.get("applied_constraints", {}).get("blocked_families") == ["fno_lite"]

    def test_budget_adjustment_applied(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        out_path = tmp_path / "plan.json"
        plan = plan_fno_training_candidates(
            evidence_paths=[],
            output_path=str(out_path),
            max_candidates=10,
            modes=[4, 8],
            widths=[16, 32],
            depths=[1],
            loss_types=["mse"],
            forecast_steps_values=[12],
            train_steps_values=[2, 4],
            budget_adjustment={"train_steps_delta": 2, "width_delta": 8},
            force=True,
        )
        configs = plan.get("candidate_configs", [])
        assert len(configs) > 0
        for config in configs:
            assert config["train_steps"] in {4, 6}
            assert config["width"] in {24, 40}
        assert plan.get("applied_constraints", {}).get("train_steps_delta") == 2

    def test_operator_policy_guidance_changes_loss_and_budget(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        out_path = tmp_path / "policy_plan.json"
        plan = plan_fno_training_candidates(
            evidence_paths=[],
            output_path=str(out_path),
            max_candidates=4,
            modes=[4],
            widths=[16],
            depths=[1],
            loss_types=["mse"],
            forecast_steps_values=[12],
            train_steps_values=[2],
            search_guidance=[
                {
                    "guidance_type": "operator_training_policy",
                    "required_axis_changes": ["loss_type", "train_steps"],
                    "preferred_loss_types": [
                        "early_weighted_mse",
                        "horizon_weighted_mse",
                    ],
                    "budget_adjustments": {"train_steps_delta": 2},
                    "reason_codes": ["loss_score_mismatch"],
                }
            ],
            force=True,
        )

        assert plan["applied_constraints"]["operator_policy"]["required_axis_changes"] == [
            "loss_type",
            "train_steps",
        ]
        assert plan["applied_constraints"]["train_steps_delta"] == 2
        assert {item["loss_type"] for item in plan["candidate_configs"]} & {
            "early_weighted_mse",
            "horizon_weighted_mse",
        }
        assert all(item["train_steps"] >= 4 for item in plan["candidate_configs"])

    def test_operator_policy_guidance_can_request_new_model_family(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        plan = plan_fno_training_candidates(
            evidence_paths=[],
            output_path=tmp_path / "new_family_plan.json",
            max_candidates=3,
            modes=[4, 8],
            widths=[16, 32],
            depths=[1],
            loss_types=["early_weighted_mse"],
            forecast_steps_values=[12],
            train_steps_values=[10],
            search_guidance=[
                {
                    "guidance_type": "operator_training_policy",
                    "next_action": "request_new_dynamics_source",
                    "required_axis_changes": ["model_family"],
                    "preferred_model_families": ["fno_ar1", "conv1d_residual"],
                    "reason_codes": ["short_horizon_dynamics_gap"],
                }
            ],
            force=True,
        )

        families = {item["model_family"] for item in plan["candidate_configs"]}
        assert families & {"fno_ar1", "conv1d_residual"}
        assert plan["applied_constraints"]["operator_policy"]["next_action"] == (
            "request_new_dynamics_source"
        )

    def test_operator_policy_model_family_axis_forces_diversified_families(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        plan = plan_fno_training_candidates(
            evidence_paths=[],
            output_path=tmp_path / "diversified_family_plan.json",
            max_candidates=3,
            modes=[4],
            widths=[16],
            depths=[1],
            loss_types=["mse"],
            forecast_steps_values=[12],
            train_steps_values=[2],
            search_guidance=[
                {
                    "guidance_type": "operator_training_policy",
                    "next_action": "expand_search_radius",
                    "required_axis_changes": ["model_family"],
                    "preferred_model_families": ["fno_lite"],
                    "reason_codes": ["stagnation"],
                }
            ],
            force=True,
        )

        families = [item["model_family"] for item in plan["candidate_configs"]]
        assert families[0] != "fno_lite"
        assert {"fno_ar1", "conv1d_residual"} <= set(families)
        for item in plan["candidate_configs"]:
            assert isinstance(item["modes"], int)
            assert isinstance(item["width"], int)
            assert isinstance(item["depth"], int)
            assert isinstance(item["forecast_steps"], int)
            assert isinstance(item["train_steps"], int)

    def test_operator_policy_repair_regression_uses_anchor_config(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        evidence_path = tmp_path / "best_evidence.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                    "selected_candidate": {
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 8,
                            "width": 32,
                            "depth": 1,
                            "loss_type": "mse",
                            "normalization": "none",
                            "forecast_steps": 20,
                            "train_steps": 4,
                        },
                        "fno_selection_score": 55.9,
                    },
                    "candidate_records": [],
                }
            )
        )
        plan = plan_fno_training_candidates(
            evidence_paths=[evidence_path],
            output_path=tmp_path / "repair_plan.json",
            max_candidates=3,
            modes=[4, 8],
            widths=[16, 32],
            depths=[1],
            loss_types=["mse", "early_weighted_mse"],
            forecast_steps_values=[12, 20],
            train_steps_values=[2, 4],
            search_guidance=[
                {
                    "guidance_type": "operator_training_policy",
                    "next_action": "repair_regression",
                    "required_axis_changes": ["capacity"],
                    "reason_codes": ["score_regression"],
                }
            ],
            force=True,
        )

        assert plan["applied_constraints"]["operator_policy"]["next_action"] == (
            "repair_regression"
        )
        assert any(
            item["modes"] == 8
            and item["width"] == 32
            and item["forecast_steps"] == 20
            and item["train_steps"] == 4
            and item["loss_type"] == "early_weighted_mse"
            and "regression_repair_anchor_capacity" in item["rationale"]
            for item in plan["candidate_configs"]
        )

    def test_operator_policy_repair_regression_can_use_memory_anchor(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        memory_path = tmp_path / "memory.json"
        memory_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "best_so_far": {
                        "score": 55.9,
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 8,
                            "width": 32,
                            "depth": 1,
                            "loss_type": "mse",
                            "normalization": "none",
                            "forecast_steps": 20,
                            "train_steps": 4,
                        },
                    },
                }
            )
        )

        plan = plan_fno_training_candidates(
            evidence_paths=[memory_path],
            output_path=tmp_path / "memory_repair_plan.json",
            max_candidates=3,
            modes=[4, 8],
            widths=[16, 32],
            depths=[1],
            loss_types=["mse", "early_weighted_mse"],
            forecast_steps_values=[12, 20],
            train_steps_values=[2, 4],
            search_guidance=[
                {
                    "guidance_type": "operator_training_policy",
                    "next_action": "repair_regression",
                    "required_axis_changes": ["capacity"],
                    "reason_codes": ["score_regression"],
                }
            ],
            force=True,
        )

        assert any(
            item["modes"] == 8
            and item["width"] == 32
            and item["forecast_steps"] == 20
            and item["train_steps"] == 4
            and item["loss_type"] == "early_weighted_mse"
            and "regression_repair_anchor_capacity" in item["rationale"]
            for item in plan["candidate_configs"]
        )

    def test_planner_uses_training_memory_anchor_preferences(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        memory_path = tmp_path / "memory_with_anchor.json"
        memory_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "best_so_far": {"score": 59.7},
                    "training_memory_anchors": [
                        {
                            "anchor_id": "training_anchor_fno_ar1",
                            "anchor_type": "operator_training_skill_candidate",
                            "status": "promoted_from_training_memory",
                            "model_family": "fno_ar1",
                            "loss_type": "early_weighted_mse",
                            "score": 59.7,
                            "config": {
                                "model_family": "fno_ar1",
                                "loss_type": "early_weighted_mse",
                                "forecast_steps": 12,
                                "train_steps": 4,
                                "modes": 8,
                                "width": 40,
                                "depth": 1,
                                "normalization": "none",
                            },
                        }
                    ],
                }
            )
        )

        plan = plan_fno_training_candidates(
            evidence_paths=[memory_path],
            output_path=tmp_path / "anchor_plan.json",
            max_candidates=3,
            modes=[4, 8],
            widths=[16, 32],
            depths=[1],
            loss_types=["mse", "early_weighted_mse"],
            forecast_steps_values=[12, 20],
            train_steps_values=[2, 4],
            force=True,
        )

        assert plan["planning_preferences"]["best_selected_config"]["model_family"] == (
            "fno_ar1"
        )
        assert any(
            item["model_family"] == "fno_ar1"
            and item["loss_type"] == "early_weighted_mse"
            and item["width"] == 40
            for item in plan["candidate_configs"]
        )
        assert any(
            "training_memory_anchor_neighborhood" in item["rationale"]
            for item in plan["candidate_configs"]
        )
        assert any(
            item["model_family"] != "fno_ar1"
            and "training_memory_anchor_family_challenge" in item["rationale"]
            for item in plan["candidate_configs"]
        )

    def test_constraint_propagation_in_iteration(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import run_fno_training_iteration

        plan = {
            "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
            "candidate_configs": [
                {
                    "candidate_id": "c1",
                    "model_family": "fno_lite",
                    "modes": 4,
                    "width": 16,
                    "depth": 1,
                    "loss_type": "mse",
                    "normalization": "none",
                    "forecast_steps": 12,
                    "train_steps": 2,
                }
            ],
        }
        opt_report = {
            "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
            "candidate_records": [
                {
                    "candidate_id": "c1",
                    "score": 0.5,
                    "model_family": "fno_lite",
                    "loss_type": "mse",
                    "config": {"modes": 4, "width": 16},
                }
            ],
            "selected_candidate": {
                "candidate_id": "c1",
                "score": 0.5,
                "model_family": "fno_lite",
            },
        }
        memory_path = tmp_path / "memory.json"
        memory_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "iteration_index": 0,
                    "best_so_far": {"score": 0.3},
                    "family_outcomes": {
                        "conv1d_residual": {
                            "status": "blocked",
                            "attempt_count": 2,
                        }
                    },
                    "search_guidance": [
                        {"guidance_type": "try_spectral", "reason": "test"}
                    ],
                    "convergence_history": {
                        "no_improvement_count": 3,
                        "best_score_history": [0.5, 0.4, 0.3],
                    },
                }
            )
        )

        with patch(
            "pde_agent.core.fno_training_closed_loop.plan_fno_training_candidates",
            return_value=plan,
        ) as mock_plan, patch(
            "pde_agent.core.fno_training_closed_loop.run_fno_training_optimization_loop",
            return_value=opt_report,
        ):
            result = run_fno_training_iteration(
                workspace_dir=tmp_path,
                source_path=tmp_path / "dummy.h5",
                memory_path=memory_path,
                max_candidates=1,
                modes=[4],
                widths=[16],
                depths=[1],
                loss_types=["mse"],
                forecast_steps_values=[12],
                train_steps_values=[2],
                force=True,
            )
            assert result["status"] == "ok"
            call_kwargs = mock_plan.call_args.kwargs
            assert call_kwargs.get("blocked_families") == ["conv1d_residual"]
            assert call_kwargs.get("budget_adjustment") == {"train_steps_delta": 2}


class TestConvergence:
    def test_target_score_met(self) -> None:
        memory = build_training_iteration_memory(
            optimization_report=_make_opt_report(
                candidate_records=[_make_candidate("c1", 1.5, effective_improvement=True)]
            ),
            iteration_index=0,
        )
        result = check_training_convergence(
            memory=memory,
            target_score=1.0,
        )
        assert result["decision"] == "converged"
        assert result["confidence"] == 0.95

    def test_single_iteration_candidate_comparison_is_not_doom_loop(self) -> None:
        memory = {
            "state_hint": "FNO_TRAINING_MEMORY_BUILT",
            "best_so_far": {"score": 55.2},
            "convergence_history": {"best_score_history": [55.2]},
            "family_outcomes": {"fno_lite": {"status": "positive_signal"}},
            "trajectory_summary": [
                {"model_family": "fno_lite", "score": 55.2},
                {"model_family": "fno_lite", "score": 55.0},
                {"model_family": "fno_lite", "score": 55.1},
            ],
        }
        result = check_training_convergence(memory=memory)

        assert result["decision"] == "continue"

    def test_patience_exhausted(self) -> None:
        records = [
            _make_candidate("c1", 0.5, model_family="fno_lite", effective_improvement=True),
            _make_candidate("c2", 0.4, model_family="conv1d_residual", effective_improvement=False),
            _make_candidate("c3", 0.3, model_family="fno_ar1", effective_improvement=False),
        ]
        previous_memory = {
            "state_hint": "FNO_TRAINING_MEMORY_BUILT",
            "iteration_index": 2,
            "best_so_far": {"score": 0.6},
            "convergence_history": {
                "no_improvement_count": 2,
                "best_score_history": [0.8, 0.7, 0.6],
            },
        }
        memory = build_training_iteration_memory(
            optimization_report=_make_opt_report(candidate_records=records),
            iteration_index=3,
            previous_memory=previous_memory,
        )
        result = check_training_convergence(
            memory=memory,
            patience=2,
        )
        assert result["decision"] == "stagnant"
        assert any(
            a["action"] == "expand_search_radius"
            for a in result["recommended_actions"]
        )

    def test_single_iteration_non_improving_family_continues(self) -> None:
        records = [
            _make_candidate("c1", 0.5, model_family="fno_lite", effective_improvement=False),
            _make_candidate("c2", 0.51, model_family="fno_lite", effective_improvement=False),
            _make_candidate("c3", 0.52, model_family="fno_lite", effective_improvement=False),
        ]
        memory = build_training_iteration_memory(
            optimization_report=_make_opt_report(candidate_records=records),
            iteration_index=0,
        )
        result = check_training_convergence(
            memory=memory,
            min_score_delta=0.1,
        )
        assert result["decision"] == "continue"

    def test_all_families_stagnant(self) -> None:
        records = [
            _make_candidate("c1", -0.1, model_family="fno_lite", effective_improvement=False),
            _make_candidate("c2", -0.2, model_family="conv1d_residual", effective_improvement=False),
        ]
        memory = build_training_iteration_memory(
            optimization_report=_make_opt_report(candidate_records=records),
            iteration_index=0,
        )
        result = check_training_convergence(memory=memory)
        assert result["decision"] == "stagnant"
        assert result["reason"] == "all_families_stagnant"

    def test_continue(self) -> None:
        records = [
            _make_candidate("c1", 0.5, model_family="fno_lite", effective_improvement=True),
        ]
        memory = build_training_iteration_memory(
            optimization_report=_make_opt_report(candidate_records=records),
            iteration_index=0,
        )
        result = check_training_convergence(memory=memory)
        assert result["decision"] == "continue"

    def test_invalid_memory(self) -> None:
        with pytest.raises(FnoTrainingConvergenceError):
            check_training_convergence(memory={"state_hint": "INVALID"})

    def test_doom_loop_detection(self) -> None:
        trajectory = [
            _make_candidate("c1", 0.5, model_family="fno_lite"),
            _make_candidate("c2", 0.51, model_family="fno_lite"),
            _make_candidate("c3", 0.52, model_family="fno_lite"),
        ]
        result = check_training_doom_loop(
            trajectory=trajectory,
            best_score_history=[0.6, 0.5, 0.51, 0.52],
            no_improvement_count=2,
            min_score_delta=0.1,
        )
        assert result["guard_active"] is True
        assert result["triggers"][0]["trigger"] == "repeated_non_improving_family"

    def test_overfitting_detected(self) -> None:
        curves = {
            "train": [1.0, 0.8, 0.6, 0.5],
            "val": [1.0, 0.9, 0.95, 1.1],
        }
        assert detect_overfitting(loss_curves=curves) is True

    def test_no_overfitting(self) -> None:
        curves = {
            "train": [1.0, 0.8, 0.6],
            "val": [1.0, 0.9, 0.8],
        }
        assert detect_overfitting(loss_curves=curves) is False


class TestCheckpoint:
    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        memory = {"state_hint": "FNO_TRAINING_MEMORY_BUILT", "iteration_index": 2}
        convergence = {"decision": "continue"}
        path = checkpoint_training_loop(
            workspace_dir=tmp_path,
            iteration_index=2,
            memory=memory,
            convergence=convergence,
            best_config={"modes": 8},
            force=True,
        )
        assert path.exists()
        state = resume_training_loop(checkpoint_path=path)
        assert state["iteration_index"] == 2
        assert state["memory"]["iteration_index"] == 2
        assert state["best_config"]["modes"] == 8

    def test_checkpoint_no_overwrite(self, tmp_path: Path) -> None:
        checkpoint_training_loop(
            workspace_dir=tmp_path,
            iteration_index=1,
            memory={},
            convergence={},
            force=True,
        )
        with pytest.raises(FnoTrainingCheckpointError):
            checkpoint_training_loop(
                workspace_dir=tmp_path,
                iteration_index=1,
                memory={},
                convergence={},
                force=False,
            )

    def test_resume_missing_file(self) -> None:
        with pytest.raises(FnoTrainingCheckpointError):
            resume_training_loop(checkpoint_path="/nonexistent.json")

    def test_list_checkpoints(self, tmp_path: Path) -> None:
        checkpoint_training_loop(
            workspace_dir=tmp_path,
            iteration_index=1,
            memory={},
            convergence={},
            force=True,
        )
        checkpoint_training_loop(
            workspace_dir=tmp_path,
            iteration_index=2,
            memory={},
            convergence={},
            force=True,
        )
        checkpoints = list_training_checkpoints(workspace_dir=tmp_path)
        assert len(checkpoints) == 2


class TestAgentInterface:
    def test_iteration_report_structure(self) -> None:
        iteration_result = {
            "iteration_index": 3,
            "summary": {
                "best_candidate_id": "c1",
                "best_score": 0.9,
                "score_delta_vs_previous": 0.1,
                "model_family": "fno_lite",
                "config": {"modes": 8},
            },
            "convergence": {
                "decision": "continue",
                "reason": "room_for_improvement",
                "recommended_actions": [{"action": "continue"}],
            },
            "memory": {},
            "artifacts": {
                "optimization_report_path": "/tmp/opt.json",
                "memory_path": "/tmp/mem.json",
                "convergence_path": "/tmp/conv.json",
            },
        }
        report = build_fno_training_iteration_report(
            iteration_result=iteration_result,
        )
        assert report["state_hint"] == "FNO_TRAINING_ITERATION_REPORTED"
        assert report["iteration_index"] == 3
        assert report["summary"]["best_score"] == 0.9
        assert "agent_directives" in report
        assert len(report["agent_directives"]) > 0

    def test_contract_structure(self, tmp_path: Path) -> None:
        contract = build_fno_training_closed_loop_contract(
            workspace_dir=tmp_path / "ws",
            source_path=tmp_path / "source.h5",
        )
        assert contract["state_hint"] == "FNO_TRAINING_CONTRACT_BUILT"
        assert "required_commands" in contract
        assert "tool_policy" in contract
        assert "bash_policy" in contract
        assert "checklist" in contract

    def test_contract_output_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "contract.json"
        build_fno_training_closed_loop_contract(
            workspace_dir=tmp_path / "ws",
            source_path=tmp_path / "source.h5",
            output_path=output_path,
        )
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["state_hint"] == "FNO_TRAINING_CONTRACT_BUILT"

    def test_contract_no_overwrite(self, tmp_path: Path) -> None:
        output_path = tmp_path / "contract.json"
        output_path.write_text("{}")
        with pytest.raises(FnoTrainingAgentInterfaceError):
            build_fno_training_closed_loop_contract(
                workspace_dir=tmp_path / "ws",
                source_path=tmp_path / "source.h5",
                output_path=output_path,
            )


class TestClosedLoopSmoke:
    def test_single_iteration_mocked(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import evaluate_fno_training_iteration

        opt_report = _make_opt_report(
            candidate_records=[
                _make_candidate("c1", 0.5, model_family="fno_lite"),
            ]
        )
        eval_report = evaluate_fno_training_iteration(opt_report=opt_report)
        assert eval_report["state_hint"] == "FNO_TRAINING_ITERATION_EVALUATED"
        assert eval_report["best_candidate_id"] == "c1"
        assert eval_report["best_score"] == 0.5

    def test_evaluate_invalid_hint(self) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            FnoTrainingClosedLoopError,
            evaluate_fno_training_iteration,
        )

        with pytest.raises(FnoTrainingClosedLoopError):
            evaluate_fno_training_iteration(opt_report={"state_hint": "INVALID"})

    def test_run_iteration_mocked(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import run_fno_training_iteration

        plan = {
            "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
            "candidate_configs": [
                {
                    "candidate_id": "c1",
                    "model_family": "fno_lite",
                    "modes": 4,
                    "width": 16,
                    "depth": 1,
                    "loss_type": "mse",
                    "normalization": "none",
                    "forecast_steps": 12,
                    "train_steps": 2,
                }
            ],
        }
        opt_report = {
            "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
            "candidate_records": [
                {
                    "candidate_id": "c1",
                    "score": 0.5,
                    "model_family": "fno_lite",
                    "loss_type": "mse",
                    "config": {"modes": 4, "width": 16},
                }
            ],
            "selected_candidate": {
                "candidate_id": "c1",
                "score": 0.5,
                "model_family": "fno_lite",
            },
        }

        with patch(
            "pde_agent.core.fno_training_closed_loop.plan_fno_training_candidates",
            return_value=plan,
        ) as mock_plan, patch(
            "pde_agent.core.fno_training_closed_loop.run_fno_training_optimization_loop",
            return_value=opt_report,
        ) as mock_opt:
            result = run_fno_training_iteration(
                workspace_dir=tmp_path,
                source_path=tmp_path / "dummy.h5",
                max_candidates=1,
                modes=[4],
                widths=[16],
                depths=[1],
                loss_types=["mse"],
                forecast_steps_values=[12],
                train_steps_values=[2],
                force=True,
            )
            assert result["state_hint"] == "FNO_TRAINING_ITERATION_REPORTED"
            assert result["status"] == "ok"
            mock_plan.assert_called_once()
            mock_opt.assert_called_once()

    def test_run_iteration_with_memory(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import run_fno_training_iteration

        plan = {
            "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
            "candidate_configs": [
                {
                    "candidate_id": "c1",
                    "model_family": "fno_lite",
                    "modes": 4,
                    "width": 16,
                    "depth": 1,
                    "loss_type": "mse",
                    "normalization": "none",
                    "forecast_steps": 12,
                    "train_steps": 2,
                }
            ],
        }
        opt_report = {
            "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
            "candidate_records": [
                {
                    "candidate_id": "c1",
                    "score": 0.5,
                    "model_family": "fno_lite",
                    "loss_type": "mse",
                    "config": {"modes": 4, "width": 16},
                }
            ],
            "selected_candidate": {
                "candidate_id": "c1",
                "score": 0.5,
            },
        }
        memory_path = tmp_path / "memory.json"
        memory_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "iteration_index": 0,
                    "best_so_far": {"score": 0.3},
                }
            )
        )

        with patch(
            "pde_agent.core.fno_training_closed_loop.plan_fno_training_candidates",
            return_value=plan,
        ), patch(
            "pde_agent.core.fno_training_closed_loop.run_fno_training_optimization_loop",
            return_value=opt_report,
        ):
            result = run_fno_training_iteration(
                workspace_dir=tmp_path,
                source_path=tmp_path / "dummy.h5",
                memory_path=memory_path,
                max_candidates=1,
                modes=[4],
                widths=[16],
                depths=[1],
                loss_types=["mse"],
                forecast_steps_values=[12],
                train_steps_values=[2],
                force=True,
            )
            assert result["metrics"]["iteration_index"] == 1

    def test_run_iteration_passes_decision_policy_to_planner(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import run_fno_training_iteration

        plan = {
            "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
            "candidate_configs": [
                {
                    "candidate_id": "c1",
                    "model_family": "fno_lite",
                    "modes": 4,
                    "width": 16,
                    "depth": 1,
                    "loss_type": "early_weighted_mse",
                    "normalization": "none",
                    "forecast_steps": 12,
                    "train_steps": 4,
                }
            ],
        }
        opt_report = {
            "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
            "candidate_records": [
                {
                    "candidate_id": "c1",
                    "score": 0.5,
                    "model_family": "fno_lite",
                    "config": {
                        "model_family": "fno_lite",
                        "loss_type": "early_weighted_mse",
                        "forecast_steps": 12,
                        "train_steps": 4,
                    },
                }
            ],
            "selected_candidate": {
                "candidate_id": "c1",
                "score": 0.5,
                "config": {
                    "model_family": "fno_lite",
                    "loss_type": "early_weighted_mse",
                    "forecast_steps": 12,
                    "train_steps": 4,
                },
            },
        }
        memory_path = tmp_path / "memory.json"
        memory_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "iteration_index": 0,
                    "best_so_far": {"score": 0.3},
                    "decision_policy": {
                        "budget_adjustments": {"train_steps_delta": 2},
                    },
                    "search_guidance": [
                        {
                            "guidance_type": "operator_training_policy",
                            "preferred_loss_types": ["early_weighted_mse"],
                        }
                    ],
                    "convergence_history": {"no_improvement_count": 0},
                }
            )
        )

        with patch(
            "pde_agent.core.fno_training_closed_loop.plan_fno_training_candidates",
            return_value=plan,
        ) as mock_plan, patch(
            "pde_agent.core.fno_training_closed_loop.run_fno_training_optimization_loop",
            return_value=opt_report,
        ):
            run_fno_training_iteration(
                workspace_dir=tmp_path,
                source_path=tmp_path / "dummy.h5",
                memory_path=memory_path,
                max_candidates=1,
                modes=[4],
                widths=[16],
                depths=[1],
                loss_types=["mse"],
                forecast_steps_values=[12],
                train_steps_values=[2],
                force=True,
            )

        kwargs = mock_plan.call_args.kwargs
        assert kwargs["budget_adjustment"] == {"train_steps_delta": 2}
        assert kwargs["search_guidance"][0]["guidance_type"] == (
            "operator_training_policy"
        )

    def test_iteration_plan_failure_recovery(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import run_fno_training_iteration

        plan = {
            "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
            "candidate_configs": [
                {
                    "candidate_id": "c1",
                    "model_family": "fno_lite",
                    "modes": 4,
                    "width": 16,
                    "depth": 1,
                    "loss_type": "mse",
                    "normalization": "none",
                    "forecast_steps": 12,
                    "train_steps": 2,
                }
            ],
        }
        opt_report = {
            "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
            "candidate_records": [
                {
                    "candidate_id": "c1",
                    "score": 0.5,
                    "model_family": "fno_lite",
                    "loss_type": "mse",
                    "config": {"modes": 4, "width": 16},
                }
            ],
            "selected_candidate": {
                "candidate_id": "c1",
                "score": 0.5,
                "model_family": "fno_lite",
            },
        }

        call_count = 0

        def flaky_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("plan failed")
            return plan

        with patch(
            "pde_agent.core.fno_training_closed_loop.plan_fno_training_candidates",
            side_effect=flaky_plan,
        ), patch(
            "pde_agent.core.fno_training_closed_loop.run_fno_training_optimization_loop",
            return_value=opt_report,
        ):
            result = run_fno_training_iteration(
                workspace_dir=tmp_path,
                source_path=tmp_path / "dummy.h5",
                max_candidates=1,
                modes=[4],
                widths=[16],
                depths=[1],
                loss_types=["mse"],
                forecast_steps_values=[12],
                train_steps_values=[2],
                force=True,
            )
            assert result["state_hint"] == "FNO_TRAINING_ITERATION_REPORTED"
            assert result["status"] == "ok"
            assert call_count == 2

    def test_iteration_optimization_failure_recovery(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import run_fno_training_iteration

        plan = {
            "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
            "candidate_configs": [
                {
                    "candidate_id": "c1",
                    "model_family": "fno_lite",
                    "modes": 4,
                    "width": 16,
                    "depth": 1,
                    "loss_type": "mse",
                    "normalization": "none",
                    "forecast_steps": 12,
                    "train_steps": 2,
                }
            ],
        }

        with patch(
            "pde_agent.core.fno_training_closed_loop.plan_fno_training_candidates",
            return_value=plan,
        ), patch(
            "pde_agent.core.fno_training_closed_loop.run_fno_training_optimization_loop",
            side_effect=RuntimeError("optimization failed"),
        ):
            result = run_fno_training_iteration(
                workspace_dir=tmp_path,
                source_path=tmp_path / "dummy.h5",
                max_candidates=1,
                modes=[4],
                widths=[16],
                depths=[1],
                loss_types=["mse"],
                forecast_steps_values=[12],
                train_steps_values=[2],
                force=True,
            )
            assert result["state_hint"] == "FNO_TRAINING_ITERATION_REPORTED"
            assert result["status"] == "ok"
            assert result["metrics"]["convergence_decision"] == "error"
            assert "recovery_actions" in result["artifacts"]

    def test_iteration_full_failure(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import run_fno_training_iteration

        with patch(
            "pde_agent.core.fno_training_closed_loop.plan_fno_training_candidates",
            side_effect=RuntimeError("plan failed permanently"),
        ):
            result = run_fno_training_iteration(
                workspace_dir=tmp_path,
                source_path=tmp_path / "dummy.h5",
                max_candidates=1,
                modes=[4],
                widths=[16],
                depths=[1],
                loss_types=["mse"],
                forecast_steps_values=[12],
                train_steps_values=[2],
                force=True,
            )
            assert result["state_hint"] == "FNO_TRAINING_ITERATION_REPORTED"
            assert result["status"] == "ok"
            assert result["metrics"]["convergence_decision"] == "error"
            assert "recovery_actions" in result["artifacts"]


class TestCLICommands:
    def test_build_memory_command(self, tmp_path: Path) -> None:
        from pde_agent.commands.fno_training_closed_loop import (
            build_fno_training_memory_command,
        )

        opt_path = tmp_path / "opt.json"
        opt_path.write_text(json.dumps(_make_opt_report([_make_candidate("c1", 0.5)])))
        out_path = tmp_path / "mem.json"
        result = build_fno_training_memory_command(
            optimization_report_path=str(opt_path),
            iteration_index=0,
            previous_memory_path=None,
            output_path=str(out_path),
            force=True,
        )
        assert result["state_hint"] == "FNO_TRAINING_MEMORY_BUILT"
        assert out_path.exists()

    def test_check_convergence_command(self, tmp_path: Path) -> None:
        from pde_agent.commands.fno_training_closed_loop import (
            check_fno_training_convergence_command,
        )

        mem_path = tmp_path / "mem.json"
        report = _make_opt_report([_make_candidate("c1", 0.5, effective_improvement=True)])
        mem = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )
        mem_path.write_text(json.dumps(mem))
        result = check_fno_training_convergence_command(
            memory_path=str(mem_path),
            target_score=None,
            patience=3,
            output_path=None,
            force=True,
        )
        assert result["state_hint"] == "FNO_TRAINING_CONVERGENCE_CHECKED"

    def test_escalate_command(self, tmp_path: Path) -> None:
        from pde_agent.commands.fno_training_closed_loop import (
            escalate_fno_training_to_hypothesis_command,
        )

        mem_path = tmp_path / "mem.json"
        report = _make_opt_report([_make_candidate("c1", 0.5)])
        mem = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )
        mem_path.write_text(json.dumps(mem))
        out_path = tmp_path / "escalation.json"
        result = escalate_fno_training_to_hypothesis_command(
            workspace_dir=str(tmp_path),
            memory_path=str(mem_path),
            output_path=str(out_path),
            max_strategy_candidates=5,
            force=True,
        )
        assert result["state_hint"] in {
            "FNO_TRAINING_ESCALATION_PIPELINE_RAN",
            "FNO_TRAINING_ESCALATION_CANDIDATES_DISCOVERED",
            "FNO_TRAINING_ESCALATION_NO_NEW_CANDIDATES",
        }
        assert out_path.exists()

    def test_integrate_feedback_command(self, tmp_path: Path) -> None:
        from pde_agent.commands.fno_training_closed_loop import (
            integrate_hypothesis_training_feedback_command,
        )

        mem_path = tmp_path / "mem.json"
        report = _make_opt_report([_make_candidate("c1", 0.5)])
        mem = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )
        mem_path.write_text(json.dumps(mem))
        hyp_path = tmp_path / "hyp.json"
        hyp_path.write_text(
            json.dumps(
                {
                    "state_hint": "HYPOTHESIS_LAB_STRATEGY_FEEDBACK_INTEGRATED",
                    "feedback": {
                        "blocked_families": ["conv1d_residual"],
                    },
                }
            )
        )
        out_path = tmp_path / "merged.json"
        result = integrate_hypothesis_training_feedback_command(
            hypothesis_report_path=str(hyp_path),
            memory_path=str(mem_path),
            output_path=str(out_path),
            force=True,
        )
        assert result["state_hint"] == "HYPOTHESIS_LAB_STRATEGY_FEEDBACK_INTEGRATED"
        assert out_path.exists()


class TestFinalReport:
    def test_build_final_report_collects_best_candidate(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            FINAL_REPORT_STATE_HINT,
            build_fno_training_final_report,
        )

        mem_path = tmp_path / "mem.json"
        report = _make_opt_report([_make_candidate("c1", 0.5)])
        mem = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=2,
        )
        mem_path.write_text(json.dumps(mem))
        out_path = tmp_path / "final_report.json"

        result = build_fno_training_final_report(
            memory_path=str(mem_path),
            workspace_dir=str(tmp_path),
            output_path=str(out_path),
            force=True,
        )

        assert result["state_hint"] == FINAL_REPORT_STATE_HINT
        assert result["best_candidate"]["candidate_id"] == "c1"
        assert result["best_candidate"]["score"] == 0.5
        assert result["provenance"]["iteration_count"] == 3
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["state_hint"] == FINAL_REPORT_STATE_HINT

    def test_build_final_report_missing_memory(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            FnoTrainingClosedLoopError,
            build_fno_training_final_report,
        )

        with pytest.raises(FnoTrainingClosedLoopError) as exc_info:
            build_fno_training_final_report(
                memory_path=str(tmp_path / "nonexistent.json"),
                workspace_dir=str(tmp_path),
                force=True,
            )
        assert "could not load memory" in str(exc_info.value)

    def test_cli_build_final_report_command(self, tmp_path: Path) -> None:
        from pde_agent.commands.fno_training_closed_loop import (
            build_fno_training_final_report_command,
        )

        mem_path = tmp_path / "mem.json"
        report = _make_opt_report([_make_candidate("c1", 0.5)])
        mem = build_training_iteration_memory(
            optimization_report=report,
            iteration_index=0,
        )
        mem_path.write_text(json.dumps(mem))
        out_path = tmp_path / "final_report.json"
        result = build_fno_training_final_report_command(
            workspace_dir=str(tmp_path),
            memory_path=str(mem_path),
            output_path=str(out_path),
            force=True,
        )
        assert result["state_hint"] == "FNO_TRAINING_FINAL_REPORT_BUILT"
        assert out_path.exists()

    def test_health_report_summarizes_closed_loop_exploration(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            build_fno_training_health_report,
        )

        workspace = tmp_path / "health_workspace"
        workspace.mkdir()
        (workspace / "fno_training_plan_0000.json").write_text(
            json.dumps(
                {
                    "candidate_configs": [
                        {
                            "model_family": "fno_lite",
                            "rationale": ["training_memory_anchor_reuse"],
                        },
                        {
                            "model_family": "fno_ar1",
                            "rationale": ["training_memory_anchor_family_challenge"],
                        },
                        {
                            "model_family": "fno_lite",
                            "rationale": ["training_memory_anchor_neighborhood"],
                        },
                    ]
                }
            )
        )
        (workspace / "fno_training_memory_0000.json").write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "iteration_index": 0,
                    "current_best": {"score": 1.0},
                    "best_so_far": {"score": 1.0},
                    "training_memory_anchors": [
                        {"status": "active", "score": 1.0}
                    ],
                }
            )
        )
        (workspace / "fno_training_convergence_0000.json").write_text(
            json.dumps({"decision": "continue"})
        )

        report = build_fno_training_health_report(
            workspace_dir=workspace,
            force=True,
        )

        assert report["state_hint"] == "FNO_TRAINING_HEALTH_REPORT_BUILT"
        assert report["iteration_count"] == 1
        assert report["score"]["global_best"] == 1.0
        assert report["exploration"]["candidate_family_counts"]["fno_lite"] == 2
        assert report["exploration"]["candidate_family_counts"]["fno_ar1"] == 1
        assert report["health"]["status"] == "healthy"
        assert (workspace / "fno_training_health_report.json").exists()
        assert (workspace / "fno_training_health_feedback.json").exists()

    def test_health_feedback_rebalances_family_bias(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            build_fno_training_health_feedback,
        )

        report = {
            "state_hint": "FNO_TRAINING_HEALTH_REPORT_BUILT",
            "workspace_dir": str(tmp_path),
            "health": {
                "status": "needs_attention",
                "issues": [
                    {
                        "code": "convergence_warning",
                        "severity": "high",
                        "evidence": {},
                    }
                ],
            },
            "exploration": {
                "candidate_family_counts": {
                    "fno_lite": 27,
                    "fno_ar1": 1,
                }
            },
        }

        feedback = build_fno_training_health_feedback(health_report=report)

        assert feedback["state_hint"] == "FNO_TRAINING_HEALTH_FEEDBACK_BUILT"
        assert any(
            guidance["next_action"] == "rebalance_candidate_families"
            for guidance in feedback["search_guidance"]
        )
        assert any(
            "fno_lite" not in guidance.get("preferred_model_families", [])
            for guidance in feedback["search_guidance"]
            if guidance["next_action"] == "rebalance_candidate_families"
        )

    def test_planner_consumes_health_feedback_as_family_rebalance(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        feedback_path = tmp_path / "health_feedback.json"
        feedback_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_HEALTH_FEEDBACK_BUILT",
                    "search_guidance": [
                        {
                            "guidance_type": "operator_training_policy",
                            "next_action": "rebalance_candidate_families",
                            "required_axis_changes": ["model_family"],
                            "preferred_model_families": [
                                "fno_ar1",
                                "conv1d_residual",
                            ],
                            "preferred_loss_types": ["early_weighted_mse"],
                            "reason_codes": ["health_family_bias"],
                        }
                    ],
                }
            )
        )

        plan = plan_fno_training_candidates(
            evidence_paths=[feedback_path],
            output_path=tmp_path / "plan.json",
            max_candidates=3,
            modes=[4],
            widths=[16],
            depths=[1],
            loss_types=["mse", "early_weighted_mse"],
            forecast_steps_values=[12],
            train_steps_values=[2],
            force=True,
        )

        families = [item["model_family"] for item in plan["candidate_configs"]]
        assert families[0] in {"fno_ar1", "conv1d_residual"}
        assert {"fno_ar1", "conv1d_residual"} <= set(families)

    def test_health_feedback_repairs_latest_non_improvement(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            build_fno_training_health_feedback,
        )

        report = {
            "state_hint": "FNO_TRAINING_HEALTH_REPORT_BUILT",
            "workspace_dir": str(tmp_path),
            "score": {
                "latest_current_best": 59.1,
                "global_best": 59.2,
            },
            "health": {
                "status": "needs_attention",
                "issues": [
                    {
                        "code": "latest_iteration_not_improving",
                        "severity": "low",
                        "evidence": {},
                    }
                ],
            },
            "exploration": {"candidate_family_counts": {"fno_lite": 6, "fno_ar1": 3}},
        }

        feedback = build_fno_training_health_feedback(health_report=report)

        repair = [
            guidance
            for guidance in feedback["search_guidance"]
            if guidance["next_action"] == "repair_regression"
        ]
        assert repair
        assert repair[0]["reason_codes"] == ["health_latest_iteration_not_improving"]
        assert repair[0]["score_context"] == {
            "latest_current_best": 59.1,
            "global_best": 59.2,
        }

    def test_health_feedback_escalates_blocked_doom_loop_family(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            build_fno_training_health_feedback,
            build_fno_training_health_report,
        )

        workspace = tmp_path / "blocked_family_health_workspace"
        workspace.mkdir()
        for index, score in enumerate([59.2, 58.9, 58.7, 58.6]):
            (workspace / f"fno_training_memory_{index:04d}.json").write_text(
                json.dumps(
                    {
                        "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                        "iteration_index": index,
                        "current_best": {"score": score},
                        "best_so_far": {"score": max(59.2, score)},
                        "training_memory_anchors": [],
                    }
                ),
                encoding="utf-8",
            )
        (workspace / "fno_training_convergence_0000.json").write_text(
            json.dumps({"decision": "continue"}),
            encoding="utf-8",
        )
        (workspace / "fno_training_convergence_0001.json").write_text(
            json.dumps({"decision": "doom_loop", "blocked_families": ["fno_lite"]}),
            encoding="utf-8",
        )

        report = build_fno_training_health_report(
            workspace_dir=workspace,
            force=True,
        )
        feedback = build_fno_training_health_feedback(health_report=report)

        convergence_guidance = [
            item
            for item in feedback["search_guidance"]
            if "health_convergence_warning" in item.get("reason_codes", [])
        ]
        assert convergence_guidance
        assert convergence_guidance[0]["next_action"] == "request_new_dynamics_source"
        assert "fno_lite" not in convergence_guidance[0]["preferred_model_families"]
        assert convergence_guidance[0]["blocked_families"] == ["fno_lite"]
        assert convergence_guidance[0]["hypothesis_lab_requests"][0]["request_type"] == (
            "expand_candidate_family"
        )

    def test_planner_consumes_health_feedback_as_regression_repair(self, tmp_path: Path) -> None:
        from pde_agent.core.task1_short_window_training import plan_fno_training_candidates

        memory_path = tmp_path / "memory.json"
        feedback_path = tmp_path / "health_feedback.json"
        memory_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "best_so_far": {
                        "score": 59.2,
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 12,
                            "width": 48,
                            "depth": 2,
                            "loss_type": "mse",
                            "normalization": "none",
                            "forecast_steps": 47,
                            "train_steps": 100,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        feedback_path.write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_HEALTH_FEEDBACK_BUILT",
                    "search_guidance": [
                        {
                            "guidance_type": "operator_training_policy",
                            "next_action": "repair_regression",
                            "required_axis_changes": ["loss_type"],
                            "preferred_loss_types": ["early_weighted_mse", "mse"],
                            "reason_codes": ["health_latest_iteration_not_improving"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        plan = plan_fno_training_candidates(
            evidence_paths=[memory_path, feedback_path],
            output_path=tmp_path / "repair_from_health_plan.json",
            max_candidates=2,
            modes=[4, 8, 12],
            widths=[16, 32, 48],
            depths=[1, 2],
            loss_types=["mse", "early_weighted_mse"],
            forecast_steps_values=[12, 20, 47],
            train_steps_values=[2, 4, 100],
            force=True,
        )

        assert plan["applied_constraints"]["health_feedback_actions"] == [
            "repair_regression"
        ]
        assert "regression_repair_anchor_capacity" in plan["candidate_configs"][0][
            "rationale"
        ]
        assert any(
            item["modes"] == 12
            and item["width"] == 48
            and item["forecast_steps"] == 47
            and item["train_steps"] == 100
            and item["loss_type"] == "early_weighted_mse"
            and "regression_repair_anchor_capacity" in item["rationale"]
            for item in plan["candidate_configs"]
        )

    def test_health_report_command_returns_payload(self, tmp_path: Path) -> None:
        from pde_agent.commands.fno_training_closed_loop import (
            build_fno_training_health_report_command,
        )

        workspace = tmp_path / "health_command_workspace"
        workspace.mkdir()
        (workspace / "fno_training_memory_0000.json").write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "iteration_index": 0,
                    "current_best": {"score": 1.0},
                    "best_so_far": {"score": 1.0},
                    "training_memory_anchors": [],
                }
            )
        )

        result = build_fno_training_health_report_command(
            workspace_dir=str(workspace),
            output_path=None,
            force=True,
        )

        assert result["state_hint"] == "FNO_TRAINING_HEALTH_REPORT_BUILT"
        assert result["metrics"]["iteration_count"] == 1
        assert result["artifacts"]["health_report_path"].endswith(
            "fno_training_health_report.json"
        )

    def test_health_report_flags_iteration_errors(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            build_fno_training_health_report,
        )

        workspace = tmp_path / "error_health_workspace"
        workspace.mkdir()
        (workspace / "fno_training_memory_0000.json").write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_MEMORY_BUILT",
                    "iteration_index": 0,
                    "best_so_far": {"score": 1.0},
                    "trajectory_summary": [],
                }
            )
        )
        (workspace / "fno_training_convergence_0000.json").write_text(
            json.dumps({"decision": "error"})
        )

        report = build_fno_training_health_report(
            workspace_dir=workspace,
            force=True,
        )

        codes = {item["code"] for item in report["health"]["issues"]}
        assert report["health"]["status"] == "needs_attention"
        assert "iteration_errors_detected" in codes
        assert "no_successful_score_observed" in codes

    def test_iteration_auto_triggers_final_report_when_converged(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_closed_loop import (
            FINAL_REPORT_STATE_HINT,
            run_fno_training_iteration,
        )

        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps({"candidates": []}))

        def fake_plan(*, evidence_paths, output_path, max_candidates, **kwargs):
            Path(output_path).write_text(json.dumps({"candidates": []}))
            return {"candidates": []}

        def fake_opt(*, candidate_plan_path, source_path, output_dir, **kwargs):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            opt_report = _make_opt_report([_make_candidate("c1", 0.9)])
            Path(output_dir, "fno_training_optimization_report.json").write_text(
                json.dumps(opt_report)
            )
            return opt_report

        with (
            patch(
                "pde_agent.core.fno_training_closed_loop.plan_fno_training_candidates",
                side_effect=fake_plan,
            ),
            patch(
                "pde_agent.core.fno_training_closed_loop.run_fno_training_optimization_loop",
                side_effect=fake_opt,
            ),
            patch(
                "pde_agent.core.fno_training_closed_loop.check_training_convergence",
                return_value={
                    "decision": "converged",
                    "confidence": 1.0,
                    "reason": "target reached",
                },
            ),
        ):
            result = run_fno_training_iteration(
                workspace_dir=str(tmp_path),
                source_path=str(tmp_path / "data.h5"),
                memory_path=None,
                force=True,
            )

        assert result["status"] == "ok"
        assert result["metrics"]["convergence_decision"] == "converged"
        assert "final_report_path" in result["artifacts"]
        assert Path(result["artifacts"]["final_report_path"]).exists()
        final_report = json.loads(Path(result["artifacts"]["final_report_path"]).read_text())
        assert final_report["state_hint"] == FINAL_REPORT_STATE_HINT
        assert final_report["best_candidate"]["score"] == 0.9
