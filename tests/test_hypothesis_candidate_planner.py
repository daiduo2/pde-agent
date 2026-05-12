from __future__ import annotations

import json
from pathlib import Path


def test_hypothesis_candidate_planner_uses_guard_kpi_and_registry(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    doom_path = tmp_path / "doom.json"
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "postprocess",
                    "physics_residual_nudge",
                    "sem_window_gate_blend_multi_early_smoke",
                ],
            }
        ),
        encoding="utf-8",
    )
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "best_family_by_score_delta": {
                    "candidate_family": "sem_window_gate_blend_multi_search",
                    "best_score_delta": 0.6,
                    "mean_score_delta": 0.4,
                },
                "candidate_family_score_summary": {
                    "sem_window_gate_blend": {
                        "count": 3,
                        "failed_count": 3,
                        "best_score_delta": -0.01,
                        "mean_score_delta": -0.03,
                    },
                    "fast_baseline_family": {
                        "count": 1,
                        "failed_count": 0,
                        "best_score_delta": 0.08,
                        "mean_score_delta": 0.08,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        doom_loop_report_path=doom_path,
        hypothesis_kpi_report_path=kpi_path,
        preferred_error_modes=["early_segment"],
        max_schedule_length=3,
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_CANDIDATE_PLAN_BUILT"
    assert report["candidate_generator_schedule"][0] == "fast_baseline_family"
    assert "sem_window_gate_blend_multi_search" not in report[
        "candidate_generator_schedule"
    ]
    assert "postprocess" not in report["candidate_generator_schedule"]
    assert report["selection_rationale"][0]["candidate_generator"] == (
        "fast_baseline_family"
    )
    assert report["selection_rationale"][0]["decision"] == "selected"
    assert report["selection_rationale"][0]["signals"]["positive_kpi"] is True
    assert Path(report["candidate_plan_path"]).exists()


def test_hypothesis_candidate_planner_allows_bounded_directive_guard_override(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    doom_path = tmp_path / "doom.json"
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": ["postprocess"],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=doom_path,
        hypothesis_kpi_report_path=None,
        guard_override_candidate_generators=["postprocess"],
        preferred_error_modes=["late_stability"],
        max_schedule_length=1,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"] == ["postprocess"]
    assert report["directive_guard_overrides"] == ["postprocess"]
    assert report["selection_rationale"][0]["candidate_generator"] == "postprocess"
    assert report["selection_rationale"][0]["decision"] == "selected"
    assert report["selection_rationale"][0]["directive_guard_override"] is True


def test_hypothesis_candidate_planner_consumes_repair_policy_summary(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_policy_path = tmp_path / "repair_policy.json"
    repair_policy_path.write_text(
        json.dumps(
            {
                "state_hint": "REPAIR_POLICY_SUMMARY_BUILT",
                "converged_repair_errors": ["learning_constraint_smoke_gate"],
                "nonconverged_repair_errors": [
                    "repair_convergence_consistency"
                ],
                "repair_error_stats": {
                    "learning_constraint_smoke_gate": {
                        "attempt_count": 1,
                        "converged_count": 1,
                        "nonconverged_count": 0,
                        "last_decision": "converged",
                    }
                },
                "recommended_actions": [
                    {
                        "audit_error": "repair_convergence_consistency",
                        "action": (
                            "switch_repair_strategy_or_reload_missing_evidence"
                        ),
                        "reason": "repair_nonconverged",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        repair_policy_summary_path=repair_policy_path,
        preferred_error_modes=[],
        max_schedule_length=2,
        force=False,
    )

    assert report["learning_constraints"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert report["learning_constraints"]["repair_nonconvergence_errors"] == [
        "repair_convergence_consistency"
    ]
    assert report["learning_constraints"]["repair_policy_recommended_actions"] == [
        {
            "audit_error": "repair_convergence_consistency",
            "action": "switch_repair_strategy_or_reload_missing_evidence",
            "reason": "repair_nonconverged",
        }
    ]
    assert "repair_policy_summary" in report["learning_constraints"]["reason_codes"]


def test_hypothesis_candidate_planner_consumes_research_chronicle_repair_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    chronicle_path = tmp_path / "research_chronicle.md"
    chronicle_path.write_text(
        "\n".join(
            [
                "Hermes Research Chronicle",
                "",
                "Repair Policy Memory:",
                "- runtime path=runtime.json runtime_action_count=1 "
                "nonconverged_repair_errors=repair_convergence_consistency "
                "converged_repair_errors=learning_constraint_smoke_gate",
                "- repair_action source=runtime "
                "audit_error=repair_convergence_consistency "
                "action=switch_repair_strategy_or_reload_missing_evidence "
                "reason=repair_nonconverged",
                "- heartbeat_batch=1 candidate_generator=physics_residual_nudge "
                "repair_policy_action_count=1",
                "",
                "Submission Decision:",
            ]
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        research_chronicle_path=chronicle_path,
        preferred_error_modes=[],
        max_schedule_length=2,
        force=False,
    )

    constraints = report["learning_constraints"]
    assert constraints["research_chronicle_repair_policy_memory"] == {
        "path": str(chronicle_path),
        "repair_action_count": 1,
        "nonconverged_repair_errors": ["repair_convergence_consistency"],
        "converged_repair_errors": ["learning_constraint_smoke_gate"],
    }
    assert constraints["repair_policy_recommended_actions"] == [
        {
            "audit_error": "repair_convergence_consistency",
            "action": "switch_repair_strategy_or_reload_missing_evidence",
            "reason": "repair_nonconverged",
        }
    ]
    assert "research_chronicle_repair_policy_memory" in constraints["reason_codes"]


def test_hypothesis_candidate_planner_consumes_auto_repair_chronicle_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    chronicle_path = tmp_path / "research_chronicle.md"
    chronicle_path.write_text(
        "\n".join(
            [
                "Hermes Research Chronicle",
                "",
                "Repair Policy Memory:",
                "- auto_repair path=chain.json auto_repair_decision=passed "
                "memory_bridge_passed=True auto_repair_action_count=1 "
                "nonconverged_repair_errors=none "
                "converged_repair_errors=runtime_memory_bridge_smoke_gate",
                "- repair_action source=auto_repair "
                "audit_error=runtime_memory_bridge_smoke_gate "
                "action=prefer_previously_converged_repair_path "
                "reason=repair_converged",
                "",
                "Submission Decision:",
            ]
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        research_chronicle_path=chronicle_path,
        preferred_error_modes=[],
        max_schedule_length=2,
        force=False,
    )

    constraints = report["learning_constraints"]
    assert constraints["research_chronicle_repair_policy_memory"] == {
        "path": str(chronicle_path),
        "repair_action_count": 1,
        "nonconverged_repair_errors": [],
        "converged_repair_errors": ["runtime_memory_bridge_smoke_gate"],
    }
    assert constraints["repair_convergence_errors"] == [
        "runtime_memory_bridge_smoke_gate"
    ]
    assert constraints["repair_policy_recommended_actions"] == [
        {
            "audit_error": "runtime_memory_bridge_smoke_gate",
            "action": "prefer_previously_converged_repair_path",
            "reason": "repair_converged",
        }
    ]


def test_hypothesis_candidate_planner_falls_back_to_safe_unblocked_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    doom_path = tmp_path / "doom.json"
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": ["postprocess"],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=doom_path,
        hypothesis_kpi_report_path=None,
        preferred_error_modes=[],
        max_schedule_length=2,
        force=False,
    )

    assert report["candidate_generator_schedule"] == [
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
    ]
    assert report["next_command"] == "run-claude-agent-sdk-runtime"


def test_hypothesis_candidate_planner_marks_candidate_space_exhausted(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    doom_path = tmp_path / "doom.json"
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "postprocess",
                    "physics_residual_nudge",
                    "sem_window_gate_blend_multi_early_smoke",
                    "sem_window_gate_blend",
                    "sem_window_blend",
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_blend",
        doom_loop_report_path=doom_path,
        hypothesis_kpi_report_path=None,
        preferred_error_modes=["early_segment"],
        max_schedule_length=3,
        min_viable_candidate_count=2,
        force=False,
    )

    assert report["candidate_generator_schedule"] == ["fast_baseline_family"]
    assert report["decision"] == "candidate_space_exhausted"
    assert report["expansion_required"] is True
    assert report["next_command"] == "run-candidate-expansion-handoff"
    assert report["exhaustion_reason"] == "eligible_candidate_count_below_minimum"
    assert report["expansion_handoff"] == {
        "reason": "candidate_space_exhausted",
        "next_command": "build-local-asset-algorithm-candidates",
        "learning_constraints": {
            "avoid_candidate_families": [],
            "mutation_required_for_families": [],
            "reason_codes": [],
        },
        "recommended_inputs": {
            "local_strategy_library_path": "runs/local_strategy_library.json",
            "checkpoint_paths": [
                "data/pdebench/checkpoints/burgers_FNO.tar",
                "data/pdebench/checkpoints/burgers_Unet.tar",
            ],
            "max_strategy_candidates": 5,
        },
        "follow_up_command": "run-candidate-expansion-handoff",
    }


def test_hypothesis_candidate_planner_excludes_generators_blocked_by_cluster_saturation(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    doom_path = tmp_path / "doom.json"
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "postprocess",
                    "physics_residual_nudge",
                    "fast_baseline_family",
                ],
            }
        ),
        encoding="utf-8",
    )
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "metrics": {"candidate_count": 12, "promotion_rate": 0.0},
                "saturated_candidate_families": ["local_strategy_skill"],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="hypothesis_lab_early_subsegment_patch_search",
        doom_loop_report_path=doom_path,
        hypothesis_kpi_report_path=kpi_path,
        preferred_error_modes=["early_segment", "transition_window"],
        max_schedule_length=3,
        min_viable_candidate_count=2,
        force=False,
    )

    excluded = {
        item["candidate_generator"]: item["reason"]
        for item in report["selection_rationale"]
        if item["decision"] == "excluded"
    }
    assert report["candidate_generator_schedule"] == [
        "hypothesis_lab_early_subsegment_patch_search"
    ]
    assert report["decision"] == "candidate_space_exhausted"
    assert report["expansion_required"] is True
    assert excluded["sem_window_gate_blend_multi_early_smoke"] == (
        "blocked_by_cluster_saturation"
    )
    assert excluded["sem_window_gate_blend"] == "blocked_by_cluster_saturation"
    assert excluded["sem_window_blend"] == "blocked_by_cluster_saturation"
    assert report["learning_constraints"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert report["expansion_handoff"]["reason"] == "candidate_space_exhausted"


def test_hypothesis_candidate_planner_carries_deprioritized_executor_constraints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    doom_path = tmp_path / "doom.json"
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "postprocess",
                    "physics_residual_nudge",
                    "sem_window_gate_blend_multi_early_smoke",
                    "sem_window_gate_blend",
                    "sem_window_blend",
                ],
            }
        ),
        encoding="utf-8",
    )
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "deprioritized_candidate_families": ["bounded_executor"],
                "candidate_family_score_summary": {
                    "bounded_executor": {
                        "count": 1,
                        "failed_count": 1,
                        "best_score_delta": -35.0,
                        "mean_score_delta": -35.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_blend",
        doom_loop_report_path=doom_path,
        hypothesis_kpi_report_path=kpi_path,
        preferred_error_modes=["early_segment"],
        max_schedule_length=3,
        min_viable_candidate_count=2,
        force=False,
    )

    constraints = report["learning_constraints"]
    assert constraints["avoid_candidate_families"] == ["bounded_executor"]
    assert constraints["mutation_required_for_families"] == ["bounded_executor"]
    assert constraints["reason_codes"] == ["deprioritized_completion_trace"]
    assert report["expansion_handoff"]["learning_constraints"] == constraints


def test_hypothesis_candidate_planner_excludes_repeated_failed_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "deprioritized_candidate_families": ["postprocess"],
                "deprioritization_reason_codes": {
                    "postprocess": ["repeated_failed_family"]
                },
                "candidate_family_score_summary": {
                    "postprocess": {
                        "count": 2,
                        "failed_count": 2,
                        "best_score_delta": -0.1,
                        "mean_score_delta": -0.2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        preferred_error_modes=[],
        max_schedule_length=3,
        force=False,
    )

    assert "postprocess" not in report["candidate_generator_schedule"]
    constraints = report["learning_constraints"]
    assert constraints["avoid_candidate_families"] == ["postprocess"]
    assert constraints["mutation_required_for_families"] == ["postprocess"]
    assert constraints["reason_codes"] == ["repeated_failed_family"]
    excluded = next(
        item
        for item in report["selection_rationale"]
        if item["candidate_generator"] == "postprocess"
    )
    assert excluded["decision"] == "excluded"
    assert excluded["reason"] == "blocked_by_learning_constraint"


def test_hypothesis_candidate_planner_carries_expansion_axis_exhaustion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    handoff_path = tmp_path / "candidate_expansion_handoff_report.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CANDIDATE_EXPANSION_HANDOFF_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_candidate_families": ["local_strategy_skill"],
                    "attempted_mutation_types": [
                        "robust_reference_ablation_retry",
                        "source_stability_ablation_retry",
                        "handoff_smoothing_ablation_retry",
                    ],
                    "recommended_next_command": "plan-hypothesis-candidates",
                    "recommended_learning_objective": (
                        "switch_candidate_family_after_axis_exhaustion"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=handoff_path,
        preferred_error_modes=["early_segment"],
        max_schedule_length=2,
        force=False,
    )

    assert report["prior_expansion_exhaustion"] == {
        "exhausted": True,
        "reason": "local_strategy_mutation_axes_exhausted",
        "exhausted_candidate_families": ["local_strategy_skill"],
        "attempted_mutation_types": [
            "robust_reference_ablation_retry",
            "source_stability_ablation_retry",
            "handoff_smoothing_ablation_retry",
        ],
        "recommended_next_command": "plan-hypothesis-candidates",
        "recommended_learning_objective": (
            "switch_candidate_family_after_axis_exhaustion"
        ),
    }
    assert report["learning_constraints"]["avoid_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert report["learning_constraints"]["mutation_required_for_families"] == [
        "local_strategy_skill"
    ]


def test_hypothesis_candidate_planner_does_not_family_block_node_level_exhaustion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CANDIDATE_EXPANSION_HANDOFF_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_candidate_families": [],
                    "exhausted_algorithm_candidate_ids": [
                        "candidate_local_strategy_task1_early_subsegment_patch_v0"
                    ],
                    "exhausted_algorithm_node_ids": [
                        "local_strategy_task1_early_subsegment_patch_v0"
                    ],
                    "attempted_mutation_types": [
                        "robust_reference_ablation_retry",
                        "source_stability_ablation_retry",
                        "handoff_smoothing_ablation_retry",
                    ],
                    "recommended_next_command": "plan-hypothesis-candidates",
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=handoff_path,
        preferred_error_modes=["early_segment"],
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["prior_expansion_exhaustion"]["exhausted"] is True
    assert report["prior_expansion_exhaustion"]["exhausted_candidate_families"] == []
    assert report["prior_expansion_exhaustion"]["exhausted_algorithm_node_ids"] == [
        "local_strategy_task1_early_subsegment_patch_v0"
    ]
    assert "local_strategy_skill" not in report["learning_constraints"][
        "avoid_candidate_families"
    ]
    assert report["learning_constraints"]["avoid_algorithm_candidate_ids"] == [
        "candidate_local_strategy_task1_early_subsegment_patch_v0"
    ]
    assert "mutation_required_for_algorithm_candidate_ids" not in report[
        "learning_constraints"
    ]
    assert report["decision"] == "schedule_candidates"
    assert report["learning_constraints"]["reason_codes"] == [
        "local_strategy_mutation_axes_exhausted"
    ]
    assert report["candidate_generator_schedule"] == [
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
    ]


def test_hypothesis_candidate_planner_exhausted_node_overrides_mutation_required_id(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    candidate_id = "candidate_local_strategy_task1_early_transition_expert_v0"
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CANDIDATE_EXPANSION_HANDOFF_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_candidate_families": [],
                    "exhausted_algorithm_candidate_ids": [candidate_id],
                    "exhausted_algorithm_node_ids": [
                        "local_strategy_task1_early_transition_expert_v0"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "deprioritized_candidate_families": ["local_strategy_skill"],
                "deprioritized_algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": "local_strategy_skill",
                    }
                ],
                "deprioritization_reason_codes_by_candidate": {
                    candidate_id: ["score_regressed"]
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=handoff_path,
        max_schedule_length=1,
        force=False,
    )

    constraints = report["learning_constraints"]
    assert candidate_id in constraints["avoid_algorithm_candidate_ids"]
    assert candidate_id not in constraints.get(
        "mutation_required_for_algorithm_candidate_ids",
        [],
    )


def test_hypothesis_candidate_planner_marks_local_strategy_cluster_saturation(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CANDIDATE_EXPANSION_HANDOFF_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_candidate_families": [],
                    "exhausted_algorithm_candidate_ids": [
                        "candidate_local_strategy_task1_early_subsegment_patch_v0",
                        "candidate_local_strategy_task1_early_transition_expert_v0",
                    ],
                    "exhausted_algorithm_node_ids": [
                        "local_strategy_task1_early_subsegment_patch_v0",
                        "local_strategy_task1_early_transition_expert_v0",
                    ],
                    "attempted_mutation_types": [
                        "robust_reference_ablation_retry",
                        "source_stability_ablation_retry",
                        "handoff_smoothing_ablation_retry",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=handoff_path,
        max_schedule_length=1,
        force=False,
    )

    assert report["cluster_saturation_signals"] == [
        {
            "candidate_family": "local_strategy_skill",
            "exhausted_node_count": 2,
            "exhausted_algorithm_node_ids": [
                "local_strategy_task1_early_subsegment_patch_v0",
                "local_strategy_task1_early_transition_expert_v0",
            ],
            "recommended_policy": "prefer_non_local_strategy_or_untried_sources",
            "signal": "local_strategy_cluster_saturated",
        }
    ]
    assert report["learning_constraints"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]


def test_hypothesis_candidate_planner_consumes_kpi_saturation_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "deprioritized_candidate_families": [],
                "saturated_candidate_families": ["local_strategy_skill"],
                "cluster_saturation_signals": [
                    {
                        "candidate_family": "local_strategy_skill",
                        "exhausted_node_count": 2,
                        "exhausted_algorithm_node_ids": [
                            "local_strategy_task1_early_subsegment_patch_v0",
                            "local_strategy_task1_early_transition_expert_v0",
                        ],
                        "recommended_policy": (
                            "prefer_non_local_strategy_or_untried_sources"
                        ),
                        "signal": "local_strategy_cluster_saturated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        max_schedule_length=1,
        force=False,
    )

    assert report["learning_constraints"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert report["cluster_saturation_signals"][0]["signal"] == (
        "local_strategy_cluster_saturated"
    )


def test_hypothesis_candidate_planner_consumes_kpi_search_guidance(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    guidance = {
        "guidance_type": "escalate_after_guided_search_exhaustion",
        "reason": "memory_guided_specs_failed_to_improve",
        "source": "sem_window_gate_blend_multi_search",
        "recommended_action": "switch_gate_feature_family_or_try_new_dynamics_source",
    }
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "deprioritized_candidate_families": [],
                "persistent_search_guidance": [guidance],
                "search_guidance_action_counts": {
                    "switch_gate_feature_family_or_try_new_dynamics_source": 1
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    constraints = report["learning_constraints"]

    assert constraints["persistent_search_guidance"] == [guidance]
    assert constraints["search_guidance_action_counts"] == {
        "switch_gate_feature_family_or_try_new_dynamics_source": 1
    }
    assert "switch_gate_feature_family_or_try_new_dynamics_source" in constraints[
        "reason_codes"
    ]


def test_hypothesis_candidate_planner_consumes_audit_repair_tasks(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_task = {
        "task_id": "repair_learning_constraint_enforcement",
        "audit_error": "learning_constraint_enforcement",
        "priority": "high",
        "repair_scope": "candidate_plan_and_heartbeat_schedule",
        "recommended_action": (
            "rebuild_candidate_plan_or_adjust_heartbeat_schedule_to_respect_learning_constraints"
        ),
        "next_command": "run-claude-agent-sdk-runtime",
    }
    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [repair_task],
                "metrics": {"repair_task_count": 1},
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=repair_path,
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    constraints = report["learning_constraints"]

    assert constraints["audit_repair_required"] is True
    assert constraints["audit_repair_tasks"] == [repair_task]
    assert "audit_repair:learning_constraint_enforcement" in constraints[
        "reason_codes"
    ]
    assert report["audit_repair_tasks"] == [repair_task]


def test_hypothesis_candidate_planner_uses_audit_repair_policy_for_saturation(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_heartbeat_saturation_policy",
                        "audit_error": "heartbeat_saturation_policy",
                        "priority": "high",
                        "repair_scope": "heartbeat_candidate_generator_policy",
                        "recommended_action": (
                            "switch_blocked_generator_to_registry_allowed_fallback_or_mark_valid_probe"
                        ),
                        "next_command": "run-claude-agent-sdk-runtime",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=repair_path,
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"][0] == "physics_residual_nudge"
    assert report["learning_constraints"]["audit_repair_policies"] == [
        {
            "audit_error": "heartbeat_saturation_policy",
            "preferred_candidate_generators": ["physics_residual_nudge"],
            "avoid_candidate_generators": [
                "sem_window_gate_blend_multi_early_smoke",
                "sem_window_gate_blend",
                "sem_window_blend",
            ],
        }
    ]
    assert report["selection_rationale"][0]["signals"][
        "matches_audit_repair_policy"
    ] is True


def test_hypothesis_candidate_planner_uses_audit_repair_policy_for_guidance_drift(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_persistent_search_guidance_consistency",
                        "audit_error": "persistent_search_guidance_consistency",
                        "priority": "high",
                        "repair_scope": "runtime_memory_bridge",
                        "recommended_action": (
                            "rebuild_kpi_candidate_plan_and_heartbeat_trace_with_consistent_persistent_search_guidance"
                        ),
                        "next_command": "run-claude-agent-sdk-runtime",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=repair_path,
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"][0] == "physics_residual_nudge"
    assert "sem_window_blend" not in report["candidate_generator_schedule"][:2]
    assert report["learning_constraints"]["audit_repair_policies"][0][
        "audit_error"
    ] == "persistent_search_guidance_consistency"


def test_hypothesis_candidate_planner_uses_audit_repair_policy_for_cluster_saturation(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_cluster_saturation_consistency",
                        "audit_error": "cluster_saturation_consistency",
                        "priority": "high",
                        "repair_scope": "candidate_cluster_saturation_memory",
                        "recommended_action": (
                            "rebuild_candidate_plan_from_latest_expansion_handoff_and_kpi_saturation_signals"
                        ),
                        "next_command": "run-claude-agent-sdk-runtime",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_gate_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=repair_path,
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"][0] == "fast_baseline_family"
    assert report["learning_constraints"]["audit_repair_policies"][0] == {
        "audit_error": "cluster_saturation_consistency",
        "preferred_candidate_generators": ["fast_baseline_family"],
        "avoid_candidate_generators": [
            "sem_window_gate_blend_multi_early_smoke",
            "sem_window_gate_blend",
            "sem_window_blend",
        ],
    }


def test_hypothesis_candidate_planner_recovers_active_audit_repair_from_kpi(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "active_audit_repair_errors": [
                    "cluster_saturation_consistency",
                ],
                "audit_repair_error_counts": {
                    "cluster_saturation_consistency": 1,
                },
                "audit_repair_consumed_error_counts": {},
                "metrics": {
                    "audit_repair_task_count": 1,
                    "audit_repair_consumed_count": 0,
                    "active_audit_repair_error_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_gate_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"][0] == "fast_baseline_family"
    assert report["learning_constraints"]["audit_repair_required"] is True
    assert report["learning_constraints"]["audit_repair_tasks"][0]["audit_error"] == (
        "cluster_saturation_consistency"
    )
    assert report["learning_constraints"]["audit_repair_tasks"][0]["source"] == (
        "hypothesis_kpi_report"
    )


def test_hypothesis_candidate_planner_switches_strategy_for_nonconverged_repair(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "active_audit_repair_errors": [
                    "cluster_saturation_consistency",
                ],
                "repair_nonconvergence_errors": [
                    "cluster_saturation_consistency",
                ],
                "chain_learning_signal_counts": {
                    "audit_repair_nonconvergence": 1,
                },
                "metrics": {
                    "active_audit_repair_error_count": 1,
                    "repair_nonconvergence_error_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_gate_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"][0] == "physics_residual_nudge"
    assert report["learning_constraints"]["repair_nonconvergence_errors"] == [
        "cluster_saturation_consistency"
    ]
    assert report["learning_constraints"]["audit_repair_tasks"][0][
        "repair_nonconverged"
    ] is True
    assert report["learning_constraints"]["audit_repair_policies"][0] == {
        "audit_error": "cluster_saturation_consistency",
        "preferred_candidate_generators": ["physics_residual_nudge"],
        "avoid_candidate_generators": [
            "fast_baseline_family",
            "sem_window_gate_blend_multi_early_smoke",
            "sem_window_gate_blend",
            "sem_window_blend",
        ],
        "repair_nonconverged": True,
    }


def test_hypothesis_candidate_planner_preserves_repair_convergence_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "repair_convergence_errors": ["learning_constraint_smoke_gate"],
                "audit_repair_converged_error_counts": {
                    "learning_constraint_smoke_gate": 2,
                },
                "chain_learning_signal_counts": {"audit_repair_convergence": 2},
                "metrics": {
                    "repair_convergence_error_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_gate_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["learning_constraints"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert report["learning_constraints"]["audit_repair_converged_error_counts"] == {
        "learning_constraint_smoke_gate": 2
    }
    assert "audit_repair_convergence" in report["learning_constraints"]["reason_codes"]
    assert "audit_repair_required" not in report["learning_constraints"]


def test_hypothesis_candidate_planner_does_not_reopen_converged_active_repair(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "active_audit_repair_errors": ["learning_constraint_smoke_gate"],
                "repair_convergence_errors": ["learning_constraint_smoke_gate"],
                "audit_repair_converged_error_counts": {
                    "learning_constraint_smoke_gate": 1,
                },
                "metrics": {
                    "active_audit_repair_error_count": 1,
                    "repair_convergence_error_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_gate_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["audit_repair_tasks"] == []
    assert "audit_repair_required" not in report["learning_constraints"]
    assert report["learning_constraints"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]


def test_hypothesis_candidate_planner_switches_strategy_for_nonconverged_guidance_repair(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "active_audit_repair_errors": [
                    "persistent_search_guidance_consistency",
                ],
                "repair_nonconvergence_errors": [
                    "persistent_search_guidance_consistency",
                ],
                "metrics": {
                    "active_audit_repair_error_count": 1,
                    "repair_nonconvergence_error_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_gate_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"][0] == "fast_baseline_family"
    assert report["learning_constraints"]["audit_repair_policies"][0] == {
        "audit_error": "persistent_search_guidance_consistency",
        "preferred_candidate_generators": ["fast_baseline_family"],
        "avoid_candidate_generators": [
            "physics_residual_nudge",
            "sem_window_gate_blend_multi_early_smoke",
            "sem_window_gate_blend",
            "sem_window_blend",
        ],
        "repair_nonconverged": True,
    }


def test_hypothesis_candidate_planner_uses_audit_repair_policy_for_memory_bridge(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_memory_bridge_inputs",
                        "audit_error": "memory_bridge_inputs",
                        "priority": "high",
                        "repair_scope": "runtime_input_memory_bridge",
                        "recommended_action": (
                            "reload_input_expansion_handoff_and_registry_summary_before_candidate_planning"
                        ),
                        "next_command": "run-claude-agent-sdk-runtime",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=repair_path,
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"][0] == "postprocess"
    assert report["learning_constraints"]["audit_repair_policies"][0] == {
        "audit_error": "memory_bridge_inputs",
        "preferred_candidate_generators": ["postprocess"],
        "requires_input_bridge_reload": True,
    }


def test_hypothesis_candidate_planner_marks_learning_smoke_repair_policy(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_learning_constraint_smoke_gate",
                        "audit_error": "learning_constraint_smoke_gate",
                        "priority": "high",
                        "repair_scope": "learning_constraint_smoke_artifact_bridge",
                        "recommended_action": (
                            "rerun_hypothesis_learning_constraint_smoke_and_attach_passed_report_before_runtime_audit"
                        ),
                        "next_command": "run-hypothesis-learning-constraint-smoke",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=repair_path,
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["learning_constraints"]["audit_repair_policies"][0] == {
        "audit_error": "learning_constraint_smoke_gate",
        "preferred_candidate_generators": ["postprocess"],
        "requires_learning_constraint_smoke_reload": True,
    }
    assert "audit_repair:learning_constraint_smoke_gate" in report[
        "learning_constraints"
    ]["reason_codes"]


def test_hypothesis_candidate_planner_switches_strategy_for_nonconverged_memory_bridge(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "active_audit_repair_errors": ["memory_bridge_inputs"],
                "repair_nonconvergence_errors": ["memory_bridge_inputs"],
                "metrics": {
                    "active_audit_repair_error_count": 1,
                    "repair_nonconvergence_error_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"][0] == "fast_baseline_family"
    assert report["learning_constraints"]["audit_repair_policies"][0] == {
        "audit_error": "memory_bridge_inputs",
        "preferred_candidate_generators": ["fast_baseline_family"],
        "avoid_candidate_generators": ["postprocess"],
        "requires_input_bridge_reload": True,
        "repair_nonconverged": True,
    }


def test_hypothesis_candidate_planner_covers_runtime_audit_repair_errors(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_errors = [
        "learning_constraint_enforcement",
        "persistent_search_guidance_consistency",
        "cluster_saturation_consistency",
        "heartbeat_saturation_policy",
        "memory_bridge_inputs",
        "repair_nonconvergence_consistency",
        "repair_convergence_consistency",
    ]
    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": f"repair_{error}",
                        "audit_error": error,
                        "priority": "high",
                        "repair_scope": "runtime_repair",
                        "recommended_action": "repair_failed_runtime_audit_gate",
                        "next_command": "run-claude-agent-sdk-runtime",
                    }
                    for error in repair_errors
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=repair_path,
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=False,
    )

    constraints = report["learning_constraints"]
    policy_errors = {
        policy["audit_error"] for policy in constraints["audit_repair_policies"]
    }

    assert constraints["audit_repair_required"] is True
    assert all(
        f"audit_repair:{error}" in constraints["reason_codes"]
        for error in repair_errors
    )
    assert {
        "persistent_search_guidance_consistency",
        "cluster_saturation_consistency",
        "heartbeat_saturation_policy",
        "memory_bridge_inputs",
        "repair_nonconvergence_consistency",
        "repair_convergence_consistency",
    }.issubset(policy_errors)
    assert "learning_constraint_enforcement" in constraints[
        "generic_audit_repair_errors"
    ]


def test_hypothesis_candidate_planner_covers_nonconverged_runtime_audit_repair_errors(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    repair_errors = [
        "persistent_search_guidance_consistency",
        "cluster_saturation_consistency",
        "memory_bridge_inputs",
    ]
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "active_audit_repair_errors": repair_errors,
                "repair_nonconvergence_errors": repair_errors,
                "metrics": {
                    "active_audit_repair_error_count": len(repair_errors),
                    "repair_nonconvergence_error_count": len(repair_errors),
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_gate_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        max_schedule_length=4,
        min_viable_candidate_count=1,
        force=False,
    )

    policies = report["learning_constraints"]["audit_repair_policies"]
    assert {
        policy["audit_error"] for policy in policies if policy.get("repair_nonconverged")
    } == set(repair_errors)
    assert all(policy.get("repair_nonconverged") is True for policy in policies)
    assert "generic_audit_repair_errors" not in report["learning_constraints"]


def test_hypothesis_candidate_planner_keeps_node_scoped_deprioritization_out_of_family_avoid(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "deprioritized_candidate_families": [
                    "official_checkpoint_family",
                    "local_strategy_skill",
                ],
                "deprioritized_algorithm_candidates": [
                    {
                        "algorithm_candidate_id": (
                            "candidate_local_checkpoint_burgers_fno"
                        ),
                        "candidate_family": "official_checkpoint_family",
                        "reason_codes": [
                            "deprioritized_completion_trace",
                            "score_regressed",
                        ],
                    },
                    {
                        "algorithm_candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "reason_codes": [
                            "deprioritized_completion_trace",
                            "score_regressed",
                        ],
                    },
                ],
                "deprioritization_reason_codes": {
                    "official_checkpoint_family": [
                        "deprioritized_completion_trace",
                        "score_regressed",
                    ],
                    "local_strategy_skill": [
                        "deprioritized_completion_trace",
                        "score_regressed",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="sem_window_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        preferred_error_modes=["early_segment"],
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    constraints = report["learning_constraints"]
    assert constraints["avoid_candidate_families"] == []
    assert constraints["avoid_algorithm_candidate_ids"] == [
        "candidate_local_checkpoint_burgers_fno",
        "candidate_local_strategy_task1_early_subsegment_patch_v0",
    ]
    assert constraints["mutation_required_for_algorithm_candidate_ids"] == [
        "candidate_local_checkpoint_burgers_fno",
        "candidate_local_strategy_task1_early_subsegment_patch_v0",
    ]
    assert report["decision"] == "schedule_candidates"


def test_plan_hypothesis_candidates_cli_returns_json_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    doom_path = tmp_path / "doom.json"
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": ["postprocess"],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "plan.json"

    result = CliRunner().invoke(
        app,
        [
            "plan-hypothesis-candidates",
            "--output-path",
            str(output_path),
            "--candidate-generator",
            "postprocess",
            "--doom-loop-report-path",
            str(doom_path),
            "--preferred-error-mode",
            "early_segment",
            "--min-viable-candidate-count",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HYPOTHESIS_CANDIDATE_PLAN_BUILT"
    assert payload["artifacts"]["candidate_plan_path"] == str(output_path)
    assert payload["metrics"]["candidate_plan_length"] > 0


def test_plan_hypothesis_candidates_cli_accepts_hypothesis_lab_next_policy(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    policy_path = tmp_path / "next_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_LAB_NEXT_POLICY_BUILT",
                "policy": {
                    "next_action": "retry_with_policy_constraints",
                    "next_command": "plan-hypothesis-candidates",
                    "candidate_generator": "hypothesis_lab_front25_dynamics_smoke",
                    "learning_objective": "early_segment",
                    "preferred_error_modes": [
                        "initial_derivative_mismatch",
                        "early_segment",
                    ],
                    "required_constraints": {
                        "avoid_algorithm_candidate_ids": [
                            "candidate_task1_front25_dynamics_v0"
                        ],
                    },
                    "hypothesis_candidate_options": {
                        "candidate_pool_axis": "dynamics_mode",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "plan.json"

    result = CliRunner().invoke(
        app,
        [
            "plan-hypothesis-candidates",
            "--output-path",
            str(output_path),
            "--hypothesis-lab-next-policy-path",
            str(policy_path),
            "--min-viable-candidate-count",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    plan = json.loads(output_path.read_text(encoding="utf-8"))
    assert plan["candidate_generator"] == "hypothesis_lab_front25_dynamics_smoke"
    assert plan["candidate_generator_schedule"][0] == (
        "hypothesis_lab_front25_dynamics_smoke"
    )
    assert plan["preferred_error_modes"] == [
        "initial_derivative_mismatch",
        "early_segment",
    ]
    constraints = plan["hypothesis_lab_policy_constraints"]
    assert constraints["learning_objective"] == "early_segment"
    assert constraints["guard_override_candidate_generator"] == (
        "hypothesis_lab_front25_dynamics_smoke"
    )
    assert payload["artifacts"]["hypothesis_lab_policy_constraints"] == constraints


def test_plan_hypothesis_candidates_cli_accepts_expansion_handoff_report(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CANDIDATE_EXPANSION_HANDOFF_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_candidate_families": ["local_strategy_skill"],
                    "attempted_mutation_types": [
                        "robust_reference_ablation_retry",
                        "source_stability_ablation_retry",
                        "handoff_smoothing_ablation_retry",
                    ],
                    "recommended_next_command": "plan-hypothesis-candidates",
                    "recommended_learning_objective": (
                        "switch_candidate_family_after_axis_exhaustion"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "plan.json"

    result = CliRunner().invoke(
        app,
        [
            "plan-hypothesis-candidates",
            "--output-path",
            str(output_path),
            "--candidate-generator",
            "postprocess",
            "--expansion-handoff-report-path",
            str(handoff_path),
            "--preferred-error-mode",
            "early_segment",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    plan = json.loads(output_path.read_text(encoding="utf-8"))
    assert plan["prior_expansion_exhaustion"]["exhausted"] is True
    assert plan["learning_constraints"]["avoid_candidate_families"] == [
        "local_strategy_skill"
    ]


def test_plan_hypothesis_candidates_cli_accepts_direct_expansion_report(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    expansion_path = tmp_path / "literature_strategy_expansion.json"
    expansion_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_STRATEGY_EXPANSION_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_algorithm_candidate_ids": [
                        "candidate_local_strategy_task1_early_transition_expert_v0"
                    ],
                    "attempted_mutation_types": [
                        "robust_reference_ablation_retry",
                        "source_stability_ablation_retry",
                        "handoff_smoothing_ablation_retry",
                    ],
                    "recommended_next_command": "plan-hypothesis-candidates",
                    "recommended_learning_objective": (
                        "switch_candidate_family_after_axis_exhaustion"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "plan.json"

    result = CliRunner().invoke(
        app,
        [
            "plan-hypothesis-candidates",
            "--output-path",
            str(output_path),
            "--candidate-generator",
            "postprocess",
            "--expansion-report-path",
            str(expansion_path),
            "--preferred-error-mode",
            "early_segment",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    plan = json.loads(output_path.read_text(encoding="utf-8"))
    assert plan["prior_expansion_exhaustion"]["exhausted"] is True
    assert plan["learning_constraints"]["avoid_algorithm_candidate_ids"] == [
        "candidate_local_strategy_task1_early_transition_expert_v0"
    ]


def test_plan_hypothesis_candidates_prefers_factory_dynamics_after_axis_exhaustion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_candidate_planner import (
        build_hypothesis_candidate_plan,
    )

    expansion_path = tmp_path / "literature_strategy_expansion.json"
    expansion_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_STRATEGY_EXPANSION_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_algorithm_candidate_ids": [
                        "candidate_local_strategy_task1_early_transition_expert_v0"
                    ],
                    "attempted_mutation_types": [
                        "robust_reference_ablation_retry",
                        "source_stability_ablation_retry",
                        "handoff_smoothing_ablation_retry",
                    ],
                    "recommended_next_command": "plan-hypothesis-candidates",
                    "recommended_learning_objective": (
                        "switch_candidate_family_after_axis_exhaustion"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_candidate_plan(
        output_path=tmp_path / "plan.json",
        candidate_generator="postprocess",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        expansion_report_path=expansion_path,
        preferred_error_modes=["early_segment", "transition_window"],
        max_schedule_length=2,
        min_viable_candidate_count=1,
        force=False,
    )

    assert report["candidate_generator_schedule"] == [
        "hypothesis_lab_front25_dynamics_smoke",
        "hypothesis_lab_early_subsegment_patch_search",
    ]
    assert report["learning_constraints"]["prefer_candidate_generators"] == [
        "hypothesis_lab_front25_dynamics_smoke",
        "hypothesis_lab_early_subsegment_patch_search",
    ]
