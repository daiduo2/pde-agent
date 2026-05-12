from __future__ import annotations

import json
from pathlib import Path


def test_hypothesis_lab_next_policy_expands_after_rejected_new_mechanism(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_policy import build_hypothesis_lab_next_policy

    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "state_hint": "LITERATURE_FAILURE_MEMORY_RECORDED",
                "pareto_learning_events": [
                    {
                        "source": "hypothesis_lab_sandbox",
                        "candidate_family": "task1_front25_dynamics",
                        "candidate_id": "task1_front25_dynamics_k32",
                        "algorithm_candidate_id": "candidate_task1_front25_dynamics_v0",
                        "candidate_key": (
                            "task1_front25_dynamics_v0|factory_candidate_family"
                        ),
                        "decision": "reject_or_revise",
                        "outcome": "rejected",
                        "bottleneck_type": "initial_derivative_mismatch",
                        "action_directives": [
                            "expand_linear_velocity_anchor_search",
                            "prioritize_axis:k",
                        ],
                        "retry_directives": ["revise_candidate_family_or_axis"],
                        "avoid_directives": ["do_not_repeat_without_new_mechanism"],
                        "candidate_params": {"k": 32},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_lab_next_policy(
        failure_memory_path=memory_path,
        output_path=tmp_path / "policy.json",
        force=False,
    )

    policy = report["policy"]
    assert report["state_hint"] == "HYPOTHESIS_LAB_NEXT_POLICY_BUILT"
    assert policy["next_action"] == "expand_new_mechanism"
    assert policy["next_command"] == "generate-task1-candidate-families"
    assert policy["learning_objective"] == "early_segment"
    assert policy["preferred_error_modes"] == [
        "initial_derivative_mismatch",
        "early_segment",
        "transition_window",
    ]
    assert policy["required_constraints"]["avoid_algorithm_candidate_ids"] == [
        "candidate_task1_front25_dynamics_v0"
    ]


def test_hypothesis_lab_next_policy_retries_guarded_local_search(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_policy import build_hypothesis_lab_next_policy

    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "state_hint": "LITERATURE_FAILURE_MEMORY_RECORDED",
                "pareto_learning_events": [
                    {
                        "candidate_family": "task1_early_refinement",
                        "decision": "keep_as_evidence",
                        "outcome": "evidence_only",
                        "bottleneck_type": "phase_shift",
                        "retry_directives": ["continue_guarded_local_search"],
                        "effective_axes": ["early_anchor"],
                        "candidate_pool_attribution": {"axis": "early_anchor"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_lab_next_policy(
        failure_memory_path=memory_path,
        output_path=tmp_path / "policy.json",
        force=False,
    )

    assert report["policy"]["next_action"] == "retry_with_policy_constraints"
    assert report["policy"]["candidate_generator"] == (
        "hypothesis_lab_early_subsegment_patch_search"
    )
    assert report["policy"]["hypothesis_candidate_options"]["focus_axes"] == [
        "early_anchor"
    ]
