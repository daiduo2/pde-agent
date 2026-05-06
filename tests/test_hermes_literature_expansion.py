from __future__ import annotations

import json
from pathlib import Path


def _write_hdf5(path: Path, data: object, key: str = "tensor") -> Path:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def _blocked_fno_candidates(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "parent_algorithm_card_id": "card_fno",
                        "candidate_family": "unsupported",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "hypothesis": "FNO improves Burgers rollout.",
                        "mechanism": "Fourier kernels learn global modes.",
                        "executor_spec": {
                            "cli_command": "blocked_requires_executor_design",
                            "unsupported_hook": "fourier_neural_operator_architecture",
                            "required_options": [],
                        },
                        "risk_controls": [
                            "blocked until a bounded executor is designed"
                        ],
                        "source_grounding": {
                            "source_ids": ["source_fno"],
                            "algorithm_card_id": "card_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _mixed_candidates(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_residual",
                        "candidate_family": "lightweight_residual_corrector",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "hypothesis": "Residual correction may improve rollout.",
                        "mechanism": "Correct horizon bias.",
                        "executor_spec": {
                            "cli_command": "run-blueprint-sandbox",
                            "required_options": [],
                        },
                        "risk_controls": ["overfit"],
                        "source_grounding": {
                            "source_ids": ["source_residual"],
                            "algorithm_card_id": "card_residual",
                        },
                    },
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "parent_algorithm_card_id": "card_fno",
                        "candidate_family": "unsupported",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "hypothesis": "FNO improves Burgers rollout.",
                        "mechanism": "Fourier kernels learn global modes.",
                        "executor_spec": {
                            "cli_command": "blocked_requires_executor_design",
                            "unsupported_hook": "fourier_neural_operator_architecture",
                            "required_options": [],
                        },
                        "risk_controls": [
                            "blocked until a bounded executor is designed"
                        ],
                        "source_grounding": {
                            "source_ids": ["source_fno"],
                            "algorithm_card_id": "card_fno",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_hermes_literature_expansion_adapts_fno_candidate_to_checkpoint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _blocked_fno_candidates(candidates_path)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        force=False,
    )

    assert report["state_hint"] == "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN"
    assert report["action_count"] == 1
    action = report["actions"][0]
    assert action["action"] == "adapt_to_checkpoint_sandbox"
    assert action["executable"] is True
    assert action["next_command"] == "run-checkpoint-family-sandbox"
    assert Path(action["adapted_candidate_path"]).exists()
    assert Path(action["sandbox_plan_path"]).exists()
    assert report["sandbox_queue"] == [action["sandbox_plan_path"]]
    assert report["strategy_growth_queue"] == []
    assert report["next_command"] == "run-checkpoint-family-sandbox"


def test_hermes_literature_expansion_ranks_checkpoint_adapters_first(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _mixed_candidates(candidates_path)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        force=False,
    )

    rankings = report["candidate_rankings"]
    assert rankings[0]["algorithm_candidate_id"] == "candidate_fno_blocked"
    assert rankings[0]["route"] == "checkpoint_adapter"
    assert rankings[0]["selection_score"] > rankings[1]["selection_score"]
    assert report["actions"][0]["algorithm_candidate_id"] == "candidate_fno_blocked"


def test_hermes_literature_expansion_skips_failed_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _mixed_candidates(candidates_path)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": ["candidate_fno_blocked"],
                "failed_candidate_keys": [],
            }
        ),
        encoding="utf-8",
    )

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        failure_memory_path=memory_path,
        force=False,
    )

    assert report["skipped_count"] == 1
    assert report["skipped_candidates"][0]["algorithm_candidate_id"] == (
        "candidate_fno_blocked"
    )
    assert report["actions"][0]["algorithm_candidate_id"] == "candidate_residual"


def test_hermes_literature_expansion_allows_mutation_despite_parent_failed_key(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    candidates_path = tmp_path / "algorithm_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": (
                            "candidate_local_checkpoint_burgers_fno_"
                            "mutation_finetune_calibration_v0"
                        ),
                        "parent_algorithm_card_id": "local_checkpoint_burgers_fno",
                        "candidate_family": "official_checkpoint_family",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "executor_spec": {
                            "cli_command": "run-checkpoint-family-sandbox"
                        },
                        "resource_bindings": {
                            "checkpoint_paths": [str(checkpoint)]
                        },
                        "source_grounding": {
                            "source_ids": [str(checkpoint)],
                            "algorithm_card_id": "local_checkpoint_burgers_fno",
                        },
                        "mutation_evidence": {
                            "parent_candidate_family": "official_checkpoint_family",
                            "parent_algorithm_candidate_id": (
                                "candidate_local_checkpoint_burgers_fno"
                            ),
                            "mutation_type": "checkpoint_finetune_calibration_retry",
                            "reason": (
                                "responds_to_deprioritized_checkpoint_family"
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [
                    "candidate_local_checkpoint_burgers_fno"
                ],
                "failed_candidate_keys": [
                    "local_checkpoint_burgers_fno|official_checkpoint_family"
                ],
                "failed_candidate_families": [],
                "outcomes": [],
            }
        ),
        encoding="utf-8",
    )

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        failure_memory_path=memory_path,
        force=False,
    )

    assert report["skipped_count"] == 0
    assert report["actions"][0]["algorithm_candidate_id"].endswith(
        "_mutation_finetune_calibration_v0"
    )
    assert report["actions"][0]["sandbox_plan_path"].endswith(
        "_mutation_finetune_calibration_v0_sandbox_plan.json"
    )


def test_hermes_literature_expansion_uses_registry_summary_to_skip_repetition(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "FNO_ARCHITECTURE_CANDIDATE_ADAPTED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_checkpoint_adapter",
                        "candidate_family": "official_checkpoint_family",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "executor_spec": {
                            "cli_command": "run-checkpoint-family-sandbox"
                        },
                        "resource_bindings": {
                            "checkpoint_paths": [str(tmp_path / "burgers_FNO.tar")]
                        },
                        "source_grounding": {
                            "source_ids": ["source_fno"],
                            "algorithm_card_id": "card_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "hermes_registry_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "avoid_repetition": [
                    {
                        "candidate_id": "candidate_checkpoint_adapter",
                        "reason": "sandbox_score_below_current_best",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[],
        registry_summary_path=summary_path,
        force=False,
    )

    assert report["action_count"] == 0
    assert report["skipped_count"] == 1
    assert report["skipped_candidates"][0]["skip_reason"] == (
        "registry_avoid_repetition"
    )
    assert report["next_command"] == "build-local-asset-algorithm-candidates"


def test_hermes_literature_expansion_records_proposal_for_blocked_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _blocked_fno_candidates(candidates_path)

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[],
        force=False,
    )

    action = report["actions"][0]
    assert action["action"] == "record_executor_design_proposal"
    assert action["executable"] is False
    assert action["next_command"] == "build-strategy-rubric"
    proposal = json.loads(Path(action["strategy_proposal_path"]).read_text())
    assert proposal["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert proposal["source_type"] == "external_retrieval"
    assert proposal["implementation_plan"]["executor_design_request_path"] == (
        action["executor_design_request_path"]
    )
    assert report["proposal_count"] == 1
    assert report["sandbox_queue"] == []
    assert report["strategy_growth_queue"] == [action["strategy_proposal_path"]]


def test_record_hermes_literature_candidate_outcome_adds_failure_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        record_hermes_literature_candidate_outcome,
    )

    report = record_hermes_literature_candidate_outcome(
        algorithm_candidate_id="candidate_fno_blocked",
        candidate_key="card_fno|fourier_neural_operator_architecture",
        outcome="failed",
        reason_codes=["sandbox_score_regressed"],
        output_path=tmp_path / "failure_memory.json",
        force=False,
    )

    assert report["state_hint"] == "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED"
    assert report["failed_algorithm_candidate_ids"] == ["candidate_fno_blocked"]
    assert report["failed_candidate_keys"] == [
        "card_fno|fourier_neural_operator_architecture"
    ]


def test_summarize_literature_registry_for_hermes_builds_learning_queue(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        summarize_literature_registry_for_hermes,
    )

    registry_path = tmp_path / "literature_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_UPDATED",
                "sources": [
                    {
                        "source_id": "source_downloaded",
                        "title": "Downloaded Paper",
                        "download_status": "downloaded",
                        "parse_status": "not_parsed",
                    },
                    {
                        "source_id": "source_parsed",
                        "title": "Parsed Paper",
                        "download_status": "downloaded",
                        "parse_status": "usable",
                        "algorithm_card_ids": [],
                    },
                ],
                "algorithm_nodes": [
                    {
                        "node_id": "algorithm_card_checkpoint",
                        "algorithm_card_id": "card_checkpoint",
                        "strategy_status": "sandbox_verified",
                        "candidate_ids": ["candidate_checkpoint_adapter"],
                        "source_ids": ["source_fno"],
                    },
                    {
                        "node_id": "algorithm_card_executor_gap",
                        "algorithm_card_id": "card_executor_gap",
                        "strategy_status": "blocked",
                        "candidate_ids": ["candidate_blocked"],
                        "source_ids": ["source_blocked"],
                    },
                ],
                "candidate_nodes": [
                    {
                        "candidate_id": "candidate_residual",
                        "candidate_family": "lightweight_residual_corrector",
                        "status": "compiled",
                        "executor": "run-blueprint-sandbox",
                        "source_ids": ["source_residual"],
                    },
                    {
                        "candidate_id": "candidate_checkpoint_adapter",
                        "candidate_family": "official_checkpoint_family",
                        "status": "sandbox_verified",
                        "executor": "run-checkpoint-family-sandbox",
                        "source_ids": ["source_fno"],
                        "sandbox_evidence": [
                            {
                                "best_score": 83.6,
                                "rules_passed": True,
                                "inference_time": 5.9,
                                "artifact_path": "checkpoint_report.json",
                            }
                        ],
                    },
                    {
                        "candidate_id": "candidate_blocked",
                        "candidate_family": "unsupported",
                        "status": "blocked",
                        "executor": "blocked_requires_executor_design",
                        "source_ids": ["source_blocked"],
                    },
                ],
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_literature_registry_for_hermes(
        registry_path=registry_path,
        output_path=tmp_path / "hermes_registry_summary.json",
        current_best_score=86.0,
        force=False,
    )

    assert report["state_hint"] == "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT"
    assert report["next_command"] == "run-blueprint-sandbox"
    assert report["source_backlog"][0]["next_command"] == "submit-mineru-parse"
    assert report["source_backlog"][1]["next_command"] == "extract-algorithm-cards"
    actions = report["candidate_action_queue"]
    assert actions[0]["candidate_id"] == "candidate_residual"
    assert actions[0]["recommended_action"] == "run_sandbox"
    assert actions[1]["candidate_id"] == "candidate_checkpoint_adapter"
    assert actions[1]["recommended_action"] == "refine_or_ablate"
    assert actions[1]["score_delta_vs_current_best"] == -2.4
    assert actions[2]["recommended_action"] == "design_executor"
    signals = report["learning_signals"]
    assert signals[0]["signal"] == "sandbox_underperformed_current_best"
    assert signals[0]["candidate_id"] == "candidate_checkpoint_adapter"
    assert report["avoid_repetition"][0]["candidate_id"] == "candidate_checkpoint_adapter"
    family_summary = {
        family["candidate_family"]: family for family in report["family_summary"]
    }
    assert family_summary["lightweight_residual_corrector"] == {
        "candidate_family": "lightweight_residual_corrector",
        "candidate_count": 1,
        "status_counts": {"compiled": 1},
        "best_score": None,
        "best_candidate_id": None,
        "recommended_action": "run_sandbox",
        "next_command": "run-blueprint-sandbox",
    }
    assert family_summary["official_checkpoint_family"]["candidate_count"] == 1
    assert family_summary["official_checkpoint_family"]["best_score"] == 83.6
    assert family_summary["official_checkpoint_family"]["recommended_action"] == (
        "refine_or_ablate"
    )
    assert family_summary["unsupported"]["recommended_action"] == "design_executor"


def test_summarize_literature_registry_for_hermes_uses_failure_history(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        summarize_literature_registry_for_hermes,
    )

    registry_path = tmp_path / "literature_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_UPDATED",
                "sources": [],
                "algorithm_nodes": [
                    {
                        "node_id": "algorithm_card_residual",
                        "algorithm_card_id": "card_residual",
                        "candidate_ids": ["candidate_residual"],
                        "strategy_status": "rejected",
                    }
                ],
                "candidate_nodes": [
                    {
                        "candidate_id": "candidate_residual",
                        "candidate_family": "lightweight_residual_corrector",
                        "status": "rejected",
                        "executor": "run-blueprint-sandbox",
                        "source_ids": ["source_residual"],
                        "failure_history": [
                            {
                                "artifact_path": str(tmp_path / "registry_event.json"),
                                "failure_codes": [
                                    "sandbox_rule_failed",
                                    "score_regressed",
                                ],
                                "recommended_next": (
                                    "avoid_repeating_without_new_evidence"
                                ),
                                "retry_allowed": False,
                            }
                        ],
                    }
                ],
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_literature_registry_for_hermes(
        registry_path=registry_path,
        output_path=tmp_path / "summary.json",
        current_best_score=86.0,
        force=False,
    )

    action = report["candidate_action_queue"][0]
    assert action["recommended_action"] == "deprioritize"
    assert action["reason"] == "candidate_failure_history_blocks_retry"
    assert action["failure_codes"] == ["sandbox_rule_failed", "score_regressed"]
    assert action["retry_allowed"] is False
    assert report["avoid_repetition"] == [
        {
            "candidate_id": "candidate_residual",
            "reason": "candidate_failure_history_blocks_retry",
            "failure_codes": ["sandbox_rule_failed", "score_regressed"],
            "recommended_next": "avoid_repeating_without_new_evidence",
            "retry_allowed": False,
        }
    ]
    assert report["learning_signals"][0]["signal"] == "candidate_failure_recorded"


def test_summarize_literature_registry_routes_failed_local_strategy_to_variant(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        summarize_literature_registry_for_hermes,
    )

    registry_path = tmp_path / "literature_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_UPDATED",
                "sources": [],
                "algorithm_nodes": [
                    {
                        "node_id": "algorithm_local_strategy",
                        "algorithm_card_id": "task1_early_subsegment_patch_v0",
                        "candidate_ids": [
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                        ],
                    }
                ],
                "candidate_nodes": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "status": "deprioritized",
                        "executor": "run-hermes-batch-optimization-loop",
                        "source_ids": ["task1_early_subsegment_patch_v0"],
                        "failure_history": [
                            {
                                "artifact_path": str(tmp_path / "event.json"),
                                "failure_codes": [
                                    "score_regressed",
                                    "ablation_missing_or_weak",
                                ],
                                "recommended_next": (
                                    "avoid_repeating_without_new_evidence"
                                ),
                                "retry_allowed": False,
                            }
                        ],
                    }
                ],
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_literature_registry_for_hermes(
        registry_path=registry_path,
        output_path=tmp_path / "summary.json",
        current_best_score=118.39,
        force=False,
    )

    action = report["candidate_action_queue"][0]
    assert action["recommended_action"] == "generate_variant_or_ablation"
    assert action["next_command"] == "run-candidate-expansion-handoff"
    assert action["variant_constraints"] == {
        "parent_candidate_id": (
            "candidate_local_strategy_task1_early_subsegment_patch_v0"
        ),
        "required_mutation_evidence": True,
        "forbidden_retry_without_new_ablation": True,
        "target_failure_codes": [
            "score_regressed",
            "ablation_missing_or_weak",
        ],
    }
    assert report["learning_signals"][0]["signal"] == (
        "local_strategy_variant_required"
    )
    assert report["next_command"] == "run-candidate-expansion-handoff"


def test_summarize_literature_registry_for_hermes_routes_promoted_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        summarize_literature_registry_for_hermes,
    )

    registry_event_path = tmp_path / "literature_registry_event.json"
    registry_event_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_EVENT_RECORDED",
                "event_type": "candidate_promoted",
                "algorithm_candidate_id": "candidate_local_strategy",
                "event_source_path": "candidate_promotion.json",
            }
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "literature_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_UPDATED",
                "sources": [],
                "algorithm_nodes": [
                    {
                        "node_id": "algorithm_local_strategy",
                        "algorithm_card_id": "local_strategy",
                        "strategy_status": "promoted",
                        "candidate_ids": ["candidate_local_strategy"],
                        "source_ids": ["local_strategy"],
                    }
                ],
                "candidate_nodes": [
                    {
                        "candidate_id": "candidate_local_strategy",
                        "candidate_family": "local_strategy_skill",
                        "status": "promoted",
                        "executor": "run-hermes-batch-optimization-loop",
                        "source_ids": ["local_strategy"],
                    }
                ],
                "events": [
                    {
                        "event_type": "candidate_promoted",
                        "algorithm_candidate_id": "candidate_local_strategy",
                        "algorithm_node_id": "algorithm_local_strategy",
                        "artifact_path": str(registry_event_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_literature_registry_for_hermes(
        registry_path=registry_path,
        output_path=tmp_path / "summary.json",
        current_best_score=118.0,
        force=False,
    )

    action = report["candidate_action_queue"][0]
    assert action["candidate_id"] == "candidate_local_strategy"
    assert action["recommended_action"] == "record_strategy_proposal"
    assert action["next_command"] == "record-promoted-algorithm-strategy-proposal"
    assert action["reason"] == "candidate_promotion_recorded"
    assert action["candidate_promotion_path"] == "candidate_promotion.json"
    assert action["registry_event_path"] == str(registry_event_path)
    assert report["next_command"] == "record-promoted-algorithm-strategy-proposal"


def test_record_promoted_algorithm_strategy_proposal_from_candidate_promotion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        record_promoted_algorithm_strategy_proposal,
    )

    candidates_path = tmp_path / "selected_algorithm_candidates.json"
    candidate_id = "candidate_local_strategy_task1_early_subsegment_patch_v0"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates_path": str(candidates_path),
                "algorithm_candidate_count": 1,
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "parent_algorithm_card_id": (
                            "local_strategy_task1_early_subsegment_patch_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "hypothesis": "Sub-window source selection improves early Burgers accuracy.",
                        "mechanism": "Splice legally validated early sub-window sources.",
                        "executor_spec": {
                            "cli_command": "run-hermes-batch-optimization-loop",
                            "implementation_hooks": [
                                "hypothesis_lab_early_subsegment_patch_search"
                            ],
                        },
                        "risk_controls": [
                            "local validation overfit",
                            "handoff discontinuity",
                        ],
                        "source_grounding": {
                            "algorithm_card_id": (
                                "local_strategy_task1_early_subsegment_patch_v0"
                            ),
                            "source_type": "local_canonical_skill",
                            "source_ids": ["task1_early_subsegment_patch_v0"],
                            "title": "Task 1 early subsegment source selection patch",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "sandbox_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "SANDBOX_EVIDENCE_RECORDED",
                "evidence_id": "evidence_candidate",
                "candidate_id": candidate_id,
                "rules_passed": True,
                "score": 128.77,
                "score_delta_vs_current_best": 10.38,
                "runtime": {"inference_seconds": 0.7},
                "physics_metrics": {"residual_delta": 0.0},
            }
        ),
        encoding="utf-8",
    )
    promotion_path = tmp_path / "candidate_promotion.json"
    promotion_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED",
                "algorithm_candidate_promotion_path": str(promotion_path),
                "algorithm_candidates_path": str(candidates_path),
                "algorithm_candidate_id": candidate_id,
                "candidate_family": "local_strategy_skill",
                "sandbox_evidence_paths": [str(evidence_path)],
                "decision": "promote",
                "promote_to_strategy_growth": True,
                "reason_codes": ["all_rubric_checks_passed"],
                "metric_results": {
                    "score": 128.77,
                    "current_best_score": 118.39,
                    "score_delta_vs_current_best": 10.38,
                    "inference_seconds": 0.7,
                    "physics_residual_delta": 0.0,
                    "rules_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )

    report = record_promoted_algorithm_strategy_proposal(
        candidate_promotion_path=promotion_path,
        output_path=tmp_path / "strategy_proposal.json",
        force=False,
    )

    assert report["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert report["strategy_id"] == "task1_early_subsegment_patch_v0"
    assert report["source_type"] == "local_library"
    assert report["title"] == "Task 1 early subsegment source selection patch"
    assert report["implementation_plan"]["algorithm_candidate_id"] == candidate_id
    assert report["implementation_plan"]["candidate_promotion_path"] == str(
        promotion_path
    )
    assert report["implementation_plan"]["allowed_commands"] == [
        "run-hermes-batch-optimization-loop"
    ]
    assert "score_delta_vs_current_best=10.38" in report["expected_observables"]
    assert report["known_risks"] == [
        "local validation overfit",
        "handoff discontinuity",
    ]


def test_convert_candidate_promotion_to_strategy_evidence_preserves_counts(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        convert_candidate_promotion_to_strategy_evidence,
    )

    promotion_path = tmp_path / "candidate_promotion.json"
    promotion_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED",
                "algorithm_candidate_promotion_path": str(promotion_path),
                "algorithm_candidates_path": str(tmp_path / "candidates.json"),
                "algorithm_candidate_id": "candidate_local_strategy",
                "candidate_family": "local_strategy_skill",
                "sandbox_evidence_paths": [str(tmp_path / "evidence.json")],
                "evidence_count": 1,
                "decision": "promote",
                "promote_to_strategy_growth": True,
                "reason_codes": ["all_rubric_checks_passed"],
                "metric_results": {
                    "score": 128.77,
                    "current_best_score": 118.39,
                    "score_delta_vs_current_best": 10.38,
                    "inference_seconds": 0.7,
                    "physics_residual_delta": 0.0,
                    "rules_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = convert_candidate_promotion_to_strategy_evidence(
        candidate_promotion_path=promotion_path,
        output_path=tmp_path / "strategy_evidence_summary.json",
        force=False,
    )

    assert evidence["evidence_summary_path"] == str(
        tmp_path / "strategy_evidence_summary.json"
    )
    assert evidence["source_candidate_promotion_path"] == str(promotion_path)
    assert evidence["algorithm_candidate_id"] == "candidate_local_strategy"
    assert evidence["evaluation_window_count"] == 1
    assert evidence["mean_score_delta"] == 10.38
    assert evidence["score_delta_std"] == 0.0
    assert evidence["long_window_score_delta"] == 10.38
    assert evidence["physics_residual_delta"] == 0.0
    assert evidence["inference_time_regression"] == 0.0
    assert evidence["failure_rate"] == 0.0
    assert evidence["next_command"] == "evaluate-strategy-promotion"


def test_convert_candidate_promotion_to_strategy_evidence_accumulates_windows(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        convert_candidate_promotion_to_strategy_evidence,
    )

    paths = []
    for index, delta in enumerate([10.0, 8.0, 6.0], start=1):
        path = tmp_path / f"candidate_promotion_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state_hint": "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED",
                    "algorithm_candidate_promotion_path": str(path),
                    "algorithm_candidates_path": str(tmp_path / "candidates.json"),
                    "algorithm_candidate_id": "candidate_local_strategy",
                    "candidate_family": "local_strategy_skill",
                    "sandbox_evidence_paths": [
                        str(tmp_path / f"evidence_{index}.json")
                    ],
                    "evidence_count": 1,
                    "decision": "promote",
                    "promote_to_strategy_growth": True,
                    "reason_codes": ["all_rubric_checks_passed"],
                    "metric_results": {
                        "score": 118.0 + delta,
                        "current_best_score": 118.0,
                        "score_delta_vs_current_best": delta,
                        "inference_seconds": 0.5 + index,
                        "physics_residual_delta": -0.01 * index,
                        "rules_passed": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    evidence = convert_candidate_promotion_to_strategy_evidence(
        candidate_promotion_path=paths[0],
        extra_candidate_promotion_paths=paths[1:],
        output_path=tmp_path / "strategy_evidence_summary.json",
        force=False,
    )

    assert evidence["source_candidate_promotion_paths"] == [
        str(path) for path in paths
    ]
    assert evidence["evaluation_window_count"] == 3
    assert evidence["mean_score_delta"] == 8.0
    assert evidence["score_delta_std"] == 2.0
    assert evidence["long_window_score_delta"] == 6.0
    assert evidence["physics_residual_delta"] == -0.01
    assert evidence["failure_rate"] == 0.0
    assert evidence["sandbox_evidence_paths"] == [
        str(tmp_path / "evidence_1.json"),
        str(tmp_path / "evidence_2.json"),
        str(tmp_path / "evidence_3.json"),
    ]


def test_plan_candidate_evidence_accumulation_from_insufficient_windows(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        plan_candidate_evidence_accumulation,
    )

    proposal_path = tmp_path / "strategy_proposal.json"
    rubric_path = tmp_path / "rubric.json"
    evidence_path = tmp_path / "strategy_evidence_summary.json"
    promotion_path = tmp_path / "strategy_promotion.json"
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_PROPOSAL_RECORDED",
                "strategy_id": "task1_early_subsegment_patch_v0",
            }
        ),
        encoding="utf-8",
    )
    rubric_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_RUBRIC_BUILT",
                "strategy_id": "task1_early_subsegment_patch_v0",
                "promotion_thresholds": {"min_evaluation_windows": 3},
            }
        ),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
                "algorithm_candidate_id": "candidate_local_strategy",
                "evaluation_window_count": 1,
                "source_candidate_promotion_paths": [
                    str(tmp_path / "candidate_promotion_1.json")
                ],
            }
        ),
        encoding="utf-8",
    )
    promotion_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_PROMOTION_EVALUATED",
                "strategy_id": "task1_early_subsegment_patch_v0",
                "strategy_proposal_path": str(proposal_path),
                "experiment_rubric_path": str(rubric_path),
                "evidence_summary_path": str(evidence_path),
                "promotion_report_path": str(promotion_path),
                "decision": "reject",
                "promote_to_skill_candidate": False,
                "reason_codes": ["insufficient_evaluation_windows"],
            }
        ),
        encoding="utf-8",
    )

    plan = plan_candidate_evidence_accumulation(
        strategy_promotion_report_path=promotion_path,
        strategy_evidence_summary_path=evidence_path,
        output_path=tmp_path / "accumulation_plan.json",
        force=False,
    )

    assert plan["state_hint"] == "CANDIDATE_EVIDENCE_ACCUMULATION_PLANNED"
    assert plan["decision"] == "accumulate_more_evidence"
    assert plan["strategy_id"] == "task1_early_subsegment_patch_v0"
    assert plan["algorithm_candidate_id"] == "candidate_local_strategy"
    assert plan["current_window_count"] == 1
    assert plan["required_window_count"] == 3
    assert plan["missing_window_count"] == 2
    assert plan["source_candidate_promotion_paths"] == [
        str(tmp_path / "candidate_promotion_1.json")
    ]
    assert plan["recommended_actions"][0]["suggested_sandbox_args"] == {
        "reference_sample_limit": 128,
    }
    assert plan["next_command"] == "run-hermes-runtime-closed-loop"


def test_summarize_literature_registry_for_hermes_recalls_legal_retest(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        summarize_literature_registry_for_hermes,
    )

    registry_path = tmp_path / "literature_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_UPDATED",
                "sources": [],
                "algorithm_nodes": [],
                "candidate_nodes": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "status": "needs_refinement",
                        "executor": "run-hermes-batch-optimization-loop",
                        "source_ids": ["local_strategy_library"],
                        "failure_history": [
                            {
                                "artifact_path": str(tmp_path / "registry_event.json"),
                                "failure_codes": ["target_leakage_risk"],
                                "recommended_next": (
                                    "refine_with_new_ablation_or_variant"
                                ),
                                "retry_allowed": True,
                                "metric_results": {
                                    "score": 128.77,
                                    "score_delta_vs_current_best": 10.38,
                                    "candidate_gate_issues": [
                                        "target_reference_local_validation_only"
                                    ],
                                    "diagnostic_only": True,
                                },
                            }
                        ],
                    }
                ],
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_literature_registry_for_hermes(
        registry_path=registry_path,
        output_path=tmp_path / "summary.json",
        current_best_score=118.39,
        force=False,
    )

    action = report["candidate_action_queue"][0]
    assert action["recommended_action"] == "refine_with_legal_reference"
    assert action["reason"] == "target_reference_signal_requires_legal_retest"
    assert action["best_score"] == 128.77
    assert action["score_delta_vs_current_best"] == 10.38
    assert action["candidate_gate_issues"] == [
        "target_reference_local_validation_only"
    ]
    assert action["next_command"] == "refine-algorithm-candidate"
    assert report["learning_signals"][0]["signal"] == (
        "target_reference_signal_requires_legal_retest"
    )
    assert report["learning_signals"][0]["recommended_learning_objective"] == (
        "legal_reference_retest_before_promotion"
    )
    assert report["next_command"] == "refine-algorithm-candidate"


def test_refine_algorithm_candidate_prepares_legal_reference_handoff(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import refine_algorithm_candidate

    summary_path = tmp_path / "hermes_registry_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "current_best_score": 118.39,
                "candidate_action_queue": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "recommended_action": "refine_with_legal_reference",
                        "next_command": "refine-algorithm-candidate",
                        "reason": "target_reference_signal_requires_legal_retest",
                        "best_score": 128.77,
                        "score_delta_vs_current_best": 10.38,
                        "candidate_gate_issues": [
                            "target_reference_local_validation_only"
                        ],
                        "executor": "run-hermes-batch-optimization-loop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = refine_algorithm_candidate(
        registry_summary_path=summary_path,
        output_path=tmp_path / "candidate_refinement.json",
        algorithm_candidate_id=None,
        force=False,
    )

    assert report["state_hint"] == "HERMES_ALGORITHM_CANDIDATE_REFINEMENT_PREPARED"
    assert report["algorithm_candidate_id"] == (
        "candidate_local_strategy_task1_early_subsegment_patch_v0"
    )
    assert report["refinement_type"] == "legal_reference_retest"
    assert report["learning_objective"] == "early_accuracy"
    assert report["candidate_generator"] == (
        "hypothesis_lab_early_subsegment_patch_search"
    )
    assert report["legal_reference_constraints"]["forbidden_reference_modes"] == [
        "target_as_reference",
        "local_validation_only",
    ]
    assert report["suggested_sandbox_args"]["reference_path_required"] is True
    assert report["suggested_sandbox_args"]["reference_sample_limit"] == 64
    assert report["success_criteria"] == [
        "rules_passed must be true",
        "candidate_gate_issues must not include target_reference_local_validation_only",
        "score_delta_vs_current_best should remain positive on legal reference",
    ]
    assert report["next_command"] == "run-expansion-sandbox-queue"


def test_refine_algorithm_candidate_command_returns_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.run_hermes_literature_strategy_expansion import (
        refine_algorithm_candidate_command,
    )

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "current_best_score": 118.39,
                "candidate_action_queue": [
                    {
                        "candidate_id": "candidate_local_strategy",
                        "candidate_family": "local_strategy_skill",
                        "recommended_action": "refine_with_legal_reference",
                        "reason": "target_reference_signal_requires_legal_retest",
                        "best_score": 128.77,
                        "score_delta_vs_current_best": 10.38,
                        "executor": "run-hermes-batch-optimization-loop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = refine_algorithm_candidate_command(
        registry_summary_path=str(summary_path),
        output_path=str(tmp_path / "candidate_refinement.json"),
        algorithm_candidate_id=None,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["command"] == "refine-algorithm-candidate"
    assert payload["artifacts"]["candidate_refinement_path"].endswith(
        "candidate_refinement.json"
    )
    assert payload["metrics"]["refinement_type"] == "legal_reference_retest"
    assert payload["next_hint"] == "run-expansion-sandbox-queue"


def test_checkpoint_failure_memory_skips_local_checkpoint_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        build_local_asset_algorithm_candidates,
        record_hermes_literature_candidate_outcome,
        run_hermes_literature_strategy_expansion,
    )
    from pde_agent.core.local_strategy_library import build_local_strategy_library

    library_path = tmp_path / "library.json"
    build_local_strategy_library(output_path=library_path, force=False)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    candidates = build_local_asset_algorithm_candidates(
        local_strategy_library_path=library_path,
        checkpoint_paths=[checkpoint],
        output_path=tmp_path / "local_asset_candidates.json",
        max_strategy_candidates=2,
        force=False,
    )
    memory = record_hermes_literature_candidate_outcome(
        algorithm_candidate_id="candidate_fno_blocked",
        candidate_key="card_fno|fourier_neural_operator_architecture",
        outcome="failed",
        reason_codes=["checkpoint_adapter_best_is_persistence"],
        output_path=tmp_path / "failure_memory.json",
        force=False,
    )

    assert memory["failed_candidate_families"] == ["official_checkpoint_family"]

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates["algorithm_candidates_path"],
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        failure_memory_path=tmp_path / "failure_memory.json",
        force=False,
    )

    assert report["skipped_candidates"][0]["candidate_family"] == (
        "official_checkpoint_family"
    )
    assert report["actions"][0]["algorithm_candidate_id"].startswith(
        "candidate_local_strategy_"
    )


def test_hypothesis_lab_builds_card_from_literature_expansion_failure(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards

    expansion_path = tmp_path / "expansion.json"
    expansion_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN",
                "actions": [
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "action": "record_executor_design_proposal",
                        "executable": False,
                        "next_command": "build-strategy-rubric",
                    }
                ],
                "skipped_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_old",
                        "skip_reason": "failed_memory",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        literature_paths=[expansion_path],
        output_path=tmp_path / "cards.json",
        force=False,
    )

    card = cards["cards"][0]
    assert card["source_type"] == "literature_expansion_report"
    assert card["implementation_hooks"] == ["avoid_failed_literature_candidate"]
    assert "candidate_old" in card["known_risks"][0]


def test_build_local_asset_algorithm_candidates_from_library_and_checkpoints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        build_local_asset_algorithm_candidates,
    )
    from pde_agent.core.local_strategy_library import build_local_strategy_library

    library_path = tmp_path / "library.json"
    build_local_strategy_library(output_path=library_path, force=False)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")

    report = build_local_asset_algorithm_candidates(
        local_strategy_library_path=library_path,
        checkpoint_paths=[checkpoint],
        output_path=tmp_path / "local_asset_candidates.json",
        max_strategy_candidates=4,
        force=False,
    )

    assert report["state_hint"] == "ALGORITHM_CANDIDATES_COMPILED"
    assert report["source_state_hint"] == "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT"
    families = {
        candidate["candidate_family"] for candidate in report["algorithm_candidates"]
    }
    assert "official_checkpoint_family" in families
    assert "local_strategy_skill" in families
    checkpoint_candidate = next(
        candidate
        for candidate in report["algorithm_candidates"]
        if candidate["candidate_family"] == "official_checkpoint_family"
    )
    assert checkpoint_candidate["resource_bindings"]["checkpoint_paths"] == [
        str(checkpoint)
    ]
    strategy_candidate = next(
        candidate
        for candidate in report["algorithm_candidates"]
        if candidate["candidate_family"] == "local_strategy_skill"
    )
    assert strategy_candidate["executor_spec"]["cli_command"] in (
        "predict-task1-retrieval-prior",
        "run-hypothesis-lab",
        "run-hermes-operator-tuning-loop",
        "run-hermes-batch-optimization-loop",
        "run-burgers-spectral-viscosity-benchmark",
    )


def test_run_candidate_expansion_handoff_builds_assets_and_expands(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    missing_checkpoint = tmp_path / "missing_Unet.tar"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [str(checkpoint), str(missing_checkpoint)],
                        "max_strategy_candidates": 2,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        force=False,
    )

    assert report["state_hint"] == "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN"
    assert Path(report["artifacts"]["local_strategy_library_path"]).exists()
    assert Path(report["artifacts"]["algorithm_candidates_path"]).exists()
    assert Path(report["artifacts"]["expansion_report_path"]).exists()
    assert report["metrics"]["existing_checkpoint_count"] == 1
    assert report["metrics"]["missing_checkpoint_count"] == 1
    assert report["metrics"]["algorithm_candidate_count"] >= 2
    assert report["next_command"] == report["expansion"]["next_command"]


def test_run_candidate_expansion_handoff_applies_learning_constraints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["local_strategy_skill"],
                        "mutation_required_for_families": ["local_strategy_skill"],
                        "reason_codes": ["deprioritized_completion_trace"],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [str(checkpoint)],
                        "max_strategy_candidates": 2,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )
    families = {
        candidate["candidate_family"]
        for candidate in candidates["algorithm_candidates"]
    }
    assert families == {"official_checkpoint_family"}
    assert report["metrics"]["learning_constraint_skipped_count"] == 2
    assert report["learning_constraint_skipped_candidates"][0]["reason"] == (
        "deprioritized_completion_trace"
    )


def test_run_candidate_expansion_handoff_deprioritizes_saturated_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "saturated_candidate_families": ["local_strategy_skill"],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [str(checkpoint)],
                        "max_strategy_candidates": 2,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )
    families = [
        candidate["candidate_family"]
        for candidate in candidates["algorithm_candidates"]
    ]
    assert families == ["official_checkpoint_family"]
    assert candidates["learning_constraints"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert report["metrics"]["learning_constraint_skipped_count"] == 2
    assert {
        item["reason"] for item in report["learning_constraint_skipped_candidates"]
    } == {"candidate_family_saturated"}


def test_run_candidate_expansion_handoff_keeps_mutated_same_family_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "mutated_local_strategy_library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT",
                "entries": [
                    {
                        "strategy_id": "plain_strategy_v0",
                        "title": "Plain strategy",
                        "source_type": "manual",
                        "hypothesis": "Plain repeated local strategy.",
                        "mechanism": "Repeat the previous local executor family.",
                        "implementation_hooks": ["run-hermes-batch-optimization-loop"],
                        "known_risks": ["repeat failure"],
                        "priority": 1,
                    },
                    {
                        "strategy_id": "mutated_strategy_v1",
                        "title": "Mutated strategy",
                        "source_type": "manual",
                        "hypothesis": "A mutated local strategy changes the failed family.",
                        "mechanism": "Add a new ablation axis before retrying.",
                        "implementation_hooks": ["run-hermes-batch-optimization-loop"],
                        "known_risks": ["needs ablation"],
                        "priority": 2,
                        "mutation_evidence": {
                            "parent_candidate_family": "local_strategy_skill",
                            "mutation_type": "new_ablation_axis",
                            "reason": "responds_to_deprioritized_completion_trace",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["local_strategy_skill"],
                        "mutation_required_for_families": ["local_strategy_skill"],
                        "reason_codes": ["deprioritized_completion_trace"],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 2,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert [candidate["algorithm_candidate_id"] for candidate in candidates] == [
        "candidate_local_strategy_mutated_strategy_v1"
    ]
    assert candidates[0]["mutation_evidence"]["mutation_type"] == "new_ablation_axis"
    assert report["metrics"]["learning_constraint_skipped_count"] == 1


def test_run_candidate_expansion_handoff_generates_local_strategy_variant(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["local_strategy_skill"],
                        "mutation_required_for_families": ["local_strategy_skill"],
                        "reason_codes": [
                            "local_strategy_failed_legal_retest_requires_variant"
                        ],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 1,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["candidate_family"] == "local_strategy_skill"
    assert candidate["algorithm_candidate_id"].endswith(
        "_mutation_robust_reference_ablation_v0"
    )
    assert candidate["mutation_evidence"] == {
        "parent_candidate_family": "local_strategy_skill",
        "parent_algorithm_candidate_id": "candidate_local_strategy_task1_early_subsegment_patch_v0",
        "mutation_type": "robust_reference_ablation_retry",
        "reason": "local_strategy_failed_legal_retest_requires_variant",
    }
    assert "robust_reference_ablation" in candidate["executor_spec"][
        "implementation_hooks"
    ]
    assert report["metrics"]["learning_constraint_skipped_count"] == 1


def test_run_candidate_expansion_handoff_uses_registry_to_generate_next_local_variant_axis(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["local_strategy_skill"],
                        "mutation_required_for_families": ["local_strategy_skill"],
                        "reason_codes": [
                            "score_regressed",
                            "ablation_missing_or_weak",
                        ],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 1,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )
    registry_summary_path = tmp_path / "registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
                {
                    "schema_version": 1,
                    "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                    "avoid_repetition": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                            "_mutation_robust_reference_ablation_v0"
                        ),
                        "reason": "local_strategy_failed_legal_retest_requires_variant",
                        "recommended_next": (
                            "only_retry_with_new_variant_or_ablation"
                        ),
                        "retry_allowed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        registry_summary_path=registry_summary_path,
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["algorithm_candidate_id"].endswith(
        "_mutation_source_stability_ablation_v0"
    )
    assert candidate["mutation_evidence"]["mutation_type"] == (
        "source_stability_ablation_retry"
    )
    assert candidate["mutation_evidence"]["previous_mutation_types"] == [
        "robust_reference_ablation_retry"
    ]
    assert "source_stability_ablation" in candidate["executor_spec"][
        "implementation_hooks"
    ]
    assert report["metrics"]["expansion_executable_count"] == 1


def test_run_candidate_expansion_handoff_turns_registry_variant_action_into_mutation(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["fast_baseline_family"],
                        "mutation_required_for_families": ["fast_baseline_family"],
                        "reason_codes": ["repeated_failed_family"],
                    },
                    "recommended_inputs": {
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 1,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )
    registry_summary_path = tmp_path / "registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "candidate_action_queue": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "recommended_action": "generate_variant_or_ablation",
                        "reason": "local_strategy_failed_legal_retest_requires_variant",
                        "failure_codes": [
                            "score_regressed",
                            "ablation_missing_or_weak",
                        ],
                        "variant_constraints": {
                            "required_mutation_evidence": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        registry_summary_path=registry_summary_path,
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["algorithm_candidate_id"].endswith(
        "_mutation_robust_reference_ablation_v0"
    )
    assert candidate["mutation_evidence"]["parent_algorithm_candidate_id"] == (
        "candidate_local_strategy_task1_early_subsegment_patch_v0"
    )
    assert candidate["mutation_evidence"]["mutation_type"] == (
        "robust_reference_ablation_retry"
    )
    assert report["metrics"]["learning_constraint_skipped_count"] == 1


def test_run_candidate_expansion_handoff_advances_registry_variant_failure_to_next_axis(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["fast_baseline_family"],
                        "mutation_required_for_families": ["fast_baseline_family"],
                        "reason_codes": ["repeated_failed_family"],
                    },
                    "recommended_inputs": {
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 1,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )
    registry_summary_path = tmp_path / "registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "avoid_repetition": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                            "_mutation_robust_reference_ablation_v0"
                        ),
                        "reason": "local_strategy_failed_legal_retest_requires_variant",
                        "retry_allowed": False,
                    }
                ],
                "candidate_action_queue": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                            "_mutation_robust_reference_ablation_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "recommended_action": "generate_variant_or_ablation",
                        "reason": "local_strategy_failed_legal_retest_requires_variant",
                        "failure_codes": [
                            "score_regressed",
                            "ablation_missing_or_weak",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        registry_summary_path=registry_summary_path,
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["algorithm_candidate_id"].endswith(
        "_mutation_source_stability_ablation_v0"
    )
    assert candidate["mutation_evidence"]["parent_algorithm_candidate_id"] == (
        "candidate_local_strategy_task1_early_subsegment_patch_v0"
    )
    assert candidate["mutation_evidence"]["mutation_type"] == (
        "source_stability_ablation_retry"
    )
    assert candidate["mutation_evidence"]["previous_mutation_types"] == [
        "robust_reference_ablation_retry"
    ]


def test_run_candidate_expansion_handoff_advances_past_second_local_variant_axis(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["local_strategy_skill"],
                        "mutation_required_for_families": ["local_strategy_skill"],
                        "reason_codes": [
                            "score_regressed",
                            "ablation_missing_or_weak",
                        ],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 1,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )
    registry_summary_path = tmp_path / "registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "avoid_repetition": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                            "_mutation_source_stability_ablation_v0"
                        ),
                        "reason": "local_strategy_failed_legal_retest_requires_variant",
                        "recommended_next": (
                            "only_retry_with_new_variant_or_ablation"
                        ),
                        "retry_allowed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        registry_summary_path=registry_summary_path,
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    candidate = candidates[0]
    assert candidate["algorithm_candidate_id"].endswith(
        "_mutation_handoff_smoothing_ablation_v0"
    )
    assert candidate["mutation_evidence"]["mutation_type"] == (
        "handoff_smoothing_ablation_retry"
    )
    assert candidate["mutation_evidence"]["previous_mutation_types"] == [
        "robust_reference_ablation_retry",
        "source_stability_ablation_retry",
    ]
    assert "handoff_smoothing_ablation" in candidate["executor_spec"][
        "implementation_hooks"
    ]


def test_run_candidate_expansion_handoff_exhausts_local_variant_axes_without_retrying_base(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["local_strategy_skill"],
                        "mutation_required_for_families": ["local_strategy_skill"],
                        "reason_codes": [
                            "score_regressed",
                            "ablation_missing_or_weak",
                        ],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 1,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )
    registry_summary_path = tmp_path / "registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "avoid_repetition": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                            "_mutation_handoff_smoothing_ablation_v0"
                        ),
                        "reason": "local_strategy_failed_legal_retest_requires_variant",
                        "recommended_next": (
                            "only_retry_with_new_variant_or_ablation"
                        ),
                        "retry_allowed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        registry_summary_path=registry_summary_path,
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert candidates == []
    assert report["metrics"]["learning_constraint_skipped_count"] == 1
    assert report["metrics"]["expansion_action_count"] == 0
    assert report["expansion_exhaustion"] == {
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
        "recommended_learning_objective": "switch_candidate_family_after_axis_exhaustion",
    }
    assert report["next_command"] == "plan-hypothesis-candidates"


def test_run_candidate_expansion_handoff_carries_prior_node_exhaustion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "prior_expansion_exhaustion": {
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
                    "recommended_learning_objective": (
                        "switch_candidate_family_after_axis_exhaustion"
                    ),
                },
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": ["local_strategy_skill"],
                        "mutation_required_for_families": ["local_strategy_skill"],
                        "reason_codes": [
                            "score_regressed",
                            "ablation_missing_or_weak",
                        ],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 2,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )
    registry_summary_path = tmp_path / "registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "avoid_repetition": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_transition_expert_v0"
                            "_mutation_handoff_smoothing_ablation_v0"
                        ),
                        "reason": "local_strategy_failed_legal_retest_requires_variant",
                        "retry_allowed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        registry_summary_path=registry_summary_path,
        force=False,
    )

    exhaustion = report["expansion_exhaustion"]
    assert exhaustion["exhausted"] is True
    assert exhaustion["exhausted_algorithm_candidate_ids"] == [
        "candidate_local_strategy_task1_early_subsegment_patch_v0",
        "candidate_local_strategy_task1_early_transition_expert_v0",
    ]
    assert exhaustion["exhausted_algorithm_node_ids"] == [
        "local_strategy_task1_early_subsegment_patch_v0",
        "local_strategy_task1_early_transition_expert_v0",
    ]


def test_run_candidate_expansion_handoff_mutates_deprioritized_fno_checkpoint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    fno_checkpoint = tmp_path / "burgers_FNO.tar"
    unet_checkpoint = tmp_path / "burgers_Unet.tar"
    fno_checkpoint.write_bytes(b"fno")
    unet_checkpoint.write_bytes(b"unet")
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": [
                            "official_checkpoint_family",
                            "local_strategy_skill",
                        ],
                        "mutation_required_for_families": [
                            "official_checkpoint_family",
                            "local_strategy_skill",
                        ],
                        "reason_codes": [
                            "deprioritized_completion_trace",
                            "repeated_failed_family",
                        ],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [
                            str(fno_checkpoint),
                            str(unet_checkpoint),
                        ],
                        "max_strategy_candidates": 2,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert [candidate["candidate_family"] for candidate in candidates] == [
        "official_checkpoint_family"
    ]
    candidate = candidates[0]
    assert candidate["algorithm_candidate_id"].endswith(
        "_mutation_finetune_calibration_v0"
    )
    assert candidate["resource_bindings"]["checkpoint_paths"] == [str(fno_checkpoint)]
    assert candidate["parameter_grid"]["fno_finetune_steps"] == [1]
    assert candidate["parameter_grid"]["calibration_weights"] == [0.1, 0.25, 0.5]
    assert candidate["mutation_evidence"] == {
        "parent_candidate_family": "official_checkpoint_family",
        "parent_algorithm_candidate_id": "candidate_local_checkpoint_burgers_fno",
        "mutation_type": "checkpoint_finetune_calibration_retry",
        "reason": "responds_to_deprioritized_checkpoint_family",
    }
    skipped_ids = {
        item["algorithm_candidate_id"]
        for item in report["learning_constraint_skipped_candidates"]
    }
    assert "candidate_local_checkpoint_burgers_fno" in skipped_ids
    assert "candidate_local_checkpoint_burgers_unet" in skipped_ids
    assert report["metrics"]["learning_constraint_skipped_count"] >= 2
    assert report["metrics"]["expansion_action_count"] == 1
    assert report["next_command"] != "plan-hypothesis-candidates"


def test_run_candidate_expansion_handoff_prefers_viable_actions_over_partial_local_exhaustion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_candidate_expansion_handoff,
    )

    library_path = tmp_path / "generated_local_strategy_library.json"
    fno_checkpoint = tmp_path / "burgers_FNO.tar"
    unet_checkpoint = tmp_path / "burgers_Unet.tar"
    fno_checkpoint.write_bytes(b"fno")
    unet_checkpoint.write_bytes(b"unet")
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "learning_constraints": {
                        "avoid_candidate_families": [
                            "official_checkpoint_family",
                            "local_strategy_skill",
                        ],
                        "mutation_required_for_families": [
                            "official_checkpoint_family",
                            "local_strategy_skill",
                        ],
                        "reason_codes": [
                            "deprioritized_completion_trace",
                            "repeated_failed_family",
                        ],
                    },
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [
                            str(fno_checkpoint),
                            str(unet_checkpoint),
                        ],
                        "max_strategy_candidates": 5,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )
    registry_summary_path = tmp_path / "registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "avoid_repetition": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                            "_mutation_handoff_smoothing_ablation_v0"
                        ),
                        "reason": "local_strategy_failed_legal_retest_requires_variant",
                        "recommended_next": (
                            "only_retry_with_new_variant_or_ablation"
                        ),
                        "retry_allowed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=candidate_plan_path,
        workspace_dir=tmp_path / "workspace",
        registry_summary_path=registry_summary_path,
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert [candidate["candidate_family"] for candidate in candidates] == [
        "official_checkpoint_family"
    ]
    assert report["expansion_exhaustion"]["exhausted"] is True
    assert report["metrics"]["expansion_action_count"] == 1
    assert report["next_command"] != "plan-hypothesis-candidates"


def test_run_candidate_expansion_handoff_cli_records_trace(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    library_path = tmp_path / "library.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [],
                        "max_strategy_candidates": 1,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "run-candidate-expansion-handoff",
            "--candidate-plan-path",
            str(candidate_plan_path),
            "--workspace-dir",
            str(tmp_path / "workspace"),
            "--trace-path",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN"
    assert payload["metrics"]["trace_event_recorded"] is True
    event = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "hermes_candidate_expansion_handoff"
    assert event["payload"]["expansion_required"] is True


def test_run_candidate_expansion_handoff_cli_uses_recommended_checkpoints(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    library_path = tmp_path / "library.json"
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    candidate_plan_path = tmp_path / "candidate_plan.json"
    candidate_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "decision": "candidate_space_exhausted",
                "expansion_required": True,
                "expansion_handoff": {
                    "reason": "candidate_space_exhausted",
                    "next_command": "build-local-asset-algorithm-candidates",
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library_path),
                        "checkpoint_paths": [str(checkpoint)],
                        "max_strategy_candidates": 1,
                    },
                    "follow_up_command": "run-hermes-literature-strategy-expansion",
                },
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run-candidate-expansion-handoff",
            "--candidate-plan-path",
            str(candidate_plan_path),
            "--workspace-dir",
            str(tmp_path / "workspace"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metrics"]["existing_checkpoint_count"] == 1


def test_run_expansion_sandbox_queue_executes_hypothesis_lab_plan(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.hermes_literature_expansion import (
        run_expansion_sandbox_queue,
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_residual",
                        "candidate_family": "local_strategy_skill",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "model_family": "FNO",
                        "hypothesis": "Residual correction may reduce rollout bias.",
                        "mechanism": "Apply bounded horizon residual correction.",
                        "executor_spec": {
                            "cli_command": "run-hypothesis-lab",
                            "implementation_hooks": ["learned_residual_correction"],
                            "required_options": [],
                        },
                        "risk_controls": ["overfit"],
                        "source_grounding": {
                            "source_ids": ["local_residual"],
                            "algorithm_card_id": "local_strategy_residual",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    expansion = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "expansion",
        checkpoint_paths=[],
        force=False,
    )
    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    context = np.repeat(np.sin(x)[None, None, :], 10, axis=1).astype(np.float32)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    target[:, 10:, :] = context[:, -1:, :] + np.float32(0.1)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    report = run_expansion_sandbox_queue(
        expansion_report_path=expansion["expansion_report_path"],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "queue",
        submission_id="team_queue_smoke",
        max_items=1,
        failure_memory_path=tmp_path / "failure_memory.json",
        force=False,
    )

    assert report["state_hint"] == "HERMES_EXPANSION_SANDBOX_QUEUE_RAN"
    assert report["metrics"]["executed_count"] == 1
    assert report["metrics"]["skipped_count"] == 0
    item = report["items"][0]
    assert item["algorithm_candidate_id"] == "candidate_residual"
    assert item["outcome"] == "passed"
    assert Path(item["sandbox_report_path"]).exists()
    assert Path(report["failure_memory_path"]).exists()
    memory = json.loads(Path(report["failure_memory_path"]).read_text())
    assert memory["outcomes"][0]["algorithm_candidate_id"] == "candidate_residual"
    assert memory["outcomes"][0]["outcome"] == "passed"


def test_run_expansion_sandbox_queue_executes_bounded_local_strategy_plan(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.hermes_literature_expansion import run_expansion_sandbox_queue

    plan_path = tmp_path / "early_subsegment_patch_sandbox_plan.json"
    candidate = {
        "algorithm_candidate_id": "candidate_local_strategy_task1_early_subsegment_patch_v0",
        "candidate_family": "local_strategy_skill",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "hypothesis": "Early subsegments need separate source selection.",
        "mechanism": "Run bounded early-subsegment patch search.",
        "executor_spec": {
            "cli_command": "run-hermes-batch-optimization-loop",
            "implementation_hooks": [
                "run-hermes-batch-optimization-loop",
                "hypothesis_lab_early_subsegment_patch_search",
                "early_subsegment_patch",
            ],
        },
        "source_grounding": {
            "source_ids": ["local_strategy_library"],
            "algorithm_card_id": "task1_early_subsegment_patch_v0",
        },
        "parameter_grid": {
            "reference_sample_limits": [128],
            "requires_wider_reference_ablation": [True],
            "source_stability_guard": [True],
            "handoff_smoothing_weights": [0.5],
        },
        "mutation_evidence": {
            "parent_candidate_family": "local_strategy_skill",
            "parent_algorithm_candidate_id": (
                "candidate_local_strategy_task1_early_subsegment_patch_v0"
            ),
            "mutation_type": "robust_reference_ablation_retry",
            "reason": "local_strategy_failed_legal_retest_requires_variant",
        },
    }
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "sandbox_plan_path": str(plan_path),
                "algorithm_candidate_id": candidate["algorithm_candidate_id"],
                "candidate_family": "local_strategy_skill",
                "executor": {"cli_command": "run-hermes-batch-optimization-loop"},
                "blocked": False,
                "selected_candidate": candidate,
                "next_command": "run-hermes-batch-optimization-loop",
            }
        ),
        encoding="utf-8",
    )
    expansion_report = tmp_path / "expansion.json"
    expansion_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN",
                "expansion_report_path": str(expansion_report),
                "sandbox_queue": [str(plan_path)],
            }
        ),
        encoding="utf-8",
    )
    data = np.zeros((1, 200, 256), dtype=np.float32)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", data[:, :10, :])
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", data)
    initial_submission_dir = tmp_path / "initial_submission"
    initial_submission_dir.mkdir()
    _write_hdf5(initial_submission_dir / "task1_pred.hdf5", data)
    calls: list[dict[str, object]] = []

    def fake_local_strategy_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        report_path = output_dir / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "best_candidate": {
                "candidate_id": "early_subsegment_patch_candidate",
                "score": 119.5,
                "rules_passed": True,
                "objective_signals": {
                    "rules_passed": True,
                    "inference_time": 1.25,
                    "segment_score": 92.0,
                    "early_segment_score": 88.0,
                },
                "segments": {
                    "early": {"score": 88.0, "rel_mse": 0.01},
                    "middle": {"score": 91.0},
                    "late": {"score": 94.0},
                },
            },
            "decision": {
                "decision": "iterate_more",
                "next_command": "run-hermes-batch-optimization-loop",
            },
            "artifacts": {
                "best_submission_dir": str(initial_submission_dir),
                "best_judge_report_path": str(output_dir / "judge.json"),
            },
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_expansion_sandbox_queue(
        expansion_report_path=expansion_report,
        context_path=context_path,
        target_path=target_path,
        initial_submission_dir=initial_submission_dir,
        output_dir=tmp_path / "queue",
        submission_id="team_local_strategy_queue",
        max_items=1,
        failure_memory_path=tmp_path / "failure_memory.json",
        local_strategy_runner=fake_local_strategy_runner,
        force=False,
    )

    assert report["metrics"]["executed_count"] == 1
    assert report["metrics"]["skipped_count"] == 0
    item = report["items"][0]
    assert item["status"] == "executed"
    assert item["outcome"] == "passed"
    assert item["next_command"] == "evaluate-expansion-sandbox-queue"
    assert report["next_command"] == "evaluate-expansion-sandbox-queue"
    assert item["local_strategy_loop_report_path"].endswith(
        "hermes_batch_optimization_loop.json"
    )
    assert calls[0]["initial_submission_dir"] == initial_submission_dir
    assert calls[0]["candidate_generator"] == (
        "hypothesis_lab_early_subsegment_patch_search"
    )
    assert calls[0]["hypothesis_reference_sample_limit"] == 128
    assert calls[0]["hypothesis_candidate_options"] == {
        "source_stability_guard": True,
        "handoff_smoothing_weight": 0.5,
    }
    assert calls[0]["max_rounds"] == 5
    assert calls[0]["deadline_seconds"] == 300.0
    assert calls[0]["learning_objective"] == "early_accuracy"


def test_run_expansion_sandbox_queue_executes_checkpoint_family_plan(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.hermes_literature_expansion import run_expansion_sandbox_queue

    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    plan_path = tmp_path / "checkpoint_sandbox_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "sandbox_plan_path": str(plan_path),
                "algorithm_candidate_id": "candidate_local_checkpoint_burgers_fno",
                "candidate_family": "official_checkpoint_family",
                "blocked": False,
                "executor": {"cli_command": "run-checkpoint-family-sandbox"},
                "selected_candidate": {
                    "algorithm_candidate_id": "candidate_local_checkpoint_burgers_fno",
                    "parent_algorithm_card_id": "local_checkpoint_burgers_fno",
                    "candidate_family": "official_checkpoint_family",
                    "executor_spec": {
                        "cli_command": "run-checkpoint-family-sandbox"
                    },
                    "resource_bindings": {"checkpoint_paths": [str(checkpoint)]},
                    "promotion_criteria": {
                        "min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                        "requires_ablation": False,
                    },
                    "source_grounding": {
                        "algorithm_card_id": "local_checkpoint_burgers_fno",
                        "source_ids": [str(checkpoint)],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    expansion_path = tmp_path / "expansion.json"
    expansion_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN",
                "expansion_report_path": str(expansion_path),
                "sandbox_queue": [str(plan_path)],
            }
        ),
        encoding="utf-8",
    )
    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    calls = []

    def fake_checkpoint_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "checkpoint_family_sandbox_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "state_hint": "CHECKPOINT_FAMILY_SANDBOX_RAN",
            "checkpoint_family_report_path": str(report_path),
            "best_candidate": {
                "candidate_id": "fno_calibrated_w1",
                "checkpoint_path": str(checkpoint),
                "score": 92.0,
                "rules_passed": True,
                "inference_time": 3.0,
                "calibration_weight": 1.0,
                "submission_dir": str(tmp_path / "submission"),
            },
            "artifacts": {
                "checkpoint_family_report_path": str(report_path),
                "best_submission_dir": str(tmp_path / "submission"),
            },
            "next_command": "record-strategy-proposal",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_expansion_sandbox_queue(
        expansion_report_path=expansion_path,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "queue",
        submission_id="team_checkpoint_queue",
        max_items=1,
        failure_memory_path=tmp_path / "failure_memory.json",
        checkpoint_family_runner=fake_checkpoint_runner,
        force=False,
    )

    assert calls
    assert calls[0]["checkpoint_paths"] == [checkpoint]
    assert calls[0]["task_id"] == 1
    assert calls[0]["test_path"] == context_path
    assert calls[0]["target_path"] == target_path
    assert report["metrics"]["executed_count"] == 1
    assert report["metrics"]["skipped_count"] == 0
    item = report["items"][0]
    assert item["status"] == "executed"
    assert item["outcome"] == "passed"
    assert item["next_command"] == "evaluate-expansion-sandbox-queue"
    assert report["next_command"] == "evaluate-expansion-sandbox-queue"
    assert item["checkpoint_family_report_path"].endswith(
        "checkpoint_family_sandbox_report.json"
    )
    assert Path(item["selected_algorithm_candidates_path"]).exists()
    memory = json.loads((tmp_path / "failure_memory.json").read_text())
    assert memory["outcomes"][0]["outcome"] == "passed"


def test_run_expansion_sandbox_queue_filters_selected_algorithm_candidate_id(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.hermes_literature_expansion import run_expansion_sandbox_queue

    checkpoint_a = tmp_path / "burgers_FNO.tar"
    checkpoint_b = tmp_path / "burgers_UNET.tar"
    checkpoint_a.write_bytes(b"checkpoint-a")
    checkpoint_b.write_bytes(b"checkpoint-b")
    plan_paths = []
    for candidate_id, checkpoint in [
        ("candidate_local_checkpoint_burgers_fno", checkpoint_a),
        ("candidate_local_checkpoint_burgers_unet", checkpoint_b),
    ]:
        plan_path = tmp_path / f"{candidate_id}.json"
        plan_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                    "sandbox_plan_path": str(plan_path),
                    "algorithm_candidate_id": candidate_id,
                    "candidate_family": "official_checkpoint_family",
                    "blocked": False,
                    "executor": {"cli_command": "run-checkpoint-family-sandbox"},
                    "selected_candidate": {
                        "algorithm_candidate_id": candidate_id,
                        "parent_algorithm_card_id": candidate_id.replace(
                            "candidate_", "algorithm_"
                        ),
                        "candidate_family": "official_checkpoint_family",
                        "executor_spec": {
                            "cli_command": "run-checkpoint-family-sandbox"
                        },
                        "resource_bindings": {"checkpoint_paths": [str(checkpoint)]},
                        "promotion_criteria": {
                            "min_score_delta": 0.01,
                            "max_inference_seconds": 120.0,
                            "requires_ablation": False,
                        },
                        "source_grounding": {
                            "algorithm_card_id": candidate_id.replace(
                                "candidate_", "algorithm_"
                            ),
                            "source_ids": [str(checkpoint)],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        plan_paths.append(plan_path)
    expansion_path = tmp_path / "expansion.json"
    expansion_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN",
                "expansion_report_path": str(expansion_path),
                "sandbox_queue": [str(path) for path in plan_paths],
            }
        ),
        encoding="utf-8",
    )
    context_path = _write_hdf5(
        tmp_path / "task1_test.hdf5",
        np.zeros((1, 10, 256), dtype=np.float32),
    )
    target_path = _write_hdf5(
        tmp_path / "task1_val.hdf5",
        np.zeros((1, 200, 256), dtype=np.float32),
    )
    calls = []

    def fake_checkpoint_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "checkpoint_family_sandbox_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "state_hint": "CHECKPOINT_FAMILY_SANDBOX_RAN",
            "checkpoint_family_report_path": str(report_path),
            "best_candidate": {
                "candidate_id": "selected_checkpoint",
                "checkpoint_path": str(kwargs["checkpoint_paths"][0]),
                "score": 92.0,
                "rules_passed": True,
                "inference_time": 3.0,
                "calibration_weight": 1.0,
                "submission_dir": str(tmp_path / "submission"),
            },
            "artifacts": {
                "checkpoint_family_report_path": str(report_path),
                "best_submission_dir": str(tmp_path / "submission"),
            },
            "next_command": "record-strategy-proposal",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_expansion_sandbox_queue(
        expansion_report_path=expansion_path,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "queue",
        submission_id="team_checkpoint_queue",
        max_items=1,
        selected_algorithm_candidate_id="candidate_local_checkpoint_burgers_unet",
        checkpoint_family_runner=fake_checkpoint_runner,
        force=False,
    )

    assert len(calls) == 1
    assert calls[0]["checkpoint_paths"] == [checkpoint_b]
    assert report["metrics"]["queued_count"] == 2
    assert report["metrics"]["candidate_filter_applied"] is True
    assert report["metrics"]["candidate_filter_match_count"] == 1
    assert report["items"][0]["algorithm_candidate_id"] == (
        "candidate_local_checkpoint_burgers_unet"
    )


def test_run_expansion_sandbox_queue_cli_records_skip_trace(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    sandbox_plan = tmp_path / "checkpoint_sandbox_plan.json"
    sandbox_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "sandbox_plan_path": str(sandbox_plan),
                "algorithm_candidates_path": str(tmp_path / "candidates.json"),
                "algorithm_candidate_id": "candidate_checkpoint",
                "candidate_family": "official_checkpoint_family",
                "executor": {"cli_command": "run-checkpoint-family-sandbox"},
                "blocked": False,
                "selected_candidate": {
                    "algorithm_candidate_id": "candidate_checkpoint",
                    "candidate_family": "official_checkpoint_family",
                    "source_grounding": {
                        "algorithm_card_id": "local_checkpoint",
                        "source_ids": ["checkpoint"],
                    },
                },
                "next_command": "run-checkpoint-family-sandbox",
            }
        ),
        encoding="utf-8",
    )
    expansion_report = tmp_path / "expansion.json"
    expansion_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN",
                "expansion_report_path": str(expansion_report),
                "sandbox_queue": [str(sandbox_plan)],
                "actions": [],
            }
        ),
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "run-expansion-sandbox-queue",
            "--expansion-report-path",
            str(expansion_report),
            "--context-path",
            str(tmp_path / "missing_context.hdf5"),
            "--target-path",
            str(tmp_path / "missing_target.hdf5"),
            "--output-dir",
            str(tmp_path / "queue"),
            "--failure-memory-path",
            str(tmp_path / "failure_memory.json"),
            "--trace-path",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_EXPANSION_SANDBOX_QUEUE_RAN"
    assert payload["metrics"]["executed_count"] == 0
    assert payload["metrics"]["skipped_count"] == 1
    event = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "hermes_expansion_sandbox_queue"
    assert event["payload"]["skipped_count"] == 1


def test_run_expansion_sandbox_queue_records_item_error_and_continues(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_expansion_sandbox_queue,
    )

    first_plan = tmp_path / "first_sandbox_plan.json"
    second_plan = tmp_path / "second_sandbox_plan.json"
    selected = {
        "algorithm_candidate_id": "candidate_residual",
        "candidate_family": "local_strategy_skill",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "hypothesis": "Residual correction may reduce rollout bias.",
        "mechanism": "Apply bounded residual correction.",
        "executor_spec": {
            "cli_command": "run-hypothesis-lab",
            "implementation_hooks": ["learned_residual_correction"],
        },
        "source_grounding": {
            "algorithm_card_id": "local_strategy_residual",
            "source_ids": ["local_residual"],
        },
    }
    first_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "sandbox_plan_path": str(first_plan),
                "algorithm_candidate_id": "candidate_residual",
                "candidate_family": "local_strategy_skill",
                "executor": {"cli_command": "run-hypothesis-lab"},
                "blocked": False,
                "selected_candidate": selected,
                "next_command": "run-hypothesis-lab",
            }
        ),
        encoding="utf-8",
    )
    second_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "sandbox_plan_path": str(second_plan),
                "algorithm_candidate_id": "candidate_checkpoint",
                "candidate_family": "official_checkpoint_family",
                "executor": {"cli_command": "run-checkpoint-family-sandbox"},
                "blocked": False,
                "selected_candidate": {
                    "algorithm_candidate_id": "candidate_checkpoint",
                    "candidate_family": "official_checkpoint_family",
                    "source_grounding": {"algorithm_card_id": "checkpoint"},
                },
                "next_command": "run-checkpoint-family-sandbox",
            }
        ),
        encoding="utf-8",
    )
    expansion_report = tmp_path / "expansion.json"
    expansion_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN",
                "expansion_report_path": str(expansion_report),
                "sandbox_queue": [str(first_plan), str(second_plan)],
            }
        ),
        encoding="utf-8",
    )

    report = run_expansion_sandbox_queue(
        expansion_report_path=expansion_report,
        context_path=tmp_path / "missing_context.hdf5",
        target_path=tmp_path / "missing_target.hdf5",
        output_dir=tmp_path / "queue",
        submission_id="team_error_smoke",
        max_items=2,
        failure_memory_path=tmp_path / "failure_memory.json",
        force=False,
    )

    assert report["metrics"]["attempted_count"] == 2
    assert report["metrics"]["error_count"] == 1
    assert report["metrics"]["skipped_count"] == 1
    assert report["items"][0]["status"] == "error"
    assert report["items"][0]["outcome"] == "failed"
    assert report["items"][1]["status"] == "skipped"
    memory = json.loads((tmp_path / "failure_memory.json").read_text())
    assert len(memory["outcomes"]) == 2


def test_evaluate_expansion_sandbox_queue_promotes_and_updates_registry(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        evaluate_expansion_sandbox_queue,
    )

    candidate_id = "candidate_residual"
    candidates_path = tmp_path / "selected_algorithm_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates_path": str(candidates_path),
                "algorithm_candidate_count": 1,
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "parent_algorithm_card_id": "local_strategy_residual",
                        "candidate_family": "local_strategy_skill",
                        "executor_spec": {"cli_command": "run-hypothesis-lab"},
                        "promotion_criteria": {
                            "min_score_delta": 0.01,
                            "max_inference_seconds": 120.0,
                            "requires_ablation": True,
                        },
                        "source_grounding": {
                            "source_ids": ["local_residual"],
                            "algorithm_card_id": "local_strategy_residual",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    lab_report_path = tmp_path / "hypothesis_lab_report.json"
    sandbox_report_path = tmp_path / "hypothesis_lab_sandbox_report.json"
    lab_report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_LAB_RAN",
                "artifacts": {
                    "sandbox_report_path": str(sandbox_report_path),
                    "best_submission_dir": str(tmp_path / "submission"),
                },
                "sandbox": {
                    "best_candidate": {
                        "candidate_id": "residual_candidate_v0",
                        "elapsed_seconds": 4.0,
                        "segment_score": 91.0,
                        "task1_total_score_proxy": 91.5,
                        "segments": {
                            "early": {"score": 90.0},
                            "middle": {"score": 92.0},
                            "late": {"score": 91.0},
                        },
                    },
                    "verifier": {
                        "decision": "promote_to_strategy_proposal",
                        "context_preserved": True,
                        "score_delta_vs_baseline": 1.5,
                        "best_segment_score": 91.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    queue_report_path = tmp_path / "queue_report.json"
    queue_report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
                "queue_report_path": str(queue_report_path),
                "items": [
                    {
                        "status": "executed",
                        "outcome": "passed",
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": "local_strategy_skill",
                        "selected_algorithm_candidates_path": str(candidates_path),
                        "sandbox_report_path": str(lab_report_path),
                    },
                    {
                        "status": "skipped",
                        "outcome": "blocked",
                        "algorithm_candidate_id": "candidate_checkpoint",
                        "candidate_family": "official_checkpoint_family",
                        "candidate_key": "checkpoint|official_checkpoint_family",
                        "reason_codes": [
                            "unsupported_for_auto_short_sandbox:run-checkpoint-family-sandbox"
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_expansion_sandbox_queue(
        queue_report_path=queue_report_path,
        output_dir=tmp_path / "promotion",
        registry_path=tmp_path / "literature_registry.json",
        current_best_score=90.0,
        force=False,
    )

    assert report["state_hint"] == "HERMES_EXPANSION_SANDBOX_QUEUE_EVALUATED"
    assert report["metrics"]["evidence_count"] == 1
    assert report["metrics"]["promotion_count"] == 1
    assert report["metrics"]["registry_event_count"] == 2
    assert report["items"][0]["decision"] == "promote"
    evidence = json.loads(Path(report["items"][0]["evidence_path"]).read_text())
    assert evidence["state_hint"] == "SANDBOX_EVIDENCE_RECORDED"
    assert evidence["score"] == 91.5
    registry = json.loads(Path(report["artifacts"]["registry_path"]).read_text())
    by_candidate = {
        candidate["candidate_id"]: candidate
        for candidate in registry["candidate_nodes"]
    }
    assert by_candidate[candidate_id]["status"] == "promoted"
    assert by_candidate["candidate_checkpoint"]["status"] == "blocked"
    assert by_candidate["candidate_checkpoint"]["candidate_family"] == (
        "official_checkpoint_family"
    )
    assert report["next_command"] == "record-strategy-proposal"


def test_evaluate_expansion_sandbox_queue_refines_local_validation_signal(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        evaluate_expansion_sandbox_queue,
    )

    candidate_id = "candidate_local_strategy_task1_early_subsegment_patch_v0"
    candidates_path = tmp_path / "selected_algorithm_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates_path": str(candidates_path),
                "algorithm_candidate_count": 1,
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "parent_algorithm_card_id": "task1_early_subsegment_patch_v0",
                        "candidate_family": "local_strategy_skill",
                        "executor_spec": {
                            "cli_command": "run-hermes-batch-optimization-loop"
                        },
                        "promotion_criteria": {
                            "min_score_delta": 0.01,
                            "max_inference_seconds": 300.0,
                            "requires_ablation": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loop_report_path = tmp_path / "hermes_batch_optimization_loop.json"
    loop_report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
                "loop_report_path": str(loop_report_path),
                "best_candidate": {
                    "candidate_id": "initial",
                    "score": 118.39,
                    "rules_passed": True,
                    "objective_signals": {
                        "rules_passed": True,
                        "inference_time": 2.0,
                    },
                    "segments": {
                        "early": {"score": 82.0},
                        "middle": {"score": 89.0},
                        "late": {"score": 130.0},
                    },
                },
                "trajectory": [
                    {
                        "candidate_id": "initial",
                        "score": 118.39,
                        "rules_passed": True,
                    },
                    {
                        "candidate_id": (
                            "early_subsegment_patch_into_task1_retrieval_refinement_"
                            "k64_affine_eb0_pw0_s0"
                        ),
                        "score": 128.77,
                        "rules_passed": False,
                        "candidate_gate_issues": [
                            "target_reference_local_validation_only"
                        ],
                        "objective_signals": {
                            "rules_passed": False,
                            "inference_time": 3.0,
                        },
                        "segments": {
                            "early": {"score": 96.0},
                            "middle": {"score": 93.0},
                            "late": {"score": 142.0},
                        },
                    },
                ],
                "artifacts": {
                    "best_submission_dir": str(tmp_path / "submission"),
                    "best_judge_report_path": str(tmp_path / "judge.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    queue_report_path = tmp_path / "queue_report.json"
    queue_report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
                "queue_report_path": str(queue_report_path),
                "items": [
                    {
                        "status": "executed",
                        "outcome": "passed",
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": "local_strategy_skill",
                        "selected_algorithm_candidates_path": str(candidates_path),
                        "local_strategy_loop_report_path": str(loop_report_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_expansion_sandbox_queue(
        queue_report_path=queue_report_path,
        output_dir=tmp_path / "promotion",
        registry_path=tmp_path / "literature_registry.json",
        current_best_score=118.39,
        force=False,
    )

    assert report["items"][0]["decision"] == "refine"
    assert report["items"][0]["reason_codes"] == ["target_leakage_risk"]
    assert report["items"][0]["score"] == 128.77
    evidence = json.loads(Path(report["items"][0]["evidence_path"]).read_text())
    assert evidence["rules_passed"] is False
    assert evidence["score"] == 128.77
    assert evidence["candidate_gate_issues"] == [
        "target_reference_local_validation_only"
    ]
    assert evidence["diagnostic_only"] is True
    assert evidence["best_valid_score"] == 118.39
    promotion = json.loads(Path(report["items"][0]["promotion_path"]).read_text())
    assert promotion["failed_checks"]["validity"]["failure_code"] == (
        "target_leakage_risk"
    )
    registry = json.loads(Path(report["artifacts"]["registry_path"]).read_text())
    [node] = registry["candidate_nodes"]
    assert node["status"] == "needs_refinement"


def test_evaluate_expansion_sandbox_queue_promotes_checkpoint_family_result(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        evaluate_expansion_sandbox_queue,
    )

    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    candidate_id = "candidate_local_checkpoint_burgers_fno"
    candidates_path = tmp_path / "selected_algorithm_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates_path": str(candidates_path),
                "algorithm_candidate_count": 1,
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "parent_algorithm_card_id": "local_checkpoint_burgers_fno",
                        "candidate_family": "official_checkpoint_family",
                        "executor_spec": {
                            "cli_command": "run-checkpoint-family-sandbox"
                        },
                        "resource_bindings": {
                            "checkpoint_paths": [str(checkpoint)]
                        },
                        "promotion_criteria": {
                            "min_score_delta": 0.01,
                            "max_inference_seconds": 120.0,
                            "requires_ablation": False,
                        },
                        "source_grounding": {
                            "source_ids": [str(checkpoint)],
                            "algorithm_card_id": "local_checkpoint_burgers_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_report_path = tmp_path / "checkpoint_family_sandbox_report.json"
    checkpoint_report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CHECKPOINT_FAMILY_SANDBOX_RAN",
                "checkpoint_family_report_path": str(checkpoint_report_path),
                "best_candidate": {
                    "candidate_id": "fno_calibrated_w1",
                    "checkpoint_path": str(checkpoint),
                    "score": 92.0,
                    "rules_passed": True,
                    "inference_time": 3.0,
                    "calibration_weight": 1.0,
                    "submission_dir": str(tmp_path / "submission"),
                },
                "artifacts": {
                    "checkpoint_family_report_path": str(checkpoint_report_path),
                    "best_submission_dir": str(tmp_path / "submission"),
                },
            }
        ),
        encoding="utf-8",
    )
    queue_report_path = tmp_path / "queue_report.json"
    queue_report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
                "queue_report_path": str(queue_report_path),
                "items": [
                    {
                        "status": "executed",
                        "outcome": "passed",
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": "official_checkpoint_family",
                        "selected_algorithm_candidates_path": str(candidates_path),
                        "checkpoint_family_report_path": str(checkpoint_report_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_expansion_sandbox_queue(
        queue_report_path=queue_report_path,
        output_dir=tmp_path / "promotion",
        registry_path=tmp_path / "literature_registry.json",
        current_best_score=90.0,
        force=False,
    )

    assert report["metrics"]["evidence_count"] == 1
    assert report["metrics"]["promotion_count"] == 1
    assert report["items"][0]["decision"] == "promote"
    evidence = json.loads(Path(report["items"][0]["evidence_path"]).read_text())
    assert evidence["source_report_path"] == str(checkpoint_report_path)
    assert evidence["score"] == 92.0
    assert evidence["runtime"]["inference_seconds"] == 3.0
    registry = json.loads(Path(report["artifacts"]["registry_path"]).read_text())
    by_candidate = {
        candidate["candidate_id"]: candidate
        for candidate in registry["candidate_nodes"]
    }
    assert by_candidate[candidate_id]["status"] == "promoted"
    assert report["next_command"] == "record-strategy-proposal"


def test_evaluate_expansion_sandbox_queue_cli_records_trace(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    queue_report_path = tmp_path / "queue_report.json"
    queue_report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
                "queue_report_path": str(queue_report_path),
                "items": [
                    {
                        "status": "skipped",
                        "outcome": "blocked",
                        "algorithm_candidate_id": "candidate_checkpoint",
                        "candidate_family": "official_checkpoint_family",
                        "reason_codes": [
                            "unsupported_for_auto_short_sandbox:run-checkpoint-family-sandbox"
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-expansion-sandbox-queue",
            "--queue-report-path",
            str(queue_report_path),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--registry-path",
            str(tmp_path / "literature_registry.json"),
            "--current-best-score",
            "90.0",
            "--trace-path",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_EXPANSION_SANDBOX_QUEUE_EVALUATED"
    assert payload["metrics"]["registry_event_count"] == 1
    assert payload["metrics"]["trace_event_recorded"] is True
    event = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "hermes_expansion_sandbox_queue_evaluated"
    assert event["payload"]["registry_event_count"] == 1


def test_prepare_expansion_blocked_executor_authoring_tasks(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.hermes_literature_expansion import (
        prepare_expansion_blocked_executor_authoring_tasks,
    )

    sandbox_plan = tmp_path / "blocked_sandbox_plan.json"
    sandbox_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "sandbox_plan_path": str(sandbox_plan),
                "algorithm_candidate_id": "candidate_batch_strategy",
                "candidate_family": "local_strategy_skill",
                "executor": {"cli_command": "run-hermes-batch-optimization-loop"},
                "blocked": False,
                "selected_candidate": {
                    "algorithm_candidate_id": "candidate_batch_strategy",
                    "candidate_family": "local_strategy_skill",
                    "hypothesis": "Batch search may find a better early patch.",
                    "mechanism": "Run a bounded local search over early patches.",
                    "executor_spec": {
                        "cli_command": "run-hermes-batch-optimization-loop",
                        "implementation_hooks": [
                            "run-hermes-batch-optimization-loop"
                        ],
                    },
                    "source_grounding": {
                        "source_ids": ["local_batch"],
                        "algorithm_card_id": "local_strategy_batch",
                    },
                    "risk_controls": ["can exceed short sandbox budget"],
                },
                "next_command": "run-hermes-batch-optimization-loop",
            }
        ),
        encoding="utf-8",
    )
    queue_report = tmp_path / "queue_report.json"
    queue_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
                "queue_report_path": str(queue_report),
                "items": [
                    {
                        "status": "skipped",
                        "outcome": "blocked",
                        "algorithm_candidate_id": "candidate_batch_strategy",
                        "candidate_family": "local_strategy_skill",
                        "sandbox_plan_path": str(sandbox_plan),
                        "reason_codes": [
                            "unsupported_for_auto_short_sandbox:run-hermes-batch-optimization-loop"
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    target_path = _write_hdf5(
        tmp_path / "target.hdf5",
        np.zeros((2, 200, 256), dtype=np.float32),
    )

    report = prepare_expansion_blocked_executor_authoring_tasks(
        queue_report_path=queue_report,
        output_dir=tmp_path / "authoring",
        context_path=target_path,
        target_path=target_path,
        submission_id="team_blocked_author",
        max_tasks=1,
        force=False,
    )

    assert report["state_hint"] == "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED"
    assert report["metrics"]["task_count"] == 1
    task = report["tasks"][0]
    assert task["algorithm_candidate_id"] == "candidate_batch_strategy"
    assert task["requested_executor_id"] == (
        "bounded_run_hermes_batch_optimization_loop_executor"
    )
    assert Path(task["executor_path"]).exists()
    authoring = json.loads(Path(task["authoring_task_path"]).read_text())
    assert authoring["state_hint"] == "BOUNDED_EXECUTOR_AUTHORING_TASK_PREPARED"
    assert authoring["allowed_edit_paths"] == [task["executor_path"]]
    assert str(target_path) not in authoring["allowed_read_paths"]
    assert authoring["allowed_read_paths"][-1].endswith("context_fixture.hdf5")
    assert report["next_command"] == "audit-bounded-executor-authoring-task"


def test_prepare_expansion_blocked_executor_authoring_tasks_uses_tensor_fixture(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from pde_agent.core.hermes_literature_expansion import (
        prepare_expansion_blocked_executor_authoring_tasks,
    )

    sandbox_plan = tmp_path / "blocked_sandbox_plan.json"
    sandbox_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "sandbox_plan_path": str(sandbox_plan),
                "algorithm_candidate_id": "candidate_batch_strategy",
                "candidate_family": "local_strategy_skill",
                "executor": {"cli_command": "run-hermes-batch-optimization-loop"},
                "selected_candidate": {
                    "algorithm_candidate_id": "candidate_batch_strategy",
                    "candidate_family": "local_strategy_skill",
                    "hypothesis": "Batch search may improve early segment.",
                    "mechanism": "Bound the batch search into a predict adapter.",
                    "executor_spec": {
                        "cli_command": "run-hermes-batch-optimization-loop"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    queue_report = tmp_path / "queue_report.json"
    queue_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
                "items": [
                    {
                        "status": "skipped",
                        "outcome": "blocked",
                        "algorithm_candidate_id": "candidate_batch_strategy",
                        "candidate_family": "local_strategy_skill",
                        "sandbox_plan_path": str(sandbox_plan),
                        "reason_codes": [
                            "unsupported_for_auto_short_sandbox:run-hermes-batch-optimization-loop"
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    target_path = tmp_path / "target.hdf5"
    with h5py.File(target_path, "w") as handle:
        handle.create_dataset("t-coordinate", data=np.arange(200, dtype=np.float32))
        handle.create_dataset(
            "tensor",
            data=np.zeros((2, 200, 256), dtype=np.float32),
        )

    report = prepare_expansion_blocked_executor_authoring_tasks(
        queue_report_path=queue_report,
        output_dir=tmp_path / "authoring",
        context_path=target_path,
        target_path=target_path,
        submission_id="team_blocked_author",
        max_tasks=1,
        force=False,
    )

    authoring = json.loads(Path(report["tasks"][0]["authoring_task_path"]).read_text())
    context_fixture = Path(authoring["allowed_read_paths"][-1])
    with h5py.File(context_fixture, "r") as handle:
        assert list(handle.keys()) == ["tensor"]
        assert handle["tensor"].shape == (2, 10, 256)


def test_prepare_expansion_blocked_executor_authoring_tasks_cli(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    sandbox_plan = tmp_path / "blocked_sandbox_plan.json"
    sandbox_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "sandbox_plan_path": str(sandbox_plan),
                "algorithm_candidate_id": "candidate_batch_strategy",
                "candidate_family": "local_strategy_skill",
                "executor": {"cli_command": "run-hermes-batch-optimization-loop"},
                "selected_candidate": {
                    "algorithm_candidate_id": "candidate_batch_strategy",
                    "candidate_family": "local_strategy_skill",
                    "hypothesis": "Batch search may improve early segment.",
                    "mechanism": "Bound the batch search into a predict adapter.",
                    "executor_spec": {
                        "cli_command": "run-hermes-batch-optimization-loop"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    queue_report = tmp_path / "queue_report.json"
    queue_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
                "items": [
                    {
                        "status": "skipped",
                        "outcome": "blocked",
                        "algorithm_candidate_id": "candidate_batch_strategy",
                        "candidate_family": "local_strategy_skill",
                        "sandbox_plan_path": str(sandbox_plan),
                        "reason_codes": [
                            "unsupported_for_auto_short_sandbox:run-hermes-batch-optimization-loop"
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "prepare-expansion-blocked-executor-authoring-tasks",
            "--queue-report-path",
            str(queue_report),
            "--output-dir",
            str(tmp_path / "authoring"),
            "--context-path",
            str(tmp_path / "context.hdf5"),
            "--target-path",
            str(tmp_path / "target.hdf5"),
            "--submission-id",
            "team_cli_authoring",
            "--max-tasks",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == (
        "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED"
    )
    assert payload["metrics"]["task_count"] == 1
    assert payload["next_hint"] == "audit-bounded-executor-authoring-task"


def test_empty_literature_queue_points_to_local_asset_candidate_recovery(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _blocked_fno_candidates(candidates_path)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [
                    "candidate_fno_blocked",
                ],
                "failed_candidate_keys": [],
                "outcomes": [],
            }
        ),
        encoding="utf-8",
    )

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        failure_memory_path=memory_path,
        force=False,
    )

    assert report["action_count"] == 0
    assert report["skipped_count"] == 1
    assert report["next_command"] == "build-local-asset-algorithm-candidates"
    assert report["recovery_plan"]["reason"] == "candidate_pool_exhausted"
    assert report["recovery_plan"]["next_command"] == (
        "build-local-asset-algorithm-candidates"
    )


def test_build_local_asset_algorithm_candidates_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.local_strategy_library import build_local_strategy_library

    library_path = tmp_path / "library.json"
    output_path = tmp_path / "local_asset_candidates.json"
    build_local_strategy_library(output_path=library_path, force=False)

    result = CliRunner().invoke(
        app,
        [
            "build-local-asset-algorithm-candidates",
            "--local-strategy-library-path",
            str(library_path),
            "--output-path",
            str(output_path),
            "--max-strategy-candidates",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "ALGORITHM_CANDIDATES_COMPILED"
    assert payload["metrics"]["algorithm_candidate_count"] == 2
    assert payload["next_hint"] == "run-hermes-literature-strategy-expansion"


def test_record_hermes_literature_candidate_outcome_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    result = CliRunner().invoke(
        app,
        [
            "record-hermes-literature-candidate-outcome",
            "--algorithm-candidate-id",
            "candidate_fno_blocked",
            "--candidate-key",
            "card_fno|fourier_neural_operator_architecture",
            "--outcome",
            "failed",
            "--reason-code",
            "sandbox_score_regressed",
            "--output-path",
            str(tmp_path / "failure_memory.json"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED"
    assert payload["metrics"]["failed_candidate_count"] == 1


def test_summarize_literature_registry_for_hermes_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    registry_path = tmp_path / "literature_registry.json"
    summary_path = tmp_path / "hermes_registry_summary.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_UPDATED",
                "sources": [],
                "algorithm_nodes": [],
                "candidate_nodes": [
                    {
                        "candidate_id": "candidate_residual",
                        "candidate_family": "lightweight_residual_corrector",
                        "status": "compiled",
                        "executor": "run-blueprint-sandbox",
                        "source_ids": [],
                    }
                ],
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "summarize-literature-registry-for-hermes",
            "--registry-path",
            str(registry_path),
            "--output-path",
            str(summary_path),
            "--current-best-score",
            "86.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT"
    assert payload["metrics"]["candidate_action_count"] == 1
    assert payload["next_hint"] == "run-blueprint-sandbox"
    assert summary_path.exists()


def test_hermes_literature_expansion_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    candidates_path = tmp_path / "algorithm_candidates.json"
    _blocked_fno_candidates(candidates_path)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-literature-strategy-expansion",
            "--algorithm-candidates-path",
            str(candidates_path),
            "--workspace-dir",
            str(tmp_path / "workspace"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN"
    assert payload["metrics"]["proposal_count"] == 1
    assert payload["next_hint"] == "build-strategy-rubric"
