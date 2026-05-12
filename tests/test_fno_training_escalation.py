"""TDD tests for FNO training escalation pipeline."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import h5py
import numpy as np
import pytest


class TestEscalationPipeline:
    def test_pipeline_discovers_candidates(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_escalation import (
            CANDIDATES_DISCOVERED_STATE_HINT,
            run_fno_training_escalation_pipeline,
        )

        memory = {
            "schema_version": 1,
            "family_outcomes": {
                "fno": {"status": "blocked", "failure_reasons": ["diverged"]},
            },
        }
        mem_path = tmp_path / "memory.json"
        mem_path.write_text(json.dumps(memory))
        out_path = tmp_path / "escalation_report.json"

        mock_candidates = {
            "algorithm_candidate_count": 2,
            "algorithm_candidates_path": str(tmp_path / "candidates.json"),
            "algorithm_candidates": [
                {"candidate_family": "unet", "priority": 1},
                {"candidate_family": "deeponet", "priority": 2},
            ],
        }
        mock_expansion = {
            "expansion_report_path": str(tmp_path / "expansion.json"),
            "action_count": 1,
            "executable_count": 1,
            "proposal_count": 0,
            "sandbox_queue": [str(tmp_path / "sandbox_plan.json")],
            "strategy_growth_queue": [],
        }

        with patch(
            "pde_agent.core.literature_expansion.build_local_asset_algorithm_candidates",
            return_value=mock_candidates,
        ), patch(
            "pde_agent.core.literature_expansion.run_literature_strategy_expansion",
            return_value=mock_expansion,
        ):
            result = run_fno_training_escalation_pipeline(
                workspace_dir=str(tmp_path),
                memory_path=str(mem_path),
                output_path=str(out_path),
                max_strategy_candidates=5,
                force=True,
            )

        assert result["state_hint"] == CANDIDATES_DISCOVERED_STATE_HINT
        assert result["candidate_count"] == 2
        assert result["expansion_action_count"] == 1
        assert result["expansion_executable_count"] == 1
        assert result["sandbox_queue"] == [str(tmp_path / "sandbox_plan.json")]
        assert result["next_command"] == "run-expansion-sandbox-queue"
        assert result["blocked_families"] == ["fno"]
        assert result["failure_reasons"] == ["diverged"]
        assert out_path.exists()
        assert (tmp_path / "fno_training_memory_escalated.json").exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["state_hint"] == CANDIDATES_DISCOVERED_STATE_HINT

    def test_pipeline_no_candidates(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_escalation import (
            NO_CANDIDATES_STATE_HINT,
            run_fno_training_escalation_pipeline,
        )

        memory = {
            "schema_version": 1,
            "family_outcomes": {},
        }
        mem_path = tmp_path / "memory.json"
        mem_path.write_text(json.dumps(memory))
        out_path = tmp_path / "escalation_report.json"

        mock_candidates = {
            "algorithm_candidate_count": 0,
            "algorithm_candidates": [],
        }

        with patch(
            "pde_agent.core.literature_expansion.build_local_asset_algorithm_candidates",
            return_value=mock_candidates,
        ):
            result = run_fno_training_escalation_pipeline(
                workspace_dir=str(tmp_path),
                memory_path=str(mem_path),
                output_path=str(out_path),
                max_strategy_candidates=5,
                force=True,
            )

        assert result["state_hint"] == NO_CANDIDATES_STATE_HINT
        assert result["blocked_families"] == []
        assert "agent_instructions" in result
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["state_hint"] == NO_CANDIDATES_STATE_HINT

    def test_pipeline_inherits_blocked_families_from_health_feedback(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_escalation import (
            run_fno_training_escalation_pipeline,
        )

        memory = {
            "schema_version": 1,
            "family_outcomes": {},
        }
        mem_path = tmp_path / "memory.json"
        mem_path.write_text(json.dumps(memory))
        (tmp_path / "fno_training_health_feedback.json").write_text(
            json.dumps(
                {
                    "state_hint": "FNO_TRAINING_HEALTH_FEEDBACK_BUILT",
                    "search_guidance": [
                        {
                            "guidance_type": "operator_training_policy",
                            "blocked_families": ["fno_lite"],
                        }
                    ],
                }
            )
        )

        mock_candidates = {
            "algorithm_candidate_count": 1,
            "algorithm_candidates_path": str(tmp_path / "candidates.json"),
            "algorithm_candidates": [{"candidate_family": "local_strategy_skill"}],
        }
        mock_expansion = {
            "expansion_report_path": str(tmp_path / "expansion.json"),
            "action_count": 0,
            "executable_count": 0,
            "proposal_count": 0,
            "sandbox_queue": [],
            "strategy_growth_queue": [],
        }

        with patch(
            "pde_agent.core.literature_expansion.build_local_asset_algorithm_candidates",
            return_value=mock_candidates,
        ), patch(
            "pde_agent.core.literature_expansion.run_literature_strategy_expansion",
            return_value=mock_expansion,
        ):
            result = run_fno_training_escalation_pipeline(
                workspace_dir=str(tmp_path),
                memory_path=str(mem_path),
                max_strategy_candidates=5,
                force=True,
            )

        assert result["blocked_families"] == ["fno_lite"]
        assert "blocked_by_fno_training_health_feedback" in result["failure_reasons"]
        merged = json.loads((tmp_path / "fno_training_memory_escalated.json").read_text())
        assert merged["family_outcomes"]["fno_lite"]["status"] == "blocked"

    def test_pipeline_missing_memory(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_escalation import (
            FnoTrainingEscalationError,
            run_fno_training_escalation_pipeline,
        )

        with pytest.raises(FnoTrainingEscalationError) as exc_info:
            run_fno_training_escalation_pipeline(
                workspace_dir=str(tmp_path),
                memory_path=str(tmp_path / "nonexistent.json"),
                max_strategy_candidates=5,
                force=True,
            )
        assert "could not load memory" in str(exc_info.value)

    def test_pipeline_candidate_discovery_failure(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_escalation import (
            FnoTrainingEscalationError,
            run_fno_training_escalation_pipeline,
        )

        memory = {"schema_version": 1, "family_outcomes": {}}
        mem_path = tmp_path / "memory.json"
        mem_path.write_text(json.dumps(memory))

        with patch(
            "pde_agent.core.literature_expansion.build_local_asset_algorithm_candidates",
            side_effect=RuntimeError("library corrupt"),
        ):
            with pytest.raises(FnoTrainingEscalationError) as exc_info:
                run_fno_training_escalation_pipeline(
                    workspace_dir=str(tmp_path),
                    memory_path=str(mem_path),
                    max_strategy_candidates=5,
                    force=True,
                )
        assert "candidate discovery failed" in str(exc_info.value)
        assert "library corrupt" in str(exc_info.value)


def test_expansion_sandbox_queue_skips_local_strategy_when_reference_horizon_is_short(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_expansion import run_expansion_sandbox_queue

    context = tmp_path / "context.hdf5"
    target = tmp_path / "target.hdf5"
    reference = tmp_path / "reference_41.hdf5"
    with h5py.File(context, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))
    with h5py.File(target, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    with h5py.File(reference, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((64, 41, 256), dtype=np.float32))
    initial_submission = tmp_path / "submission"
    initial_submission.mkdir()

    candidate = {
        "algorithm_candidate_id": "candidate_local_strategy_task1_early_subsegment_patch_v0",
        "candidate_family": "local_strategy_skill",
        "executor_spec": {
            "cli_command": "run-batch-optimization-loop",
            "implementation_hooks": [
                "hypothesis_lab_early_subsegment_patch_search"
            ],
        },
        "parameter_grid": {},
    }
    plan = tmp_path / "sandbox_plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "algorithm_candidate_id": candidate["algorithm_candidate_id"],
                "candidate_family": "local_strategy_skill",
                "selected_candidate": candidate,
                "executor": {"cli_command": "run-batch-optimization-loop"},
            }
        ),
        encoding="utf-8",
    )
    expansion = tmp_path / "expansion.json"
    expansion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_STRATEGY_EXPANSION_RAN",
                "sandbox_queue": [str(plan)],
            }
        ),
        encoding="utf-8",
    )

    report = run_expansion_sandbox_queue(
        expansion_report_path=expansion,
        context_path=context,
        target_path=target,
        initial_submission_dir=initial_submission,
        output_dir=tmp_path / "queue",
        submission_id="test",
        max_items=None,
        failure_memory_path=tmp_path / "failure_memory.json",
        reference_path=reference,
        reference_sample_limit=None,
        timing_profile=None,
        timing_factor=None,
        selected_algorithm_candidate_id=None,
        local_strategy_runner=lambda **_: pytest.fail("runner should be skipped"),
        force=False,
    )

    assert report["metrics"]["skipped_count"] == 1
    item = report["items"][0]
    assert item["status"] == "skipped"
    assert item["outcome"] == "blocked"
    assert item["reason_codes"] == [
        "local_strategy_reference_horizon_incompatible:expected_at_least_200_got_41"
    ]


def test_expansion_sandbox_queue_classifies_reference_shape_errors_as_blocked(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_expansion import (
        evaluate_expansion_sandbox_queue,
        _error_expansion_sandbox_item,
    )
    from pde_agent.core.task1_retrieval_prior import Task1RetrievalPriorError

    context = tmp_path / "context.hdf5"
    target = tmp_path / "target.hdf5"
    reference = tmp_path / "reference_41.hdf5"
    with h5py.File(context, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))
    with h5py.File(target, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    with h5py.File(reference, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((64, 41, 256), dtype=np.float32))

    candidate_id = "candidate_local_strategy_task1_retrieval_refinement_v0"
    candidate = {
        "algorithm_candidate_id": candidate_id,
        "candidate_family": "local_strategy_skill",
        "executor_spec": {
            "cli_command": "run-hypothesis-lab",
            "implementation_hooks": ["run-hypothesis-lab", "task1_retrieval_refinement"],
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": False,
        },
    }
    plan = tmp_path / "sandbox_plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_SANDBOX_PLANNED",
                "algorithm_candidate_id": candidate_id,
                "candidate_family": "local_strategy_skill",
                "selected_candidate": candidate,
                "executor": {"cli_command": "run-hypothesis-lab"},
            }
        ),
        encoding="utf-8",
    )
    expansion = tmp_path / "expansion.json"
    expansion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_STRATEGY_EXPANSION_RAN",
                "sandbox_queue": [str(plan)],
            }
        ),
        encoding="utf-8",
    )

    def failing_hypothesis_lab(**_: Any) -> dict[str, Any]:
        raise Task1RetrievalPriorError(
            "expected reference shape [N, 200, 256], got (512, 41, 256)"
        )

    item = _error_expansion_sandbox_item(
        plan=json.loads(plan.read_text(encoding="utf-8")),
        plan_file=plan,
        failure_memory_path=tmp_path / "failure_memory.json",
        error=Task1RetrievalPriorError(
            "expected reference shape [N, 200, 256], got (512, 41, 256)"
        ),
    )

    assert item["status"] == "error"
    assert item["outcome"] == "blocked"
    assert "reference_horizon_incompatible:expected_200_got_41" in item["reason_codes"]

    queue_report = tmp_path / "queue_report.json"
    queue_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXPANSION_SANDBOX_QUEUE_RAN",
                "queue_report_path": str(queue_report),
                "items": [item],
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_expansion_sandbox_queue(
        queue_report_path=queue_report,
        output_dir=tmp_path / "eval",
        registry_path=tmp_path / "registry.json",
        current_best_score=59.0,
        force=False,
    )
    evaluated_item = evaluation["items"][0]
    assert evaluated_item["decision"] == "blocked"
    assert evaluated_item["reason_codes"] == ["data_contract_incompatible"]


def test_summarize_literature_registry_cli_command_is_available(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_UPDATED",
                "registry_path": str(registry),
                "sources": [],
                "source_count": 0,
                "algorithm_nodes": [],
                "algorithm_node_count": 0,
                "candidate_nodes": [],
                "candidate_node_count": 0,
                "events": [],
                "event_count": 0,
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "summarize-literature-registry",
            "--registry-path",
            str(registry),
            "--output-path",
            str(tmp_path / "summary.json"),
            "--current-best-score",
            "59.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "summarize-literature-registry"
    assert payload["state_hint"] == "LITERATURE_REGISTRY_SUMMARY_BUILT"


def test_run_literature_strategy_expansion_cli_command_is_available(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    candidates = tmp_path / "algorithm_candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates_path": str(candidates),
                "algorithm_candidate_count": 1,
                "algorithm_candidates": [
                        {
                            "algorithm_candidate_id": "candidate_review_only",
                            "candidate_family": "review_only",
                            "hypothesis": "A bounded executor is needed before testing.",
                            "mechanism": "Route unsupported hooks to executor design.",
                            "executor_spec": {
                            "cli_command": "blocked_requires_executor_design",
                            "unsupported_hook": "review_only_hook",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run-literature-strategy-expansion",
            "--algorithm-candidates-path",
            str(candidates),
            "--workspace-dir",
            str(tmp_path / "workspace"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "run-literature-strategy-expansion"
    assert payload["state_hint"] == "LITERATURE_STRATEGY_EXPANSION_RAN"


def test_registry_summary_routes_data_contract_and_refinement_actions(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_expansion import summarize_literature_registry

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_UPDATED",
                "registry_path": str(registry),
                "sources": [],
                "source_count": 0,
                "algorithm_nodes": [
                    {
                        "node_id": "algorithm_refine",
                        "algorithm_card_id": "local_strategy_refine",
                        "candidate_ids": ["candidate_refine"],
                    }
                ],
                "algorithm_node_count": 1,
                "candidate_nodes": [
                    {
                        "candidate_id": "candidate_data_contract",
                        "candidate_family": "local_strategy_skill",
                        "status": "blocked",
                        "failure_history": [
                            {
                                "failure_codes": ["data_contract_incompatible"],
                                "retry_allowed": True,
                            }
                        ],
                    },
                    {
                        "candidate_id": "candidate_refine",
                        "candidate_family": "local_strategy_skill",
                        "status": "needs_refinement",
                        "executor": "run-hypothesis-lab",
                        "failure_history": [
                            {
                                "failure_codes": ["ablation_missing_or_weak"],
                                "retry_allowed": True,
                                "metric_results": {
                                    "score": 81.0,
                                    "score_delta_vs_current_best": 22.0,
                                },
                            }
                        ],
                    },
                ],
                "candidate_node_count": 2,
                "events": [],
                "event_count": 0,
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_literature_registry(
        registry_path=registry,
        output_path=tmp_path / "summary.json",
        current_best_score=59.0,
        force=False,
    )

    actions = {
        item["candidate_id"]: item for item in summary["candidate_action_queue"]
    }
    assert actions["candidate_data_contract"]["recommended_action"] == (
        "provide_compatible_reference_data"
    )
    assert actions["candidate_data_contract"]["next_command"] == (
        "prepare-compatible-reference-data"
    )
    assert actions["candidate_refine"]["recommended_action"] == (
        "refine_with_ablation_or_variant"
    )
    assert actions["candidate_refine"]["next_command"] == "refine-algorithm-candidate"
    signals = {item["signal"] for item in summary["learning_signals"]}
    assert "data_contract_blocks_candidate" in signals
    assert "candidate_refinement_required" in signals

    refinement = __import__(
        "pde_agent.core.literature_expansion",
        fromlist=["refine_algorithm_candidate"],
    ).refine_algorithm_candidate(
        registry_summary_path=summary["summary_path"],
        output_path=tmp_path / "refinement.json",
        algorithm_candidate_id="candidate_refine",
        force=False,
    )
    assert refinement["refinement_type"] == "ablation_or_variant_refinement"
    assert refinement["suggested_sandbox_args"]["reference_path_required"] is False
    assert refinement["suggested_sandbox_args"]["learning_objective"] == (
        "ablation_or_variant_evidence"
    )


def test_candidate_expansion_handoff_mutates_refinement_actions(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_expansion import run_candidate_expansion_handoff

    library = tmp_path / "local_strategy_library.json"
    library.write_text(
        json.dumps(
                {
                    "schema_version": 1,
                    "state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT",
                    "entries": [
                    {
                        "strategy_id": "task1_early_transition_expert_v0",
                        "priority": 1,
                        "title": "Early transition expert",
                        "source_type": "local_strategy",
                        "model_families": ["FNO"],
                        "hypothesis": "Improve early transition dynamics.",
                        "mechanism": "Fit a local early transition map.",
                        "implementation_hooks": [
                            "run-hypothesis-lab",
                            "task1_early_transition_expert",
                        ],
                        "known_risks": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "candidate_plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "expansion_required": True,
                "expansion_handoff": {
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library),
                        "max_strategy_candidates": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    registry_summary = tmp_path / "registry_summary.json"
    registry_summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LITERATURE_REGISTRY_SUMMARY_BUILT",
                "candidate_action_queue": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_transition_expert_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "recommended_action": "refine_with_ablation_or_variant",
                        "failure_codes": ["ablation_missing_or_weak"],
                        "reason": "candidate_needs_refinement",
                    }
                ],
                "avoid_repetition": [],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_expansion_handoff(
        candidate_plan_path=plan,
        workspace_dir=tmp_path / "workspace",
        local_strategy_library_path=library,
        checkpoint_paths=None,
        max_strategy_candidates=1,
        failure_memory_path=None,
        registry_summary_path=registry_summary,
        force=False,
    )

    candidates_path = Path(report["artifacts"]["algorithm_candidates_path"])
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    [candidate] = candidates["algorithm_candidates"]
    assert candidate["algorithm_candidate_id"].endswith(
        "mutation_robust_reference_ablation_v0"
    )
    assert candidate["mutation_evidence"]["parent_algorithm_candidate_id"] == (
        "candidate_local_strategy_task1_early_transition_expert_v0"
    )
    assert candidates["learning_constraint_skipped_candidates"][0][
        "mutation_generated"
    ] is True


def test_registry_summary_uses_mutation_evidence_to_advance_mutation_axis(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        evaluate_algorithm_candidate_promotion,
        record_literature_registry_event,
        update_literature_registry,
    )
    from pde_agent.core.literature_expansion import (
        run_candidate_expansion_handoff,
        summarize_literature_registry,
    )

    parent_id = "candidate_local_strategy_task1_early_transition_expert_v0"
    mutation_id = f"{parent_id}_mutation_robust_reference_ablation_v0"
    candidates = tmp_path / "algorithm_candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": mutation_id,
                        "candidate_family": "local_strategy_skill",
                        "executor_spec": {"cli_command": "run-hypothesis-lab"},
                        "promotion_criteria": {
                            "min_score_delta": 0.01,
                            "max_inference_seconds": 120.0,
                            "requires_ablation": True,
                        },
                        "mutation_evidence": {
                            "parent_algorithm_candidate_id": parent_id,
                            "parent_candidate_family": "local_strategy_skill",
                            "mutation_type": "robust_reference_ablation_retry",
                            "reason": "local_strategy_failed_legal_retest_requires_variant",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "sandbox_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "state_hint": "SANDBOX_EVIDENCE_RECORDED",
                "evidence_id": "evidence_mutation",
                "candidate_id": mutation_id,
                "rules_passed": True,
                "score": 81.0,
                "runtime": {"inference_seconds": 1.0},
                "physics_metrics": {"residual_delta": 0.0},
                "ablation_metrics": {"score_delta": 0.0},
                "score_delta_vs_current_best": 22.0,
                "candidate_gate_issues": [],
                "artifact_paths": [],
                "reproducibility": {"deterministic": True},
            }
        ),
        encoding="utf-8",
    )
    promotion = evaluate_algorithm_candidate_promotion(
        algorithm_candidates_path=candidates,
        algorithm_candidate_id=mutation_id,
        sandbox_evidence_paths=[evidence],
        output_path=tmp_path / "promotion.json",
        current_best_score=59.0,
        force=False,
    )
    assert promotion["mutation_evidence"]["mutation_type"] == (
        "robust_reference_ablation_retry"
    )
    event = record_literature_registry_event(
        event_source_path=promotion["algorithm_candidate_promotion_path"],
        output_path=tmp_path / "event.json",
        force=False,
    )
    registry = update_literature_registry(
        registry_path=tmp_path / "registry.json",
        algorithm_candidates_paths=[candidates],
        event_paths=[event["literature_registry_event_path"]],
        force=False,
    )
    summary = summarize_literature_registry(
        registry_path=registry["registry_path"],
        output_path=tmp_path / "summary.json",
        current_best_score=59.0,
        force=False,
    )
    [action] = summary["candidate_action_queue"]
    assert action["mutation_evidence"]["mutation_type"] == (
        "robust_reference_ablation_retry"
    )

    library = tmp_path / "local_strategy_library.json"
    library.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT",
                "entries": [
                    {
                        "strategy_id": "task1_early_transition_expert_v0",
                        "priority": 1,
                        "title": "Early transition expert",
                        "source_type": "local_strategy",
                        "model_families": ["FNO"],
                        "hypothesis": "Improve early transition dynamics.",
                        "mechanism": "Fit a local early transition map.",
                        "implementation_hooks": [
                            "run-hypothesis-lab",
                            "task1_early_transition_expert",
                        ],
                        "known_risks": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "candidate_plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
                "expansion_required": True,
                "expansion_handoff": {
                    "recommended_inputs": {
                        "local_strategy_library_path": str(library),
                        "max_strategy_candidates": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    handoff = run_candidate_expansion_handoff(
        candidate_plan_path=plan,
        workspace_dir=tmp_path / "workspace",
        local_strategy_library_path=library,
        checkpoint_paths=None,
        max_strategy_candidates=1,
        failure_memory_path=None,
        registry_summary_path=summary["summary_path"],
        force=False,
    )
    generated = json.loads(
        Path(handoff["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"][0]
    assert generated["mutation_evidence"]["mutation_type"] == (
        "source_stability_ablation_retry"
    )
