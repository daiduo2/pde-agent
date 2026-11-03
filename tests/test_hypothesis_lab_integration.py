"""End-to-end integration tests for the full hypothesis-lab autonomous pipeline.

Tests the complete chain: session → cards → hypotheses → blueprints → sandbox
→ outcome → doom-loop → failure taxonomy → strategy recommendation →
creative escalation trigger.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def _build_report_path(tmp_path: Path, name: str) -> Path:
    return tmp_path / name


def _linear_trend_algorithm_candidates(path: Path) -> Path:
    payload = {
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "algorithm_candidates": [
            {
                "algorithm_candidate_id": "lt_001",
                "candidate_family": "linear_trend_spectral",
                "executor_spec": {
                    "cli_command": "run-blueprint-sandbox",
                    "implementation_hooks": ["linear_trend_spectral"],
                },
                "hypothesis": "Linear trend with spectral smoothing should improve late segments.",
                "mechanism": "Extrapolate linear velocity and apply spectral filter.",
                "target_problem": "PDE_Burgers",
                "params": {
                    "slope_window": 5,
                    "gain": -0.1,
                    "decay": 60.0,
                    "spectral_strength": 1.0,
                },
                "score": 14.94,
                "task1_total_score_proxy": 86.19,
                "source_type": "manual_operator_tuning",
            }
        ]
    }
    path.write_text(json.dumps(payload))
    return path


def _make_synthetic_burgers(
    n_samples: int = 1,
    t_total: int = 200,
    x_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic Burgers-like data for sandbox testing."""
    rng = np.random.default_rng(42)
    context = rng.random((n_samples, 10, x_size), dtype=np.float32)
    target = rng.random((n_samples, t_total, x_size), dtype=np.float32)
    # Make target somewhat correlated with context for realism
    target[:, :10, :] = context
    return context, target


# ---------------------------------------------------------------------------
# Scenario 1: Full pipeline on session (end-to-end)
# ---------------------------------------------------------------------------


class TestFullPipelineOnSession:
    def test_pipeline_completes_all_stages(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_primitives import (
            build_knowledge_cards_primitive,
            design_algorithm_blueprints_primitive,
            generate_hypotheses_primitive,
            record_sandbox_outcome_primitive,
            run_blueprint_sandbox_primitive,
        )
        from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

        workspace = tmp_path / "session_pipeline"
        session = HypothesisLabSession.create(workspace=str(workspace))

        # Stage 1: build cards
        cards = build_knowledge_cards_primitive(
            session,
            manual_report_paths=[],
            algorithm_candidate_paths=[
                str(_linear_trend_algorithm_candidates(tmp_path / "candidates.json"))
            ],
        )
        assert cards["card_count"] >= 1
        assert session.get_artifact("knowledge_cards", "knowledge_cards") is not None

        # Stage 2: generate hypotheses
        hypotheses = generate_hypotheses_primitive(session)
        assert hypotheses["hypothesis_count"] >= 1
        assert session.get_artifact("hypotheses", "hypotheses") is not None

        # Stage 3: design blueprints
        blueprints = design_algorithm_blueprints_primitive(session)
        assert blueprints["blueprint_count"] >= 1
        assert session.get_artifact("blueprints", "blueprints") is not None

        # Stage 4: run sandbox (with synthetic data)
        context, target = _make_synthetic_burgers()
        context_path = _write_hdf5(tmp_path / "ctx.h5", context)
        target_path = _write_hdf5(tmp_path / "tgt.h5", target)

        sandbox = run_blueprint_sandbox_primitive(
            session,
            context_path=context_path,
            target_path=target_path,
            blueprint_index=0,
        )
        assert sandbox["state_hint"] == "HYPOTHESIS_LAB_SANDBOX_VERIFIED"
        assert "best_candidate" in sandbox
        assert "post_mortem" in sandbox  # Layer 1: diagnostic
        assert sandbox["post_mortem"]["worst_segment"] in {"early", "mid", "late", "ok"}
        assert "root_cause_hint" in sandbox["post_mortem"]
        assert session.get_artifact("sandbox", "sandbox") is not None

        # Verify EVENT_STAGE_COMPLETED has enriched payload (doom-loop fix)
        events = session.get_events(event_type="stage_completed")
        sandbox_events = [e for e in events if e.get("payload", {}).get("stage") == "run_blueprint_sandbox"]
        assert len(sandbox_events) >= 1
        payload = sandbox_events[0]["payload"]
        assert "candidate_family" in payload
        assert "score_delta" in payload
        assert "outcome" in payload
        assert "worst_segment" in payload
        assert "root_cause_hint" in payload

        # Stage 5: record outcome (best-effort: may not have candidate_outcome
        # if executable_blueprint cannot be matched)
        outcome = record_sandbox_outcome_primitive(session)
        assert "sandbox_artifact_id" in outcome
        assert session.get_artifact("outcome", "outcome") is not None

        # Verify session state summary reflects all stages
        summary = session.get_state_summary()
        assert summary["artifact_counts"]["knowledge_cards"] >= 1
        assert summary["artifact_counts"]["hypotheses"] >= 1
        assert summary["artifact_counts"]["blueprints"] >= 1
        assert summary["artifact_counts"]["sandbox"] >= 1
        assert summary["artifact_counts"]["outcome"] >= 1


# ---------------------------------------------------------------------------
# Scenario 2: Autonomous loop deterministic mode
# ---------------------------------------------------------------------------


class TestAutonomousLoopDeterministic:
    def test_runs_full_pipeline_and_reports_results(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_autonomous_loop import (
            AutonomousHypothesisLab,
        )
        from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

        workspace = tmp_path / "autonomous_test"
        session = HypothesisLabSession.create(workspace=str(workspace))

        context, target = _make_synthetic_burgers()
        context_path = _write_hdf5(tmp_path / "ctx.h5", context)
        target_path = _write_hdf5(tmp_path / "tgt.h5", target)

        engine = AutonomousHypothesisLab(
            session=session,
            context_path=str(context_path),
            target_path=str(target_path),
            manual_report_paths=[
                str(_linear_trend_algorithm_candidates(tmp_path / "candidates.json"))
            ],
        )

        result = engine.run(max_iterations=10, autonomy_level="deterministic")

        assert result["status"] == "ok"
        assert result["state_hint"] == "HYPOTHESIS_LAB_AUTONOMOUS_LOOP_RAN"
        # The loop runs until report_results; with synthetic data some stages
        # may fail but the loop should still report results.
        assert result["iterations"] >= 1
        assert result["metrics"]["total_artifacts"] >= 1

        # Verify all stage artifacts exist
        artifacts = result["artifacts"]
        assert artifacts["session_id"] == session.session_id
        assert "workspace" in artifacts


# ---------------------------------------------------------------------------
# Scenario 3: Doom-loop detection triggers after repeated failures
# ---------------------------------------------------------------------------


class TestDoomLoopIntegration:
    def test_doom_loop_detects_repeated_family(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_doom_loop import check_hypothesis_doom_loop
        from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

        workspace = tmp_path / "doom_test"
        session = HypothesisLabSession.create(workspace=str(workspace))

        # Emit repeated sandbox events for the same family with poor scores
        for i in range(5):
            session.emit(
                event_type="stage_completed",
                payload={
                    "stage": "run_blueprint_sandbox",
                    "candidate_family": "linear_trend_spectral",
                    "algorithm_candidate_id": f"candidate_{i}",
                    "score_delta": -0.5,
                    "outcome": "failed",
                },
                tags=["stage", "sandbox"],
            )

        trace_path = workspace / "events.jsonl"
        assert trace_path.exists()

        report = check_hypothesis_doom_loop(
            trace_path=trace_path,
            output_path=None,
            recent_window=20,
            repeated_family_threshold=3,
            failed_family_threshold=2,
            min_score_delta=0.1,
            force=True,
        )

        assert report["guard_active"] is True
        assert "linear_trend_spectral" in report["blocked_candidate_families"]
        assert report["decision"] == "inject_corrective_prompt"


# ---------------------------------------------------------------------------
# Scenario 4: Failure taxonomy + strategy recommendation
# ---------------------------------------------------------------------------


class TestFailureTaxonomyAndStrategy:
    def test_classify_failure_tags_sandbox_result(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_failure_taxonomy import classify_failure

        sandbox_result = {
            "post_mortem": {
                "root_cause_hint": "early_shock_smearing",
                "context_preserved": True,
            },
            "verifier": {
                "rules_passed": False,
                "score_delta_vs_baseline": -1.5,
            },
        }

        result = classify_failure(sandbox_result)
        assert result["primary_type"] == "PHYSICAL"
        assert result["primary_subtype"] == "shock_capturing_failure"
        assert result["root_cause_hint"] == "early_shock_smearing"
        assert result["confidence"] in {"high", "medium", "low"}

    def test_strategy_recommends_family_from_diagnosis(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_failure_strategy import (
            recommend_family_given_diagnosis,
        )

        diagnosis = {
            "failed_family": "task1_retrieval_refinement",
            "root_cause": "early_shock_smearing",
            "worst_segment": "early",
        }

        rules = [
            {
                "rule_id": "rule_001",
                "condition": {
                    "failed_family": "task1_retrieval_refinement",
                    "root_cause": "early_shock_smearing",
                },
                "action": "try_family",
                "target": "task1_early_transition_expert",
                "confidence": 0.8,
                "support": 4,
                "supporting_sessions": ["sess_001", "sess_002", "sess_003", "sess_004"],
            }
        ]

        available = {"task1_early_transition_expert", "task1_sem_burgers_rollout"}
        rec = recommend_family_given_diagnosis(diagnosis, rules, available)

        assert rec is not None
        assert rec["target_family"] == "task1_early_transition_expert"
        assert rec["confidence"] == 0.8

    def test_diagnose_session_family_aggregates_correctly(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_failure_taxonomy import (
            diagnose_session_family,
        )
        from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

        workspace = tmp_path / "diag_test"
        session = HypothesisLabSession.create(workspace=str(workspace))

        # Put a sandbox artifact with post_mortem
        session.put_artifact(
            artifact_type="sandbox",
            artifact_id="sb_001",
            payload={
                "candidate_family": "linear_trend_spectral",
                "best_candidate": {
                    "candidate_family": "linear_trend_spectral",
                    "task1_total_score_proxy": 10.0,
                },
                "post_mortem": {
                    "root_cause_hint": "late_drift",
                    "worst_segment": "late",
                },
                "verifier": {
                    "rules_passed": False,
                    "score_delta_vs_baseline": -0.8,
                },
            },
        )

        diagnosis = diagnose_session_family(session, "linear_trend_spectral")
        assert diagnosis["family"] == "linear_trend_spectral"
        assert diagnosis["sandbox_count"] >= 1
        assert "root_causes" in diagnosis
        assert "late_drift" in diagnosis["root_causes"]


# ---------------------------------------------------------------------------
# Scenario 5: Creative escalation trigger
# ---------------------------------------------------------------------------


class TestCreativeEscalationTrigger:
    def test_triggers_when_all_families_blocked(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_autonomous_loop import (
            AutonomousHypothesisLab,
        )
        from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

        workspace = tmp_path / "creative_test"
        session = HypothesisLabSession.create(workspace=str(workspace))

        # Seed the session with a sandbox artifact so recommendation reaches report_results
        session.put_artifact(
            artifact_type="knowledge_cards",
            artifact_id="knowledge_cards",
            payload={"card_count": 1, "cards": []},
        )
        session.put_artifact(
            artifact_type="hypotheses",
            artifact_id="hypotheses",
            payload={"hypothesis_count": 1, "hypotheses": []},
        )
        session.put_artifact(
            artifact_type="blueprints",
            artifact_id="blueprints",
            payload={"blueprint_count": 1, "blueprints": []},
        )
        session.put_artifact(
            artifact_type="sandbox",
            artifact_id="sandbox",
            payload={
                "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
                "candidate_family": "linear_trend_spectral",
                "best_candidate": {
                    "candidate_id": "lt_001",
                    "candidate_family": "linear_trend_spectral",
                    "task1_total_score_proxy": 10.0,
                },
                "verifier": {
                    "rules_passed": False,
                    "score_delta_vs_baseline": -0.5,
                },
                "post_mortem": {
                    "root_cause_hint": "late_drift",
                    "worst_segment": "late",
                },
            },
        )
        session.put_artifact(
            artifact_type="outcome",
            artifact_id="outcome",
            payload={"has_pareto_memory": True},
        )

        # Emit repeated sandbox failure events so doom-loop triggers
        for i in range(3):
            session.emit(
                event_type="stage_completed",
                payload={
                    "stage": "run_blueprint_sandbox",
                    "candidate_family": "linear_trend_spectral",
                    "algorithm_candidate_id": f"lt_{i}",
                    "score_delta": -0.5,
                    "outcome": "failed",
                },
                tags=["stage", "sandbox"],
            )

        engine = AutonomousHypothesisLab(session=session)

        # In deterministic mode: should NOT escalate, just report results
        decision = engine._decide({}, autonomy_level="deterministic")
        assert decision["action"] == "report_results"

        # In full mode: should trigger generate_new_executor with escalation
        decision = engine._decide({}, autonomy_level="full")
        assert decision["action"] == "generate_new_executor"
        assert decision["escalation_required"] is True
        assert "creative_context" in decision

    def test_builds_creative_context(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_creative_escalation import (
            build_creative_escalation_context,
        )
        from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

        workspace = tmp_path / "creative_ctx_test"
        session = HypothesisLabSession.create(workspace=str(workspace))

        # Seed with minimal sandbox artifact
        session.put_artifact(
            artifact_type="sandbox",
            artifact_id="sandbox",
            payload={
                "candidate_family": "linear_trend_spectral",
                "best_candidate": {
                    "candidate_id": "lt_001",
                    "candidate_family": "linear_trend_spectral",
                    "task1_total_score_proxy": 10.0,
                },
                "verifier": {"score_delta_vs_baseline": -0.5},
            },
        )

        ctx = build_creative_escalation_context(session)
        assert "session_summary" in ctx
        assert "best_evidence" in ctx
        assert "failure_patterns" in ctx
        assert "score_trajectory" in ctx
        assert "prompt_template" in ctx
        assert "Write a Python function `predict(context, target=None, reference=None, **kwargs)`" in ctx["prompt_template"]


# ---------------------------------------------------------------------------
# Scenario 6: Memory cross-session persistence
# ---------------------------------------------------------------------------


class TestMemoryCrossSession:
    def test_memory_records_and_queries_experiments(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_memory import HypothesisLabMemory

        memory_path = tmp_path / "memory.jsonl"
        memory = HypothesisLabMemory(memory_path)

        memory.record_experiment("sess_001", {
            "best_candidate": {
                "candidate_family": "task1_retrieval_refinement",
                "task1_total_score_proxy": 95.0,
            },
        })
        memory.record_failure_pattern("sess_001", "task1_retrieval_refinement", {
            "decision": "rejected",
            "reason_codes": ["early_segment_low"],
        })
        memory.record_candidate("sess_001", {
            "candidate_id": "rc_001",
            "candidate_family": "task1_retrieval_refinement",
            "task1_total_score_proxy": 95.0,
            "inference_seconds": 2.0,
        })

        # Query back
        failures = memory.get_similar_failures("task1_retrieval_refinement")
        assert len(failures) == 1

        history = memory.get_family_performance_history("task1_retrieval_refinement")
        assert len(history) == 1
        assert history[0]["score"] == 95.0

        pareto = memory.get_global_pareto_front()
        assert len(pareto) >= 1

        lessons = memory.get_lessons_learned()
        assert lessons["total_experiments"] == 1
        assert "task1_retrieval_refinement" in lessons["family_performance_summary"]

    def test_memory_failure_rules_extraction(self, tmp_path: Path) -> None:
        from pde_agent.core.hypothesis_lab_failure_strategy import (
            extract_rules_from_memory,
        )
        from pde_agent.core.hypothesis_lab_memory import HypothesisLabMemory

        memory_path = tmp_path / "memory.jsonl"
        memory = HypothesisLabMemory(memory_path)

        # Record a failure
        memory.record_failure_pattern("sess_001", "task1_retrieval_refinement", {
            "root_cause": "early_shock_smearing",
            "worst_segment": "early",
            "score_delta": -0.5,
        })
        # Record a success in same session for different family
        memory.record_experiment("sess_001", {
            "best_candidate": {
                "candidate_family": "task1_early_transition_expert",
                "task1_total_score_proxy": 98.0,
            },
        })

        rules = extract_rules_from_memory(memory)
        assert len(rules) >= 1
        rule = rules[0]
        assert rule["action"] == "try_family"
        assert rule["target"] == "task1_early_transition_expert"
        assert rule["condition"]["failed_family"] == "task1_retrieval_refinement"


# ---------------------------------------------------------------------------
# Scenario 7: Tool registry catalog
# ---------------------------------------------------------------------------


class TestToolRegistryIntegration:
    def test_catalog_has_all_15_tools(self) -> None:
        from pde_agent.core.hypothesis_lab_tool_registry import (
            build_hypothesis_lab_tool_catalog,
        )

        catalog = build_hypothesis_lab_tool_catalog()
        assert len(catalog) == 15

        names = {spec.name for spec in catalog}
        assert "agent-hypothesis-run-sandbox" in names
        assert "agent-hypothesis-run-pipeline" in names
        assert "agent-hypothesis-generate-hypotheses" in names

        # Verify risk levels
        for spec in catalog:
            assert spec.risk_level in {"read_only", "artifact_write", "sandbox", "long_running"}

    def test_payload_has_schema_metadata(self) -> None:
        from pde_agent.core.hypothesis_lab_tool_registry import (
            build_hypothesis_lab_tool_catalog_payload,
        )

        payload = build_hypothesis_lab_tool_catalog_payload()
        assert payload["schema_version"] == 1
        assert payload["tool_count"] == 15
        assert len(payload["tools"]) == 15
        assert "risk_levels" in payload
