from __future__ import annotations

import json
from pathlib import Path


def test_trace_store_appends_events_and_summarizes_by_family(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_trace import (
        HypothesisLabTraceStore,
        summarize_hypothesis_lab_trace,
    )

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)

    first = store.append_event(
        event_type="sandbox_result",
        payload={
            "algorithm_candidate_id": "candidate_a",
            "candidate_family": "task1_retrieval_refinement",
            "outcome": "promoted",
            "score_delta": 1.25,
        },
        run_id="run_001",
        tags=["task1", "sandbox"],
    )
    second = store.append_event(
        event_type="sandbox_result",
        payload={
            "algorithm_candidate_id": "candidate_b",
            "candidate_family": "task1_sem_burgers_rollout",
            "outcome": "failed",
            "score_delta": -0.4,
        },
        run_id="run_001",
    )

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_id"] == first["event_id"]
    assert json.loads(lines[1])["event_id"] == second["event_id"]

    summary = summarize_hypothesis_lab_trace(
        trace_path=trace_path,
        output_path=tmp_path / "summary.json",
        force=False,
    )

    assert summary["state_hint"] == "HYPOTHESIS_LAB_TRACE_SUMMARIZED"
    assert summary["event_count"] == 2
    assert summary["event_counts_by_type"] == {"sandbox_result": 2}
    assert summary["candidate_family_counts"] == {
        "task1_retrieval_refinement": 1,
        "task1_sem_burgers_rollout": 1,
    }
    assert summary["outcome_counts"] == {"promoted": 1, "failed": 1}
    assert summary["promoted_candidate_ids"] == ["candidate_a"]
    assert summary["failed_candidate_families"] == ["task1_sem_burgers_rollout"]


def test_doom_loop_detector_flags_repeated_non_improving_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_doom_loop import check_hypothesis_doom_loop
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for index in range(3):
        store.append_event(
            event_type="sandbox_result",
            payload={
                "algorithm_candidate_id": f"candidate_{index}",
                "candidate_family": "task1_early_refinement",
                "outcome": "failed",
                "score_delta": 0.02,
            },
        )

    report = check_hypothesis_doom_loop(
        trace_path=trace_path,
        output_path=tmp_path / "doom.json",
        repeated_family_threshold=3,
        min_score_delta=0.1,
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_DOOM_LOOP_CHECKED"
    assert report["decision"] == "inject_corrective_prompt"
    assert report["guard_active"] is True
    assert report["blocked_candidate_families"] == ["task1_early_refinement"]
    assert "task1_early_refinement" in report["corrective_prompt"]
    assert "different candidate family" in report["corrective_prompt"]


def test_scientific_tool_catalog_exposes_risk_and_state_hints(tmp_path: Path) -> None:
    from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog

    catalog = build_scientific_tool_catalog(
        output_path=tmp_path / "tool_catalog.json",
        force=False,
    )

    assert catalog["state_hint"] == "SCIENTIFIC_TOOL_CATALOG_BUILT"
    tools = {tool["name"]: tool for tool in catalog["tools"]}
    assert tools["run-hypothesis-lab"]["risk_level"] == "sandbox"
    assert "HYPOTHESIS_LAB_RAN" in tools["run-hypothesis-lab"]["state_hints"]
    assert tools["detect-hypothesis-doom-loop"]["risk_level"] == "read_only"
    assert "active_audit_repair_errors" in tools["build-hypothesis-kpi-report"][
        "produced_artifacts"
    ]
    assert "repair_nonconvergence_errors" in tools["build-hypothesis-kpi-report"][
        "produced_artifacts"
    ]
    assert "repair_convergence_errors" in tools["build-hypothesis-kpi-report"][
        "produced_artifacts"
    ]
    assert "audit_repair_converged_error_counts" in tools[
        "build-hypothesis-kpi-report"
    ]["produced_artifacts"]
    assert "persistent_search_guidance" in tools["build-hypothesis-kpi-report"][
        "produced_artifacts"
    ]
    assert "cluster_saturation_signals" in tools["build-hypothesis-kpi-report"][
        "produced_artifacts"
    ]
    assert "saturated_candidate_families" in tools["build-hypothesis-kpi-report"][
        "produced_artifacts"
    ]
    assert "active_audit_repair_errors" in tools["build-hypothesis-kpi-report"][
        "agent_guidance"
    ]
    assert "repair_convergence_errors" in tools["build-hypothesis-kpi-report"][
        "agent_guidance"
    ]
    assert "persistent_search_guidance" in tools["build-hypothesis-kpi-report"][
        "agent_guidance"
    ]
    assert tools["plan-hypothesis-candidates"]["risk_level"] == "read_only"
    assert "candidate_plan_json" in tools["plan-hypothesis-candidates"][
        "produced_artifacts"
    ]
    assert "candidate_generator_schedule" in tools["plan-hypothesis-candidates"][
        "produced_artifacts"
    ]
    assert "audit_repair_policies" in tools["plan-hypothesis-candidates"][
        "produced_artifacts"
    ]
    assert "repair_nonconvergence_errors" in tools["plan-hypothesis-candidates"][
        "produced_artifacts"
    ]
    assert "repair_convergence_errors" in tools["plan-hypothesis-candidates"][
        "produced_artifacts"
    ]
    assert "audit_repair_converged_error_counts" in tools[
        "plan-hypothesis-candidates"
    ]["produced_artifacts"]
    assert "repair_policy_recommended_actions" in tools[
        "plan-hypothesis-candidates"
    ]["produced_artifacts"]
    assert "research_chronicle_repair_policy_memory" in tools[
        "plan-hypothesis-candidates"
    ]["produced_artifacts"]
    assert "audit_repair_policies" in tools["plan-hypothesis-candidates"][
        "agent_guidance"
    ]
    assert "repair_convergence_errors" in tools["plan-hypothesis-candidates"][
        "agent_guidance"
    ]
    assert "research_chronicle_path" in tools["plan-hypothesis-candidates"][
        "agent_guidance"
    ]
    assert tools["build-hypothesis-algorithm-design"]["risk_level"] == "artifact_write"
    assert "evidence_paths" in tools["build-hypothesis-algorithm-design"][
        "required_inputs"
    ]
    assert "--evidence-path" in tools["build-hypothesis-algorithm-design"][
        "agent_guidance"
    ]
    assert "HYPOTHESIS_ALGORITHM_DESIGN_BUILT" in tools[
        "build-hypothesis-algorithm-design"
    ]["state_hints"]
    assert "algorithm_blueprints" in tools["build-hypothesis-algorithm-design"][
        "produced_artifacts"
    ]
    assert tools["validate-hypothesis-algorithm-design"]["risk_level"] == "read_only"
    assert "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED" in tools[
        "validate-hypothesis-algorithm-design"
    ]["state_hints"]
    assert tools["record-hypothesis-algorithm-evidence"]["risk_level"] == (
        "artifact_write"
    )
    assert tools["build-fno-training-harness-contract"]["risk_level"] == (
        "artifact_write"
    )
    assert "FNO_TRAINING_HARNESS_CONTRACT_BUILT" in tools[
        "build-fno-training-harness-contract"
    ]["state_hints"]
    assert "available_actions" in tools["build-fno-training-harness-contract"][
        "produced_artifacts"
    ]
    assert "not a fixed pipeline" in tools["build-fno-training-harness-contract"][
        "agent_guidance"
    ]
    assert tools["run-fno-training-optimization-loop"]["risk_level"] == "sandbox"
    assert "FNO_TRAINING_OPTIMIZATION_LOOP_RAN" in tools[
        "run-fno-training-optimization-loop"
    ]["state_hints"]
    assert "candidate_plan_path" in tools["run-fno-training-optimization-loop"][
        "required_inputs"
    ]
    assert "budget_contract" in tools["run-fno-training-optimization-loop"][
        "agent_guidance"
    ]
    assert tools["record-fno-training-optimization-evidence"]["risk_level"] == (
        "artifact_write"
    )
    assert "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED" in tools[
        "record-fno-training-optimization-evidence"
    ]["state_hints"]
    assert tools["audit-hermes-availability"]["risk_level"] == "read_only"
    assert "AGENT_RUNTIME_AVAILABILITY_AUDITED" in tools[
        "audit-hermes-availability"
    ]["state_hints"]
    assert tools["audit-agent-runtime-availability"]["risk_level"] == "read_only"
    assert "AGENT_RUNTIME_AVAILABILITY_AUDITED" in tools[
        "audit-agent-runtime-availability"
    ]["state_hints"]
    assert "claude_agent_sdk" in tools["audit-agent-runtime-availability"][
        "agent_guidance"
    ]
    assert tools["build-agent-runtime-contract"]["risk_level"] == "artifact_write"
    assert "AGENT_RUNTIME_CONTRACT_BUILT" in tools[
        "build-agent-runtime-contract"
    ]["state_hints"]
    assert "claude_agent_sdk" in tools["build-agent-runtime-contract"][
        "agent_guidance"
    ]
    assert tools["run-claude-agent-sdk-runtime"]["risk_level"] == "sandbox"
    assert "CLAUDE_AGENT_SDK_RUNTIME_RAN" in tools[
        "run-claude-agent-sdk-runtime"
    ]["state_hints"]
    assert "uv run --frozen pde-agent" in tools["run-claude-agent-sdk-runtime"][
        "agent_guidance"
    ]
    assert "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED" in tools[
        "record-hypothesis-algorithm-evidence"
    ]["state_hints"]
    assert tools["plan-hypothesis-algorithm-sandbox"]["risk_level"] == (
        "artifact_write"
    )
    assert "HYPOTHESIS_ALGORITHM_SANDBOX_PLANNED" in tools[
        "plan-hypothesis-algorithm-sandbox"
    ]["state_hints"]
    assert "sandbox_items" in tools["plan-hypothesis-algorithm-sandbox"][
        "produced_artifacts"
    ]
    assert tools["adapt-hypothesis-algorithm-sandbox-item"]["risk_level"] == (
        "artifact_write"
    )
    assert "HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED" in tools[
        "adapt-hypothesis-algorithm-sandbox-item"
    ]["state_hints"]
    assert tools["run-hypothesis-algorithm-sandbox"]["risk_level"] == "sandbox"
    assert "HYPOTHESIS_ALGORITHM_SANDBOX_RAN" in tools[
        "run-hypothesis-algorithm-sandbox"
    ]["state_hints"]
    assert tools["run-hypothesis-algorithm-designer-stability-batch"][
        "risk_level"
    ] == "sandbox"
    assert "HYPOTHESIS_ALGORITHM_DESIGNER_STABILITY_BATCH_RAN" in tools[
        "run-hypothesis-algorithm-designer-stability-batch"
    ]["state_hints"]
    assert tools["build-repair-policy-summary"]["risk_level"] == "read_only"
    assert "HERMES_REPAIR_POLICY_SUMMARY_BUILT" in tools[
        "build-repair-policy-summary"
    ]["state_hints"]
    assert "repair_policy_summary_json" in tools[
        "build-repair-policy-summary"
    ]["produced_artifacts"]
    assert "nonconverged_repair_errors" in tools[
        "build-repair-policy-summary"
    ]["produced_artifacts"]
    assert "recommended_actions" in tools["build-repair-policy-summary"][
        "agent_guidance"
    ]
    assert tools["run-hypothesis-learning-constraint-smoke"]["risk_level"] == (
        "artifact_write"
    )
    assert "learning_constraint_smoke_json" in tools[
        "run-hypothesis-learning-constraint-smoke"
    ]["produced_artifacts"]
    assert tools["run-hermes-runtime-memory-bridge-smoke"]["risk_level"] == (
        "sandbox"
    )
    assert "runtime_memory_bridge_smoke_json" in tools[
        "run-hermes-runtime-memory-bridge-smoke"
    ]["produced_artifacts"]
    assert "memory_constraints" in tools[
        "run-hermes-runtime-memory-bridge-smoke"
    ]["agent_guidance"]
    assert tools["run-hermes-auto-repair-memory-acceptance-smoke"]["risk_level"] == (
        "sandbox"
    )
    assert "auto_repair_memory_acceptance_smoke_json" in tools[
        "run-hermes-auto-repair-memory-acceptance-smoke"
    ]["produced_artifacts"]
    assert "planner_internalized" in tools[
        "run-hermes-auto-repair-memory-acceptance-smoke"
    ]["produced_artifacts"]
    assert tools["audit-hermes-hypothesis-lab-infrastructure"]["risk_level"] == (
        "read_only"
    )
    assert "infrastructure_audit_json" in tools[
        "audit-hermes-hypothesis-lab-infrastructure"
    ]["produced_artifacts"]
    assert "stability_chain_report_path" in tools[
        "audit-hermes-hypothesis-lab-infrastructure"
    ]["produced_artifacts"]
    assert "candidate_pressure_chain_report_path" in tools[
        "audit-hermes-hypothesis-lab-infrastructure"
    ]["produced_artifacts"]
    assert "preflight_acceptance_smoke_path" in tools[
        "audit-hermes-hypothesis-lab-infrastructure"
    ]["agent_guidance"]
    assert "candidate_pressure_chain_report_path" in tools[
        "audit-hermes-hypothesis-lab-infrastructure"
    ]["agent_guidance"]
    assert tools["run-candidate-expansion-handoff"]["risk_level"] == "artifact_write"
    assert "expansion_report_json" in tools["run-candidate-expansion-handoff"][
        "produced_artifacts"
    ]
    assert tools["prepare-expansion-blocked-executor-authoring-tasks"][
        "risk_level"
    ] == "artifact_write"
    assert "bounded_executor_authoring_task_json" in tools[
        "prepare-expansion-blocked-executor-authoring-tasks"
    ]["produced_artifacts"]
    assert tools["audit-bounded-executor-authoring-task"]["risk_level"] == "read_only"
    assert "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED" in tools[
        "audit-bounded-executor-authoring-task"
    ]["state_hints"]
    assert tools["run-bounded-executor-sandbox"]["required_inputs"] == [
        "design_spec_path",
        "manifest_path",
        "executor_path",
        "context_path",
        "target_path",
        "output_dir",
    ]
    assert "bounded_executor_sandbox_report_json" in tools[
        "run-bounded-executor-sandbox"
    ]["produced_artifacts"]
    assert tools["record-bounded-executor-sandbox-evidence"][
        "risk_level"
    ] == "artifact_write"
    assert "SANDBOX_EVIDENCE_RECORDED" in tools[
        "record-bounded-executor-sandbox-evidence"
    ]["state_hints"]
    assert tools["complete-bounded-executor-authoring-task"][
        "risk_level"
    ] == "sandbox"
    assert "BOUNDED_EXECUTOR_AUTHORING_COMPLETED" in tools[
        "complete-bounded-executor-authoring-task"
    ]["state_hints"]
    assert "hypothesis_lab_trace_event" in tools[
        "complete-bounded-executor-authoring-task"
    ]["produced_artifacts"]
    assert "trace_path" in tools["complete-bounded-executor-authoring-task"][
        "agent_guidance"
    ]
    assert tools["run-hermes-heartbeat"]["risk_level"] == "long_running"
    assert "audit_command" in tools["run-hermes-runtime-closed-loop"][
        "produced_artifacts"
    ]
    assert "repair_nonconvergence_errors" in tools[
        "run-hermes-runtime-closed-loop"
    ]["produced_artifacts"]
    assert "audit_repair_policies" in tools["run-hermes-runtime-closed-loop"][
        "produced_artifacts"
    ]
    assert "task1_reference_path" in tools["run-hermes-runtime-closed-loop"][
        "agent_guidance"
    ]
    assert "evidence_accumulation_plan_path" in tools[
        "run-hermes-runtime-closed-loop"
    ]["agent_guidance"]
    assert "audit_repair_tasks_path" in tools[
        "run-hermes-runtime-closed-loop"
    ]["agent_guidance"]
    assert "audit_command" in tools["run-hermes-runtime-closed-loop"][
        "agent_guidance"
    ]
    assert "audit_repair_policies" in tools["run-hermes-runtime-closed-loop"][
        "agent_guidance"
    ]
    assert "learning_constraint_enforcement" in tools[
        "audit-hermes-runtime-closed-loop"
    ]["agent_guidance"]
    assert "runtime_memory_bridge_smoke" in tools[
        "audit-hermes-runtime-closed-loop"
    ]["agent_guidance"]
    assert "audit_errors" in tools["audit-hermes-runtime-closed-loop"][
        "produced_artifacts"
    ]
    assert "failed_checks" in tools["audit-hermes-runtime-closed-loop"][
        "produced_artifacts"
    ]
    assert "failed_checks" in tools["audit-hermes-runtime-closed-loop"][
        "agent_guidance"
    ]
    assert tools["run-hermes-runtime-audit-chain"]["risk_level"] == "long_running"
    assert "runtime_audit_chain_report_json" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "task1_reference_path" in tools["run-hermes-runtime-audit-chain"][
        "agent_guidance"
    ]
    assert "audit_repair_tasks_path" in tools["run-hermes-runtime-audit-chain"][
        "agent_guidance"
    ]
    assert "hypothesis_lab_trace_event" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "audit_repair_tasks_json" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "repair_convergence" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "auto_repair_report_path" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "auto_repair_memory_bridge_passed" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "auto_repair_repair_policy_recommended_action_count" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "preflight_acceptance_smoke_path" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "preflight_infrastructure_audit_path" in tools[
        "run-hermes-runtime-audit-chain"
    ]["produced_artifacts"]
    assert "preflight_acceptance_smoke_path" in tools[
        "run-hermes-runtime-audit-chain"
    ]["agent_guidance"]
    assert "preflight_infrastructure_audit_path" in tools[
        "run-hermes-runtime-audit-chain"
    ]["agent_guidance"]
    assert "audit_repair_tasks" in tools["run-hermes-runtime-audit-chain"][
        "agent_guidance"
    ]
    assert "repair_convergence" in tools["run-hermes-runtime-audit-chain"][
        "agent_guidance"
    ]
    assert "auto_repair" in tools["run-hermes-runtime-audit-chain"]["agent_guidance"]
    assert "suggested_command_args" in tools["run-hermes-runtime-audit-chain"][
        "agent_guidance"
    ]
    assert "missing_inputs" in tools["run-hermes-runtime-audit-chain"][
        "agent_guidance"
    ]
    assert tools["resolve-audit-repair-task"]["risk_level"] == "read_only"
    assert "repair_task_resolution_json" in tools["resolve-audit-repair-task"][
        "produced_artifacts"
    ]
    assert "command_string" in tools["resolve-audit-repair-task"][
        "produced_artifacts"
    ]
    assert "suggested_command_args" in tools["resolve-audit-repair-task"][
        "agent_guidance"
    ]
    assert "provided-input" in tools["resolve-audit-repair-task"]["agent_guidance"]
    assert tools["run-resolved-audit-repair-task"]["risk_level"] == "sandbox"
    assert "HERMES_AUDIT_REPAIR_TASK_RAN" in tools[
        "run-resolved-audit-repair-task"
    ]["state_hints"]
    assert "runtime_loop_report_json" in tools[
        "run-resolved-audit-repair-task"
    ]["produced_artifacts"]
    assert "runtime_memory_bridge_smoke_json" in tools[
        "run-resolved-audit-repair-task"
    ]["produced_artifacts"]
    assert "smoke_executed" in tools[
        "run-resolved-audit-repair-task"
    ]["produced_artifacts"]
    assert "memory_bridge_passed" in tools[
        "run-resolved-audit-repair-task"
    ]["produced_artifacts"]
    assert "repair_policy_recommended_action_count" in tools[
        "run-resolved-audit-repair-task"
    ]["produced_artifacts"]
    assert "Unsupported next_command" in tools[
        "run-resolved-audit-repair-task"
    ]["agent_guidance"]
    assert "run-hermes-runtime-memory-bridge-smoke" in tools[
        "run-resolved-audit-repair-task"
    ]["agent_guidance"]
    assert tools["refine-algorithm-candidate"]["risk_level"] == "artifact_write"
    assert "HERMES_ALGORITHM_CANDIDATE_REFINEMENT_PREPARED" in tools[
        "refine-algorithm-candidate"
    ]["state_hints"]
    assert "legal reference" in tools["refine-algorithm-candidate"]["agent_guidance"]
    assert catalog["next_command"] == "summarize-hypothesis-lab-trace"


def test_runtime_command_wrappers_return_agent_payloads(tmp_path: Path) -> None:
    from pde_agent.commands.hypothesis_lab_runtime import (
        build_scientific_tool_catalog_command,
        detect_hypothesis_doom_loop_command,
        record_hypothesis_lab_trace_event_command,
        summarize_hypothesis_lab_trace_command,
    )

    trace_path = tmp_path / "trace.jsonl"
    record = record_hypothesis_lab_trace_event_command(
        trace_path=str(trace_path),
        event_type="sandbox_result",
        payload_json=json.dumps(
            {
                "algorithm_candidate_id": "candidate_a",
                "candidate_family": "task1_early_refinement",
                "outcome": "failed",
                "score_delta": 0.0,
            }
        ),
        run_id="run_001",
        source_artifact_path=None,
        tags=["task1"],
    )
    assert record["status"] == "ok"
    assert record["state_hint"] == "HYPOTHESIS_LAB_TRACE_EVENT_RECORDED"
    assert Path(record["artifacts"]["trace_path"]).exists()

    summary = summarize_hypothesis_lab_trace_command(
        trace_path=str(trace_path),
        output_path=str(tmp_path / "summary.json"),
        force=False,
    )
    assert summary["status"] == "ok"
    assert summary["state_hint"] == "HYPOTHESIS_LAB_TRACE_SUMMARIZED"
    assert summary["artifacts"]["trace_summary_json"].endswith("summary.json")
    assert summary["metrics"]["event_count"] == 1

    doom = detect_hypothesis_doom_loop_command(
        trace_path=str(trace_path),
        output_path=str(tmp_path / "doom.json"),
        recent_window=20,
        repeated_family_threshold=3,
        failed_family_threshold=1,
        min_score_delta=0.1,
        force=False,
    )
    assert doom["status"] == "ok"
    assert doom["state_hint"] == "HYPOTHESIS_DOOM_LOOP_CHECKED"
    assert doom["artifacts"]["doom_loop_report_json"].endswith("doom.json")
    assert doom["metrics"]["guard_active"] is True

    catalog = build_scientific_tool_catalog_command(
        output_path=str(tmp_path / "catalog.json"),
        force=False,
    )
    assert catalog["status"] == "ok"
    assert catalog["state_hint"] == "SCIENTIFIC_TOOL_CATALOG_BUILT"
    assert catalog["metrics"]["tool_count"] >= 3


def test_build_hypothesis_kpi_report_command_returns_repair_signal_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    calls = []

    def fake_kpi(**kwargs):
        calls.append(kwargs)
        return {
            "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
            "metrics": {
                "candidate_count": 1,
                "active_audit_repair_error_count": 1,
                "repair_nonconvergence_error_count": 1,
                "repair_convergence_error_count": 1,
            },
            "active_audit_repair_errors": [
                "cluster_saturation_consistency",
            ],
            "repair_nonconvergence_errors": [
                "cluster_saturation_consistency",
            ],
            "repair_convergence_errors": [
                "learning_constraint_smoke_gate",
            ],
            "audit_repair_error_counts": {
                "cluster_saturation_consistency": 2,
            },
            "audit_repair_consumed_error_counts": {
                "cluster_saturation_consistency": 1,
            },
            "audit_repair_converged_error_counts": {
                "learning_constraint_smoke_gate": 1,
            },
            "persistent_search_guidance": [
                {
                    "guidance_type": "escalate_after_guided_search_exhaustion",
                    "recommended_action": (
                        "switch_gate_feature_family_or_try_new_dynamics_source"
                    ),
                }
            ],
            "search_guidance_action_counts": {
                "switch_gate_feature_family_or_try_new_dynamics_source": 1
            },
            "cluster_saturation_signals": [
                {
                    "candidate_family": "local_strategy_skill",
                    "signal": "local_strategy_cluster_saturated",
                }
            ],
            "saturated_candidate_families": ["local_strategy_skill"],
            "next_command": "detect-hypothesis-doom-loop",
        }

    monkeypatch.setattr(
        runtime_commands,
        "build_hypothesis_kpi_report",
        fake_kpi,
    )

    payload = runtime_commands.build_hypothesis_kpi_report_command(
        trace_path=str(tmp_path / "trace.jsonl"),
        output_path=str(tmp_path / "kpi.json"),
        recent_window=20,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["artifacts"]["hypothesis_kpi_report_json"].endswith("kpi.json")
    assert payload["artifacts"]["active_audit_repair_errors"] == [
        "cluster_saturation_consistency"
    ]
    assert payload["artifacts"]["repair_nonconvergence_errors"] == [
        "cluster_saturation_consistency"
    ]
    assert payload["artifacts"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert payload["artifacts"]["audit_repair_error_counts"] == {
        "cluster_saturation_consistency": 2
    }
    assert payload["artifacts"]["audit_repair_consumed_error_counts"] == {
        "cluster_saturation_consistency": 1
    }
    assert payload["artifacts"]["audit_repair_converged_error_counts"] == {
        "learning_constraint_smoke_gate": 1
    }
    assert payload["artifacts"]["persistent_search_guidance"] == [
        {
            "guidance_type": "escalate_after_guided_search_exhaustion",
            "recommended_action": (
                "switch_gate_feature_family_or_try_new_dynamics_source"
            ),
        }
    ]
    assert payload["artifacts"]["search_guidance_action_counts"] == {
        "switch_gate_feature_family_or_try_new_dynamics_source": 1
    }
    assert payload["artifacts"]["cluster_saturation_signals"] == [
        {
            "candidate_family": "local_strategy_skill",
            "signal": "local_strategy_cluster_saturated",
        }
    ]
    assert payload["artifacts"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert calls[0]["recent_window"] == 20


def test_build_repair_policy_summary_command_returns_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    calls = []

    def fake_summary(**kwargs):
        calls.append(kwargs)
        return {
            "state_hint": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
            "repair_policy_summary_path": str(kwargs["output_path"]),
            "converged_repair_errors": ["learning_constraint_smoke_gate"],
            "nonconverged_repair_errors": ["repair_convergence_consistency"],
            "repair_error_stats": {
                "learning_constraint_smoke_gate": {
                    "attempt_count": 1,
                    "converged_count": 1,
                    "nonconverged_count": 0,
                    "last_decision": "converged",
                },
                "repair_convergence_consistency": {
                    "attempt_count": 1,
                    "converged_count": 0,
                    "nonconverged_count": 1,
                    "last_decision": "nonconverged",
                },
            },
            "recommended_actions": [
                {
                    "audit_error": "repair_convergence_consistency",
                    "action": "switch_repair_strategy_or_reload_missing_evidence",
                    "reason": "repair_nonconverged",
                }
            ],
            "metrics": {
                "repair_attempt_count": 2,
                "converged_repair_error_count": 1,
                "nonconverged_repair_error_count": 1,
                "event_count": 2,
            },
            "next_command": "plan-hypothesis-candidates",
        }

    monkeypatch.setattr(
        runtime_commands,
        "build_repair_policy_summary",
        fake_summary,
    )

    payload = runtime_commands.build_repair_policy_summary_command(
        trace_path=str(tmp_path / "trace.jsonl"),
        output_path=str(tmp_path / "repair_policy.json"),
        recent_window=20,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_REPAIR_POLICY_SUMMARY_BUILT"
    assert payload["artifacts"]["repair_policy_summary_json"].endswith(
        "repair_policy.json"
    )
    assert payload["artifacts"]["converged_repair_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert payload["artifacts"]["nonconverged_repair_errors"] == [
        "repair_convergence_consistency"
    ]
    assert payload["artifacts"]["recommended_actions"] == [
        {
            "audit_error": "repair_convergence_consistency",
            "action": "switch_repair_strategy_or_reload_missing_evidence",
            "reason": "repair_nonconverged",
        }
    ]
    assert payload["metrics"]["repair_attempt_count"] == 2
    assert calls[0]["recent_window"] == 20


def test_plan_hypothesis_candidates_command_returns_repair_policy_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    calls = []

    def fake_plan(**kwargs):
        calls.append(kwargs)
        return {
            "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
            "candidate_plan_path": str(kwargs["output_path"]),
            "candidate_generator_schedule": ["physics_residual_nudge"],
            "eligible_candidate_count": 3,
            "expansion_required": False,
            "decision": "schedule_candidates",
            "learning_constraints": {
                "repair_nonconvergence_errors": [
                    "cluster_saturation_consistency"
                ],
                "repair_convergence_errors": [
                    "learning_constraint_smoke_gate",
                ],
                "audit_repair_converged_error_counts": {
                    "learning_constraint_smoke_gate": 1,
                },
                "audit_repair_policies": [
                    {
                        "audit_error": "cluster_saturation_consistency",
                        "preferred_candidate_generators": [
                            "physics_residual_nudge"
                        ],
                        "repair_nonconverged": True,
                    }
                ],
                "research_chronicle_repair_policy_memory": {
                    "path": str(tmp_path / "prior_chronicle.md"),
                    "repair_action_count": 1,
                },
            },
            "audit_repair_tasks": [
                {
                    "audit_error": "cluster_saturation_consistency",
                    "repair_nonconverged": True,
                }
            ],
            "audit_repair_policies": [
                {
                    "audit_error": "cluster_saturation_consistency",
                    "preferred_candidate_generators": ["physics_residual_nudge"],
                    "repair_nonconverged": True,
                }
            ],
            "next_command": "run-hermes-heartbeat",
        }

    monkeypatch.setattr(
        runtime_commands,
        "build_hypothesis_candidate_plan",
        fake_plan,
    )

    payload = runtime_commands.plan_hypothesis_candidates_command(
        output_path=str(tmp_path / "candidate_plan.json"),
        candidate_generator="sem_window_gate_blend",
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=None,
        repair_policy_summary_path=str(tmp_path / "repair_policy.json"),
        research_chronicle_path=str(tmp_path / "prior_chronicle.md"),
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        preferred_error_modes=[],
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["artifacts"]["candidate_plan_json"].endswith(
        "candidate_plan.json"
    )
    assert payload["artifacts"]["candidate_generator_schedule"] == [
        "physics_residual_nudge"
    ]
    assert payload["artifacts"]["repair_nonconvergence_errors"] == [
        "cluster_saturation_consistency"
    ]
    assert payload["artifacts"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert payload["artifacts"]["audit_repair_converged_error_counts"] == {
        "learning_constraint_smoke_gate": 1
    }
    assert payload["artifacts"]["audit_repair_policies"] == [
        {
            "audit_error": "cluster_saturation_consistency",
            "preferred_candidate_generators": ["physics_residual_nudge"],
            "repair_nonconverged": True,
        }
    ]
    assert payload["artifacts"]["audit_repair_tasks"] == [
        {
            "audit_error": "cluster_saturation_consistency",
            "repair_nonconverged": True,
        }
    ]
    assert payload["artifacts"]["research_chronicle_repair_policy_memory"] == {
        "path": str(tmp_path / "prior_chronicle.md"),
        "repair_action_count": 1,
    }
    assert calls[0]["repair_policy_summary_path"] == tmp_path / "repair_policy.json"
    assert calls[0]["research_chronicle_path"] == tmp_path / "prior_chronicle.md"
    assert calls[0]["output_path"] == tmp_path / "candidate_plan.json"


def test_hypothesis_lab_runtime_cli_smoke(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    output_path = tmp_path / "catalog.json"
    result = CliRunner().invoke(
        app,
        [
            "build-scientific-tool-catalog",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "SCIENTIFIC_TOOL_CATALOG_BUILT" in result.stdout
    assert "run-hypothesis-lab" in output_path.read_text(encoding="utf-8")


def test_repair_policy_summary_cli_smoke(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from typer.testing import CliRunner

    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "passed",
            "repair_convergence": {
                "decision": "converged",
                "input_repair_errors": ["learning_constraint_smoke_gate"],
                "repeated_audit_errors": [],
            },
        },
    )
    output_path = tmp_path / "repair_policy.json"
    result = CliRunner().invoke(
        app,
        [
            "build-repair-policy-summary",
            "--trace-path",
            str(trace_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "HERMES_REPAIR_POLICY_SUMMARY_BUILT" in result.stdout
    assert "learning_constraint_smoke_gate" in output_path.read_text(
        encoding="utf-8"
    )


def test_run_hypothesis_lab_command_can_record_trace_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import run_hypothesis_lab as command_module

    def fake_run_hypothesis_lab(**_kwargs):
        return {
            "artifacts": {
                "hypothesis_lab_report_path": str(tmp_path / "report.json"),
                "sandbox_report_path": str(tmp_path / "sandbox.json"),
            },
            "cards": {"count": 2},
            "hypotheses": {"count": 1},
            "blueprints": {"count": 1},
            "sandbox": {
                "candidate_family": "task1_early_refinement",
                "best_candidate": {
                    "candidate_id": "candidate_a",
                    "segment_score": 48.0,
                },
                "verifier": {
                    "best_segment_score": 48.0,
                    "score_delta_vs_baseline": 1.5,
                    "decision": "promote_to_strategy_proposal",
                },
            },
            "next_command": "record-strategy-proposal",
        }

    monkeypatch.setattr(
        command_module,
        "run_hypothesis_lab",
        fake_run_hypothesis_lab,
    )
    trace_path = tmp_path / "trace.jsonl"

    payload = command_module.run_hypothesis_lab_command(
        manual_report_paths=[],
        context_path=str(tmp_path / "context.hdf5"),
        target_path=str(tmp_path / "target.hdf5"),
        output_dir=str(tmp_path / "out"),
        submission_id="team_hermes",
        force=False,
        trace_path=str(trace_path),
    )

    assert payload["status"] == "ok"
    assert payload["artifacts"]["trace_path"] == str(trace_path)
    assert payload["metrics"]["trace_event_recorded"] is True
    event = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "hypothesis_lab_run"
    assert event["payload"]["candidate_family"] == "task1_early_refinement"
    assert event["payload"]["outcome"] == "promoted"
    assert event["payload"]["score_delta"] == 1.5


def test_more_runtime_commands_record_trace_events(tmp_path: Path, monkeypatch) -> None:
    from pde_agent.commands import literature_algorithm_lab as lit_commands
    from pde_agent.commands import run_hermes_heartbeat as heartbeat_commands
    from pde_agent.commands import run_hermes_literature_strategy_expansion as expansion_commands

    monkeypatch.setattr(
        heartbeat_commands,
        "run_hermes_heartbeat",
        lambda **_kwargs: {
            "heartbeat_report_path": str(tmp_path / "heartbeat.json"),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "candidate_h",
                "family": "task1_early_refinement",
            },
            "best_score": 112.0,
            "best_objective": "score",
            "best_objective_score": 112.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue",
            "parameters": {
                "saturated_candidate_families": ["local_strategy_skill"],
                "persistent_search_guidance_count": 1,
            },
            "persistent_search_guidance": [
                {
                    "guidance_type": "escalate_after_guided_search_exhaustion",
                    "reason": "memory_guided_specs_failed_to_improve",
                    "source": "sem_window_gate_blend_multi_search",
                    "recommended_action": (
                        "switch_gate_feature_family_or_try_new_dynamics_source"
                    ),
                }
            ],
            "next_command": "run-hermes-heartbeat",
        },
    )
    monkeypatch.setattr(
        expansion_commands,
        "run_hermes_literature_strategy_expansion",
        lambda **_kwargs: {
            "expansion_report_path": str(tmp_path / "expansion.json"),
            "action_count": 2,
            "executable_count": 1,
            "proposal_count": 1,
            "skipped_count": 0,
            "actions": [
                {
                    "algorithm_candidate_id": "candidate_lit",
                    "candidate_family": "official_checkpoint_family",
                    "executable": True,
                }
            ],
            "next_command": "run-checkpoint-family-sandbox",
        },
    )
    monkeypatch.setattr(
        expansion_commands,
        "record_hermes_literature_candidate_outcome",
        lambda **_kwargs: {
            "failed_algorithm_candidate_ids": ["candidate_failed"],
            "outcomes": [
                {
                    "algorithm_candidate_id": "candidate_failed",
                    "candidate_key": "key_failed",
                    "outcome": "failed",
                    "reason_codes": ["score_regressed"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        lit_commands,
        "run_bounded_executor_sandbox",
        lambda **_kwargs: {
            "algorithm_candidate_id": "candidate_exec",
            "candidate_family": "task1_pde_executor",
            "rules_passed": True,
            "artifacts": {"sandbox_report_path": str(tmp_path / "sandbox.json")},
            "metric_results": {
                "score_delta_vs_persistence": 2.0,
                "inference_seconds": 0.5,
            },
            "next_command": "record-bounded-executor-sandbox-event",
        },
    )
    trace_path = tmp_path / "trace.jsonl"

    heartbeat_payload = heartbeat_commands.run_hermes_heartbeat_command(
        output_dir=str(tmp_path / "hb"),
        initial_submission_dir=str(tmp_path / "initial"),
        max_batches=1,
        batch_deadline_seconds=1.0,
        max_rounds_per_batch=1,
        checkpoint_every=1,
        max_stagnant_batches=1,
        min_batch_score_improvement=0.0,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        learning_objective_schedule=[],
        patience=1,
        task1_test_path=None,
        task1_target_path=None,
        task2_test_path=None,
        task2_target_path=None,
        unet_checkpoint_path=None,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        candidate_generator="postprocess",
        candidate_generator_schedule=[],
        hypothesis_blueprint_path=None,
        hypothesis_reference_path=None,
        hypothesis_reference_sample_limit=None,
        hypothesis_reference_sample_limits=[],
        postprocess_search_mode="adaptive",
        postprocess_strategies=[],
        postprocess_blend_weights=[],
        postprocess_adaptive_step=0.125,
        submission_id="team_hermes",
        timing_profile=None,
        timing_factor=None,
        force=False,
        trace_path=str(trace_path),
    )
    assert heartbeat_payload["metrics"]["trace_event_recorded"] is True

    expansion_payload = expansion_commands.run_hermes_literature_strategy_expansion_command(
        algorithm_candidates_path=str(tmp_path / "candidates.json"),
        workspace_dir=str(tmp_path / "workspace"),
        checkpoint_paths=[],
        failure_memory_path=None,
        registry_summary_path=None,
        force=False,
        trace_path=str(trace_path),
    )
    assert expansion_payload["metrics"]["trace_event_recorded"] is True

    outcome_payload = expansion_commands.record_hermes_literature_candidate_outcome_command(
        algorithm_candidate_id="candidate_failed",
        candidate_key="key_failed",
        outcome="failed",
        reason_codes=["score_regressed"],
        output_path=str(tmp_path / "failure_memory.json"),
        force=False,
        trace_path=str(trace_path),
    )
    assert outcome_payload["metrics"]["trace_event_recorded"] is True

    sandbox_payload = lit_commands.run_bounded_executor_sandbox_command(
        design_spec_path=str(tmp_path / "design.json"),
        manifest_path=str(tmp_path / "manifest.json"),
        executor_path=str(tmp_path / "executor.py"),
        context_path=str(tmp_path / "context.hdf5"),
        target_path=str(tmp_path / "target.hdf5"),
        output_dir=str(tmp_path / "sandbox"),
        submission_id="team_hermes",
        force=False,
        trace_path=str(trace_path),
    )
    assert sandbox_payload["metrics"]["trace_event_recorded"] is True

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "hermes_heartbeat_run",
        "hermes_literature_strategy_expansion",
        "literature_candidate_outcome",
        "bounded_executor_sandbox",
    ]
    assert events[0]["payload"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert events[0]["payload"]["persistent_search_guidance"][0][
        "guidance_type"
    ] == "escalate_after_guided_search_exhaustion"


def test_heartbeat_uses_doom_loop_guard_to_switch_candidate_generator(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    guard_path = tmp_path / "doom.json"
    guard_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": ["task1_early_refinement"],
                "corrective_prompt": "switch",
            }
        ),
        encoding="utf-8",
    )
    batch_calls = []

    def fake_batch_runner(**kwargs):
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "batch.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "loop_report_path": str(report_path),
            "best_candidate": {
                "candidate_id": "best",
                "score": 100.0,
                "submission_dir": str(best_dir),
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs):
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="task1_early_refinement",
        doom_loop_report_path=guard_path,
        candidate_generator_fallback="postprocess",
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "postprocess"
    assert report["batches"][0]["candidate_generator"] == "postprocess"
    assert report["batches"][0]["guarded_candidate_generator"] is True
    assert report["guard"]["blocked_candidate_families"] == ["task1_early_refinement"]


def test_hypothesis_kpi_report_rolls_up_trace(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    store = HypothesisLabTraceStore(tmp_path / "trace.jsonl")
    store.append_event(
        event_type="hypothesis_lab_run",
        payload={
            "algorithm_candidate_id": "candidate_a",
            "candidate_family": "family_a",
            "outcome": "promoted",
            "score_delta": 2.0,
        },
    )
    store.append_event(
        event_type="hypothesis_lab_run",
        payload={
            "algorithm_candidate_id": "candidate_b",
            "candidate_family": "family_b",
            "outcome": "failed",
            "score_delta": -0.5,
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=tmp_path / "trace.jsonl",
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_KPI_REPORT_BUILT"
    assert report["metrics"]["candidate_count"] == 2
    assert report["metrics"]["promotion_rate"] == 0.5
    assert report["best_family_by_score_delta"]["candidate_family"] == "family_a"
    assert report["candidate_family_score_summary"]["family_a"] == {
        "count": 1,
        "failed_count": 0,
        "best_score_delta": 2.0,
        "mean_score_delta": 2.0,
    }
    assert report["candidate_family_score_summary"]["family_b"] == {
        "count": 1,
        "failed_count": 1,
        "best_score_delta": -0.5,
        "mean_score_delta": -0.5,
    }
    assert report["trajectory"]["direction"] == "mixed"


def test_hypothesis_kpi_report_marks_bounded_executor_deprioritization(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    store = HypothesisLabTraceStore(tmp_path / "trace.jsonl")
    store.append_event(
        event_type="bounded_executor_completion",
        payload={
            "algorithm_candidate_id": "candidate_bounded_bad",
            "candidate_family": "bounded_executor",
            "outcome": "deprioritize",
            "score_delta": -35.0,
            "rules_passed": True,
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=tmp_path / "trace.jsonl",
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["metrics"]["candidate_count"] == 1
    assert report["metrics"]["failed_count"] == 1
    assert report["outcome_counts"]["deprioritize"] == 1
    assert report["deprioritized_candidate_families"] == ["bounded_executor"]
    assert report["candidate_family_score_summary"]["bounded_executor"] == {
        "count": 1,
        "failed_count": 1,
        "best_score_delta": -35.0,
        "mean_score_delta": -35.0,
    }


def test_hypothesis_kpi_report_deprioritizes_repeated_failed_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    store = HypothesisLabTraceStore(tmp_path / "trace.jsonl")
    for index in range(2):
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"postprocess_bad_{index}",
                "candidate_family": "postprocess",
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )

    report = build_hypothesis_kpi_report(
        trace_path=tmp_path / "trace.jsonl",
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["deprioritized_candidate_families"] == ["postprocess"]
    assert "repeated_failed_family" in report["deprioritization_reason_codes"][
        "postprocess"
    ]


def test_hypothesis_kpi_report_preserves_deprioritize_reason_codes(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    store = HypothesisLabTraceStore(tmp_path / "trace.jsonl")
    store.append_event(
        event_type="hypothesis_lab_run",
        payload={
            "algorithm_candidate_id": "candidate_local_strategy",
            "candidate_family": "local_strategy_skill",
            "outcome": "deprioritize",
            "score_delta": 0.0,
            "reason_codes": [
                "score_regressed",
                "ablation_missing_or_weak",
            ],
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=tmp_path / "trace.jsonl",
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["deprioritized_candidate_families"] == ["local_strategy_skill"]
    assert report["deprioritization_reason_codes"]["local_strategy_skill"] == [
        "ablation_missing_or_weak",
        "deprioritized_completion_trace",
        "score_regressed",
    ]
    assert report["deprioritized_algorithm_candidates"] == [
        {
            "algorithm_candidate_id": "candidate_local_strategy",
            "candidate_family": "local_strategy_skill",
            "reason_codes": [
                "ablation_missing_or_weak",
                "deprioritized_completion_trace",
                "score_regressed",
            ],
        }
    ]
    assert report["deprioritization_reason_codes_by_candidate"] == {
        "candidate_local_strategy": [
            "ablation_missing_or_weak",
            "deprioritized_completion_trace",
            "score_regressed",
        ]
    }


def test_hypothesis_kpi_report_summarizes_runtime_chain_signals(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    store = HypothesisLabTraceStore(tmp_path / "trace.jsonl")
    store.append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "failed",
            "requested_iteration_count": 3,
            "completed_iteration_count": 1,
            "failed_audit_count": 1,
            "last_heartbeat_best_score": 100.0,
            "score_delta": 0.0,
            "candidate_space_exhausted_count": 1,
            "chain_learning_signals": [
                "audit_gate_failure",
                "candidate_space_exhaustion",
                "score_stagnation",
            ],
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=tmp_path / "trace.jsonl",
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["chain_learning_signal_counts"] == {
        "audit_gate_failure": 1,
        "candidate_space_exhaustion": 1,
        "score_stagnation": 1,
    }
    assert report["metrics"]["chain_event_count"] == 1
    assert report["metrics"]["chain_failed_count"] == 1
    assert report["metrics"]["chain_audit_failure_count"] == 1


def test_hypothesis_kpi_report_summarizes_cluster_saturation_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    store.append_event(
        event_type="hermes_runtime_closed_loop",
        payload={
            "algorithm_candidate_id": "best_runtime",
            "candidate_family": "sem_window_gate_blend_multi_early_smoke",
            "outcome": "continue_next_heartbeat",
            "score_delta": None,
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
            "saturated_candidate_families": ["local_strategy_skill"],
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=trace_path,
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["metrics"]["cluster_saturation_signal_count"] == 1
    assert report["saturated_candidate_families"] == ["local_strategy_skill"]
    assert report["cluster_saturation_signals"][0]["signal"] == (
        "local_strategy_cluster_saturated"
    )


def test_hypothesis_kpi_report_summarizes_persistent_search_guidance(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    guidance = {
        "guidance_type": "escalate_after_guided_search_exhaustion",
        "reason": "memory_guided_specs_failed_to_improve",
        "source": "sem_window_gate_blend_multi_search",
        "recommended_action": "switch_gate_feature_family_or_try_new_dynamics_source",
    }
    store.append_event(
        event_type="hermes_runtime_closed_loop",
        payload={
            "algorithm_candidate_id": "best_runtime",
            "candidate_family": "sem_window_gate_blend_multi_early_smoke",
            "outcome": "continue_next_heartbeat",
            "score_delta": None,
            "persistent_search_guidance": [guidance],
        },
    )
    store.append_event(
        event_type="hermes_heartbeat_run",
        payload={
            "algorithm_candidate_id": "heartbeat",
            "candidate_family": "hermes_heartbeat",
            "outcome": "continue_next_heartbeat",
            "score_delta": None,
            "persistent_search_guidance": [guidance],
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=trace_path,
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["metrics"]["persistent_search_guidance_count"] == 1
    assert report["persistent_search_guidance"] == [guidance]
    assert report["search_guidance_action_counts"] == {
        "switch_gate_feature_family_or_try_new_dynamics_source": 1
    }


def test_hypothesis_kpi_report_summarizes_audit_repair_lifecycle(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    trace_path = tmp_path / "trace.jsonl"
    repair_task = {
        "task_id": "repair_cluster_saturation_consistency",
        "audit_error": "cluster_saturation_consistency",
        "priority": "high",
        "repair_scope": "candidate_cluster_saturation_memory",
        "recommended_action": "reload_cluster_saturation_memory_before_runtime",
        "next_command": "run-hermes-runtime-closed-loop",
    }
    store = HypothesisLabTraceStore(trace_path)
    store.append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "failed",
            "failed_audit_count": 1,
            "audit_errors": ["cluster_saturation_consistency"],
            "audit_repair_task_count": 1,
            "audit_repair_tasks_path": str(tmp_path / "audit_repair_tasks.json"),
            "score_delta": 0.0,
        },
    )
    store.append_event(
        event_type="hermes_runtime_closed_loop",
        payload={
            "algorithm_candidate_id": "best_runtime",
            "candidate_family": "fast_baseline_family",
            "outcome": "continue_next_heartbeat",
            "audit_repair_tasks": [repair_task],
            "score_delta": None,
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=trace_path,
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["audit_repair_error_counts"] == {
        "cluster_saturation_consistency": 1
    }
    assert report["audit_repair_consumed_error_counts"] == {
        "cluster_saturation_consistency": 1
    }
    assert report["active_audit_repair_errors"] == []
    assert report["metrics"]["audit_repair_task_count"] == 1
    assert report["metrics"]["audit_repair_consumed_count"] == 1
    assert report["metrics"]["active_audit_repair_error_count"] == 0


def test_hypothesis_kpi_report_summarizes_repair_nonconvergence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    store.append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "failed",
            "failed_audit_count": 1,
            "audit_errors": ["cluster_saturation_consistency"],
            "repair_convergence": {
                "decision": "nonconverged",
                "input_repair_errors": ["cluster_saturation_consistency"],
                "repeated_audit_errors": ["cluster_saturation_consistency"],
            },
            "chain_learning_signals": ["audit_repair_nonconvergence"],
            "score_delta": 0.0,
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=trace_path,
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["repair_nonconvergence_errors"] == [
        "cluster_saturation_consistency"
    ]
    assert report["metrics"]["repair_nonconvergence_error_count"] == 1
    assert report["chain_learning_signal_counts"]["audit_repair_nonconvergence"] == 1


def test_hypothesis_kpi_report_summarizes_repair_convergence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "passed",
            "failed_audit_count": 0,
            "audit_errors": [],
            "repair_convergence": {
                "decision": "converged",
                "input_repair_errors": ["learning_constraint_smoke_gate"],
                "repeated_audit_errors": [],
            },
            "chain_learning_signals": ["audit_repair_convergence"],
            "score_delta": 0.0,
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=trace_path,
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert report["audit_repair_converged_error_counts"] == {
        "learning_constraint_smoke_gate": 1
    }
    assert report["metrics"]["repair_convergence_error_count"] == 1
    assert report["chain_learning_signal_counts"]["audit_repair_convergence"] == 1


def test_hypothesis_kpi_report_keeps_repair_convergence_without_candidate_id(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report

    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "repair_convergence": {
                "decision": "converged",
                "input_repair_errors": ["repair_convergence_consistency"],
                "repeated_audit_errors": [],
            },
            "chain_learning_signals": ["audit_repair_convergence"],
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=trace_path,
        output_path=tmp_path / "kpi.json",
        recent_window=10,
        force=False,
    )

    assert report["metrics"]["candidate_count"] == 0
    assert report["metrics"]["chain_event_count"] == 1
    assert report["repair_convergence_errors"] == [
        "repair_convergence_consistency"
    ]
    assert report["audit_repair_converged_error_counts"] == {
        "repair_convergence_consistency": 1
    }


def test_hypothesis_kpi_report_summarizes_auto_repair_memory_bridge(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "repaired",
            "repair_convergence": {
                "decision": "converged",
                "input_repair_errors": ["runtime_memory_bridge_smoke_gate"],
                "repeated_audit_errors": [],
            },
            "chain_learning_signals": ["audit_repair_auto_repaired"],
            "auto_repair_decision": "passed",
            "auto_repair_metrics": {
                "smoke_executed": True,
                "memory_bridge_passed": True,
                "repair_policy_recommended_action_count": 1,
                "batch_memory_constraint_action_count": 1,
            },
        },
    )

    report = build_hypothesis_kpi_report(
        trace_path=trace_path,
        output_path=tmp_path / "kpi.json",
        recent_window=20,
        force=False,
    )

    assert report["metrics"]["auto_repair_event_count"] == 1
    assert report["metrics"]["auto_repair_passed_count"] == 1
    assert report["metrics"]["auto_repair_memory_bridge_success_count"] == 1
    assert report["auto_repair_decision_counts"] == {"passed": 1}
    assert report["auto_repair_memory_bridge"] == {
        "success_count": 1,
        "smoke_executed_count": 1,
        "repair_policy_recommended_action_count": 1,
        "batch_memory_constraint_action_count": 1,
    }


def test_repair_policy_summary_rolls_up_converged_and_nonconverged_repairs(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_repair_policy_summary import (
        build_repair_policy_summary,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    store.append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "passed",
            "repair_convergence": {
                "decision": "converged",
                "input_repair_errors": ["learning_constraint_smoke_gate"],
                "repeated_audit_errors": [],
            },
            "chain_learning_signals": ["audit_repair_convergence"],
        },
    )
    store.append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "failed",
            "repair_convergence": {
                "decision": "nonconverged",
                "input_repair_errors": ["repair_convergence_consistency"],
                "repeated_audit_errors": ["repair_convergence_consistency"],
            },
            "chain_learning_signals": ["audit_repair_nonconvergence"],
        },
    )

    report = build_repair_policy_summary(
        trace_path=trace_path,
        output_path=tmp_path / "repair_policy.json",
        recent_window=10,
        force=False,
    )

    assert report["state_hint"] == "HERMES_REPAIR_POLICY_SUMMARY_BUILT"
    assert report["converged_repair_errors"] == ["learning_constraint_smoke_gate"]
    assert report["nonconverged_repair_errors"] == [
        "repair_convergence_consistency"
    ]
    assert report["repair_error_stats"]["learning_constraint_smoke_gate"] == {
        "attempt_count": 1,
        "converged_count": 1,
        "nonconverged_count": 0,
        "last_decision": "converged",
    }
    assert report["repair_error_stats"]["repair_convergence_consistency"] == {
        "attempt_count": 1,
        "converged_count": 0,
        "nonconverged_count": 1,
        "last_decision": "nonconverged",
    }
    assert report["recommended_actions"] == [
        {
            "audit_error": "repair_convergence_consistency",
            "action": "switch_repair_strategy_or_reload_missing_evidence",
            "reason": "repair_nonconverged",
        },
        {
            "audit_error": "learning_constraint_smoke_gate",
            "action": "prefer_previously_converged_repair_path",
            "reason": "repair_converged",
        },
    ]
    assert report["metrics"] == {
        "event_count": 2,
        "repair_attempt_count": 2,
        "converged_repair_error_count": 1,
        "nonconverged_repair_error_count": 1,
        "recommended_action_count": 2,
    }
    assert Path(report["repair_policy_summary_path"]).exists()


def test_prepare_pde_literature_extraction_skill(tmp_path: Path) -> None:
    from pde_agent.core.pde_literature_extraction_skill import (
        prepare_pde_literature_extraction_skill,
    )

    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {
                "state_hint": "LITERATURE_EXTRACTION_CONTEXT_PREPARED",
                "source_id": "arxiv:0000.0000",
                "context_units": [
                    {
                        "unit_id": "u1",
                        "heading": "Method",
                        "text": "Fourier modes are used for operator learning.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    skill = prepare_pde_literature_extraction_skill(
        literature_extraction_context_path=context_path,
        output_path=tmp_path / "skill_task.json",
        operator_family="FNO",
        target_problem="PDE_Burgers",
        force=False,
    )

    assert skill["state_hint"] == "PDE_LITERATURE_EXTRACTION_SKILL_PREPARED"
    assert skill["agent_contract"]["output_state_hint"] == "ALGORITHM_CARDS_EXTRACTED"
    assert skill["agent_contract"]["candidate_novelty_policy"] == {
        "new_family": "Use normal AlgorithmCard fields for a new mechanism family.",
        "same_family_retry": (
            "Include mutation_evidence when the card is a deliberate variant "
            "of a previously failed or deprioritized candidate family."
        ),
    }
    assert skill["agent_contract"]["optional_card_fields"]["mutation_evidence"][
        "required_when_present"
    ] == ["parent_candidate_family", "mutation_type", "reason"]
    assert "AlgorithmCard v2" in skill["agent_prompt"]
    assert "mutation_evidence" in skill["agent_prompt"]
    assert "new algorithm family" in skill["agent_prompt"]
    assert "Do not invent results" in skill["agent_prompt"]


def test_hypothesis_learning_constraint_smoke_runs_trace_to_handoff(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_learning_constraint_smoke import (
        run_hypothesis_learning_constraint_smoke,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="bounded_executor_completion",
        payload={
            "algorithm_candidate_id": "candidate_repeated_local_strategy",
            "candidate_family": "local_strategy_skill",
            "outcome": "deprioritize",
            "score_delta": -12.0,
            "rules_passed": True,
        },
    )
    library_path = tmp_path / "local_strategy_library.json"
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
                        "hypothesis": "Repeat local strategy.",
                        "mechanism": "Same family retry without mutation.",
                        "implementation_hooks": ["run-hermes-batch-optimization-loop"],
                        "known_risks": ["repeat failure"],
                        "priority": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")

    report = run_hypothesis_learning_constraint_smoke(
        trace_path=trace_path,
        output_dir=tmp_path / "smoke",
        local_strategy_library_path=library_path,
        checkpoint_paths=[checkpoint],
        candidate_generator="sem_window_blend",
        min_viable_candidate_count=999,
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN"
    assert report["decision"] == "passed"
    assert report["learning_constraints"]["avoid_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert report["metrics"]["learning_constraint_skipped_count"] == 1
    assert report["candidate_families_after_handoff"] == [
        "official_checkpoint_family"
    ]
    assert Path(report["artifacts"]["smoke_report_path"]).exists()
    assert Path(report["artifacts"]["candidate_plan_path"]).exists()
    assert Path(report["artifacts"]["expansion_handoff_report_path"]).exists()


def test_hypothesis_learning_constraint_smoke_mutates_local_strategy_reason_codes(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_learning_constraint_smoke import (
        run_hypothesis_learning_constraint_smoke,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="hypothesis_lab_run",
        payload={
            "algorithm_candidate_id": (
                "candidate_local_strategy_task1_early_subsegment_patch_v0"
            ),
            "candidate_family": "local_strategy_skill",
            "outcome": "deprioritize",
            "score_delta": 0.0,
            "reason_codes": ["score_regressed", "ablation_missing_or_weak"],
        },
    )
    library_path = tmp_path / "local_strategy_library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT",
                "entries": [
                    {
                        "strategy_id": "task1_early_subsegment_patch_v0",
                        "title": "Task 1 early subsegment patch",
                        "source_type": "manual",
                        "hypothesis": "Patch early subsegments.",
                        "mechanism": "Select early sub-window sources.",
                        "implementation_hooks": [
                            "run-hermes-batch-optimization-loop",
                            "hypothesis_lab_early_subsegment_patch_search",
                        ],
                        "known_risks": ["narrow-window overfit"],
                        "priority": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_hypothesis_learning_constraint_smoke(
        trace_path=trace_path,
        output_dir=tmp_path / "smoke",
        local_strategy_library_path=library_path,
        checkpoint_paths=[],
        candidate_generator="sem_window_blend",
        min_viable_candidate_count=999,
        force=False,
    )

    candidates = json.loads(
        Path(report["artifacts"]["algorithm_candidates_path"]).read_text(
            encoding="utf-8"
        )
    )["algorithm_candidates"]
    assert report["decision"] == "passed"
    assert candidates[0]["candidate_family"] == "local_strategy_skill"
    assert candidates[0]["algorithm_candidate_id"].endswith(
        "_mutation_robust_reference_ablation_v0"
    )
    assert candidates[0]["mutation_evidence"]["mutation_type"] == (
        "robust_reference_ablation_retry"
    )


def test_hypothesis_learning_constraint_smoke_accepts_scheduler_exclusion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_learning_constraint_smoke import (
        run_hypothesis_learning_constraint_smoke,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for index in range(2):
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"postprocess_bad_{index}",
                "candidate_family": "postprocess",
                "outcome": "failed",
                "score_delta": -0.2,
            },
        )
    library_path = tmp_path / "local_strategy_library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT",
                "entries": [
                    {
                        "strategy_id": "fresh_strategy_v0",
                        "title": "Fresh strategy",
                        "source_type": "manual",
                        "hypothesis": "Try a new mechanism.",
                        "mechanism": "Avoid the repeated failed postprocess family.",
                        "implementation_hooks": ["run-hermes-batch-optimization-loop"],
                        "known_risks": [],
                        "priority": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_hypothesis_learning_constraint_smoke(
        trace_path=trace_path,
        output_dir=tmp_path / "smoke",
        local_strategy_library_path=library_path,
        checkpoint_paths=[],
        candidate_generator="postprocess",
        min_viable_candidate_count=999,
        force=False,
    )

    assert report["decision"] == "passed"
    assert report["learning_constraints"]["avoid_candidate_families"] == [
        "postprocess"
    ]
    assert report["metrics"]["learning_constraint_excluded_count"] == 1
    assert report["metrics"]["learning_constraint_skipped_count"] == 0


def test_hermes_runtime_memory_bridge_smoke_runs_runtime_to_audit(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_runtime_memory_bridge_smoke import (
        run_hermes_runtime_memory_bridge_smoke,
    )

    context = np.ones((1, 10, 256), dtype=np.float32)
    future = np.ones((1, 190, 256), dtype=np.float32)
    target = np.concatenate([context, future], axis=1)
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_target.hdf5", target)
    future_path = tmp_path / "future.npz"
    np.savez(future_path, prediction=future)
    submission = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_package",
        submission_id="team_hermes",
        method="external_prediction",
        source_prediction_path=future_path,
        train_time=0.0,
        inference_time=0.0,
        force=True,
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_audit_chain",
                "payload": {
                    "repair_convergence": {
                        "decision": "converged",
                        "input_repair_errors": ["repair_convergence_consistency"],
                        "repeated_audit_errors": [],
                    },
                    "chain_learning_signals": ["audit_repair_convergence"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_convergence_memory_smoke",
                        "audit_error": "repair_convergence_consistency",
                        "priority": "high",
                        "repair_scope": "candidate_plan_and_heartbeat_schedule",
                        "recommended_action": (
                            "prefer_previously_converged_repair_path"
                        ),
                        "next_command": "run-hermes-runtime-closed-loop",
                    }
                ],
                "metrics": {"repair_task_count": 1},
            }
        ),
        encoding="utf-8",
    )

    report = run_hermes_runtime_memory_bridge_smoke(
        trace_path=trace_path,
        output_dir=tmp_path / "runtime_memory_bridge_smoke",
        initial_submission_dir=submission["artifacts"]["submission_dir"],
        task1_test_path=test_path,
        task1_target_path=target_path,
        audit_repair_tasks_path=repair_path,
        candidate_generator="postprocess",
        force=False,
    )

    assert report["state_hint"] == "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN"
    assert report["decision"] == "passed"
    assert report["metrics"]["runtime_audit_failed_check_count"] == 0
    assert report["metrics"]["repair_policy_recommended_action_count"] == 1
    assert report["metrics"]["batch_memory_constraint_action_count"] == 1
    assert report["memory_bridge"]["passed"] is True
    assert Path(report["artifacts"]["runtime_loop_report_path"]).exists()
    assert Path(report["artifacts"]["runtime_closed_loop_audit_path"]).exists()
    runtime_payload = json.loads(
        Path(report["artifacts"]["runtime_loop_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert runtime_payload["artifacts"][
        "runtime_memory_bridge_smoke_report_path"
    ] == report["artifacts"]["smoke_report_path"]
    assert runtime_payload["audit_command"][
        "require_runtime_memory_bridge_smoke"
    ] is True


def test_hermes_auto_repair_memory_acceptance_smoke_runs_full_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import hermes_auto_repair_memory_acceptance_smoke as smoke

    calls = []

    def fake_chain(**kwargs):
        calls.append(("chain", kwargs))
        chain_report_path = (
            Path(kwargs["output_dir"]) / "hermes_runtime_audit_chain_report.json"
        )
        chain_report_path.parent.mkdir(parents=True, exist_ok=True)
        chain_report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
            "chain_report_path": str(chain_report_path),
            "decision": "repaired",
            "metrics": {
                "auto_repair_attempted": True,
                "auto_repair_decision": "passed",
                "auto_repair_memory_bridge_passed": True,
                "auto_repair_repair_policy_recommended_action_count": 1,
                "auto_repair_batch_memory_constraint_action_count": 1,
            },
            "artifacts": {
                "auto_repair_report_path": str(tmp_path / "auto_repair.json"),
                "audit_repair_tasks_path": str(tmp_path / "audit_repair_tasks.json"),
            },
            "auto_repair": {
                "artifacts": {
                    "runtime_loop_report_path": str(tmp_path / "runtime.json"),
                    "smoke_report_path": str(tmp_path / "smoke.json"),
                    "audit_report_path": str(tmp_path / "audit.json"),
                },
                "metrics": {
                    "memory_bridge_passed": True,
                    "repair_policy_recommended_action_count": 1,
                    "batch_memory_constraint_action_count": 1,
                },
            },
            "repair_convergence": {
                "decision": "converged",
                "input_repair_errors": ["runtime_memory_bridge_smoke_gate"],
                "repeated_audit_errors": [],
            },
            "next_command": "run-hermes-runtime-audit-chain",
        }

    def fake_kpi(**kwargs):
        calls.append(("kpi", kwargs))
        return {
            "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
            "metrics": {"auto_repair_memory_bridge_success_count": 1},
            "auto_repair_memory_bridge": {"success_count": 1},
        }

    def fake_repair_policy(**kwargs):
        calls.append(("repair_policy", kwargs))
        return {
            "state_hint": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
            "repair_policy_summary_path": str(kwargs["output_path"]),
            "converged_repair_errors": ["runtime_memory_bridge_smoke_gate"],
            "recommended_actions": [
                {
                    "audit_error": "runtime_memory_bridge_smoke_gate",
                    "action": "prefer_previously_converged_repair_path",
                    "reason": "repair_converged",
                }
            ],
        }

    def fake_chronicle(**kwargs):
        calls.append(("chronicle", kwargs))
        Path(kwargs["output_path"]).write_text(
            "Hermes Research Chronicle\n\nRepair Policy Memory:\n"
            "- auto_repair path=chain.json auto_repair_decision=passed "
            "memory_bridge_passed=True auto_repair_action_count=1 "
            "nonconverged_repair_errors=none "
            "converged_repair_errors=runtime_memory_bridge_smoke_gate\n"
            "- repair_action source=auto_repair "
            "audit_error=runtime_memory_bridge_smoke_gate "
            "action=prefer_previously_converged_repair_path "
            "reason=repair_converged\n",
            encoding="utf-8",
        )
        return {
            "state_hint": "RESEARCH_CHRONICLE_BUILT",
            "artifacts": {"chronicle_path": str(kwargs["output_path"])},
            "metrics": {"artifact_count": 1},
        }

    def fake_planner(**kwargs):
        calls.append(("planner", kwargs))
        return {
            "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
            "candidate_plan_path": str(kwargs["output_path"]),
            "learning_constraints": {
                "repair_convergence_errors": ["runtime_memory_bridge_smoke_gate"],
                "repair_policy_recommended_actions": [
                    {
                        "audit_error": "runtime_memory_bridge_smoke_gate",
                        "action": "prefer_previously_converged_repair_path",
                        "reason": "repair_converged",
                    }
                ],
            },
            "next_command": "run-hermes-heartbeat",
        }

    def fake_audit(**kwargs):
        calls.append(("strict_audit", kwargs))
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "decision": "passed",
            "summary": {"check_count": 20, "failed_check_count": 0},
            "errors": [],
            "audit_report_path": str(kwargs["output_path"]),
        }

    monkeypatch.setattr(smoke, "run_hermes_runtime_audit_chain", fake_chain)
    monkeypatch.setattr(smoke, "build_hypothesis_kpi_report", fake_kpi)
    monkeypatch.setattr(smoke, "build_repair_policy_summary", fake_repair_policy)
    monkeypatch.setattr(smoke, "build_research_chronicle", fake_chronicle)
    monkeypatch.setattr(smoke, "build_hypothesis_candidate_plan", fake_planner)
    monkeypatch.setattr(smoke, "audit_hermes_runtime_closed_loop", fake_audit)

    initial_dir = tmp_path / "submission"
    initial_dir.mkdir()
    report = smoke.run_hermes_auto_repair_memory_acceptance_smoke(
        output_dir=tmp_path / "acceptance",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        task1_test_path=tmp_path / "task1_test.hdf5",
        task1_target_path=tmp_path / "task1_target.hdf5",
        candidate_generator="postprocess",
        force=False,
    )

    assert report["state_hint"] == "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN"
    assert report["decision"] == "passed"
    assert report["checks"] == {
        "auto_repair_passed": True,
        "memory_bridge_passed": True,
        "kpi_internalized": True,
        "repair_policy_internalized": True,
        "chronicle_internalized": True,
        "planner_internalized": True,
        "strict_audit_passed": True,
    }
    assert report["metrics"]["passed_check_count"] == 7
    assert report["artifacts"]["smoke_report_path"].endswith(
        "hermes_auto_repair_memory_acceptance_smoke.json"
    )
    assert report["artifacts"]["candidate_plan_path"].endswith(
        "candidate_plan_after_acceptance.json"
    )
    assert calls[0][0] == "chain"
    assert calls[0][1]["auto_repair"] is True
    assert calls[-1][0] == "strict_audit"
    assert calls[-1][1]["require_runtime_memory_bridge_smoke"] is True
    assert Path(report["artifacts"]["smoke_report_path"]).exists()


def test_hypothesis_learning_constraint_smoke_command_returns_payload(
    tmp_path: Path, monkeypatch
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    def fake_smoke(**kwargs):
        return {
            "state_hint": "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN",
            "decision": "passed",
            "artifacts": {
                "smoke_report_path": str(tmp_path / "smoke.json"),
                "hypothesis_kpi_report_path": str(tmp_path / "kpi.json"),
                "candidate_plan_path": str(tmp_path / "plan.json"),
                "expansion_handoff_report_path": str(tmp_path / "handoff.json"),
            },
            "metrics": {
                "learning_constraint_skipped_count": 1,
                "deprioritized_family_count": 1,
            },
            "learning_constraints": {
                "avoid_candidate_families": ["local_strategy_skill"],
            },
            "next_command": "run-hermes-runtime-closed-loop",
        }

    monkeypatch.setattr(
        runtime_commands,
        "run_hypothesis_learning_constraint_smoke",
        fake_smoke,
    )

    payload = runtime_commands.run_hypothesis_learning_constraint_smoke_command(
        trace_path=str(tmp_path / "trace.jsonl"),
        output_dir=str(tmp_path / "smoke"),
        local_strategy_library_path=str(tmp_path / "library.json"),
        checkpoint_paths=[str(tmp_path / "checkpoint.tar")],
        candidate_generator="sem_window_blend",
        min_viable_candidate_count=999,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN"
    assert payload["metrics"]["learning_constraint_skipped_count"] == 1
    assert payload["artifacts"]["learning_constraint_smoke_json"].endswith(
        "smoke.json"
    )
    assert payload["artifacts"]["hypothesis_kpi_report_json"].endswith("kpi.json")
    assert payload["artifacts"]["candidate_plan_json"].endswith("plan.json")
    assert payload["artifacts"]["expansion_handoff_report_json"].endswith(
        "handoff.json"
    )
    assert payload["artifacts"]["candidate_plan_path"].endswith("plan.json")


def test_hermes_runtime_memory_bridge_smoke_command_returns_payload(
    tmp_path: Path, monkeypatch
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    def fake_smoke(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
            "decision": "passed",
            "artifacts": {
                "smoke_report_path": str(tmp_path / "smoke.json"),
                "runtime_loop_report_path": str(tmp_path / "runtime.json"),
                "runtime_closed_loop_audit_path": str(tmp_path / "audit.json"),
                "heartbeat_report_path": str(tmp_path / "heartbeat.json"),
            },
            "metrics": {
                "runtime_audit_failed_check_count": 0,
                "repair_policy_recommended_action_count": 1,
                "batch_memory_constraint_action_count": 1,
            },
            "memory_bridge": {"passed": True},
            "next_command": "use-hermes-runtime-closed-loop",
        }

    monkeypatch.setattr(
        runtime_commands,
        "run_hermes_runtime_memory_bridge_smoke",
        fake_smoke,
    )

    payload = runtime_commands.run_hermes_runtime_memory_bridge_smoke_command(
        trace_path=str(tmp_path / "trace.jsonl"),
        output_dir=str(tmp_path / "smoke"),
        initial_submission_dir=str(tmp_path / "submission"),
        task1_test_path=str(tmp_path / "task1_test.hdf5"),
        task1_target_path=str(tmp_path / "task1_target.hdf5"),
        audit_repair_tasks_path=str(tmp_path / "audit_repair_tasks.json"),
        candidate_generator="postprocess",
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN"
    assert payload["metrics"]["decision"] == "passed"
    assert payload["metrics"]["batch_memory_constraint_action_count"] == 1
    assert payload["artifacts"]["runtime_memory_bridge_smoke_json"].endswith(
        "smoke.json"
    )
    assert payload["artifacts"]["runtime_closed_loop_audit_json"].endswith(
        "audit.json"
    )
    assert payload["artifacts"]["memory_bridge"] == {"passed": True}


def test_hermes_auto_repair_memory_acceptance_smoke_command_returns_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    def fake_smoke(**kwargs):
        return {
            "state_hint": "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN",
            "decision": "passed",
            "checks": {"planner_internalized": True},
            "artifacts": {
                "smoke_report_path": str(tmp_path / "acceptance.json"),
                "chain_report_path": str(tmp_path / "chain.json"),
                "candidate_plan_path": str(tmp_path / "plan.json"),
            },
            "metrics": {
                "passed_check_count": 7,
                "check_count": 7,
            },
            "next_command": "run-hermes-runtime-audit-chain",
        }

    monkeypatch.setattr(
        runtime_commands,
        "run_hermes_auto_repair_memory_acceptance_smoke",
        fake_smoke,
    )

    payload = runtime_commands.run_hermes_auto_repair_memory_acceptance_smoke_command(
        output_dir=str(tmp_path / "acceptance"),
        trace_path=str(tmp_path / "trace.jsonl"),
        initial_submission_dir=str(tmp_path / "submission"),
        task1_test_path=str(tmp_path / "task1_test.hdf5"),
        task1_target_path=str(tmp_path / "task1_target.hdf5"),
        candidate_generator="postprocess",
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN"
    assert payload["artifacts"]["auto_repair_memory_acceptance_smoke_json"].endswith(
        "acceptance.json"
    )
    assert payload["artifacts"]["planner_internalized"] is True
    assert payload["metrics"]["passed_check_count"] == 7
    assert payload["next_hint"] == "run-hermes-runtime-audit-chain"


def test_hermes_runtime_memory_bridge_smoke_cli(tmp_path: Path, monkeypatch) -> None:
    import pde_agent.cli as cli_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    def fake_command(**kwargs):
        return {
            "status": "ok",
            "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
            "artifacts": {
                "runtime_memory_bridge_smoke_json": str(tmp_path / "smoke.json"),
            },
            "metrics": {"decision": "passed"},
            "next_hint": "use-hermes-runtime-closed-loop",
        }

    monkeypatch.setattr(
        cli_module,
        "run_hermes_runtime_memory_bridge_smoke_command",
        fake_command,
    )

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-runtime-memory-bridge-smoke",
            "--trace-path",
            str(tmp_path / "trace.jsonl"),
            "--output-dir",
            str(tmp_path / "smoke"),
            "--initial-submission-dir",
            str(tmp_path / "submission"),
            "--task1-test-path",
            str(tmp_path / "task1_test.hdf5"),
            "--task1-target-path",
            str(tmp_path / "task1_target.hdf5"),
            "--audit-repair-tasks-path",
            str(tmp_path / "audit_repair_tasks.json"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN"
    assert payload["metrics"]["decision"] == "passed"


def test_hermes_auto_repair_memory_acceptance_smoke_cli(
    tmp_path: Path, monkeypatch
) -> None:
    import pde_agent.cli as cli_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    calls: list[dict[str, object]] = []

    def fake_command(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "state_hint": "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN",
            "artifacts": {
                "auto_repair_memory_acceptance_smoke_json": str(
                    tmp_path / "acceptance.json"
                ),
                "candidate_plan_json": str(tmp_path / "candidate_plan.json"),
            },
            "metrics": {"passed_check_count": 7},
            "next_hint": "run-hermes-runtime-audit-chain",
        }

    monkeypatch.setattr(
        cli_module,
        "run_hermes_auto_repair_memory_acceptance_smoke_command",
        fake_command,
    )

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-auto-repair-memory-acceptance-smoke",
            "--trace-path",
            str(tmp_path / "trace.jsonl"),
            "--output-dir",
            str(tmp_path / "acceptance"),
            "--initial-submission-dir",
            str(tmp_path / "submission"),
            "--task1-test-path",
            str(tmp_path / "task1_test.hdf5"),
            "--task1-target-path",
            str(tmp_path / "task1_target.hdf5"),
            "--candidate-generator",
            "postprocess",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "trace_path": str(tmp_path / "trace.jsonl"),
            "output_dir": str(tmp_path / "acceptance"),
            "initial_submission_dir": str(tmp_path / "submission"),
            "task1_test_path": str(tmp_path / "task1_test.hdf5"),
            "task1_target_path": str(tmp_path / "task1_target.hdf5"),
            "candidate_generator": "postprocess",
            "force": True,
        }
    ]
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN"
    assert payload["metrics"]["passed_check_count"] == 7
    assert payload["artifacts"]["candidate_plan_json"].endswith("candidate_plan.json")


def test_hermes_hypothesis_lab_infrastructure_audit_validates_gate_sequence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_hypothesis_lab_infrastructure_audit import (
        audit_hermes_hypothesis_lab_infrastructure,
    )

    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN",
                "decision": "passed",
                "checks": {
                    "auto_repair_passed": True,
                    "memory_bridge_passed": True,
                    "kpi_internalized": True,
                    "repair_policy_internalized": True,
                    "chronicle_internalized": True,
                    "planner_internalized": True,
                    "strict_audit_passed": True,
                },
                "metrics": {"passed_check_count": 7, "check_count": 7},
            }
        ),
        encoding="utf-8",
    )
    chain_path = tmp_path / "chain.json"
    chain_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "artifacts": {
                    "preflight_acceptance_smoke_path": str(acceptance_path),
                },
                "metrics": {
                    "completed_iteration_count": 2,
                    "failed_audit_count": 0,
                    "preflight_acceptance_smoke_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    runbook_path = tmp_path / "runbook.md"
    runbook_path.write_text(
        "run-hermes-auto-repair-memory-acceptance-smoke\n"
        "--preflight-acceptance-smoke-path\n"
        "--preflight-infrastructure-audit-path\n"
        "run_hermes_hypothesis_lab_infrastructure.py\n"
        "run-hermes-runtime-audit-chain\n",
        encoding="utf-8",
    )

    report = audit_hermes_hypothesis_lab_infrastructure(
        acceptance_smoke_path=acceptance_path,
        preflight_chain_report_path=chain_path,
        runbook_path=runbook_path,
        output_path=tmp_path / "audit.json",
        min_completed_iterations=2,
        force=False,
    )

    assert report["state_hint"] == "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED"
    assert report["decision"] == "passed"
    assert report["summary"]["check_count"] >= 13
    assert report["summary"]["failed_check_count"] == 0
    assert report["artifacts"] == {
        "acceptance_smoke_path": str(acceptance_path),
        "preflight_chain_report_path": str(chain_path),
        "runbook_path": str(runbook_path),
    }
    assert {check["check_id"] for check in report["checks"]} >= {
        "acceptance_decision_passed",
        "acceptance_all_checks_passed",
        "preflight_chain_completed_iterations",
        "preflight_artifact_matches_acceptance",
        "runbook_mentions_preflight_gate",
    }
    assert Path(report["audit_report_path"]).exists()


def test_hermes_hypothesis_lab_infrastructure_audit_accepts_readiness_chains(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_hypothesis_lab_infrastructure_audit import (
        audit_hermes_hypothesis_lab_infrastructure,
    )

    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN",
                "decision": "passed",
                "checks": {
                    "auto_repair_passed": True,
                    "memory_bridge_passed": True,
                    "kpi_internalized": True,
                    "repair_policy_internalized": True,
                    "chronicle_internalized": True,
                    "planner_internalized": True,
                    "strict_audit_passed": True,
                },
                "metrics": {"passed_check_count": 7, "check_count": 7},
            }
        ),
        encoding="utf-8",
    )
    infrastructure_audit_path = tmp_path / "real_hermes_infrastructure_audit.json"
    infrastructure_audit_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "session_id": "session_001",
                "infrastructure": {
                    "decision": "passed",
                    "check_count": 13,
                    "failed_check_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    base_chain = {
        "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
        "decision": "passed",
        "artifacts": {
            "preflight_acceptance_smoke_path": str(acceptance_path),
            "preflight_infrastructure_audit_path": str(infrastructure_audit_path),
        },
        "metrics": {
            "completed_iteration_count": 2,
            "failed_audit_count": 0,
            "preflight_acceptance_smoke_passed": True,
            "preflight_infrastructure_audit_passed": True,
            "preflight_infrastructure_check_count": 13,
        },
    }
    preflight_chain_path = tmp_path / "preflight_chain.json"
    preflight_chain_path.write_text(json.dumps(base_chain), encoding="utf-8")
    stability_chain_path = tmp_path / "stability_chain.json"
    stability_chain_path.write_text(json.dumps(base_chain), encoding="utf-8")

    heartbeat_path = tmp_path / "pressure" / "heartbeat.json"
    heartbeat_path.parent.mkdir()
    heartbeat_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_RAN",
                "batches": [
                    {
                        "candidate_generator": "sem_window_gate_blend_multi_early_smoke",
                        "candidate_diagnostics": {"attempted_candidate_count": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_path = tmp_path / "pressure" / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {"heartbeat_report_path": str(heartbeat_path)},
            }
        ),
        encoding="utf-8",
    )
    pressure_chain = {
        **base_chain,
        "iterations": [{"runtime_report_path": str(runtime_path)}],
    }
    candidate_pressure_chain_path = tmp_path / "candidate_pressure_chain.json"
    candidate_pressure_chain_path.write_text(
        json.dumps(pressure_chain),
        encoding="utf-8",
    )
    runbook_path = tmp_path / "runbook.md"
    runbook_path.write_text(
        "run-hermes-auto-repair-memory-acceptance-smoke\n"
        "--preflight-acceptance-smoke-path\n"
        "--preflight-infrastructure-audit-path\n"
        "run_hermes_hypothesis_lab_infrastructure.py\n"
        "run-hermes-runtime-audit-chain\n",
        encoding="utf-8",
    )

    report = audit_hermes_hypothesis_lab_infrastructure(
        acceptance_smoke_path=acceptance_path,
        preflight_chain_report_path=preflight_chain_path,
        runbook_path=runbook_path,
        output_path=tmp_path / "audit.json",
        min_completed_iterations=2,
        stability_chain_report_path=stability_chain_path,
        candidate_pressure_chain_report_path=candidate_pressure_chain_path,
        force=False,
    )

    assert report["decision"] == "passed"
    assert report["artifacts"]["stability_chain_report_path"] == str(
        stability_chain_path
    )
    assert report["artifacts"]["candidate_pressure_chain_report_path"] == str(
        candidate_pressure_chain_path
    )
    assert {check["check_id"] for check in report["checks"]} >= {
        "stability_chain_dual_preflight_passed",
        "candidate_pressure_chain_dual_preflight_passed",
        "candidate_pressure_attempted_nontrivial_family",
        "runbook_mentions_infrastructure_preflight_gate",
    }


def test_hermes_hypothesis_lab_infrastructure_audit_command_and_cli(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pde_agent.cli as cli_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    calls: list[dict[str, object]] = []

    def fake_command(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "state_hint": "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED",
            "artifacts": {
                "infrastructure_audit_json": str(tmp_path / "audit.json"),
            },
            "metrics": {"decision": "passed", "failed_check_count": 0},
            "next_hint": "run-hermes-runtime-audit-chain",
        }

    monkeypatch.setattr(
        cli_module,
        "audit_hermes_hypothesis_lab_infrastructure_command",
        fake_command,
    )

    result = CliRunner().invoke(
        app,
        [
            "audit-hermes-hypothesis-lab-infrastructure",
            "--acceptance-smoke-path",
            str(tmp_path / "acceptance.json"),
            "--preflight-chain-report-path",
            str(tmp_path / "chain.json"),
            "--runbook-path",
            str(tmp_path / "runbook.md"),
            "--output-path",
            str(tmp_path / "audit.json"),
            "--min-completed-iterations",
            "2",
            "--stability-chain-report-path",
            str(tmp_path / "stability.json"),
            "--candidate-pressure-chain-report-path",
            str(tmp_path / "pressure.json"),
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "acceptance_smoke_path": str(tmp_path / "acceptance.json"),
            "preflight_chain_report_path": str(tmp_path / "chain.json"),
            "runbook_path": str(tmp_path / "runbook.md"),
            "output_path": str(tmp_path / "audit.json"),
            "min_completed_iterations": 2,
            "stability_chain_report_path": str(tmp_path / "stability.json"),
            "candidate_pressure_chain_report_path": str(tmp_path / "pressure.json"),
            "force": True,
        }
    ]
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED"
    assert payload["metrics"]["failed_check_count"] == 0


def test_resolve_audit_repair_task_command_builds_runnable_argv(
    tmp_path: Path,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    repair_path = tmp_path / "audit_repair_tasks.json"
    output_path = tmp_path / "repair_resolution.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_learning_constraint_smoke_gate",
                        "audit_error": "learning_constraint_smoke_gate",
                        "next_command": "run-hypothesis-learning-constraint-smoke",
                        "suggested_command_args": {
                            "trace_path": str(tmp_path / "trace.jsonl"),
                            "output_dir": str(tmp_path / "learning_constraint_smoke"),
                            "local_strategy_library_path": str(
                                tmp_path / "local_strategy_library.json"
                            ),
                            "candidate_generator": "postprocess",
                            "checkpoint_paths": [],
                            "min_viable_candidate_count": 999,
                            "force": True,
                        },
                        "missing_inputs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = runtime_commands.resolve_audit_repair_task_command(
        audit_repair_tasks_path=str(repair_path),
        output_path=str(output_path),
        task_id=None,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_AUDIT_REPAIR_TASK_RESOLVED"
    assert payload["metrics"]["ready_to_run"] is True
    assert payload["artifacts"]["next_command"] == (
        "run-hypothesis-learning-constraint-smoke"
    )
    assert payload["artifacts"]["argv"] == [
        "uv",
        "run",
        "--frozen",
        "pde-agent",
        "run-hypothesis-learning-constraint-smoke",
        "--trace-path",
        str(tmp_path / "trace.jsonl"),
        "--output-dir",
        str(tmp_path / "learning_constraint_smoke"),
        "--local-strategy-library-path",
        str(tmp_path / "local_strategy_library.json"),
        "--candidate-generator",
        "postprocess",
        "--min-viable-candidate-count",
        "999",
        "--force",
    ]
    assert payload["artifacts"]["command_string"] == (
        "uv run --frozen pde-agent run-hypothesis-learning-constraint-smoke "
        f"--trace-path {tmp_path / 'trace.jsonl'} "
        f"--output-dir {tmp_path / 'learning_constraint_smoke'} "
        f"--local-strategy-library-path {tmp_path / 'local_strategy_library.json'} "
        "--candidate-generator postprocess --min-viable-candidate-count 999 --force"
    )
    assert payload["artifacts"]["repair_task_resolution_json"] == str(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8"))["argv"] == payload[
        "artifacts"
    ]["argv"]
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "command_string"
    ] == payload["artifacts"]["command_string"]


def test_resolve_audit_repair_task_command_builds_runtime_memory_smoke_argv(
    tmp_path: Path,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    repair_path = tmp_path / "audit_repair_tasks.json"
    output_path = tmp_path / "repair_resolution.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_runtime_memory_bridge_smoke_gate",
                        "audit_error": "runtime_memory_bridge_smoke_gate",
                        "next_command": "run-hermes-runtime-memory-bridge-smoke",
                        "suggested_command_args": {
                            "trace_path": str(tmp_path / "trace.jsonl"),
                            "output_dir": str(
                                tmp_path / "runtime_memory_bridge_smoke"
                            ),
                            "initial_submission_dir": str(tmp_path / "submission"),
                            "task1_test_path": str(tmp_path / "task1_test.hdf5"),
                            "task1_target_path": str(tmp_path / "task1_target.hdf5"),
                            "audit_repair_tasks_path": str(
                                tmp_path / "audit_repair_tasks.json"
                            ),
                            "candidate_generator": "postprocess",
                            "force": True,
                        },
                        "missing_inputs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = runtime_commands.resolve_audit_repair_task_command(
        audit_repair_tasks_path=str(repair_path),
        output_path=str(output_path),
        task_id=None,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["metrics"]["ready_to_run"] is True
    assert payload["artifacts"]["next_command"] == (
        "run-hermes-runtime-memory-bridge-smoke"
    )
    assert payload["artifacts"]["argv"] == [
        "uv",
        "run",
        "--frozen",
        "pde-agent",
        "run-hermes-runtime-memory-bridge-smoke",
        "--trace-path",
        str(tmp_path / "trace.jsonl"),
        "--output-dir",
        str(tmp_path / "runtime_memory_bridge_smoke"),
        "--initial-submission-dir",
        str(tmp_path / "submission"),
        "--task1-test-path",
        str(tmp_path / "task1_test.hdf5"),
        "--task1-target-path",
        str(tmp_path / "task1_target.hdf5"),
        "--audit-repair-tasks-path",
        str(tmp_path / "audit_repair_tasks.json"),
        "--candidate-generator",
        "postprocess",
        "--force",
    ]


def test_resolve_audit_repair_task_command_reports_missing_inputs(
    tmp_path: Path,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_learning_constraint_smoke_gate",
                        "audit_error": "learning_constraint_smoke_gate",
                        "next_command": "run-hypothesis-learning-constraint-smoke",
                        "suggested_command_args": {
                            "trace_path": str(tmp_path / "trace.jsonl"),
                            "output_dir": str(tmp_path / "learning_constraint_smoke"),
                        },
                        "missing_inputs": ["local_strategy_library_path"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = runtime_commands.resolve_audit_repair_task_command(
        audit_repair_tasks_path=str(repair_path),
        output_path=str(tmp_path / "repair_resolution.json"),
        task_id=None,
        provided_inputs=[],
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["metrics"]["ready_to_run"] is False
    assert payload["artifacts"]["missing_inputs"] == ["local_strategy_library_path"]
    assert payload["next_hint"] == "fill-repair-task-missing-inputs"


def test_resolve_audit_repair_task_command_accepts_provided_inputs(
    tmp_path: Path,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    repair_path = tmp_path / "audit_repair_tasks.json"
    library_path = tmp_path / "local_strategy_library.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_learning_constraint_smoke_gate",
                        "audit_error": "learning_constraint_smoke_gate",
                        "next_command": "run-hypothesis-learning-constraint-smoke",
                        "suggested_command_args": {
                            "trace_path": str(tmp_path / "trace.jsonl"),
                            "output_dir": str(tmp_path / "learning_constraint_smoke"),
                        },
                        "missing_inputs": ["local_strategy_library_path"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = runtime_commands.resolve_audit_repair_task_command(
        audit_repair_tasks_path=str(repair_path),
        output_path=str(tmp_path / "repair_resolution.json"),
        task_id=None,
        provided_inputs=[f"local_strategy_library_path={library_path}"],
        force=False,
    )

    assert payload["metrics"]["ready_to_run"] is True
    assert payload["artifacts"]["missing_inputs"] == []
    assert "--local-strategy-library-path" in payload["artifacts"]["argv"]
    assert str(library_path) in payload["artifacts"]["argv"]
    assert payload["next_hint"] == "run-hypothesis-learning-constraint-smoke"


def test_resolve_audit_repair_task_cli(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    repair_path = tmp_path / "audit_repair_tasks.json"
    output_path = tmp_path / "repair_resolution.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_learning_constraint_smoke_gate",
                        "audit_error": "learning_constraint_smoke_gate",
                        "next_command": "run-hypothesis-learning-constraint-smoke",
                        "suggested_command_args": {
                            "trace_path": str(tmp_path / "trace.jsonl"),
                            "output_dir": str(tmp_path / "learning_constraint_smoke"),
                            "local_strategy_library_path": str(
                                tmp_path / "local_strategy_library.json"
                            ),
                            "candidate_generator": "postprocess",
                            "force": True,
                        },
                        "missing_inputs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "resolve-audit-repair-task",
            "--audit-repair-tasks-path",
            str(repair_path),
            "--output-path",
            str(output_path),
            "--provided-input",
            f"local_strategy_library_path={tmp_path / 'local_strategy_library.json'}",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_AUDIT_REPAIR_TASK_RESOLVED"
    assert payload["metrics"]["ready_to_run"] is True
    assert payload["artifacts"]["argv"][:5] == [
        "uv",
        "run",
        "--frozen",
        "pde-agent",
        "run-hypothesis-learning-constraint-smoke",
    ]
    assert output_path.exists()


def test_run_resolved_audit_repair_task_executes_runtime_and_audit(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_resolved_audit_repair_task,
    )

    repair_path = tmp_path / "audit_repair_tasks.json"
    trace_path = tmp_path / "trace.jsonl"
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_repair_convergence_consistency",
                        "audit_error": "repair_convergence_consistency",
                        "next_command": "run-hermes-runtime-closed-loop",
                        "suggested_command_args": {
                            "output_dir": str(tmp_path / "runtime_repair"),
                            "trace_path": str(trace_path),
                            "initial_submission_dir": str(initial_dir),
                            "candidate_generator": "postprocess",
                            "max_batches": 1,
                            "force": True,
                        },
                        "missing_inputs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    runtime_calls = []
    audit_calls = []

    def fake_runtime(**kwargs):
        runtime_calls.append(kwargs)
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            json.dumps({"state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN"}),
            encoding="utf-8",
        )
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 101.0},
            "audit_command": {
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_guard_effect": False,
                "require_learning_constraint_smoke": False,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        audit_calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "passed",
            "errors": [],
            "summary": {"failed_check_count": 0},
            "next_command": "run-hermes-runtime-audit-chain",
        }

    report = run_resolved_audit_repair_task(
        audit_repair_tasks_path=repair_path,
        output_dir=tmp_path / "resolved_repair",
        task_id=None,
        provided_inputs=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    assert report["state_hint"] == "HERMES_AUDIT_REPAIR_TASK_RAN"
    assert report["decision"] == "passed"
    assert report["repair_task_resolution"]["ready_to_run"] is True
    assert report["repair_task_resolution"]["next_command"] == (
        "run-hermes-runtime-closed-loop"
    )
    assert runtime_calls[0]["audit_repair_tasks_path"] == repair_path
    assert runtime_calls[0]["output_dir"] == tmp_path / "runtime_repair"
    assert audit_calls[0]["runtime_report_path"] == (
        tmp_path / "runtime_repair" / "hermes_runtime_closed_loop_report.json"
    )
    assert Path(report["artifacts"]["repair_task_run_report_path"]).exists()


def test_run_resolved_audit_repair_task_executes_runtime_memory_bridge_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import hermes_runtime_audit_chain as audit_chain

    repair_path = tmp_path / "audit_repair_tasks.json"
    trace_path = tmp_path / "trace.jsonl"
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_target.hdf5"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_runtime_memory_bridge_smoke_gate",
                        "audit_error": "runtime_memory_bridge_smoke_gate",
                        "next_command": "run-hermes-runtime-memory-bridge-smoke",
                        "suggested_command_args": {
                            "trace_path": str(trace_path),
                            "output_dir": str(
                                tmp_path / "runtime_memory_bridge_smoke"
                            ),
                            "initial_submission_dir": str(initial_dir),
                            "task1_test_path": str(task1_test),
                            "task1_target_path": str(task1_target),
                            "audit_repair_tasks_path": str(repair_path),
                            "candidate_generator": "postprocess",
                            "force": True,
                        },
                        "missing_inputs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    smoke_calls = []

    def fake_smoke(**kwargs):
        smoke_calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
            "decision": "passed",
            "artifacts": {
                "smoke_report_path": str(tmp_path / "smoke.json"),
                "runtime_loop_report_path": str(tmp_path / "runtime.json"),
                "runtime_closed_loop_audit_path": str(tmp_path / "audit.json"),
            },
            "metrics": {
                "runtime_audit_failed_check_count": 0,
                "repair_policy_recommended_action_count": 1,
                "batch_memory_constraint_action_count": 1,
            },
            "memory_bridge": {"passed": True},
            "next_command": "use-hermes-runtime-closed-loop",
        }

    monkeypatch.setattr(
        audit_chain,
        "run_hermes_runtime_memory_bridge_smoke",
        fake_smoke,
    )

    report = audit_chain.run_resolved_audit_repair_task(
        audit_repair_tasks_path=repair_path,
        output_dir=tmp_path / "resolved_repair",
        task_id=None,
        provided_inputs=None,
        force=False,
    )

    assert report["state_hint"] == "HERMES_AUDIT_REPAIR_TASK_RAN"
    assert report["decision"] == "passed"
    assert report["repair_task_resolution"]["next_command"] == (
        "run-hermes-runtime-memory-bridge-smoke"
    )
    assert report["metrics"]["smoke_executed"] is True
    assert report["metrics"]["memory_bridge_passed"] is True
    assert report["metrics"]["repair_policy_recommended_action_count"] == 1
    assert report["metrics"]["batch_memory_constraint_action_count"] == 1
    assert report["metrics"]["runtime_executed"] is False
    assert report["metrics"]["audit_executed"] is False
    assert report["smoke"]["memory_bridge"] == {"passed": True}
    assert report["artifacts"]["smoke_report_path"] == str(tmp_path / "smoke.json")
    assert smoke_calls[0]["trace_path"] == trace_path
    assert smoke_calls[0]["output_dir"] == tmp_path / "runtime_memory_bridge_smoke"
    assert smoke_calls[0]["initial_submission_dir"] == initial_dir
    assert smoke_calls[0]["task1_test_path"] == task1_test
    assert smoke_calls[0]["task1_target_path"] == task1_target
    assert smoke_calls[0]["audit_repair_tasks_path"] == repair_path


def test_run_resolved_audit_repair_task_command_returns_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    def fake_run(**kwargs):
        return {
            "state_hint": "HERMES_AUDIT_REPAIR_TASK_RAN",
            "repair_task_run_report_path": str(tmp_path / "repair_run.json"),
            "decision": "passed",
            "repair_task_resolution": {
                "repair_task_resolution_path": str(tmp_path / "resolution.json"),
                "next_command": "run-hermes-runtime-closed-loop",
                "ready_to_run": True,
            },
            "runtime": {
                "artifacts": {
                    "runtime_loop_report_path": str(tmp_path / "runtime.json")
                },
                "metrics": {"heartbeat_best_score": 101.0},
            },
            "audit": {
                "audit_report_path": str(tmp_path / "audit.json"),
                "decision": "passed",
                "errors": [],
            },
            "artifacts": {
                "repair_task_run_report_path": str(tmp_path / "repair_run.json"),
                "repair_task_resolution_path": str(tmp_path / "resolution.json"),
                "runtime_loop_report_path": str(tmp_path / "runtime.json"),
                "audit_report_path": str(tmp_path / "audit.json"),
            },
            "metrics": {
                "ready_to_run": True,
                "audit_failed_check_count": 0,
            },
            "next_command": "run-hermes-runtime-audit-chain",
        }

    monkeypatch.setattr(
        runtime_commands,
        "run_resolved_audit_repair_task",
        fake_run,
    )

    payload = runtime_commands.run_resolved_audit_repair_task_command(
        audit_repair_tasks_path=str(tmp_path / "audit_repair_tasks.json"),
        output_dir=str(tmp_path / "repair_run"),
        task_id=None,
        provided_inputs=[],
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_AUDIT_REPAIR_TASK_RAN"
    assert payload["artifacts"]["runtime_loop_report_json"].endswith("runtime.json")
    assert payload["artifacts"]["audit_report_json"].endswith("audit.json")
    assert payload["metrics"]["ready_to_run"] is True
    assert payload["next_hint"] == "run-hermes-runtime-audit-chain"


def test_run_resolved_audit_repair_task_command_returns_memory_bridge_smoke_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    def fake_run(**kwargs):
        return {
            "state_hint": "HERMES_AUDIT_REPAIR_TASK_RAN",
            "repair_task_run_report_path": str(tmp_path / "repair_run.json"),
            "decision": "passed",
            "repair_task_resolution": {
                "repair_task_resolution_path": str(tmp_path / "resolution.json"),
                "next_command": "run-hermes-runtime-memory-bridge-smoke",
                "ready_to_run": True,
            },
            "smoke": {
                "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
                "memory_bridge": {"passed": True},
            },
            "artifacts": {
                "repair_task_run_report_path": str(tmp_path / "repair_run.json"),
                "repair_task_resolution_path": str(tmp_path / "resolution.json"),
                "smoke_report_path": str(tmp_path / "smoke.json"),
                "runtime_loop_report_path": str(tmp_path / "runtime.json"),
                "audit_report_path": str(tmp_path / "audit.json"),
            },
            "metrics": {
                "ready_to_run": True,
                "runtime_executed": False,
                "audit_executed": False,
                "smoke_executed": True,
            },
            "next_command": "use-hermes-runtime-closed-loop",
        }

    monkeypatch.setattr(
        runtime_commands,
        "run_resolved_audit_repair_task",
        fake_run,
    )

    payload = runtime_commands.run_resolved_audit_repair_task_command(
        audit_repair_tasks_path=str(tmp_path / "audit_repair_tasks.json"),
        output_dir=str(tmp_path / "repair_run"),
        task_id=None,
        provided_inputs=[],
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_AUDIT_REPAIR_TASK_RAN"
    assert payload["artifacts"]["runtime_memory_bridge_smoke_json"].endswith(
        "smoke.json"
    )
    assert payload["artifacts"]["runtime_memory_bridge_smoke_path"].endswith(
        "smoke.json"
    )
    assert payload["artifacts"]["runtime_loop_report_json"].endswith("runtime.json")
    assert payload["artifacts"]["audit_report_json"].endswith("audit.json")
    assert payload["metrics"]["smoke_executed"] is True
    assert payload["metrics"]["runtime_executed"] is False
    assert payload["metrics"]["audit_executed"] is False
    assert payload["next_hint"] == "use-hermes-runtime-closed-loop"


def test_run_resolved_audit_repair_task_cli_reports_unsupported_command(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_learning_constraint_smoke_gate",
                        "audit_error": "learning_constraint_smoke_gate",
                        "next_command": "run-hypothesis-learning-constraint-smoke",
                        "suggested_command_args": {
                            "trace_path": str(tmp_path / "trace.jsonl"),
                            "output_dir": str(tmp_path / "learning_constraint_smoke"),
                            "local_strategy_library_path": str(
                                tmp_path / "local_strategy_library.json"
                            ),
                        },
                        "missing_inputs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run-resolved-audit-repair-task",
            "--audit-repair-tasks-path",
            str(repair_path),
            "--output-dir",
            str(tmp_path / "repair_run"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_AUDIT_REPAIR_TASK_RAN"
    assert payload["artifacts"]["decision"] == "unsupported_next_command"
    assert payload["metrics"]["runtime_executed"] is False


def test_hermes_runtime_closed_loop_builds_guarded_heartbeat(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for index in range(3):
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"candidate_{index}",
                "candidate_family": "task1_early_refinement",
                "outcome": "failed",
                "score_delta": 0.0,
            },
        )
    heartbeat_calls = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "repair_policy_summary_path": str(
                    Path(kwargs["output_dir"]).parent / "repair_policy_summary.json"
                ),
                "repair_policy_recommended_action_count": 0,
            },
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    smoke_path = (
        tmp_path
        / "runtime_loop"
        / "learning_constraint_smoke"
        / "hypothesis_learning_constraint_smoke.json"
    )
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN",
                "decision": "passed",
                "trace_path": str(tmp_path / "different_trace.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    prior_chronicle_path = tmp_path / "prior_research_chronicle.md"
    prior_chronicle_path.write_text(
        "\n".join(
            [
                "Hermes Research Chronicle",
                "",
                "Repair Policy Memory:",
                "- repair_action source=runtime "
                "audit_error=repair_convergence_consistency "
                "action=switch_repair_strategy_or_reload_missing_evidence "
                "reason=repair_nonconverged",
                "",
                "Submission Decision:",
            ]
        ),
        encoding="utf-8",
    )
    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="task1_early_refinement",
        candidate_generator_fallback="postprocess",
        max_batches=1,
        input_research_chronicle_path=prior_chronicle_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["state_hint"] == "HERMES_RUNTIME_CLOSED_LOOP_RAN"
    assert report["guard"]["guard_active"] is True
    assert heartbeat_calls[0]["doom_loop_report_path"] == Path(
        report["artifacts"]["doom_loop_report_path"]
    )
    assert heartbeat_calls[0]["candidate_generator_schedule"]
    assert heartbeat_calls[0]["candidate_generator_schedule"][0] != (
        "task1_early_refinement"
    )
    assert heartbeat_calls[0]["candidate_generator"] == "task1_early_refinement"
    assert heartbeat_calls[0]["candidate_generator_fallback"] == "postprocess"
    assert Path(report["artifacts"]["hypothesis_kpi_report_path"]).exists()
    assert Path(report["artifacts"]["candidate_plan_path"]).exists()
    assert report["candidate_plan"]["expansion_required"] is False
    assert report["artifacts"]["input_research_chronicle_path"] == str(
        prior_chronicle_path
    )
    assert report["candidate_plan"]["learning_constraints"][
        "research_chronicle_repair_policy_memory"
    ]["repair_action_count"] == 1
    assert "research_chronicle_repair_policy_memory" in report["candidate_plan"][
        "learning_constraints"
    ]["reason_codes"]
    assert Path(report["artifacts"]["heartbeat_report_path"]).exists()
    assert Path(report["artifacts"]["research_chronicle_path"]).exists()
    chronicle = Path(report["artifacts"]["research_chronicle_path"]).read_text(
        encoding="utf-8"
    )
    assert "Hermes Research Chronicle" in chronicle
    assert "kind=hermes_runtime_closed_loop" in chronicle
    assert "kind=hermes_heartbeat" in chronicle
    assert "Repair Policy Memory:" in chronicle
    assert report["metrics"]["research_chronicle_built"] is True
    assert report["artifacts"]["learning_constraint_smoke_report_path"] == str(
        smoke_path
    )
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert events[-1]["event_type"] == "hermes_runtime_closed_loop"
    assert events[-1]["payload"]["outcome"] == "continue_next_heartbeat"
    assert events[-1]["payload"]["learning_constraints"] == report["candidate_plan"][
        "learning_constraints"
    ]
    assert report["audit_command"] == {
        "command": "audit-hermes-runtime-closed-loop",
        "runtime_report_path": report["artifacts"]["runtime_loop_report_path"],
        "output_path": str(tmp_path / "runtime_loop" / "hermes_runtime_closed_loop_audit.json"),
        "require_learning_constraint_smoke": True,
    }
    assert report["next_command"] == "audit-hermes-runtime-closed-loop"


def test_hermes_runtime_closed_loop_forwards_task1_reference_to_hypothesis_runner(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    heartbeat_calls = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {},
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    task1_reference = tmp_path / "task1_reference.hdf5"
    h5py = __import__("h5py")
    with h5py.File(task1_reference, "w") as handle:
        handle.create_dataset("tensor", shape=(1, 200, 256), dtype="float32")

    run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="hypothesis_lab_front25_dynamics_smoke",
        candidate_generator_fallback="auto",
        max_batches=1,
        task1_reference_path=task1_reference,
        hypothesis_candidate_options={"source_stability_guard": True},
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert heartbeat_calls[0]["candidate_generator_schedule"]
    assert heartbeat_calls[0]["candidate_generator"] == (
        "hypothesis_lab_front25_dynamics_smoke"
    )
    assert heartbeat_calls[0]["hypothesis_reference_path"] == task1_reference
    assert heartbeat_calls[0]["hypothesis_candidate_options"] == {
        "source_stability_guard": True
    }


def test_hermes_runtime_closed_loop_surfaces_candidate_space_exhaustion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )
    heartbeat_calls = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["metrics"]["candidate_space_exhausted"] is True
    assert report["candidate_plan"]["decision"] == "candidate_space_exhausted"
    assert report["candidate_plan"]["next_command"] == (
        "run-hermes-literature-strategy-expansion"
    )
    assert report["candidate_plan"]["expansion_handoff"]["next_command"] == (
        "build-local-asset-algorithm-candidates"
    )
    assert report["next_command"] == "run-hermes-literature-strategy-expansion"
    assert heartbeat_calls[0]["candidate_generator_schedule"] == [
        "fast_baseline_family"
    ]


def test_hermes_runtime_closed_loop_replans_after_expansion_axis_exhaustion(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    store.append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "failed",
            "repair_convergence": {
                "decision": "nonconverged",
                "input_repair_errors": ["repair_convergence_consistency"],
                "repeated_audit_errors": ["repair_convergence_consistency"],
            },
        },
    )
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_val.hdf5"
    with h5py.File(task1_test, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))
    with h5py.File(task1_target, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    prior_chronicle_path = tmp_path / "prior_research_chronicle.md"
    prior_chronicle_path.write_text(
        "\n".join(
            [
                "Hermes Research Chronicle",
                "",
                "Repair Policy Memory:",
                "- repair_action source=runtime "
                "audit_error=repair_convergence_consistency "
                "action=switch_repair_strategy_or_reload_missing_evidence "
                "reason=repair_nonconverged",
                "",
                "Submission Decision:",
            ]
        ),
        encoding="utf-8",
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    best_dir = tmp_path / "best"
    best_dir.mkdir()
    heartbeat_calls: list[dict[str, object]] = []
    queue_calls: list[dict[str, object]] = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": f"best_{len(heartbeat_calls)}",
                "family": kwargs["candidate_generator"],
                "score": 101.0 + len(heartbeat_calls),
                "submission_dir": str(best_dir),
            },
            "best_score": 101.0 + len(heartbeat_calls),
            "best_objective": "score",
            "best_objective_score": 101.0 + len(heartbeat_calls),
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_handoff_runner(**kwargs):
        workspace = Path(kwargs["workspace_dir"])
        workspace.mkdir(parents=True, exist_ok=True)
        handoff_path = workspace / "candidate_expansion_handoff_report.json"
        candidates_path = workspace / "local_asset_algorithm_candidates.json"
        expansion_path = workspace / "literature_strategy_expansion.json"
        handoff = {
            "schema_version": 1,
            "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
            "handoff_report_path": str(handoff_path),
            "artifacts": {
                "algorithm_candidates_path": str(candidates_path),
                "expansion_report_path": str(expansion_path),
            },
            "metrics": {
                "algorithm_candidate_count": 0,
                "expansion_action_count": 0,
            },
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
            "next_command": "plan-hypothesis-candidates",
        }
        handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
        candidates_path.write_text("{}", encoding="utf-8")
        expansion_path.write_text("{}", encoding="utf-8")
        return handoff

    def fail_queue_runner(**kwargs):
        queue_calls.append(kwargs)
        raise AssertionError("sandbox queue should not run after axis exhaustion")

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        task1_test_path=task1_test,
        task1_target_path=task1_target,
        input_research_chronicle_path=prior_chronicle_path,
        heartbeat_runner=fake_heartbeat_runner,
        expansion_handoff_runner=fake_handoff_runner,
        sandbox_queue_runner=fail_queue_runner,
        force=False,
    )

    assert len(heartbeat_calls) == 2
    assert queue_calls == []
    assert heartbeat_calls[1]["candidate_generator_schedule"] == [
        "fast_baseline_family"
    ]
    assert report["expansion_pipeline"]["expansion_exhaustion"]["exhausted"] is True
    assert Path(
        report["artifacts"]["post_expansion_candidate_plan_path"]
    ).exists()
    assert Path(
        report["artifacts"]["post_expansion_heartbeat_report_path"]
    ).exists()
    post_plan = json.loads(
        Path(report["artifacts"]["post_expansion_candidate_plan_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert post_plan["learning_constraints"]["avoid_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert post_plan["learning_constraints"]["repair_nonconvergence_errors"] == [
        "repair_convergence_consistency"
    ]
    assert post_plan["learning_constraints"]["repair_policy_recommended_actions"] == [
        {
            "audit_error": "repair_convergence_consistency",
            "action": "switch_repair_strategy_or_reload_missing_evidence",
            "reason": "repair_nonconverged",
        }
    ]
    assert post_plan["learning_constraints"][
        "research_chronicle_repair_policy_memory"
    ]["repair_action_count"] == 1
    assert report["next_command"] == "audit-hermes-runtime-closed-loop"


def test_hermes_runtime_closed_loop_passes_expansion_registry_summary_to_handoff(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_val.hdf5"
    with h5py.File(task1_test, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))
    with h5py.File(task1_target, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    registry_summary_path = tmp_path / "hermes_registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "summary_path": str(registry_summary_path),
                "metrics": {"learning_signal_count": 1},
                "learning_signals": [
                    {
                        "candidate_family": "local_strategy_skill",
                        "reason_codes": ["score_regressed"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    best_dir = tmp_path / "best"
    best_dir.mkdir()
    handoff_calls: list[dict[str, object]] = []

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        guidance = {
            "guidance_type": "escalate_after_guided_search_exhaustion",
            "reason": "memory_guided_specs_failed_to_improve",
            "source": "sem_window_gate_blend_multi_search",
            "recommended_action": "switch_gate_feature_family_or_try_new_dynamics_source",
        }
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [guidance],
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_handoff_runner(**kwargs):
        handoff_calls.append(kwargs)
        workspace = Path(kwargs["workspace_dir"])
        workspace.mkdir(parents=True, exist_ok=True)
        handoff_path = workspace / "candidate_expansion_handoff_report.json"
        candidates_path = workspace / "local_asset_algorithm_candidates.json"
        expansion_path = workspace / "literature_strategy_expansion.json"
        handoff = {
            "schema_version": 1,
            "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
            "handoff_report_path": str(handoff_path),
            "artifacts": {
                "algorithm_candidates_path": str(candidates_path),
                "expansion_report_path": str(expansion_path),
            },
            "metrics": {
                "algorithm_candidate_count": 0,
                "expansion_action_count": 0,
            },
            "expansion_exhaustion": {
                "exhausted": True,
                "reason": "local_strategy_mutation_axes_exhausted",
                "exhausted_candidate_families": ["local_strategy_skill"],
                "recommended_next_command": "plan-hypothesis-candidates",
            },
            "next_command": "plan-hypothesis-candidates",
        }
        handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
        candidates_path.write_text("{}", encoding="utf-8")
        expansion_path.write_text("{}", encoding="utf-8")
        return handoff

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        task1_test_path=task1_test,
        task1_target_path=task1_target,
        expansion_registry_summary_path=registry_summary_path,
        heartbeat_runner=fake_heartbeat_runner,
        expansion_handoff_runner=fake_handoff_runner,
        force=False,
    )

    assert handoff_calls[0]["registry_summary_path"] == registry_summary_path
    assert report["artifacts"]["input_expansion_registry_summary_path"] == str(
        registry_summary_path
    )
    assert report["metrics"]["expansion_registry_summary_loaded"] is True


def test_hermes_runtime_closed_loop_runs_queue_when_handoff_has_actions_despite_partial_exhaustion(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_val.hdf5"
    with h5py.File(task1_test, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))
    with h5py.File(task1_target, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    best_dir = tmp_path / "best"
    best_dir.mkdir()
    queue_calls: list[dict[str, object]] = []

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        guidance = {
            "guidance_type": "escalate_after_guided_search_exhaustion",
            "reason": "memory_guided_specs_failed_to_improve",
            "source": "sem_window_gate_blend_multi_search",
            "recommended_action": "switch_gate_feature_family_or_try_new_dynamics_source",
        }
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [guidance],
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_handoff_runner(**kwargs):
        workspace = Path(kwargs["workspace_dir"])
        workspace.mkdir(parents=True, exist_ok=True)
        handoff_path = workspace / "candidate_expansion_handoff_report.json"
        candidates_path = workspace / "local_asset_algorithm_candidates.json"
        expansion_path = workspace / "literature_strategy_expansion.json"
        handoff = {
            "schema_version": 1,
            "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
            "handoff_report_path": str(handoff_path),
            "artifacts": {
                "algorithm_candidates_path": str(candidates_path),
                "expansion_report_path": str(expansion_path),
            },
            "metrics": {
                "algorithm_candidate_count": 1,
                "expansion_action_count": 1,
                "expansion_executable_count": 1,
            },
            "expansion_exhaustion": {
                "exhausted": True,
                "reason": "local_strategy_mutation_axes_exhausted",
                "exhausted_candidate_families": ["local_strategy_skill"],
                "recommended_next_command": "plan-hypothesis-candidates",
            },
            "next_command": "plan-hypothesis-candidates",
        }
        handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
        candidates_path.write_text("{}", encoding="utf-8")
        expansion_path.write_text("{}", encoding="utf-8")
        return handoff

    def fake_queue_runner(**kwargs):
        queue_calls.append(kwargs)
        return {
            "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
            "queue_report_path": str(tmp_path / "queue.json"),
            "metrics": {
                "executed_count": 1,
                "skipped_count": 0,
                "passed_count": 1,
                "failed_count": 0,
                "error_count": 0,
            },
            "next_command": "evaluate-expansion-sandbox-queue",
        }

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        task1_test_path=task1_test,
        task1_target_path=task1_target,
        heartbeat_runner=fake_heartbeat_runner,
        expansion_handoff_runner=fake_handoff_runner,
        sandbox_queue_runner=fake_queue_runner,
        queue_evaluation_runner=lambda **_kwargs: {
            "queue_evaluation_report_path": str(tmp_path / "evaluation.json"),
            "artifacts": {"registry_path": str(tmp_path / "registry.json")},
            "items": [],
            "metrics": {"promotion_count": 0, "registry_event_count": 0},
            "next_command": "summarize-literature-registry-for-hermes",
        },
        registry_summary_runner=lambda **_kwargs: {
            "summary_path": str(tmp_path / "summary.json"),
            "metrics": {"learning_signal_count": 0},
            "next_command": "review-literature-registry",
        },
        force=False,
    )

    assert len(queue_calls) == 1
    assert "post_expansion_candidate_plan_path" not in report["artifacts"]
    assert report["metrics"]["expansion_sandbox_executed_count"] == 1


def test_hermes_runtime_closed_loop_carries_prior_expansion_handoff_into_plan(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    prior_handoff_path = tmp_path / "prior_handoff.json"
    prior_handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
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
                },
                "next_command": "plan-hypothesis-candidates",
            }
        ),
        encoding="utf-8",
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    heartbeat_calls = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(initial_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        expansion_handoff_report_path=prior_handoff_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["candidate_plan"]["learning_constraints"][
        "avoid_candidate_families"
    ] == ["local_strategy_skill"]
    assert report["candidate_plan"]["prior_expansion_exhaustion"]["exhausted"] is True
    assert report["artifacts"]["input_expansion_handoff_report_path"] == str(
        prior_handoff_path
    )
    assert report["metrics"]["expansion_handoff_report_loaded"] is True


def test_hermes_runtime_closed_loop_surfaces_cluster_saturation_signals(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    prior_handoff_path = tmp_path / "prior_handoff.json"
    prior_handoff_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
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
                    "recommended_next_command": "plan-hypothesis-candidates",
                },
                "next_command": "plan-hypothesis-candidates",
            }
        ),
        encoding="utf-8",
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    heartbeat_calls = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        guidance = {
            "guidance_type": "escalate_after_guided_search_exhaustion",
            "reason": "memory_guided_specs_failed_to_improve",
            "source": "sem_window_gate_blend_multi_search",
            "recommended_action": "switch_gate_feature_family_or_try_new_dynamics_source",
        }
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(initial_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [guidance],
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        expansion_handoff_report_path=prior_handoff_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["metrics"]["cluster_saturation_signal_count"] == 1
    assert report["metrics"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert report["candidate_plan"]["cluster_saturation_signals"] == [
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
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert events[-1]["payload"]["cluster_saturation_signals"] == report[
        "candidate_plan"
    ]["cluster_saturation_signals"]
    assert events[-1]["payload"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]
    assert events[-1]["payload"]["persistent_search_guidance"][0][
        "guidance_type"
    ] == "escalate_after_guided_search_exhaustion"


def test_hermes_runtime_closed_loop_does_not_heartbeat_empty_candidate_schedule(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
        "fast_baseline_family",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )
    heartbeat_calls = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        raise AssertionError("heartbeat must not run when candidate schedule is empty")

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert heartbeat_calls == []
    assert report["metrics"]["candidate_space_exhausted"] is True
    assert report["candidate_plan"]["candidate_generator_schedule"] == []
    assert report["heartbeat"]["decision"] == "candidate_space_exhausted"
    assert report["heartbeat"]["stop_reason"] == "candidate_schedule_empty"
    heartbeat_report_path = Path(report["artifacts"]["heartbeat_report_path"])
    heartbeat_report = json.loads(heartbeat_report_path.read_text(encoding="utf-8"))
    assert heartbeat_report["state_hint"] == "HERMES_HEARTBEAT_RAN"
    assert heartbeat_report["batch_count"] == 0
    assert heartbeat_report["batches"] == []
    assert heartbeat_report["decision"] == "candidate_space_exhausted"
    assert report["next_command"] == "run-hermes-literature-strategy-expansion"


def test_hermes_runtime_closed_loop_carries_audit_repair_tasks_into_plan(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    repair_task = {
        "task_id": "repair_learning_constraint_enforcement",
        "audit_error": "learning_constraint_enforcement",
        "priority": "high",
        "repair_scope": "candidate_plan_and_heartbeat_schedule",
        "recommended_action": (
            "rebuild_candidate_plan_or_adjust_heartbeat_schedule_to_respect_learning_constraints"
        ),
        "next_command": "run-hermes-runtime-closed-loop",
    }
    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [repair_task],
                "metrics": {"repair_task_count": 1},
            }
        ),
        encoding="utf-8",
    )

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        repair_policy_path = kwargs.get("repair_policy_summary_path")
        repair_actions = []
        if repair_policy_path is not None:
            repair_policy = json.loads(
                Path(repair_policy_path).read_text(encoding="utf-8")
            )
            repair_actions = repair_policy.get("recommended_actions", [])
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "repair_policy_recommended_action_count": len(repair_actions)
            },
            "batch_count": 1,
            "batches": [
                {
                    "repair_policy_recommended_actions": repair_actions,
                    "repair_policy_recommended_action_count": len(repair_actions),
                    "candidate_diagnostics": {
                        "memory_constraints": {
                            "repair_policy_recommended_actions": repair_actions,
                            "repair_policy_recommended_action_count": len(
                                repair_actions
                            ),
                        }
                    },
                }
            ],
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(initial_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [],
            "repair_policy_recommended_actions": repair_actions,
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        max_batches=1,
        audit_repair_tasks_path=repair_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["artifacts"]["input_audit_repair_tasks_path"] == str(repair_path)
    assert report["metrics"]["audit_repair_tasks_loaded"] is True
    assert report["metrics"]["audit_repair_task_count"] == 1
    assert report["candidate_plan"]["audit_repair_tasks"] == [repair_task]
    assert report["candidate_plan"]["learning_constraints"][
        "audit_repair_required"
    ] is True
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["payload"]["audit_repair_tasks"] == [repair_task]


def test_hermes_runtime_closed_loop_surfaces_repair_nonconvergence_from_plan(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    store.append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "failed",
            "failed_audit_count": 1,
            "audit_errors": ["cluster_saturation_consistency"],
            "repair_convergence": {
                "decision": "nonconverged",
                "input_repair_errors": ["cluster_saturation_consistency"],
                "repeated_audit_errors": ["cluster_saturation_consistency"],
            },
            "chain_learning_signals": ["audit_repair_nonconvergence"],
            "score_delta": 0.0,
        },
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    heartbeat_calls = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [],
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_gate_blend",
        max_batches=1,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["metrics"]["repair_nonconvergence_error_count"] == 1
    assert report["metrics"]["repair_nonconvergence_errors"] == [
        "cluster_saturation_consistency"
    ]
    assert Path(report["artifacts"]["repair_policy_summary_path"]).exists()
    assert report["metrics"]["repair_policy_recommended_action_count"] == 1
    assert heartbeat_calls[0]["repair_policy_summary_path"] == Path(
        report["artifacts"]["repair_policy_summary_path"]
    )
    assert report["candidate_plan"]["repair_nonconvergence_errors"] == [
        "cluster_saturation_consistency"
    ]
    assert report["candidate_plan"]["learning_constraints"][
        "repair_policy_recommended_actions"
    ] == [
        {
            "audit_error": "cluster_saturation_consistency",
            "action": "switch_repair_strategy_or_reload_missing_evidence",
            "reason": "repair_nonconverged",
        }
    ]
    assert report["candidate_plan"]["audit_repair_policies"][0][
        "repair_nonconverged"
    ] is True
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["payload"]["repair_nonconvergence_errors"] == [
        "cluster_saturation_consistency"
    ]


def test_hermes_runtime_closed_loop_surfaces_repair_convergence_from_plan(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    store.append_event(
        event_type="hermes_runtime_audit_chain",
        payload={
            "algorithm_candidate_id": "hermes_runtime_audit_chain",
            "candidate_family": "hermes_runtime_audit_chain",
            "outcome": "passed",
            "failed_audit_count": 0,
            "audit_errors": [],
            "repair_convergence": {
                "decision": "converged",
                "input_repair_errors": ["learning_constraint_smoke_gate"],
                "repeated_audit_errors": [],
            },
            "chain_learning_signals": ["audit_repair_convergence"],
            "score_delta": 0.0,
        },
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [],
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_gate_blend",
        max_batches=1,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["metrics"]["repair_convergence_error_count"] == 1
    assert report["metrics"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert Path(report["artifacts"]["repair_policy_summary_path"]).exists()
    assert report["candidate_plan"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert report["candidate_plan"]["learning_constraints"][
        "audit_repair_converged_error_counts"
    ] == {"learning_constraint_smoke_gate": 1}
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["payload"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]


def test_hermes_runtime_closed_loop_requires_memory_bridge_inputs_for_repair(
    tmp_path: Path,
) -> None:
    import pytest

    from pde_agent.core.hermes_runtime_closed_loop import (
        HermesRuntimeClosedLoopError,
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_memory_bridge_inputs",
                        "audit_error": "memory_bridge_inputs",
                        "priority": "high",
                        "repair_scope": "runtime_input_memory_bridge",
                        "recommended_action": (
                            "reload_input_expansion_handoff_and_registry_summary_before_candidate_planning"
                        ),
                        "next_command": "run-hermes-runtime-closed-loop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        HermesRuntimeClosedLoopError,
        match="memory_bridge_inputs repair requires",
    ):
        run_hermes_runtime_closed_loop(
            output_dir=tmp_path / "runtime_loop",
            trace_path=trace_path,
            initial_submission_dir=initial_dir,
            candidate_generator="postprocess",
            max_batches=1,
            audit_repair_tasks_path=repair_path,
            force=False,
        )


def test_hermes_runtime_closed_loop_accepts_memory_bridge_repair_with_registry_summary(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_memory_bridge_inputs",
                        "audit_error": "memory_bridge_inputs",
                        "priority": "high",
                        "repair_scope": "runtime_input_memory_bridge",
                        "recommended_action": (
                            "reload_input_expansion_handoff_and_registry_summary_before_candidate_planning"
                        ),
                        "next_command": "run-hermes-runtime-closed-loop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry_summary_path = tmp_path / "registry_summary.json"
    registry_summary_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "metrics": {"learning_signal_count": 0},
                "learning_signals": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        repair_policy_path = kwargs.get("repair_policy_summary_path")
        repair_actions = []
        if repair_policy_path is not None:
            repair_policy = json.loads(
                Path(repair_policy_path).read_text(encoding="utf-8")
            )
            repair_actions = repair_policy.get("recommended_actions", [])
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "repair_policy_recommended_action_count": len(repair_actions)
            },
            "batch_count": 1,
            "batches": [
                {
                    "repair_policy_recommended_actions": repair_actions,
                    "repair_policy_recommended_action_count": len(repair_actions),
                    "candidate_diagnostics": {
                        "memory_constraints": {
                            "repair_policy_recommended_actions": repair_actions,
                            "repair_policy_recommended_action_count": len(
                                repair_actions
                            ),
                        }
                    },
                }
            ],
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(initial_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [],
            "repair_policy_recommended_actions": repair_actions,
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        max_batches=1,
        expansion_registry_summary_path=registry_summary_path,
        audit_repair_tasks_path=repair_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["artifacts"]["input_expansion_registry_summary_path"] == str(
        registry_summary_path
    )
    assert report["metrics"]["expansion_registry_summary_loaded"] is True
    assert report["metrics"]["audit_repair_tasks_loaded"] is True


def test_hermes_runtime_closed_loop_requires_cluster_saturation_inputs_for_repair(
    tmp_path: Path,
) -> None:
    import pytest

    from pde_agent.core.hermes_runtime_closed_loop import (
        HermesRuntimeClosedLoopError,
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_cluster_saturation_consistency",
                        "audit_error": "cluster_saturation_consistency",
                        "priority": "high",
                        "repair_scope": "candidate_cluster_saturation_memory",
                        "recommended_action": (
                            "rebuild_candidate_plan_from_latest_expansion_handoff_and_kpi_saturation_signals"
                        ),
                        "next_command": "run-hermes-runtime-closed-loop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        HermesRuntimeClosedLoopError,
        match="cluster_saturation_consistency repair requires",
    ):
        run_hermes_runtime_closed_loop(
            output_dir=tmp_path / "runtime_loop",
            trace_path=trace_path,
            initial_submission_dir=initial_dir,
            candidate_generator="postprocess",
            max_batches=1,
            audit_repair_tasks_path=repair_path,
            force=False,
        )


def test_hermes_runtime_closed_loop_accepts_cluster_saturation_repair_with_handoff(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    repair_path = tmp_path / "audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_cluster_saturation_consistency",
                        "audit_error": "cluster_saturation_consistency",
                        "priority": "high",
                        "repair_scope": "candidate_cluster_saturation_memory",
                        "recommended_action": (
                            "rebuild_candidate_plan_from_latest_expansion_handoff_and_kpi_saturation_signals"
                        ),
                        "next_command": "run-hermes-runtime-closed-loop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_algorithm_node_ids": [
                        "local_strategy_a",
                        "local_strategy_b",
                    ],
                    "recommended_next_command": "plan-hypothesis-candidates",
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        repair_policy_path = kwargs.get("repair_policy_summary_path")
        repair_actions = []
        if repair_policy_path is not None:
            repair_policy = json.loads(
                Path(repair_policy_path).read_text(encoding="utf-8")
            )
            repair_actions = repair_policy.get("recommended_actions", [])
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "repair_policy_recommended_action_count": len(repair_actions)
            },
            "batch_count": 1,
            "batches": [
                {
                    "repair_policy_recommended_actions": repair_actions,
                    "repair_policy_recommended_action_count": len(repair_actions),
                    "candidate_diagnostics": {
                        "memory_constraints": {
                            "repair_policy_recommended_actions": repair_actions,
                            "repair_policy_recommended_action_count": len(
                                repair_actions
                            ),
                        }
                    },
                }
            ],
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(initial_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [],
            "repair_policy_recommended_actions": repair_actions,
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        max_batches=1,
        expansion_handoff_report_path=handoff_path,
        audit_repair_tasks_path=repair_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert report["artifacts"]["input_expansion_handoff_report_path"] == str(
        handoff_path
    )
    assert report["metrics"]["expansion_handoff_report_loaded"] is True
    assert report["metrics"]["audit_repair_tasks_loaded"] is True


def test_hermes_runtime_closed_loop_scores_initial_submission_when_schedule_empty(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    context = np.ones((1, 10, 256), dtype=np.float32)
    future = np.ones((1, 190, 256), dtype=np.float32)
    target = np.concatenate([context, future], axis=1)
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_target.hdf5", target)
    future_path = tmp_path / "future.npz"
    np.savez(future_path, prediction=future)
    submission = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_package",
        submission_id="team_hermes",
        method="external_prediction",
        source_prediction_path=future_path,
        train_time=0.0,
        inference_time=0.0,
        force=True,
    )
    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
        "fast_baseline_family",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=submission["artifacts"]["submission_dir"],
        candidate_generator="postprocess",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        task1_test_path=test_path,
        task1_target_path=target_path,
        force=False,
    )

    assert report["heartbeat"]["decision"] == "candidate_space_exhausted"
    assert report["metrics"]["heartbeat_best_score"] == 150.0
    assert report["heartbeat"]["best_candidate"]["candidate_id"] == "initial"
    heartbeat_report = json.loads(
        Path(report["artifacts"]["heartbeat_report_path"]).read_text(encoding="utf-8")
    )
    assert heartbeat_report["best_score"] == 150.0
    assert heartbeat_report["best_candidate"]["submission_dir"] == (
        submission["artifacts"]["submission_dir"]
    )
    assert Path(heartbeat_report["artifacts"]["initial_judge_report_path"]).exists()


def test_hermes_runtime_closed_loop_runs_expansion_pipeline_with_validation_pair(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_val.hdf5"
    with h5py.File(task1_test, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))
    with h5py.File(task1_target, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    calls: dict[str, list[dict[str, object]]] = {
        "handoff": [],
        "queue": [],
        "evaluation": [],
        "summary": [],
        "blocked_authoring": [],
        "authoring_audit": [],
    }

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 120.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 120.0,
            "decision": "continue_next_heartbeat",
            "stop_reason": "max_batches_exhausted",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_handoff_runner(**kwargs):
        calls["handoff"].append(kwargs)
        assert kwargs["failure_memory_path"] is None
        return {
            "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
            "handoff_report_path": str(tmp_path / "handoff.json"),
            "artifacts": {
                "algorithm_candidates_path": str(tmp_path / "candidates.json"),
                "expansion_report_path": str(tmp_path / "expansion.json"),
            },
            "metrics": {
                "algorithm_candidate_count": 2,
                "expansion_action_count": 2,
            },
            "next_command": "run-expansion-sandbox-queue",
        }

    def fake_queue_runner(**kwargs):
        calls["queue"].append(kwargs)
        return {
            "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
            "queue_report_path": str(tmp_path / "queue.json"),
            "metrics": {
                "executed_count": 1,
                "skipped_count": 1,
                "passed_count": 1,
                "failed_count": 0,
                "error_count": 0,
            },
            "next_command": "evaluate-expansion-sandbox-queue",
        }

    def fake_evaluation_runner(**kwargs):
        calls["evaluation"].append(kwargs)
        return {
            "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_EVALUATED",
            "queue_evaluation_report_path": str(tmp_path / "evaluation.json"),
            "artifacts": {"registry_path": str(tmp_path / "registry.json")},
            "items": [
                {
                    "status": "evaluated",
                    "algorithm_candidate_id": "candidate_local_checkpoint_burgers_fno",
                    "candidate_family": "official_checkpoint_family",
                    "decision": "deprioritize",
                    "score": 83.0,
                    "score_delta_vs_current_best": -37.0,
                    "evidence_path": str(tmp_path / "checkpoint_evidence.json"),
                    "promotion_path": str(tmp_path / "checkpoint_promotion.json"),
                    "registry_event_path": str(tmp_path / "checkpoint_event.json"),
                }
            ],
            "metrics": {
                "evidence_count": 1,
                "promotion_count": 0,
                "registry_event_count": 2,
                "registry_candidate_count": 2,
            },
            "next_command": "summarize-literature-registry-for-hermes",
        }

    def fake_summary_runner(**kwargs):
        calls["summary"].append(kwargs)
        return {
            "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
            "summary_path": str(tmp_path / "summary.json"),
            "metrics": {
                "candidate_action_count": 2,
                "learning_signal_count": 2,
            },
            "next_command": "create-executor-design-request",
        }

    def fake_blocked_authoring_runner(**kwargs):
        calls["blocked_authoring"].append(kwargs)
        return {
            "state_hint": "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED",
            "blocked_executor_tasks_path": str(tmp_path / "blocked_tasks.json"),
            "metrics": {"blocked_item_count": 1, "task_count": 1},
            "tasks": [
                {
                    "algorithm_candidate_id": "candidate_blocked",
                    "authoring_task_path": str(tmp_path / "authoring_task.json"),
                    "executor_path": str(tmp_path / "executor.py"),
                    "next_command": "audit-bounded-executor-authoring-task",
                }
            ],
            "next_command": "audit-bounded-executor-authoring-task",
        }

    def fake_authoring_audit_runner(**kwargs):
        calls["authoring_audit"].append(kwargs)
        return {
            "state_hint": "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED",
            "authoring_audit_path": str(tmp_path / "authoring_audit.json"),
            "authoring_task_path": str(kwargs["authoring_task_path"]),
            "decision": "approved",
            "issue_count": 0,
            "skill_path": "skills/pde_bounded_executor_author/SKILL.md",
            "next_command": "agent-author-bounded-executor",
        }

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        task1_test_path=task1_test,
        task1_target_path=task1_target,
        heartbeat_runner=fake_heartbeat_runner,
        expansion_handoff_runner=fake_handoff_runner,
        sandbox_queue_runner=fake_queue_runner,
        queue_evaluation_runner=fake_evaluation_runner,
        registry_summary_runner=fake_summary_runner,
        blocked_executor_authoring_runner=fake_blocked_authoring_runner,
        authoring_audit_runner=fake_authoring_audit_runner,
        force=False,
    )

    assert calls["handoff"][0]["candidate_plan_path"] == Path(
        report["artifacts"]["candidate_plan_path"]
    )
    assert calls["queue"][0]["context_path"] == task1_target
    assert calls["queue"][0]["target_path"] == task1_target
    assert calls["queue"][0]["initial_submission_dir"] == tmp_path / "best"
    assert calls["queue"][0]["max_items"] == 3
    assert calls["evaluation"][0]["current_best_score"] == 120.0
    assert calls["summary"][0]["current_best_score"] == 120.0
    assert calls["blocked_authoring"][0]["queue_report_path"] == Path(
        tmp_path / "queue.json"
    )
    assert calls["blocked_authoring"][0]["context_path"] == task1_target
    assert calls["blocked_authoring"][0]["target_path"] == task1_target
    assert calls["authoring_audit"][0]["authoring_task_path"] == Path(
        tmp_path / "authoring_task.json"
    )
    assert calls["authoring_audit"][0]["skill_root"] == Path("skills")
    assert report["metrics"]["expansion_sandbox_executed_count"] == 1
    assert report["metrics"]["expansion_promotion_count"] == 0
    assert report["metrics"]["blocked_executor_authoring_task_count"] == 1
    assert report["metrics"]["blocked_executor_authoring_approved_count"] == 1
    assert report["artifacts"]["expansion_registry_summary_path"] == (
        str(tmp_path / "summary.json")
    )
    assert report["artifacts"]["blocked_executor_tasks_path"] == (
        str(tmp_path / "blocked_tasks.json")
    )
    completion_paths = report["artifacts"]["blocked_executor_completion_task_paths"]
    assert len(completion_paths) == 1
    completion = json.loads(Path(completion_paths[0]).read_text(encoding="utf-8"))
    assert completion["state_hint"] == "BOUNDED_EXECUTOR_COMPLETION_TASK_PREPARED"
    assert completion["algorithm_candidates_path"] == str(tmp_path / "candidates.json")
    assert completion["registry_path"] == str(tmp_path / "registry.json")
    assert completion["current_best_score"] == 120.0
    assert completion["next_command"] == "complete-bounded-executor-authoring-task"
    assert completion["trace_path"] == str(trace_path)
    assert f"--trace-path {trace_path}" in completion["command"]
    assert report["next_command"] == "agent-author-bounded-executor"
    expansion_events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("event_type") == "hermes_runtime_expansion_pipeline"
    ]
    assert expansion_events[-1]["payload"][
        "blocked_executor_authoring_task_count"
    ] == 1
    assert expansion_events[-1]["payload"][
        "blocked_executor_authoring_approved_count"
    ] == 1
    candidate_events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("event_type") == "hypothesis_lab_run"
        and "expansion_candidate" in json.loads(line).get("tags", [])
    ]
    assert candidate_events[-1]["payload"] == {
        "algorithm_candidate_id": "candidate_local_checkpoint_burgers_fno",
        "candidate_family": "official_checkpoint_family",
        "implementation_source": "hermes_runtime_expansion_pipeline",
        "outcome": "deprioritize",
        "score": 83.0,
        "score_delta": -37.0,
        "evidence_path": str(tmp_path / "checkpoint_evidence.json"),
        "promotion_path": str(tmp_path / "checkpoint_promotion.json"),
        "registry_event_path": str(tmp_path / "checkpoint_event.json"),
    }


def test_hermes_runtime_closed_loop_retests_refinement_on_legal_reference(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_val.hdf5"
    task1_reference = tmp_path / "task1_reference.hdf5"
    with h5py.File(task1_test, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))
    with h5py.File(task1_target, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    with h5py.File(task1_reference, "w") as handle:
        handle.create_dataset("tensor", data=np.ones((4, 200, 256), dtype=np.float32))
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    calls: dict[str, list[dict[str, object]]] = {
        "queue": [],
        "evaluation": [],
        "summary": [],
        "refinement": [],
        "strategy_proposal": [],
        "strategy_evidence": [],
        "strategy_rubric": [],
        "strategy_promotion": [],
        "evidence_accumulation": [],
    }

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 118.39,
                "submission_dir": str(best_dir),
            },
            "best_score": 118.39,
            "decision": "continue_next_heartbeat",
            "stop_reason": "max_batches_exhausted",
        }

    def fake_handoff_runner(**_kwargs):
        return {
            "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
            "handoff_report_path": str(tmp_path / "handoff.json"),
            "artifacts": {
                "algorithm_candidates_path": str(tmp_path / "candidates.json"),
                "expansion_report_path": str(tmp_path / "expansion.json"),
            },
            "metrics": {
                "algorithm_candidate_count": 1,
                "expansion_action_count": 1,
            },
            "next_command": "run-expansion-sandbox-queue",
        }

    def fake_queue_runner(**kwargs):
        calls["queue"].append(kwargs)
        index = len(calls["queue"])
        return {
            "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
            "queue_report_path": str(tmp_path / f"queue_{index}.json"),
            "metrics": {
                "executed_count": 1,
                "skipped_count": 0,
                "passed_count": 1,
                "failed_count": 0,
                "error_count": 0,
            },
            "next_command": "evaluate-expansion-sandbox-queue",
        }

    def fake_evaluation_runner(**kwargs):
        calls["evaluation"].append(kwargs)
        index = len(calls["evaluation"])
        return {
            "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_EVALUATED",
            "queue_evaluation_report_path": str(tmp_path / f"evaluation_{index}.json"),
            "artifacts": {"registry_path": str(tmp_path / f"registry_{index}.json")},
            "items": [
                {
                    "status": "evaluated",
                    "algorithm_candidate_id": (
                        "candidate_local_strategy_task1_early_subsegment_patch_v0"
                    ),
                    "candidate_family": "local_strategy_skill",
                    "decision": "refine" if index == 1 else "promote",
                    "score": 128.77 if index == 1 else 121.0,
                    "score_delta_vs_current_best": 10.38 if index == 1 else 2.61,
                    "evidence_path": str(tmp_path / f"evidence_{index}.json"),
                    "promotion_path": str(tmp_path / f"promotion_{index}.json"),
                    "registry_event_path": str(tmp_path / f"event_{index}.json"),
                }
            ],
            "metrics": {
                "evidence_count": 1,
                "promotion_count": 0 if index == 1 else 1,
                "registry_event_count": 1,
                "registry_candidate_count": 1,
            },
            "next_command": "summarize-literature-registry-for-hermes",
        }

    def fake_summary_runner(**kwargs):
        calls["summary"].append(kwargs)
        index = len(calls["summary"])
        if index == 1:
            return {
                "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
                "summary_path": str(tmp_path / "summary_1.json"),
                "metrics": {
                    "candidate_action_count": 1,
                    "learning_signal_count": 1,
                },
                "candidate_action_queue": [
                    {
                        "candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                        ),
                        "candidate_family": "local_strategy_skill",
                        "recommended_action": "refine_with_legal_reference",
                        "reason": "target_reference_signal_requires_legal_retest",
                        "best_score": 128.77,
                        "score_delta_vs_current_best": 10.38,
                        "candidate_gate_issues": [
                            "target_reference_local_validation_only"
                        ],
                        "executor": "run-hermes-batch-optimization-loop",
                    }
                ],
                "next_command": "refine-algorithm-candidate",
            }
        return {
            "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
            "summary_path": str(tmp_path / "summary_2.json"),
            "metrics": {
                "candidate_action_count": 1,
                "learning_signal_count": 0,
            },
            "candidate_action_queue": [
                {
                    "candidate_id": (
                        "candidate_local_strategy_task1_early_subsegment_patch_v0"
                    ),
                    "candidate_family": "local_strategy_skill",
                    "recommended_action": "record_strategy_proposal",
                    "reason": "candidate_promotion_recorded",
                    "candidate_promotion_path": str(tmp_path / "promotion_2.json"),
                    "next_command": (
                        "record-promoted-algorithm-strategy-proposal"
                    ),
                }
            ],
            "next_command": "record-promoted-algorithm-strategy-proposal",
        }

    def fake_refinement_runner(**kwargs):
        calls["refinement"].append(kwargs)
        return {
            "state_hint": "HERMES_ALGORITHM_CANDIDATE_REFINEMENT_PREPARED",
            "candidate_refinement_path": str(tmp_path / "candidate_refinement.json"),
            "algorithm_candidate_id": (
                "candidate_local_strategy_task1_early_subsegment_patch_v0"
            ),
            "candidate_family": "local_strategy_skill",
            "refinement_type": "legal_reference_retest",
            "candidate_generator": "hypothesis_lab_early_subsegment_patch_search",
            "learning_objective": "early_accuracy",
            "suggested_sandbox_args": {
                "reference_path_required": True,
                "reference_sample_limit": 64,
            },
            "next_command": "run-expansion-sandbox-queue",
        }

    def fail_blocked_authoring_runner(**_kwargs):
        raise AssertionError("blocked executor authoring should not run")

    def fail_authoring_audit_runner(**_kwargs):
        raise AssertionError("authoring audit should not run")

    def fake_strategy_proposal_runner(**kwargs):
        calls["strategy_proposal"].append(kwargs)
        return {
            "state_hint": "STRATEGY_PROPOSAL_RECORDED",
            "strategy_proposal_path": str(tmp_path / "strategy_proposal.json"),
            "strategy_id": "task1_early_subsegment_patch_v0",
            "next_command": "build-strategy-rubric",
        }

    def fake_strategy_evidence_runner(**kwargs):
        calls["strategy_evidence"].append(kwargs)
        return {
            "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
            "evidence_summary_path": str(tmp_path / "strategy_evidence.json"),
            "evaluation_window_count": 1,
            "mean_score_delta": 2.61,
            "next_command": "evaluate-strategy-promotion",
        }

    def fake_strategy_rubric_runner(**kwargs):
        calls["strategy_rubric"].append(kwargs)
        return {
            "state_hint": "STRATEGY_RUBRIC_BUILT",
            "experiment_rubric_path": str(tmp_path / "strategy_rubric.json"),
            "strategy_id": "task1_early_subsegment_patch_v0",
            "next_command": "evaluate-strategy-promotion",
        }

    def fake_strategy_promotion_runner(**kwargs):
        calls["strategy_promotion"].append(kwargs)
        return {
            "state_hint": "STRATEGY_PROMOTION_EVALUATED",
            "promotion_report_path": str(tmp_path / "strategy_promotion.json"),
            "decision": "reject",
            "promote_to_skill_candidate": False,
            "reason_codes": ["insufficient_evaluation_windows"],
            "next_command": "append-strategy-ledger",
        }

    def fake_evidence_accumulation_runner(**kwargs):
        calls["evidence_accumulation"].append(kwargs)
        return {
            "state_hint": "CANDIDATE_EVIDENCE_ACCUMULATION_PLANNED",
            "evidence_accumulation_plan_path": str(tmp_path / "accumulation_plan.json"),
            "decision": "accumulate_more_evidence",
            "current_window_count": 1,
            "required_window_count": 2,
            "missing_window_count": 1,
            "next_command": "run-hermes-runtime-closed-loop",
        }

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        task1_test_path=task1_test,
        task1_target_path=task1_target,
        task1_reference_path=task1_reference,
        heartbeat_runner=fake_heartbeat_runner,
        expansion_handoff_runner=fake_handoff_runner,
        sandbox_queue_runner=fake_queue_runner,
        queue_evaluation_runner=fake_evaluation_runner,
        registry_summary_runner=fake_summary_runner,
        refinement_runner=fake_refinement_runner,
        blocked_executor_authoring_runner=fail_blocked_authoring_runner,
        authoring_audit_runner=fail_authoring_audit_runner,
        strategy_proposal_runner=fake_strategy_proposal_runner,
        strategy_evidence_runner=fake_strategy_evidence_runner,
        strategy_rubric_runner=fake_strategy_rubric_runner,
        strategy_promotion_runner=fake_strategy_promotion_runner,
        evidence_accumulation_runner=fake_evidence_accumulation_runner,
        force=False,
    )

    assert len(calls["queue"]) == 2
    assert calls["refinement"][0]["registry_summary_path"] == Path(
        tmp_path / "summary_1.json"
    )
    assert calls["queue"][1]["reference_path"] == task1_reference
    assert calls["queue"][1]["reference_sample_limit"] == 64
    assert calls["queue"][1]["max_items"] == 1
    assert calls["queue"][1]["selected_algorithm_candidate_id"] == (
        "candidate_local_strategy_task1_early_subsegment_patch_v0"
    )
    assert calls["evaluation"][1]["queue_report_path"] == Path(tmp_path / "queue_2.json")
    assert calls["summary"][1]["registry_path"] == Path(tmp_path / "registry_2.json")
    assert report["artifacts"]["candidate_refinement_path"] == str(
        tmp_path / "candidate_refinement.json"
    )
    assert report["artifacts"]["refinement_sandbox_queue_report_path"] == str(
        tmp_path / "queue_2.json"
    )
    assert report["metrics"]["refinement_sandbox_executed_count"] == 1
    assert report["metrics"]["refinement_promotion_count"] == 1
    assert report["metrics"]["recorded_refinement_candidate_count"] == 1
    assert calls["strategy_proposal"][0]["candidate_promotion_path"] == Path(
        tmp_path / "promotion_2.json"
    )
    assert calls["strategy_evidence"][0]["candidate_promotion_path"] == Path(
        tmp_path / "promotion_2.json"
    )
    assert calls["strategy_rubric"][0]["proposal_path"] == Path(
        tmp_path / "strategy_proposal.json"
    )
    assert calls["strategy_promotion"][0]["evidence_summary_path"] == Path(
        tmp_path / "strategy_evidence.json"
    )
    assert calls["evidence_accumulation"][0]["strategy_promotion_report_path"] == Path(
        tmp_path / "strategy_promotion.json"
    )
    assert report["metrics"]["strategy_growth_bridge_executed"] is True
    assert report["metrics"]["evidence_accumulation_missing_window_count"] == 1
    assert report["artifacts"]["evidence_accumulation_plan_path"] == str(
        tmp_path / "accumulation_plan.json"
    )
    assert report["next_command"] == "run-hermes-runtime-closed-loop"


def test_hermes_runtime_closed_loop_consumes_evidence_accumulation_plan(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
            source_artifact_path="failed.json",
            tags=["hermes"],
        )
    initial_dir = tmp_path / "initial_submission"
    initial_dir.mkdir()
    task1_target = tmp_path / "task1_target.hdf5"
    task1_reference = tmp_path / "task1_reference.hdf5"
    data = np.zeros((4, 200, 256), dtype=np.float32)
    for path in [task1_target, task1_reference]:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("tensor", data=data)

    previous_promotion = tmp_path / "promotion_previous.json"
    previous_promotion.write_text("{}", encoding="utf-8")
    plan_path = tmp_path / "accumulation_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CANDIDATE_EVIDENCE_ACCUMULATION_PLANNED",
                "algorithm_candidate_id": (
                    "candidate_local_strategy_task1_early_subsegment_patch_v0"
                ),
                "source_candidate_promotion_paths": [str(previous_promotion)],
                "recommended_actions": [
                    {
                        "action": "run_legal_reference_retest",
                        "algorithm_candidate_id": (
                            "candidate_local_strategy_task1_early_subsegment_patch_v0"
                        ),
                        "suggested_sandbox_args": {
                            "reference_sample_limit": 128,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: dict[str, list[dict[str, object]]] = {
        "queue": [],
        "summary": [],
        "refinement": [],
        "strategy_evidence": [],
    }

    def fake_heartbeat_runner(**_kwargs):
        return {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(tmp_path / "heartbeat.json"),
            "artifacts": {
                "best_submission_dir": str(initial_dir),
                "best_judge_report_path": str(tmp_path / "judge.json"),
            },
            "best_candidate": {
                "candidate_id": "initial",
                "score": 118.39,
                "rules_passed": True,
                "submission_dir": str(initial_dir),
            },
            "best_score": 118.39,
            "decision": "continue_next_heartbeat",
        }

    def fake_handoff_runner(**_kwargs):
        return {
            "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
            "handoff_report_path": str(tmp_path / "handoff.json"),
            "artifacts": {
                "algorithm_candidates_path": str(tmp_path / "candidates.json"),
                "expansion_report_path": str(tmp_path / "expansion.json"),
            },
            "metrics": {
                "algorithm_candidate_count": 1,
                "expansion_action_count": 1,
            },
            "next_command": "run-expansion-sandbox-queue",
        }

    def fake_queue_runner(**kwargs):
        calls["queue"].append(kwargs)
        index = len(calls["queue"])
        return {
            "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_RAN",
            "queue_report_path": str(tmp_path / f"queue_{index}.json"),
            "metrics": {
                "executed_count": 1,
                "skipped_count": 0,
                "passed_count": 1,
                "failed_count": 0,
                "error_count": 0,
            },
            "next_command": "evaluate-expansion-sandbox-queue",
        }

    def fake_evaluation_runner(**_kwargs):
        index = len(calls["queue"])
        return {
            "state_hint": "HERMES_EXPANSION_SANDBOX_QUEUE_EVALUATED",
            "queue_evaluation_report_path": str(tmp_path / f"evaluation_{index}.json"),
            "artifacts": {"registry_path": str(tmp_path / f"registry_{index}.json")},
            "metrics": {
                "evidence_count": 1,
                "promotion_count": 0 if index == 1 else 1,
                "registry_event_count": 1,
                "registry_candidate_count": 1,
            },
            "next_command": "summarize-literature-registry-for-hermes",
        }

    def fake_summary_runner(**kwargs):
        calls["summary"].append(kwargs)
        index = len(calls["summary"])
        action = {
            "candidate_id": "candidate_local_strategy_task1_early_subsegment_patch_v0",
            "candidate_family": "local_strategy_skill",
            "next_command": "refine-algorithm-candidate",
        }
        if index == 1:
            action.update(
                {
                    "recommended_action": "refine_with_legal_reference",
                    "reason": "target_reference_signal_requires_legal_retest",
                    "executor": "run-hermes-batch-optimization-loop",
                }
            )
        else:
            action.update(
                {
                    "recommended_action": "record_strategy_proposal",
                    "reason": "candidate_promotion_recorded",
                    "candidate_promotion_path": str(tmp_path / "promotion_current.json"),
                }
            )
        return {
            "state_hint": "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT",
            "summary_path": str(tmp_path / f"summary_{index}.json"),
            "metrics": {
                "candidate_action_count": 1,
                "learning_signal_count": 0,
            },
            "candidate_action_queue": [action],
            "next_command": action["next_command"],
        }

    def fake_refinement_runner(**kwargs):
        calls["refinement"].append(kwargs)
        return {
            "state_hint": "HERMES_ALGORITHM_CANDIDATE_REFINEMENT_PREPARED",
            "candidate_refinement_path": str(tmp_path / "candidate_refinement.json"),
            "algorithm_candidate_id": (
                "candidate_local_strategy_task1_early_subsegment_patch_v0"
            ),
            "suggested_sandbox_args": {"reference_sample_limit": 64},
            "next_command": "run-expansion-sandbox-queue",
        }

    def fake_strategy_evidence_runner(**kwargs):
        calls["strategy_evidence"].append(kwargs)
        return {
            "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
            "evidence_summary_path": str(tmp_path / "strategy_evidence.json"),
            "evaluation_window_count": 2,
            "mean_score_delta": 2.0,
            "next_command": "evaluate-strategy-promotion",
        }

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        task1_test_path=task1_target,
        task1_target_path=task1_target,
        task1_reference_path=task1_reference,
        evidence_accumulation_plan_path=plan_path,
        heartbeat_runner=fake_heartbeat_runner,
        expansion_handoff_runner=fake_handoff_runner,
        sandbox_queue_runner=fake_queue_runner,
        queue_evaluation_runner=fake_evaluation_runner,
        registry_summary_runner=fake_summary_runner,
        refinement_runner=fake_refinement_runner,
        blocked_executor_authoring_runner=lambda **_kwargs: {
            "metrics": {},
            "artifacts": {},
            "next_command": None,
        },
        authoring_audit_runner=lambda **_kwargs: {},
        strategy_proposal_runner=lambda **_kwargs: {
            "strategy_proposal_path": str(tmp_path / "proposal.json"),
            "next_command": "build-strategy-rubric",
        },
        strategy_evidence_runner=fake_strategy_evidence_runner,
        strategy_rubric_runner=lambda **_kwargs: {
            "experiment_rubric_path": str(tmp_path / "rubric.json"),
            "next_command": "evaluate-strategy-promotion",
        },
        strategy_promotion_runner=lambda **_kwargs: {
            "promotion_report_path": str(tmp_path / "strategy_promotion.json"),
            "decision": "promote",
            "promote_to_skill_candidate": True,
            "reason_codes": ["all_rubric_checks_passed"],
            "next_command": "append-strategy-ledger",
        },
        evidence_accumulation_runner=lambda **_kwargs: {
            "missing_window_count": 0,
            "evidence_accumulation_plan_path": str(tmp_path / "unused.json"),
        },
        force=False,
    )

    assert calls["queue"][1]["reference_sample_limit"] == 128
    assert calls["strategy_evidence"][0]["candidate_promotion_path"] == Path(
        tmp_path / "promotion_current.json"
    )
    assert calls["strategy_evidence"][0]["extra_candidate_promotion_paths"] == [
        Path(previous_promotion)
    ]
    assert report["metrics"]["strategy_growth_evaluation_window_count"] == 2
    assert report["metrics"]["strategy_growth_promote_to_skill_candidate"] is True


def test_hermes_runtime_closed_loop_skips_blocked_authoring_without_executor_gap(
    tmp_path: Path,
) -> None:
    import h5py
    import numpy as np

    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for family in [
        "postprocess",
        "physics_residual_nudge",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    ]:
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{family}",
                "candidate_family": family,
                "outcome": "failed",
                "score_delta": -0.1,
            },
        )
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_val.hdf5"
    with h5py.File(task1_test, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 10, 256), dtype=np.float32))
    with h5py.File(task1_target, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 200, 256), dtype=np.float32))
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "best_score": 120.0,
            "decision": "continue_next_heartbeat",
            "stop_reason": "max_batches_exhausted",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_handoff_runner(**_kwargs):
        return {
            "handoff_report_path": str(tmp_path / "handoff.json"),
            "artifacts": {
                "algorithm_candidates_path": str(tmp_path / "candidates.json"),
                "expansion_report_path": str(tmp_path / "expansion.json"),
            },
            "metrics": {
                "algorithm_candidate_count": 2,
                "expansion_action_count": 2,
            },
            "next_command": "run-expansion-sandbox-queue",
        }

    def fake_queue_runner(**_kwargs):
        return {
            "queue_report_path": str(tmp_path / "queue.json"),
            "metrics": {"executed_count": 1, "skipped_count": 0},
            "next_command": "evaluate-expansion-sandbox-queue",
        }

    def fake_evaluation_runner(**_kwargs):
        return {
            "queue_evaluation_report_path": str(tmp_path / "evaluation.json"),
            "artifacts": {"registry_path": str(tmp_path / "registry.json")},
            "metrics": {"promotion_count": 1, "registry_event_count": 1},
            "next_command": "summarize-literature-registry-for-hermes",
        }

    def fake_summary_runner(**_kwargs):
        return {
            "summary_path": str(tmp_path / "summary.json"),
            "metrics": {"learning_signal_count": 0},
            "learning_signals": [],
            "next_command": "run-hermes-runtime-closed-loop",
        }

    def fail_blocked_authoring_runner(**_kwargs):
        raise AssertionError("blocked executor authoring should not run")

    def fail_authoring_audit_runner(**_kwargs):
        raise AssertionError("authoring audit should not run")

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_blend",
        max_batches=1,
        recent_window=10,
        failed_family_threshold=1,
        task1_test_path=task1_test,
        task1_target_path=task1_target,
        heartbeat_runner=fake_heartbeat_runner,
        expansion_handoff_runner=fake_handoff_runner,
        sandbox_queue_runner=fake_queue_runner,
        queue_evaluation_runner=fake_evaluation_runner,
        registry_summary_runner=fake_summary_runner,
        blocked_executor_authoring_runner=fail_blocked_authoring_runner,
        authoring_audit_runner=fail_authoring_audit_runner,
        force=False,
    )

    assert report["metrics"]["blocked_executor_authoring_task_count"] == 0
    assert report["next_command"] == "run-hermes-runtime-closed-loop"


def test_hermes_runtime_closed_loop_command_returns_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    calls = []

    def fake_runtime(**kwargs):
        calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {
                "runtime_loop_report_path": str(tmp_path / "runtime.json"),
                "trace_path": str(tmp_path / "trace.jsonl"),
                "trace_summary_path": str(tmp_path / "summary.json"),
                "hypothesis_kpi_report_path": str(tmp_path / "kpi.json"),
                "repair_policy_summary_path": str(tmp_path / "repair_policy.json"),
                "doom_loop_report_path": str(tmp_path / "doom.json"),
                "candidate_plan_path": str(tmp_path / "candidate_plan.json"),
                "heartbeat_report_path": str(tmp_path / "heartbeat.json"),
            },
            "metrics": {
                "guard_active": True,
                "heartbeat_best_score": 101.0,
                "heartbeat_decision": "continue_next_heartbeat",
                "repair_nonconvergence_error_count": 1,
                "repair_nonconvergence_errors": [
                    "cluster_saturation_consistency"
                ],
                "repair_convergence_error_count": 1,
                "repair_convergence_errors": [
                    "learning_constraint_smoke_gate",
                ],
            },
            "candidate_plan": {
                "repair_nonconvergence_errors": [
                    "cluster_saturation_consistency"
                ],
                "repair_convergence_errors": [
                    "learning_constraint_smoke_gate",
                ],
                "learning_constraints": {
                    "audit_repair_converged_error_counts": {
                        "learning_constraint_smoke_gate": 1,
                    },
                    "repair_policy_recommended_actions": [
                        {
                            "audit_error": "cluster_saturation_consistency",
                            "action": (
                                "switch_repair_strategy_or_reload_missing_evidence"
                            ),
                            "reason": "repair_nonconverged",
                        }
                    ],
                },
                "audit_repair_policies": [
                    {
                        "audit_error": "cluster_saturation_consistency",
                        "preferred_candidate_generators": [
                            "physics_residual_nudge"
                        ],
                        "repair_nonconverged": True,
                    }
                ],
            },
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(tmp_path / "runtime.json"),
                "output_path": str(tmp_path / "audit.json"),
                "require_learning_constraint_smoke": True,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    monkeypatch.setattr(
        runtime_commands,
        "run_hermes_runtime_closed_loop",
        fake_runtime,
    )

    payload = runtime_commands.run_hermes_runtime_closed_loop_command(
        output_dir=str(tmp_path / "runtime"),
        trace_path=str(tmp_path / "trace.jsonl"),
        initial_submission_dir=str(tmp_path / "initial"),
        candidate_generator="task1_early_refinement",
        candidate_generator_fallback="postprocess",
        max_batches=1,
        recent_window=3,
        repeated_family_threshold=2,
        failed_family_threshold=1,
        min_score_delta=0.05,
        task1_reference_path=str(tmp_path / "task1_reference.hdf5"),
        expansion_registry_summary_path=str(tmp_path / "registry_summary.json"),
        expansion_handoff_report_path=str(tmp_path / "handoff.json"),
        audit_repair_tasks_path=str(tmp_path / "audit_repair_tasks.json"),
        input_research_chronicle_path=str(tmp_path / "prior_chronicle.md"),
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_RUNTIME_CLOSED_LOOP_RAN"
    assert payload["metrics"]["guard_active"] is True
    assert payload["artifacts"]["trace_summary_json"].endswith("summary.json")
    assert payload["artifacts"]["hypothesis_kpi_report_json"].endswith("kpi.json")
    assert payload["artifacts"]["repair_policy_summary_json"].endswith(
        "repair_policy.json"
    )
    assert payload["artifacts"]["doom_loop_report_json"].endswith("doom.json")
    assert payload["artifacts"]["candidate_plan_json"].endswith(
        "candidate_plan.json"
    )
    assert payload["artifacts"]["heartbeat_report_json"].endswith("heartbeat.json")
    assert payload["artifacts"]["repair_nonconvergence_errors"] == [
        "cluster_saturation_consistency"
    ]
    assert payload["artifacts"]["repair_convergence_errors"] == [
        "learning_constraint_smoke_gate"
    ]
    assert payload["artifacts"]["audit_repair_converged_error_counts"] == {
        "learning_constraint_smoke_gate": 1
    }
    assert payload["artifacts"]["audit_repair_policies"] == [
        {
            "audit_error": "cluster_saturation_consistency",
            "preferred_candidate_generators": ["physics_residual_nudge"],
            "repair_nonconverged": True,
        }
    ]
    assert payload["artifacts"]["repair_policy_recommended_actions"] == [
        {
            "audit_error": "cluster_saturation_consistency",
            "action": "switch_repair_strategy_or_reload_missing_evidence",
            "reason": "repair_nonconverged",
        }
    ]
    assert payload["artifacts"]["heartbeat_report_path"].endswith("heartbeat.json")
    assert payload["artifacts"]["audit_command"] == {
        "command": "audit-hermes-runtime-closed-loop",
        "runtime_report_path": str(tmp_path / "runtime.json"),
        "output_path": str(tmp_path / "audit.json"),
        "require_learning_constraint_smoke": True,
    }
    assert payload["next_hint"] == "audit-hermes-runtime-closed-loop"
    assert calls[0]["recent_window"] == 3
    assert calls[0]["repeated_family_threshold"] == 2
    assert calls[0]["failed_family_threshold"] == 1
    assert calls[0]["min_score_delta"] == 0.05
    assert calls[0]["task1_reference_path"] == tmp_path / "task1_reference.hdf5"
    assert calls[0]["expansion_registry_summary_path"] == (
        tmp_path / "registry_summary.json"
    )
    assert calls[0]["expansion_handoff_report_path"] == tmp_path / "handoff.json"
    assert calls[0]["audit_repair_tasks_path"] == (
        tmp_path / "audit_repair_tasks.json"
    )
    assert calls[0]["input_research_chronicle_path"] == (
        tmp_path / "prior_chronicle.md"
    )


def test_hermes_runtime_audit_chain_runs_runtime_audit_iterations(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    runtime_calls = []
    audit_calls = []

    def fake_runtime(**kwargs):
        iteration = len(runtime_calls) + 1
        runtime_calls.append(kwargs)
        best_dir = tmp_path / f"best_{iteration}"
        best_dir.mkdir()
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0 + iteration},
            "heartbeat": {
                "best_candidate": {
                    "candidate_id": f"best_{iteration}",
                    "submission_dir": str(best_dir),
                }
            },
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_learning_constraint_smoke": True,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        audit_calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "passed",
            "summary": {"check_count": 11, "failed_check_count": 0},
            "errors": [],
            "next_command": "use-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    infrastructure_path = tmp_path / "passed_infrastructure_audit.json"
    infrastructure_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "session_id": "session_infra_001",
                "infrastructure": {
                    "decision": "passed",
                    "check_count": 13,
                    "failed_check_count": 0,
                },
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=2,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        audit_repair_tasks_path=tmp_path / "audit_repair_tasks.json",
        preflight_infrastructure_audit_path=infrastructure_path,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    assert report["state_hint"] == "HERMES_RUNTIME_AUDIT_CHAIN_RAN"
    assert report["decision"] == "passed"
    assert report["metrics"]["completed_iteration_count"] == 2
    assert len(runtime_calls) == 2
    assert len(audit_calls) == 2
    assert runtime_calls[0]["initial_submission_dir"] == initial_dir
    assert runtime_calls[0]["audit_repair_tasks_path"] == (
        tmp_path / "audit_repair_tasks.json"
    )
    assert runtime_calls[1]["initial_submission_dir"] == tmp_path / "best_1"
    assert audit_calls[0]["require_learning_constraint_smoke"] is True
    assert report["iterations"][1]["initial_submission_dir"] == str(tmp_path / "best_1")
    assert report["next_command"] == "run-hermes-runtime-audit-chain"
    assert report["artifacts"]["preflight_infrastructure_audit_path"] == str(
        infrastructure_path
    )
    assert report["metrics"]["preflight_infrastructure_audit_passed"] is True
    assert report["metrics"]["preflight_infrastructure_check_count"] == 13
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["event_type"] == "hermes_runtime_audit_chain"
    assert events[-1]["source_artifact_path"] == report["chain_report_path"]
    assert events[-1]["tags"] == ["hermes", "runtime_audit_chain"]
    assert events[-1]["payload"] == {
        "algorithm_candidate_id": "hermes_runtime_audit_chain",
        "candidate_family": "hermes_runtime_audit_chain",
        "outcome": "passed",
        "requested_iteration_count": 2,
        "completed_iteration_count": 2,
        "failed_audit_count": 0,
        "last_heartbeat_best_score": 102.0,
        "score_delta": 1.0,
        "candidate_space_exhausted_count": 0,
        "audit_errors": [],
        "repair_convergence": {
            "decision": "not_applicable",
            "input_repair_errors": [],
            "repeated_audit_errors": [],
        },
        "audit_repair_task_count": 0,
        "chain_learning_signals": [],
        "chain_report_path": report["chain_report_path"],
        "preflight_infrastructure_audit_path": str(infrastructure_path),
        "preflight_infrastructure_check_count": 13,
        "preflight_infrastructure_session_id": "session_infra_001",
    }
    assert report["trace_event_id"] == events[-1]["event_id"]


def test_hermes_runtime_audit_chain_auto_repairs_supported_runtime_task(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    runtime_calls = []
    audit_calls = []

    def fake_runtime(**kwargs):
        runtime_calls.append(kwargs)
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            json.dumps({"state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN"}),
            encoding="utf-8",
        )
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 101.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        audit_calls.append(kwargs)
        if len(audit_calls) == 1:
            return {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
                "audit_report_path": str(kwargs["output_path"]),
                "decision": "failed",
                "summary": {"check_count": 14, "failed_check_count": 1},
                "errors": ["repair_convergence_consistency"],
                "next_command": "run-hermes-runtime-closed-loop",
            }
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "passed",
            "summary": {"check_count": 14, "failed_check_count": 0},
            "errors": [],
            "next_command": "run-hermes-runtime-audit-chain",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        auto_repair=True,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    assert report["decision"] == "repaired"
    assert report["auto_repair"]["decision"] == "passed"
    assert report["repair_convergence"] == {
        "decision": "converged",
        "input_repair_errors": ["repair_convergence_consistency"],
        "repeated_audit_errors": [],
    }
    assert "audit_repair_convergence" in report["chain_learning_signals"]
    assert report["metrics"]["auto_repair_attempted"] is True
    assert report["artifacts"]["auto_repair_report_path"].endswith(
        "audit_repair_task_run_report.json"
    )
    assert len(runtime_calls) == 2
    assert len(audit_calls) == 2
    assert runtime_calls[1]["audit_repair_tasks_path"] == Path(
        report["artifacts"]["audit_repair_tasks_path"]
    )
    assert runtime_calls[1]["trace_path"] == tmp_path / "trace.jsonl"
    assert runtime_calls[1]["initial_submission_dir"] == initial_dir
    assert report["next_command"] == "run-hermes-runtime-audit-chain"


def test_hermes_runtime_audit_chain_auto_repairs_memory_bridge_smoke_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import hermes_runtime_audit_chain as audit_chain

    runtime_calls = []
    audit_calls = []
    smoke_calls = []

    def fake_runtime(**kwargs):
        runtime_calls.append(kwargs)
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            json.dumps({"state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN"}),
            encoding="utf-8",
        )
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 101.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_runtime_memory_bridge_smoke": True,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        audit_calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 20, "failed_check_count": 1},
            "errors": ["runtime_memory_bridge_smoke_gate"],
            "next_command": "run-hermes-runtime-memory-bridge-smoke",
        }

    def fake_smoke(**kwargs):
        smoke_calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
            "decision": "passed",
            "artifacts": {
                "smoke_report_path": str(tmp_path / "smoke.json"),
                "runtime_loop_report_path": str(tmp_path / "runtime.json"),
                "runtime_closed_loop_audit_path": str(tmp_path / "audit.json"),
            },
            "metrics": {
                "runtime_audit_failed_check_count": 0,
                "repair_policy_recommended_action_count": 1,
                "batch_memory_constraint_action_count": 1,
            },
            "memory_bridge": {"passed": True},
            "next_command": "use-hermes-runtime-closed-loop",
        }

    monkeypatch.setattr(
        audit_chain,
        "run_hermes_runtime_memory_bridge_smoke",
        fake_smoke,
    )

    initial_dir = tmp_path / "initial"
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_target.hdf5"
    initial_dir.mkdir()
    task1_test.write_text("context", encoding="utf-8")
    task1_target.write_text("target", encoding="utf-8")

    report = audit_chain.run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=task1_test,
        task1_target_path=task1_target,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        auto_repair=True,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    assert report["decision"] == "repaired"
    assert report["auto_repair"]["decision"] == "passed"
    assert report["auto_repair"]["metrics"]["memory_bridge_passed"] is True
    assert report["repair_convergence"] == {
        "decision": "converged",
        "input_repair_errors": ["runtime_memory_bridge_smoke_gate"],
        "repeated_audit_errors": [],
    }
    assert report["metrics"]["auto_repair_attempted"] is True
    assert report["metrics"]["auto_repair_memory_bridge_passed"] is True
    assert report["metrics"]["auto_repair_smoke_executed"] is True
    assert report["metrics"]["auto_repair_repair_policy_recommended_action_count"] == 1
    assert report["metrics"]["auto_repair_batch_memory_constraint_action_count"] == 1
    assert "audit_repair_auto_repaired" in report["chain_learning_signals"]
    assert len(runtime_calls) == 1
    assert len(audit_calls) == 1
    assert len(smoke_calls) == 1
    assert smoke_calls[0]["audit_repair_tasks_path"] == Path(
        report["artifacts"]["audit_repair_tasks_path"]
    )
    assert smoke_calls[0]["output_dir"] == tmp_path / "chain" / "runtime_memory_bridge_smoke"
    assert smoke_calls[0]["initial_submission_dir"] == initial_dir
    assert smoke_calls[0]["task1_test_path"] == task1_test
    assert smoke_calls[0]["task1_target_path"] == task1_target
    assert report["artifacts"]["auto_repair_report_path"].endswith(
        "audit_repair_task_run_report.json"
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["event_type"] == "hermes_runtime_audit_chain"
    assert events[-1]["payload"]["auto_repair_decision"] == "passed"
    assert events[-1]["payload"]["auto_repair_metrics"] == {
        "smoke_executed": True,
        "memory_bridge_passed": True,
        "repair_policy_recommended_action_count": 1,
        "batch_memory_constraint_action_count": 1,
    }
    assert events[-1]["payload"]["auto_repair_artifacts"] == {
        "repair_task_run_report_path": report["artifacts"][
            "auto_repair_report_path"
        ],
        "smoke_report_path": str(tmp_path / "smoke.json"),
        "runtime_loop_report_path": str(tmp_path / "runtime.json"),
        "audit_report_path": str(tmp_path / "audit.json"),
    }


def test_hermes_runtime_audit_chain_rejects_failed_acceptance_preflight(
    tmp_path: Path,
) -> None:
    import pytest

    from pde_agent.core.hermes_runtime_audit_chain import (
        HermesRuntimeAuditChainError,
        run_hermes_runtime_audit_chain,
    )

    preflight_path = tmp_path / "failed_acceptance.json"
    preflight_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN",
                "decision": "failed",
                "checks": {
                    "auto_repair_passed": True,
                    "planner_internalized": False,
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("runtime should not run after failed preflight")

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    with pytest.raises(
        HermesRuntimeAuditChainError,
        match="preflight acceptance smoke failed",
    ):
        run_hermes_runtime_audit_chain(
            output_dir=tmp_path / "chain",
            trace_path=tmp_path / "trace.jsonl",
            initial_submission_dir=initial_dir,
            candidate_generator="postprocess",
            candidate_generator_fallback="auto",
            iterations=1,
            max_batches=1,
            recent_window=50,
            repeated_family_threshold=3,
            failed_family_threshold=2,
            min_score_delta=0.1,
            max_rounds_per_batch=1,
            batch_deadline_seconds=60.0,
            submit_score_threshold=130.0,
            learning_objective="score",
            task1_test_path=None,
            task1_target_path=None,
            task1_reference_path=None,
            task2_test_path=None,
            task2_target_path=None,
            timing_profile=None,
            timing_factor=None,
            preflight_acceptance_smoke_path=preflight_path,
            runtime_runner=fail_if_called,
            force=False,
        )


def test_hermes_runtime_audit_chain_rejects_failed_infrastructure_preflight(
    tmp_path: Path,
) -> None:
    import pytest

    from pde_agent.core.hermes_runtime_audit_chain import (
        HermesRuntimeAuditChainError,
        run_hermes_runtime_audit_chain,
    )

    infrastructure_path = tmp_path / "failed_infrastructure_audit.json"
    infrastructure_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "session_id": "session_failed",
                "infrastructure": {
                    "decision": "failed",
                    "check_count": 13,
                    "failed_check_count": 1,
                },
                "errors": ["allowed_command_sequence"],
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("runtime should not run after infrastructure preflight failure")

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    with pytest.raises(
        HermesRuntimeAuditChainError,
        match="preflight infrastructure audit failed",
    ):
        run_hermes_runtime_audit_chain(
            output_dir=tmp_path / "chain",
            trace_path=tmp_path / "trace.jsonl",
            initial_submission_dir=initial_dir,
            candidate_generator="postprocess",
            candidate_generator_fallback="auto",
            iterations=1,
            max_batches=1,
            recent_window=50,
            repeated_family_threshold=3,
            failed_family_threshold=2,
            min_score_delta=0.1,
            max_rounds_per_batch=1,
            batch_deadline_seconds=60.0,
            submit_score_threshold=130.0,
            learning_objective="score",
            task1_test_path=None,
            task1_target_path=None,
            task1_reference_path=None,
            task2_test_path=None,
            task2_target_path=None,
            timing_profile=None,
            timing_factor=None,
            preflight_infrastructure_audit_path=infrastructure_path,
            runtime_runner=fail_if_called,
            force=False,
        )


def test_hermes_runtime_audit_chain_marks_auto_repair_nonconverged(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    audit_calls = []

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            json.dumps({"state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN"}),
            encoding="utf-8",
        )
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 101.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        audit_calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 14, "failed_check_count": 1},
            "errors": ["repair_convergence_consistency"],
            "next_command": "run-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        auto_repair=True,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    assert len(audit_calls) == 2
    assert report["decision"] == "failed"
    assert report["auto_repair"]["decision"] == "failed"
    assert report["repair_convergence"] == {
        "decision": "nonconverged",
        "input_repair_errors": ["repair_convergence_consistency"],
        "repeated_audit_errors": ["repair_convergence_consistency"],
    }
    assert report["metrics"]["repair_nonconverged_error_count"] == 1
    assert "audit_repair_nonconvergence" in report["chain_learning_signals"]
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["payload"]["repair_convergence"]["decision"] == "nonconverged"
    assert "audit_repair_nonconvergence" in events[-1]["payload"][
        "chain_learning_signals"
    ]


def test_hermes_runtime_audit_chain_stops_on_failed_audit(tmp_path: Path) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_learning_constraint_smoke": True,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 11, "failed_check_count": 1},
            "errors": ["learning_constraint_enforcement"],
            "next_command": "run-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=3,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    assert report["decision"] == "failed"
    assert report["metrics"]["completed_iteration_count"] == 1
    assert report["metrics"]["failed_audit_count"] == 1
    assert report["iterations"][0]["audit_decision"] == "failed"
    assert report["iterations"][0]["audit_errors"] == [
        "learning_constraint_enforcement"
    ]
    assert report["metrics"]["audit_repair_task_count"] == 1
    assert report["artifacts"]["audit_repair_tasks_path"].endswith(
        "audit_repair_tasks.json"
    )
    repair_tasks = json.loads(
        Path(report["artifacts"]["audit_repair_tasks_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert repair_tasks["state_hint"] == "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED"
    assert repair_tasks["repair_tasks"] == [
        {
            "task_id": "repair_learning_constraint_enforcement",
            "audit_error": "learning_constraint_enforcement",
            "priority": "high",
            "repair_scope": "candidate_plan_and_heartbeat_schedule",
            "recommended_action": (
                "rebuild_candidate_plan_or_adjust_heartbeat_schedule_to_respect_learning_constraints"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        }
    ]
    assert report["next_command"] == "run-hermes-runtime-closed-loop"
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["event_type"] == "hermes_runtime_audit_chain"
    assert events[-1]["payload"]["outcome"] == "failed"
    assert events[-1]["payload"]["failed_audit_count"] == 1
    assert events[-1]["payload"]["audit_errors"] == [
        "learning_constraint_enforcement"
    ]
    assert events[-1]["payload"]["audit_repair_task_count"] == 1
    assert events[-1]["payload"]["audit_repair_tasks_path"] == report["artifacts"][
        "audit_repair_tasks_path"
    ]
    assert report["trace_event_id"] == events[-1]["event_id"]


def test_hermes_runtime_audit_chain_marks_repair_nonconvergence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    repair_path = tmp_path / "input_audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_cluster_saturation_consistency",
                        "audit_error": "cluster_saturation_consistency",
                        "priority": "high",
                        "repair_scope": "candidate_cluster_saturation_memory",
                        "recommended_action": (
                            "rebuild_candidate_plan_from_latest_expansion_handoff_and_kpi_saturation_signals"
                        ),
                        "next_command": "run-hermes-runtime-closed-loop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 12, "failed_check_count": 1},
            "errors": ["cluster_saturation_consistency"],
            "next_command": "run-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        audit_repair_tasks_path=repair_path,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    assert report["repair_convergence"] == {
        "decision": "nonconverged",
        "input_repair_errors": ["cluster_saturation_consistency"],
        "repeated_audit_errors": ["cluster_saturation_consistency"],
    }
    assert report["metrics"]["repair_nonconverged_error_count"] == 1
    assert "audit_repair_nonconvergence" in report["chain_learning_signals"]
    event = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert event["payload"]["repair_convergence"]["decision"] == "nonconverged"
    assert "audit_repair_nonconvergence" in event["payload"][
        "chain_learning_signals"
    ]


def test_hermes_runtime_audit_chain_prepares_repair_nonconvergence_consistency_task(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 13, "failed_check_count": 1},
            "errors": ["repair_nonconvergence_consistency"],
            "next_command": "run-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    repair_tasks = json.loads(
        Path(report["artifacts"]["audit_repair_tasks_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert repair_tasks["repair_tasks"] == [
        {
            "task_id": "repair_repair_nonconvergence_consistency",
            "audit_error": "repair_nonconvergence_consistency",
            "priority": "high",
            "repair_scope": "runtime_repair_nonconvergence_signal_bridge",
            "recommended_action": (
                "rebuild_kpi_candidate_plan_runtime_trace_with_consistent_repair_nonconvergence_signals"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        }
    ]


def test_hermes_runtime_audit_chain_prepares_repair_convergence_consistency_task(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 14, "failed_check_count": 1},
            "errors": ["repair_convergence_consistency"],
            "next_command": "run-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    repair_tasks = json.loads(
        Path(report["artifacts"]["audit_repair_tasks_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert repair_tasks["repair_tasks"] == [
        {
            "task_id": "repair_repair_convergence_consistency",
            "audit_error": "repair_convergence_consistency",
            "priority": "high",
            "repair_scope": "runtime_repair_convergence_signal_bridge",
            "recommended_action": (
                "rebuild_kpi_candidate_plan_runtime_trace_with_consistent_repair_convergence_signals"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        }
    ]


def test_hermes_runtime_audit_chain_prepares_learning_smoke_repair_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    monkeypatch.chdir(tmp_path)

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_learning_constraint_smoke": True,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 13, "failed_check_count": 1},
            "errors": ["learning_constraint_smoke_gate"],
            "next_command": "run-hypothesis-learning-constraint-smoke",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    repair_tasks = json.loads(
        Path(report["artifacts"]["audit_repair_tasks_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert repair_tasks["next_command"] == "run-hypothesis-learning-constraint-smoke"
    assert report["next_command"] == "run-hypothesis-learning-constraint-smoke"
    assert repair_tasks["repair_tasks"] == [
        {
            "task_id": "repair_learning_constraint_smoke_gate",
            "audit_error": "learning_constraint_smoke_gate",
            "priority": "high",
            "repair_scope": "learning_constraint_smoke_artifact_bridge",
            "recommended_action": (
                "rerun_hypothesis_learning_constraint_smoke_and_attach_passed_report_before_runtime_audit"
            ),
            "next_command": "run-hypothesis-learning-constraint-smoke",
            "suggested_command_args": {
                "trace_path": str(tmp_path / "trace.jsonl"),
                "output_dir": str(tmp_path / "chain" / "learning_constraint_smoke"),
                "candidate_generator": "postprocess",
                "checkpoint_paths": [],
                "min_viable_candidate_count": 999,
                "force": True,
            },
            "missing_inputs": ["local_strategy_library_path"],
        }
    ]


def test_hermes_runtime_audit_chain_prepares_runtime_memory_bridge_smoke_repair_task(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    repair_path = tmp_path / "input_audit_repair_tasks.json"
    repair_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
                "repair_tasks": [
                    {
                        "task_id": "repair_repair_convergence_consistency",
                        "audit_error": "repair_convergence_consistency",
                        "next_command": "run-hermes-runtime-closed-loop",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_runtime_memory_bridge_smoke": True,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 20, "failed_check_count": 1},
            "errors": ["runtime_memory_bridge_smoke_gate"],
            "next_command": "run-hermes-runtime-memory-bridge-smoke",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    task1_test = tmp_path / "task1_test.hdf5"
    task1_target = tmp_path / "task1_target.hdf5"
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=task1_test,
        task1_target_path=task1_target,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        audit_repair_tasks_path=repair_path,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    repair_tasks = json.loads(
        Path(report["artifacts"]["audit_repair_tasks_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert repair_tasks["next_command"] == "run-hermes-runtime-memory-bridge-smoke"
    assert report["next_command"] == "run-hermes-runtime-memory-bridge-smoke"
    assert repair_tasks["repair_tasks"] == [
        {
            "task_id": "repair_runtime_memory_bridge_smoke_gate",
            "audit_error": "runtime_memory_bridge_smoke_gate",
            "priority": "high",
            "repair_scope": "runtime_memory_bridge_smoke_artifact_bridge",
            "recommended_action": (
                "rerun_hermes_runtime_memory_bridge_smoke_and_attach_passed_report_before_runtime_audit"
            ),
            "next_command": "run-hermes-runtime-memory-bridge-smoke",
            "suggested_command_args": {
                "trace_path": str(tmp_path / "trace.jsonl"),
                "output_dir": str(tmp_path / "chain" / "runtime_memory_bridge_smoke"),
                "initial_submission_dir": str(initial_dir),
                "task1_test_path": str(task1_test),
                "task1_target_path": str(task1_target),
                "audit_repair_tasks_path": str(repair_path),
                "candidate_generator": "postprocess",
                "force": True,
            },
            "missing_inputs": [],
        }
    ]


def test_hermes_runtime_audit_chain_fills_learning_smoke_library_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    monkeypatch.chdir(tmp_path)
    chain_dir = tmp_path / "chain"
    library_path = chain_dir / "local_strategy_library.json"
    library_path.parent.mkdir(parents=True)
    library_path.write_text(
        json.dumps({"state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT"}),
        encoding="utf-8",
    )

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_learning_constraint_smoke": True,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 13, "failed_check_count": 1},
            "errors": ["learning_constraint_smoke_gate"],
            "next_command": "run-hypothesis-learning-constraint-smoke",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=chain_dir,
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    repair_tasks = json.loads(
        Path(report["artifacts"]["audit_repair_tasks_path"]).read_text(
            encoding="utf-8"
        )
    )
    repair_task = repair_tasks["repair_tasks"][0]
    assert repair_task["suggested_command_args"]["local_strategy_library_path"] == str(
        library_path
    )
    assert repair_task["missing_inputs"] == []


def test_hermes_runtime_audit_chain_records_chain_learning_signals(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    runtime_calls = []

    def fake_runtime(**kwargs):
        runtime_calls.append(kwargs)
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {
                "heartbeat_best_score": 100.0,
                "candidate_space_exhausted": True,
            },
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_learning_constraint_smoke": True,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "passed",
            "summary": {"check_count": 11, "failed_check_count": 0},
            "errors": [],
            "next_command": "use-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=2,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    assert report["metrics"]["score_delta"] == 0.0
    assert report["metrics"]["candidate_space_exhausted_count"] == 2
    assert report["chain_learning_signals"] == [
        "candidate_space_exhaustion",
        "score_stagnation",
    ]
    event = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert event["payload"]["chain_learning_signals"] == [
        "candidate_space_exhaustion",
        "score_stagnation",
    ]


def test_hermes_runtime_audit_chain_inherits_root_learning_smoke(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    smoke_path = (
        tmp_path
        / "chain"
        / "learning_constraint_smoke"
        / "hypothesis_learning_constraint_smoke.json"
    )
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN",
                "decision": "passed",
                "trace_path": str(tmp_path / "trace.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    audit_calls = []

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        runtime_path.parent.mkdir(parents=True)
        report = {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
                "require_learning_constraint_smoke": False,
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }
        runtime_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_audit(**kwargs):
        audit_calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "passed",
            "summary": {"check_count": 11, "failed_check_count": 0},
            "errors": [],
            "next_command": "use-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    runtime_payload = json.loads(
        Path(report["iterations"][0]["runtime_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert runtime_payload["artifacts"]["learning_constraint_smoke_report_path"] == str(
        smoke_path
    )
    assert runtime_payload["audit_command"]["require_learning_constraint_smoke"] is True
    assert audit_calls[0]["require_learning_constraint_smoke"] is True


def test_hermes_runtime_audit_chain_inherits_root_runtime_memory_bridge_smoke(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )

    smoke_path = (
        tmp_path
        / "chain"
        / "runtime_memory_bridge_smoke"
        / "hermes_runtime_memory_bridge_smoke.json"
    )
    smoke_path.parent.mkdir(parents=True)
    audit_calls = []

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        runtime_path.parent.mkdir(parents=True)
        smoke_path.write_text(
            json.dumps(
                {
                    "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
                    "decision": "passed",
                    "artifacts": {"runtime_loop_report_path": str(runtime_path)},
                    "memory_bridge": {"passed": True},
                }
            ),
            encoding="utf-8",
        )
        report = {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }
        runtime_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_audit(**kwargs):
        audit_calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "passed",
            "summary": {"check_count": 20, "failed_check_count": 0},
            "errors": [],
            "next_command": "use-hermes-runtime-closed-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )

    runtime_payload = json.loads(
        Path(report["iterations"][0]["runtime_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert runtime_payload["artifacts"][
        "runtime_memory_bridge_smoke_report_path"
    ] == str(smoke_path)
    assert runtime_payload["audit_command"][
        "require_runtime_memory_bridge_smoke"
    ] is True
    assert audit_calls[0]["require_runtime_memory_bridge_smoke"] is True


def test_hermes_runtime_audit_chain_command_returns_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    calls = []

    def fake_chain(**kwargs):
        calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
            "chain_report_path": str(kwargs["output_dir"] / "chain.json"),
            "artifacts": {
                "audit_repair_tasks_path": str(
                    kwargs["output_dir"] / "audit_repair_tasks.json"
                ),
                "preflight_infrastructure_audit_path": str(
                    kwargs["preflight_infrastructure_audit_path"]
                ),
            },
            "trace_event_id": "event_chain_001",
            "decision": "passed",
            "iterations": [{"iteration_index": 1}],
            "metrics": {
                "requested_iteration_count": kwargs["iterations"],
                "completed_iteration_count": 1,
                "failed_audit_count": 0,
                "audit_repair_task_count": 0,
                "auto_repair_attempted": kwargs["auto_repair"],
                "preflight_infrastructure_audit_passed": (
                    kwargs["preflight_infrastructure_audit_path"] is not None
                ),
                "last_heartbeat_best_score": 101.0,
            },
            "repair_convergence": {
                "decision": "nonconverged",
                "input_repair_errors": ["cluster_saturation_consistency"],
                "repeated_audit_errors": ["cluster_saturation_consistency"],
            },
            "next_command": "run-hermes-runtime-audit-chain",
        }

    monkeypatch.setattr(
        runtime_commands,
        "run_hermes_runtime_audit_chain",
        fake_chain,
    )

    payload = runtime_commands.run_hermes_runtime_audit_chain_command(
        output_dir=str(tmp_path / "chain"),
        trace_path=str(tmp_path / "trace.jsonl"),
        initial_submission_dir=str(tmp_path / "initial"),
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=2,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=str(tmp_path / "task1_reference.hdf5"),
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        audit_repair_tasks_path=str(tmp_path / "audit_repair_tasks.json"),
        preflight_acceptance_smoke_path=str(tmp_path / "acceptance.json"),
        preflight_infrastructure_audit_path=str(tmp_path / "infrastructure.json"),
        auto_repair=True,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_RUNTIME_AUDIT_CHAIN_RAN"
    assert payload["artifacts"]["runtime_audit_chain_report_json"].endswith(
        "chain.json"
    )
    assert payload["artifacts"]["chain_report_path"].endswith("chain.json")
    assert payload["artifacts"]["audit_repair_tasks_json"].endswith(
        "audit_repair_tasks.json"
    )
    assert payload["artifacts"]["audit_repair_tasks_path"].endswith(
        "audit_repair_tasks.json"
    )
    assert calls[0]["task1_reference_path"] == tmp_path / "task1_reference.hdf5"
    assert calls[0]["audit_repair_tasks_path"] == (
        tmp_path / "audit_repair_tasks.json"
    )
    assert calls[0]["preflight_acceptance_smoke_path"] == tmp_path / "acceptance.json"
    assert calls[0]["preflight_infrastructure_audit_path"] == (
        tmp_path / "infrastructure.json"
    )
    assert calls[0]["auto_repair"] is True
    assert payload["artifacts"]["trace_event_id"] == "event_chain_001"
    assert payload["artifacts"]["hypothesis_lab_trace_event"] == "event_chain_001"
    assert payload["artifacts"]["repair_convergence"] == {
        "decision": "nonconverged",
        "input_repair_errors": ["cluster_saturation_consistency"],
        "repeated_audit_errors": ["cluster_saturation_consistency"],
    }
    assert payload["metrics"]["completed_iteration_count"] == 1
    assert payload["metrics"]["auto_repair_attempted"] is True
    assert payload["metrics"]["preflight_infrastructure_audit_passed"] is True
    assert payload["next_hint"] == "run-hermes-runtime-audit-chain"


def test_hermes_runtime_audit_chain_cli_passes_auto_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pde_agent.cli as cli_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    calls = []

    def fake_command(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
            "artifacts": {
                "runtime_audit_chain_report_json": str(tmp_path / "chain.json"),
                "auto_repair_report_json": str(tmp_path / "repair.json"),
            },
            "metrics": {"auto_repair_attempted": kwargs["auto_repair"]},
            "next_hint": "run-hermes-runtime-audit-chain",
        }

    monkeypatch.setattr(
        cli_module,
        "run_hermes_runtime_audit_chain_command",
        fake_command,
    )

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-runtime-audit-chain",
            "--output-dir",
            str(tmp_path / "chain"),
            "--trace-path",
            str(tmp_path / "trace.jsonl"),
            "--initial-submission-dir",
            str(tmp_path / "initial"),
            "--preflight-acceptance-smoke-path",
            str(tmp_path / "acceptance.json"),
            "--auto-repair",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert calls[0]["preflight_acceptance_smoke_path"] == str(
        tmp_path / "acceptance.json"
    )
    assert calls[0]["auto_repair"] is True
    assert payload["metrics"]["auto_repair_attempted"] is True
    assert payload["artifacts"]["auto_repair_report_json"].endswith("repair.json")


def test_hermes_runtime_audit_chain_cli_passes_infrastructure_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pde_agent.cli as cli_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    calls = []

    def fake_command(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
            "artifacts": {
                "runtime_audit_chain_report_json": str(tmp_path / "chain.json"),
                "preflight_infrastructure_audit_path": kwargs[
                    "preflight_infrastructure_audit_path"
                ],
            },
            "metrics": {"preflight_infrastructure_audit_passed": True},
            "next_hint": "run-hermes-runtime-audit-chain",
        }

    monkeypatch.setattr(
        cli_module,
        "run_hermes_runtime_audit_chain_command",
        fake_command,
    )

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-runtime-audit-chain",
            "--output-dir",
            str(tmp_path / "chain"),
            "--trace-path",
            str(tmp_path / "trace.jsonl"),
            "--initial-submission-dir",
            str(tmp_path / "initial"),
            "--preflight-infrastructure-audit-path",
            str(tmp_path / "infrastructure_audit.json"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert calls[0]["preflight_infrastructure_audit_path"] == str(
        tmp_path / "infrastructure_audit.json"
    )
    assert payload["metrics"]["preflight_infrastructure_audit_passed"] is True


def test_hermes_runtime_closed_loop_bootstraps_empty_trace_and_passes_heartbeat_options(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    heartbeat_calls = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_bootstrap",
                "family": kwargs["candidate_generator"],
                "score": 99.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 99.0,
            "best_objective": kwargs["learning_objective"],
            "best_objective_score": 99.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    trace_path = tmp_path / "new_trace.jsonl"
    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        max_batches=2,
        max_rounds_per_batch=7,
        batch_deadline_seconds=42.0,
        submit_score_threshold=999.0,
        learning_objective="early_accuracy",
        task1_test_path=tmp_path / "task1_test.hdf5",
        task1_target_path=tmp_path / "task1_target.hdf5",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=0.05,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert trace_path.exists()
    assert heartbeat_calls[0]["max_batches"] == 2
    assert heartbeat_calls[0]["max_rounds_per_batch"] == 7
    assert heartbeat_calls[0]["batch_deadline_seconds"] == 42.0
    assert heartbeat_calls[0]["submit_score_threshold"] == 999.0
    assert heartbeat_calls[0]["learning_objective"] == "early_accuracy"
    assert heartbeat_calls[0]["task1_test_path"] == tmp_path / "task1_test.hdf5"
    assert heartbeat_calls[0]["task1_target_path"] == tmp_path / "task1_target.hdf5"
    assert heartbeat_calls[0]["timing_profile"] == "mac_mini_m4_local_dev"
    assert heartbeat_calls[0]["timing_factor"] == 0.05
    assert heartbeat_calls[0]["hypothesis_kpi_report_path"] == Path(
        report["artifacts"]["hypothesis_kpi_report_path"]
    )
    assert report["metrics"]["trace_bootstrapped"] is True


def test_hermes_runtime_closed_loop_rejects_task1_sample_count_mismatch(
    tmp_path: Path,
) -> None:
    import numpy as np
    import pytest

    from pde_agent.core.hermes_runtime_closed_loop import (
        HermesRuntimeClosedLoopError,
        run_hermes_runtime_closed_loop,
    )

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    test_path = _write_hdf5_tensor(
        tmp_path / "task1_test.hdf5",
        np.zeros((3, 10, 256), dtype=np.float32),
    )
    target_path = _write_hdf5_tensor(
        tmp_path / "task1_target.hdf5",
        np.zeros((2, 200, 256), dtype=np.float32),
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("heartbeat should not run after preflight failure")

    with pytest.raises(HermesRuntimeClosedLoopError, match="Task 1 sample count"):
        run_hermes_runtime_closed_loop(
            output_dir=tmp_path / "runtime_loop",
            trace_path=tmp_path / "trace.jsonl",
            initial_submission_dir=initial_dir,
            candidate_generator="postprocess",
            task1_test_path=test_path,
            task1_target_path=target_path,
            heartbeat_runner=fail_if_called,
            force=False,
        )


def test_hermes_runtime_closed_loop_rejects_invalid_task1_reference_shape(
    tmp_path: Path,
) -> None:
    import numpy as np
    import pytest

    from pde_agent.core.hermes_runtime_closed_loop import (
        HermesRuntimeClosedLoopError,
        run_hermes_runtime_closed_loop,
    )

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    test_path = _write_hdf5_tensor(
        tmp_path / "task1_test.hdf5",
        np.zeros((2, 10, 256), dtype=np.float32),
    )
    target_path = _write_hdf5_tensor(
        tmp_path / "task1_target.hdf5",
        np.zeros((2, 200, 256), dtype=np.float32),
    )
    reference_path = _write_hdf5_tensor(
        tmp_path / "task1_reference_short.hdf5",
        np.zeros((64, 41, 256), dtype=np.float32),
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("heartbeat should not run after reference preflight failure")

    with pytest.raises(HermesRuntimeClosedLoopError, match="Task 1 reference"):
        run_hermes_runtime_closed_loop(
            output_dir=tmp_path / "runtime_loop",
            trace_path=tmp_path / "trace.jsonl",
            initial_submission_dir=initial_dir,
            candidate_generator="postprocess",
            task1_test_path=test_path,
            task1_target_path=target_path,
            task1_reference_path=reference_path,
            heartbeat_runner=fail_if_called,
            force=False,
        )


def test_hermes_runtime_closed_loop_records_heartbeat_candidate_attempts(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    def fake_heartbeat_runner(**kwargs):
        heartbeat_root = Path(kwargs["output_dir"])
        batch_report_path = heartbeat_root / "batch_001" / "loop.json"
        heartbeat_report_path = heartbeat_root / "hermes_heartbeat_report.json"
        candidate_dir = tmp_path / "candidate"
        candidate_dir.mkdir()
        best_dir = tmp_path / "best"
        best_dir.mkdir()
        batch_report_path.parent.mkdir(parents=True, exist_ok=True)
        batch_report_path.write_text(
            json.dumps(
                {
                    "trajectory": [
                        {
                            "round_index": 0,
                            "candidate_id": "initial",
                            "score": 100.0,
                            "score_delta_vs_best": 0.0,
                            "metadata": {"source": "initial_submission"},
                        },
                        {
                            "round_index": 1,
                            "candidate_id": "candidate_a",
                            "score": 99.5,
                            "score_delta_vs_best": -0.5,
                            "rules_passed": True,
                            "judge_report_path": str(tmp_path / "judge.json"),
                            "submission_dir": str(candidate_dir),
                            "metadata": {
                                "source": "sem_window_gate_blend_multi_search"
                            },
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(heartbeat_report_path),
            "batch_count": 1,
            "batches": [
                {
                    "batch_index": 1,
                    "batch_report_path": str(batch_report_path),
                    "candidate_generator": kwargs["candidate_generator"],
                }
            ],
            "best_candidate": {
                "candidate_id": "initial",
                "score": 100.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 100.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }
        heartbeat_report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    trace_path = tmp_path / "trace.jsonl"
    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attempt = next(
        event
        for event in events
        if event["event_type"] == "hypothesis_lab_run"
        and event["payload"]["algorithm_candidate_id"] == "candidate_a"
    )
    assert attempt["payload"]["candidate_family"] == (
        "sem_window_gate_blend_multi_early_smoke"
    )
    assert attempt["payload"]["implementation_source"] == (
        "sem_window_gate_blend_multi_search"
    )
    assert attempt["payload"]["outcome"] == "failed"
    assert attempt["payload"]["score_delta"] == -0.5
    assert report["metrics"]["recorded_candidate_attempt_count"] == 1


def test_audit_hermes_runtime_closed_loop_validates_artifacts(tmp_path: Path) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tool_catalog_path = tmp_path / "catalog.json"
    trace_summary_path = tmp_path / "summary.json"
    kpi_path = tmp_path / "kpi.json"
    doom_path = tmp_path / "doom.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    heartbeat_path = tmp_path / "heartbeat.json"
    blocked_tasks_path = tmp_path / "blocked_tasks.json"
    authoring_audit_path = tmp_path / "authoring_audit.json"
    completion_task_path = tmp_path / "completion_task.json"
    chronicle_path = tmp_path / "research_chronicle.md"
    runtime_path = tmp_path / "runtime.json"
    tool_catalog_path.write_text(
        json.dumps({"state_hint": "SCIENTIFIC_TOOL_CATALOG_BUILT", "tool_count": 8}),
        encoding="utf-8",
    )
    trace_summary_path.write_text(
        json.dumps({"state_hint": "HYPOTHESIS_LAB_TRACE_SUMMARIZED"}),
        encoding="utf-8",
    )
    kpi_path.write_text(
        json.dumps({"state_hint": "HYPOTHESIS_KPI_REPORT_BUILT"}),
        encoding="utf-8",
    )
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": ["task1_early_refinement"],
            }
        ),
        encoding="utf-8",
    )
    candidate_plan_path.write_text(
        json.dumps({"state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT"}),
        encoding="utf-8",
    )
    heartbeat_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_RAN",
                "guard": {
                    "guard_active": True,
                    "blocked_candidate_families": ["task1_early_refinement"],
                },
                "batches": [
                        {
                            "candidate_generator": "physics_residual_nudge",
                            "candidate_generator_metadata": {
                                "auto_fallback_allowed": True,
                                "cost_level": "light",
                                "default_budget": "smoke",
                                "requires_reference": False,
                                "requires_target": False,
                                "targets_error_modes": [
                                    "physical_residual",
                                    "late_stability",
                                ],
                            },
                            "guarded_candidate_generator": False,
                            "original_candidate_generator": "physics_residual_nudge",
                        }
                    ],
                    "parameters": {
                        "candidate_generator": "task1_early_refinement",
                        "candidate_generator_schedule": ["physics_residual_nudge"],
                    },
                }
            ),
            encoding="utf-8",
    )
    blocked_tasks_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED",
                "metrics": {"task_count": 1},
            }
        ),
        encoding="utf-8",
    )
    authoring_audit_path.write_text(
        json.dumps(
            {
                "state_hint": "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED",
                "authoring_task_path": str(tmp_path / "authoring.json"),
                "decision": "approved",
                "issue_count": 0,
            }
        ),
        encoding="utf-8",
    )
    completion_task_path.write_text(
        json.dumps(
            {
                "state_hint": "BOUNDED_EXECUTOR_COMPLETION_TASK_PREPARED",
                "authoring_task_path": str(tmp_path / "authoring.json"),
                "next_command": "complete-bounded-executor-authoring-task",
                "trace_path": str(trace_path),
                "command": (
                    "uv run --frozen pde-agent "
                    "complete-bounded-executor-authoring-task "
                    f"--trace-path {trace_path} --force"
                ),
            }
        ),
        encoding="utf-8",
    )
    chronicle_path.write_text(
        (
            "Hermes Research Chronicle\n\n"
            "Repair Policy Memory:\n"
            "- No repair policy memory artifact was provided.\n"
        ),
        encoding="utf-8",
    )
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    "tool_catalog_path": str(tool_catalog_path),
                    "trace_summary_path": str(trace_summary_path),
                    "hypothesis_kpi_report_path": str(kpi_path),
                    "doom_loop_report_path": str(doom_path),
                    "candidate_plan_path": str(candidate_plan_path),
                    "heartbeat_report_path": str(heartbeat_path),
                    "blocked_executor_tasks_path": str(blocked_tasks_path),
                    "blocked_executor_authoring_audit_paths": [
                        str(authoring_audit_path)
                    ],
                    "blocked_executor_completion_task_paths": [
                        str(completion_task_path)
                    ],
                    "research_chronicle_path": str(chronicle_path),
                },
                "metrics": {
                    "guard_active": True,
                    "blocked_executor_authoring_task_count": 1,
                    "blocked_executor_authoring_approved_count": 1,
                },
                "guard": {
                    "guard_active": True,
                    "blocked_candidate_families": ["task1_early_refinement"],
                },
                "next_command": "agent-author-bounded-executor",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        require_guard_effect=True,
        force=False,
    )

    assert audit["state_hint"] == "HERMES_RUNTIME_CLOSED_LOOP_AUDITED"
    assert audit["decision"] == "passed"
    assert all(check["passed"] for check in audit["checks"])
    chronicle_check = next(
        check
        for check in audit["checks"]
        if check["check_id"] == "research_chronicle_repair_policy_memory"
    )
    assert chronicle_check["passed"] is True
    assert audit["summary"]["artifact_count"] == 12


def test_audit_hermes_runtime_closed_loop_rejects_bad_repair_policy_summary(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "repair_policy_summary_path": tmp_path / "repair_policy.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "repair_policy_summary_path": "WRONG_STATE_HINT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "candidate_plan_path":
            payload.update({"candidate_generator_schedule": []})
        if key == "heartbeat_report_path":
            payload.update({"batches": [], "parameters": {}})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {"guard_active": False},
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "next_command": "audit-hermes-runtime-closed-loop",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "artifact_state_hints" in audit["errors"]
    state_check = next(
        check for check in audit["checks"] if check["check_id"] == "artifact_state_hints"
    )
    assert state_check["details"]["actual"]["repair_policy_summary_path"] == (
        "WRONG_STATE_HINT"
    )


def test_audit_hermes_runtime_closed_loop_rejects_uncarried_repair_policy_actions(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {
                    "outcome": "continue_next_heartbeat",
                    "repair_policy_recommended_actions": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    action = {
        "audit_error": "repair_convergence_consistency",
        "action": "switch_repair_strategy_or_reload_missing_evidence",
        "reason": "repair_nonconverged",
    }
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "repair_policy_summary_path": tmp_path / "repair_policy.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "repair_policy_summary_path": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "repair_policy_summary_path":
            payload.update(
                {
                    "recommended_actions": [action],
                    "metrics": {"recommended_action_count": 1},
                }
            )
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "candidate_plan_path":
            payload.update(
                {
                    "candidate_generator_schedule": [],
                    "learning_constraints": {
                        "repair_policy_recommended_actions": []
                    },
                }
            )
        if key == "heartbeat_report_path":
            payload.update({"batches": [], "parameters": {}})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {
                    "guard_active": False,
                    "repair_policy_recommended_action_count": 0,
                },
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "candidate_plan": {
                    "learning_constraints": {
                        "repair_policy_recommended_actions": []
                    },
                },
                "next_command": "audit-hermes-runtime-closed-loop",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "repair_policy_summary_consistency" in audit["errors"]
    policy_check = next(
        check
        for check in audit["checks"]
        if check["check_id"] == "repair_policy_summary_consistency"
    )
    assert policy_check["details"]["violations"][0]["reason"] == (
        "repair_policy_recommended_actions_mismatch"
    )


def test_audit_hermes_runtime_closed_loop_rejects_heartbeat_policy_drift(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    action = {
        "audit_error": "repair_convergence_consistency",
        "action": "switch_repair_strategy_or_reload_missing_evidence",
        "reason": "repair_nonconverged",
    }
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {
                    "outcome": "continue_next_heartbeat",
                    "repair_policy_recommended_actions": [action],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "repair_policy_summary_path": tmp_path / "repair_policy.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "repair_policy_summary_path": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "repair_policy_summary_path":
            payload.update(
                {
                    "recommended_actions": [action],
                    "metrics": {"recommended_action_count": 1},
                }
            )
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "candidate_plan_path":
            payload.update(
                {
                    "candidate_generator_schedule": [],
                    "learning_constraints": {
                        "repair_policy_recommended_actions": [action]
                    },
                }
            )
        if key == "heartbeat_report_path":
            payload.update(
                {
                    "batches": [
                        {
                            "repair_policy_recommended_actions": [],
                            "repair_policy_recommended_action_count": 0,
                        }
                    ],
                    "parameters": {"repair_policy_recommended_action_count": 0},
                    "repair_policy_recommended_actions": [],
                }
            )
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {
                    "guard_active": False,
                    "repair_policy_recommended_action_count": 1,
                },
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "candidate_plan": {
                    "learning_constraints": {
                        "repair_policy_recommended_actions": [action]
                    },
                    "repair_policy_recommended_actions": [action],
                },
                "next_command": "audit-hermes-runtime-closed-loop",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    policy_check = next(
        check
        for check in audit["checks"]
        if check["check_id"] == "repair_policy_summary_consistency"
    )
    assert any(
        violation["source"] == "heartbeat_report_path"
        for violation in policy_check["details"]["violations"]
    )


def test_audit_hermes_runtime_closed_loop_rejects_missing_batch_memory_constraints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    action = {
        "audit_error": "repair_convergence_consistency",
        "action": "switch_repair_strategy_or_reload_missing_evidence",
        "reason": "repair_nonconverged",
    }
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {
                    "outcome": "continue_next_heartbeat",
                    "repair_policy_recommended_actions": [action],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "repair_policy_summary_path": tmp_path / "repair_policy.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "repair_policy_summary_path": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "repair_policy_summary_path":
            payload.update(
                {
                    "recommended_actions": [action],
                    "metrics": {"recommended_action_count": 1},
                }
            )
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "candidate_plan_path":
            payload.update(
                {
                    "candidate_generator_schedule": [],
                    "learning_constraints": {
                        "repair_policy_recommended_actions": [action]
                    },
                }
            )
        if key == "heartbeat_report_path":
            payload.update(
                {
                    "batches": [
                        {
                            "repair_policy_recommended_actions": [action],
                            "repair_policy_recommended_action_count": 1,
                            "candidate_diagnostics": {},
                        }
                    ],
                    "parameters": {"repair_policy_recommended_action_count": 1},
                    "repair_policy_recommended_actions": [action],
                }
            )
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {
                    "guard_active": False,
                    "repair_policy_recommended_action_count": 1,
                },
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "candidate_plan": {
                    "learning_constraints": {
                        "repair_policy_recommended_actions": [action]
                    },
                    "repair_policy_recommended_actions": [action],
                },
                "next_command": "audit-hermes-runtime-closed-loop",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "repair_policy_memory_constraints_consistency" in audit["errors"]
    memory_check = next(
        check
        for check in audit["checks"]
        if check["check_id"] == "repair_policy_memory_constraints_consistency"
    )
    assert memory_check["details"]["violations"] == [
        {
            "reason": "missing_batch_memory_constraints",
            "source": "heartbeat_report_path",
            "batch_index": 1,
            "expected_count": 1,
        }
    ]


def test_audit_hermes_runtime_closed_loop_fails_when_memory_bridge_not_loaded(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
        "input_expansion_handoff_report_path": tmp_path / "handoff.json",
        "input_expansion_registry_summary_path": tmp_path / "registry_summary.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
        "input_expansion_handoff_report_path": (
            "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN"
        ),
        "input_expansion_registry_summary_path": (
            "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT"
        ),
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "candidate_plan_path":
            payload.update(
                {
                    "prior_expansion_exhaustion": {"exhausted": True},
                    "learning_constraints": {},
                    "candidate_generator_schedule": [],
                }
            )
        if key == "input_expansion_handoff_report_path":
            payload.update({"expansion_exhaustion": {"exhausted": True}})
        if key == "heartbeat_report_path":
            payload.update({"batches": [], "parameters": {}})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {
                    "guard_active": False,
                    "expansion_handoff_report_loaded": False,
                    "expansion_registry_summary_loaded": False,
                },
                "guard": {
                    "guard_active": False,
                    "blocked_candidate_families": [],
                },
                "next_command": "audit-hermes-runtime-closed-loop",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "memory_bridge_inputs" in audit["errors"]


def test_audit_hermes_runtime_closed_loop_fails_on_cluster_saturation_drift(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "candidate_plan_path":
            payload.update(
                {
                    "candidate_generator_schedule": [],
                    "learning_constraints": {
                        "saturated_candidate_families": ["local_strategy_skill"]
                    },
                }
            )
        if key == "heartbeat_report_path":
            payload.update({"batches": [], "parameters": {}})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {
                    "guard_active": False,
                    "cluster_saturation_signal_count": 1,
                    "saturated_candidate_families": ["local_strategy_skill"],
                },
                "guard": {
                    "guard_active": False,
                    "blocked_candidate_families": [],
                },
                "candidate_plan": {
                    "learning_constraints": {
                        "saturated_candidate_families": ["local_strategy_skill"]
                    }
                },
                "next_command": "audit-hermes-runtime-closed-loop",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "cluster_saturation_consistency" in audit["errors"]


def test_audit_hermes_runtime_closed_loop_fails_on_repair_nonconvergence_drift(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {
                    "outcome": "continue_next_heartbeat",
                    "repair_nonconvergence_errors": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "hypothesis_kpi_report_path":
            payload.update(
                {
                    "repair_nonconvergence_errors": [
                        "cluster_saturation_consistency"
                    ],
                    "metrics": {"repair_nonconvergence_error_count": 1},
                }
            )
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "candidate_plan_path":
            payload.update(
                {
                    "candidate_generator_schedule": ["physics_residual_nudge"],
                    "learning_constraints": {
                        "repair_nonconvergence_errors": [
                            "cluster_saturation_consistency"
                        ]
                    },
                }
            )
        if key == "heartbeat_report_path":
            payload.update({"batches": [], "parameters": {}})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {
                    "guard_active": False,
                    "repair_nonconvergence_error_count": 1,
                    "repair_nonconvergence_errors": [
                        "cluster_saturation_consistency"
                    ],
                },
                "guard": {
                    "guard_active": False,
                    "blocked_candidate_families": [],
                },
                "candidate_plan": {
                    "learning_constraints": {
                        "repair_nonconvergence_errors": [
                            "cluster_saturation_consistency"
                        ]
                    },
                    "repair_nonconvergence_errors": [
                        "cluster_saturation_consistency"
                    ],
                },
                "next_command": "audit-hermes-runtime-closed-loop",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "repair_nonconvergence_consistency" in audit["errors"]
    check = next(
        item
        for item in audit["checks"]
        if item["check_id"] == "repair_nonconvergence_consistency"
    )
    assert check["details"]["violations"][0]["reason"] == (
        "repair_nonconvergence_errors_mismatch"
    )


def test_audit_hermes_runtime_closed_loop_fails_on_repair_convergence_drift(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {
                    "outcome": "continue_next_heartbeat",
                    "repair_convergence_errors": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "hypothesis_kpi_report_path":
            payload.update(
                {
                    "repair_convergence_errors": [
                        "learning_constraint_smoke_gate"
                    ],
                    "metrics": {"repair_convergence_error_count": 1},
                }
            )
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "candidate_plan_path":
            payload.update(
                {
                    "candidate_generator_schedule": ["postprocess"],
                    "learning_constraints": {
                        "repair_convergence_errors": [
                            "learning_constraint_smoke_gate"
                        ],
                    },
                }
            )
        if key == "heartbeat_report_path":
            payload.update({"batches": [], "parameters": {}})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {
                    "guard_active": False,
                    "repair_convergence_error_count": 1,
                    "repair_convergence_errors": [
                        "learning_constraint_smoke_gate"
                    ],
                },
                "guard": {
                    "guard_active": False,
                    "blocked_candidate_families": [],
                },
                "candidate_plan": {
                    "learning_constraints": {
                        "repair_convergence_errors": [
                            "learning_constraint_smoke_gate"
                        ],
                    },
                    "repair_convergence_errors": [
                        "learning_constraint_smoke_gate"
                    ],
                },
                "next_command": "audit-hermes-runtime-closed-loop",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "repair_convergence_consistency" in audit["errors"]
    check = next(
        item
        for item in audit["checks"]
        if item["check_id"] == "repair_convergence_consistency"
    )
    assert check["details"]["violations"][0]["reason"] == (
        "repair_convergence_errors_mismatch"
    )


def test_audit_hermes_runtime_closed_loop_fails_when_attempt_trace_missing(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {"recorded_candidate_attempt_count": 1},
                "guard": {"guard_active": False, "blocked_candidate_families": []},
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        require_guard_effect=False,
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "candidate_attempt_trace_sync" in audit["errors"]


def test_audit_hermes_runtime_closed_loop_fails_when_bounded_authoring_rejected(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tool_catalog_path = tmp_path / "catalog.json"
    trace_summary_path = tmp_path / "summary.json"
    kpi_path = tmp_path / "kpi.json"
    doom_path = tmp_path / "doom.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    heartbeat_path = tmp_path / "heartbeat.json"
    blocked_tasks_path = tmp_path / "blocked_tasks.json"
    authoring_audit_path = tmp_path / "authoring_audit.json"
    runtime_path = tmp_path / "runtime.json"
    artifacts = [
        (tool_catalog_path, "SCIENTIFIC_TOOL_CATALOG_BUILT"),
        (trace_summary_path, "HYPOTHESIS_LAB_TRACE_SUMMARIZED"),
        (kpi_path, "HYPOTHESIS_KPI_REPORT_BUILT"),
        (doom_path, "HYPOTHESIS_DOOM_LOOP_CHECKED"),
        (candidate_plan_path, "HYPOTHESIS_CANDIDATE_PLAN_BUILT"),
        (heartbeat_path, "HERMES_HEARTBEAT_RAN"),
    ]
    for path, state_hint in artifacts:
        payload = {"state_hint": state_hint}
        if path == doom_path:
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
    blocked_tasks_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED",
                "metrics": {"task_count": 1},
            }
        ),
        encoding="utf-8",
    )
    authoring_audit_path.write_text(
        json.dumps(
            {
                "state_hint": "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED",
                "decision": "rejected",
                "issue_count": 1,
            }
        ),
        encoding="utf-8",
    )
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    "tool_catalog_path": str(tool_catalog_path),
                    "trace_summary_path": str(trace_summary_path),
                    "hypothesis_kpi_report_path": str(kpi_path),
                    "doom_loop_report_path": str(doom_path),
                    "candidate_plan_path": str(candidate_plan_path),
                    "heartbeat_report_path": str(heartbeat_path),
                    "blocked_executor_tasks_path": str(blocked_tasks_path),
                    "blocked_executor_authoring_audit_paths": [
                        str(authoring_audit_path)
                    ],
                },
                "metrics": {
                    "guard_active": False,
                    "blocked_executor_authoring_task_count": 1,
                    "blocked_executor_authoring_approved_count": 0,
                },
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "next_command": "agent-author-bounded-executor",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "bounded_executor_authoring_gate" in audit["errors"]


def test_audit_hermes_runtime_closed_loop_fails_without_completion_task(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tool_catalog_path = tmp_path / "catalog.json"
    trace_summary_path = tmp_path / "summary.json"
    kpi_path = tmp_path / "kpi.json"
    doom_path = tmp_path / "doom.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    heartbeat_path = tmp_path / "heartbeat.json"
    blocked_tasks_path = tmp_path / "blocked_tasks.json"
    authoring_audit_path = tmp_path / "authoring_audit.json"
    runtime_path = tmp_path / "runtime.json"
    artifacts = [
        (tool_catalog_path, "SCIENTIFIC_TOOL_CATALOG_BUILT"),
        (trace_summary_path, "HYPOTHESIS_LAB_TRACE_SUMMARIZED"),
        (kpi_path, "HYPOTHESIS_KPI_REPORT_BUILT"),
        (doom_path, "HYPOTHESIS_DOOM_LOOP_CHECKED"),
        (candidate_plan_path, "HYPOTHESIS_CANDIDATE_PLAN_BUILT"),
        (heartbeat_path, "HERMES_HEARTBEAT_RAN"),
    ]
    for path, state_hint in artifacts:
        payload = {"state_hint": state_hint}
        if path == doom_path:
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
    blocked_tasks_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED",
                "metrics": {"task_count": 1},
            }
        ),
        encoding="utf-8",
    )
    authoring_audit_path.write_text(
        json.dumps(
            {
                "state_hint": "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED",
                "decision": "approved",
                "issue_count": 0,
            }
        ),
        encoding="utf-8",
    )
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    "tool_catalog_path": str(tool_catalog_path),
                    "trace_summary_path": str(trace_summary_path),
                    "hypothesis_kpi_report_path": str(kpi_path),
                    "doom_loop_report_path": str(doom_path),
                    "candidate_plan_path": str(candidate_plan_path),
                    "heartbeat_report_path": str(heartbeat_path),
                    "blocked_executor_tasks_path": str(blocked_tasks_path),
                    "blocked_executor_authoring_audit_paths": [
                        str(authoring_audit_path)
                    ],
                },
                "metrics": {
                    "guard_active": False,
                    "blocked_executor_authoring_task_count": 1,
                    "blocked_executor_authoring_approved_count": 1,
                },
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "next_command": "agent-author-bounded-executor",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "bounded_executor_authoring_gate" in audit["errors"]


def test_audit_hermes_runtime_closed_loop_rejects_completion_task_without_trace(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "agent-author-bounded-executor"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.json"
    tool_catalog_path = tmp_path / "catalog.json"
    trace_summary_path = tmp_path / "summary.json"
    kpi_path = tmp_path / "kpi.json"
    doom_path = tmp_path / "doom.json"
    candidate_plan_path = tmp_path / "candidate_plan.json"
    heartbeat_path = tmp_path / "heartbeat.json"
    blocked_tasks_path = tmp_path / "blocked_tasks.json"
    authoring_path = tmp_path / "authoring.json"
    authoring_audit_path = tmp_path / "authoring_audit.json"
    completion_task_path = tmp_path / "completion_task.json"
    for path, state_hint in [
        (tool_catalog_path, "SCIENTIFIC_TOOL_CATALOG_BUILT"),
        (trace_summary_path, "HYPOTHESIS_LAB_TRACE_SUMMARIZED"),
        (kpi_path, "HYPOTHESIS_KPI_REPORT_BUILT"),
        (candidate_plan_path, "HYPOTHESIS_CANDIDATE_PLAN_BUILT"),
        (heartbeat_path, "HERMES_HEARTBEAT_RAN"),
    ]:
        path.write_text(json.dumps({"state_hint": state_hint}), encoding="utf-8")
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": False,
                "blocked_candidate_families": [],
            }
        ),
        encoding="utf-8",
    )
    blocked_tasks_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED",
                "metrics": {"task_count": 1},
            }
        ),
        encoding="utf-8",
    )
    authoring_audit_path.write_text(
        json.dumps(
            {
                "state_hint": "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED",
                "authoring_task_path": str(authoring_path),
                "decision": "approved",
                "issue_count": 0,
            }
        ),
        encoding="utf-8",
    )
    completion_task_path.write_text(
        json.dumps(
            {
                "state_hint": "BOUNDED_EXECUTOR_COMPLETION_TASK_PREPARED",
                "authoring_task_path": str(authoring_path),
                "next_command": "complete-bounded-executor-authoring-task",
                "command": "uv run --frozen pde-agent complete-bounded-executor-authoring-task --force",
            }
        ),
        encoding="utf-8",
    )
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    "tool_catalog_path": str(tool_catalog_path),
                    "trace_summary_path": str(trace_summary_path),
                    "hypothesis_kpi_report_path": str(kpi_path),
                    "doom_loop_report_path": str(doom_path),
                    "candidate_plan_path": str(candidate_plan_path),
                    "heartbeat_report_path": str(heartbeat_path),
                    "blocked_executor_tasks_path": str(blocked_tasks_path),
                    "blocked_executor_authoring_audit_paths": [
                        str(authoring_audit_path)
                    ],
                    "blocked_executor_completion_task_paths": [
                        str(completion_task_path)
                    ],
                },
                "metrics": {
                    "guard_active": False,
                    "blocked_executor_authoring_task_count": 1,
                    "blocked_executor_authoring_approved_count": 1,
                },
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "next_command": "agent-author-bounded-executor",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        require_guard_effect=False,
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "bounded_executor_authoring_gate" in audit["errors"]
    bounded_check = next(
        check
        for check in audit["checks"]
        if check["check_id"] == "bounded_executor_authoring_gate"
    )
    assert bounded_check["details"]["reason"] == "completion_task_missing_or_invalid"
    assert bounded_check["details"]["completion_tasks"][0]["trace_matches"] is False


def test_audit_hermes_runtime_closed_loop_rejects_mismatched_completion_task(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
    blocked_tasks_path = tmp_path / "blocked_tasks.json"
    blocked_tasks_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED",
                "tasks": [{"authoring_task_path": str(tmp_path / "authoring.json")}],
            }
        ),
        encoding="utf-8",
    )
    authoring_audit_path = tmp_path / "authoring_audit.json"
    authoring_audit_path.write_text(
        json.dumps(
            {
                "state_hint": "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED",
                "authoring_task_path": str(tmp_path / "authoring.json"),
                "decision": "approved",
                "issue_count": 0,
            }
        ),
        encoding="utf-8",
    )
    completion_task_path = tmp_path / "completion_task.json"
    completion_task_path.write_text(
        json.dumps(
            {
                "state_hint": "BOUNDED_EXECUTOR_COMPLETION_TASK_PREPARED",
                "authoring_task_path": str(tmp_path / "other_authoring.json"),
                "next_command": "complete-bounded-executor-authoring-task",
                "command": "uv run --frozen pde-agent complete-bounded-executor-authoring-task --force",
            }
        ),
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                    "blocked_executor_tasks_path": str(blocked_tasks_path),
                    "blocked_executor_authoring_audit_paths": [
                        str(authoring_audit_path)
                    ],
                    "blocked_executor_completion_task_paths": [
                        str(completion_task_path)
                    ],
                },
                "metrics": {
                    "guard_active": False,
                    "blocked_executor_authoring_task_count": 1,
                    "blocked_executor_authoring_approved_count": 1,
                },
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "next_command": "agent-author-bounded-executor",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "bounded_executor_authoring_gate" in audit["errors"]


def test_audit_hermes_runtime_closed_loop_rejects_guarded_heavy_generator(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update(
                {
                    "guard_active": True,
                    "blocked_candidate_families": ["postprocess"],
                }
            )
        if key == "heartbeat_report_path":
            payload.update(
                {
                    "guard": {
                        "guard_active": True,
                        "blocked_candidate_families": ["postprocess"],
                    },
                    "batches": [
                        {
                            "candidate_generator": (
                                "sem_window_gate_blend_multi_search"
                            ),
                            "candidate_generator_metadata": {
                                "auto_fallback_allowed": False,
                                "cost_level": "heavy",
                                "default_budget": "explicit",
                                "requires_reference": False,
                                "requires_target": True,
                            },
                            "guarded_candidate_generator": True,
                        }
                    ],
                }
            )
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {"guard_active": True},
                "guard": {
                    "guard_active": True,
                    "blocked_candidate_families": ["postprocess"],
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        require_guard_effect=True,
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "candidate_generator_registry_policy" in audit["errors"]


def test_audit_hermes_runtime_closed_loop_rejects_saturated_heartbeat_generator(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "heartbeat_report_path":
            payload.update(
                {
                    "parameters": {
                        "saturated_candidate_families": ["local_strategy_skill"]
                    },
                    "batches": [
                        {
                            "candidate_generator": (
                                "sem_window_gate_blend_multi_early_smoke"
                            ),
                            "candidate_generator_metadata": {
                                "auto_fallback_allowed": True,
                                "cost_level": "medium",
                                "default_budget": "smoke",
                                "requires_reference": False,
                                "requires_target": True,
                                "saturation_blocked_by": ["local_strategy_skill"],
                                "targets_error_modes": [
                                    "early_segment",
                                    "transition_window",
                                ],
                            },
                            "guarded_candidate_generator": False,
                            "original_candidate_generator": (
                                "sem_window_gate_blend_multi_early_smoke"
                            ),
                        }
                    ],
                }
            )
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {"guard_active": False},
                "guard": {
                    "guard_active": False,
                    "blocked_candidate_families": [],
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "heartbeat_saturation_policy" in audit["errors"]


def test_audit_hermes_runtime_closed_loop_allows_marked_saturation_probe(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        if key == "heartbeat_report_path":
            payload.update(
                {
                    "parameters": {
                        "saturated_candidate_families": ["local_strategy_skill"],
                        "saturation_probe_interval_batches": 2,
                    },
                    "batches": [
                        {
                            "batch_index": 2,
                            "candidate_generator": (
                                "sem_window_gate_blend_multi_early_smoke"
                            ),
                            "candidate_generator_metadata": {
                                "auto_fallback_allowed": True,
                                "cost_level": "medium",
                                "default_budget": "smoke",
                                "requires_reference": False,
                                "requires_target": True,
                                "saturation_blocked_by": ["local_strategy_skill"],
                                "targets_error_modes": [
                                    "early_segment",
                                    "transition_window",
                                ],
                            },
                            "guarded_candidate_generator": False,
                            "original_candidate_generator": (
                                "sem_window_gate_blend_multi_early_smoke"
                            ),
                            "saturation_probe": True,
                            "saturation_probe_reason": (
                                "interval_reprobe_of_saturated_candidate_family"
                            ),
                        }
                    ],
                }
            )
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {"guard_active": False},
                "guard": {
                    "guard_active": False,
                    "blocked_candidate_families": [],
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "passed"


def test_audit_hermes_runtime_closed_loop_command_and_cli(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {"guard_active": False},
                "guard": {"guard_active": False, "blocked_candidate_families": []},
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "audit.json"

    result = CliRunner().invoke(
        app,
        [
            "audit-hermes-runtime-closed-loop",
            "--runtime-report-path",
            str(runtime_path),
            "--output-path",
            str(audit_path),
            "--require-learning-constraint-smoke",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_RUNTIME_CLOSED_LOOP_AUDITED"
    assert payload["metrics"]["decision"] == "failed"
    assert payload["artifacts"]["runtime_closed_loop_audit_json"] == str(audit_path)
    assert "learning_constraint_smoke_gate" in payload["artifacts"]["audit_errors"]
    assert "learning_constraint_smoke_gate" in payload["artifacts"]["errors"]
    assert payload["artifacts"]["failed_checks"][0]["check_id"] == (
        "learning_constraint_smoke_gate"
    )
    assert "learning_constraint_smoke_gate" in json.loads(
        audit_path.read_text(encoding="utf-8")
    )["errors"]
    assert audit_path.exists()


def test_audit_hermes_runtime_closed_loop_requires_learning_constraint_smoke(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_payload = {
        "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
        "artifacts": {
            "runtime_loop_report_path": str(runtime_path),
            "trace_path": str(trace_path),
            **{key: str(path) for key, path in paths.items()},
        },
        "metrics": {"guard_active": False},
        "guard": {"guard_active": False, "blocked_candidate_families": []},
    }
    runtime_path.write_text(json.dumps(runtime_payload), encoding="utf-8")

    missing = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit_missing.json",
        require_guard_effect=False,
        require_learning_constraint_smoke=True,
        force=False,
    )

    assert missing["decision"] == "failed"
    assert "learning_constraint_smoke_gate" in missing["errors"]

    smoke_path = tmp_path / "learning_constraint_smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN",
                "decision": "passed",
                "trace_path": str(tmp_path / "different_trace.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    runtime_payload["artifacts"]["learning_constraint_smoke_report_path"] = str(
        smoke_path
    )
    runtime_path.write_text(json.dumps(runtime_payload), encoding="utf-8")

    mismatched = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit_mismatched.json",
        require_guard_effect=False,
        require_learning_constraint_smoke=True,
        force=False,
    )

    assert mismatched["decision"] == "failed"
    mismatch_check = next(
        check
        for check in mismatched["checks"]
        if check["check_id"] == "learning_constraint_smoke_gate"
    )
    assert mismatch_check["details"]["trace_matches"] is False

    smoke_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN",
                "decision": "passed",
                "trace_path": str(trace_path),
            }
        ),
        encoding="utf-8",
    )

    passed = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit_passed.json",
        require_guard_effect=False,
        require_learning_constraint_smoke=True,
        force=False,
    )

    assert passed["decision"] == "passed"
    smoke_check = next(
        check
        for check in passed["checks"]
        if check["check_id"] == "learning_constraint_smoke_gate"
    )
    assert smoke_check["details"]["decision"] == "passed"


def test_audit_hermes_runtime_closed_loop_requires_runtime_memory_bridge_smoke(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "tool_catalog_path": tmp_path / "catalog.json",
        "trace_summary_path": tmp_path / "summary.json",
        "hypothesis_kpi_report_path": tmp_path / "kpi.json",
        "doom_loop_report_path": tmp_path / "doom.json",
        "candidate_plan_path": tmp_path / "candidate_plan.json",
        "heartbeat_report_path": tmp_path / "heartbeat.json",
    }
    state_hints = {
        "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
        "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
        "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
        "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
        "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
    }
    for key, path in paths.items():
        payload = {"state_hint": state_hints[key]}
        if key == "doom_loop_report_path":
            payload.update({"guard_active": False, "blocked_candidate_families": []})
        path.write_text(json.dumps(payload), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_payload = {
        "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
        "artifacts": {
            "runtime_loop_report_path": str(runtime_path),
            "trace_path": str(trace_path),
            **{key: str(path) for key, path in paths.items()},
        },
        "metrics": {"guard_active": False},
        "guard": {"guard_active": False, "blocked_candidate_families": []},
    }
    runtime_path.write_text(json.dumps(runtime_payload), encoding="utf-8")

    missing = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit_missing.json",
        require_guard_effect=False,
        require_runtime_memory_bridge_smoke=True,
        force=False,
    )

    assert missing["decision"] == "failed"
    assert "runtime_memory_bridge_smoke_gate" in missing["errors"]

    smoke_path = tmp_path / "runtime_memory_bridge_smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
                "decision": "failed",
                "artifacts": {"runtime_loop_report_path": str(runtime_path)},
                "memory_bridge": {"passed": False},
            }
        ),
        encoding="utf-8",
    )
    runtime_payload["artifacts"]["runtime_memory_bridge_smoke_report_path"] = str(
        smoke_path
    )
    runtime_path.write_text(json.dumps(runtime_payload), encoding="utf-8")

    failed = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit_failed.json",
        require_guard_effect=False,
        require_runtime_memory_bridge_smoke=True,
        force=False,
    )

    assert failed["decision"] == "failed"

    smoke_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
                "decision": "passed",
                "artifacts": {"runtime_loop_report_path": str(runtime_path)},
                "memory_bridge": {"passed": True},
            }
        ),
        encoding="utf-8",
    )

    passed = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit_passed.json",
        require_guard_effect=False,
        require_runtime_memory_bridge_smoke=True,
        force=False,
    )

    assert passed["decision"] == "passed"
    smoke_check = next(
        check
        for check in passed["checks"]
        if check["check_id"] == "runtime_memory_bridge_smoke_gate"
    )
    assert smoke_check["details"]["memory_bridge_passed"] is True


def test_audit_hermes_runtime_closed_loop_command_forwards_runtime_memory_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as runtime_commands

    calls = []

    def fake_audit(**kwargs):
        calls.append(kwargs)
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "passed",
            "summary": {"check_count": 20, "failed_check_count": 0},
            "errors": [],
            "checks": [],
            "next_command": "use-hermes-runtime-closed-loop",
        }

    monkeypatch.setattr(
        runtime_commands,
        "audit_hermes_runtime_closed_loop",
        fake_audit,
    )

    payload = runtime_commands.audit_hermes_runtime_closed_loop_command(
        runtime_report_path=str(tmp_path / "runtime.json"),
        output_path=str(tmp_path / "audit.json"),
        require_guard_effect=False,
        require_learning_constraint_smoke=False,
        require_runtime_memory_bridge_smoke=True,
        force=False,
    )

    assert payload["status"] == "ok"
    assert payload["metrics"]["decision"] == "passed"
    assert calls[0]["require_runtime_memory_bridge_smoke"] is True


def test_audit_hermes_runtime_closed_loop_rejects_unenforced_learning_constraint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {"outcome": "continue_next_heartbeat"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payloads = {
        "tool_catalog_path": {"state_hint": "SCIENTIFIC_TOOL_CATALOG_BUILT"},
        "trace_summary_path": {"state_hint": "HYPOTHESIS_LAB_TRACE_SUMMARIZED"},
        "hypothesis_kpi_report_path": {
            "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT"
        },
        "doom_loop_report_path": {
            "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
            "guard_active": False,
            "blocked_candidate_families": [],
        },
        "candidate_plan_path": {
            "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
            "learning_constraints": {
                "avoid_candidate_families": ["postprocess"],
                "mutation_required_for_families": ["postprocess"],
                "reason_codes": ["repeated_failed_family"],
            },
            "candidate_generator_schedule": ["postprocess"],
        },
        "heartbeat_report_path": {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "batches": [{"batch_index": 1, "candidate_generator": "postprocess"}],
        },
    }
    paths = {}
    for key, payload in payloads.items():
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[key] = path
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {"guard_active": False},
                "guard": {"guard_active": False, "blocked_candidate_families": []},
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        require_guard_effect=False,
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "learning_constraint_enforcement" in audit["errors"]
    check = next(
        item
        for item in audit["checks"]
        if item["check_id"] == "learning_constraint_enforcement"
    )
    assert check["details"]["violations"] == [
        {"location": "candidate_plan_schedule", "candidate_generator": "postprocess"},
        {
            "location": "heartbeat_batch",
            "batch_index": 1,
            "candidate_generator": "postprocess",
        },
    ]


def test_audit_hermes_runtime_closed_loop_fails_on_persistent_search_guidance_drift(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    guidance = {
        "guidance_type": "escalate_after_guided_search_exhaustion",
        "reason": "memory_guided_specs_failed_to_improve",
        "source": "sem_window_gate_blend_multi_search",
        "recommended_action": "switch_gate_feature_family_or_try_new_dynamics_source",
    }
    drifted_guidance = {
        **guidance,
        "recommended_action": "refine_gate_quantiles_before_escalation",
    }
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "hermes_runtime_closed_loop",
                "payload": {
                    "outcome": "continue_next_heartbeat",
                    "persistent_search_guidance": [guidance],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payloads = {
        "tool_catalog_path": {"state_hint": "SCIENTIFIC_TOOL_CATALOG_BUILT"},
        "trace_summary_path": {"state_hint": "HYPOTHESIS_LAB_TRACE_SUMMARIZED"},
        "hypothesis_kpi_report_path": {
            "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
            "persistent_search_guidance": [drifted_guidance],
            "metrics": {"persistent_search_guidance_count": 1},
        },
        "doom_loop_report_path": {
            "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
            "guard_active": False,
            "blocked_candidate_families": [],
        },
        "candidate_plan_path": {
            "state_hint": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
            "learning_constraints": {
                "persistent_search_guidance": [guidance],
                "search_guidance_action_counts": {
                    "switch_gate_feature_family_or_try_new_dynamics_source": 1
                },
            },
            "candidate_generator_schedule": ["physics_residual_nudge"],
        },
        "heartbeat_report_path": {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "persistent_search_guidance": [guidance],
            "parameters": {"persistent_search_guidance_count": 1},
            "batches": [
                {"batch_index": 1, "candidate_generator": "physics_residual_nudge"}
            ],
        },
    }
    paths = {}
    for key, payload in payloads.items():
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[key] = path
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {
                    "runtime_loop_report_path": str(runtime_path),
                    "trace_path": str(trace_path),
                    **{key: str(path) for key, path in paths.items()},
                },
                "metrics": {
                    "guard_active": False,
                    "persistent_search_guidance_count": 1,
                },
                "guard": {"guard_active": False, "blocked_candidate_families": []},
                "candidate_plan": {
                    "learning_constraints": {
                        "persistent_search_guidance": [guidance],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_path,
        output_path=tmp_path / "audit.json",
        require_guard_effect=False,
        force=False,
    )

    assert audit["decision"] == "failed"
    assert "persistent_search_guidance_consistency" in audit["errors"]
    check = next(
        item
        for item in audit["checks"]
        if item["check_id"] == "persistent_search_guidance_consistency"
    )
    assert check["details"]["violations"] == [
        {
            "reason": "persistent_search_guidance_payload_mismatch",
            "source": "hypothesis_kpi_report_path",
        }
    ]


def test_hermes_runtime_closed_loop_real_smoke_with_tiny_task1_submission(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    context = np.ones((1, 10, 256), dtype=np.float32)
    future = np.ones((1, 190, 256), dtype=np.float32)
    target = np.concatenate([context, future], axis=1)
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_target.hdf5", target)
    future_path = tmp_path / "future.npz"
    np.savez(future_path, prediction=future)
    submission = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_package",
        submission_id="team_hermes",
        method="external_prediction",
        source_prediction_path=future_path,
        train_time=0.0,
        inference_time=0.0,
        force=True,
    )
    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    for index in range(3):
        store.append_event(
            event_type="hypothesis_lab_run",
            payload={
                "algorithm_candidate_id": f"failed_{index}",
                "candidate_family": "task1_early_refinement",
                "outcome": "failed",
                "score_delta": 0.0,
            },
        )

    runtime = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=trace_path,
        initial_submission_dir=submission["artifacts"]["submission_dir"],
        candidate_generator="task1_early_refinement",
        candidate_generator_fallback="none",
        max_batches=1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=30.0,
        task1_test_path=test_path,
        task1_target_path=target_path,
        force=False,
    )
    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime["artifacts"]["runtime_loop_report_path"],
        output_path=tmp_path / "runtime_audit.json",
        require_guard_effect=True,
        force=False,
    )

    assert runtime["state_hint"] == "HERMES_RUNTIME_CLOSED_LOOP_RAN"
    assert runtime["metrics"]["guard_active"] is True
    assert runtime["metrics"]["heartbeat_best_score"] == 150.0
    assert runtime["heartbeat"]["best_candidate"]["candidate_id"] == "initial"
    assert audit["decision"] == "passed"
    assert Path(runtime["artifacts"]["heartbeat_report_path"]).exists()


def test_runtime_audit_chain_repair_tasks_feed_next_runtime_smoke(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 14, "failed_check_count": 1},
            "errors": ["cluster_saturation_consistency"],
            "next_command": "run-hermes-runtime-closed-loop",
        }

    chain = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "chain",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=2,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_audit,
        force=False,
    )
    repair_path = Path(chain["artifacts"]["audit_repair_tasks_path"])
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
                "expansion_exhaustion": {
                    "exhausted": True,
                    "reason": "local_strategy_mutation_axes_exhausted",
                    "exhausted_algorithm_node_ids": [
                        "local_strategy_a",
                        "local_strategy_b",
                    ],
                    "recommended_next_command": "plan-hypothesis-candidates",
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        repair_policy_path = kwargs.get("repair_policy_summary_path")
        repair_actions = []
        if repair_policy_path is not None:
            repair_policy = json.loads(
                Path(repair_policy_path).read_text(encoding="utf-8")
            )
            repair_actions = repair_policy.get("recommended_actions", [])
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "repair_policy_recommended_action_count": len(repair_actions)
            },
            "batch_count": 1,
            "batches": [
                {
                    "repair_policy_recommended_actions": repair_actions,
                    "repair_policy_recommended_action_count": len(repair_actions),
                    "candidate_diagnostics": {
                        "memory_constraints": {
                            "repair_policy_recommended_actions": repair_actions,
                            "repair_policy_recommended_action_count": len(
                                repair_actions
                            ),
                        }
                    },
                }
            ],
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(initial_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [],
            "repair_policy_recommended_actions": repair_actions,
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    runtime = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_repair_consumer",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        max_batches=1,
        expansion_handoff_report_path=handoff_path,
        audit_repair_tasks_path=repair_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert chain["decision"] == "failed"
    assert repair_path.exists()
    assert runtime["metrics"]["audit_repair_tasks_loaded"] is True
    assert runtime["candidate_plan"]["audit_repair_tasks"][0]["audit_error"] == (
        "cluster_saturation_consistency"
    )
    assert runtime["candidate_plan"]["candidate_generator_schedule"][0] == (
        "fast_baseline_family"
    )
    assert runtime["candidate_plan"]["learning_constraints"][
        "audit_repair_policies"
    ][0]["audit_error"] == "cluster_saturation_consistency"
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-2]["event_type"] == "hermes_runtime_audit_chain"
    assert events[-2]["payload"]["audit_repair_tasks_path"] == str(repair_path)
    assert events[-1]["event_type"] == "hermes_runtime_closed_loop"
    assert events[-1]["payload"]["audit_repair_tasks"][0]["audit_error"] == (
        "cluster_saturation_consistency"
    )


def test_runtime_audit_chain_convergence_repair_memory_roundtrip_smoke(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_audit_chain import (
        run_hermes_runtime_audit_chain,
    )
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )
    from pde_agent.core.hermes_runtime_closed_loop_audit import (
        audit_hermes_runtime_closed_loop,
    )

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    def fake_runtime(**kwargs):
        runtime_path = Path(kwargs["output_dir"]) / "hermes_runtime_closed_loop_report.json"
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
            "artifacts": {"runtime_loop_report_path": str(runtime_path)},
            "metrics": {"heartbeat_best_score": 100.0},
            "heartbeat": {"best_candidate": {}},
            "audit_command": {
                "command": "audit-hermes-runtime-closed-loop",
                "runtime_report_path": str(runtime_path),
                "output_path": str(Path(kwargs["output_dir"]) / "audit.json"),
            },
            "next_command": "audit-hermes-runtime-closed-loop",
        }

    def fake_failed_convergence_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "failed",
            "summary": {"check_count": 14, "failed_check_count": 1},
            "errors": ["repair_convergence_consistency"],
            "next_command": "run-hermes-runtime-closed-loop",
        }

    failed_chain = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "failed_chain",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        runtime_runner=fake_runtime,
        audit_runner=fake_failed_convergence_audit,
        force=False,
    )
    repair_path = Path(failed_chain["artifacts"]["audit_repair_tasks_path"])

    def fake_passed_audit(**kwargs):
        return {
            "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_AUDITED",
            "audit_report_path": str(kwargs["output_path"]),
            "decision": "passed",
            "summary": {"check_count": 14, "failed_check_count": 0},
            "errors": [],
            "next_command": "run-hermes-runtime-audit-chain",
        }

    passed_chain = run_hermes_runtime_audit_chain(
        output_dir=tmp_path / "passed_chain",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=60.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=None,
        task1_target_path=None,
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        audit_repair_tasks_path=repair_path,
        runtime_runner=fake_runtime,
        audit_runner=fake_passed_audit,
        force=False,
    )

    def fake_heartbeat_runner(**kwargs):
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        repair_policy_path = kwargs.get("repair_policy_summary_path")
        repair_actions = []
        if repair_policy_path is not None:
            repair_policy = json.loads(
                Path(repair_policy_path).read_text(encoding="utf-8")
            )
            repair_actions = repair_policy.get("recommended_actions", [])
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "repair_policy_recommended_action_count": len(repair_actions)
            },
            "batch_count": 1,
            "batches": [
                {
                    "repair_policy_recommended_actions": repair_actions,
                    "repair_policy_recommended_action_count": len(repair_actions),
                    "candidate_diagnostics": {
                        "memory_constraints": {
                            "repair_policy_recommended_actions": repair_actions,
                            "repair_policy_recommended_action_count": len(
                                repair_actions
                            ),
                        }
                    },
                }
            ],
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 101.0,
                "submission_dir": str(initial_dir),
            },
            "best_score": 101.0,
            "best_objective": "score",
            "best_objective_score": 101.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "persistent_search_guidance": [],
            "repair_policy_recommended_actions": repair_actions,
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    runtime = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_convergence_consumer",
        trace_path=trace_path,
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        max_batches=1,
        audit_repair_tasks_path=repair_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )
    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime["artifacts"]["runtime_loop_report_path"],
        output_path=tmp_path / "runtime_convergence_audit.json",
        require_guard_effect=False,
        force=False,
    )

    assert failed_chain["decision"] == "failed"
    assert repair_path.exists()
    assert passed_chain["repair_convergence"] == {
        "decision": "converged",
        "input_repair_errors": ["repair_convergence_consistency"],
        "repeated_audit_errors": [],
    }
    assert runtime["metrics"]["repair_convergence_errors"] == [
        "repair_convergence_consistency"
    ]
    assert runtime["candidate_plan"]["learning_constraints"][
        "audit_repair_converged_error_counts"
    ] == {"repair_convergence_consistency": 1}
    assert audit["decision"] == "passed"


def _write_hdf5_tensor(path: Path, data) -> Path:
    h5py = __import__("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path
