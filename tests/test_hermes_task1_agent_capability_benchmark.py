from __future__ import annotations

import json
from pathlib import Path


def test_task1_agent_capability_benchmark_passes_only_with_score_and_capability(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_task1_agent_capability_benchmark import (
        evaluate_hermes_task1_agent_capability,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    chain_path = _write_chain_report(
        tmp_path / "chain.json",
        scores=[86.0, 88.4],
        exhausted=[False, False],
    )
    trace_path = tmp_path / "trace.jsonl"
    store = HypothesisLabTraceStore(trace_path)
    store.append_event(
        event_type="algorithm_candidate_sandbox_result",
        payload={
            "algorithm_candidate_id": "cand_early_sem",
            "candidate_family": "task1_sem_front25_dynamics",
            "outcome": "promoted",
            "score_delta": 1.4,
            "sandbox_evidence_path": "runs/cand_early/evidence.json",
            "pareto_gate_passed": True,
            "independent_slice_evidence": True,
        },
        tags=["task1", "sandbox"],
    )
    store.append_event(
        event_type="candidate_failure_recorded",
        payload={
            "algorithm_candidate_id": "cand_bad_gate",
            "candidate_family": "task1_scalar_gate",
            "outcome": "rejected",
            "score_delta": -0.2,
            "failure_mode": "early_segment_regression",
            "failure_memory_written": True,
            "sandbox_evidence_path": "runs/cand_bad/evidence.json",
        },
        tags=["task1", "sandbox", "failure_memory"],
    )

    report = evaluate_hermes_task1_agent_capability(
        chain_report_path=chain_path,
        trace_path=trace_path,
        output_path=tmp_path / "benchmark.json",
        baseline_score=None,
        min_score_delta=0.5,
        min_unique_candidate_families=2,
        min_sandbox_evidence_events=2,
        require_independent_slice_evidence=True,
        force=False,
    )

    assert report["state_hint"] == "HERMES_TASK1_AGENT_CAPABILITY_BENCHMARKED"
    assert report["decision"] == "passed"
    assert report["score_progress"]["score_delta"] == 2.4
    assert report["candidate_generation"]["unique_candidate_family_count"] == 2
    assert report["candidate_generation"]["non_postprocess_ratio"] == 1.0
    assert report["sandbox_validation"]["evidence_event_count"] == 2
    assert report["sandbox_validation"]["outcome_counts"] == {
        "promoted": 1,
        "rejected": 1,
    }
    assert report["learning_memory"]["failure_memory_event_count"] == 1
    assert report["promotion_quality"]["invalid_promotion_count"] == 0
    assert report["promotion_quality"]["independent_slice_evidence_count"] == 1


def test_task1_agent_capability_benchmark_marks_score_only_when_trace_is_weak(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_task1_agent_capability_benchmark import (
        evaluate_hermes_task1_agent_capability,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    chain_path = _write_chain_report(
        tmp_path / "chain.json",
        scores=[86.0, 89.0],
        exhausted=[False, False],
    )
    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="algorithm_candidate_sandbox_result",
        payload={
            "algorithm_candidate_id": "cand_post",
            "candidate_family": "postprocess",
            "outcome": "promoted",
            "score_delta": 3.0,
        },
    )

    report = evaluate_hermes_task1_agent_capability(
        chain_report_path=chain_path,
        trace_path=trace_path,
        output_path=tmp_path / "benchmark.json",
        baseline_score=None,
        min_score_delta=0.5,
        min_unique_candidate_families=2,
        min_sandbox_evidence_events=1,
        require_independent_slice_evidence=False,
        force=False,
    )

    assert report["decision"] == "score_only"
    assert report["capability_checks"]["unique_candidate_families"]["passed"] is False
    assert report["capability_checks"]["evidence_events"]["passed"] is False


def test_task1_agent_capability_benchmark_accepts_candidate_source_artifact_as_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_task1_agent_capability_benchmark import (
        evaluate_hermes_task1_agent_capability,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    chain_path = _write_chain_report(tmp_path / "chain.json", scores=[1.0, 2.0])
    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="hypothesis_lab_run",
        payload={
            "algorithm_candidate_id": "cand_judged",
            "candidate_family": "postprocess",
            "outcome": "promoted",
        },
        source_artifact_path="runs/cand/judges/round_001_judge.json",
        tags=["hermes", "candidate_attempt"],
    )

    report = evaluate_hermes_task1_agent_capability(
        chain_report_path=chain_path,
        trace_path=trace_path,
        output_path=tmp_path / "benchmark.json",
        min_score_delta=0.5,
        min_unique_candidate_families=1,
        min_sandbox_evidence_events=1,
        require_independent_slice_evidence=False,
        force=False,
    )

    assert report["sandbox_validation"]["evidence_event_count"] == 1
    assert report["promotion_quality"]["evidence_backed_promotion_count"] == 1
    assert report["promotion_quality"]["invalid_promotion_count"] == 0


def test_task1_agent_capability_benchmark_scores_control_layer_memory_and_schedule(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_task1_agent_capability_benchmark import (
        evaluate_hermes_task1_agent_capability,
    )
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    chain_path = _write_chain_report(tmp_path / "chain.json", scores=[120.0, 120.4])
    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="algorithm_candidate_sandbox_result",
        payload={
            "algorithm_candidate_id": "cand_smooth",
            "candidate_family": "postprocess",
            "outcome": "promoted",
            "sandbox_evidence_path": "runs/cand_smooth/evidence.json",
            "pareto_gate_passed": True,
        },
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
                "avoid_candidate_families": [
                    {
                        "candidate_family": "linear_trend_spectral",
                        "reason": "severe_score_regression",
                    }
                ],
                "positive_signals": [
                    {
                        "candidate_id": "postprocess_spectral_smooth_w0.7325",
                        "candidate_family": "postprocess_generator",
                    }
                ],
                "family_outcomes": {
                    "linear_trend_spectral": {"status": "blocked"},
                    "postprocess_generator": {"status": "positive_signal"},
                },
            }
        ),
        encoding="utf-8",
    )
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_EXPLORATION_SCHEDULE_BUILT",
                "decision": "refine_positive_signal",
                "schedule": {
                    "candidate_generator": "postprocess",
                    "blocked_candidate_families": ["linear_trend_spectral"],
                    "promotion_gate": {"min_score_delta": 0.0001},
                },
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_hermes_task1_agent_capability(
        chain_report_path=chain_path,
        trace_path=trace_path,
        output_path=tmp_path / "benchmark.json",
        min_score_delta=0.1,
        min_unique_candidate_families=1,
        min_sandbox_evidence_events=1,
        memory_json_path=memory_path,
        exploration_schedule_path=schedule_path,
        require_control_layer_evidence=True,
        force=False,
    )

    assert report["decision"] == "passed"
    assert report["control_layer"]["failure_memory_family_count"] == 1
    assert report["control_layer"]["positive_signal_count"] == 1
    assert report["control_layer"]["schedule_decision"] == "refine_positive_signal"
    assert report["capability_checks"]["control_layer_evidence"]["passed"] is True


def test_task1_agent_capability_benchmark_cli_and_tool_catalog(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog
    from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore

    chain_path = _write_chain_report(tmp_path / "chain.json", scores=[1.0, 1.0])
    trace_path = tmp_path / "trace.jsonl"
    HypothesisLabTraceStore(trace_path).append_event(
        event_type="algorithm_candidate_sandbox_result",
        payload={
            "algorithm_candidate_id": "cand",
            "candidate_family": "task1_front25",
            "outcome": "failed",
            "sandbox_evidence_path": "runs/cand/evidence.json",
        },
    )
    output_path = tmp_path / "benchmark.json"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-hermes-task1-agent-capability",
            "--chain-report-path",
            str(chain_path),
            "--trace-path",
            str(trace_path),
            "--output-path",
            str(output_path),
            "--min-score-delta",
            "0.5",
            "--min-unique-candidate-families",
            "1",
            "--min-sandbox-evidence-events",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state_hint"] == "HERMES_TASK1_AGENT_CAPABILITY_BENCHMARKED"
    assert payload["artifacts"]["task1_agent_capability_benchmark_json"] == str(
        output_path
    )
    assert output_path.exists()

    catalog = build_scientific_tool_catalog(
        output_path=tmp_path / "catalog.json",
        force=False,
    )
    tools = {tool["name"]: tool for tool in catalog["tools"]}
    assert tools["evaluate-hermes-task1-agent-capability"]["risk_level"] == (
        "read_only"
    )
    assert "task1_agent_capability_benchmark_json" in tools[
        "evaluate-hermes-task1-agent-capability"
    ]["produced_artifacts"]


def _write_chain_report(
    path: Path,
    *,
    scores: list[float],
    exhausted: list[bool] | None = None,
) -> Path:
    exhausted_values = exhausted or [False for _ in scores]
    iterations = []
    for index, score in enumerate(scores, start=1):
        iterations.append(
            {
                "iteration_index": index,
                "audit_decision": "passed",
                "audit_failed_check_count": 0,
                "heartbeat_best_score": score,
                "candidate_space_exhausted": exhausted_values[index - 1],
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "iterations": iterations,
                "metrics": {
                    "completed_iteration_count": len(iterations),
                    "failed_audit_count": 0,
                    "candidate_space_exhausted_count": sum(exhausted_values),
                },
            }
        ),
        encoding="utf-8",
    )
    return path
