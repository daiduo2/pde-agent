"""Benchmark Hermes Task1 score improvement as reusable agent capability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_trace import read_hypothesis_lab_trace


STATE_HINT = "HERMES_TASK1_AGENT_CAPABILITY_BENCHMARKED"
CHAIN_STATE_HINT = "HERMES_RUNTIME_AUDIT_CHAIN_RAN"


class HermesTask1AgentCapabilityBenchmarkError(RuntimeError):
    pass


def evaluate_hermes_task1_agent_capability(
    *,
    chain_report_path: str | Path,
    trace_path: str | Path,
    output_path: str | Path,
    baseline_score: float | None = None,
    min_score_delta: float = 0.5,
    min_unique_candidate_families: int = 2,
    min_sandbox_evidence_events: int = 1,
    require_independent_slice_evidence: bool = False,
    memory_json_path: str | Path | None = None,
    exploration_schedule_path: str | Path | None = None,
    require_control_layer_evidence: bool = False,
    force: bool,
) -> dict[str, Any]:
    if min_unique_candidate_families < 1:
        raise HermesTask1AgentCapabilityBenchmarkError(
            "min_unique_candidate_families must be positive"
        )
    if min_sandbox_evidence_events < 0:
        raise HermesTask1AgentCapabilityBenchmarkError(
            "min_sandbox_evidence_events must be non-negative"
        )
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise HermesTask1AgentCapabilityBenchmarkError(
            f"Task1 agent capability benchmark exists: {output_file}"
        )

    chain_path = Path(chain_report_path)
    chain = _read_chain_report(chain_path)
    events = read_hypothesis_lab_trace(trace_path)
    score_progress = _score_progress(
        chain=chain,
        baseline_score=baseline_score,
        min_score_delta=min_score_delta,
    )
    candidate_generation = _candidate_generation(events)
    sandbox_validation = _sandbox_validation(events)
    learning_memory = _learning_memory(events)
    promotion_quality = _promotion_quality(events)
    control_layer = _control_layer(
        memory_json_path=memory_json_path,
        exploration_schedule_path=exploration_schedule_path,
    )
    capability_checks = _capability_checks(
        candidate_generation=candidate_generation,
        sandbox_validation=sandbox_validation,
        promotion_quality=promotion_quality,
        control_layer=control_layer,
        min_unique_candidate_families=min_unique_candidate_families,
        min_sandbox_evidence_events=min_sandbox_evidence_events,
        require_independent_slice_evidence=require_independent_slice_evidence,
        require_control_layer_evidence=require_control_layer_evidence,
    )
    score_improved = (
        isinstance(score_progress["score_delta"], float)
        and score_progress["score_delta"] >= min_score_delta
    )
    capability_passed = all(check["passed"] for check in capability_checks.values())
    decision = _decision(
        score_improved=score_improved,
        capability_passed=capability_passed,
        score_progress=score_progress,
        chain=chain,
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "decision": decision,
        "inputs": {
            "chain_report_path": str(chain_report_path),
            "trace_path": str(trace_path),
            "baseline_score": baseline_score,
            "min_score_delta": min_score_delta,
            "min_unique_candidate_families": min_unique_candidate_families,
            "min_sandbox_evidence_events": min_sandbox_evidence_events,
            "require_independent_slice_evidence": require_independent_slice_evidence,
            "memory_json_path": str(memory_json_path)
            if memory_json_path is not None
            else None,
            "exploration_schedule_path": str(exploration_schedule_path)
            if exploration_schedule_path is not None
            else None,
            "require_control_layer_evidence": require_control_layer_evidence,
        },
        "score_progress": score_progress,
        "candidate_generation": candidate_generation,
        "sandbox_validation": sandbox_validation,
        "learning_memory": learning_memory,
        "promotion_quality": promotion_quality,
        "control_layer": control_layer,
        "capability_checks": capability_checks,
        "next_command": _next_command(decision),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report["benchmark_report_path"] = str(output_file)
    return report


def _read_chain_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HermesTask1AgentCapabilityBenchmarkError(
            f"chain report does not exist: {path}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesTask1AgentCapabilityBenchmarkError(
            f"chain report is not valid JSON: {path}"
        ) from exc
    if not isinstance(report, dict):
        raise HermesTask1AgentCapabilityBenchmarkError(
            "chain report must be a JSON object"
        )
    if report.get("state_hint") != CHAIN_STATE_HINT:
        raise HermesTask1AgentCapabilityBenchmarkError(
            "chain report has invalid state_hint"
        )
    return report


def _score_progress(
    *,
    chain: dict[str, Any],
    baseline_score: float | None,
    min_score_delta: float,
) -> dict[str, Any]:
    iterations = chain.get("iterations")
    scores: list[float] = []
    if isinstance(iterations, list):
        for item in iterations:
            if not isinstance(item, dict):
                continue
            score = item.get("heartbeat_best_score")
            if isinstance(score, int | float) and not isinstance(score, bool):
                scores.append(float(score))
    initial = float(baseline_score) if baseline_score is not None else (
        scores[0] if scores else None
    )
    final = scores[-1] if scores else None
    best = max(scores) if scores else None
    score_delta = None
    if initial is not None and final is not None:
        score_delta = round(final - initial, 10)
    stagnation_count = 0
    for previous, current in zip(scores, scores[1:]):
        if current - previous < min_score_delta:
            stagnation_count += 1
    return {
        "initial_score": initial,
        "final_score": final,
        "best_score": best,
        "score_delta": score_delta,
        "score_series": scores,
        "stagnation_count": stagnation_count,
    }


def _candidate_generation(events: list[dict[str, Any]]) -> dict[str, Any]:
    families: list[str] = []
    for payload in _candidate_payloads(events):
        family = _optional_text(payload.get("candidate_family"))
        if family:
            families.append(family)
    unique_families = sorted(set(families))
    non_postprocess_count = sum(
        1 for family in families if "postprocess" not in family.lower()
    )
    ratio = non_postprocess_count / len(families) if families else 0.0
    return {
        "candidate_event_count": len(families),
        "unique_candidate_family_count": len(unique_families),
        "unique_candidate_families": unique_families,
        "non_postprocess_candidate_count": non_postprocess_count,
        "non_postprocess_ratio": round(ratio, 10),
    }


def _sandbox_validation(events: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_counts: dict[str, int] = {}
    evidence_event_count = 0
    for event in events:
        payload = _payload(event)
        family = _optional_text(payload.get("candidate_family"))
        if not family or family == "hermes_runtime_audit_chain":
            continue
        outcome = _optional_text(payload.get("outcome"))
        if outcome:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if _has_sandbox_evidence(event):
            evidence_event_count += 1
    return {
        "evidence_event_count": evidence_event_count,
        "outcome_counts": outcome_counts,
        "validated_outcome_count": sum(outcome_counts.values()),
    }


def _learning_memory(events: list[dict[str, Any]]) -> dict[str, Any]:
    failure_memory_event_count = 0
    taxonomies: list[str] = []
    for event in events:
        payload = _payload(event)
        tags = event.get("tags") if isinstance(event.get("tags"), list) else []
        event_type = str(event.get("event_type") or "")
        outcome = _optional_text(payload.get("outcome"))
        is_failure = outcome in {"failed", "rejected", "blocked"}
        if (
            payload.get("failure_memory_written") is True
            or "failure" in event_type
            or "failure_memory" in tags
        ) and is_failure:
            failure_memory_event_count += 1
        taxonomy = _optional_text(payload.get("failure_taxonomy")) or _optional_text(
            payload.get("failure_mode")
        )
        if taxonomy:
            taxonomies.append(taxonomy)
    return {
        "failure_memory_event_count": failure_memory_event_count,
        "failure_taxonomy_count": len(set(taxonomies)),
        "failure_taxonomies": sorted(set(taxonomies)),
    }


def _promotion_quality(events: list[dict[str, Any]]) -> dict[str, Any]:
    promoted_count = 0
    evidence_backed_promotion_count = 0
    pareto_gated_promotion_count = 0
    independent_slice_evidence_count = 0
    invalid_promotion_count = 0
    for event in events:
        payload = _payload(event)
        if payload.get("independent_slice_evidence") is True:
            independent_slice_evidence_count += 1
        if _optional_text(payload.get("outcome")) != "promoted":
            continue
        promoted_count += 1
        has_evidence = _has_sandbox_evidence(event)
        if has_evidence:
            evidence_backed_promotion_count += 1
        else:
            invalid_promotion_count += 1
        if payload.get("pareto_gate_passed") is True:
            pareto_gated_promotion_count += 1
    return {
        "promoted_count": promoted_count,
        "evidence_backed_promotion_count": evidence_backed_promotion_count,
        "pareto_gated_promotion_count": pareto_gated_promotion_count,
        "independent_slice_evidence_count": independent_slice_evidence_count,
        "invalid_promotion_count": invalid_promotion_count,
    }


def _control_layer(
    *,
    memory_json_path: str | Path | None,
    exploration_schedule_path: str | Path | None,
) -> dict[str, Any]:
    memory = _optional_json(memory_json_path)
    schedule = _optional_json(exploration_schedule_path)
    avoid_families = []
    positive_signals = []
    family_outcomes = {}
    if isinstance(memory, dict):
        avoid_families = [
            item
            for item in memory.get("avoid_candidate_families", [])
            if isinstance(item, dict)
        ]
        positive_signals = [
            item
            for item in memory.get("positive_signals", [])
            if isinstance(item, dict)
        ]
        if isinstance(memory.get("family_outcomes"), dict):
            family_outcomes = dict(memory["family_outcomes"])
    schedule_payload = schedule.get("schedule") if isinstance(schedule, dict) else {}
    schedule_payload = schedule_payload if isinstance(schedule_payload, dict) else {}
    return {
        "memory_loaded": isinstance(memory, dict),
        "schedule_loaded": isinstance(schedule, dict),
        "failure_memory_family_count": len(avoid_families),
        "positive_signal_count": len(positive_signals),
        "family_outcome_count": len(family_outcomes),
        "schedule_decision": schedule.get("decision")
        if isinstance(schedule, dict)
        else None,
        "scheduled_candidate_generator": schedule_payload.get("candidate_generator"),
        "scheduled_blocked_family_count": len(
            [
                item
                for item in schedule_payload.get("blocked_candidate_families", [])
                if isinstance(item, str)
            ]
        )
        if isinstance(schedule_payload.get("blocked_candidate_families"), list)
        else 0,
    }


def _capability_checks(
    *,
    candidate_generation: dict[str, Any],
    sandbox_validation: dict[str, Any],
    promotion_quality: dict[str, Any],
    control_layer: dict[str, Any],
    min_unique_candidate_families: int,
    min_sandbox_evidence_events: int,
    require_independent_slice_evidence: bool,
    require_control_layer_evidence: bool,
) -> dict[str, dict[str, Any]]:
    unique_family_count = candidate_generation["unique_candidate_family_count"]
    evidence_count = sandbox_validation["evidence_event_count"]
    invalid_promotion_count = promotion_quality["invalid_promotion_count"]
    independent_count = promotion_quality["independent_slice_evidence_count"]
    control_evidence_passed = (
        not require_control_layer_evidence
        or (
            control_layer["memory_loaded"]
            and control_layer["schedule_loaded"]
            and (
                control_layer["failure_memory_family_count"] > 0
                or control_layer["positive_signal_count"] > 0
            )
        )
    )
    return {
        "unique_candidate_families": {
            "passed": unique_family_count >= min_unique_candidate_families,
            "actual": unique_family_count,
            "minimum": min_unique_candidate_families,
        },
        "evidence_events": {
            "passed": evidence_count >= min_sandbox_evidence_events,
            "actual": evidence_count,
            "minimum": min_sandbox_evidence_events,
        },
        "no_invalid_promotion": {
            "passed": invalid_promotion_count == 0,
            "actual": invalid_promotion_count,
            "maximum": 0,
        },
        "independent_slice_evidence": {
            "passed": (not require_independent_slice_evidence)
            or independent_count > 0,
            "actual": independent_count,
            "required": require_independent_slice_evidence,
        },
        "control_layer_evidence": {
            "passed": control_evidence_passed,
            "memory_loaded": control_layer["memory_loaded"],
            "schedule_loaded": control_layer["schedule_loaded"],
            "failure_memory_family_count": control_layer[
                "failure_memory_family_count"
            ],
            "positive_signal_count": control_layer["positive_signal_count"],
            "required": require_control_layer_evidence,
        },
    }


def _decision(
    *,
    score_improved: bool,
    capability_passed: bool,
    score_progress: dict[str, Any],
    chain: dict[str, Any],
) -> str:
    if score_improved and capability_passed:
        return "passed"
    if score_improved:
        return "score_only"
    if capability_passed:
        return "infrastructure_only"
    metrics = chain.get("metrics")
    exhausted_count = 0
    if isinstance(metrics, dict):
        value = metrics.get("candidate_space_exhausted_count")
        if isinstance(value, int) and not isinstance(value, bool):
            exhausted_count = value
    if exhausted_count > 0 or score_progress.get("stagnation_count", 0) > 0:
        return "stagnated"
    return "failed"


def _next_command(decision: str) -> str:
    if decision == "passed":
        return "start-long-hermes-task1-agent-capability-goal"
    if decision == "score_only":
        return "require-evidence-backed-candidate-generation"
    if decision == "infrastructure_only":
        return "expand-task1-candidate-family-or-data-slice"
    if decision == "stagnated":
        return "expand-hypothesis-lab-candidate-space"
    return "inspect-hermes-task1-agent-capability-artifacts"


def _candidate_payloads(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads = []
    for event in events:
        payload = _payload(event)
        family = _optional_text(payload.get("candidate_family"))
        if family and family != "hermes_runtime_audit_chain":
            payloads.append(payload)
    return payloads


def _has_sandbox_evidence(event: dict[str, Any]) -> bool:
    payload = _payload(event)
    if any(
        _optional_text(payload.get(key))
        for key in (
            "sandbox_evidence_path",
            "evidence_path",
            "evidence_report_path",
            "sandbox_report_path",
        )
    ):
        return True
    evidence = payload.get("evidence")
    if isinstance(evidence, dict) and evidence:
        return True
    source_artifact_path = _optional_text(event.get("source_artifact_path"))
    if source_artifact_path and _is_candidate_evidence_source(event):
        return True
    tags = event.get("tags") if isinstance(event.get("tags"), list) else []
    return "sandbox" in tags and payload.get("validated") is True


def _is_candidate_evidence_source(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "")
    tags = event.get("tags") if isinstance(event.get("tags"), list) else []
    return (
        "candidate" in event_type
        or event_type == "hypothesis_lab_run"
        or "candidate_attempt" in tags
        or "sandbox" in tags
    )


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    file = Path(path)
    if not file.is_file():
        raise HermesTask1AgentCapabilityBenchmarkError(
            f"control-layer artifact missing: {file}"
        )
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesTask1AgentCapabilityBenchmarkError(
            f"control-layer artifact is not valid JSON: {file}"
        ) from exc
    if not isinstance(payload, dict):
        raise HermesTask1AgentCapabilityBenchmarkError(
            "control-layer artifact must be a JSON object"
        )
    return payload


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()
