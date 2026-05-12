"""KPI rollups for Hypothesis Lab trace trajectories."""


import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_trace import read_hypothesis_lab_trace


STATE_HINT = "HYPOTHESIS_KPI_REPORT_BUILT"
NEGATIVE_OUTCOMES = {"failed", "rejected", "blocked", "deprioritize", "deprioritized"}
REPEATED_FAILURE_DEPRIORITIZE_THRESHOLD = 2


class HypothesisKpiError(RuntimeError):
    pass


def build_hypothesis_kpi_report(
    *,
    trace_path: str | Path,
    output_path: str | Path | None,
    recent_window: int = 50,
    force: bool,
) -> dict[str, Any]:
    if recent_window < 1:
        raise HypothesisKpiError("recent_window must be positive")
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    events = read_hypothesis_lab_trace(trace_path)
    recent_events = events[-recent_window:]
    observations = [_observation(event) for event in recent_events]

    outcome_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    family_score_deltas: dict[str, list[float]] = {}
    family_failed_counts: dict[str, int] = {}
    deprioritized_candidate_families: list[str] = []
    deprioritization_reason_codes: dict[str, list[str]] = {}
    deprioritization_reason_codes_by_candidate: dict[str, list[str]] = {}
    deprioritized_algorithm_candidates_by_id: dict[str, dict[str, Any]] = {}
    cluster_saturation_signals: list[dict[str, Any]] = []
    saturated_candidate_families: list[str] = []
    persistent_search_guidance: list[dict[str, Any]] = []
    search_guidance_action_counts: dict[str, int] = {}
    audit_repair_error_counts: dict[str, int] = {}
    audit_repair_consumed_error_counts: dict[str, int] = {}
    repair_nonconvergence_errors: list[str] = []
    repair_convergence_errors: list[str] = []
    audit_repair_converged_error_counts: dict[str, int] = {}
    chain_learning_signal_counts: dict[str, int] = {}
    auto_repair_decision_counts: dict[str, int] = {}
    auto_repair_event_count = 0
    auto_repair_passed_count = 0
    auto_repair_memory_bridge_success_count = 0
    auto_repair_smoke_executed_count = 0
    auto_repair_repair_policy_action_count = 0
    auto_repair_batch_memory_constraint_action_count = 0
    chain_event_count = 0
    chain_failed_count = 0
    chain_audit_failure_count = 0
    for obs in observations:
        outcome = str(obs.get("outcome") or "observed")
        if obs.get("algorithm_candidate_id"):
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if obs.get("event_type") == "runtime_audit_chain":
            chain_event_count += 1
            if outcome == "failed":
                chain_failed_count += 1
            failed_audit_count = obs.get("failed_audit_count")
            if isinstance(failed_audit_count, int):
                chain_audit_failure_count += failed_audit_count
            auto_repair_decision = obs.get("auto_repair_decision")
            if isinstance(auto_repair_decision, str) and auto_repair_decision:
                auto_repair_event_count += 1
                auto_repair_decision_counts[auto_repair_decision] = (
                    auto_repair_decision_counts.get(auto_repair_decision, 0) + 1
                )
                if auto_repair_decision == "passed":
                    auto_repair_passed_count += 1
            auto_repair_metrics = obs.get("auto_repair_metrics")
            if isinstance(auto_repair_metrics, dict):
                if auto_repair_metrics.get("memory_bridge_passed") is True:
                    auto_repair_memory_bridge_success_count += 1
                if auto_repair_metrics.get("smoke_executed") is True:
                    auto_repair_smoke_executed_count += 1
                auto_repair_repair_policy_action_count += _int_metric(
                    auto_repair_metrics.get("repair_policy_recommended_action_count")
                )
                auto_repair_batch_memory_constraint_action_count += _int_metric(
                    auto_repair_metrics.get("batch_memory_constraint_action_count")
                )
            for signal in obs.get("chain_learning_signals", []):
                if isinstance(signal, str) and signal:
                    chain_learning_signal_counts[signal] = (
                        chain_learning_signal_counts.get(signal, 0) + 1
                    )
            for audit_error in obs.get("audit_errors", []):
                if isinstance(audit_error, str) and audit_error:
                    audit_repair_error_counts[audit_error] = (
                        audit_repair_error_counts.get(audit_error, 0) + 1
                    )
            for audit_error in obs.get("repair_nonconvergence_errors", []):
                if isinstance(audit_error, str) and audit_error:
                    _append_unique(repair_nonconvergence_errors, audit_error)
            for audit_error in obs.get("repair_convergence_errors", []):
                if isinstance(audit_error, str) and audit_error:
                    _append_unique(repair_convergence_errors, audit_error)
                    audit_repair_converged_error_counts[audit_error] = (
                        audit_repair_converged_error_counts.get(audit_error, 0) + 1
                    )
        for repair_task in obs.get("audit_repair_tasks", []):
            if not isinstance(repair_task, dict):
                continue
            audit_error = repair_task.get("audit_error")
            if isinstance(audit_error, str) and audit_error:
                audit_repair_consumed_error_counts[audit_error] = (
                    audit_repair_consumed_error_counts.get(audit_error, 0) + 1
                )
        for signal in obs.get("cluster_saturation_signals", []):
            if isinstance(signal, dict) and signal not in cluster_saturation_signals:
                cluster_saturation_signals.append(signal)
        for family in obs.get("saturated_candidate_families", []):
            if (
                isinstance(family, str)
                and family
                and family not in saturated_candidate_families
            ):
                saturated_candidate_families.append(family)
        for guidance in obs.get("persistent_search_guidance", []):
            if isinstance(guidance, dict) and guidance not in persistent_search_guidance:
                persistent_search_guidance.append(guidance)
                action = guidance.get("recommended_action")
                if isinstance(action, str) and action:
                    search_guidance_action_counts[action] = (
                        search_guidance_action_counts.get(action, 0) + 1
                    )
        family = obs.get("candidate_family")
        if isinstance(family, str) and family:
            family_counts[family] = family_counts.get(family, 0) + 1
            if outcome in NEGATIVE_OUTCOMES:
                family_failed_counts[family] = family_failed_counts.get(family, 0) + 1
            if outcome in {"deprioritize", "deprioritized"} and family not in (
                deprioritized_candidate_families
            ):
                deprioritized_candidate_families.append(family)
                _append_reason_code(
                    deprioritization_reason_codes,
                    family=family,
                    reason_code="deprioritized_completion_trace",
                )
            if outcome in {"deprioritize", "deprioritized"}:
                candidate_id = obs.get("algorithm_candidate_id")
                if isinstance(candidate_id, str) and candidate_id:
                    _append_reason_code_by_candidate(
                        deprioritization_reason_codes_by_candidate,
                        candidate_id=candidate_id,
                        reason_code="deprioritized_completion_trace",
                    )
                for reason_code in obs.get("reason_codes", []):
                    if isinstance(reason_code, str) and reason_code:
                        _append_reason_code(
                            deprioritization_reason_codes,
                            family=family,
                            reason_code=reason_code,
                        )
                        candidate_id = obs.get("algorithm_candidate_id")
                        if isinstance(candidate_id, str) and candidate_id:
                            _append_reason_code_by_candidate(
                                deprioritization_reason_codes_by_candidate,
                                candidate_id=candidate_id,
                                reason_code=reason_code,
                            )
                candidate_id = obs.get("algorithm_candidate_id")
                if isinstance(candidate_id, str) and candidate_id:
                    deprioritized_algorithm_candidates_by_id[candidate_id] = {
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": family,
                    }
            score_delta = obs.get("score_delta")
            if isinstance(score_delta, float):
                family_score_deltas.setdefault(family, []).append(score_delta)

    for family, failed_count in family_failed_counts.items():
        values = family_score_deltas.get(family, [])
        if (
            failed_count >= REPEATED_FAILURE_DEPRIORITIZE_THRESHOLD
            and values
            and max(values) <= 0.0
        ):
            if family not in deprioritized_candidate_families:
                deprioritized_candidate_families.append(family)
            _append_reason_code(
                deprioritization_reason_codes,
                family=family,
                reason_code="repeated_failed_family",
            )

    candidate_count = sum(
        1 for obs in observations if isinstance(obs.get("algorithm_candidate_id"), str)
    )
    promoted = outcome_counts.get("promoted", 0)
    failed = sum(outcome_counts.get(key, 0) for key in NEGATIVE_OUTCOMES)
    best_family = _best_family_by_score_delta(family_score_deltas)
    score_deltas = [
        obs["score_delta"]
        for obs in observations
        if isinstance(obs.get("score_delta"), float)
    ]
    active_audit_repair_errors = _active_count_keys(
        produced_counts=audit_repair_error_counts,
        consumed_counts=audit_repair_consumed_error_counts,
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "trace_path": str(trace_path),
        "recent_window": recent_window,
        "metrics": {
            "event_count": len(recent_events),
            "candidate_count": candidate_count,
            "promoted_count": promoted,
            "failed_count": failed,
            "promotion_rate": promoted / candidate_count if candidate_count else 0.0,
            "failure_rate": failed / candidate_count if candidate_count else 0.0,
            "distinct_candidate_family_count": len(family_counts),
            "chain_event_count": chain_event_count,
            "chain_failed_count": chain_failed_count,
            "chain_audit_failure_count": chain_audit_failure_count,
            "auto_repair_event_count": auto_repair_event_count,
            "auto_repair_passed_count": auto_repair_passed_count,
            "auto_repair_memory_bridge_success_count": (
                auto_repair_memory_bridge_success_count
            ),
            "cluster_saturation_signal_count": len(cluster_saturation_signals),
            "saturated_candidate_family_count": len(saturated_candidate_families),
            "persistent_search_guidance_count": len(persistent_search_guidance),
            "audit_repair_task_count": sum(audit_repair_error_counts.values()),
            "audit_repair_consumed_count": sum(
                audit_repair_consumed_error_counts.values()
            ),
            "active_audit_repair_error_count": len(active_audit_repair_errors),
            "repair_nonconvergence_error_count": len(repair_nonconvergence_errors),
            "repair_convergence_error_count": len(repair_convergence_errors),
        },
        "outcome_counts": outcome_counts,
        "chain_learning_signal_counts": chain_learning_signal_counts,
        "auto_repair_decision_counts": auto_repair_decision_counts,
        "auto_repair_memory_bridge": {
            "success_count": auto_repair_memory_bridge_success_count,
            "smoke_executed_count": auto_repair_smoke_executed_count,
            "repair_policy_recommended_action_count": (
                auto_repair_repair_policy_action_count
            ),
            "batch_memory_constraint_action_count": (
                auto_repair_batch_memory_constraint_action_count
            ),
        },
        "audit_repair_error_counts": audit_repair_error_counts,
        "audit_repair_consumed_error_counts": audit_repair_consumed_error_counts,
        "active_audit_repair_errors": active_audit_repair_errors,
        "repair_nonconvergence_errors": sorted(repair_nonconvergence_errors),
        "repair_convergence_errors": sorted(repair_convergence_errors),
        "audit_repair_converged_error_counts": audit_repair_converged_error_counts,
        "cluster_saturation_signals": cluster_saturation_signals,
        "saturated_candidate_families": sorted(saturated_candidate_families),
        "persistent_search_guidance": persistent_search_guidance,
        "search_guidance_action_counts": search_guidance_action_counts,
        "deprioritized_candidate_families": sorted(deprioritized_candidate_families),
        "deprioritization_reason_codes": {
            family: sorted(set(reason_codes))
            for family, reason_codes in sorted(
                deprioritization_reason_codes.items()
            )
        },
        "deprioritization_reason_codes_by_candidate": {
            candidate_id: sorted(set(reason_codes))
            for candidate_id, reason_codes in sorted(
                deprioritization_reason_codes_by_candidate.items()
            )
        },
        "deprioritized_algorithm_candidates": [
            {
                **candidate,
                "reason_codes": sorted(
                    set(
                        deprioritization_reason_codes_by_candidate.get(
                            str(candidate["algorithm_candidate_id"]),
                            [],
                        )
                    )
                ),
            }
            for candidate in sorted(
                deprioritized_algorithm_candidates_by_id.values(),
                key=lambda item: str(item["algorithm_candidate_id"]),
            )
        ],
        "candidate_family_counts": family_counts,
        "candidate_family_score_summary": _candidate_family_score_summary(
            family_counts=family_counts,
            family_failed_counts=family_failed_counts,
            family_score_deltas=family_score_deltas,
        ),
        "best_family_by_score_delta": best_family,
        "trajectory": _trajectory(score_deltas),
        "next_command": "detect-hypothesis-doom-loop",
    }
    _write_optional_json(output_file, report)
    return report


def _append_reason_code(
    values: dict[str, list[str]],
    *,
    family: str,
    reason_code: str,
) -> None:
    values.setdefault(family, []).append(reason_code)


def _append_reason_code_by_candidate(
    values: dict[str, list[str]],
    *,
    candidate_id: str,
    reason_code: str,
) -> None:
    values.setdefault(candidate_id, []).append(reason_code)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _active_count_keys(
    *,
    produced_counts: dict[str, int],
    consumed_counts: dict[str, int],
) -> list[str]:
    active = []
    for key, produced_count in produced_counts.items():
        if produced_count > consumed_counts.get(key, 0):
            active.append(key)
    return sorted(active)


def _observation(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    score_delta = payload.get("score_delta")
    signals = payload.get("chain_learning_signals")
    cluster_saturation_signals = payload.get("cluster_saturation_signals")
    saturated_candidate_families = payload.get("saturated_candidate_families")
    persistent_search_guidance = payload.get("persistent_search_guidance")
    audit_errors = payload.get("audit_errors")
    audit_repair_tasks = payload.get("audit_repair_tasks")
    repair_convergence = payload.get("repair_convergence")
    return {
        "event_type": event.get("event_type"),
        "algorithm_candidate_id": payload.get("algorithm_candidate_id"),
        "candidate_family": payload.get("candidate_family"),
        "outcome": payload.get("outcome"),
        "failed_audit_count": payload.get("failed_audit_count"),
        "audit_errors": audit_errors if isinstance(audit_errors, list) else [],
        "audit_repair_tasks": audit_repair_tasks
        if isinstance(audit_repair_tasks, list)
        else [],
        "repair_nonconvergence_errors": _repair_nonconvergence_errors(
            repair_convergence
        ),
        "repair_convergence_errors": _repair_convergence_errors(
            repair_convergence
        ),
        "auto_repair_decision": payload.get("auto_repair_decision"),
        "auto_repair_metrics": payload.get("auto_repair_metrics")
        if isinstance(payload.get("auto_repair_metrics"), dict)
        else {},
        "chain_learning_signals": signals if isinstance(signals, list) else [],
        "cluster_saturation_signals": cluster_saturation_signals
        if isinstance(cluster_saturation_signals, list)
        else [],
        "saturated_candidate_families": saturated_candidate_families
        if isinstance(saturated_candidate_families, list)
        else [],
        "persistent_search_guidance": persistent_search_guidance
        if isinstance(persistent_search_guidance, list)
        else [],
        "reason_codes": payload.get("reason_codes")
        if isinstance(payload.get("reason_codes"), list)
        else [],
        "score_delta": float(score_delta)
        if isinstance(score_delta, int | float) and not isinstance(score_delta, bool)
        else None,
    }


def _repair_nonconvergence_errors(value: Any) -> list[str]:
    if not isinstance(value, dict) or value.get("decision") != "nonconverged":
        return []
    repeated = value.get("repeated_audit_errors")
    if not isinstance(repeated, list):
        return []
    return [item for item in repeated if isinstance(item, str) and item]


def _repair_convergence_errors(value: Any) -> list[str]:
    if not isinstance(value, dict) or value.get("decision") != "converged":
        return []
    input_errors = value.get("input_repair_errors")
    if not isinstance(input_errors, list):
        return []
    return [item for item in input_errors if isinstance(item, str) and item]


def _int_metric(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    return 0


def _best_family_by_score_delta(
    family_score_deltas: dict[str, list[float]],
) -> dict[str, Any] | None:
    if not family_score_deltas:
        return None
    summaries = []
    for family, values in family_score_deltas.items():
        summaries.append(
            {
                "candidate_family": family,
                "count": len(values),
                "best_score_delta": max(values),
                "mean_score_delta": sum(values) / len(values),
            }
        )
    return max(summaries, key=lambda item: item["best_score_delta"])


def _candidate_family_score_summary(
    *,
    family_counts: dict[str, int],
    family_failed_counts: dict[str, int],
    family_score_deltas: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    summary = {}
    for family, count in family_counts.items():
        values = family_score_deltas.get(family, [])
        summary[family] = {
            "count": count,
            "failed_count": family_failed_counts.get(family, 0),
            "best_score_delta": max(values) if values else None,
            "mean_score_delta": sum(values) / len(values) if values else None,
        }
    return summary


def _trajectory(score_deltas: list[float]) -> dict[str, Any]:
    if not score_deltas:
        return {"direction": "unknown", "mean_score_delta": None}
    positive = sum(1 for value in score_deltas if value > 0.0)
    negative = sum(1 for value in score_deltas if value < 0.0)
    if positive and negative:
        direction = "mixed"
    elif positive:
        direction = "improving"
    elif negative:
        direction = "regressing"
    else:
        direction = "flat"
    return {
        "direction": direction,
        "mean_score_delta": sum(score_deltas) / len(score_deltas),
        "best_score_delta": max(score_deltas),
        "worst_score_delta": min(score_deltas),
    }


def _guard_output(path: Path | None, *, force: bool) -> None:
    if path is not None and path.exists() and not force:
        raise HypothesisKpiError(f"output already exists: {path}")


def _write_optional_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
