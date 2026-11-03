"""Agent query interface for HypothesisLabSession introspection.

Provides high-level query methods that let an agent understand the current
state of a scientific experiment session without knowing file paths or
internal JSON schemas.
"""

from __future__ import annotations

from typing import Any

from pde_agent.core.hypothesis_lab_session import HypothesisLabSession


class QueryError(RuntimeError):
    pass


def list_candidates(
    session: HypothesisLabSession,
    *,
    family: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List candidates from sandbox artifacts.

    Args:
        family: Filter by candidate family (e.g., "task1_retrieval_refinement")
        status: Filter by status (e.g., "promoted", "rejected", "evidence")

    Returns:
        List of candidate summaries with id, family, score, and status.
    """
    candidates: list[dict[str, Any]] = []

    # Search sandbox artifacts for candidates
    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        if not isinstance(best, dict):
            continue

        candidate_family = best.get("candidate_family", "unknown")
        if family is not None and candidate_family != family:
            continue

        entry = {
            "candidate_id": best.get("candidate_id"),
            "candidate_family": candidate_family,
            "score": best.get("task1_total_score_proxy"),
            "segment_score": best.get("segment_score"),
            "artifact_type": "sandbox",
            "artifact_id": artifact.get("artifact_id"),
        }

        # Infer status from outcome artifacts
        entry["status"] = _infer_candidate_status(session, entry["candidate_id"])
        if status is not None and entry["status"] != status:
            continue

        candidates.append(entry)

    # Sort by score descending
    candidates.sort(key=lambda c: c.get("score") or 0.0, reverse=True)
    return candidates


def get_best_evidence(
    session: HypothesisLabSession,
    *,
    metric: str = "score",
) -> dict[str, Any] | None:
    """Get the best experimental evidence from sandbox artifacts.

    Args:
        metric: Metric to optimize ("score" for task1_total_score_proxy)

    Returns:
        Best evidence summary or None if no sandbox results exist.
    """
    candidates = list_candidates(session)
    if not candidates:
        return None

    best = candidates[0]
    sandbox = session.get_artifact("sandbox", best["artifact_id"])
    if sandbox is None:
        return best

    return {
        "candidate_id": best["candidate_id"],
        "candidate_family": best["candidate_family"],
        "score": best["score"],
        "segment_score": best["segment_score"],
        "sandbox_artifact_id": best["artifact_id"],
        "sandbox": sandbox,
    }


def get_failure_pattern(
    session: HypothesisLabSession,
    family: str,
) -> dict[str, Any]:
    """Analyze failure patterns for a candidate family.

    Args:
        family: Candidate family to analyze

    Returns:
        Summary of failures including count, common reasons, and score stats.
    """
    failures: list[dict[str, Any]] = []
    scores: list[float] = []

    # Check outcome artifacts for this family
    for artifact in session.list_artifacts("outcome"):
        payload = artifact.get("payload", {})
        pareto = payload.get("pareto_memory", {})
        events = pareto.get("pareto_learning_events", [])
        for event in events:
            if not isinstance(event, dict):
                continue
            evt_family = event.get("candidate_family")
            if evt_family != family:
                continue
            outcome = event.get("outcome", "unknown")
            if outcome in {"failed", "rejected", "blocked"}:
                failures.append(event)
            score_delta = event.get("score_delta")
            if isinstance(score_delta, (int, float)) and not isinstance(score_delta, bool):
                scores.append(float(score_delta))

    # Also check sandbox artifacts for low scores
    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        if best.get("candidate_family") != family:
            continue
        score = best.get("task1_total_score_proxy")
        if isinstance(score, (int, float)) and score < 30.0:
            scores.append(float(score))

    reasons: dict[str, int] = {}
    for failure in failures:
        decision = failure.get("decision", "unknown")
        reasons[decision] = reasons.get(decision, 0) + 1
        # Also check guardrail regressions
        for reg in failure.get("guardrail_regressions", []):
            if isinstance(reg, str):
                reasons[reg] = reasons.get(reg, 0) + 1

    # Aggregate failure taxonomy from sandbox post_mortem
    failure_types: dict[str, int] = {}
    root_causes: list[str] = []
    from pde_agent.core.hypothesis_lab_failure_taxonomy import classify_failure

    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        if best.get("candidate_family") != family:
            continue
        classification = classify_failure(payload)
        ptype = classification["primary_type"]
        if ptype != "NONE":
            failure_types[ptype] = failure_types.get(ptype, 0) + 1
        rch = classification["root_cause_hint"]
        if rch and rch != "ok" and rch not in root_causes:
            root_causes.append(rch)

    return {
        "family": family,
        "failure_count": len(failures),
        "failure_reasons": reasons,
        "failure_types": failure_types,
        "root_causes": root_causes,
        "score_stats": {
            "count": len(scores),
            "best": max(scores) if scores else None,
            "worst": min(scores) if scores else None,
            "mean": sum(scores) / len(scores) if scores else None,
        },
        "recommendation": _failure_recommendation(family, reasons, scores),
    }


def recommend_next_action(session: HypothesisLabSession) -> dict[str, Any]:
    """Recommend the next action based on current session state.

    Returns:
        Recommendation with action, reasoning, and required inputs.
    """
    summary = session.get_state_summary()
    artifact_counts = summary.get("artifact_counts", {})

    # Determine current pipeline stage
    if artifact_counts.get("sandbox", 0) > 0:
        # Have sandbox results - check if we need more or should record outcome
        if artifact_counts.get("outcome", 0) == 0:
            return {
                "action": "record_sandbox_outcome",
                "reasoning": "Sandbox completed but outcome not recorded yet",
                "required_inputs": ["sandbox artifact"],
            }
        # Have outcome - maybe run another blueprint
        pending = list_pending_blueprints(session)
        if pending:
            sandboxed_count = sum(1 for p in pending if p.get("sandboxed"))
            total_count = len(pending)
            if sandboxed_count < total_count:
                return {
                    "action": "run_next_blueprint_sandbox",
                    "reasoning": f"Only {sandboxed_count}/{total_count} current blueprints tested",
                    "required_inputs": ["context_path", "target_path"],
                }
        return {
            "action": "report_results",
            "reasoning": "Pipeline complete",
            "required_inputs": [],
        }

    if artifact_counts.get("blueprints", 0) > 0:
        return {
            "action": "run_blueprint_sandbox",
            "reasoning": "Blueprints ready but not yet tested",
            "required_inputs": ["context_path", "target_path"],
        }

    if artifact_counts.get("hypotheses", 0) > 0:
        return {
            "action": "design_algorithm_blueprints",
            "reasoning": "Hypotheses ready but blueprints not yet designed",
            "required_inputs": [],
        }

    if artifact_counts.get("knowledge_cards", 0) > 0:
        return {
            "action": "generate_hypotheses",
            "reasoning": "Knowledge cards ready but hypotheses not yet generated",
            "required_inputs": [],
        }

    return {
        "action": "build_knowledge_cards",
        "reasoning": "No artifacts found - start from beginning",
        "required_inputs": ["manual_report_paths"],
    }


def list_pending_blueprints(session: HypothesisLabSession) -> list[dict[str, Any]]:
    """List blueprints that have not yet been sandboxed.

    Returns:
        List of blueprints with their index and sandbox status.
    """
    blueprints_artifact = session.get_latest_artifact("blueprints")
    if blueprints_artifact is None:
        return []

    blueprint_list = blueprints_artifact.get("payload", {}).get("blueprints", [])

    # Find which blueprint_ids have been sandboxed
    sandboxed_ids: set[str] = set()
    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        bp_id = payload.get("blueprint_id")
        if bp_id:
            sandboxed_ids.add(bp_id)

    pending: list[dict[str, Any]] = []
    for idx, bp in enumerate(blueprint_list):
        if not isinstance(bp, dict):
            continue
        bp_id = bp.get("blueprint_id", f"index_{idx}")
        pending.append({
            "index": idx,
            "blueprint_id": bp_id,
            "candidate_family": bp.get("candidate_family", "unknown"),
            "hypothesis_id": bp.get("hypothesis_id"),
            "sandboxed": bp_id in sandboxed_ids,
        })

    return pending


def get_pareto_front(session: HypothesisLabSession) -> list[dict[str, Any]]:
    """Get the current Pareto front from all sandbox results.

    Returns:
        List of non-dominated candidates (score vs runtime tradeoff).
    """
    points: list[dict[str, Any]] = []
    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        if not isinstance(best, dict):
            continue
        score = best.get("task1_total_score_proxy")
        inference_seconds = best.get("inference_seconds")
        if score is None:
            continue
        points.append({
            "candidate_id": best.get("candidate_id"),
            "candidate_family": best.get("candidate_family"),
            "score": float(score),
            "runtime": float(inference_seconds) if inference_seconds is not None else None,
            "segment_score": best.get("segment_score"),
            "sandbox_artifact_id": artifact.get("artifact_id"),
        })

    if not points:
        return []

    # Simple Pareto front: not dominated in score (higher is better)
    # A point is dominated if another point has both higher score and lower runtime
    pareto: list[dict[str, Any]] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            q_better_score = q["score"] > p["score"]
            q_better_runtime = (
                q["runtime"] is not None
                and p["runtime"] is not None
                and q["runtime"] < p["runtime"]
            )
            # Dominated if another point is better in both dimensions
            if q_better_score and (q_better_runtime or p["runtime"] is None):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    pareto.sort(key=lambda x: x["score"], reverse=True)
    return pareto


def summarize_recent_events(
    session: HypothesisLabSession,
    *,
    n: int = 20,
    event_type: str | None = None,
) -> dict[str, Any]:
    """Summarize recent events for agent context.

    Args:
        n: Number of recent events to include
        event_type: Optional filter by event type

    Returns:
        Summary with event list and counts.
    """
    events = session.get_events(event_type=event_type, limit=n)
    counts: dict[str, int] = {}
    for evt in events:
        et = evt.get("event_type", "unknown")
        counts[et] = counts.get(et, 0) + 1

    return {
        "session_id": session.session_id,
        "event_count": len(events),
        "event_type_filter": event_type,
        "event_counts": counts,
        "events": events,
    }


# -- Helpers --


def _infer_candidate_status(
    session: HypothesisLabSession,
    candidate_id: str | None,
) -> str:
    if candidate_id is None:
        return "unknown"

    # Check outcome artifacts
    for artifact in session.list_artifacts("outcome"):
        payload = artifact.get("payload", {})
        outcome = payload.get("candidate_outcome", {})
        if outcome.get("candidate_id") == candidate_id:
            return str(outcome.get("status", "recorded"))

    # Check if promoted in strategy growth
    for artifact in session.list_artifacts("report"):
        payload = artifact.get("payload", {})
        if payload.get("state_hint") == "STRATEGY_PROPOSAL_RECORDED":
            return "promoted"

    return "evidence"


def _failure_recommendation(
    family: str,
    reasons: dict[str, int],
    scores: list[float],
) -> str:
    if not reasons:
        if not scores:
            return "no_data"
        avg = sum(scores) / len(scores)
        if avg < 20:
            return "family_underperforms"
        return "needs_more_evidence"

    top_reason = max(reasons, key=lambda r: reasons[r])
    if top_reason in {"target_leakage_risk", "context_mismatch"}:
        return "fix_safety"
    if top_reason in {"score_regressed", "ablation_missing_or_weak"}:
        return "refine_params"
    if top_reason in {"runtime_budget_exceeded"}:
        return "reduce_complexity"
    if top_reason in {"unsupported_executor", "requires_external_code"}:
        return "design_executor"
    return "investigate"


def recommend_family_for_sandbox(
    session: HypothesisLabSession,
    memory: Any = None,
) -> dict[str, Any]:
    """Recommend which blueprint family to run next in sandbox.

    Returns:
        {
            "action": "run_blueprint_sandbox",
            "recommended_family": "task1_early_transition_expert",
            "recommended_index": 2,
            "reasoning": "...",
            "confidence": 0.75,
            "alternative_families": [...],
        }
    """
    from pde_agent.core.hypothesis_lab_failure_strategy import (
        recommend_family_for_sandbox as _recommend_family_for_sandbox,
    )

    return _recommend_family_for_sandbox(session, memory)
