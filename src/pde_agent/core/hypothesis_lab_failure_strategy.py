"""Prescriptive strategy layer for failure-guided family selection.

Mines cross-session failure records for counterfactual rules and uses them
to recommend which blueprint family to run next in the sandbox stage.
"""

from __future__ import annotations

from typing import Any

from pde_agent.core.hypothesis_lab_queries import list_pending_blueprints
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession


def extract_rules_from_memory(memory: Any) -> list[dict[str, Any]]:
    """Mine cross-session failure records for counterfactual rules.

    A rule is a condition-action pair learned from observing:
    - Family X failed under condition C
    - Family Y succeeded under the same condition C

    Returns rules like:
    [
        {
            "rule_id": "rule_001",
            "condition": {
                "failed_family": "task1_retrieval_refinement",
                "root_cause": "early_shock_smearing",
                "worst_segment": "early",
            },
            "action": "try_family",
            "target": "task1_early_transition_expert",
            "confidence": 0.75,
            "support": 4,
            "supporting_sessions": ["sess_001", ...],
        },
    ]
    """
    rules: list[dict[str, Any]] = []
    if memory is None:
        return rules

    try:
        all_records = list(memory._read_all())
    except Exception:
        return rules

    # Group sessions by session_id
    sessions: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        sid = record.get("session_id")
        if not sid:
            continue
        sessions.setdefault(sid, []).append(record)

    # For each session, find failures and successes
    condition_outcomes: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}

    for sid, records in sessions.items():
        failures: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []

        for record in records:
            rtype = record.get("record_type")
            if rtype == "failure":
                failures.append(record)
            elif rtype == "experiment":
                outcome = record.get("outcome", {})
                cand = outcome.get("candidate", outcome.get("best_candidate", {}))
                score = cand.get("task1_total_score_proxy") or cand.get("score")
                if isinstance(score, (int, float)) and score > 30.0:
                    successes.append(record)
            elif rtype == "candidate":
                cand = record.get("candidate", {})
                score = cand.get("task1_total_score_proxy") or cand.get("score")
                if isinstance(score, (int, float)) and score > 30.0:
                    successes.append(record)

        for failure in failures:
            pattern = failure.get("pattern", {})
            failed_family = failure.get("family", "unknown")
            root_cause = pattern.get("root_cause", pattern.get("reason", "unknown"))
            worst_segment = pattern.get("worst_segment", "unknown")

            condition = (failed_family, root_cause, worst_segment)

            for success in successes:
                succ_cand = _extract_candidate(success)
                succ_family = succ_cand.get("candidate_family", "unknown")
                if succ_family == failed_family:
                    continue

                if condition not in condition_outcomes:
                    condition_outcomes[condition] = {}

                target = succ_family
                if target not in condition_outcomes[condition]:
                    condition_outcomes[condition][target] = {
                        "success_count": 0,
                        "total_count": 0,
                        "sessions": set(),
                    }

                condition_outcomes[condition][target]["success_count"] += 1
                condition_outcomes[condition][target]["total_count"] += 1
                condition_outcomes[condition][target]["sessions"].add(sid)

    rule_id = 0
    for condition, targets in condition_outcomes.items():
        failed_family, root_cause, worst_segment = condition
        for target, stats in targets.items():
            total = stats["total_count"]
            success = stats["success_count"]
            confidence = success / total if total > 0 else 0.0
            if confidence < 0.25:
                continue
            rule_id += 1
            rules.append({
                "rule_id": f"rule_{rule_id:03d}",
                "condition": {
                    "failed_family": failed_family,
                    "root_cause": root_cause,
                    "worst_segment": worst_segment,
                },
                "action": "try_family",
                "target": target,
                "confidence": round(confidence, 2),
                "support": total,
                "supporting_sessions": sorted(stats["sessions"]),
            })

    # Sort by confidence descending, then support descending
    rules.sort(key=lambda r: (r["confidence"], r["support"]), reverse=True)
    return rules


def recommend_family_given_diagnosis(
    diagnosis: dict[str, Any],
    rules: list[dict[str, Any]],
    available_families: set[str],
) -> dict[str, Any] | None:
    """Given current diagnosis, recommend a family from available ones.

    Matches diagnosis against rule conditions and returns the highest-confidence
    matching rule's target family.

    Returns None if no rule matches or target not in available_families.
    """
    matching: list[dict[str, Any]] = []
    for rule in rules:
        if _match_rule_condition(rule, diagnosis):
            target = rule.get("target", "")
            if target in available_families:
                matching.append(rule)

    if not matching:
        return None

    # Sort by confidence descending, then support descending
    matching.sort(key=lambda r: (r.get("confidence", 0.0), r.get("support", 0)), reverse=True)
    best = matching[0]
    return {
        "rule_id": best["rule_id"],
        "target_family": best["target"],
        "confidence": best["confidence"],
        "support": best["support"],
        "reasoning": (
            f"Rule {best['rule_id']} recommends {best['target']} "
            f"(confidence={best['confidence']}, support={best['support']})"
        ),
    }


def score_family_for_context(
    session: HypothesisLabSession,
    family: str,
    rules: list[dict[str, Any]],
) -> float:
    """Score how promising a family is for the current context (0.0-1.0).

    Scoring factors:
    - +0.4 if a high-confidence rule recommends this family for current conditions
    - +0.3 if this family has historically succeeded in similar sessions
    - +0.2 if this family is NOT in blocked/doom-loop list
    - +0.1 if this family has not been tried in this session yet
    - -0.5 if this family is blocked by doom-loop
    """
    score = 0.0

    # Get current diagnosis from session
    diagnosis = _extract_session_diagnosis(session)

    # Factor 1: high-confidence rule match (+0.4)
    for rule in rules:
        if rule.get("target") == family and _match_rule_condition(rule, diagnosis):
            if rule.get("confidence", 0.0) >= 0.5:
                score += 0.4
                break

    # Factor 2: historical success in similar sessions (+0.3)
    if _family_has_historical_success(session, family):
        score += 0.3

    # Factor 3 & 5: doom-loop blocked status (+0.2 / -0.5)
    blocked = _get_blocked_families(session)
    if family in blocked:
        score -= 0.5
    else:
        score += 0.2

    # Factor 4: not tried yet in this session (+0.1)
    if not _family_tried_in_session(session, family):
        score += 0.1

    return max(0.0, min(1.0, score))


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
    pending = list_pending_blueprints(session)
    if not pending:
        return {"action": "report_results", "reasoning": "No pending blueprints"}

    # Extract rules from memory
    rules: list[dict[str, Any]] = []
    if memory is not None:
        try:
            from pde_agent.core.hypothesis_lab_failure_strategy import (
                extract_rules_from_memory,
            )
            rules = extract_rules_from_memory(memory)
        except Exception:
            pass

    # Available families from pending blueprints
    available = {
        p.get("candidate_family", "unknown")
        for p in pending
        if not p.get("sandboxed")
    }

    # Current diagnosis from last failed family (if any)
    diagnosis = _extract_session_diagnosis(session)

    # Try rule-based recommendation first
    rule_rec = None
    if rules and diagnosis:
        rule_rec = recommend_family_given_diagnosis(diagnosis, rules, available)

    # Score all pending families
    family_scores: dict[str, float] = {}
    for p in pending:
        if p.get("sandboxed"):
            continue
        fam = p.get("candidate_family", "unknown")
        if fam not in family_scores:
            family_scores[fam] = score_family_for_context(session, fam, rules)

    # Sort families by score
    sorted_families = sorted(
        family_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Determine best family
    best_family: str | None = None
    best_score = -1.0
    reasoning_parts: list[str] = []

    if rule_rec is not None:
        best_family = rule_rec["target_family"]
        best_score = rule_rec["confidence"]
        reasoning_parts.append(
            f"Rule-based recommendation: {best_family} "
            f"({rule_rec['reasoning']})"
        )
    elif sorted_families:
        best_family, best_score = sorted_families[0]
        reasoning_parts.append(
            f"Score-based recommendation: {best_family} (score={best_score:.2f})"
        )

    if best_family is None:
        # Fallback: first pending blueprint
        for p in pending:
            if not p.get("sandboxed"):
                best_family = p.get("candidate_family", "unknown")
                best_index = p.get("index", 0)
                return {
                    "action": "run_blueprint_sandbox",
                    "recommended_family": best_family,
                    "recommended_index": best_index,
                    "reasoning": "Fallback: first pending blueprint (no rules or scores)",
                    "confidence": 0.0,
                    "alternative_families": [],
                }

    # Find the index of the best pending blueprint for this family
    best_index: int | None = None
    for p in pending:
        if not p.get("sandboxed") and p.get("candidate_family") == best_family:
            best_index = p.get("index")
            break

    if best_index is None:
        # Family not found in pending (shouldn't happen), fallback
        for p in pending:
            if not p.get("sandboxed"):
                best_index = p.get("index", 0)
                best_family = p.get("candidate_family", "unknown")
                break

    alternative_families = [
        fam for fam, _ in sorted_families[1:3]
        if fam != best_family
    ]

    return {
        "action": "run_blueprint_sandbox",
        "recommended_family": best_family,
        "recommended_index": best_index,
        "reasoning": "; ".join(reasoning_parts),
        "confidence": round(best_score, 2),
        "alternative_families": alternative_families,
    }


def _match_rule_condition(rule: dict[str, Any], diagnosis: dict[str, Any]) -> bool:
    """Check if a rule's condition matches the current diagnosis."""
    condition = rule.get("condition", {})
    for key, value in condition.items():
        if diagnosis.get(key) != value:
            return False
    return True


def _extract_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Extract candidate dict from experiment or candidate record."""
    if record.get("record_type") == "experiment":
        outcome = record.get("outcome", {})
        return outcome.get("candidate", outcome.get("best_candidate", {}))
    if record.get("record_type") == "candidate":
        return record.get("candidate", {})
    return {}


def _extract_session_diagnosis(session: HypothesisLabSession) -> dict[str, Any]:
    """Build a diagnosis dict from the most recent sandbox outcome."""
    diagnosis: dict[str, Any] = {}

    # Look at sandbox artifacts for the last failed family
    for artifact in reversed(list(session.list_artifacts("sandbox"))):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        score = best.get("task1_total_score_proxy")
        if isinstance(score, (int, float)) and score < 30.0:
            diagnosis["failed_family"] = best.get("candidate_family", "unknown")
            seg = best.get("segment_score", {})
            if isinstance(seg, dict) and seg:
                # Find worst segment
                worst = min(
                    seg.items(),
                    key=lambda x: x[1] if isinstance(x[1], (int, float)) else float("inf"),
                )
                diagnosis["worst_segment"] = worst[0]
            break

    # Look at outcome artifacts for root cause
    for artifact in reversed(list(session.list_artifacts("outcome"))):
        payload = artifact.get("payload", {})
        pareto = payload.get("pareto_memory", {})
        events = pareto.get("pareto_learning_events", [])
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            outcome = event.get("outcome", "")
            if outcome in ("failed", "rejected", "blocked"):
                diagnosis["root_cause"] = event.get("decision", "unknown")
                break
        if "root_cause" in diagnosis:
            break

    return diagnosis


def _family_has_historical_success(
    session: HypothesisLabSession,
    family: str,
) -> bool:
    """Check if this family has succeeded in any sandbox artifact in this session."""
    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        if best.get("candidate_family") == family:
            score = best.get("task1_total_score_proxy")
            if isinstance(score, (int, float)) and score > 30.0:
                return True
    return False


def _get_blocked_families(session: HypothesisLabSession) -> set[str]:
    """Return families blocked by doom-loop in this session."""
    blocked: list[str] = []
    # Check outcome artifacts for blocked families
    for artifact in session.list_artifacts("outcome"):
        payload = artifact.get("payload", {})
        pareto = payload.get("pareto_memory", {})
        events = pareto.get("pareto_learning_events", [])
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("outcome") == "blocked":
                fam = event.get("candidate_family")
                if fam:
                    blocked.append(fam)
    return set(blocked)


def _family_tried_in_session(session: HypothesisLabSession, family: str) -> bool:
    """Check if this family has been sandboxed in this session."""
    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        if best.get("candidate_family") == family:
            return True
    return False
