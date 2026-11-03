"""Creative escalation protocol for autonomous Hypothesis Lab.

Assembles full session context for LLM code generation escalation,
parses generated Python code, and creates agent-generated executor blueprints.
No LLM API calls here — the outer agent runtime handles that.
"""

from __future__ import annotations

import re
from typing import Any

from pde_agent.core.hypothesis_lab_queries import (
    get_best_evidence,
    get_failure_pattern,
    list_candidates,
)
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession
from pde_agent.core.hypothesis_lab_utils import HypothesisLabError


STATE_HINT = "HYPOTHESIS_LAB_CREATIVE_ESCALATION_CONTEXT_BUILT"
EVENT_CREATIVE_ESCALATION_TRIGGERED = "creative_escalation_triggered"


class CreativeEscalationError(RuntimeError):
    pass


def build_creative_escalation_context(
    session: HypothesisLabSession,
    memory: Any | None = None,
) -> dict[str, Any]:
    """Assemble full context for LLM code generation escalation.

    Returns dict with session summary, best evidence, failure patterns,
    blocked families, score trajectory, proven strategies, and a prompt
    template the LLM can consume.
    """
    summary = session.get_state_summary()
    artifact_counts = summary.get("artifact_counts", {})

    # Best evidence so far
    best_evidence = get_best_evidence(session)

    # Failure patterns per family from session
    families_seen = _collect_families_from_session(session)
    failure_patterns: dict[str, Any] = {}
    for family in families_seen:
        fp = get_failure_pattern(session, family)
        if fp.get("failure_count", 0) > 0:
            failure_patterns[family] = fp

    # Blocked families from doom-loop events
    blocked_families = _extract_blocked_families(session)

    # Score trajectory from sandbox events
    score_trajectory = _build_score_trajectory(session)

    # Proven strategies from memory
    proven_strategies: list[dict[str, Any]] = []
    if memory is not None:
        try:
            proven_strategies = memory.get_proven_hypotheses()
        except Exception:
            pass

    # Cross-session failure patterns from memory
    memory_failures: dict[str, Any] = {}
    if memory is not None:
        for family in families_seen:
            try:
                failures = memory.get_similar_failures(family)
                if failures:
                    memory_failures[family] = {
                        "count": len(failures),
                        "patterns": failures[:5],
                    }
            except Exception:
                pass

    prompt_template = _build_prompt_template(
        best_evidence=best_evidence,
        failure_patterns=failure_patterns,
        blocked_families=blocked_families,
        score_trajectory=score_trajectory,
        proven_strategies=proven_strategies,
    )

    context = {
        "session_summary": {
            "session_id": session.session_id,
            "workspace": str(session.workspace),
            "artifact_counts": artifact_counts,
            "total_artifacts": summary.get("total_artifacts", 0),
            "event_count": summary.get("event_count", 0),
            "current_stage": _infer_current_stage(artifact_counts),
        },
        "best_evidence": best_evidence,
        "failure_patterns": failure_patterns,
        "blocked_families": blocked_families,
        "score_trajectory": score_trajectory,
        "proven_strategies": proven_strategies,
        "memory_failures": memory_failures,
        "prompt_template": prompt_template,
    }

    session.emit(
        event_type=EVENT_CREATIVE_ESCALATION_TRIGGERED,
        payload={
            "session_id": session.session_id,
            "blocked_families": blocked_families,
            "families_seen": list(families_seen),
            "best_score": best_evidence.get("score") if best_evidence else None,
        },
        tags=["creative_escalation", "context_built"],
    )

    return context


def parse_generated_executor(response: str) -> dict[str, Any]:
    """Extract Python code from LLM markdown-fenced response.

    Looks for ```python ... ``` blocks. Validates extracted code has
    ``def predict(context, **kwargs)`` or
    ``def predict(context, target=None, reference=None, **kwargs)``.

    Returns {"executor_source": str, "valid": bool, "issues": list[str]}.
    """
    # Extract code blocks
    code_blocks = re.findall(
        r"```python\n(.*?)\n```",
        response,
        re.DOTALL,
    )
    if not code_blocks:
        # Try without language specifier
        code_blocks = re.findall(
            r"```\n(.*?)\n```",
            response,
            re.DOTALL,
        )

    if not code_blocks:
        return {
            "executor_source": "",
            "valid": False,
            "issues": ["no_python_code_block_found"],
        }

    source = code_blocks[0].strip()
    if not source:
        return {
            "executor_source": "",
            "valid": False,
            "issues": ["empty_code_block"],
        }

    # Validate predict function signature
    issues: list[str] = []

    # Check for predict function definition
    predict_pattern = r"def\s+predict\s*\("
    if not re.search(predict_pattern, source):
        issues.append("missing_predict_function")

    # Check first arg is context
    sig_pattern = r"def\s+predict\s*\(\s*context"
    if not re.search(sig_pattern, source):
        issues.append("predict_first_arg_must_be_context")

    # Check for forbidden calls (basic regex audit)
    forbidden = {"open(", "eval(", "exec(", "compile(", "__import__(", "input(", "getattr("}
    for bad in forbidden:
        if bad in source:
            issues.append(f"forbidden_call:{bad.rstrip('(')}")

    if issues:
        return {
            "executor_source": source,
            "valid": False,
            "issues": issues,
        }

    return {
        "executor_source": source,
        "valid": True,
        "issues": [],
    }


def build_agent_generated_blueprint(
    executor_source: str,
    hypothesis_id: str = "agent_generated_hypothesis_001",
    **sandbox_kwargs: Any,
) -> dict[str, Any]:
    """Create an agent_generated_executor blueprint from source code.

    Returns blueprint dict compatible with run_blueprint_sandbox.
    """
    import uuid

    blueprint_id = f"bp_agent_generated_{uuid.uuid4().hex[:8]}"

    blueprint = {
        "blueprint_id": blueprint_id,
        "hypothesis_id": hypothesis_id,
        "candidate_family": "agent_generated_executor",
        "parameter_grid": {
            "candidate_params": [{"executor_source": executor_source}],
        },
        "sandbox": {
            "allowed_executor": "agent_generated_executor",
            "primary_metric": "task1_total_score_proxy",
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 60.0,
            **sandbox_kwargs,
        },
    }

    return blueprint


# -- Internal helpers --


def _collect_families_from_session(session: HypothesisLabSession) -> set[str]:
    """Collect all candidate families seen in sandbox artifacts."""
    families: set[str] = set()
    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        family = best.get("candidate_family")
        if family:
            families.add(str(family))
    return families


def _extract_blocked_families(session: HypothesisLabSession) -> list[str]:
    """Extract blocked families from doom-loop events in session."""
    blocked: set[str] = set()
    events = session.get_events(event_type="doom_loop_detected", limit=50)
    for evt in events:
        payload = evt.get("payload", {})
        families = payload.get("blocked_families", [])
        if isinstance(families, list):
            blocked.update(str(f) for f in families if f)
        # Also check direct blocked_candidate_families key
        alt = payload.get("blocked_candidate_families", [])
        if isinstance(alt, list):
            blocked.update(str(f) for f in alt if f)

    # Also check autonomous step events for doom_loop_triggered
    step_events = session.get_events(
        event_type="autonomous_step_completed",
        limit=100,
    )
    for evt in step_events:
        payload = evt.get("payload", {})
        # The decision payload may have blocked_families
        if isinstance(payload, dict):
            decision = payload.get("decision", {})
            if isinstance(decision, dict):
                bfs = decision.get("blocked_families", [])
                if isinstance(bfs, list):
                    blocked.update(str(f) for f in bfs if f)

    return sorted(blocked)


def _build_score_trajectory(session: HypothesisLabSession) -> list[dict[str, Any]]:
    """Build list of (iteration, score_delta) from session events."""
    trajectory: list[dict[str, Any]] = []

    # Look at sandbox completion events
    events = session.get_events(limit=200)
    iteration = 0
    for evt in events:
        payload = evt.get("payload", {})
        if not isinstance(payload, dict):
            continue

        # Check for score_delta in various payload shapes
        score_delta = payload.get("score_delta")
        if score_delta is None:
            # Check nested decision
            decision = payload.get("decision", {})
            if isinstance(decision, dict):
                score_delta = decision.get("score_delta")

        if isinstance(score_delta, (int, float)) and not isinstance(score_delta, bool):
            iteration += 1
            trajectory.append({
                "iteration": iteration,
                "score_delta": float(score_delta),
                "event_type": evt.get("event_type", "unknown"),
                "timestamp": evt.get("timestamp"),
            })

    # Also extract from sandbox artifacts directly
    if not trajectory:
        for artifact in session.list_artifacts("sandbox"):
            payload = artifact.get("payload", {})
            best = payload.get("best_candidate", {})
            score = best.get("task1_total_score_proxy")
            if isinstance(score, (int, float)):
                iteration += 1
                trajectory.append({
                    "iteration": iteration,
                    "score": float(score),
                    "candidate_family": best.get("candidate_family", "unknown"),
                })

    return trajectory


def _infer_current_stage(artifact_counts: dict[str, int]) -> str:
    """Infer the current pipeline stage from artifact counts."""
    if artifact_counts.get("outcome", 0) > 0:
        return "outcome_recorded"
    if artifact_counts.get("sandbox", 0) > 0:
        return "sandbox_complete"
    if artifact_counts.get("blueprints", 0) > 0:
        return "blueprints_ready"
    if artifact_counts.get("hypotheses", 0) > 0:
        return "hypotheses_ready"
    if artifact_counts.get("knowledge_cards", 0) > 0:
        return "knowledge_cards_ready"
    return "initial"


def _build_prompt_template(
    best_evidence: dict[str, Any] | None,
    failure_patterns: dict[str, Any],
    blocked_families: list[str],
    score_trajectory: list[dict[str, Any]],
    proven_strategies: list[dict[str, Any]],
) -> str:
    """Build a structured prompt template for the LLM."""
    best_score = best_evidence.get("score") if best_evidence else None
    best_family = best_evidence.get("candidate_family") if best_evidence else None

    trajectory_str = "\n".join(
        f"  - iter {t['iteration']}: score_delta={t.get('score_delta', 'N/A')}, "
        f"score={t.get('score', 'N/A')}"
        for t in score_trajectory[-10:]
    ) or "  (no score trajectory yet)"

    failures_str = "\n".join(
        f"  - {family}: {fp.get('failure_count', 0)} failures, "
        f"reasons={fp.get('failure_reasons', {})}"
        for family, fp in failure_patterns.items()
    ) or "  (no failures recorded)"

    proven_str = "\n".join(
        f"  - {p.get('hypothesis', {}).get('name', 'unknown')}: "
        f"status={p.get('hypothesis', {}).get('status', 'unknown')}"
        for p in proven_strategies[:5]
    ) or "  (no proven strategies yet)"

    template = f"""# Creative Escalation: Generate New Prediction Algorithm

You are being asked to invent a novel prediction algorithm for Burgers' equation (Task 1).

## Current Best Result
- Family: {best_family or "none"}
- Score: {best_score or "N/A"}

## Blocked Families (Doom-Loop)
{chr(10).join(f"  - {f}" for f in blocked_families) or "  (none blocked)"}

## Score Trajectory (last 10)
{trajectory_str}

## Failure Patterns
{failures_str}

## Proven Strategies
{proven_str}

## Your Task
Write a Python function `predict(context, target=None, reference=None, **kwargs)` that:
1. Takes `context` as a numpy array of shape `[N, T_context, X]`
2. Returns a numpy array of shape `[N, T_forecast, X]`
3. Uses only `numpy`, `math`, and `typing` imports
4. Avoids file I/O, eval/exec, and other unsafe operations
5. Implements a novel approach not covered by the blocked families above

## Output Format
Wrap your code in a ```python block. Include ONLY the function definition and any helper functions.
"""
    return template
