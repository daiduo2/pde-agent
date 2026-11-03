"""Action dispatcher for Hypothesis Lab autonomous loop.

Maps action names to primitive function calls with validation,
error handling, and event emission.
"""

from __future__ import annotations

import traceback
from typing import Any

from pde_agent.core.hypothesis_lab_events import (
    EVENT_STAGE_FAILED,
    EVENT_STAGE_STARTED,
)
from pde_agent.core.hypothesis_lab_primitives import (
    PrimitiveError,
    build_knowledge_cards_primitive,
    design_algorithm_blueprints_primitive,
    generate_hypotheses_primitive,
    record_sandbox_outcome_primitive,
    run_blueprint_sandbox_primitive,
)
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession


class ActionDispatchError(RuntimeError):
    pass


def _generate_new_executor_task(
    session: HypothesisLabSession,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a formal creative-escalation task package for external Claude.

    Instead of calling an LLM API directly, this writes a structured task
    file that can be consumed by:
      - `claude -p < task_file.md`  (Claude CLI)
      - Claude SDK  (claude_agent_sdk.query)
      - The current Claude Code session

    The task file contains the full prompt, session context, and output schema.
    After an external Claude generates code, call
    `ingest_creative_escalation_response(session, response_text)` to continue.
    """
    import json
    import time
    from datetime import datetime, timezone

    # Extract creative context (passed from AutonomousHypothesisLab._execute)
    creative_context = kwargs.get("creative_context", {})
    prompt_template = creative_context.get("prompt_template", "")

    # Build formal task package
    task_package = {
        "schema_version": 1,
        "task_type": "creative_escalation",
        "task_id": f"ce_{session.session_id}_{int(time.time())}",
        "session_id": session.session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_external_claude",
        "input": {
            "prompt": prompt_template,
            "session_summary": creative_context.get("session_summary", {}),
            "best_evidence": creative_context.get("best_evidence"),
            "failure_patterns": creative_context.get("failure_patterns", {}),
            "blocked_families": creative_context.get("blocked_families", []),
            "score_trajectory": creative_context.get("score_trajectory", []),
        },
        "expected_output": {
            "format": "markdown_fenced_python",
            "required_function": "predict(context, target=None, reference=None, **kwargs)",
            "return_type": "numpy.ndarray of shape [N, T_forecast, X]",
            "allowed_imports": ["numpy", "math", "typing"],
            "forbidden_calls": ["open", "eval", "exec", "compile", "__import__", "input", "getattr"],
            "audit_required": True,
        },
        "next_step_after_response": {
            "action": "ingest_creative_escalation_response",
            "description": "Call ingest_creative_escalation_response(session, response_text) with the Claude-generated code.",
        },
    }

    # Write task package to session workspace
    task_path = session.workspace / "creative_escalation_task.json"
    task_path.write_text(json.dumps(task_package, indent=2, ensure_ascii=False))

    # Also write a human-readable markdown version
    md_path = session.workspace / "creative_escalation_task.md"
    md_content = _build_creative_escalation_markdown(task_package)
    md_path.write_text(md_content)

    return {
        "action": "generate_new_executor",
        "status": "awaiting_external_claude",
        "message": (
            "Creative escalation task prepared. "
            f"Invoke external Claude with: claude -p < {md_path} "
            f"Then call ingest_creative_escalation_response(session, '<response>')"
        ),
        "task_paths": {
            "json": str(task_path),
            "markdown": str(md_path),
        },
        "task_id": task_package["task_id"],
    }


def _build_creative_escalation_markdown(task_package: dict[str, Any]) -> str:
    """Build human-readable markdown task for Claude CLI consumption."""
    inp = task_package["input"]
    out = task_package["expected_output"]

    lines = [
        "# Creative Escalation Task: Generate New PDE Prediction Algorithm",
        "",
        f"Task ID: `{task_package['task_id']}`",
        f"Session: `{task_package['session_id']}`",
        "",
        "## Context",
        "",
    ]

    summary = inp.get("session_summary", {})
    if summary:
        lines.append("### Session Summary")
        lines.append(f"- Total artifacts: {summary.get('total_artifacts', 0)}")
        lines.append(f"- Event count: {summary.get('event_count', 0)}")
        lines.append(f"- Current stage: {summary.get('current_stage', 'unknown')}")
        lines.append("")

    best = inp.get("best_evidence")
    if best:
        lines.append("### Best Evidence So Far")
        lines.append(f"- Family: `{best.get('candidate_family', 'unknown')}`")
        lines.append(f"- Score: `{best.get('score', 'N/A')}`")
        lines.append(f"- Segment score: `{best.get('segment_score', 'N/A')}`")
        lines.append("")

    blocked = inp.get("blocked_families", [])
    if blocked:
        lines.append("### Blocked Families (Doom-Loop)")
        for f in blocked:
            lines.append(f"- `{f}`")
        lines.append("")

    failures = inp.get("failure_patterns", {})
    if failures:
        lines.append("### Failure Patterns")
        for family, fp in failures.items():
            lines.append(f"- `{family}`: {fp.get('failure_count', 0)} failures")
        lines.append("")

    trajectory = inp.get("score_trajectory", [])
    if trajectory:
        lines.append("### Score Trajectory")
        for t in trajectory[-5:]:
            lines.append(f"- iter {t.get('iteration')}: score={t.get('score', 'N/A')}, delta={t.get('score_delta', 'N/A')}")
        lines.append("")

    lines.extend([
        "## Prompt",
        "",
        "```",
        inp.get("prompt", "(no prompt provided)"),
        "```",
        "",
        "## Expected Output",
        "",
        f"Write a Python function `{out['required_function']}` that:",
        "1. Takes `context` as a numpy array of shape `[N, T_context, X]`",
        "2. Returns a numpy array of shape `[N, T_forecast, X]`",
        f"3. Uses only these imports: {', '.join(f'`{m}`' for m in out['allowed_imports'])}",
        f"4. Avoids: {', '.join(f'`{c}`' for c in out['forbidden_calls'])}",
        "",
        "Wrap your code in a ````python` block. Include ONLY the function definition and helper functions.",
        "",
    ])

    return "\n".join(lines)


def ingest_creative_escalation_response(
    session: HypothesisLabSession,
    response_text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Ingest a Claude-generated code response into the session.

    Parses the response, runs AST audit, builds an agent_generated_executor
    blueprint, and stores it in the session artifact store.

    Args:
        session: The HypothesisLabSession to inject into.
        response_text: Raw text response from external Claude (may contain markdown).
        **kwargs: Optional sandbox overrides.

    Returns:
        Dict with blueprint_id, audit_result, and status.
    """
    from pde_agent.core.hypothesis_lab_agent_executor import (
        _audit_agent_executor_source,
    )
    from pde_agent.core.hypothesis_lab_creative_escalation import (
        build_agent_generated_blueprint,
        parse_generated_executor,
    )

    # Parse code from markdown; fallback to raw text if no fenced block found
    parsed = parse_generated_executor(response_text)
    source = parsed.get("executor_source", "")
    is_valid = parsed.get("valid", False)
    parse_issues = parsed.get("issues", [])

    # Fallback: if no markdown block found, try the raw text directly
    if not source and "no_python_code_block_found" in parse_issues:
        stripped = response_text.strip()
        if stripped.startswith("def ") or "import " in stripped:
            source = stripped
            parse_issues = []
            is_valid = True

    if not source:
        return {
            "action": "ingest_creative_escalation_response",
            "status": "error",
            "error": "no_code_extracted",
            "issues": parse_issues,
        }

    # AST audit
    audit_issues = _audit_agent_executor_source(source)
    if audit_issues:
        return {
            "action": "ingest_creative_escalation_response",
            "status": "error",
            "error": "audit_failed",
            "audit_issues": audit_issues,
            "parse_issues": parse_issues,
            "executor_source_preview": source[:500],
        }

    # Build blueprint
    blueprint = build_agent_generated_blueprint(
        executor_source=source,
        **{k: v for k, v in kwargs.items() if k in {"promotion_min_score_delta", "max_inference_seconds"}},
    )

    # Store blueprint in session
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id=f"agent_generated_{blueprint['blueprint_id']}",
        payload={
            "blueprint_count": 1,
            "blueprints": [blueprint],
            "source": "creative_escalation",
        },
    )

    session.emit(
        event_type="creative_escalation_response_ingested",
        payload={
            "blueprint_id": blueprint["blueprint_id"],
            "candidate_family": "agent_generated_executor",
            "audit_passed": True,
            "parse_issues": parse_issues,
        },
        tags=["creative_escalation", "ingested"],
    )

    return {
        "action": "ingest_creative_escalation_response",
        "status": "ok",
        "blueprint_id": blueprint["blueprint_id"],
        "candidate_family": "agent_generated_executor",
        "audit_passed": True,
        "parse_issues": parse_issues,
    }


# Registry mapping action names to their primitive functions
_ACTION_REGISTRY: dict[str, Any] = {
    "build_knowledge_cards": build_knowledge_cards_primitive,
    "generate_hypotheses": generate_hypotheses_primitive,
    "design_algorithm_blueprints": design_algorithm_blueprints_primitive,
    "run_blueprint_sandbox": run_blueprint_sandbox_primitive,
    "record_sandbox_outcome": record_sandbox_outcome_primitive,
    "generate_new_executor": _generate_new_executor_task,
    "ingest_creative_escalation_response": ingest_creative_escalation_response,
}

# Required kwargs per action (beyond session)
_REQUIRED_KWARGS: dict[str, set[str]] = {
    "build_knowledge_cards": {"manual_report_paths"},
    "generate_hypotheses": set(),
    "design_algorithm_blueprints": set(),
    "run_blueprint_sandbox": {"context_path", "target_path"},
    "record_sandbox_outcome": set(),
    "generate_new_executor": set(),
    "ingest_creative_escalation_response": {"response_text"},
}

# Actions that are deterministic (no external LLM required)
_DETERMINISTIC_ACTIONS: set[str] = {
    "build_knowledge_cards",
    "generate_hypotheses",
    "design_algorithm_blueprints",
    "run_blueprint_sandbox",
    "run_next_blueprint_sandbox",
    "record_sandbox_outcome",
    "report_results",
    "ingest_creative_escalation_response",
}


def _dispatch_action(
    action_name: str,
    session: HypothesisLabSession,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch an action to its primitive function.

    Args:
        action_name: The action to dispatch (e.g. "build_knowledge_cards").
        session: The hypothesis lab session.
        **kwargs: Additional arguments required by the action.

    Returns:
        The primitive result dict.

    Raises:
        ActionDispatchError: If the action is unknown, required kwargs are
            missing, or the primitive fails.
    """
    if action_name not in _ACTION_REGISTRY:
        if action_name == "report_results":
            return {"action": "report_results", "status": "ok"}
        raise ActionDispatchError(f"unknown action: {action_name}")

    required = _REQUIRED_KWARGS.get(action_name, set())
    missing = required - set(kwargs.keys())
    if missing:
        raise ActionDispatchError(
            f"action '{action_name}' missing required kwargs: {sorted(missing)}"
        )

    primitive = _ACTION_REGISTRY[action_name]

    session.emit(
        event_type=EVENT_STAGE_STARTED,
        payload={"stage": "dispatch_action", "action": action_name},
        tags=["dispatch", action_name],
    )

    try:
        result = primitive(session, **kwargs)
        return {
            "action": action_name,
            "status": "ok",
            "result": result,
        }
    except PrimitiveError as exc:
        session.emit(
            event_type=EVENT_STAGE_FAILED,
            payload={
                "stage": "dispatch_action",
                "action": action_name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            tags=["dispatch", action_name, "error"],
        )
        raise ActionDispatchError(
            f"action '{action_name}' failed: {exc}"
        ) from exc
    except Exception as exc:
        session.emit(
            event_type=EVENT_STAGE_FAILED,
            payload={
                "stage": "dispatch_action",
                "action": action_name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            tags=["dispatch", action_name, "error"],
        )
        raise ActionDispatchError(
            f"action '{action_name}' failed unexpectedly: {exc}"
        ) from exc


def is_deterministic_action(action_name: str) -> bool:
    """Return True if the action does not require LLM escalation."""
    return action_name in _DETERMINISTIC_ACTIONS
