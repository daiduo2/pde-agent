"""CLI command for creative escalation context assembly.

Thin wrapper around build_creative_escalation_context that follows the
JSON-only CLI contract.
"""

from __future__ import annotations

from typing import Any

from pde_agent.core.hypothesis_lab_creative_escalation import (
    STATE_HINT,
    build_creative_escalation_context,
)
from pde_agent.core.hypothesis_lab_memory import HypothesisLabMemory
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession
from pde_agent.core.json_output import success_payload


def build_creative_escalation_context_command(
    *,
    workspace: str,
    memory_path: str | None = None,
) -> dict[str, object]:
    """Build creative escalation context for LLM code generation.

    Args:
        workspace: Session workspace directory.
        memory_path: Optional path to cross-session memory JSONL.

    Returns:
        JSON payload following the CLI contract.
    """
    session = HypothesisLabSession.load(workspace=workspace)
    memory = HypothesisLabMemory(memory_path) if memory_path else None

    context = build_creative_escalation_context(session, memory=memory)

    return success_payload(
        run_id=session.session_id,
        command="build-creative-escalation-context",
        state_hint=STATE_HINT,
        artifacts={
            "workspace": str(session.workspace),
            "session_id": session.session_id,
        },
        metrics={
            "total_artifacts": context["session_summary"]["total_artifacts"],
            "event_count": context["session_summary"]["event_count"],
            "blocked_families_count": len(context["blocked_families"]),
            "families_seen_count": len(context["failure_patterns"]),
            "score_trajectory_length": len(context["score_trajectory"]),
            "proven_strategies_count": len(context["proven_strategies"]),
        },
        next_hint="feed prompt_template to LLM and parse response with parse_generated_executor",
    )
