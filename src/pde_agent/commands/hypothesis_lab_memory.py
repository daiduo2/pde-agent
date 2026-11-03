"""CLI commands for Hypothesis Lab global memory queries.

Thin wrappers around HypothesisLabMemory. Each returns a JSON payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_memory import (
    HypothesisLabMemory,
    get_memory_path_from_env,
)
from pde_agent.core.json_output import success_payload


def _resolve_memory(memory_path: str | None = None) -> HypothesisLabMemory:
    if memory_path:
        return HypothesisLabMemory(memory_path)
    env_path = get_memory_path_from_env()
    if env_path:
        return HypothesisLabMemory(env_path)
    return HypothesisLabMemory()


def agent_hypothesis_get_global_pareto_front_command(
    *,
    workspace: str,
    memory_path: str | None = None,
) -> dict[str, object]:
    """Get the global Pareto front across all sessions."""
    memory = _resolve_memory(memory_path)
    pareto = memory.get_global_pareto_front()
    return success_payload(
        run_id=workspace,
        command="agent-hypothesis-get-global-pareto-front",
        state_hint="HYPOTHESIS_LAB_GLOBAL_PARETO_QUERIED",
        artifacts={"pareto_front": pareto},
        metrics={
            "pareto_size": len(pareto),
            "best_score": pareto[0].get("score") if pareto else None,
        },
        next_hint="explore global tradeoffs or refine best candidates",
    )


def agent_hypothesis_get_family_history_command(
    *,
    family: str,
    memory_path: str | None = None,
) -> dict[str, object]:
    """Get score trajectory for a candidate family across all sessions."""
    memory = _resolve_memory(memory_path)
    history = memory.get_family_performance_history(family)
    scores = [
        h["score"] for h in history
        if isinstance(h.get("score"), (int, float))
    ]
    return success_payload(
        run_id=family,
        command="agent-hypothesis-get-family-history",
        state_hint="HYPOTHESIS_LAB_FAMILY_HISTORY_QUERIED",
        artifacts={"history": history},
        metrics={
            "family": family,
            "data_points": len(history),
            "best_score": max(scores) if scores else None,
            "latest_score": scores[-1] if scores else None,
        },
        next_hint="identify trends or saturation in this family",
    )


def agent_hypothesis_get_lessons_learned_command(
    *,
    n: int = 10,
    memory_path: str | None = None,
) -> dict[str, object]:
    """Get cross-session insights and lessons learned."""
    memory = _resolve_memory(memory_path)
    lessons = memory.get_lessons_learned(n=n)
    return success_payload(
        run_id="global",
        command="agent-hypothesis-get-lessons-learned",
        state_hint="HYPOTHESIS_LAB_LESSONS_QUERIED",
        artifacts={"lessons": lessons},
        metrics={
            "total_sessions": lessons.get("total_sessions", 0),
            "total_experiments": lessons.get("total_experiments", 0),
            "top_pattern_count": len(lessons.get("top_failure_patterns", [])),
        },
        next_hint="apply lessons to next experiment design",
    )
