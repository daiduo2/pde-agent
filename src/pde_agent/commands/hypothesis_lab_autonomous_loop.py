"""CLI command for autonomous Hypothesis Lab loop.

Thin wrapper around AutonomousHypothesisLab that follows the JSON-only
CLI contract.
"""

from __future__ import annotations

from typing import Any

from pde_agent.core.hypothesis_lab_autonomous_loop import (
    STATE_HINT,
    AutonomousHypothesisLab,
)
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession
from pde_agent.core.json_output import success_payload


def agent_hypothesis_run_autonomous_loop_command(
    *,
    workspace: str,
    context_path: str,
    target_path: str,
    reference_path: str | None = None,
    reference_sample_limit: int | None = None,
    manual_report_paths: list[str] | None = None,
    max_iterations: int = 10,
    autonomy_level: str = "deterministic",
    submission_id: str = "autonomous_loop",
    force: bool = True,
) -> dict[str, object]:
    """Run the autonomous hypothesis-lab loop.

    Args:
        workspace: Session workspace directory.
        context_path: Path to context NPZ file.
        target_path: Path to target NPZ file.
        reference_path: Optional reference NPZ file.
        reference_sample_limit: Optional sample limit for reference.
        manual_report_paths: Optional list of manual report paths.
        max_iterations: Maximum loop iterations.
        autonomy_level: "deterministic" | "guarded" | "full".
        submission_id: Run identifier.
        force: Overwrite existing artifacts.

    Returns:
        JSON payload following the CLI contract.
    """
    session = HypothesisLabSession.load(workspace=workspace)
    engine = AutonomousHypothesisLab(
        session=session,
        context_path=context_path,
        target_path=target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        manual_report_paths=manual_report_paths,
    )
    result = engine.run(
        max_iterations=max_iterations,
        autonomy_level=autonomy_level,
    )

    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-run-autonomous-loop",
        state_hint=STATE_HINT,
        artifacts={
            "workspace": str(session.workspace),
            "session_id": session.session_id,
            "iterations": result.get("iterations", 0),
        },
        metrics={
            "iterations": result.get("iterations", 0),
            "total_artifacts": result.get("metrics", {}).get("total_artifacts", 0),
            "event_count": result.get("metrics", {}).get("event_count", 0),
        },
        next_hint="review results or resume loop",
    )
