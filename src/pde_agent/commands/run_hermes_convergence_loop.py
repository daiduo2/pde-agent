from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_convergence_loop import (
    STATE_HINT,
    run_hermes_convergence_loop,
)
from pde_agent.core.json_output import success_payload


def run_hermes_convergence_loop_command(
    *,
    initial_artifact_path: str,
    output_dir: str,
    max_steps: int,
    max_inference_seconds: float,
    min_score_delta: float,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_convergence_loop(
        initial_artifact_path=Path(initial_artifact_path),
        output_dir=Path(output_dir),
        max_steps=max_steps,
        max_inference_seconds=max_inference_seconds,
        min_score_delta=min_score_delta,
        action_executor=None,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-convergence-loop",
        state_hint=STATE_HINT,
        artifacts={"convergence_loop_report_path": report["loop_report_path"]},
        metrics={
            "iteration_count": report["iteration_count"],
            "final_action": report["final_decision"]["action"],
            "final_command": report["final_decision"]["command"],
        },
        next_hint=report["next_command"],
    )
