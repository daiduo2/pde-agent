from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_local_strategy_runner import (
    RUNNER_STATE_HINT,
    run_hermes_local_strategy_loop,
)
from pde_agent.core.json_output import success_payload


def run_hermes_local_strategy_loop_command(
    *,
    output_dir: str,
    applicable_pde: str,
    model_family: str,
    failure_modes: list[str],
    max_experiments: int,
    min_score_delta: float,
    max_inference_seconds: float,
    dry_run: bool,
    force: bool,
) -> dict[str, object]:
    if not dry_run:
        raise RuntimeError(
            "run-hermes-local-strategy-loop currently requires --dry-run unless "
            "called from Python with an executor"
        )
    report = run_hermes_local_strategy_loop(
        output_dir=Path(output_dir),
        applicable_pde=applicable_pde,
        model_family=model_family,
        failure_modes=failure_modes,
        max_experiments=max_experiments,
        min_score_delta=min_score_delta,
        max_inference_seconds=max_inference_seconds,
        executor=None,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-local-strategy-loop",
        state_hint=RUNNER_STATE_HINT,
        artifacts={"hermes_local_strategy_loop_report_path": report["loop_report_path"]},
        metrics={
            "iteration_count": report["iteration_count"],
            "decision": report["decision"]["decision"],
            "route_action": report["final_route"]["route"]["action"],
        },
        next_hint=str(report["next_command"]),
    )
