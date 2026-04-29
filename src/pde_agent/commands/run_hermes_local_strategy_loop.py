from __future__ import annotations

from pathlib import Path

from pde_agent.core.local_strategy_competition_executor import (
    build_fno_competition_executor,
)
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
    executor_name: str,
    task_id: int | None,
    test_path: str | None,
    checkpoint_path: str | None,
    target_path: str | None,
    submission_id: str,
    dataset_key: str,
    baseline_score: float | None,
    dry_run: bool,
    force: bool,
) -> dict[str, object]:
    executor = None
    if executor_name == "dry-run":
        if not dry_run:
            raise RuntimeError("dry-run executor requires --dry-run")
    elif executor_name == "fno-competition":
        if task_id is None or test_path is None or checkpoint_path is None:
            raise RuntimeError(
                "fno-competition executor requires task_id, test_path, "
                "and checkpoint_path"
            )
        executor = build_fno_competition_executor(
            task_id=task_id,
            test_path=Path(test_path),
            checkpoint_path=Path(checkpoint_path),
            submission_id=submission_id,
            target_path=Path(target_path) if target_path is not None else None,
            dataset_key=dataset_key,
            force=force,
        )
    else:
        raise RuntimeError(
            "unsupported executor for run-hermes-local-strategy-loop: "
            f"{executor_name}"
        )
    report = run_hermes_local_strategy_loop(
        output_dir=Path(output_dir),
        applicable_pde=applicable_pde,
        model_family=model_family,
        failure_modes=failure_modes,
        max_experiments=max_experiments,
        min_score_delta=min_score_delta,
        max_inference_seconds=max_inference_seconds,
        executor=executor,
        force=force,
        baseline_score=baseline_score,
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
            "executor": executor_name,
        },
        next_hint=str(report["next_command"]),
    )
