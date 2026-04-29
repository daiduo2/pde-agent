"""Hermes local strategy loop runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.local_strategy_leaderboard import record_local_strategy_result
from pde_agent.core.local_strategy_library import (
    build_local_strategy_library,
    lookup_local_strategies,
)
from pde_agent.core.local_strategy_result_ingestion import ingest_local_strategy_result
from pde_agent.core.local_strategy_route import route_local_strategy_next_action
from pde_agent.core.local_strategy_scheduler import (
    claim_next_scheduled_experiment,
    schedule_local_strategy_evaluation,
)


RUNNER_STATE_HINT = "HERMES_LOCAL_STRATEGY_LOOP_RAN"
Executor = Callable[[dict[str, Any], Path], Path]


class HermesLocalStrategyRunnerError(RuntimeError):
    pass


def run_hermes_local_strategy_loop(
    *,
    output_dir: str | Path,
    applicable_pde: str,
    model_family: str,
    failure_modes: list[str],
    max_experiments: int,
    min_score_delta: float,
    max_inference_seconds: float,
    executor: Executor | None,
    force: bool,
) -> dict[str, Any]:
    if max_experiments < 1:
        raise HermesLocalStrategyRunnerError("max_experiments must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = _bootstrap(
        root=root,
        applicable_pde=applicable_pde,
        model_family=model_family,
        failure_modes=failure_modes,
        max_experiments=max_experiments,
        max_inference_seconds=max_inference_seconds,
        force=force,
    )
    run_executor = executor or _dry_run_executor
    completed_ids: list[str] = []
    iterations = []
    leaderboard: dict[str, Any] | None = None
    final_route: dict[str, Any] | None = None
    for index in range(1, max_experiments + 1):
        claim_path = root / f"claim_{index:03d}.json"
        claim = claim_next_scheduled_experiment(
            schedule_path=paths["schedule_path"],
            completed_experiment_ids=completed_ids,
            output_path=claim_path,
        )
        selected = claim["selected_experiment"]
        workspace_dir = Path(selected["workspace_dir"])
        workspace_dir.mkdir(parents=True, exist_ok=True)
        source_result_path = run_executor(claim, workspace_dir)
        ingested_path = root / f"ingested_result_{index:03d}.json"
        ingested = ingest_local_strategy_result(
            source_path=source_result_path,
            output_path=ingested_path,
            baseline_score=None,
            result_status="completed",
            failure_reason=None,
        )
        leaderboard = record_local_strategy_result(
            schedule_path=paths["schedule_path"],
            experiment_id=selected["experiment_id"],
            result_summary_path=ingested_path,
            leaderboard_path=root / "leaderboard.json",
            result_status=str(ingested["result_status"]),
            min_score_delta=min_score_delta,
            max_inference_seconds=max_inference_seconds,
        )
        completed_ids = list(leaderboard["completed_experiment_ids"])
        route = route_local_strategy_next_action(
            artifact_path=root / "leaderboard.json",
            output_path=root / f"route_{index:03d}.json",
        )
        final_route = route
        iterations.append(
            {
                "iteration": index,
                "claim_path": claim["claim_path"],
                "experiment_id": selected["experiment_id"],
                "strategy_id": selected["strategy_id"],
                "source_result_path": str(source_result_path),
                "ingested_result_path": str(ingested_path),
                "leaderboard_decision": leaderboard["decision"]["decision"],
                "route_action": route["route"]["action"],
                "route_command": route["route"]["command"],
            }
        )
        if leaderboard["decision"]["decision"] != "continue_scheduled_experiments":
            break
    if leaderboard is None or final_route is None:
        raise HermesLocalStrategyRunnerError("local strategy loop produced no iterations")
    report_path = root / "hermes_local_strategy_loop_report.json"
    report = {
        "schema_version": 1,
        "state_hint": RUNNER_STATE_HINT,
        "loop_report_path": str(report_path),
        "output_dir": str(root),
        "library_path": paths["library_path"],
        "lookup_path": paths["lookup_path"],
        "schedule_path": paths["schedule_path"],
        "iteration_count": len(iterations),
        "iterations": iterations,
        "leaderboard": leaderboard,
        "decision": leaderboard["decision"],
        "final_route": final_route,
        "next_command": final_route["route"]["command"],
    }
    _write_json(report_path, report)
    return report


def _bootstrap(
    *,
    root: Path,
    applicable_pde: str,
    model_family: str,
    failure_modes: list[str],
    max_experiments: int,
    max_inference_seconds: float,
    force: bool,
) -> dict[str, str]:
    library_path = root / "local_strategy_library.json"
    lookup_path = root / "local_strategy_lookup.json"
    schedule_path = root / "local_strategy_schedule.json"
    build_local_strategy_library(output_path=library_path, force=force)
    lookup_local_strategies(
        library_path=library_path,
        applicable_pde=applicable_pde,
        model_family=model_family,
        failure_modes=failure_modes,
        max_results=max_experiments,
        output_path=lookup_path,
    )
    schedule_local_strategy_evaluation(
        lookup_report_path=lookup_path,
        output_path=schedule_path,
        workspace_root=root / "workspace",
        max_experiments=max_experiments,
        max_train_seconds=3600.0,
        max_inference_seconds=max_inference_seconds,
        force=force,
    )
    return {
        "library_path": str(library_path),
        "lookup_path": str(lookup_path),
        "schedule_path": str(schedule_path),
    }


def _dry_run_executor(claim: dict[str, Any], workspace_dir: Path) -> Path:
    selected = claim["selected_experiment"]
    strategy_id = str(selected["strategy_id"])
    result_path = workspace_dir / "dry_run_result.json"
    if strategy_id == "fno_spectral_viscosity_filter_v0":
        score = 76.0
        baseline = 72.0
    else:
        score = 70.0
        baseline = 72.0
    _write_json(
        result_path,
        {
            "state_hint": "LOCAL_EXPERIMENT_RESULT",
            "score": score,
            "baseline_score": baseline,
            "inference_time_seconds": 1.0,
            "result_summary": "Hermes local strategy runner dry-run result",
        },
    )
    return result_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
