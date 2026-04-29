"""Autonomous Hermes observe-diagnose-plan loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_next_action_policy import plan_hermes_next_action
from pde_agent.core.hermes_result_diagnosis import diagnose_hermes_result


STATE_HINT = "HERMES_CONVERGENCE_LOOP_RAN"
ActionExecutor = Callable[[dict[str, Any], Path], Path | None]


class HermesConvergenceLoopError(RuntimeError):
    pass


def run_hermes_convergence_loop(
    *,
    initial_artifact_path: str | Path,
    output_dir: str | Path,
    max_steps: int,
    max_inference_seconds: float = 120.0,
    min_score_delta: float = 1.0,
    action_executor: ActionExecutor | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if max_steps < 1:
        raise HermesConvergenceLoopError("max_steps must be positive")
    root = Path(output_dir)
    report_path = root / "hermes_convergence_loop_report.json"
    if report_path.exists() and not force:
        raise HermesConvergenceLoopError(
            f"Hermes convergence loop report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    current_artifact = Path(initial_artifact_path)
    iterations: list[dict[str, Any]] = []
    final_plan: dict[str, Any] | None = None
    for step in range(1, max_steps + 1):
        diagnosis_path = root / f"diagnosis_{step:03d}.json"
        plan_path = root / f"next_action_{step:03d}.json"
        diagnosis = diagnose_hermes_result(
            artifact_path=current_artifact,
            output_path=diagnosis_path,
            max_inference_seconds=max_inference_seconds,
            min_score_delta=min_score_delta,
        )
        plan = plan_hermes_next_action(
            diagnosis_path=diagnosis_path,
            output_path=plan_path,
        )
        final_plan = plan
        executed_artifact: Path | None = None
        if not plan.get("terminal") and action_executor is not None:
            workspace_dir = root / "workspace" / f"step_{step:03d}"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            executed_artifact = action_executor(plan, workspace_dir)
        iterations.append(
            {
                "step": step,
                "observed_artifact_path": str(current_artifact),
                "diagnosis_path": str(diagnosis_path),
                "primary_bottleneck": diagnosis["primary_bottleneck"],
                "recommended_action_family": diagnosis["recommended_action_family"],
                "plan_path": str(plan_path),
                "planned_action": plan["action"],
                "planned_command": plan["command"],
                "executed_artifact_path": (
                    str(executed_artifact) if executed_artifact is not None else None
                ),
            }
        )
        if executed_artifact is None:
            break
        current_artifact = executed_artifact

    if final_plan is None:
        raise HermesConvergenceLoopError("convergence loop produced no plan")
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "loop_report_path": str(report_path),
        "initial_artifact_path": str(initial_artifact_path),
        "iteration_count": len(iterations),
        "iterations": iterations,
        "final_decision": {
            "action": final_plan["action"],
            "command": final_plan["command"],
            "terminal": final_plan["terminal"],
            "reason_codes": final_plan["reason_codes"],
        },
        "next_command": final_plan["command"],
    }
    _write_json(report_path, report)
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
