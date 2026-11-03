"""Task 2 FNO training closed loop iteration runner.

Mirrors Task 1's fno_training_closed_loop but simplified for the
single-family fno_lite_task2 regime.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_memory import (
    HypothesisLabMemory,
    get_memory_path_from_env,
)
from pde_agent.core.json_output import success_payload
from pde_agent.core.task2_fno_training_checkpoint import (
    checkpoint_task2_training_loop,
)
from pde_agent.core.task2_fno_training_convergence import (
    Task2FnoTrainingConvergenceError,
    check_task2_fno_training_convergence,
)
from pde_agent.core.task2_fno_training_health_report import (
    build_task2_fno_training_health_report,
)
from pde_agent.core.task2_fno_training_loop import (
    Task2FnoTrainingLoopError,
    run_task2_fno_optimization_loop,
)
from pde_agent.core.task2_fno_training_memory import (
    Task2FnoTrainingMemoryError,
    build_task2_fno_training_memory,
    load_task2_fno_training_memory,
)
from pde_agent.core.task2_fno_training_policy import (
    Task2FnoTrainingPolicyError,
    plan_task2_fno_candidates,
)

STATE_HINT = "TASK2_FNO_TRAINING_ITERATION_REPORTED"
CONTRACT_STATE_HINT = "TASK2_FNO_TRAINING_CONTRACT_BUILT"
FINAL_REPORT_STATE_HINT = "TASK2_FNO_TRAINING_FINAL_REPORT_BUILT"


class Task2FnoTrainingClosedLoopError(RuntimeError):
    pass


def run_task2_fno_training_iteration(
    *,
    workspace_dir: str | Path,
    memory_path: str | Path | None = None,
    evidence_paths: list[str | Path] | None = None,
    max_candidates: int = 4,
    hidden_channels_values: list[int] | None = None,
    modes_values: list[int] | None = None,
    depths_values: list[int] | None = None,
    physics_weights_values: list[float] | None = None,
    learning_rates_values: list[float] | None = None,
    train_steps_values: list[int] | None = None,
    batch_sizes_values: list[int] | None = None,
    target_score: float | None = None,
    patience: int = 3,
    device: str = "mps",
    checkpoint_every: int = 1,
    force: bool = False,
    hypothesis_memory_path: str | Path | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run one Task 2 FNO training iteration.

    Steps:
        1. Load previous memory (if any) to get iteration_index + search guidance
        2. Plan candidates with evidence-based pruning
        3. Run optimization loop (train all candidates, select best)
        4. Evaluate best candidate
        5. Build iteration memory
        6. Check convergence
        7. Write artifacts and return report
    """
    root = Path(workspace_dir)
    root.mkdir(parents=True, exist_ok=True)

    previous_memory = None
    iteration_index = 0
    if memory_path is not None:
        previous_memory = load_task2_fno_training_memory(memory_path)
        if previous_memory is not None:
            iteration_index = int(previous_memory.get("iteration_index", 0)) + 1

    evidence_paths = _resolve_evidence_paths(
        evidence_paths=evidence_paths,
        workspace_dir=root,
        memory_path=memory_path,
        previous_memory=previous_memory,
    )

    plan_path = root / f"task2_fno_plan_{iteration_index:04d}.json"
    optimization_dir = root / f"task2_optimization_{iteration_index:04d}"
    memory_output_path = root / f"task2_fno_memory_{iteration_index:04d}.json"
    convergence_path = root / f"task2_fno_convergence_{iteration_index:04d}.json"
    report_path = root / "task2_fno_iteration_report.json"

    def _run_body(
        *,
        max_candidates_override: int | None = None,
    ) -> dict[str, Any]:
        blocked_configs: list[dict[str, Any]] = []
        search_guidance: list[dict[str, Any]] = []
        if previous_memory is not None:
            blocked_configs = list(previous_memory.get("blocked_configs", []))
            search_guidance = list(previous_memory.get("search_guidance", []))
            # If no improvement for 2+ iterations, expand train_steps
            convergence_history = previous_memory.get("convergence_history", {})
            no_improvement = convergence_history.get("no_improvement_count", 0)
            if no_improvement >= 2 and not train_steps_values:
                # Defer expansion to policy by injecting guidance
                search_guidance.append({
                    "guidance_type": "expand_budget",
                    "reason": f"no_improvement_for_{no_improvement}_iterations",
                    "recommended_action": "increase_train_steps",
                })

        plan = plan_task2_fno_candidates(
            evidence_paths=[str(p) for p in evidence_paths],
            output_path=str(plan_path),
            max_candidates=max_candidates_override or max_candidates,
            hidden_channels_values=hidden_channels_values,
            modes_values=modes_values,
            depths_values=depths_values,
            physics_weights_values=physics_weights_values,
            learning_rates_values=learning_rates_values,
            train_steps_values=train_steps_values,
            batch_sizes_values=batch_sizes_values,
            blocked_configs=blocked_configs or None,
            search_guidance=search_guidance or None,
            force=force,
        )

        # Hypothesis-lab memory bridge
        hl_memory: HypothesisLabMemory | None = None
        hl_path = hypothesis_memory_path or get_memory_path_from_env()
        if hl_path is not None:
            hl_memory = HypothesisLabMemory(memory_path=hl_path)

        opt_report = run_task2_fno_optimization_loop(
            candidate_plan_path=str(plan_path),
            output_dir=str(optimization_dir),
            device=device,
            force=force,
            memory=hl_memory,
            session_id=session_id or root.name,
        )

        eval_report = evaluate_task2_fno_training_iteration(opt_report=opt_report)

        memory = build_task2_fno_training_memory(
            optimization_report=opt_report,
            iteration_index=iteration_index,
            previous_memory=previous_memory,
            source_optimization_report_path=str(
                optimization_dir / "task2_fno_optimization_report.json"
            ),
        )

        convergence = check_task2_fno_training_convergence(
            memory=memory,
            target_score=target_score,
            patience=patience,
        )

        summary = _build_summary(
            eval_report=eval_report,
            previous_best_score=_float_or_none(
                previous_memory.get("best_so_far", {}).get("score") if previous_memory else None
            ),
        )

        _write_json(memory_output_path, memory, force=force)
        _write_json(convergence_path, convergence, force=force)

        report = build_task2_fno_training_iteration_report(
            iteration_index=iteration_index,
            summary=summary,
            convergence=convergence,
            memory=memory,
            artifacts={
                "optimization_report_path": str(
                    optimization_dir / "task2_fno_optimization_report.json"
                ),
                "memory_path": str(memory_output_path),
                "convergence_path": str(convergence_path),
                "report_path": str(report_path),
            },
        )
        _write_json(report_path, report, force=force)

        if checkpoint_every > 0 and (iteration_index + 1) % checkpoint_every == 0:
            checkpoint_task2_training_loop(
                workspace_dir=root,
                iteration_index=iteration_index,
                memory=memory,
                convergence=convergence,
                best_config=summary.get("config"),
                trajectory=memory.get("trajectory_summary", []),
                force=force,
            )

        artifacts: dict[str, Any] = {
            "optimization_report_path": str(
                optimization_dir / "task2_fno_optimization_report.json"
            ),
            "memory_path": str(memory_output_path),
            "convergence_path": str(convergence_path),
            "report_path": str(report_path),
            "plan_path": str(plan_path),
        }

        if convergence.get("decision") == "converged":
            final_report = build_task2_fno_training_final_report(
                memory_path=memory_output_path,
                workspace_dir=root,
                force=force,
            )
            artifacts["final_report_path"] = str(root / "task2_fno_training_final_report.json")

        return success_payload(
            run_id=root.name or "task2-fno-training",
            command="run-task2-fno-training-iteration",
            state_hint=STATE_HINT,
            artifacts=artifacts,
            metrics={
                "iteration_index": iteration_index,
                "best_score": summary.get("best_score"),
                "score_delta_vs_previous": summary.get("score_delta_vs_previous"),
                "convergence_decision": convergence.get("decision"),
                "candidate_count": len(opt_report.get("candidate_records", [])),
            },
            next_hint=_next_hint(convergence.get("decision", "continue")),
        )

    try:
        return _run_body()
    except Exception as exc:
        # Retry once with fewer candidates
        try:
            return _run_body(max_candidates_override=max(1, max_candidates - 1))
        except Exception:
            pass

        error_memory = build_error_memory(
            iteration_index=iteration_index,
            previous_memory=previous_memory,
            error_message=str(exc),
            error_type=type(exc).__name__,
        )
        convergence = build_error_convergence(
            error_message=str(exc),
            error_type=type(exc).__name__,
            recovered=False,
        )
        _write_json(memory_output_path, error_memory, force=force)
        _write_json(convergence_path, convergence, force=force)

        summary = {
            "best_candidate_id": None,
            "best_score": None,
            "score_delta_vs_previous": None,
            "config": None,
        }
        report = build_task2_fno_training_iteration_report(
            iteration_index=iteration_index,
            summary=summary,
            convergence=convergence,
            memory=error_memory,
            artifacts={
                "optimization_report_path": str(
                    optimization_dir / "task2_fno_optimization_report.json"
                ),
                "memory_path": str(memory_output_path),
                "convergence_path": str(convergence_path),
                "report_path": str(report_path),
            },
        )
        _write_json(report_path, report, force=force)

        return success_payload(
            run_id=root.name or "task2-fno-training",
            command="run-task2-fno-training-iteration",
            state_hint=STATE_HINT,
            artifacts={
                "optimization_report_path": str(
                    optimization_dir / "task2_fno_optimization_report.json"
                ),
                "memory_path": str(memory_output_path),
                "convergence_path": str(convergence_path),
                "report_path": str(report_path),
                "plan_path": str(plan_path),
                "recovery_actions": [
                    {
                        "action": "retry_with_degraded_params",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "suggested_params": {
                            "max_candidates": max(1, max_candidates - 1),
                            "train_steps_values": [5000],
                            "batch_sizes_values": [16],
                        },
                    },
                    {
                        "action": "escalate_to_hypothesis_lab",
                        "reason": "iteration_recovery_exhausted",
                    },
                ],
            },
            metrics={
                "iteration_index": iteration_index,
                "best_score": None,
                "score_delta_vs_previous": None,
                "convergence_decision": "error",
                "candidate_count": 0,
            },
            next_hint="iteration_failed_review_error_and_decide",
        )


def evaluate_task2_fno_training_iteration(
    *,
    opt_report: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate optimization report and return summary."""
    if opt_report.get("state_hint") != "TASK2_FNO_OPTIMIZATION_LOOP_RAN":
        raise Task2FnoTrainingClosedLoopError("optimization report has invalid state_hint")
    records = opt_report.get("candidate_records", [])
    selected = opt_report.get("selected_candidate", {})
    if not records:
        raise Task2FnoTrainingClosedLoopError("optimization report has no candidate records")
    best_record = selected if isinstance(selected, dict) and selected.get("score") else max(
        records,
        key=lambda item: float(item.get("score", float("-inf")))
        if isinstance(item.get("score"), int | float)
        else float("-inf"),
    )
    scores = [
        float(r["score"]) for r in records
        if isinstance(r.get("score"), int | float)
    ]
    return {
        "schema_version": 1,
        "state_hint": "TASK2_FNO_TRAINING_ITERATION_EVALUATED",
        "best_candidate_id": best_record.get("candidate_id"),
        "best_score": best_record.get("score"),
        "best_config": best_record.get("config"),
        "candidate_count": len(records),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }


def build_task2_fno_training_iteration_report(
    *,
    iteration_index: int,
    summary: dict[str, Any],
    convergence: dict[str, Any],
    memory: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Build standardized iteration report."""
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "iteration_index": iteration_index,
        "summary": summary,
        "convergence": convergence,
        "memory": memory,
        "artifacts": artifacts,
    }


def build_task2_fno_training_closed_loop_contract(
    *,
    workspace_dir: str | Path,
    output_path: str | Path | None = None,
    required_commands: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a contract defining the Task 2 closed loop protocol."""
    workspace = Path(workspace_dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    default_commands = [
        "run-task2-fno-training-iteration",
        "check-task2-fno-training-convergence",
        "build-task2-fno-training-memory",
        "build-task2-fno-training-final-report",
        "build-task2-fno-training-health-report",
    ]
    commands = required_commands or default_commands

    contract = {
        "schema_version": 1,
        "state_hint": CONTRACT_STATE_HINT,
        "workspace_dir": str(workspace),
        "required_commands": commands,
        "iteration_protocol": {
            "step_1": "run-task2-fno-training-iteration",
            "step_2": "check-task2-fno-training-convergence",
            "step_3": "build-task2-fno-training-final-report (if converged)",
        },
        "artifact_paths": {
            "plan_pattern": "task2_fno_plan_*.json",
            "optimization_pattern": "task2_optimization_*/task2_fno_optimization_report.json",
            "memory_pattern": "task2_fno_memory_*.json",
            "convergence_pattern": "task2_fno_convergence_*.json",
            "report": "task2_fno_iteration_report.json",
            "final_report": "task2_fno_training_final_report.json",
            "health_report": "task2_fno_training_health_report.json",
        },
    }

    out_file = Path(output_path) if output_path is not None else workspace / "task2_fno_training_contract.json"
    _write_json(out_file, contract, force=force)
    return contract


def build_task2_fno_training_final_report(
    *,
    memory_path: str | Path,
    workspace_dir: str | Path,
    output_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build final report from best memory."""
    memory_file = Path(memory_path).expanduser().resolve()
    workspace = Path(workspace_dir).expanduser().resolve()
    out_file = Path(output_path) if output_path is not None else workspace / "task2_fno_training_final_report.json"

    memory = _load_json(memory_file)
    if memory is None:
        raise Task2FnoTrainingClosedLoopError(f"could not load memory: {memory_file}")

    best_so_far = memory.get("best_so_far", {}) if isinstance(memory, dict) else {}
    trajectory = memory.get("trajectory_summary", []) if isinstance(memory, dict) else []

    best_candidate = best_so_far if best_so_far else (trajectory[-1] if trajectory else {})

    provenance = {
        "iteration_count": int(memory.get("iteration_index", 0)) + 1,
        "trajectory_length": len(trajectory),
    }

    report = {
        "schema_version": 1,
        "state_hint": FINAL_REPORT_STATE_HINT,
        "workspace_dir": str(workspace),
        "memory_path": str(memory_file),
        "best_candidate": {
            "candidate_id": best_candidate.get("candidate_id"),
            "candidate_family": best_candidate.get("candidate_family"),
            "score": best_candidate.get("score"),
            "config": best_candidate.get("config"),
        },
        "provenance": provenance,
        "artifacts": {
            "memory_path": str(memory_file),
            "convergence_path": str(workspace / f"task2_fno_convergence_{memory.get('iteration_index', 0):04d}.json"),
        },
        "agent_instructions": [
            "Task 2 training loop has converged.",
            "Review best_candidate config and artifacts.",
            "Run build-competition-submission or record-promoted-algorithm-strategy-proposal if satisfied.",
        ],
    }
    _write_json(out_file, report, force=force)
    return report


# -- Internal helpers --


def _resolve_evidence_paths(
    *,
    evidence_paths: list[str | Path] | None,
    workspace_dir: Path,
    memory_path: str | Path | None,
    previous_memory: dict[str, Any] | None,
) -> list[Path]:
    if evidence_paths:
        return [Path(p) for p in evidence_paths]
    paths: list[Path] = []
    if previous_memory is not None and memory_path is not None:
        memory_file = Path(memory_path)
        if memory_file.is_file():
            paths.append(memory_file)
        prev_artifacts = previous_memory.get("source_optimization_report_path")
        if isinstance(prev_artifacts, str):
            evidence_path = Path(prev_artifacts)
            if evidence_path.is_file():
                paths.append(evidence_path)
    for candidate in ["task2_fno_optimization_evidence.json"]:
        candidate_path = workspace_dir / candidate
        if candidate_path.is_file():
            paths.append(candidate_path)
    health_feedback_path = workspace_dir / "task2_fno_training_health_feedback.json"
    if health_feedback_path.is_file():
        paths.append(health_feedback_path)
    return paths


def _build_summary(
    *,
    eval_report: dict[str, Any],
    previous_best_score: float | None,
) -> dict[str, Any]:
    best_score = _float_or_none(eval_report.get("best_score"))
    score_delta = None
    if best_score is not None and previous_best_score is not None:
        score_delta = round(best_score - previous_best_score, 6)
    return {
        "best_candidate_id": eval_report.get("best_candidate_id"),
        "best_score": best_score,
        "score_delta_vs_previous": score_delta,
        "config": eval_report.get("best_config"),
    }


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise Task2FnoTrainingClosedLoopError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _next_hint(decision: str) -> str:
    if decision == "continue":
        return "run-task2-fno-training-iteration again with updated memory"
    if decision == "escalate":
        return "escalate to hypothesis lab for strategy discovery"
    if decision == "converged":
        return "training converged; review best candidate artifacts"
    if decision == "stagnant":
        return "training stagnant; consider escalation or parameter expansion"
    if decision == "doom_loop":
        return "doom loop detected; escalate to hypothesis lab immediately"
    if decision == "error":
        return "iteration failed; review recovery_actions and decide next step"
    return "run-task2-fno-training-iteration again"


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def build_error_memory(
    *,
    iteration_index: int,
    previous_memory: dict[str, Any] | None,
    error_message: str,
    error_type: str,
) -> dict[str, Any]:
    best_so_far = previous_memory.get("best_so_far") if previous_memory else None
    return {
        "schema_version": 1,
        "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
        "iteration_index": iteration_index,
        "source_optimization_report_path": None,
        "best_so_far": best_so_far or {},
        "trajectory_summary": [],
        "blocked_configs": [],
        "search_guidance": [
            {
                "guidance_type": "degraded_retry",
                "reason": f"Previous iteration failed with {error_type}: {error_message}",
                "recommended_action": "reduce_candidates_and_train_steps",
            }
        ],
        "convergence_history": previous_memory.get("convergence_history", {}) if previous_memory else {},
        "next_hints": [
            f"Iteration {iteration_index} failed with {error_type}. "
            "Consider retry with degraded parameters or escalate.",
        ],
    }


def build_error_convergence(
    *,
    error_message: str,
    error_type: str,
    recovered: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": "TASK2_FNO_TRAINING_CONVERGENCE_CHECKED",
        "decision": "error",
        "reason": f"iteration_failed:{error_type}: {error_message}",
        "confidence": 0.0,
        "recovered": recovered,
        "recommended_actions": [
            {"action": "retry_with_degraded_params", "reason": error_message, "params": {}},
            {"action": "escalate_to_hypothesis_lab", "reason": "iteration_recovery_exhausted", "params": {}},
        ],
        "triggers": [
            {"trigger": "iteration_failure", "error_type": error_type, "error_message": error_message}
        ],
    }
