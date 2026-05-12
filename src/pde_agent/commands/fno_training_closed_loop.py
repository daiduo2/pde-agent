"""CLI wrappers for FNO training closed-loop commands."""

import json
from pathlib import Path
from typing import Any

from pde_agent.core.fno_training_checkpoint import (
    checkpoint_training_loop,
    list_training_checkpoints,
    resume_training_loop,
)
from pde_agent.core.fno_training_closed_loop import (
    STATE_HINT as RUN_STATE_HINT,
)
from pde_agent.core.fno_training_closed_loop import (
    FnoTrainingClosedLoopError,
    build_fno_training_closed_loop_contract,
    build_fno_training_final_report,
    build_fno_training_health_report,
    evaluate_fno_training_iteration,
    run_fno_training_iteration,
)
from pde_agent.core.fno_training_escalation import (
    FnoTrainingEscalationError,
    run_fno_training_escalation_pipeline,
)
from pde_agent.core.fno_training_convergence import (
    STATE_HINT as CONVERGENCE_STATE_HINT,
)
from pde_agent.core.fno_training_convergence import check_training_convergence
from pde_agent.core.fno_training_memory import (
    STATE_HINT as MEMORY_STATE_HINT,
)
from pde_agent.core.fno_training_memory import (
    build_training_iteration_memory,
    load_training_iteration_memory,
    merge_training_memory_with_hypothesis_feedback,
)
from pde_agent.core.json_output import error_payload, success_payload


def run_fno_training_iteration_command(
    *,
    workspace_dir: str,
    source_path: str,
    memory_path: str | None,
    evidence_paths: list[str] | None,
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    time_reduction: int,
    space_reduction: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_candidates: int,
    modes: list[int] | None,
    widths: list[int] | None,
    depths: list[int] | None,
    loss_types: list[str] | None,
    forecast_steps_values: list[int] | None,
    train_steps_values: list[int] | None,
    normalizations: list[str] | None,
    target_score: float | None,
    patience: int,
    checkpoint_every: int,
    force: bool,
) -> dict[str, Any]:
    return run_fno_training_iteration(
        workspace_dir=workspace_dir,
        source_path=source_path,
        memory_path=memory_path,
        evidence_paths=evidence_paths,
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        max_candidates=max_candidates,
        modes=modes,
        widths=widths,
        depths=depths,
        loss_types=loss_types,
        forecast_steps_values=forecast_steps_values,
        train_steps_values=train_steps_values,
        normalizations=normalizations,
        target_score=target_score,
        patience=patience,
        checkpoint_every=checkpoint_every,
        force=force,
    )


def build_fno_training_closed_loop_contract_command(
    *,
    workspace_dir: str,
    source_path: str,
    output_path: str | None,
    dataset_key: str,
    baseline_prediction_path: str | None,
    task1_test_path: str | None,
    task1_target_path: str | None,
    required_commands: list[str] | None,
    force: bool,
) -> dict[str, Any]:
    contract = build_fno_training_closed_loop_contract(
        workspace_dir=workspace_dir,
        source_path=source_path,
        output_path=output_path,
        dataset_key=dataset_key,
        baseline_prediction_path=baseline_prediction_path,
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        required_commands=required_commands,
        force=force,
    )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="build-fno-training-closed-loop-contract",
        state_hint=contract["state_hint"],
        artifacts={
            "contract_path": output_path,
            "workspace_dir": workspace_dir,
        },
        metrics={
            "required_command_count": len(contract["required_commands"]),
        },
        next_hint="run-fno-training-iteration",
    )


def check_fno_training_convergence_command(
    *,
    memory_path: str,
    target_score: float | None,
    patience: int,
    output_path: str | None,
    force: bool,
) -> dict[str, Any]:
    memory = load_training_iteration_memory(memory_path)
    if memory is None:
        return error_payload(
            run_id=Path(memory_path).stem or "local",
            command="check-fno-training-convergence",
            error_code="missing_memory",
            error_type="FnoTrainingConvergenceError",
            message_short="could not load training iteration memory",
            recoverable=True,
            retryable=True,
        )
    result = check_training_convergence(
        memory=memory,
        target_score=target_score,
        patience=patience,
    )
    if output_path is not None:
        out = Path(output_path)
        if out.exists() and not force:
            return error_payload(
                run_id=Path(memory_path).stem or "local",
                command="check-fno-training-convergence",
                error_code="output_exists",
                error_type="FnoTrainingConvergenceError",
                message_short=f"output already exists: {output_path}",
                recoverable=True,
                retryable=False,
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return success_payload(
        run_id=Path(memory_path).stem or "local",
        command="check-fno-training-convergence",
        state_hint=CONVERGENCE_STATE_HINT,
        artifacts={
            "memory_path": memory_path,
            "convergence_path": output_path,
        },
        metrics={
            "decision": result.get("decision"),
            "confidence": result.get("confidence"),
        },
        next_hint="review convergence decision and decide next action",
    )


def build_fno_training_memory_command(
    *,
    optimization_report_path: str,
    iteration_index: int,
    previous_memory_path: str | None,
    output_path: str | None,
    force: bool,
) -> dict[str, Any]:
    report = _load_json(Path(optimization_report_path))
    previous_memory = None
    if previous_memory_path is not None:
        previous_memory = load_training_iteration_memory(previous_memory_path)
    memory = build_training_iteration_memory(
        optimization_report=report,
        iteration_index=iteration_index,
        previous_memory=previous_memory,
    )
    if output_path is not None:
        out = Path(output_path)
        if out.exists() and not force:
            return error_payload(
                run_id=f"iter-{iteration_index}",
                command="build-fno-training-memory",
                error_code="output_exists",
                error_type="FnoTrainingMemoryError",
                message_short=f"output already exists: {output_path}",
                recoverable=True,
                retryable=False,
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(memory, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return success_payload(
        run_id=f"iter-{iteration_index}",
        command="build-fno-training-memory",
        state_hint=MEMORY_STATE_HINT,
        artifacts={
            "optimization_report_path": optimization_report_path,
            "memory_path": output_path,
        },
        metrics={
            "iteration_index": iteration_index,
            "trajectory_count": len(memory.get("trajectory_summary", [])),
            "avoid_count": len(memory.get("avoid_repeating", [])),
        },
        next_hint="check-fno-training-convergence",
    )


def checkpoint_fno_training_loop_command(
    *,
    workspace_dir: str,
    iteration_index: int,
    memory_path: str,
    convergence_path: str,
    force: bool,
) -> dict[str, Any]:
    memory = load_training_iteration_memory(memory_path)
    if memory is None:
        return error_payload(
            run_id=f"iter-{iteration_index}",
            command="checkpoint-fno-training-loop",
            error_code="missing_memory",
            error_type="FnoTrainingCheckpointError",
            message_short="could not load training iteration memory",
            recoverable=True,
            retryable=True,
        )
    convergence = _load_json(Path(convergence_path))
    checkpoint_path = checkpoint_training_loop(
        workspace_dir=workspace_dir,
        iteration_index=iteration_index,
        memory=memory,
        convergence=convergence,
        best_config=memory.get("best_so_far", {}).get("config"),
        trajectory=memory.get("trajectory_summary", []),
        force=force,
    )
    return success_payload(
        run_id=f"iter-{iteration_index}",
        command="checkpoint-fno-training-loop",
        state_hint="FNO_TRAINING_CHECKPOINTED",
        artifacts={
            "checkpoint_path": str(checkpoint_path),
        },
        metrics={
            "iteration_index": iteration_index,
        },
        next_hint="continue training or resume from checkpoint if needed",
    )


def resume_fno_training_loop_command(
    *,
    checkpoint_path: str,
) -> dict[str, Any]:
    state = resume_training_loop(checkpoint_path=checkpoint_path)
    return success_payload(
        run_id=Path(checkpoint_path).stem or "local",
        command="resume-fno-training-loop",
        state_hint="FNO_TRAINING_RESUMED",
        artifacts={
            "checkpoint_path": checkpoint_path,
        },
        metrics={
            "iteration_index": state["iteration_index"],
        },
        next_hint="run-fno-training-iteration from resumed state",
    )


def escalate_fno_training_to_hypothesis_command(
    *,
    workspace_dir: str,
    memory_path: str,
    output_path: str | None,
    max_strategy_candidates: int,
    force: bool,
) -> dict[str, Any]:
    try:
        report = run_fno_training_escalation_pipeline(
            workspace_dir=workspace_dir,
            memory_path=memory_path,
            output_path=output_path,
            max_strategy_candidates=max_strategy_candidates,
            force=force,
        )
    except FnoTrainingEscalationError as exc:
        return error_payload(
            run_id=Path(workspace_dir).name or "local",
            command="escalate-fno-training-to-hypothesis",
            error_code="escalation_failed",
            error_type="FnoTrainingEscalationError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="escalate-fno-training-to-hypothesis",
        state_hint=report["state_hint"],
        artifacts={
            "memory_path": memory_path,
            "escalation_report_path": output_path,
            "merged_memory_path": report.get("merged_memory_path"),
            "algorithm_candidates_path": report.get("algorithm_candidates_path"),
            "expansion_report_path": report.get("expansion_report_path"),
        },
        metrics={
            "candidate_count": report.get("candidate_count", 0),
            "expansion_action_count": report.get("expansion_action_count", 0),
            "expansion_executable_count": report.get("expansion_executable_count", 0),
            "expansion_proposal_count": report.get("expansion_proposal_count", 0),
            "blocked_family_count": len(report.get("blocked_families", [])),
        },
        next_hint=report.get(
            "next_command",
            "run-fno-training-iteration with merged memory if candidates discovered",
        ),
    )


def integrate_hypothesis_training_feedback_command(
    *,
    hypothesis_report_path: str,
    memory_path: str,
    output_path: str | None,
    force: bool,
) -> dict[str, Any]:
    memory = load_training_iteration_memory(memory_path)
    if memory is None:
        return error_payload(
            run_id=Path(memory_path).stem or "local",
            command="integrate-hypothesis-training-feedback",
            error_code="missing_memory",
            error_type="FnoTrainingMemoryError",
            message_short="could not load training iteration memory",
            recoverable=True,
            retryable=True,
        )
    hypothesis_report = _load_json(Path(hypothesis_report_path))
    merged = merge_training_memory_with_hypothesis_feedback(
        memory=memory,
        hypothesis_report=hypothesis_report,
    )
    if output_path is not None:
        out = Path(output_path)
        if out.exists() and not force:
            return error_payload(
                run_id=Path(memory_path).stem or "local",
                command="integrate-hypothesis-training-feedback",
                error_code="output_exists",
                error_type="FnoTrainingMemoryError",
                message_short=f"output already exists: {output_path}",
                recoverable=True,
                retryable=False,
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(merged, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return success_payload(
        run_id=Path(memory_path).stem or "local",
        command="integrate-hypothesis-training-feedback",
        state_hint="HYPOTHESIS_LAB_STRATEGY_FEEDBACK_INTEGRATED",
        artifacts={
            "memory_path": memory_path,
            "hypothesis_report_path": hypothesis_report_path,
            "merged_memory_path": output_path,
        },
        metrics={
            "merged_hint_count": len(merged.get("next_hints", [])),
        },
        next_hint="run-fno-training-iteration with updated memory",
    )


def build_fno_training_final_report_command(
    *,
    workspace_dir: str,
    memory_path: str,
    output_path: str | None,
    force: bool,
) -> dict[str, Any]:
    try:
        report = build_fno_training_final_report(
            memory_path=memory_path,
            workspace_dir=workspace_dir,
            output_path=output_path,
            force=force,
        )
    except FnoTrainingClosedLoopError as exc:
        return error_payload(
            run_id=Path(workspace_dir).name or "local",
            command="build-fno-training-final-report",
            error_code="final_report_failed",
            error_type="FnoTrainingClosedLoopError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="build-fno-training-final-report",
        state_hint=report["state_hint"],
        artifacts={
            "memory_path": memory_path,
            "final_report_path": output_path or str(Path(workspace_dir) / "fno_training_final_report.json"),
        },
        metrics={
            "best_score": report.get("best_candidate", {}).get("score"),
            "iteration_count": report.get("provenance", {}).get("iteration_count"),
        },
        next_hint="review final report and promote candidate if satisfied",
    )


def build_fno_training_health_report_command(
    *,
    workspace_dir: str,
    output_path: str | None,
    force: bool,
) -> dict[str, Any]:
    try:
        report = build_fno_training_health_report(
            workspace_dir=workspace_dir,
            output_path=output_path,
            force=force,
        )
    except FnoTrainingClosedLoopError as exc:
        return error_payload(
            run_id=Path(workspace_dir).name or "local",
            command="build-fno-training-health-report",
            error_code="health_report_failed",
            error_type="FnoTrainingClosedLoopError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="build-fno-training-health-report",
        state_hint=report["state_hint"],
        artifacts={
            "health_report_path": output_path or str(Path(workspace_dir) / "fno_training_health_report.json"),
        },
        metrics={
            "iteration_count": report.get("iteration_count"),
            "global_best": report.get("score", {}).get("global_best"),
            "health_status": report.get("health", {}).get("status"),
            "issue_count": len(report.get("health", {}).get("issues", [])),
        },
        next_hint=report.get("next_hint"),
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON must be an object: {path}")
    return value
