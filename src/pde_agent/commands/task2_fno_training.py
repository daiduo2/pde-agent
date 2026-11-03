"""CLI command wrappers for Task 2 FNO training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.json_output import error_payload, success_payload
from pde_agent.core.task2_fno_training_core import train_task2_fno_model
from pde_agent.core.task2_fno_training_loop import run_task2_fno_optimization_loop
from pde_agent.core.task2_fno_training_policy import plan_task2_fno_candidates
from pde_agent.core.task2_fno_training_closed_loop import (
    Task2FnoTrainingClosedLoopError,
    build_task2_fno_training_closed_loop_contract,
    build_task2_fno_training_final_report,
    build_task2_fno_training_health_report,
    build_task2_fno_training_memory,
    check_task2_fno_training_convergence,
    run_task2_fno_training_iteration,
)
from pde_agent.core.task2_fno_training_escalation import (
    Task2FnoTrainingEscalationError,
    run_task2_fno_training_escalation_pipeline,
)
from pde_agent.core.task2_fno_training_memory import (
    load_task2_fno_training_memory,
    merge_task2_training_memory_with_hypothesis_feedback,
)


def run_task2_fno_training_command(
    *,
    hidden_channels: int = 96,
    modes: int = 48,
    depth: int = 5,
    batch_size: int = 32,
    train_steps: int = 10000,
    learning_rate: float = 1e-3,
    physics_weight: float = 0.01,
    device: str = "mps",
    seed: int = 42,
    output_dir: str = "/tmp/task2_fno_training",
    checkpoint_path: str | None = None,
    resume_lr: float | None = None,
    force: bool = False,
    # V2 model flags
    use_v2: bool = False,
    use_film: bool = False,
    use_coord: bool = False,
    use_residual: bool = False,
    use_groupnorm: bool = False,
    dropout: float = 0.0,
    # Forecast steps
    forecast_steps: int = 190,
    # Auxiliary loss weights
    time_weight: float = 0.0,
    grad_weight: float = 0.0,
    temporal_weight: float = 0.0,
) -> dict[str, Any]:
    """Train a single Task 2 FNO model."""
    report = train_task2_fno_model(
        hidden_channels=hidden_channels,
        modes=modes,
        depth=depth,
        batch_size=batch_size,
        train_steps=train_steps,
        learning_rate=learning_rate,
        physics_weight=physics_weight,
        device=device,
        seed=seed,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        resume_lr=resume_lr,
        force=force,
        use_v2=use_v2,
        use_film=use_film,
        use_coord=use_coord,
        use_residual=use_residual,
        use_groupnorm=use_groupnorm,
        dropout=dropout,
        forecast_steps=forecast_steps,
        time_weight=time_weight,
        grad_weight=grad_weight,
        temporal_weight=temporal_weight,
    )
    return success_payload(
        run_id=report.get("config", {}).get("candidate_id", "task2-fno"),
        command="run-task2-fno-training",
        state_hint=report["state_hint"],
        artifacts=report.get("artifacts", {}),
        metrics={
            "segment_score": report.get("segment_score"),
            "task2_proxy": report.get("task2_proxy"),
            "best_ext_val_loss": report.get("best_ext_val_loss"),
            "parameters": report.get("parameters"),
            "runtime_seconds": report.get("runtime_seconds"),
        },
        next_hint=report.get("next_hint", ""),
    )


def plan_task2_fno_candidates_command(
    *,
    evidence_paths: list[str] | None = None,
    output_path: str | None = None,
    max_candidates: int = 8,
    hidden_channels_values: list[int] | None = None,
    modes_values: list[int] | None = None,
    depths_values: list[int] | None = None,
    physics_weights_values: list[float] | None = None,
    learning_rates_values: list[float] | None = None,
    train_steps_values: list[int] | None = None,
    batch_sizes_values: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Plan Task 2 FNO training candidates."""
    plan = plan_task2_fno_candidates(
        evidence_paths=evidence_paths,
        output_path=output_path,
        max_candidates=max_candidates,
        hidden_channels_values=hidden_channels_values,
        modes_values=modes_values,
        depths_values=depths_values,
        physics_weights_values=physics_weights_values,
        learning_rates_values=learning_rates_values,
        train_steps_values=train_steps_values,
        batch_sizes_values=batch_sizes_values,
        force=force,
    )
    artifacts: dict[str, Any] = {"plan": plan}
    if output_path is not None:
        artifacts["output_path"] = output_path
    return success_payload(
        run_id="task2-fno-plan",
        command="plan-task2-fno-candidates",
        state_hint=plan["state_hint"],
        artifacts=artifacts,
        metrics={"candidate_count": plan["candidate_count"]},
        next_hint=plan["next_hint"],
    )


def run_task2_fno_optimization_loop_command(
    *,
    candidate_plan_path: str,
    output_dir: str = "/tmp/task2_fno_optimization",
    device: str = "mps",
    force: bool = False,
) -> dict[str, Any]:
    """Run Task 2 FNO optimization loop."""
    report = run_task2_fno_optimization_loop(
        candidate_plan_path=candidate_plan_path,
        output_dir=output_dir,
        device=device,
        force=force,
    )
    best_record = report.get("selected_candidate")
    best_score = report.get("best_score")
    return success_payload(
        run_id=Path(output_dir).name or "task2-fno-optimization",
        command="run-task2-fno-optimization-loop",
        state_hint=report["state_hint"],
        artifacts={
            "output_dir": output_dir,
            "report_path": str(Path(output_dir) / "task2_fno_optimization_report.json"),
        },
        metrics={
            "candidate_count": report.get("candidate_count", 0),
            "completed_count": report.get("completed_count", 0),
            "best_score": best_score,
        },
        next_hint="review best candidate and run next iteration or convergence check",
    )


# -- Closed loop commands --


def run_task2_fno_training_iteration_command(
    *,
    workspace_dir: str,
    memory_path: str | None = None,
    evidence_paths: list[str] | None = None,
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
    hypothesis_memory_path: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run one Task 2 FNO training closed-loop iteration."""
    return run_task2_fno_training_iteration(
        workspace_dir=workspace_dir,
        memory_path=memory_path,
        evidence_paths=evidence_paths,
        max_candidates=max_candidates,
        hidden_channels_values=hidden_channels_values,
        modes_values=modes_values,
        depths_values=depths_values,
        physics_weights_values=physics_weights_values,
        learning_rates_values=learning_rates_values,
        train_steps_values=train_steps_values,
        batch_sizes_values=batch_sizes_values,
        target_score=target_score,
        patience=patience,
        device=device,
        checkpoint_every=checkpoint_every,
        force=force,
        hypothesis_memory_path=hypothesis_memory_path,
        session_id=session_id,
    )


def build_task2_fno_training_contract_command(
    *,
    workspace_dir: str,
    output_path: str | None = None,
    required_commands: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build Task 2 FNO training closed-loop contract."""
    contract = build_task2_fno_training_closed_loop_contract(
        workspace_dir=workspace_dir,
        output_path=output_path,
        required_commands=required_commands,
        force=force,
    )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="build-task2-fno-training-contract",
        state_hint=contract["state_hint"],
        artifacts={
            "contract_path": output_path,
            "workspace_dir": workspace_dir,
        },
        metrics={"required_command_count": len(contract["required_commands"])},
        next_hint="run-task2-fno-training-iteration",
    )


def check_task2_fno_training_convergence_command(
    *,
    memory_path: str,
    target_score: float | None = None,
    patience: int = 3,
    output_path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Check convergence from memory file."""
    from pde_agent.core.json_output import error_payload

    memory = _load_json(Path(memory_path))
    if memory is None:
        return error_payload(
            run_id=Path(memory_path).stem or "local",
            command="check-task2-fno-training-convergence",
            error_code="missing_memory",
            error_type="Task2FnoTrainingClosedLoopError",
            message_short="could not load training iteration memory",
            recoverable=True,
            retryable=True,
        )
    result = check_task2_fno_training_convergence(
        memory=memory,
        target_score=target_score,
        patience=patience,
    )
    if output_path is not None:
        out = Path(output_path)
        if out.exists() and not force:
            return error_payload(
                run_id=Path(memory_path).stem or "local",
                command="check-task2-fno-training-convergence",
                error_code="output_exists",
                error_type="Task2FnoTrainingClosedLoopError",
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
        command="check-task2-fno-training-convergence",
        state_hint=result["state_hint"],
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


def build_task2_fno_training_memory_command(
    *,
    optimization_report_path: str,
    iteration_index: int,
    previous_memory_path: str | None = None,
    output_path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build iteration memory from optimization report."""
    from pde_agent.core.json_output import error_payload

    report = _load_json(Path(optimization_report_path))
    if report is None:
        return error_payload(
            run_id=f"iter-{iteration_index}",
            command="build-task2-fno-training-memory",
            error_code="missing_report",
            error_type="Task2FnoTrainingClosedLoopError",
            message_short=f"missing report: {optimization_report_path}",
            recoverable=True,
            retryable=True,
        )
    previous_memory = None
    if previous_memory_path is not None:
        previous_memory = _load_json(Path(previous_memory_path))
    memory = build_task2_fno_training_memory(
        optimization_report=report,
        iteration_index=iteration_index,
        previous_memory=previous_memory,
        source_optimization_report_path=optimization_report_path,
    )
    if output_path is not None:
        out = Path(output_path)
        if out.exists() and not force:
            return error_payload(
                run_id=f"iter-{iteration_index}",
                command="build-task2-fno-training-memory",
                error_code="output_exists",
                error_type="Task2FnoTrainingClosedLoopError",
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
        command="build-task2-fno-training-memory",
        state_hint=memory["state_hint"],
        artifacts={
            "optimization_report_path": optimization_report_path,
            "memory_path": output_path,
        },
        metrics={
            "iteration_index": iteration_index,
            "trajectory_count": len(memory.get("trajectory_summary", [])),
        },
        next_hint="check-task2-fno-training-convergence",
    )


def build_task2_fno_training_final_report_command(
    *,
    workspace_dir: str,
    memory_path: str,
    output_path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build final report from best memory."""
    from pde_agent.core.json_output import error_payload

    try:
        report = build_task2_fno_training_final_report(
            memory_path=memory_path,
            workspace_dir=workspace_dir,
            output_path=output_path,
            force=force,
        )
    except Task2FnoTrainingClosedLoopError as exc:
        return error_payload(
            run_id=Path(workspace_dir).name or "local",
            command="build-task2-fno-training-final-report",
            error_code="final_report_failed",
            error_type="Task2FnoTrainingClosedLoopError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="build-task2-fno-training-final-report",
        state_hint=report["state_hint"],
        artifacts={
            "memory_path": memory_path,
            "final_report_path": output_path or str(Path(workspace_dir) / "task2_fno_training_final_report.json"),
        },
        metrics={
            "best_score": report.get("best_candidate", {}).get("score"),
            "iteration_count": report.get("provenance", {}).get("iteration_count"),
        },
        next_hint="review final report and promote candidate if satisfied",
    )


def build_task2_fno_training_health_report_command(
    *,
    workspace_dir: str,
    output_path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build health report across all iterations."""
    from pde_agent.core.json_output import error_payload

    try:
        report = build_task2_fno_training_health_report(
            workspace_dir=workspace_dir,
            output_path=output_path,
            force=force,
        )
    except Task2FnoTrainingClosedLoopError as exc:
        return error_payload(
            run_id=Path(workspace_dir).name or "local",
            command="build-task2-fno-training-health-report",
            error_code="health_report_failed",
            error_type="Task2FnoTrainingClosedLoopError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="build-task2-fno-training-health-report",
        state_hint=report["state_hint"],
        artifacts={
            "health_report_path": output_path or str(Path(workspace_dir) / "task2_fno_training_health_report.json"),
        },
        metrics={
            "iteration_count": report.get("iteration_count"),
            "global_best": report.get("score", {}).get("global_best"),
            "health_status": report.get("health", {}).get("status"),
            "issue_count": len(report.get("health", {}).get("issues", [])),
        },
        next_hint=report.get("next_hint"),
    )


# -- Checkpoint commands --


def checkpoint_task2_fno_training_loop_command(
    *,
    workspace_dir: str,
    iteration_index: int,
    memory_path: str,
    convergence_path: str,
    force: bool = False,
) -> dict[str, Any]:
    """Save checkpoint of current Task 2 training loop state."""
    from pde_agent.core.json_output import error_payload
    from pde_agent.core.task2_fno_training_checkpoint import (
        Task2FnoTrainingCheckpointError,
        checkpoint_task2_training_loop,
    )

    memory = _load_json(Path(memory_path))
    if memory is None:
        return error_payload(
            run_id=f"iter-{iteration_index}",
            command="checkpoint-task2-fno-training-loop",
            error_code="missing_memory",
            error_type="Task2FnoTrainingCheckpointError",
            message_short="could not load training iteration memory",
            recoverable=True,
            retryable=True,
        )
    convergence = _load_json(Path(convergence_path))
    if convergence is None:
        convergence = {}
    try:
        checkpoint_path = checkpoint_task2_training_loop(
            workspace_dir=workspace_dir,
            iteration_index=iteration_index,
            memory=memory,
            convergence=convergence,
            best_config=memory.get("best_so_far", {}).get("config"),
            trajectory=memory.get("trajectory_summary", []),
            force=force,
        )
    except Task2FnoTrainingCheckpointError as exc:
        return error_payload(
            run_id=f"iter-{iteration_index}",
            command="checkpoint-task2-fno-training-loop",
            error_code="checkpoint_failed",
            error_type="Task2FnoTrainingCheckpointError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=f"iter-{iteration_index}",
        command="checkpoint-task2-fno-training-loop",
        state_hint="TASK2_FNO_TRAINING_CHECKPOINTED",
        artifacts={
            "checkpoint_path": str(checkpoint_path),
        },
        metrics={
            "iteration_index": iteration_index,
        },
        next_hint="continue training or resume from checkpoint if needed",
    )


def resume_task2_fno_training_loop_command(
    *,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Resume Task 2 training loop from checkpoint."""
    from pde_agent.core.json_output import error_payload
    from pde_agent.core.task2_fno_training_checkpoint import (
        Task2FnoTrainingCheckpointError,
        resume_task2_training_loop,
    )

    try:
        state = resume_task2_training_loop(checkpoint_path=checkpoint_path)
    except Task2FnoTrainingCheckpointError as exc:
        return error_payload(
            run_id=Path(checkpoint_path).stem or "local",
            command="resume-task2-fno-training-loop",
            error_code="resume_failed",
            error_type="Task2FnoTrainingCheckpointError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=Path(checkpoint_path).stem or "local",
        command="resume-task2-fno-training-loop",
        state_hint="TASK2_FNO_TRAINING_RESUMED",
        artifacts={
            "checkpoint_path": checkpoint_path,
        },
        metrics={
            "iteration_index": state["iteration_index"],
        },
        next_hint="run-task2-fno-training-iteration from resumed state",
    )


def escalate_task2_fno_training_to_hypothesis_command(
    *,
    workspace_dir: str,
    memory_path: str,
    output_path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Escalate Task 2 training to hypothesis lab for strategy discovery."""
    try:
        report = run_task2_fno_training_escalation_pipeline(
            workspace_dir=workspace_dir,
            memory_path=memory_path,
            output_path=output_path,
            force=force,
        )
    except Task2FnoTrainingEscalationError as exc:
        return error_payload(
            run_id=Path(workspace_dir).name or "local",
            command="escalate-task2-fno-training-to-hypothesis",
            error_code="escalation_failed",
            error_type="Task2FnoTrainingEscalationError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="escalate-task2-fno-training-to-hypothesis",
        state_hint=report["state_hint"],
        artifacts={
            "memory_path": memory_path,
            "escalation_report_path": output_path
            or str(Path(workspace_dir) / "task2_fno_training_escalation_report.json"),
        },
        metrics={
            "recommendation_count": len(report.get("strategy_recommendations", [])),
            "failure_mode_count": len(report.get("failure_analysis", {}).get("failure_modes", [])),
        },
        next_hint=report.get("next_command", "run-task2-fno-training-iteration"),
    )


def integrate_hypothesis_training_feedback_command(
    *,
    hypothesis_report_path: str,
    memory_path: str,
    output_path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Merge hypothesis lab feedback into Task 2 training memory."""
    memory = load_task2_fno_training_memory(memory_path)
    if memory is None:
        return error_payload(
            run_id=Path(memory_path).stem or "local",
            command="integrate-hypothesis-training-feedback",
            error_code="missing_memory",
            error_type="Task2FnoTrainingMemoryError",
            message_short="could not load training iteration memory",
            recoverable=True,
            retryable=True,
        )
    hypothesis_report = _load_json(Path(hypothesis_report_path))
    merged = merge_task2_training_memory_with_hypothesis_feedback(
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
                error_type="Task2FnoTrainingMemoryError",
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
        next_hint="run-task2-fno-training-iteration with updated memory",
    )


def build_claude_task2_fno_training_task_command(
    *,
    workspace_dir: str,
    output_dir: str | None = None,
    max_iterations: int = 10,
    target_score: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build Claude agent task prompt and contract for Task 2 closed loop."""
    from pde_agent.core.task2_fno_training_agent_interface import (
        Task2FnoTrainingAgentInterfaceError,
        build_claude_task2_fno_training_closed_loop_task,
    )

    try:
        result = build_claude_task2_fno_training_closed_loop_task(
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            max_iterations=max_iterations,
            target_score=target_score,
            force=force,
        )
    except Task2FnoTrainingAgentInterfaceError as exc:
        return error_payload(
            run_id=Path(workspace_dir).name or "local",
            command="build-claude-task2-fno-training-task",
            error_code="task_build_failed",
            error_type="Task2FnoTrainingAgentInterfaceError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="build-claude-task2-fno-training-task",
        state_hint=result["state_hint"],
        artifacts=result.get("artifacts", {}),
        metrics={
            "max_iterations": result.get("max_iterations"),
            "target_score": result.get("target_score"),
        },
        next_hint="start the autonomous loop using the generated prompt",
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
