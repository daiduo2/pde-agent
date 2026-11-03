"""CLI commands for Task 2 FNO training domain."""

from __future__ import annotations

import json

import typer

from pde_agent.commands.task2_fno_training import (
    build_claude_task2_fno_training_task_command,
    build_task2_fno_training_contract_command,
    build_task2_fno_training_final_report_command,
    build_task2_fno_training_health_report_command,
    build_task2_fno_training_memory_command,
    checkpoint_task2_fno_training_loop_command,
    check_task2_fno_training_convergence_command,
    escalate_task2_fno_training_to_hypothesis_command,
    integrate_hypothesis_training_feedback_command,
    plan_task2_fno_candidates_command,
    resume_task2_fno_training_loop_command,
    run_task2_fno_optimization_loop_command,
    run_task2_fno_training_command,
    run_task2_fno_training_iteration_command,
)
from pde_agent.commands.task2_fno_graft_evaluator import (
    run_task2_fno_graft_evaluator_command,
)


def register_commands(app: typer.Typer) -> None:
    """Register Task 2 FNO training commands."""

    @app.command("run-task2-fno-training")
    def run_task2_fno_training_cmd(
        hidden_channels: int = typer.Option(96),
        modes: int = typer.Option(48),
        depth: int = typer.Option(5),
        batch_size: int = typer.Option(32),
        train_steps: int = typer.Option(10000),
        learning_rate: float = typer.Option(1e-3),
        physics_weight: float = typer.Option(0.01),
        device: str = typer.Option("mps"),
        seed: int = typer.Option(42),
        output_dir: str = typer.Option("/tmp/task2_fno_training"),
        checkpoint_path: str | None = typer.Option(None),
        resume_lr: float | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        # V2 model flags
        use_v2: bool = typer.Option(False, "--use-v2"),
        use_film: bool = typer.Option(False, "--use-film"),
        use_coord: bool = typer.Option(False, "--use-coord"),
        use_residual: bool = typer.Option(False, "--use-residual"),
        use_groupnorm: bool = typer.Option(False, "--use-groupnorm"),
        dropout: float = typer.Option(0.0, "--dropout"),
        # Forecast steps
        forecast_steps: int = typer.Option(190, "--forecast-steps"),
        # Auxiliary loss weights
        time_weight: float = typer.Option(0.0, "--time-weight"),
        grad_weight: float = typer.Option(0.0, "--grad-weight"),
        temporal_weight: float = typer.Option(0.0, "--temporal-weight"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_task2_fno_training_command(
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
            )
        )

    @app.command("plan-task2-fno-candidates")
    def plan_task2_fno_candidates_cmd(
        evidence_paths: list[str] = typer.Option([], "--evidence-path"),
        output_path: str | None = typer.Option(None),
        max_candidates: int = typer.Option(8),
        hidden_channels_values: list[int] = typer.Option([], "--hidden-channels"),
        modes_values: list[int] = typer.Option([], "--mode"),
        depths_values: list[int] = typer.Option([], "--depth"),
        physics_weights_values: list[float] = typer.Option([], "--physics-weight"),
        learning_rates_values: list[float] = typer.Option([], "--learning-rate"),
        train_steps_values: list[int] = typer.Option([], "--train-steps"),
        batch_sizes_values: list[int] = typer.Option([], "--batch-size"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_task2_fno_candidates_command(
                    evidence_paths=evidence_paths or None,
                    output_path=output_path,
                    max_candidates=max_candidates,
                    hidden_channels_values=hidden_channels_values or None,
                    modes_values=modes_values or None,
                    depths_values=depths_values or None,
                    physics_weights_values=physics_weights_values or None,
                    learning_rates_values=learning_rates_values or None,
                    train_steps_values=train_steps_values or None,
                    batch_sizes_values=batch_sizes_values or None,
                    force=force,
                )
            )
        )

    @app.command("run-task2-fno-optimization-loop")
    def run_task2_fno_optimization_loop_cmd(
        candidate_plan_path: str = typer.Option(...),
        output_dir: str = typer.Option("/tmp/task2_fno_optimization"),
        device: str = typer.Option("mps"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_task2_fno_optimization_loop_command(
                    candidate_plan_path=candidate_plan_path,
                    output_dir=output_dir,
                    device=device,
                    force=force,
                )
            )
        )

    # -- Closed loop commands --

    @app.command("run-task2-fno-training-iteration")
    def run_task2_fno_training_iteration_cmd(
        workspace_dir: str = typer.Option(...),
        memory_path: str | None = typer.Option(None),
        evidence_paths: list[str] = typer.Option([], "--evidence-path"),
        max_candidates: int = typer.Option(4),
        hidden_channels_values: list[int] = typer.Option([], "--hidden-channels"),
        modes_values: list[int] = typer.Option([], "--mode"),
        depths_values: list[int] = typer.Option([], "--depth"),
        physics_weights_values: list[float] = typer.Option([], "--physics-weight"),
        learning_rates_values: list[float] = typer.Option([], "--learning-rate"),
        train_steps_values: list[int] = typer.Option([], "--train-steps"),
        batch_sizes_values: list[int] = typer.Option([], "--batch-size"),
        target_score: float | None = typer.Option(None),
        patience: int = typer.Option(3),
        device: str = typer.Option("mps"),
        checkpoint_every: int = typer.Option(1),
        force: bool = typer.Option(False, "--force"),
        hypothesis_memory_path: str | None = typer.Option(None, "--hypothesis-memory-path"),
        session_id: str | None = typer.Option(None, "--session-id"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_task2_fno_training_iteration_command(
                    workspace_dir=workspace_dir,
                    memory_path=memory_path,
                    evidence_paths=evidence_paths or None,
                    max_candidates=max_candidates,
                    hidden_channels_values=hidden_channels_values or None,
                    modes_values=modes_values or None,
                    depths_values=depths_values or None,
                    physics_weights_values=physics_weights_values or None,
                    learning_rates_values=learning_rates_values or None,
                    train_steps_values=train_steps_values or None,
                    batch_sizes_values=batch_sizes_values or None,
                    target_score=target_score,
                    patience=patience,
                    device=device,
                    checkpoint_every=checkpoint_every,
                    force=force,
                    hypothesis_memory_path=hypothesis_memory_path,
                    session_id=session_id,
                )
            )
        )

    @app.command("build-task2-fno-training-contract")
    def build_task2_fno_training_contract_cmd(
        workspace_dir: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        required_commands: list[str] = typer.Option([], "--required-command"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_task2_fno_training_contract_command(
                    workspace_dir=workspace_dir,
                    output_path=output_path,
                    required_commands=required_commands or None,
                    force=force,
                )
            )
        )

    @app.command("check-task2-fno-training-convergence")
    def check_task2_fno_training_convergence_cmd(
        memory_path: str = typer.Option(...),
        target_score: float | None = typer.Option(None),
        patience: int = typer.Option(3),
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                check_task2_fno_training_convergence_command(
                    memory_path=memory_path,
                    target_score=target_score,
                    patience=patience,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-task2-fno-training-memory")
    def build_task2_fno_training_memory_cmd(
        optimization_report_path: str = typer.Option(...),
        iteration_index: int = typer.Option(...),
        previous_memory_path: str | None = typer.Option(None),
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_task2_fno_training_memory_command(
                    optimization_report_path=optimization_report_path,
                    iteration_index=iteration_index,
                    previous_memory_path=previous_memory_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-task2-fno-training-final-report")
    def build_task2_fno_training_final_report_cmd(
        workspace_dir: str = typer.Option(...),
        memory_path: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_task2_fno_training_final_report_command(
                    workspace_dir=workspace_dir,
                    memory_path=memory_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-task2-fno-training-health-report")
    def build_task2_fno_training_health_report_cmd(
        workspace_dir: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_task2_fno_training_health_report_command(
                    workspace_dir=workspace_dir,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("checkpoint-task2-fno-training-loop")
    def checkpoint_task2_fno_training_loop_cmd(
        workspace_dir: str = typer.Option(...),
        iteration_index: int = typer.Option(...),
        memory_path: str = typer.Option(...),
        convergence_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                checkpoint_task2_fno_training_loop_command(
                    workspace_dir=workspace_dir,
                    iteration_index=iteration_index,
                    memory_path=memory_path,
                    convergence_path=convergence_path,
                    force=force,
                )
            )
        )

    @app.command("resume-task2-fno-training-loop")
    def resume_task2_fno_training_loop_cmd(
        checkpoint_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                resume_task2_fno_training_loop_command(
                    checkpoint_path=checkpoint_path,
                )
            )
        )

    @app.command("escalate-task2-fno-training-to-hypothesis")
    def escalate_task2_fno_training_to_hypothesis_cmd(
        workspace_dir: str = typer.Option(...),
        memory_path: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                escalate_task2_fno_training_to_hypothesis_command(
                    workspace_dir=workspace_dir,
                    memory_path=memory_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("integrate-hypothesis-training-feedback")
    def integrate_hypothesis_training_feedback_cmd(
        hypothesis_report_path: str = typer.Option(...),
        memory_path: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                integrate_hypothesis_training_feedback_command(
                    hypothesis_report_path=hypothesis_report_path,
                    memory_path=memory_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-claude-task2-fno-training-task")
    def build_claude_task2_fno_training_task_cmd(
        workspace_dir: str = typer.Option(...),
        output_dir: str | None = typer.Option(None),
        max_iterations: int = typer.Option(10),
        target_score: float | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_claude_task2_fno_training_task_command(
                    workspace_dir=workspace_dir,
                    output_dir=output_dir,
                    max_iterations=max_iterations,
                    target_score=target_score,
                    force=force,
                )
            )
        )

    @app.command("run-task2-fno-graft-evaluator")
    def run_task2_fno_graft_evaluator_cmd(
        base_checkpoint_path: str = typer.Option(...),
        front_checkpoint_path: str = typer.Option(...),
        output_dir: str = typer.Option("/tmp/task2_fno_graft_eval"),
        val_data_path: str | None = typer.Option(None),
        window_start: list[int] = typer.Option([], "--window-start"),
        window_stop: list[int] = typer.Option([], "--window-stop"),
        blend_weights: list[float] = typer.Option([], "--blend-weight"),
        residual_scales: list[float] = typer.Option([], "--residual-scale"),
        late_non_regression_tolerance: float = typer.Option(1.0),
        device: str = typer.Option("mps"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        windows = None
        if window_start and window_stop:
            windows = list(zip(window_start, window_stop))
        blend_weights_list = blend_weights if blend_weights else None
        residual_scales_list = residual_scales if residual_scales else None
        typer.echo(
            json.dumps(
                run_task2_fno_graft_evaluator_command(
                    base_checkpoint_path=base_checkpoint_path,
                    front_checkpoint_path=front_checkpoint_path,
                    output_dir=output_dir,
                    val_data_path=val_data_path,
                    windows=windows,
                    blend_weights=blend_weights_list,
                    residual_scales=residual_scales_list,
                    late_non_regression_tolerance=late_non_regression_tolerance,
                    device=device,
                    force=force,
                )
            )
        )
