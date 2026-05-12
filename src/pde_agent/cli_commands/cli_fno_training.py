"""CLI commands for fno_training domain."""

import json
import time

import typer

from pde_agent.commands.fno_training_closed_loop import (
    build_fno_training_closed_loop_contract_command,
    build_fno_training_final_report_command,
    build_fno_training_health_report_command,
    build_fno_training_memory_command,
    checkpoint_fno_training_loop_command,
    check_fno_training_convergence_command,
    escalate_fno_training_to_hypothesis_command,
    integrate_hypothesis_training_feedback_command,
    resume_fno_training_loop_command,
    run_fno_training_iteration_command,
)
from pde_agent.commands.fno_training_agent_loop import (
    build_claude_fno_training_closed_loop_task_command,
    build_fno_training_tool_catalog_command,
    run_claude_fno_training_closed_loop_command,
)
from pde_agent.commands.hypothesis_lab_runtime import (
    adapt_hypothesis_algorithm_sandbox_item_command,
    audit_agent_runtime_availability_command,
    build_fno_training_harness_contract_command,
    build_hypothesis_algorithm_design_command,
    build_hypothesis_kpi_report_command,
    build_scientific_tool_catalog_command,
    detect_hypothesis_doom_loop_command,
    plan_fno_training_candidates_command,
    plan_hypothesis_algorithm_sandbox_command,
    plan_hypothesis_candidates_command,
    prepare_pde_literature_extraction_skill_command,
    record_fno_training_optimization_evidence_command,
    record_hypothesis_algorithm_evidence_command,
    record_hypothesis_lab_trace_event_command,
    run_fno_training_optimization_loop_command,
    run_hypothesis_algorithm_sandbox_command,
    run_hypothesis_algorithm_designer_stability_batch_command,
    run_task1_short_window_family_selection_loop_command,
    run_task1_short_window_training_loop_command,
    summarize_hypothesis_lab_trace_command,
    validate_hypothesis_algorithm_design_command,
)
from pde_agent.commands.task1_candidate_factory import (
    evaluate_candidate_factory_capability_command,
    generate_task1_candidate_families_command,
    materialize_candidate_family_item_command,
    run_candidate_family_sandbox_command,
)
from pde_agent.commands.task1_fno_graft_evaluator import (
    run_task1_fno_graft_from_optimization_report_command,
    run_task1_fno_graft_evaluator_command,
)

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("build-fno-training-harness-contract")
    def build_fno_training_harness_contract_cmd(
        output_path: str = typer.Option(...),
        workspace_dir: str = typer.Option(...),
        source_path: str = typer.Option(...),
        baseline_prediction_path: str | None = typer.Option(None),
        task1_test_path: str | None = typer.Option(None),
        task1_target_path: str | None = typer.Option(None),
        previous_evidence_paths: list[str] = typer.Option([], "--previous-evidence-path"),
        dataset_key: str = typer.Option("tensor"),
        train_sample_limit: int = typer.Option(16),
        val_sample_limit: int = typer.Option(8),
        context_steps: int = typer.Option(10),
        time_reduction: int = typer.Option(5),
        space_reduction: int = typer.Option(4),
        batch_size: int = typer.Option(4),
        learning_rate: float = typer.Option(1e-3),
        seed: int = typer.Option(0),
        max_candidates: int = typer.Option(3),
        modes: list[int] = typer.Option([4, 8], "--mode"),
        widths: list[int] = typer.Option([16, 32], "--width"),
        depths: list[int] = typer.Option([1], "--depth"),
        loss_types: list[str] = typer.Option(
            ["mse", "early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
            "--loss-type",
        ),
        normalizations: list[str] = typer.Option(["none"], "--normalization"),
        forecast_steps_values: list[int] = typer.Option(
            [12, 20],
            "--forecast-steps-value",
        ),
        train_steps_values: list[int] = typer.Option([1, 2], "--train-steps-value"),
        search_guidance: list[str] = typer.Option([], "--search-guidance"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_fno_training_harness_contract_command(
                    output_path=output_path,
                    workspace_dir=workspace_dir,
                    source_path=source_path,
                    baseline_prediction_path=baseline_prediction_path,
                    task1_test_path=task1_test_path,
                    task1_target_path=task1_target_path,
                    previous_evidence_paths=previous_evidence_paths,
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
                    normalizations=normalizations,
                    forecast_steps_values=forecast_steps_values,
                    train_steps_values=train_steps_values,
                    search_guidance=search_guidance,
                    force=force,
                )
            )
        )

    @app.command("plan-fno-training-candidates")
    def plan_fno_training_candidates_cmd(
        evidence_paths: list[str] = typer.Option([], "--evidence-path"),
        output_path: str = typer.Option(...),
        max_candidates: int = typer.Option(4),
        modes: list[int] = typer.Option([4, 8], "--mode"),
        widths: list[int] = typer.Option([16, 32], "--width"),
        depths: list[int] = typer.Option([1], "--depth"),
        loss_types: list[str] = typer.Option(
            ["mse", "early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
            "--loss-type",
        ),
        normalizations: list[str] = typer.Option(["none"], "--normalization"),
        forecast_steps_values: list[int] = typer.Option(
            [12, 20],
            "--forecast-steps-value",
        ),
        train_steps_values: list[int] = typer.Option([1, 2], "--train-steps-value"),
        search_guidance: list[str] = typer.Option([], "--search-guidance"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_fno_training_candidates_command(
                    evidence_paths=evidence_paths,
                    output_path=output_path,
                    max_candidates=max_candidates,
                    modes=modes,
                    widths=widths,
                    depths=depths,
                    loss_types=loss_types,
                    normalizations=normalizations,
                    forecast_steps_values=forecast_steps_values,
                    train_steps_values=train_steps_values,
                    search_guidance=search_guidance,
                    force=force,
                )
            )
        )

    @app.command("run-fno-training-optimization-loop")
    def run_fno_training_optimization_loop_cmd(
        candidate_plan_path: str = typer.Option(...),
        source_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        train_sample_limit: int = typer.Option(64),
        val_sample_limit: int = typer.Option(32),
        context_steps: int = typer.Option(10),
        time_reduction: int = typer.Option(1),
        space_reduction: int = typer.Option(4),
        batch_size: int = typer.Option(16),
        learning_rate: float = typer.Option(1e-3),
        seed: int = typer.Option(0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_fno_training_optimization_loop_command(
                    candidate_plan_path=candidate_plan_path,
                    source_path=source_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    train_sample_limit=train_sample_limit,
                    val_sample_limit=val_sample_limit,
                    context_steps=context_steps,
                    time_reduction=time_reduction,
                    space_reduction=space_reduction,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=seed,
                    force=force,
                )
            )
        )

    @app.command("run-task1-fno-graft-evaluator")
    def run_task1_fno_graft_evaluator_cmd(
        baseline_prediction_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        test_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        prediction_dataset_key: str = typer.Option("tensor"),
        target_dataset_key: str = typer.Option("tensor"),
        windows: list[str] = typer.Option([], "--window"),
        blend_weights: list[float] = typer.Option([], "--blend-weight"),
        late_non_regression_tolerance: float = typer.Option(0.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_task1_fno_graft_evaluator_command(
                    baseline_prediction_path=baseline_prediction_path,
                    checkpoint_path=checkpoint_path,
                    test_path=test_path,
                    target_path=target_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    prediction_dataset_key=prediction_dataset_key,
                    target_dataset_key=target_dataset_key,
                    windows=windows,
                    blend_weights=blend_weights,
                    late_non_regression_tolerance=late_non_regression_tolerance,
                    force=force,
                )
            )
        )

    @app.command("run-task1-fno-graft-from-optimization-report")
    def run_task1_fno_graft_from_optimization_report_cmd(
        optimization_report_path: str = typer.Option(...),
        baseline_prediction_path: str = typer.Option(...),
        test_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        prediction_dataset_key: str = typer.Option("tensor"),
        target_dataset_key: str = typer.Option("tensor"),
        windows: list[str] = typer.Option([], "--window"),
        blend_weights: list[float] = typer.Option([], "--blend-weight"),
        late_non_regression_tolerance: float = typer.Option(0.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_task1_fno_graft_from_optimization_report_command(
                    optimization_report_path=optimization_report_path,
                    baseline_prediction_path=baseline_prediction_path,
                    test_path=test_path,
                    target_path=target_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    prediction_dataset_key=prediction_dataset_key,
                    target_dataset_key=target_dataset_key,
                    windows=windows,
                    blend_weights=blend_weights,
                    late_non_regression_tolerance=late_non_regression_tolerance,
                    force=force,
                )
            )
        )

    @app.command("record-fno-training-optimization-evidence")
    def record_fno_training_optimization_evidence_cmd(
        optimization_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        graft_report_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_fno_training_optimization_evidence_command(
                    optimization_report_path=optimization_report_path,
                    output_path=output_path,
                    graft_report_path=graft_report_path,
                    force=force,
                )
            )
        )

    @app.command("run-fno-training-iteration")
    def run_fno_training_iteration_cmd(
        workspace_dir: str = typer.Option(...),
        source_path: str = typer.Option(...),
        memory_path: str | None = typer.Option(None),
        evidence_path: list[str] | None = typer.Option(None, "--evidence-path"),
        dataset_key: str = typer.Option("tensor"),
        train_sample_limit: int = typer.Option(64),
        val_sample_limit: int = typer.Option(32),
        context_steps: int = typer.Option(10),
        time_reduction: int = typer.Option(5),
        space_reduction: int = typer.Option(4),
        batch_size: int = typer.Option(16),
        learning_rate: float = typer.Option(1e-3),
        seed: int = typer.Option(0),
        max_candidates: int = typer.Option(3),
        mode: list[int] | None = typer.Option(None, "--mode"),
        width: list[int] | None = typer.Option(None, "--width"),
        depth: list[int] | None = typer.Option(None, "--depth"),
        loss_type: list[str] | None = typer.Option(None, "--loss-type"),
        forecast_steps_value: list[int] | None = typer.Option(None, "--forecast-steps-value"),
        train_steps_value: list[int] | None = typer.Option(None, "--train-steps-value"),
        normalization: list[str] | None = typer.Option(None, "--normalization"),
        target_score: float | None = typer.Option(None),
        patience: int = typer.Option(3),
        checkpoint_every: int = typer.Option(1),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_fno_training_iteration_command(
                    workspace_dir=workspace_dir,
                    source_path=source_path,
                    memory_path=memory_path,
                    evidence_paths=evidence_path,
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
                    modes=mode,
                    widths=width,
                    depths=depth,
                    loss_types=loss_type,
                    forecast_steps_values=forecast_steps_value,
                    train_steps_values=train_steps_value,
                    normalizations=normalization,
                    target_score=target_score,
                    patience=patience,
                    checkpoint_every=checkpoint_every,
                    force=force,
                )
            )
        )

    @app.command("build-fno-training-closed-loop-contract")
    def build_fno_training_closed_loop_contract_cmd(
        workspace_dir: str = typer.Option(...),
        source_path: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        dataset_key: str = typer.Option("tensor"),
        baseline_prediction_path: str | None = typer.Option(None),
        task1_test_path: str | None = typer.Option(None),
        task1_target_path: str | None = typer.Option(None),
        required_command: list[str] | None = typer.Option(None, "--required-command"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_fno_training_closed_loop_contract_command(
                    workspace_dir=workspace_dir,
                    source_path=source_path,
                    output_path=output_path,
                    dataset_key=dataset_key,
                    baseline_prediction_path=baseline_prediction_path,
                    task1_test_path=task1_test_path,
                    task1_target_path=task1_target_path,
                    required_commands=required_command,
                    force=force,
                )
            )
        )

    @app.command("check-fno-training-convergence")
    def check_fno_training_convergence_cmd(
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
                check_fno_training_convergence_command(
                    memory_path=memory_path,
                    target_score=target_score,
                    patience=patience,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-fno-training-memory")
    def build_fno_training_memory_cmd(
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
                build_fno_training_memory_command(
                    optimization_report_path=optimization_report_path,
                    iteration_index=iteration_index,
                    previous_memory_path=previous_memory_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("checkpoint-fno-training-loop")
    def checkpoint_fno_training_loop_cmd(
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
                checkpoint_fno_training_loop_command(
                    workspace_dir=workspace_dir,
                    iteration_index=iteration_index,
                    memory_path=memory_path,
                    convergence_path=convergence_path,
                    force=force,
                )
            )
        )

    @app.command("resume-fno-training-loop")
    def resume_fno_training_loop_cmd(
        checkpoint_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                resume_fno_training_loop_command(
                    checkpoint_path=checkpoint_path,
                )
            )
        )

    @app.command("escalate-fno-training-to-hypothesis")
    def escalate_fno_training_to_hypothesis_cmd(
        workspace_dir: str = typer.Option(...),
        memory_path: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        max_strategy_candidates: int = typer.Option(5, "--max-strategy-candidates"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                escalate_fno_training_to_hypothesis_command(
                    workspace_dir=workspace_dir,
                    memory_path=memory_path,
                    output_path=output_path,
                    max_strategy_candidates=max_strategy_candidates,
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

    @app.command("build-fno-training-final-report")
    def build_fno_training_final_report_cmd(
        workspace_dir: str = typer.Option(...),
        memory_path: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_fno_training_final_report_command(
                    workspace_dir=workspace_dir,
                    memory_path=memory_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-fno-training-health-report")
    def build_fno_training_health_report_cmd(
        workspace_dir: str = typer.Option(...),
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_fno_training_health_report_command(
                    workspace_dir=workspace_dir,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-fno-training-tool-catalog")
    def build_fno_training_tool_catalog_cmd(
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_fno_training_tool_catalog_command(
                    output_path=output_path,
                    force=force,
                )
            )
        )
