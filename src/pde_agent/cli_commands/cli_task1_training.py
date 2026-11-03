"""CLI commands for task1_training domain."""

import json
import time

import typer

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

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("run-task1-short-window-training-loop")
    def run_task1_short_window_training_loop_cmd(
        source_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        train_sample_limit: int = typer.Option(64),
        val_sample_limit: int = typer.Option(32),
        context_steps: int = typer.Option(10),
        forecast_steps: int = typer.Option(47),
        time_reduction: int = typer.Option(1),
        space_reduction: int = typer.Option(4),
        model_family: str = typer.Option("conv1d_residual"),
        fno_depth: int = typer.Option(1),
        fno_width: int = typer.Option(32),
        fno_modes: int = typer.Option(8),
        loss_type: str = typer.Option("mse"),
        normalization: str = typer.Option("none"),
        train_steps: int = typer.Option(8),
        batch_size: int = typer.Option(16),
        learning_rate: float = typer.Option(1e-3),
        seed: int = typer.Option(0),
        noise_sigma: float = typer.Option(0.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_task1_short_window_training_loop_command(
                    source_path=source_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    train_sample_limit=train_sample_limit,
                    val_sample_limit=val_sample_limit,
                    context_steps=context_steps,
                    forecast_steps=forecast_steps,
                    time_reduction=time_reduction,
                    space_reduction=space_reduction,
                    model_family=model_family,
                    fno_depth=fno_depth,
                    fno_width=fno_width,
                    fno_modes=fno_modes,
                    loss_type=loss_type,
                    normalization=normalization,
                    train_steps=train_steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=seed,
                    noise_sigma=noise_sigma,
                    force=force,
                )
            )
        )

    @app.command("run-task1-short-window-family-selection-loop")
    def run_task1_short_window_family_selection_loop_cmd(
        source_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        model_families: list[str] = typer.Option(
            ["conv1d_residual", "fno_lite"],
            "--model-family",
        ),
        train_sample_limit: int = typer.Option(64),
        val_sample_limit: int = typer.Option(32),
        context_steps: int = typer.Option(10),
        forecast_steps: int = typer.Option(47),
        time_reduction: int = typer.Option(1),
        space_reduction: int = typer.Option(4),
        train_steps: int = typer.Option(8),
        batch_size: int = typer.Option(16),
        learning_rate: float = typer.Option(1e-3),
        seed: int = typer.Option(0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_task1_short_window_family_selection_loop_command(
                    source_path=source_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    model_families=model_families,
                    train_sample_limit=train_sample_limit,
                    val_sample_limit=val_sample_limit,
                    context_steps=context_steps,
                    forecast_steps=forecast_steps,
                    time_reduction=time_reduction,
                    space_reduction=space_reduction,
                    train_steps=train_steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=seed,
                    force=force,
                )
            )
        )
