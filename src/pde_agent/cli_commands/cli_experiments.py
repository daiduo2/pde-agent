"""CLI commands for experiments domain."""

import json
import time

import typer

from pde_agent.commands.audit_experiment_config import audit_experiment_config
from pde_agent.commands.convert_fno_evidence_summary import (
    convert_fno_evidence_summary_command,
)
from pde_agent.commands.convert_operator_evidence_summary import (
    convert_operator_evidence_summary_command,
)
from pde_agent.commands.convert_official_comparison_evidence import (
    convert_official_comparison_evidence_command,
)
from pde_agent.commands.generate_experiment_config import generate_experiment_config
from pde_agent.commands.plan_next_experiment import plan_next_experiment
from pde_agent.commands.plan_next_experiment_from_fno_summary import (
    plan_next_experiment_from_fno_summary_command,
)
from pde_agent.commands.run_fno_iteration_dryrun import (
    run_fno_iteration_dryrun_command,
)
from pde_agent.commands.run_fno_iteration import run_fno_iteration_command
from pde_agent.commands.run_fno_evidence_sweep import (
    run_fno_evidence_sweep_command,
)
from pde_agent.commands.run_official_operator_comparison import (
    run_official_operator_comparison_command,
)
from pde_agent.commands.run_deeponet_strategy_sweep import (
    run_deeponet_strategy_sweep_command,
)
from pde_agent.commands.run_burgers_spectral_viscosity_benchmark import (
    run_burgers_spectral_viscosity_benchmark_command,
)
from pde_agent.commands.run_unet_strategy_sweep import run_unet_strategy_sweep_command

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("run-fno-iteration-dryrun")
    def run_fno_iteration_dryrun_cmd(
        windows_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_fno_iteration_dryrun_command(
                    windows_path=windows_path,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
            )
        )

    @app.command("run-fno-iteration")
    def run_fno_iteration_cmd(
        windows_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        max_train_steps: int = typer.Option(16),
        learning_rate: float = typer.Option(1e-4),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_fno_iteration_command(
                    windows_path=windows_path,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    max_train_steps=max_train_steps,
                    learning_rate=learning_rate,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
            )
        )

    @app.command("run-fno-evidence-sweep")
    def run_fno_evidence_sweep_cmd(
        source_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        start_indices: list[int] = typer.Option(..., "--start-index"),
        sample_limit: int = typer.Option(...),
        context_steps: int = typer.Option(10),
        forecast_steps: int = typer.Option(...),
        time_reduction: int = typer.Option(5),
        space_reduction: int = typer.Option(4),
        train_steps: list[int] = typer.Option(..., "--train-steps"),
        learning_rate: float = typer.Option(1e-4),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_fno_evidence_sweep_command(
                    source_path=source_path,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    start_indices=start_indices,
                    sample_limit=sample_limit,
                    context_steps=context_steps,
                    forecast_steps=forecast_steps,
                    time_reduction=time_reduction,
                    space_reduction=space_reduction,
                    train_steps=train_steps,
                    learning_rate=learning_rate,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
            )
        )

    @app.command("run-official-operator-comparison")
    def run_official_operator_comparison_cmd(
        source_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        fno_checkpoint_path: str = typer.Option(...),
        unet_checkpoint_path: str = typer.Option(...),
        sample_limit: int = typer.Option(8),
        context_steps: int = typer.Option(10),
        forecast_steps: int = typer.Option(31),
        time_reduction: int = typer.Option(5),
        space_reduction: int = typer.Option(4),
        spectral_modes: int = typer.Option(16),
        spectral_ridge: float = typer.Option(1e-3),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_official_operator_comparison_command(
                    source_path=source_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    fno_checkpoint_path=fno_checkpoint_path,
                    unet_checkpoint_path=unet_checkpoint_path,
                    sample_limit=sample_limit,
                    context_steps=context_steps,
                    forecast_steps=forecast_steps,
                    time_reduction=time_reduction,
                    space_reduction=space_reduction,
                    spectral_modes=spectral_modes,
                    spectral_ridge=spectral_ridge,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
            )
        )

    @app.command("run-deeponet-strategy-sweep")
    def run_deeponet_strategy_sweep_cmd(
        windows_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        baseline_score_path: str = typer.Option(...),
        start_indices: list[int] = typer.Option(..., "--start-index"),
        training_strategies: list[str] = typer.Option(
            ["branch_trunk_supervised"],
            "--training-strategy",
        ),
        train_steps: list[int] = typer.Option(..., "--train-steps"),
        hidden_widths: list[int] = typer.Option(..., "--hidden-width"),
        latent_dims: list[int] = typer.Option(..., "--latent-dim"),
        learning_rates: list[float] = typer.Option(..., "--learning-rate"),
        seeds: list[int] = typer.Option(..., "--seed"),
        spectral_weights: list[float] = typer.Option([0.1], "--spectral-weight"),
        physics_weights: list[float] = typer.Option([0.01], "--physics-weight"),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_deeponet_strategy_sweep_command(
                    windows_path=windows_path,
                    output_dir=output_dir,
                    baseline_score_path=baseline_score_path,
                    start_indices=start_indices,
                    training_strategies=training_strategies,
                    train_steps=train_steps,
                    hidden_widths=hidden_widths,
                    latent_dims=latent_dims,
                    learning_rates=learning_rates,
                    seeds=seeds,
                    spectral_weights=spectral_weights,
                    physics_weights=physics_weights,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
            )
        )

    @app.command("run-unet-strategy-sweep")
    def run_unet_strategy_sweep_cmd(
        windows_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        baseline_score_path: str = typer.Option(...),
        start_indices: list[int] = typer.Option(..., "--start-index"),
        training_strategies: list[str] = typer.Option(
            ["unet_supervised"],
            "--training-strategy",
        ),
        train_steps: list[int] = typer.Option(..., "--train-steps"),
        base_channels: list[int] = typer.Option(..., "--base-channels"),
        depths: list[int] = typer.Option(..., "--depth"),
        learning_rates: list[float] = typer.Option(..., "--learning-rate"),
        seeds: list[int] = typer.Option(..., "--seed"),
        spectral_weights: list[float] = typer.Option([0.1], "--spectral-weight"),
        physics_weights: list[float] = typer.Option([0.01], "--physics-weight"),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_unet_strategy_sweep_command(
                    windows_path=windows_path,
                    output_dir=output_dir,
                    baseline_score_path=baseline_score_path,
                    start_indices=start_indices,
                    training_strategies=training_strategies,
                    train_steps=train_steps,
                    base_channels=base_channels,
                    depths=depths,
                    learning_rates=learning_rates,
                    seeds=seeds,
                    spectral_weights=spectral_weights,
                    physics_weights=physics_weights,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
            )
        )

    @app.command("run-burgers-spectral-viscosity-benchmark")
    def run_burgers_spectral_viscosity_benchmark_cmd(
        source_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        sample_limit: int = typer.Option(8),
        min_evaluation_windows: int = typer.Option(4),
        context_steps: int = typer.Option(10),
        forecast_steps: int = typer.Option(31),
        time_reduction: int = typer.Option(5),
        space_reduction: int = typer.Option(4),
        train_steps: list[int] = typer.Option([4], "--train-steps"),
        hidden_widths: list[int] = typer.Option([8], "--hidden-width"),
        latent_dims: list[int] = typer.Option([4], "--latent-dim"),
        unet_base_channels: list[int] = typer.Option([8], "--unet-base-channels"),
        unet_depths: list[int] = typer.Option([2], "--unet-depth"),
        learning_rates: list[float] = typer.Option([0.01], "--learning-rate"),
        seeds: list[int] = typer.Option([5], "--seed"),
        spectral_weights: list[float] = typer.Option([0.05, 0.1], "--spectral-weight"),
        physics_weights: list[float] = typer.Option([0.005], "--physics-weight"),
        proposal_id: str = typer.Option("burgers_spectral_viscosity_real_hdf5_v0"),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_burgers_spectral_viscosity_benchmark_command(
                    source_path=source_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    sample_limit=sample_limit,
                    min_evaluation_windows=min_evaluation_windows,
                    context_steps=context_steps,
                    forecast_steps=forecast_steps,
                    time_reduction=time_reduction,
                    space_reduction=space_reduction,
                    train_steps=train_steps,
                    hidden_widths=hidden_widths,
                    latent_dims=latent_dims,
                    unet_base_channels=unet_base_channels,
                    unet_depths=unet_depths,
                    learning_rates=learning_rates,
                    seeds=seeds,
                    spectral_weights=spectral_weights,
                    physics_weights=physics_weights,
                    proposal_id=proposal_id,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
            )
        )

    @app.command("convert-fno-evidence-summary")
    def convert_fno_evidence_summary_cmd(
        sweep_summary_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        long_window_threshold: int = typer.Option(31),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                convert_fno_evidence_summary_command(
                    sweep_summary_path=sweep_summary_path,
                    output_path=output_path,
                    long_window_threshold=long_window_threshold,
                    force=force,
                )
            )
        )

    @app.command("convert-operator-evidence-summary")
    def convert_operator_evidence_summary_cmd(
        sweep_summary_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        model_family: str = typer.Option(...),
        source_state_hint: str = typer.Option(...),
        long_window_threshold: int = typer.Option(31),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                convert_operator_evidence_summary_command(
                    sweep_summary_path=sweep_summary_path,
                    output_path=output_path,
                    model_family=model_family,
                    source_state_hint=source_state_hint,
                    long_window_threshold=long_window_threshold,
                    force=force,
                )
            )
        )

    @app.command("convert-official-comparison-evidence")
    def convert_official_comparison_evidence_cmd(
        comparison_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        long_window_threshold: int = typer.Option(31),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                convert_official_comparison_evidence_command(
                    comparison_report_path=comparison_report_path,
                    output_path=output_path,
                    long_window_threshold=long_window_threshold,
                    force=force,
                )
            )
        )

    @app.command("plan-next-experiment")
    def plan_next_experiment_cmd(
        score_path: str = typer.Option(...),
        error_diagnosis_path: str = typer.Option(...),
        physics_check_path: str = typer.Option(...),
        hypothesis_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_next_experiment(
                    score_path=score_path,
                    error_diagnosis_path=error_diagnosis_path,
                    physics_check_path=physics_check_path,
                    hypothesis_path=hypothesis_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("plan-next-experiment-from-fno-summary")
    def plan_next_experiment_from_fno_summary_cmd(
        summary_path: str = typer.Option(...),
        hypothesis_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_next_experiment_from_fno_summary_command(
                    summary_path=summary_path,
                    hypothesis_path=hypothesis_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("generate-experiment-config")
    def generate_experiment_config_cmd(
        plan_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                generate_experiment_config(
                    plan_path=plan_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("audit-experiment-config")
    def audit_experiment_config_cmd(
        config_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        plan_path: str | None = typer.Option(None),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                audit_experiment_config(
                    config_path=config_path,
                    output_path=output_path,
                    plan_path=plan_path,
                )
            )
        )
