"""CLI commands for spectral domain."""

import json
import time

import typer

from pde_agent.commands.plan_next_spectral_residual_experiment import (
    plan_next_spectral_residual_experiment_command,
)
from pde_agent.commands.plan_spectral_residual_variance_intervention import (
    plan_spectral_residual_variance_intervention_command,
)
from pde_agent.commands.run_spectral_residual_evidence_sweep import (
    run_spectral_residual_evidence_sweep_command,
)
from pde_agent.commands.run_spectral_residual_convergence_grid import (
    run_spectral_residual_convergence_grid_command,
)
from pde_agent.commands.run_spectral_residual_window_stratification import (
    run_spectral_residual_window_stratification_command,
)
from pde_agent.commands.train_spectral_residual_baseline import (
    train_spectral_residual_baseline,
)

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("train-spectral-residual-baseline")
    def train_spectral_residual_baseline_cmd(
        windows_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        modes: int = typer.Option(16),
        ridge: float = typer.Option(1e-3),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                train_spectral_residual_baseline(
                    windows_path=windows_path,
                    checkpoint_path=checkpoint_path,
                    modes=modes,
                    ridge=ridge,
                    force=force,
                )
            )
        )

    @app.command("run-spectral-residual-evidence-sweep")
    def run_spectral_residual_evidence_sweep_cmd(
        source_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        start_indices: list[int] = typer.Option(..., "--start-index"),
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
                run_spectral_residual_evidence_sweep_command(
                    source_path=source_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    start_indices=start_indices,
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

    @app.command("run-spectral-residual-convergence-grid")
    def run_spectral_residual_convergence_grid_cmd(
        audit_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        source_path: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        sample_limit: int = typer.Option(8),
        context_steps: int = typer.Option(10),
        forecast_steps: int = typer.Option(31),
        time_reduction: int = typer.Option(5),
        space_reduction: int = typer.Option(4),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_spectral_residual_convergence_grid_command(
                    audit_path=audit_path,
                    output_dir=output_dir,
                    source_path=source_path,
                    dataset_key=dataset_key,
                    sample_limit=sample_limit,
                    context_steps=context_steps,
                    forecast_steps=forecast_steps,
                    time_reduction=time_reduction,
                    space_reduction=space_reduction,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
            )
        )

    @app.command("run-spectral-residual-window-stratification")
    def run_spectral_residual_window_stratification_cmd(
        audit_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_spectral_residual_window_stratification_command(
                    audit_path=audit_path,
                    output_dir=output_dir,
                    force=force,
                )
            )
        )

    @app.command("plan-next-spectral-residual-experiment")
    def plan_next_spectral_residual_experiment_cmd(
        evidence_summary_path: str = typer.Option(...),
        promotion_report_path: str = typer.Option(...),
        ledger_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_next_spectral_residual_experiment_command(
                    evidence_summary_path=evidence_summary_path,
                    promotion_report_path=promotion_report_path,
                    ledger_path=ledger_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("plan-spectral-residual-variance-intervention")
    def plan_spectral_residual_variance_intervention_cmd(
        grid_summary_path: str = typer.Option(...),
        promotion_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_spectral_residual_variance_intervention_command(
                    grid_summary_path=grid_summary_path,
                    promotion_report_path=promotion_report_path,
                    output_path=output_path,
                )
            )
        )
