"""CLI commands for burgers domain."""

import json
import time

import typer

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.check_burgers_physics_local import check_burgers_physics_local
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.diagnose_fno_rollout import diagnose_fno_rollout
from pde_agent.commands.diagnose_unet_rollout import diagnose_unet_rollout
from pde_agent.commands.diagnose_rollout_error import diagnose_rollout_error
from pde_agent.commands.diagnose_task2_nu_shift import (
    diagnose_task2_nu_shift_command,
)
from pde_agent.commands.eval_rollout import eval_rollout

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("diagnose-task2-nu-shift")
    def diagnose_task2_nu_shift_cmd(
        target_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        reference_path: str | None = typer.Option(None),
        dataset_key: str = typer.Option("tensor"),
        nu_key: str = typer.Option("nu"),
        context_steps: int = typer.Option(10),
        sample_limit: int | None = typer.Option(None),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                diagnose_task2_nu_shift_command(
                    target_path=target_path,
                    output_path=output_path,
                    reference_path=reference_path,
                    dataset_key=dataset_key,
                    nu_key=nu_key,
                    context_steps=context_steps,
                    sample_limit=sample_limit,
                )
            )
        )

    @app.command("diagnose-fno-rollout")
    def diagnose_fno_rollout_cmd(
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
                diagnose_fno_rollout(
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

    @app.command("diagnose-unet-rollout")
    def diagnose_unet_rollout_cmd(
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
                diagnose_unet_rollout(
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

    @app.command("diagnose-rollout-error")
    def diagnose_rollout_error_cmd(
        prediction_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        score_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                diagnose_rollout_error(
                    prediction_path=prediction_path,
                    target_path=target_path,
                    score_path=score_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("check-burgers-physics-local")
    def check_burgers_physics_local_cmd(
        prediction_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        nu: float = typer.Option(0.001),
        dt: float = typer.Option(1.0),
        dx: float = typer.Option(1.0),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                check_burgers_physics_local(
                    prediction_path=prediction_path,
                    target_path=target_path,
                    output_path=output_path,
                    nu=nu,
                    dt=dt,
                    dx=dx,
                )
            )
        )

    @app.command("eval-rollout")
    def eval_rollout_cmd(
        run_id: str = typer.Option(...),
        run_root: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(json.dumps(eval_rollout(run_id=run_id, run_root=run_root)))

    @app.command("analyze-spectrum")
    def analyze_spectrum_cmd(
        run_id: str = typer.Option(...),
        run_root: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(json.dumps(analyze_spectrum(run_id=run_id, run_root=run_root)))

    @app.command("check-physics")
    def check_physics_cmd(
        run_id: str = typer.Option(...),
        run_root: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(json.dumps(check_physics(run_id=run_id, run_root=run_root)))
