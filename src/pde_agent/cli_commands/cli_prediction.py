"""CLI commands for prediction domain."""

import json
import time

import typer

from pde_agent.commands.generate_baseline_prediction import generate_baseline_prediction
from pde_agent.commands.inspect_data import inspect_data
from pde_agent.commands.inspect_fno_checkpoint import inspect_fno_checkpoint
from pde_agent.commands.inspect_unet_checkpoint import inspect_unet_checkpoint
from pde_agent.commands.predict_fno_local import predict_fno_local
from pde_agent.commands.predict_fno_competition import (
    predict_fno_competition_command,
)
from pde_agent.commands.predict_task1_retrieval_prior import (
    predict_task1_retrieval_prior_command,
)
from pde_agent.commands.predict_unet_local import predict_unet_local
from pde_agent.commands.predict_unet_competition import (
    predict_unet_competition_command,
)
from pde_agent.commands.predict_local_model import predict_local_model
from pde_agent.commands.score_local_rollout import score_local_rollout

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("score-local-rollout")
    def score_local_rollout_cmd(
        prediction_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                score_local_rollout(
                    prediction_path=prediction_path,
                    target_path=target_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("generate-baseline-prediction")
    def generate_baseline_prediction_cmd(
        windows_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        method: str = typer.Option("persistence"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                generate_baseline_prediction(
                    windows_path=windows_path,
                    output_path=output_path,
                    method=method,
                    force=force,
                )
            )
        )

    @app.command("predict-local-model")
    def predict_local_model_cmd(
        windows_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                predict_local_model(
                    windows_path=windows_path,
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("inspect-fno-checkpoint")
    def inspect_fno_checkpoint_cmd(
        checkpoint_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                inspect_fno_checkpoint(
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("predict-fno-local")
    def predict_fno_local_cmd(
        windows_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        metadata_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                predict_fno_local(
                    windows_path=windows_path,
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    metadata_path=metadata_path,
                    force=force,
                )
            )
        )

    @app.command("inspect-unet-checkpoint")
    def inspect_unet_checkpoint_cmd(
        checkpoint_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                inspect_unet_checkpoint(
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("predict-unet-local")
    def predict_unet_local_cmd(
        windows_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        metadata_path: str | None = typer.Option(None),
        inference_batch_size: int | None = typer.Option(None),
        torch_num_threads: int | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                predict_unet_local(
                    windows_path=windows_path,
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    metadata_path=metadata_path,
                    inference_batch_size=inference_batch_size,
                    torch_num_threads=torch_num_threads,
                    force=force,
                )
            )
        )

    @app.command("inspect-data")
    def inspect_data_cmd(
        run_id: str = typer.Option(...),
        run_root: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(json.dumps(inspect_data(run_id=run_id, run_root=run_root)))

    @app.command("predict-unet-competition")
    def predict_unet_competition_cmd(
        task_id: int = typer.Option(...),
        test_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        inference_batch_size: int | None = typer.Option(None),
        torch_num_threads: int | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                predict_unet_competition_command(
                    task_id=task_id,
                    test_path=test_path,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    inference_batch_size=inference_batch_size,
                    torch_num_threads=torch_num_threads,
                    force=force,
                )
            )
        )

    @app.command("predict-fno-competition")
    def predict_fno_competition_cmd(
        task_id: int = typer.Option(...),
        test_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                predict_fno_competition_command(
                    task_id=task_id,
                    test_path=test_path,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    force=force,
                )
            )
        )

    @app.command("predict-task1-retrieval-prior")
    def predict_task1_retrieval_prior_cmd(
        reference_path: str = typer.Option(...),
        test_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        k: int = typer.Option(20),
        mode: str = typer.Option("affine"),
        leave_one_out: bool = typer.Option(False, "--leave-one-out"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                predict_task1_retrieval_prior_command(
                    reference_path=reference_path,
                    test_path=test_path,
                    output_dir=output_dir,
                    dataset_key=dataset_key,
                    k=k,
                    mode=mode,
                    leave_one_out=leave_one_out,
                    force=force,
                )
            )
        )
