"""CLI commands for submission domain."""

import json
import time

import typer

from pde_agent.commands.audit_competition_submission import (
    audit_competition_submission_command,
)
from pde_agent.commands.audit_task2_source_policy import (
    audit_task2_source_policy_command,
)
from pde_agent.commands.build_competition_submission import (
    build_competition_submission_command,
)
from pde_agent.commands.build_submission import build_submission
from pde_agent.commands.materialize_competition_candidate import (
    materialize_competition_candidate_command,
)
from pde_agent.commands.merge_competition_submissions import (
    merge_competition_submissions_command,
)
from pde_agent.commands.prepare_competition_windows import (
    prepare_competition_windows_command,
)
from pde_agent.commands.run_competition_task_pipeline import (
    run_competition_task_pipeline_command,
)
from pde_agent.commands.score_competition_submission import (
    score_competition_submission_command,
)

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("build-submission")
    def build_submission_cmd(
        run_id: str = typer.Option(...),
        run_root: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(json.dumps(build_submission(run_id=run_id, run_root=run_root)))

    @app.command("build-competition-submission")
    def build_competition_submission_cmd(
        task_id: int = typer.Option(...),
        test_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        prediction_dataset_key: str = typer.Option("tensor"),
        method: str = typer.Option("persistence"),
        source_prediction_path: str | None = typer.Option(None),
        source_prediction_key: str = typer.Option("prediction"),
        train_time: float = typer.Option(0.0),
        inference_time: float | None = typer.Option(None),
        timing_profile: str | None = typer.Option(None),
        timing_factor: float | None = typer.Option(None),
        local_loop_chronicle_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_competition_submission_command(
                    task_id=task_id,
                    test_path=test_path,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    dataset_key=dataset_key,
                    prediction_dataset_key=prediction_dataset_key,
                    method=method,
                    source_prediction_path=source_prediction_path,
                    source_prediction_key=source_prediction_key,
                    train_time=train_time,
                    inference_time=inference_time,
                    timing_profile=timing_profile,
                    timing_factor=timing_factor,
                    local_loop_chronicle_path=local_loop_chronicle_path,
                    force=force,
                )
            )
        )

    @app.command("score-competition-submission")
    def score_competition_submission_cmd(
        task_id: int = typer.Option(...),
        prediction_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        prediction_dataset_key: str = typer.Option("tensor"),
        target_dataset_key: str = typer.Option("tensor"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                score_competition_submission_command(
                    task_id=task_id,
                    prediction_path=prediction_path,
                    target_path=target_path,
                    output_path=output_path,
                    prediction_dataset_key=prediction_dataset_key,
                    target_dataset_key=target_dataset_key,
                )
            )
        )

    @app.command("materialize-competition-candidate")
    def materialize_competition_candidate_cmd(
        task_id: int = typer.Option(...),
        candidate_report_path: str = typer.Option(...),
        test_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option(...),
        target_path: str | None = typer.Option(None),
        dataset_key: str = typer.Option("tensor"),
        prediction_dataset_key: str = typer.Option("tensor"),
        source_prediction_key: str = typer.Option("prediction"),
        train_time: float | None = typer.Option(None),
        timing_profile: str | None = typer.Option(None),
        timing_factor: float | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                materialize_competition_candidate_command(
                    task_id=task_id,
                    candidate_report_path=candidate_report_path,
                    test_path=test_path,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    target_path=target_path,
                    dataset_key=dataset_key,
                    prediction_dataset_key=prediction_dataset_key,
                    source_prediction_key=source_prediction_key,
                    train_time=train_time,
                    timing_profile=timing_profile,
                    timing_factor=timing_factor,
                    force=force,
                )
            )
        )

    @app.command("merge-competition-submissions")
    def merge_competition_submissions_cmd(
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option(...),
        task_submission_dirs: list[str] = typer.Option(..., "--task-submission-dir"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                merge_competition_submissions_command(
                    output_dir=output_dir,
                    submission_id=submission_id,
                    task_submission_dirs=task_submission_dirs,
                    force=force,
                )
            )
        )

    @app.command("audit-competition-submission")
    def audit_competition_submission_cmd(
        task_id: int = typer.Option(...),
        submission_dir: str = typer.Option(...),
        test_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                audit_competition_submission_command(
                    task_id=task_id,
                    submission_dir=submission_dir,
                    test_path=test_path,
                    output_path=output_path,
                    dataset_key=dataset_key,
                )
            )
        )

    @app.command("audit-task2-source-policy")
    def audit_task2_source_policy_cmd(
        train_data_paths: list[str] = typer.Option([], "--train-data-path"),
        test_data_paths: list[str] = typer.Option([], "--test-data-path"),
        checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
        candidate_source_paths: list[str] = typer.Option([], "--candidate-source-path"),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                audit_task2_source_policy_command(
                    train_data_paths=train_data_paths,
                    test_data_paths=test_data_paths,
                    checkpoint_paths=checkpoint_paths,
                    candidate_source_paths=candidate_source_paths,
                    output_path=output_path,
                )
            )
        )

    @app.command("prepare-competition-windows")
    def prepare_competition_windows_cmd(
        task_id: int = typer.Option(...),
        test_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        dataset_key: str = typer.Option("tensor"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                prepare_competition_windows_command(
                    task_id=task_id,
                    test_path=test_path,
                    output_path=output_path,
                    dataset_key=dataset_key,
                    force=force,
                )
            )
        )

    @app.command("run-competition-task-pipeline")
    def run_competition_task_pipeline_cmd(
        task_id: int = typer.Option(...),
        model_family: str = typer.Option(...),
        test_path: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option(...),
        target_path: str | None = typer.Option(None),
        dataset_key: str = typer.Option("tensor"),
        timing_profile: str | None = typer.Option(None),
        timing_factor: float | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        local_strategy_schedule_path: str | None = typer.Option(None),
        local_strategy_experiment_id: str | None = typer.Option(None),
        local_strategy_leaderboard_path: str | None = typer.Option(None),
        local_strategy_baseline_score: float | None = typer.Option(None),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_competition_task_pipeline_command(
                    task_id=task_id,
                    model_family=model_family,
                    test_path=test_path,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    target_path=target_path,
                    dataset_key=dataset_key,
                    timing_profile=timing_profile,
                    timing_factor=timing_factor,
                    force=force,
                    local_strategy_schedule_path=local_strategy_schedule_path,
                    local_strategy_experiment_id=local_strategy_experiment_id,
                    local_strategy_leaderboard_path=local_strategy_leaderboard_path,
                    local_strategy_baseline_score=local_strategy_baseline_score,
                )
            )
        )
