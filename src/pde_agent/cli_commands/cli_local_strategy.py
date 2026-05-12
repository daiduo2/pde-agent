"""CLI commands for local_strategy domain."""

import json
import time

import typer

from pde_agent.commands.lookup_strategy_skill import lookup_strategy_skill_command
from pde_agent.commands.audit_local_strategy_loop import (
    audit_local_strategy_loop_command,
)
from pde_agent.commands.build_local_strategy_library import (
    build_local_strategy_library_command,
    lookup_local_strategy_command,
)
from pde_agent.commands.ingest_local_strategy_result import (
    ingest_local_strategy_result_command,
)
from pde_agent.commands.local_strategy_scheduler import (
    claim_next_scheduled_experiment_command,
    schedule_local_strategy_evaluation_command,
)
from pde_agent.commands.local_strategy_leaderboard import (
    record_local_strategy_result_command,
)
from pde_agent.commands.route_local_strategy_next_action import (
    route_local_strategy_next_action_command,
)

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("lookup-strategy-skill")
    def lookup_strategy_skill_cmd(
        registry_path: str = typer.Option(...),
        applicable_pde: str = typer.Option(...),
        model_family: str = typer.Option(...),
        output_path: str = typer.Option(...),
        max_results: int = typer.Option(3),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                lookup_strategy_skill_command(
                    registry_path=registry_path,
                    applicable_pde=applicable_pde,
                    model_family=model_family,
                    max_results=max_results,
                    output_path=output_path,
                )
            )
        )

    @app.command("build-local-strategy-library")
    def build_local_strategy_library_cmd(
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_local_strategy_library_command(
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("lookup-local-strategy")
    def lookup_local_strategy_cmd(
        library_path: str = typer.Option(...),
        applicable_pde: str = typer.Option(...),
        model_family: str = typer.Option(...),
        failure_modes: list[str] = typer.Option(..., "--failure-mode"),
        output_path: str = typer.Option(...),
        max_results: int = typer.Option(3),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                lookup_local_strategy_command(
                    library_path=library_path,
                    applicable_pde=applicable_pde,
                    model_family=model_family,
                    failure_modes=failure_modes,
                    max_results=max_results,
                    output_path=output_path,
                )
            )
        )

    @app.command("schedule-local-strategy-evaluation")
    def schedule_local_strategy_evaluation_cmd(
        lookup_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        workspace_root: str = typer.Option(...),
        max_experiments: int = typer.Option(3),
        max_train_seconds: float = typer.Option(3600.0),
        max_inference_seconds: float = typer.Option(120.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                schedule_local_strategy_evaluation_command(
                    lookup_report_path=lookup_report_path,
                    output_path=output_path,
                    workspace_root=workspace_root,
                    max_experiments=max_experiments,
                    max_train_seconds=max_train_seconds,
                    max_inference_seconds=max_inference_seconds,
                    force=force,
                )
            )
        )

    @app.command("claim-next-scheduled-experiment")
    def claim_next_scheduled_experiment_cmd(
        schedule_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        completed_experiment_ids: list[str] | None = typer.Option(
            None,
            "--completed-experiment-id",
        ),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                claim_next_scheduled_experiment_command(
                    schedule_path=schedule_path,
                    completed_experiment_ids=completed_experiment_ids or [],
                    output_path=output_path,
                )
            )
        )

    @app.command("record-local-strategy-result")
    def record_local_strategy_result_cmd(
        schedule_path: str = typer.Option(...),
        experiment_id: str = typer.Option(...),
        result_summary_path: str = typer.Option(...),
        leaderboard_path: str = typer.Option(...),
        result_status: str = typer.Option("completed"),
        min_score_delta: float = typer.Option(1.0),
        max_inference_seconds: float = typer.Option(120.0),
        failure_reason: str | None = typer.Option(None),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_local_strategy_result_command(
                    schedule_path=schedule_path,
                    experiment_id=experiment_id,
                    result_summary_path=result_summary_path,
                    leaderboard_path=leaderboard_path,
                    result_status=result_status,
                    min_score_delta=min_score_delta,
                    max_inference_seconds=max_inference_seconds,
                    failure_reason=failure_reason,
                )
            )
        )

    @app.command("audit-local-strategy-loop")
    def audit_local_strategy_loop_cmd(
        output_dir: str = typer.Option(...),
        report_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                audit_local_strategy_loop_command(
                    output_dir=output_dir,
                    report_path=report_path,
                    force=force,
                )
            )
        )

    @app.command("route-local-strategy-next-action")
    def route_local_strategy_next_action_cmd(
        artifact_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                route_local_strategy_next_action_command(
                    artifact_path=artifact_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("ingest-local-strategy-result")
    def ingest_local_strategy_result_cmd(
        source_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        baseline_score: float | None = typer.Option(None),
        result_status: str = typer.Option("completed"),
        failure_reason: str | None = typer.Option(None),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                ingest_local_strategy_result_command(
                    source_path=source_path,
                    output_path=output_path,
                    baseline_score=baseline_score,
                    result_status=result_status,
                    failure_reason=failure_reason,
                )
            )
        )
