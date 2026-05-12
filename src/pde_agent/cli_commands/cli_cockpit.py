"""CLI commands for cockpit domain."""

import json
import time

import typer

from pde_agent.commands.interactive_cockpit import (
    build_agent_dialogue_request_command,
    build_cockpit_snapshot_command,
    build_fno_training_loop_snapshot_command,
    cockpit_chat_command,
    dispatch_agent_dialogue_request_command,
    run_interactive_cockpit_command,
)
from pde_agent.commands.supervisor_tui import supervisor_tui_run_command
from pde_agent.commands.supervisor_tui import supervisor_tui_runs_command

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("supervisor-tui-runs")
    def supervisor_tui_runs_cmd(
        runs_root: str = typer.Option("runs"),
        limit: int = typer.Option(20),
        wide: bool = typer.Option(False, "--wide"),
        watch: float = typer.Option(0.0, "--watch"),
        json_flag: bool = typer.Option(False, "--json"),
    ) -> None:
        while True:
            view = supervisor_tui_runs_command(runs_root=runs_root, limit=limit, wide=wide)
            if json_flag:
                typer.echo(json.dumps(view))
            else:
                typer.echo(view["text"])
            if watch <= 0:
                break
            time.sleep(watch)

    @app.command("supervisor-tui-run")
    def supervisor_tui_run_cmd(
        run_dir: str = typer.Option(...),
        json_flag: bool = typer.Option(False, "--json"),
    ) -> None:
        view = supervisor_tui_run_command(run_dir=run_dir)
        if json_flag:
            typer.echo(json.dumps(view))
        else:
            typer.echo(view["text"])

    @app.command("cockpit-snapshot")
    def cockpit_snapshot_cmd(
        runs_root: str = typer.Option("runs"),
        active_run_dir: str | None = typer.Option(None),
        limit: int = typer.Option(20),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_cockpit_snapshot_command(
                    runs_root=runs_root,
                    active_run_dir=active_run_dir,
                    limit=limit,
                )
            )
        )

    @app.command("cockpit-dialogue-request")
    def cockpit_dialogue_request_cmd(
        user_message: str = typer.Option(...),
        runs_root: str = typer.Option("runs"),
        active_run_dir: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_agent_dialogue_request_command(
                    user_message=user_message,
                    runs_root=runs_root,
                    active_run_dir=active_run_dir,
                    output_dir=output_dir,
                    force=force,
                )
            )
        )

    @app.command("cockpit-dispatch-dialogue")
    def cockpit_dispatch_dialogue_cmd(
        request_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        workspace_dir: str = typer.Option("."),
        dry_run: bool = typer.Option(True, "--dry-run/--execute"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                dispatch_agent_dialogue_request_command(
                    request_path=request_path,
                    output_path=output_path,
                    workspace_dir=workspace_dir,
                    dry_run=dry_run,
                    force=force,
                )
            )
        )

    @app.command("cockpit-chat")
    def cockpit_chat_cmd(
        text: str = typer.Argument(...),
        runs_root: str = typer.Option("runs"),
        active_run_dir: str | None = typer.Option(None),
        dialogue_output_dir: str = typer.Option("runs/cockpit_dialogue"),
        workspace_dir: str = typer.Option("."),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        result = cockpit_chat_command(
            text=text,
            runs_root=runs_root,
            active_run_dir=active_run_dir,
            dialogue_output_dir=dialogue_output_dir,
            workspace_dir=workspace_dir,
            force=force,
        )
        if json_flag:
            typer.echo(json.dumps(result))
        else:
            typer.echo(result.get("text", ""))

    @app.command("monitor-fno-training-loop")
    def monitor_fno_training_loop_cmd(
        workspace_dir: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        result = build_fno_training_loop_snapshot_command(
            workspace_dir=workspace_dir,
        )
        if json_flag:
            typer.echo(json.dumps(result))
        else:
            typer.echo(result.get("panel_text", ""))
