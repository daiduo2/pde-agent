
from pathlib import Path
from typing import Any

from pde_agent.core.interactive_cockpit import (
    build_agent_dialogue_request,
    build_cockpit_snapshot,
    build_fno_training_loop_snapshot,
    dispatch_agent_dialogue_request,
    handle_chat_input,
)


def build_cockpit_snapshot_command(
    *,
    runs_root: str,
    active_run_dir: str | None,
    limit: int,
) -> dict[str, Any]:
    return build_cockpit_snapshot(
        runs_root=Path(runs_root),
        active_run_dir=Path(active_run_dir) if active_run_dir is not None else None,
        limit=limit,
    )


def build_agent_dialogue_request_command(
    *,
    user_message: str,
    runs_root: str,
    active_run_dir: str,
    output_dir: str,
    force: bool,
) -> dict[str, Any]:
    return build_agent_dialogue_request(
        user_message=user_message,
        runs_root=Path(runs_root),
        active_run_dir=Path(active_run_dir),
        output_dir=Path(output_dir),
        force=force,
    )


def run_interactive_cockpit_command(
    *,
    runs_root: str,
    active_run_dir: str | None,
    dialogue_output_dir: str,
) -> None:
    from pde_agent.core.interactive_cockpit_app import PDEAgentCockpitApp

    app = PDEAgentCockpitApp(
        runs_root=Path(runs_root),
        active_run_dir=Path(active_run_dir) if active_run_dir is not None else None,
        dialogue_output_dir=Path(dialogue_output_dir),
    )
    app.run()


def dispatch_agent_dialogue_request_command(
    *,
    request_path: str,
    output_path: str,
    workspace_dir: str,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    return dispatch_agent_dialogue_request(
        request_path=Path(request_path),
        output_path=Path(output_path),
        workspace_dir=Path(workspace_dir),
        dry_run=dry_run,
        force=force,
    )


def cockpit_chat_command(
    *,
    text: str,
    runs_root: str,
    active_run_dir: str | None,
    dialogue_output_dir: str,
    workspace_dir: str,
    force: bool,
) -> dict[str, Any]:
    return handle_chat_input(
        text=text,
        runs_root=Path(runs_root),
        active_run_dir=Path(active_run_dir) if active_run_dir is not None else None,
        dialogue_output_dir=Path(dialogue_output_dir),
        workspace_dir=Path(workspace_dir),
        force=force,
    )


def build_fno_training_loop_snapshot_command(
    *,
    workspace_dir: str,
) -> dict[str, Any]:
    return build_fno_training_loop_snapshot(
        workspace_dir=Path(workspace_dir),
    )
