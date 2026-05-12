"""CLI wrappers for FNO training agent loop commands."""

from pathlib import Path
from typing import Any

from pde_agent.core.fno_training_agent_interface import (
    build_claude_fno_training_closed_loop_task,
)
from pde_agent.core.fno_training_agent_loop import (
    run_claude_fno_training_closed_loop,
)
from pde_agent.core.fno_training_tool_catalog import (
    build_fno_training_tool_catalog,
    write_tool_catalog,
)
from pde_agent.core.json_output import success_payload


def run_claude_fno_training_closed_loop_command(
    *,
    workspace_dir: str,
    source_path: str,
    dataset_key: str,
    max_iterations: int,
    target_score: float | None,
    timeout_seconds: int,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    result = run_claude_fno_training_closed_loop(
        workspace_dir=workspace_dir,
        source_path=source_path,
        dataset_key=dataset_key,
        max_iterations=max_iterations,
        target_score=target_score,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
        force=force,
    )
    return success_payload(
        run_id=Path(workspace_dir).name or "fno-training-loop",
        command="run-claude-fno-training-closed-loop",
        state_hint=result["state_hint"],
        artifacts=result["artifacts"],
        metrics={
            "max_iterations": max_iterations,
            "dry_run": dry_run,
            "target_score": target_score,
        },
        next_hint="monitor agent session for convergence",
    )


def build_claude_fno_training_closed_loop_task_command(
    *,
    workspace_dir: str,
    source_path: str,
    dataset_key: str,
    max_iterations: int,
    target_score: float | None,
    force: bool,
) -> dict[str, Any]:
    task = build_claude_fno_training_closed_loop_task(
        workspace_dir=workspace_dir,
        source_path=source_path,
        output_dir=Path(workspace_dir) / "agent_task",
        dataset_key=dataset_key,
        max_iterations=max_iterations,
        target_score=target_score,
        force=force,
    )
    return success_payload(
        run_id=Path(workspace_dir).name or "local",
        command="build-claude-fno-training-closed-loop-task",
        state_hint=task["state_hint"],
        artifacts=task["artifacts"],
        metrics={
            "max_iterations": max_iterations,
            "prompt_length": len(task["prompt"]),
        },
        next_hint="run-claude-fno-training-closed-loop",
    )


def build_fno_training_tool_catalog_command(
    *,
    output_path: str | None,
    force: bool,
) -> dict[str, Any]:
    catalog = build_fno_training_tool_catalog()
    if output_path is not None:
        write_tool_catalog(catalog, Path(output_path))
    return success_payload(
        run_id="tool-catalog",
        command="build-fno-training-tool-catalog",
        state_hint=catalog["state_hint"],
        artifacts={
            "tool_catalog_path": output_path,
        },
        metrics={
            "command_count": len(catalog["commands"]),
        },
        next_hint="build-claude-fno-training-closed-loop-task",
    )
