"""Agent loop launcher for FNO training closed loops using Claude Code SDK."""

import os
from pathlib import Path
from typing import Any

from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime
from pde_agent.core.fno_training_agent_interface import (
    build_claude_fno_training_closed_loop_task,
)


STATE_HINT = "FNO_TRAINING_CLOSED_LOOP_LAUNCHED"


class FnoTrainingAgentLoopError(RuntimeError):
    pass


def _require_anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise FnoTrainingAgentLoopError(
            "ANTHROPIC_API_KEY is not set. "
            "The Claude Agent SDK requires this environment variable. "
            "It is automatically available when using Claude CLI (from zshrc)."
        )
    return key


def _build_sdk_env() -> dict[str, str]:
    env: dict[str, str] = {}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    return env


def run_claude_fno_training_closed_loop(
    *,
    workspace_dir: str | Path,
    source_path: str | Path,
    dataset_key: str = "tensor",
    max_iterations: int = 10,
    target_score: float | None = None,
    timeout_seconds: int = 3600,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    if max_iterations < 1:
        raise FnoTrainingAgentLoopError("max_iterations must be >= 1")
    if not dry_run:
        _require_anthropic_api_key()
    task = build_claude_fno_training_closed_loop_task(
        workspace_dir=workspace,
        source_path=source,
        output_dir=workspace / "agent_task",
        dataset_key=dataset_key,
        max_iterations=max_iterations,
        target_score=target_score,
        force=force,
    )
    output_path = workspace / "agent_task" / "sdk_session.json"
    session = run_claude_agent_sdk_runtime(
        prompt=task["prompt"],
        output_path=output_path,
        workspace_dir=workspace,
        allowed_tools=["Read", "Glob", "Grep", "Bash"],
        disallowed_tools=["Edit", "Write", "NotebookEdit", "WebFetch"],
        allowed_bash_prefixes=[
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"
        ],
        allowed_pde_agent_commands=task["required_commands"],
        dry_run=dry_run,
        force=force,
        timeout_seconds=timeout_seconds,
        env=_build_sdk_env(),
    )
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "status": session["status"],
        "max_iterations": max_iterations,
        "target_score": target_score,
        "dry_run": dry_run,
        "artifacts": {
            "contract_path": task["artifacts"]["contract_path"],
            "prompt_path": task["artifacts"]["prompt_path"],
            "tool_catalog_path": task["artifacts"]["tool_catalog_path"],
            "session_path": str(output_path),
        },
        "session": session,
    }
