from __future__ import annotations

from pathlib import Path

from pde_agent.core.agent_runtime import CLAUDE_RUNTIME_STATE_HINT
from pde_agent.core.agent_runtime import CLAUDE_FNO_TRAINING_SMOKE_TASK_STATE_HINT
from pde_agent.core.agent_runtime import CLAUDE_PROMOTION_SMOKE_TASK_STATE_HINT
from pde_agent.core.agent_runtime import CONTRACT_STATE_HINT
from pde_agent.core.agent_runtime import build_agent_runtime_contract
from pde_agent.core.agent_runtime import build_claude_fno_training_smoke_task
from pde_agent.core.agent_runtime import build_claude_promotion_smoke_task
from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime
from pde_agent.core.json_output import success_payload


def build_agent_runtime_contract_command(
    *,
    output_path: str,
    runtime_name: str,
    workspace_dir: str,
    tool_catalog_path: str,
    required_commands: list[str],
    force: bool,
) -> dict[str, object]:
    contract = build_agent_runtime_contract(
        runtime_name=runtime_name,
        output_path=Path(output_path),
        workspace_dir=Path(workspace_dir),
        tool_catalog_path=Path(tool_catalog_path),
        required_commands=required_commands,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="build-agent-runtime-contract",
        state_hint=CONTRACT_STATE_HINT,
        artifacts={
            "runtime_contract_path": output_path,
            "tool_catalog_path": tool_catalog_path,
        },
        metrics={
            "required_command_count": len(contract["required_commands"]),
            "allowed_tool_count": len(contract["tool_policy"]["allowed_tools"]),
            "allowed_bash_prefix_count": len(
                contract["bash_policy"]["allowed_prefixes"]
            ),
        },
        next_hint="run-claude-agent-sdk-runtime",
    )


def run_claude_agent_sdk_runtime_command(
    *,
    prompt: str | None,
    prompt_file: str | None,
    output_path: str,
    workspace_dir: str,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    allowed_bash_prefixes: list[str],
    dry_run: bool,
    force: bool,
) -> dict[str, object]:
    if prompt_file is not None:
        prompt_text = Path(prompt_file).read_text(encoding="utf-8")
    elif prompt is not None:
        prompt_text = prompt
    else:
        raise ValueError("either prompt or prompt_file is required")
    session = run_claude_agent_sdk_runtime(
        prompt=prompt_text,
        output_path=Path(output_path),
        workspace_dir=Path(workspace_dir),
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        allowed_bash_prefixes=allowed_bash_prefixes,
        dry_run=dry_run,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="run-claude-agent-sdk-runtime",
        state_hint=CLAUDE_RUNTIME_STATE_HINT,
        artifacts=session["artifacts"],
        metrics={
            "sdk_message_count": session["sdk_message_count"],
            "dry_run": session["status"] == "dry_run",
            "allowed_tool_count": len(session["tool_policy"]["allowed_tools"]),
        },
        next_hint="audit-agent-runtime-availability",
    )


def build_claude_fno_training_smoke_task_command(
    *,
    output_dir: str,
    workspace_dir: str,
    runtime_contract_path: str,
    tool_catalog_path: str,
    source_path: str,
    previous_evidence_paths: list[str],
    force: bool,
) -> dict[str, object]:
    task = build_claude_fno_training_smoke_task(
        output_dir=Path(output_dir),
        workspace_dir=Path(workspace_dir),
        runtime_contract_path=Path(runtime_contract_path),
        tool_catalog_path=Path(tool_catalog_path),
        source_path=Path(source_path),
        previous_evidence_paths=[Path(path) for path in previous_evidence_paths],
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="build-claude-fno-training-smoke-task",
        state_hint=CLAUDE_FNO_TRAINING_SMOKE_TASK_STATE_HINT,
        artifacts=task["artifacts"],
        metrics={
            "required_command_count": len(task["required_commands"]),
            "expected_artifact_count": len(task["expected_artifacts"]),
        },
        next_hint="run-claude-agent-sdk-runtime",
    )


def build_claude_promotion_smoke_task_command(
    *,
    output_dir: str,
    workspace_dir: str,
    runtime_contract_path: str,
    tool_catalog_path: str,
    force: bool,
) -> dict[str, object]:
    task = build_claude_promotion_smoke_task(
        output_dir=Path(output_dir),
        workspace_dir=Path(workspace_dir),
        runtime_contract_path=Path(runtime_contract_path),
        tool_catalog_path=Path(tool_catalog_path),
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="build-claude-promotion-smoke-task",
        state_hint=CLAUDE_PROMOTION_SMOKE_TASK_STATE_HINT,
        artifacts=task["artifacts"],
        metrics={
            "required_command_count": len(task["required_commands"]),
            "expected_artifact_count": len(task["expected_artifacts"]),
        },
        next_hint="run-claude-agent-sdk-runtime",
    )
