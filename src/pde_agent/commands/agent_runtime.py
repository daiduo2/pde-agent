
from pathlib import Path

from pde_agent.core.agent_runtime import build_agent_runtime_contract
from pde_agent.core.agent_runtime import export_competition_llm_log
from pde_agent.core.agent_runtime import record_agent_runtime_issue
from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime
from pde_agent.core.agent_runtime import run_claude_cli_runtime
from pde_agent.core.agent_runtime_fno_task import build_claude_fno_training_smoke_task
from pde_agent.core.agent_runtime_hypothesis_task import (
    build_claude_hypothesis_lab_closure_task,
)
from pde_agent.core.agent_runtime_literature_task import (
    build_claude_literature_candidate_closure_task,
)
from pde_agent.core.agent_runtime_promotion_task import build_claude_promotion_smoke_task
from pde_agent.core.agent_runtime_shared import (
    AGENT_RUNTIME_ISSUE_RECORDED_STATE_HINT,
    CLAUDE_CLI_RUNTIME_STATE_HINT,
    CLAUDE_FNO_TRAINING_SMOKE_TASK_STATE_HINT,
    CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_STATE_HINT,
    CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_STATE_HINT,
    CLAUDE_PROMOTION_SMOKE_TASK_STATE_HINT,
    CLAUDE_RUNTIME_STATE_HINT,
    COMPETITION_LLM_LOG_STATE_HINT,
    CONTRACT_STATE_HINT,
)
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
    event_log_path: str | None,
    workspace_dir: str,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    allowed_bash_prefixes: list[str],
    allowed_pde_agent_commands: list[str],
    dry_run: bool,
    force: bool,
    timeout_seconds: int = 3600,
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
        event_log_path=Path(event_log_path) if event_log_path is not None else None,
        workspace_dir=Path(workspace_dir),
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        allowed_bash_prefixes=allowed_bash_prefixes,
        allowed_pde_agent_commands=allowed_pde_agent_commands,
        dry_run=dry_run,
        force=force,
        timeout_seconds=timeout_seconds,
    )
    payload = success_payload(
        run_id=Path(output_path).stem or "local",
        command="run-claude-agent-sdk-runtime",
        state_hint=CLAUDE_RUNTIME_STATE_HINT,
        artifacts=session["artifacts"],
        metrics={
            "sdk_message_count": session["sdk_message_count"],
            "dry_run": session["status"] == "dry_run",
            "allowed_tool_count": len(session["tool_policy"]["allowed_tools"]),
            "allowed_pde_agent_command_count": len(
                session["bash_policy"]["allowed_pde_agent_commands"]
            ),
        },
        next_hint="audit-agent-runtime-availability",
    )
    payload["runtime_status"] = session["status"]
    payload["errors"] = session["errors"]
    if session["status"] in {"failed", "timeout"}:
        payload["status"] = session["status"]
    return payload


def run_claude_cli_runtime_command(
    *,
    prompt: str | None,
    prompt_file: str | None,
    output_path: str,
    workspace_dir: str,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    permission_mode: str,
    output_format: str,
    dry_run: bool,
    force: bool,
    timeout_seconds: int,
) -> dict[str, object]:
    if prompt_file is not None:
        prompt_text = Path(prompt_file).read_text(encoding="utf-8")
    elif prompt is not None:
        prompt_text = prompt
    else:
        raise ValueError("either prompt or prompt_file is required")
    session = run_claude_cli_runtime(
        prompt=prompt_text,
        output_path=Path(output_path),
        workspace_dir=Path(workspace_dir),
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        permission_mode=permission_mode,
        output_format=output_format,
        dry_run=dry_run,
        force=force,
        timeout_seconds=timeout_seconds,
    )
    payload = success_payload(
        run_id=Path(output_path).stem or "local",
        command="run-claude-cli-runtime",
        state_hint=CLAUDE_CLI_RUNTIME_STATE_HINT,
        artifacts=session["artifacts"],
        metrics={
            "sdk_message_count": session["sdk_message_count"],
            "dry_run": session["status"] == "dry_run",
            "exit_code": session["process"]["exit_code"],
            "allowed_tool_count": len(session["tool_policy"]["allowed_tools"]),
        },
        next_hint="export-competition-llm-log",
    )
    payload["errors"] = session["errors"]
    return payload


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


def build_claude_hypothesis_lab_closure_task_command(
    *,
    output_dir: str,
    workspace_dir: str,
    runtime_contract_path: str,
    tool_catalog_path: str,
    candidate_family_sandbox_path: str,
    context_path: str,
    target_path: str,
    reference_path: str | None,
    force: bool,
) -> dict[str, object]:
    task = build_claude_hypothesis_lab_closure_task(
        output_dir=Path(output_dir),
        workspace_dir=Path(workspace_dir),
        runtime_contract_path=Path(runtime_contract_path),
        tool_catalog_path=Path(tool_catalog_path),
        candidate_family_sandbox_path=Path(candidate_family_sandbox_path),
        context_path=Path(context_path),
        target_path=Path(target_path),
        reference_path=Path(reference_path) if reference_path is not None else None,
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="build-claude-hypothesis-lab-closure-task",
        state_hint=CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_STATE_HINT,
        artifacts=task["artifacts"],
        metrics={
            "required_command_count": len(task["required_commands"]),
            "expected_artifact_count": len(task["expected_artifacts"]),
        },
        next_hint="run-claude-agent-sdk-runtime",
    )


def build_claude_literature_candidate_closure_task_command(
    *,
    output_dir: str,
    workspace_dir: str,
    runtime_contract_path: str,
    tool_catalog_path: str,
    paper_parse_path: str,
    operator_family: str,
    target_problem: str,
    registry_path: str,
    current_best_score: float,
    force: bool,
) -> dict[str, object]:
    task = build_claude_literature_candidate_closure_task(
        output_dir=Path(output_dir),
        workspace_dir=Path(workspace_dir),
        runtime_contract_path=Path(runtime_contract_path),
        tool_catalog_path=Path(tool_catalog_path),
        paper_parse_path=Path(paper_parse_path),
        operator_family=operator_family,
        target_problem=target_problem,
        registry_path=Path(registry_path),
        current_best_score=current_best_score,
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="build-claude-literature-candidate-closure-task",
        state_hint=CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_STATE_HINT,
        artifacts=task["artifacts"],
        metrics={
            "required_command_count": len(task["required_commands"]),
            "expected_artifact_count": len(task["expected_artifacts"]),
        },
        next_hint="run-claude-agent-sdk-runtime",
    )


def record_agent_runtime_issue_command(
    *,
    output_path: str,
    failed_command: str,
    artifact_path: str | None,
    stderr_summary: str,
    likely_schema_gap: bool,
    next_action: str,
    force: bool,
) -> dict[str, object]:
    issue = record_agent_runtime_issue(
        output_path=Path(output_path),
        failed_command=failed_command,
        artifact_path=Path(artifact_path) if artifact_path else None,
        stderr_summary=stderr_summary,
        likely_schema_gap=likely_schema_gap,
        next_action=next_action,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="record-agent-runtime-issue",
        state_hint=AGENT_RUNTIME_ISSUE_RECORDED_STATE_HINT,
        artifacts={"runtime_issue_path": output_path},
        metrics={"likely_schema_gap": issue["likely_schema_gap"]},
        next_hint=issue["next_action"],
    )


def export_competition_llm_log_command(
    *,
    session_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = export_competition_llm_log(
        session_path=Path(session_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="export-competition-llm-log",
        state_hint=COMPETITION_LLM_LOG_STATE_HINT,
        artifacts={
            "source_session_path": report["source_session_path"],
            "llm_log_path": report["output_path"],
        },
        metrics={
            "line_count": report["line_count"],
            "max_window_hours": report["max_window_hours"],
        },
        next_hint="copy-to-submission-task-log",
    )
