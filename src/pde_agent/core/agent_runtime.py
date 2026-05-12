
import asyncio
import importlib
import json
import shlex
import subprocess
import shutil
from pathlib import Path
from typing import Any
from datetime import datetime
from datetime import timedelta
from datetime import timezone


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
    AgentRuntimeError,
    _write_json,
)
from pde_agent.core.agent_runtime_fno_task import (
    build_claude_fno_training_smoke_task,
)
from pde_agent.core.agent_runtime_hypothesis_task import (
    build_claude_hypothesis_lab_closure_task,
)
from pde_agent.core.agent_runtime_literature_task import (
    build_claude_literature_candidate_closure_task,
)
from pde_agent.core.agent_runtime_promotion_task import (
    build_claude_promotion_smoke_task,
)


def build_agent_runtime_contract(
    *,
    runtime_name: str,
    output_path: Path,
    workspace_dir: Path,
    tool_catalog_path: Path,
    required_commands: list[str],
    force: bool,
) -> dict[str, Any]:
    if runtime_name != "claude_agent_sdk":
        raise AgentRuntimeError(f"unsupported runtime: {runtime_name}")
    workspace_dir = workspace_dir.expanduser().resolve()
    tool_catalog_path = tool_catalog_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    contract: dict[str, Any] = {
        "state_hint": CONTRACT_STATE_HINT,
        "runtime": {
            "name": "claude_agent_sdk",
            "primary": True,
            "compatibility_backends": [],
            "legacy_backends": [],
            "legacy_backend_policy": "not exposed in active runtime contracts",
            "sdk_package": "claude-agent-sdk",
            "sdk_import": "claude_agent_sdk",
        },
        "workspace_dir": str(workspace_dir),
        "tool_catalog_path": str(tool_catalog_path),
        "required_commands": list(required_commands),
        "tool_policy": {
            "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
            "disallowed_tools": ["Edit", "Write", "NotebookEdit", "WebFetch"],
            "permission_mode": "dontAsk",
            "bash_guard_hook": "enabled",
        },
        "bash_policy": {
            "allowed_prefixes": [
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"
            ],
            "allowed_pde_agent_commands": list(required_commands),
            "forbidden_patterns": [
                "mkdir ",
                "rm ",
                "python -c",
                "python3 -c",
                "pip install",
                "uv add",
                ";",
                "&&",
                "||",
                "|",
                "$(",
                "`",
            ],
        },
        "agent_guidance": (
            "Use the scientific tool catalog and pde-agent CLI artifacts as the "
            "execution surface. Do not patch project files or invent shell "
            "commands outside the allowed pde-agent prefix."
        ),
    }
    _write_json(output_path, contract, force=force)
    return contract


def build_bash_prefix_guard_hook(
    allowed_bash_prefixes: list[str],
    allowed_pde_agent_commands: list[str] | None = None,
) -> Any:
    command_allowlist = set(allowed_pde_agent_commands or [])

    async def guard_hook(
        input_data: dict[str, Any],
        tool_use_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        _ = tool_use_id, context
        tool_input = input_data.get("tool_input", {})
        command = tool_input.get("command") if isinstance(tool_input, dict) else None
        if not isinstance(command, str):
            return {}
        decision = _validate_bash_command(
            command=command,
            allowed_bash_prefixes=allowed_bash_prefixes,
            allowed_pde_agent_commands=command_allowlist,
        )
        if decision is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": decision,
            }
        }

    return guard_hook


def _validate_bash_command(
    *,
    command: str,
    allowed_bash_prefixes: list[str],
    allowed_pde_agent_commands: set[str],
) -> str | None:
    if _contains_shell_control(command):
        return "Bash command contains shell control syntax outside the typed tool gateway."
    matching_prefix = next(
        (prefix for prefix in allowed_bash_prefixes if command.startswith(prefix)),
        None,
    )
    if matching_prefix is None:
        return "Bash command is outside allowed pde-agent prefixes."
    if not allowed_pde_agent_commands:
        return None
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Bash command could not be parsed safely: {exc}"
    if parts and parts[0].startswith("UV_CACHE_DIR="):
        parts = parts[1:]
    if len(parts) < 5 or parts[:4] != ["uv", "run", "--frozen", "pde-agent"]:
        return "Bash command does not match the controlled pde-agent invocation shape."
    command_name = parts[4]
    if command_name not in allowed_pde_agent_commands:
        return (
            f"pde-agent command `{command_name}` is not in the allowed command "
            "gateway for this task."
        )
    return None


def _contains_shell_control(command: str) -> bool:
    forbidden = [
        ";",
        "&&",
        "||",
        "|",
        "$(",
        "`",
        "\n",
        "\r",
        ">",
        "<",
        "2>",
        "1>",
        ">>",
        "<<",
        "tee ",
        "tail ",
        "head ",
        "grep ",
    ]
    return any(token in command for token in forbidden)


def _import_claude_agent_sdk() -> Any:
    try:
        return importlib.import_module("claude_agent_sdk")
    except ModuleNotFoundError as exc:
        raise AgentRuntimeError(
            "Claude Agent SDK is not installed. Install optional package "
            "`claude-agent-sdk` so import `claude_agent_sdk` is available."
        ) from exc


async def _collect_sdk_messages(
    *,
    sdk_module: Any,
    prompt: str,
    workspace_dir: Path,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    allowed_bash_prefixes: list[str],
    allowed_pde_agent_commands: list[str] | None = None,
    env: dict[str, str] | None = None,
    on_message: Any | None = None,
) -> list[str]:
    options_cls = getattr(sdk_module, "ClaudeAgentOptions")
    hook_matcher_cls = getattr(sdk_module, "HookMatcher", None)
    query = getattr(sdk_module, "query")
    hooks = {}
    if hook_matcher_cls is not None:
        hooks = {
            "PreToolUse": [
                hook_matcher_cls(
                    matcher="Bash",
                    hooks=[
                        build_bash_prefix_guard_hook(
                            allowed_bash_prefixes,
                            allowed_pde_agent_commands=allowed_pde_agent_commands,
                        )
                    ],
                )
            ]
        }
    options_kwargs: dict[str, Any] = dict(
        cwd=str(workspace_dir),
        cli_path=shutil.which("claude"),
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        permission_mode="dontAsk",
        hooks=hooks,
    )
    if env:
        options_kwargs["env"] = env
    options = options_cls(**options_kwargs)
    messages = []
    async for message in query(prompt=prompt, options=options):
        for event in _sdk_message_events(message):
            messages.append(event)
            if on_message is not None:
                on_message(event)
    return messages


def _sdk_message_events(message: Any) -> list[str]:
    if isinstance(message, str):
        return [message]
    class_name = type(message).__name__
    if class_name == "AssistantMessage":
        events = []
        for block in getattr(message, "content", []) or []:
            block_name = type(block).__name__
            if block_name == "TextBlock":
                events.append(
                    _json_event(
                        {
                            "type": "assistant",
                            "text": str(getattr(block, "text", "")),
                        }
                    )
                )
            elif block_name == "ToolUseBlock":
                events.append(
                    _json_event(
                        {
                            "type": "tool_use",
                            "name": getattr(block, "name", None),
                            "input": getattr(block, "input", None),
                        }
                    )
                )
            elif block_name == "ToolResultBlock":
                events.append(
                    _json_event(
                        {
                            "type": "tool_result",
                            "content": getattr(block, "content", None),
                            "is_error": getattr(block, "is_error", None),
                        }
                    )
                )
        return events or [_json_event({"type": "assistant", "text": str(message)})]
    if class_name == "ResultMessage":
        return [
            _json_event(
                {
                    "type": "result",
                    "subtype": getattr(message, "subtype", None),
                    "is_error": getattr(message, "is_error", None),
                    "num_turns": getattr(message, "num_turns", None),
                    "duration_ms": getattr(message, "duration_ms", None),
                    "result": getattr(message, "result", None),
                    "errors": getattr(message, "errors", None),
                }
            )
        ]
    return [_json_event({"type": class_name, "text": str(message)})]


def _json_event(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)


def run_claude_agent_sdk_runtime(
    *,
    prompt: str,
    output_path: Path,
    event_log_path: Path | None = None,
    workspace_dir: Path,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    allowed_bash_prefixes: list[str],
    allowed_pde_agent_commands: list[str] | None = None,
    dry_run: bool,
    force: bool,
    timeout_seconds: int = 3600,
    sdk_module: Any | None = "auto",
    on_message: Any | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_path = output_path.expanduser().resolve()
    event_log_file = event_log_path.expanduser().resolve() if event_log_path is not None else None
    workspace_dir = workspace_dir.expanduser().resolve()
    if event_log_file is not None:
        if event_log_file.exists() and not force:
            raise AgentRuntimeError(f"refusing to overwrite existing file: {event_log_file}")
        event_log_file.parent.mkdir(parents=True, exist_ok=True)
        event_log_file.write_text("", encoding="utf-8")

    def _append_event(event: str) -> None:
        if event_log_file is None:
            return
        with event_log_file.open("a", encoding="utf-8") as handle:
            handle.write(event)
            handle.write("\n")

    if dry_run:
        sdk_messages: list[str] = []
        status = "dry_run"
        error_message = None
    else:
        module = _import_claude_agent_sdk() if sdk_module == "auto" else sdk_module
        if module is None:
            raise AgentRuntimeError(
                "Claude Agent SDK is not installed. Install optional package "
                "`claude-agent-sdk` so import `claude_agent_sdk` is available."
            )
        try:
            sdk_messages = asyncio.run(
                asyncio.wait_for(
                    _collect_sdk_messages(
                        sdk_module=module,
                        prompt=prompt,
                        workspace_dir=workspace_dir,
                        allowed_tools=allowed_tools,
                        disallowed_tools=disallowed_tools,
                        allowed_bash_prefixes=allowed_bash_prefixes,
                        allowed_pde_agent_commands=allowed_pde_agent_commands,
                        env=env,
                        on_message=lambda event: (
                            _append_event(event),
                            on_message(event) if on_message is not None else None,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            )
        except TimeoutError:
            sdk_messages = []
            status = "timeout"
            error_message = f"Claude Agent SDK runtime timed out after {timeout_seconds} seconds"
        except Exception as exc:
            sdk_messages = []
            status = "failed"
            error_message = _format_sdk_exception(exc)
        else:
            status = "completed"
            error_message = None

    session: dict[str, Any] = {
        "state_hint": CLAUDE_RUNTIME_STATE_HINT,
        "status": status,
        "runtime": {
            "name": "claude_agent_sdk",
            "sdk_package": "claude-agent-sdk",
            "sdk_import": "claude_agent_sdk",
        },
        "workspace_dir": str(workspace_dir),
        "prompt_preview": prompt[:200],
        "tool_policy": {
            "allowed_tools": list(allowed_tools),
            "disallowed_tools": list(disallowed_tools),
            "permission_mode": "dontAsk",
            "bash_guard_hook": "enabled",
            "timeout_seconds": timeout_seconds,
        },
        "bash_policy": {
            "allowed_prefixes": list(allowed_bash_prefixes),
            "allowed_pde_agent_commands": list(allowed_pde_agent_commands or []),
        },
        "sdk_message_count": len(sdk_messages),
        "sdk_messages": sdk_messages,
        "errors": [error_message] if error_message else [],
        "artifacts": {
            "session_path": str(output_path),
            "event_log_path": str(event_log_file) if event_log_file is not None else None,
        },
    }
    _write_json(output_path, session, force=force)
    return session


def run_claude_cli_runtime(
    *,
    prompt: str,
    output_path: Path,
    workspace_dir: Path,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    permission_mode: str,
    output_format: str,
    dry_run: bool,
    force: bool,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    output_path = output_path.expanduser().resolve()
    workspace_dir = workspace_dir.expanduser().resolve()
    if permission_mode not in {"dontAsk", "default", "auto", "plan", "bypassPermissions"}:
        raise AgentRuntimeError(f"unsupported permission mode: {permission_mode}")
    if output_format not in {"text", "json", "stream-json"}:
        raise AgentRuntimeError(f"unsupported output format: {output_format}")
    command = [
        "claude",
        "-p",
        "--permission-mode",
        permission_mode,
        "--output-format",
        output_format,
    ]
    if allowed_tools:
        command.extend(["--allowedTools", ",".join(allowed_tools)])
    if disallowed_tools:
        command.extend(["--disallowedTools", ",".join(disallowed_tools)])
    command.extend(["--", prompt])
    if dry_run:
        completed = None
        stdout = ""
        stderr = ""
        exit_code = None
        status = "dry_run"
    else:
        try:
            completed = subprocess.run(
                command,
                cwd=workspace_dir,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            completed = None
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            exit_code = None
            status = "timeout"
        else:
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = int(completed.returncode)
            status = "completed" if completed.returncode == 0 else "failed"
    messages = _claude_cli_stdout_to_messages(stdout=stdout, output_format=output_format)
    errors = []
    if status in {"failed", "timeout"}:
        errors.append(_claude_cli_error_summary(status=status, stderr=stderr, stdout=stdout))
    session: dict[str, Any] = {
        "state_hint": CLAUDE_CLI_RUNTIME_STATE_HINT,
        "status": status,
        "runtime": {
            "name": "claude_cli",
            "command": "claude -p",
            "output_format": output_format,
        },
        "workspace_dir": str(workspace_dir),
        "prompt_preview": prompt[:200],
        "tool_policy": {
            "allowed_tools": list(allowed_tools),
            "disallowed_tools": list(disallowed_tools),
            "permission_mode": permission_mode,
        },
        "process": {
            "argv": [command[0], *command[1:-1], "<prompt>"],
            "shell_command_preview": " ".join(
                shlex.quote(part) for part in [command[0], *command[1:-1], "<prompt>"]
            ),
            "exit_code": exit_code,
            "stdout_preview": stdout[:4000],
            "stderr_preview": stderr[:4000],
        },
        "sdk_message_count": len(messages),
        "sdk_messages": messages,
        "errors": errors,
        "artifacts": {
            "session_path": str(output_path),
        },
    }
    _write_json(output_path, session, force=force)
    return session


def _claude_cli_stdout_to_messages(*, stdout: str, output_format: str) -> list[str]:
    if not stdout.strip():
        return []
    if output_format == "stream-json":
        messages = []
        for line in stdout.splitlines():
            if not line.strip():
                continue
            messages.append(_claude_cli_stream_line_to_event(line))
        return messages
    if output_format == "json":
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return [_json_event({"type": "assistant", "text": stdout.strip()})]
        text = payload.get("result") if isinstance(payload, dict) else None
        return [_json_event({"type": "result", "result": text or stdout.strip()})]
    return [_json_event({"type": "assistant", "text": stdout.strip()})]


def _claude_cli_stream_line_to_event(line: str) -> str:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return _json_event({"type": "raw", "text": line})
    if not isinstance(payload, dict):
        return _json_event({"type": "raw", "text": line})
    event_type = str(payload.get("type") or "")
    if event_type == "assistant":
        text_parts = []
        tool_events = []
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else payload.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block.get("type") == "tool_use":
                    tool_events.append(
                        _json_event(
                            {
                                "type": "tool_use",
                                "name": block.get("name"),
                                "input": block.get("input"),
                            }
                        )
                    )
        events = []
        if text_parts:
            events.append(_json_event({"type": "assistant", "text": "\n".join(text_parts)}))
        events.extend(tool_events)
        if events:
            return "\n".join(events)
    if event_type == "result":
        return _json_event(
            {
                "type": "result",
                "subtype": payload.get("subtype"),
                "is_error": payload.get("is_error"),
                "num_turns": payload.get("num_turns"),
                "duration_ms": payload.get("duration_ms"),
                "result": payload.get("result"),
                "errors": payload.get("errors"),
            }
        )
    return _json_event({"type": event_type or "raw", "text": line})


def _claude_cli_error_summary(*, status: str, stderr: str, stdout: str) -> str:
    parts = [f"claude cli {status}"]
    if stderr.strip():
        parts.append("stderr: " + stderr.strip()[:4000])
    if stdout.strip():
        parts.append("stdout: " + stdout.strip()[:4000])
    return "\n".join(parts)


def _format_sdk_exception(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, str) and stderr.strip():
        parts.append("stderr: " + stderr.strip()[:4000])
    exit_code = getattr(exc, "exit_code", None)
    if exit_code is not None:
        parts.append(f"exit_code: {exit_code}")
    attrs = {
        key: value
        for key in dir(exc)
        if not key.startswith("_") and key not in {"add_note", "args", "with_traceback"}
        for value in [getattr(exc, key, None)]
        if isinstance(value, str | int | float | bool | type(None))
    }
    if attrs:
        parts.append(
            "attributes: "
            + json.dumps(attrs, ensure_ascii=False, allow_nan=False, sort_keys=True)[:4000]
        )
    if exc.args:
        parts.append(f"args: {exc.args!r}"[:4000])
    return "\n".join(parts)


def export_competition_llm_log(
    *,
    session_path: Path,
    output_path: Path,
    force: bool,
) -> dict[str, Any]:
    session_path = session_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(session, dict):
        raise AgentRuntimeError("session artifact must be a JSON object")
    messages = session.get("sdk_messages")
    if not isinstance(messages, list):
        raise AgentRuntimeError("session artifact missing sdk_messages")
    _guard_log_output(output_path, force=force)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = _competition_log_start_time(session_path)
    rows = []
    for index, raw in enumerate(messages):
        row = _competition_log_row(
            raw_message=str(raw),
            timestamp=start + timedelta(seconds=index),
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise AgentRuntimeError("session artifact produced no competition log rows")
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = {
        "schema_version": 1,
        "state_hint": COMPETITION_LLM_LOG_STATE_HINT,
        "source_session_path": str(session_path),
        "output_path": str(output_path),
        "line_count": len(rows),
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
        "max_window_hours": 12,
        "competition_format": {
            "jsonl": True,
            "required_fields": ["timestamp", "elapsed_seconds"],
            "requires_response_or_tool_calls": True,
        },
    }
    return report


def _guard_log_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise AgentRuntimeError(f"refusing to overwrite existing file: {path}")


def _competition_log_start_time(session_path: Path) -> datetime:
    timestamp = session_path.stat().st_mtime
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _competition_log_row(
    *,
    raw_message: str,
    timestamp: datetime,
) -> dict[str, Any] | None:
    payload = _parse_sdk_event(raw_message)
    event_type = str(payload.get("type") or "")
    response = ""
    tool_calls = ""
    if event_type == "assistant":
        response = str(payload.get("text") or "")
    elif event_type == "tool_use":
        tool_calls = _format_competition_tool_call(payload)
    elif event_type == "result":
        response = str(payload.get("result") or "")
    else:
        text = str(payload.get("text") or "")
        if "ToolUseBlock" in text or "tool_use" in text:
            tool_calls = text
        elif text:
            response = text
    if not response and not tool_calls:
        return None
    row: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "elapsed_seconds": 0.0,
    }
    if response:
        row["response"] = response
    if tool_calls:
        row["tool_calls"] = tool_calls
    return row


def _parse_sdk_event(raw_message: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        return {"type": "raw", "text": raw_message}
    return payload if isinstance(payload, dict) else {"type": "raw", "text": raw_message}


def _format_competition_tool_call(payload: dict[str, Any]) -> str:
    name = str(payload.get("name") or "tool")
    tool_input = payload.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    return f"{name}({json.dumps(tool_input, ensure_ascii=False, sort_keys=True)})"


def record_agent_runtime_issue(
    *,
    output_path: Path,
    failed_command: str,
    artifact_path: Path | None,
    stderr_summary: str,
    likely_schema_gap: bool,
    next_action: str,
    force: bool,
) -> dict[str, Any]:
    output_path = output_path.expanduser().resolve()
    artifact_path = artifact_path.expanduser().resolve() if artifact_path else None
    payload: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": AGENT_RUNTIME_ISSUE_RECORDED_STATE_HINT,
        "failed_command": failed_command,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "stderr_summary": stderr_summary,
        "likely_schema_gap": likely_schema_gap,
        "next_action": next_action,
    }
    _write_json(output_path, payload, force=force)
    return payload
