from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


SUCCESS_STATE_HINT = "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED"


def audit_hermes_hypothesis_lab_infrastructure_session(
    *,
    session_path: Path,
    required_artifacts: list[str],
    required_reads: list[str],
    allowed_commands: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    if not session_path.exists():
        return _error_report(
            session_path,
            required_artifacts,
            f"session file not found: {session_path}",
        )

    session = json.loads(session_path.read_text(encoding="utf-8"))
    messages = session.get("messages", [])
    if not any(message.get("role") == "user" for message in messages):
        errors.append("session has no user message")

    tool_calls = _extract_tool_calls(messages)
    read_paths = [
        call["arguments"].get("path")
        for call in tool_calls
        if call["name"] == "read_file"
    ]
    terminal_commands = [
        call["arguments"].get("command")
        for call in tool_calls
        if call["name"] == "terminal"
    ]
    tool_summary = dict(Counter(call["name"] for call in tool_calls))

    _add_check(
        checks,
        errors,
        "required_reads",
        read_paths[: len(required_reads)] == required_reads,
        {"expected": required_reads, "actual_prefix": read_paths[: len(required_reads)]},
    )
    _add_check(
        checks,
        errors,
        "allowed_command_sequence",
        terminal_commands == allowed_commands,
        {"expected": allowed_commands, "actual": terminal_commands},
    )
    forbidden_calls = _forbidden_calls(tool_calls)
    _add_check(
        checks,
        errors,
        "forbidden_actions",
        not forbidden_calls,
        {"offenders": forbidden_calls},
    )
    _add_check(
        checks,
        errors,
        "terminal_exit_codes",
        _terminal_exit_codes_ok(messages),
        {"terminal_result_count": len(_terminal_results(messages))},
    )
    final_payload = _final_terminal_payload(messages)
    _add_check(
        checks,
        errors,
        "final_state",
        final_payload.get("status") == "ok"
        and final_payload.get("state_hint") == SUCCESS_STATE_HINT,
        {"final_payload": final_payload},
    )
    artifact_ok = _required_artifacts_exist(required_artifacts)
    _add_check(
        checks,
        errors,
        "required_artifacts",
        artifact_ok,
        {"required_artifacts": required_artifacts},
    )
    infrastructure_ok, infrastructure = _infrastructure_report_ok(
        Path(required_artifacts[0]) if required_artifacts else None
    )
    _add_check(
        checks,
        errors,
        "infrastructure_report",
        infrastructure_ok,
        {"infrastructure_report_path": required_artifacts[0] if required_artifacts else None},
    )

    return {
        "status": "passed" if not errors else "failed",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "required_artifacts": required_artifacts,
        "checks": checks,
        "tool_call_summary": tool_summary,
        "terminal_commands": terminal_commands,
        "infrastructure": infrastructure,
        "errors": errors,
    }


def _required_artifacts_exist(paths: list[str]) -> bool:
    return bool(paths) and all(Path(path).exists() for path in paths)


def _infrastructure_report_ok(path: Path | None) -> tuple[bool, dict[str, Any]]:
    if path is None or not path.exists():
        return False, {}
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report.get("summary")
    infrastructure = {
        "decision": report.get("decision"),
        "check_count": summary.get("check_count") if isinstance(summary, dict) else None,
        "failed_check_count": summary.get("failed_check_count")
        if isinstance(summary, dict)
        else None,
    }
    ok = (
        report.get("state_hint") == SUCCESS_STATE_HINT
        and report.get("decision") == "passed"
        and isinstance(summary, dict)
        and isinstance(summary.get("check_count"), int)
        and summary.get("check_count", 0) > 0
        and summary.get("failed_check_count") == 0
    )
    return ok, infrastructure


def _forbidden_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    forbidden_terms = [
        " --help",
        " curl ",
        " wget ",
        " cat ",
        " ls ",
        " find ",
        " grep ",
        " rg ",
        " sed ",
        " python ",
    ]
    for call in tool_calls:
        name = call["name"]
        command = call["arguments"].get("command", "")
        if name not in {"read_file", "terminal"}:
            offenders.append(call)
        elif any(term in command for term in forbidden_terms):
            offenders.append(call)
    return offenders


def _extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = json.loads(function.get("arguments") or "{}")
            calls.append({"name": function.get("name", ""), "arguments": arguments})
    return calls


def _terminal_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content") or "{}")
        except json.JSONDecodeError:
            result = {"parse_error": message.get("content")}
        if "exit_code" in result or "output" in result or "parse_error" in result:
            results.append(result)
    return results


def _terminal_exit_codes_ok(messages: list[dict[str, Any]]) -> bool:
    results = _terminal_results(messages)
    return bool(results) and all(result.get("exit_code") == 0 for result in results)


def _final_terminal_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    results = _terminal_results(messages)
    if not results:
        return {}
    try:
        return json.loads(results[-1].get("output") or "{}")
    except json.JSONDecodeError:
        return {"parse_error": results[-1].get("output")}


def _add_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    passed: bool,
    details: dict[str, Any],
) -> None:
    checks.append(
        {"name": name, "status": "passed" if passed else "failed", "details": details}
    )
    if not passed:
        errors.append(name)


def _error_report(
    session_path: Path,
    required_artifacts: list[str],
    message: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "required_artifacts": required_artifacts,
        "checks": [],
        "tool_call_summary": {},
        "terminal_commands": [],
        "infrastructure": {},
        "errors": [message],
    }


def _session_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("session_")
