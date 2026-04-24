from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def audit_hermes_diagnostic_session(
    *,
    session_path: Path,
    manifest_path: Path,
    required_reads: list[str],
    allowed_commands: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    if not session_path.exists():
        return _error_report(
            session_path, manifest_path, f"session file not found: {session_path}"
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
    _add_check(
        checks,
        errors,
        "forbidden_actions",
        not _forbidden_calls(tool_calls),
        {"offenders": _forbidden_calls(tool_calls)},
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
        and final_payload.get("state_hint") == "HYPOTHESIS_SELECTED",
        {"final_payload": final_payload},
    )
    manifest_ok, diagnosis = _manifest_ok(manifest_path)
    _add_check(
        checks,
        errors,
        "manifest",
        manifest_ok,
        {"manifest_path": str(manifest_path)},
    )

    return {
        "status": "passed" if not errors else "failed",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "manifest_path": str(manifest_path),
        "checks": checks,
        "tool_call_summary": tool_summary,
        "terminal_commands": terminal_commands,
        "diagnosis": diagnosis,
        "errors": errors,
    }


def _manifest_ok(manifest_path: Path) -> tuple[bool, dict[str, object]]:
    if not manifest_path.exists():
        return False, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    summary = manifest.get("summary", {})
    required_artifacts = [
        "rollout_metrics_path",
        "spectrum_metrics_path",
        "physics_check_path",
        "error_interpretation_path",
        "selected_hypothesis_path",
    ]
    diagnosis = {
        "dominant_failure_mode": summary.get("dominant_failure_mode"),
        "selected_hypothesis": summary.get("selected_hypothesis"),
    }
    ok = (
        manifest.get("status") == "ok"
        and manifest.get("phase") == "hypothesis_selected"
        and manifest.get("errors") == []
        and all(artifacts.get(key) for key in required_artifacts)
        and bool(summary.get("dominant_spectral_issue"))
        and bool(summary.get("physics_issue"))
        and bool(summary.get("dominant_failure_mode"))
        and bool(summary.get("selected_hypothesis"))
    )
    return ok, diagnosis


def _forbidden_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    forbidden_terms = [
        " train-baseline",
        " build-submission",
        " --help",
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


def _error_report(session_path: Path, manifest_path: Path, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "manifest_path": str(manifest_path),
        "checks": [],
        "tool_call_summary": {},
        "terminal_commands": [],
        "diagnosis": {},
        "errors": [message],
    }


def _session_id_from_path(path: Path) -> str:
    name = path.stem
    return name.removeprefix("session_")
