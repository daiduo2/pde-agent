from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pde_agent.core.autonomy_decision import (
    DECISION_NEXT_COMMANDS,
    STATE_HINT,
    VALID_REASON_CODES,
)


def audit_hermes_autonomy_session(
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
    command_ok = (
        len(terminal_commands) == 2
        and bool(allowed_commands)
        and terminal_commands[0] == allowed_commands[0]
        and terminal_commands[1] in allowed_commands[1:]
    )
    _add_check(
        checks,
        errors,
        "allowed_command_sequence",
        command_ok,
        {
            "expected_first": allowed_commands[0] if allowed_commands else None,
            "expected_second_one_of": allowed_commands[1:],
            "actual": terminal_commands,
        },
    )
    forbidden_calls = _forbidden_calls(tool_calls, required_reads)
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
        and final_payload.get("state_hint") == STATE_HINT,
        {"final_payload": final_payload},
    )
    _add_check(
        checks,
        errors,
        "required_artifacts",
        _required_artifacts_exist(required_artifacts),
        {"required_artifacts": required_artifacts},
    )
    decision_ok, decision, decision_quality_score = _decision_artifact_ok(
        Path(required_artifacts[-1]) if required_artifacts else None
    )
    _add_check(
        checks,
        errors,
        "decision_artifact",
        decision_ok,
        {"decision_artifact_path": required_artifacts[-1] if required_artifacts else None},
    )

    return {
        "status": "passed" if not errors else "failed",
        "arm": "constrained_free",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "required_artifacts": required_artifacts,
        "checks": checks,
        "tool_call_summary": tool_summary,
        "terminal_commands": terminal_commands,
        "decision": decision,
        "metrics": {
            "command_count": len(terminal_commands),
            "decision_quality_score": decision_quality_score,
        },
        "errors": errors,
    }


def _required_artifacts_exist(paths: list[str]) -> bool:
    return bool(paths) and all(Path(path).exists() for path in paths)


def _decision_artifact_ok(path: Path | None) -> tuple[bool, dict[str, Any], int]:
    if path is None or not path.exists():
        return False, {}, 0
    decision = json.loads(path.read_text(encoding="utf-8"))
    evidence_path = Path(str(decision.get("evidence_source_path", "")))
    expected_next_command = DECISION_NEXT_COMMANDS.get(str(decision.get("decision")))
    ok = (
        decision.get("schema_version") == 1
        and decision.get("state_hint") == STATE_HINT
        and decision.get("decision") in DECISION_NEXT_COMMANDS
        and decision.get("reason_code") in VALID_REASON_CODES
        and decision.get("autonomy_decision_path") == str(path)
        and decision.get("next_command") == expected_next_command
        and evidence_path.exists()
    )
    score = 4 if ok else 0
    return ok, decision, score


def _forbidden_calls(
    tool_calls: list[dict[str, Any]],
    required_reads: list[str],
) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    forbidden_terms = [
        " execute-fno-finetune-candidate",
        " evaluate-fno-candidate",
        " compare-fno-candidates",
        " build-submission",
        " train-baseline",
        " --help",
        " curl ",
        " wget ",
    ]
    forbidden_prefixes = ("cat ", "less ", "head ", "tail ", "python -c ")
    for call in tool_calls:
        name = call["name"]
        command = call["arguments"].get("command", "")
        path = call["arguments"].get("path", "")
        if name == "read_file":
            if path not in required_reads:
                offenders.append(call)
        elif name == "terminal":
            if command.startswith(forbidden_prefixes) or any(
                term in command for term in forbidden_terms
            ):
                offenders.append(call)
        else:
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
        "arm": "constrained_free",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "required_artifacts": required_artifacts,
        "checks": [],
        "tool_call_summary": {},
        "terminal_commands": [],
        "decision": {},
        "metrics": {"command_count": 0, "decision_quality_score": 0},
        "errors": [message],
    }


def _session_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("session_")
