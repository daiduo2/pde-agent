from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

STATE_HINT = "REAL_HDF5_BASELINE_SMOKE_COMPLETED"


def audit_hermes_real_hdf5_baseline_session(
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
    summary_ok, summary = _summary_artifact_ok(
        Path(required_artifacts[0]) if required_artifacts else None
    )
    _add_check(
        checks,
        errors,
        "summary_artifact",
        summary_ok,
        {
            "summary_path": required_artifacts[0] if required_artifacts else None,
            "summary": summary,
        },
    )

    return {
        "status": "passed" if not errors else "failed",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "required_artifacts": required_artifacts,
        "checks": checks,
        "tool_call_summary": tool_summary,
        "terminal_commands": terminal_commands,
        "summary": summary,
        "errors": errors,
    }


def _required_artifacts_exist(paths: list[str]) -> bool:
    return bool(paths) and all(Path(path).exists() for path in paths)


def _summary_artifact_ok(path: Path | None) -> tuple[bool, dict[str, Any]]:
    if path is None or not path.exists():
        return False, {}
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("metrics")
    artifacts = report.get("artifacts")
    if not isinstance(metrics, dict) or not isinstance(artifacts, dict):
        return False, {}

    summary = {
        "sample_count": metrics.get("sample_count"),
        "forecast_steps": metrics.get("forecast_steps"),
        "competition_like_score_short": metrics.get("competition_like_score_short"),
        "rel_mse": metrics.get("rel_mse"),
        "residual_rmse": metrics.get("residual_rmse"),
        "dominant_error_pattern": metrics.get("dominant_error_pattern"),
        "physics_issue": metrics.get("physics_issue"),
        "residual_issue": metrics.get("residual_issue"),
    }
    artifact_paths_ok = all(
        isinstance(value, str) and Path(value).exists()
        for value in artifacts.values()
    )
    ok = (
        report.get("state_hint") == STATE_HINT
        and isinstance(summary["sample_count"], int)
        and summary["sample_count"] > 0
        and isinstance(summary["forecast_steps"], int)
        and summary["forecast_steps"] > 0
        and isinstance(summary["competition_like_score_short"], int | float)
        and isinstance(summary["rel_mse"], int | float)
        and isinstance(summary["residual_rmse"], int | float)
        and isinstance(summary["dominant_error_pattern"], str)
        and bool(summary["dominant_error_pattern"])
        and isinstance(summary["physics_issue"], str)
        and bool(summary["physics_issue"])
        and isinstance(summary["residual_issue"], str)
        and bool(summary["residual_issue"])
        and artifact_paths_ok
    )
    return ok, summary


def _forbidden_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    forbidden_terms = [
        " run-fno-iteration",
        " train-baseline",
        " build-submission",
        " deeponet",
        " pideeponet",
        " --help",
        " curl ",
        " wget ",
        " python ",
        " python3 ",
        " cat ",
        " ls",
        " find ",
        " grep ",
        " sed ",
        " head ",
        " tail ",
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
        "summary": {},
        "errors": [message],
    }


def _session_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("session_")
