from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


FINAL_STATE_HINT = "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
PROPOSAL_ID = "burgers_spectral_viscosity_real_hdf5_v0"
VALID_DECISIONS = {
    "evaluate_promotion",
    "expand_evidence",
    "retrieve_external_strategy",
}


def audit_hermes_burgers_spectral_viscosity_benchmark_session(
    *,
    session_path: Path,
    required_artifacts: list[str],
    required_reads: list[str],
    allowed_commands: list[str],
    min_evaluation_windows: int,
    arm: str = "burgers_spectral_viscosity_benchmark",
    final_state_hint: str = FINAL_STATE_HINT,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    if not session_path.exists():
        return _error_report(
            session_path,
            required_artifacts,
            f"session file not found: {session_path}",
            arm=arm,
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
        and final_payload.get("state_hint") == final_state_hint,
        {"final_payload": final_payload},
    )
    _add_check(
        checks,
        errors,
        "required_artifacts",
        _required_artifacts_exist(required_artifacts),
        {"required_artifacts": required_artifacts},
    )
    report_ok, benchmark_report = _benchmark_report_ok(
        Path(required_artifacts[0]) if required_artifacts else None,
        min_evaluation_windows=min_evaluation_windows,
        final_state_hint=final_state_hint,
    )
    _add_check(
        checks,
        errors,
        "benchmark_report_semantics",
        report_ok,
        {
            "benchmark_report_path": required_artifacts[0]
            if required_artifacts
            else None,
            "min_evaluation_windows": min_evaluation_windows,
        },
    )

    decision = (
        benchmark_report.get("decision", {}).get("decision")
        if isinstance(benchmark_report.get("decision"), dict)
        else None
    )
    return {
        "status": "passed" if not errors else "failed",
        "arm": arm,
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "required_artifacts": required_artifacts,
        "checks": checks,
        "tool_call_summary": tool_summary,
        "terminal_commands": terminal_commands,
        "benchmark_report": benchmark_report,
        "metrics": {
            "command_count": len(terminal_commands),
            "evaluation_window_count": benchmark_report.get("evaluation_window_count"),
            "decision": decision,
        },
        "errors": errors,
    }


def _required_artifacts_exist(paths: list[str]) -> bool:
    return bool(paths) and all(Path(path).exists() for path in paths)


def _benchmark_report_ok(
    path: Path | None,
    *,
    min_evaluation_windows: int,
    final_state_hint: str,
) -> tuple[bool, dict[str, Any]]:
    if path is None or not path.exists():
        return False, {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, {}
    if not isinstance(report, dict):
        return False, {}
    metrics = report.get("metrics")
    decision = report.get("decision")
    artifacts = report.get("artifacts")
    ok = (
        report.get("state_hint") == final_state_hint
        and report.get("proposal_id") == PROPOSAL_ID
        and _integer_at_least(
            report.get("evaluation_window_count"),
            min_evaluation_windows,
        )
        and isinstance(metrics, dict)
        and _is_number(metrics.get("mean_score_delta"))
        and _is_number(metrics.get("long_window_score_delta"))
        and _is_number(metrics.get("failure_rate"))
        and isinstance(decision, dict)
        and decision.get("decision") in VALID_DECISIONS
        and bool(decision.get("next_command"))
        and isinstance(artifacts, dict)
        and bool(artifacts.get("strategy_decision_path"))
        and bool(artifacts.get("strategy_evidence_summary_path"))
    )
    return ok, report


def _forbidden_calls(
    tool_calls: list[dict[str, Any]],
    required_reads: list[str],
) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    forbidden_terms = [
        " run-real-hdf5-baseline-smoke",
        " run-deeponet-strategy-sweep",
        " convert-operator-evidence-summary",
        " recommend-strategy-next-action",
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
    *,
    arm: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "arm": arm,
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "required_artifacts": required_artifacts,
        "checks": [],
        "tool_call_summary": {},
        "terminal_commands": [],
        "benchmark_report": {},
        "metrics": {"command_count": 0},
        "errors": [message],
    }


def _session_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("session_")


def _integer_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
