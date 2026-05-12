
import json
from pathlib import Path
from typing import Any


STATE_HINT = "AGENT_RUNTIME_AVAILABILITY_AUDITED"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _extract_tool_use_names(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    names: list[str] = []
    for raw in messages:
        payload = raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") != "tool_use":
            continue
        name = payload.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_tool_catalog(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        return {}
    catalog: dict[str, dict[str, Any]] = {}
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        command = item.get("command", name)
        if isinstance(command, str) and command:
            catalog[command] = item
    return catalog


def _load_canary_evidence(paths: list[Path]) -> list[dict[str, Any]]:
    evidence = []
    for path in paths:
        if not path.exists():
            evidence.append(
                {
                    "path": str(path),
                    "status": "missing",
                    "passed": False,
                    "session_id": None,
                    "terminal_command_count": 0,
                }
            )
            continue
        payload = _read_json(path)
        status = payload.get("status") or payload.get("decision")
        state_hint = payload.get("state_hint")
        session_tool_policy = payload.get("tool_policy")
        allowed_tools = []
        if isinstance(session_tool_policy, dict):
            allowed_tools = _string_list(session_tool_policy.get("allowed_tools"))
        sdk_messages = payload.get("sdk_messages")
        tool_use_names = _extract_tool_use_names(sdk_messages)
        unexpected_tool_uses = [
            name for name in tool_use_names if allowed_tools and name not in allowed_tools
        ]
        passed = status == "passed" or (
            status == "completed" and state_hint == "CLAUDE_AGENT_SDK_RUNTIME_RAN"
        )
        if unexpected_tool_uses:
            passed = False
        terminal_commands = payload.get("terminal_commands", [])
        if not isinstance(terminal_commands, list):
            terminal_commands = []
        evidence.append(
            {
                "path": str(path),
                "status": status,
                "passed": passed,
                "session_id": payload.get("session_id"),
                "terminal_command_count": len(terminal_commands),
                "state_hint": state_hint,
                "sdk_message_count": payload.get("sdk_message_count", 0),
                "tool_call_summary": payload.get("tool_call_summary", {}),
                "allowed_tools": allowed_tools,
                "tool_use_names": tool_use_names,
                "unexpected_tool_uses": unexpected_tool_uses,
            }
        )
    return evidence


def _readiness(
    *,
    cli_exposed: bool,
    tool_catalog_exposed: bool,
    required: bool,
) -> str:
    if tool_catalog_exposed:
        return "discoverable"
    if cli_exposed:
        return "cli_only"
    if required:
        return "missing"
    return "unknown"


def _runtime_metadata(runtime_name: str) -> dict[str, Any]:
    if runtime_name == "claude_agent_sdk":
        return {
            "name": "claude_agent_sdk",
            "primary": True,
            "compatibility_backends": [],
        }
    if runtime_name == "claude_agent_sdk":
        return {
            "name": "claude_agent_sdk",
            "primary": False,
            "compatibility_backends": [],
        }
    return {
        "name": runtime_name,
        "primary": False,
        "compatibility_backends": [],
    }


def audit_agent_runtime_availability(
    *,
    runtime_name: str,
    output_path: Path,
    tool_catalog_path: Path,
    canary_report_paths: list[Path],
    cli_commands: list[str],
    required_commands: list[str],
    force: bool,
) -> dict[str, Any]:
    catalog = _load_tool_catalog(tool_catalog_path)
    cli_set = set(cli_commands)
    required_set = set(required_commands)
    command_names = sorted(cli_set | set(catalog) | required_set)
    canary_evidence = _load_canary_evidence(canary_report_paths)

    matrix: list[dict[str, Any]] = []
    for command in command_names:
        tool = catalog.get(command, {})
        cli_exposed = command in cli_set
        tool_catalog_exposed = command in catalog
        required = command in required_set
        gaps = []
        if not cli_exposed:
            gaps.append("not_cli_exposed")
        if not tool_catalog_exposed:
            gaps.append("not_in_tool_catalog")
        if required and not tool_catalog_exposed:
            gaps.append("required_but_not_discoverable")
        matrix.append(
            {
                "command": command,
                "required": required,
                "cli_exposed": cli_exposed,
                "tool_catalog_exposed": tool_catalog_exposed,
                "risk_level": tool.get("risk_level"),
                "state_hints": tool.get("state_hints", []),
                "readiness": _readiness(
                    cli_exposed=cli_exposed,
                    tool_catalog_exposed=tool_catalog_exposed,
                    required=required,
                ),
                "gaps": gaps,
            }
        )

    required_entries = [entry for entry in matrix if entry["required"]]
    required_ready_count = sum(
        1
        for entry in required_entries
        if entry["readiness"] == "discoverable"
    )
    passed_canary_count = sum(1 for item in canary_evidence if item["passed"])
    tool_policy_violation_count = sum(
        len(item.get("unexpected_tool_uses", [])) for item in canary_evidence
    )
    missing_required = [
        entry["command"]
        for entry in required_entries
        if entry["readiness"] != "discoverable"
    ]
    decision = (
        "passed"
        if not missing_required
        and passed_canary_count > 0
        and tool_policy_violation_count == 0
        else "needs_work"
    )

    report: dict[str, Any] = {
        "state_hint": STATE_HINT,
        "decision": decision,
        "runtime": _runtime_metadata(runtime_name),
        "tool_catalog_path": str(tool_catalog_path),
        "canary_evidence": canary_evidence,
        "matrix": matrix,
        "metrics": {
            "command_count": len(matrix),
            "cli_exposed_count": sum(1 for entry in matrix if entry["cli_exposed"]),
            "tool_catalog_exposed_count": sum(
                1 for entry in matrix if entry["tool_catalog_exposed"]
            ),
            "required_command_count": len(required_entries),
            "required_ready_count": required_ready_count,
            "passed_canary_count": passed_canary_count,
            "missing_required_count": len(missing_required),
            "tool_policy_violation_count": tool_policy_violation_count,
        },
        "gaps": {
            "missing_required_commands": missing_required,
            "cli_only_commands": [
                entry["command"]
                for entry in matrix
                if entry["readiness"] == "cli_only"
            ],
            "unexpected_tool_uses": sorted(
                {
                    name
                    for item in canary_evidence
                    for name in item.get("unexpected_tool_uses", [])
                }
            ),
        },
        "next_command": (
            "run-claude-agent-sdk-runtime"
            if runtime_name == "claude_agent_sdk"
            and (missing_required or tool_policy_violation_count > 0)
            else "legacy -p"
            if missing_required or tool_policy_violation_count > 0
            else None
        ),
    }
    _write_json(output_path, report, force=force)
    return report


def audit_availability(
    *,
    output_path: Path,
    tool_catalog_path: Path,
    canary_report_paths: list[Path],
    cli_commands: list[str],
    required_commands: list[str],
    force: bool,
) -> dict[str, Any]:
    return audit_agent_runtime_availability(
        runtime_name="legacy",
        output_path=output_path,
        tool_catalog_path=tool_catalog_path,
        canary_report_paths=canary_report_paths,
        cli_commands=cli_commands,
        required_commands=required_commands,
        force=force,
    )
