"""Audit script for FNO training agent sessions.

Reads an SDK session JSON and validates it against the expected behavior
contract: required reads, allowed commands, forbidden actions, artifacts.
"""

import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: audit_fno_training_agent_session.py <session.json> [run_state.yaml]", file=sys.stderr)
        sys.exit(1)

    session_path = Path(sys.argv[1]).expanduser().resolve()
    run_state_path = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else None

    session = _load_json(session_path)
    if session is None:
        _emit_audit_report(passed=False, errors=["could not load session JSON"])
        sys.exit(2)

    run_state: dict[str, Any] = {}
    if run_state_path is not None and run_state_path.is_file():
        run_state = _load_yaml(run_state_path) or {}

    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    # Check forbidden tool usage
    forbidden_tools = set(run_state.get("forbidden_actions", ["Edit", "Write", "NotebookEdit", "WebFetch"]))
    tool_events = _extract_tool_events(session)
    used_tools = {event.get("name") for event in tool_events if event.get("name")}
    forbidden_used = used_tools & forbidden_tools
    checks.append({
        "check": "forbidden_tools",
        "passed": not bool(forbidden_used),
        "forbidden_tools": sorted(forbidden_tools),
        "used_tools": sorted(used_tools),
        "violations": sorted(forbidden_used),
    })
    if forbidden_used:
        errors.append(f"forbidden tools used: {sorted(forbidden_used)}")

    # Check allowed commands
    allowed_commands = set(run_state.get("allowed_commands", []))
    if allowed_commands:
        bash_events = [e for e in tool_events if e.get("name") == "Bash"]
        command_violations = []
        for event in bash_events:
            command = _extract_pde_agent_command(event)
            if command and command not in allowed_commands:
                command_violations.append(command)
        checks.append({
            "check": "allowed_commands",
            "passed": not bool(command_violations),
            "allowed_commands": sorted(allowed_commands),
            "violations": command_violations,
        })
        if command_violations:
            errors.append(f"disallowed commands used: {command_violations}")

    # Check required reads
    required_reads = set(run_state.get("required_reads", []))
    if required_reads:
        read_events = [e for e in tool_events if e.get("name") == "Read"]
        read_paths = {str(event.get("input", {}).get("file_path", "")) for event in read_events}
        missing_reads = required_reads - read_paths
        checks.append({
            "check": "required_reads",
            "passed": not bool(missing_reads),
            "required_reads": sorted(required_reads),
            "read_paths": sorted(read_paths),
            "missing": sorted(missing_reads),
        })
        if missing_reads:
            errors.append(f"required reads missing: {sorted(missing_reads)}")

    # Check required artifacts exist
    required_artifacts = run_state.get("required_artifacts", [])
    workspace_dir = Path(session.get("workspace_dir", ".")).expanduser().resolve()
    artifact_checks = []
    for artifact in required_artifacts:
        path = workspace_dir / artifact if not Path(artifact).is_absolute() else Path(artifact)
        exists = path.is_file()
        artifact_checks.append({"path": str(path), "exists": exists})
        if not exists:
            errors.append(f"missing required artifact: {path}")
    checks.append({
        "check": "required_artifacts",
        "passed": all(a["exists"] for a in artifact_checks),
        "artifacts": artifact_checks,
    })

    # Check success transition
    success_transition = run_state.get("success_transition")
    if success_transition:
        final_state = _extract_final_state(session)
        passed = final_state == success_transition
        checks.append({
            "check": "success_transition",
            "passed": passed,
            "expected": success_transition,
            "actual": final_state,
        })
        if not passed:
            errors.append(f"expected final state {success_transition}, got {final_state}")

    _emit_audit_report(
        passed=not errors,
        errors=errors,
        checks=checks,
        session_path=str(session_path),
        run_state_path=str(run_state_path) if run_state_path else None,
    )

    if errors:
        sys.exit(3)


def _emit_audit_report(
    *,
    passed: bool,
    errors: list[str],
    checks: list[dict[str, Any]] | None = None,
    session_path: str | None = None,
    run_state_path: str | None = None,
) -> None:
    report = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_AGENT_SESSION_AUDITED",
        "passed": passed,
        "session_path": session_path,
        "run_state_path": run_state_path,
        "errors": errors,
        "checks": checks or [],
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))


def _extract_tool_events(session: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in session.get("sdk_messages", []):
        payload = _parse_event(raw)
        if payload.get("type") == "tool_use":
            events.append(payload)
    return events


def _extract_pde_agent_command(event: dict[str, Any]) -> str | None:
    tool_input = event.get("input", {})
    if not isinstance(tool_input, dict):
        return None
    command = tool_input.get("command", "")
    if not isinstance(command, str):
        return None
    prefix = "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
    if command.startswith(prefix):
        parts = command[len(prefix):].strip().split()
        if parts:
            return parts[0].replace("_", "-")
    return None


def _extract_final_state(session: dict[str, Any]) -> str | None:
    events = session.get("sdk_messages", [])
    for raw in reversed(events):
        payload = _parse_event(raw)
        if payload.get("type") == "result":
            return str(payload.get("result") or "")
    return session.get("status")


def _parse_event(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"type": "raw", "text": str(raw)}
    return payload if isinstance(payload, dict) else {"type": "raw", "text": str(raw)}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except ImportError:
        # Fallback: try basic parsing if yaml is not available
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


if __name__ == "__main__":
    main()
