#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pde_agent.core.hermes_fno_iteration_audit import (
    audit_hermes_fno_iteration_session,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a saved Hermes FNO iteration smoke-test session."
    )
    parser.add_argument(
        "--fixture-root",
        default="integration_tests/hermes_fno_iteration_smoke",
    )
    parser.add_argument("--session-id")
    parser.add_argument("--session-path")
    parser.add_argument("--report-path")
    args = parser.parse_args()

    repo_root = Path.cwd()
    fixture_root = (repo_root / args.fixture_root).resolve()
    session_path = _resolve_session_path(fixture_root, args.session_id, args.session_path)
    run_state = (fixture_root / "run_state.yaml").read_text(encoding="utf-8")
    required_reads = _yaml_list(run_state, "required_reads")
    required_artifacts = _yaml_list(run_state, "required_artifacts")
    allowed_commands = _yaml_command_list(run_state)

    report = audit_hermes_fno_iteration_session(
        session_path=session_path,
        required_artifacts=required_artifacts,
        required_reads=required_reads,
        allowed_commands=allowed_commands,
    )
    report_path = (
        Path(args.report_path)
        if args.report_path
        else fixture_root / "reports" / f"audit_{report['session_id']}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "report_path": str(report_path)},
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


def _resolve_session_path(
    fixture_root: Path,
    session_id: str | None,
    session_path: str | None,
) -> Path:
    if session_path:
        return Path(session_path).resolve()
    sessions_dir = fixture_root / "hermes_home" / "sessions"
    if session_id:
        return sessions_dir / f"session_{session_id}.json"
    sessions = sorted(
        sessions_dir.glob("session_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not sessions:
        return sessions_dir / "session_missing.json"
    return sessions[-1]


def _yaml_list(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    values: list[str] = []
    inside = False
    for line in lines:
        if line.strip() == f"{key}:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            break
        if inside and line.strip().startswith("- "):
            values.append(line.strip()[2:])
    return values


def _yaml_command_list(text: str) -> list[str]:
    commands: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))
    return commands


if __name__ == "__main__":
    raise SystemExit(main())
