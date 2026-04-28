#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pde_agent.core.hermes_burgers_spectral_viscosity_benchmark_audit import (
    audit_hermes_burgers_spectral_viscosity_benchmark_session,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a saved Hermes Burgers spectral-viscosity benchmark session."
    )
    parser.add_argument(
        "--fixture-root",
        default="integration_tests/hermes_burgers_spectral_viscosity_benchmark",
    )
    parser.add_argument("--session-id")
    parser.add_argument("--session-path")
    parser.add_argument("--report-path")
    parser.add_argument("--arm", default="burgers_spectral_viscosity_benchmark")
    args = parser.parse_args()

    repo_root = Path.cwd()
    fixture_root = (repo_root / args.fixture_root).resolve()
    session_path = _resolve_session_path(
        repo_root,
        fixture_root,
        args.session_id,
        args.session_path,
    )
    run_state = (fixture_root / "run_state.yaml").read_text(encoding="utf-8")
    required_reads = _yaml_list(run_state, "required_reads")
    required_artifacts = _yaml_list(run_state, "required_artifacts")
    allowed_commands = _yaml_command_list(run_state)
    final_state_hint = _yaml_success_transition(run_state)
    min_evaluation_windows = _yaml_scalar_int(run_state, "min_evaluation_windows", 4)

    report = audit_hermes_burgers_spectral_viscosity_benchmark_session(
        session_path=session_path,
        required_artifacts=required_artifacts,
        required_reads=required_reads,
        allowed_commands=allowed_commands,
        min_evaluation_windows=min_evaluation_windows,
        arm=args.arm,
        final_state_hint=final_state_hint,
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
    repo_root: Path,
    fixture_root: Path,
    session_id: str | None,
    session_path: str | None,
) -> Path:
    if session_path:
        return Path(session_path).resolve()
    if session_id:
        candidates = [
            fixture_root / "hermes_home" / "sessions" / f"session_{session_id}.json",
            repo_root
            / "integration_tests"
            / "hermes_protocol_smoke"
            / "hermes_home"
            / "sessions"
            / f"session_{session_id}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    sessions_dir = fixture_root / "hermes_home" / "sessions"
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


def _yaml_success_transition(text: str) -> str:
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "success_transition:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            break
        if inside and stripped.startswith("on_complete: "):
            return stripped.removeprefix("on_complete: ")
    return "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"


def _yaml_scalar_int(text: str, key: str, default: int) -> int:
    prefix = f"{key}: "
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return int(stripped.removeprefix(prefix))
    return default


if __name__ == "__main__":
    raise SystemExit(main())
