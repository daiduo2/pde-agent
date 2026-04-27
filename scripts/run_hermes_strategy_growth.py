#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real Hermes FNO strategy-growth test and audit it."
    )
    parser.add_argument(
        "--fixture-root",
        default="integration_tests/hermes_fno_strategy_growth",
    )
    parser.add_argument("--profile", default="hermes-protocol-smoke")
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--report-path")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--session-path")
    args = parser.parse_args()

    repo_root = Path.cwd()
    fixture_root = (repo_root / args.fixture_root).resolve()

    if not args.audit_only:
        prompt = _render_prompt(fixture_root)
        final_state_hint = _success_transition_from_run_state(fixture_root)
        session_id = _run_hermes(
            profile=args.profile,
            fixture_root=fixture_root,
            prompt=prompt,
            timeout_seconds=args.timeout_seconds,
            final_state_hint=final_state_hint,
        )
    else:
        session_id = args.session_id

    session_path = args.session_path
    if args.audit_only and session_path is None and session_id is None:
        session_path = _latest_saved_session_path(repo_root, fixture_root)
    if session_path is None and session_id:
        session_path = _resolve_saved_session_path(repo_root, fixture_root, session_id)

    audit_cmd = [
        sys.executable,
        "scripts/audit_hermes_strategy_growth.py",
        "--fixture-root",
        _fixture_arg(repo_root, fixture_root),
    ]
    if session_id:
        audit_cmd.extend(["--session-id", session_id])
    if session_path:
        audit_cmd.extend(["--session-path", session_path])
    if args.report_path:
        audit_cmd.extend(["--report-path", args.report_path])

    result = subprocess.run(
        audit_cmd,
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode


def _render_prompt(fixture_root: Path) -> str:
    template = (fixture_root / "prompt_template.md").read_text(encoding="utf-8")
    return template.format(
        run_state_path=fixture_root / "run_state.yaml",
        bootstrap_prompt_path=fixture_root / "bootstrap_prompt.md",
        task_spec_path=fixture_root / "task_spec.json",
    )


def _fixture_arg(repo_root: Path, fixture_root: Path) -> str:
    try:
        return str(fixture_root.relative_to(repo_root))
    except ValueError:
        return str(fixture_root)


def _resolve_saved_session_path(
    repo_root: Path,
    fixture_root: Path,
    session_id: str,
) -> str | None:
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
            return str(candidate)
    return None


def _latest_saved_session_path(repo_root: Path, fixture_root: Path) -> str | None:
    sessions: list[Path] = []
    for sessions_dir in (
        fixture_root / "hermes_home" / "sessions",
        repo_root
        / "integration_tests"
        / "hermes_protocol_smoke"
        / "hermes_home"
        / "sessions",
    ):
        sessions.extend(sessions_dir.glob("session_*.json"))
    if not sessions:
        return None
    return str(max(sessions, key=lambda path: path.stat().st_mtime))


def _run_hermes(
    *,
    profile: str,
    fixture_root: Path,
    prompt: str,
    timeout_seconds: int,
    final_state_hint: str,
) -> str | None:
    process = subprocess.Popen(
        ["script", "-q", "/dev/null", "hermes", "-p", profile],
        cwd=fixture_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("failed to open Hermes runner pipes")

    session_id: str | None = None
    output = ""
    deadline = time.time() + timeout_seconds
    prompt_submitted = False
    exit_submitted = False

    while time.time() < deadline:
        readable, _, _ = select.select([process.stdout], [], [], 0.5)
        if readable:
            chunk = process.stdout.read1(4096).decode("utf-8", errors="replace")
            if not chunk:
                break
            output += chunk
            print(chunk, end="", flush=True)
            session_id = session_id or _extract_session_id(output)
            if not prompt_submitted and "❯" in output:
                process.stdin.write((prompt + "\r").encode("utf-8"))
                process.stdin.flush()
                prompt_submitted = True
            elif (
                prompt_submitted
                and not exit_submitted
                and _success_transition_seen(
                    output,
                    final_state_hint=final_state_hint,
                )
            ):
                process.stdin.write(b"/exit\r")
                process.stdin.flush()
                exit_submitted = True

        if process.poll() is not None:
            break

    if process.poll() is None:
        process.terminate()
        raise TimeoutError(
            f"Hermes strategy-growth run timed out after {timeout_seconds} seconds"
        )
    if process.returncode != 0:
        raise RuntimeError(f"Hermes exited with code {process.returncode}")
    return session_id


def _success_transition_seen(
    output: str,
    *,
    final_state_hint: str = "STRATEGY_LEDGER_APPENDED",
) -> bool:
    normalized = output.replace("*", "").replace("`", "")
    if final_state_hint not in normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "Protocol complete",
            "Final command returned",
            "Success transition",
            "completed successfully",
            "No forbidden actions",
            "Final state",
            "Final state_hint",
            "state_hint",
        )
    )


def _success_transition_from_run_state(fixture_root: Path) -> str:
    run_state_path = fixture_root / "run_state.yaml"
    if not run_state_path.exists():
        return "STRATEGY_LEDGER_APPENDED"
    inside = False
    for line in run_state_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "success_transition:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            break
        if inside and stripped.startswith("on_complete: "):
            return stripped.removeprefix("on_complete: ")
    return "STRATEGY_LEDGER_APPENDED"


def _extract_session_id(output: str) -> str | None:
    for line in output.splitlines():
        if line.startswith("Session:"):
            return line.split(":", 1)[1].strip()
    return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps({"status": "error", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        raise SystemExit(1)
