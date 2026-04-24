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
        description="Run real Hermes protocol smoke test and audit it."
    )
    parser.add_argument("--fixture-root", default="integration_tests/hermes_protocol_smoke")
    parser.add_argument("--profile", default="hermes-protocol-smoke")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--report-path")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--session-path")
    args = parser.parse_args()

    repo_root = Path.cwd()
    fixture_root = (repo_root / args.fixture_root).resolve()

    if not args.audit_only:
        prompt = _render_prompt(fixture_root)
        session_id = _run_hermes(
            profile=args.profile,
            fixture_root=fixture_root,
            prompt=prompt,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        session_id = args.session_id

    audit_cmd = [
        sys.executable,
        "scripts/audit_hermes_smoke.py",
        "--fixture-root",
        _fixture_arg(repo_root, fixture_root),
    ]
    if session_id:
        audit_cmd.extend(["--session-id", session_id])
    if args.session_path:
        audit_cmd.extend(["--session-path", args.session_path])
    if args.report_path:
        audit_cmd.extend(["--report-path", args.report_path])

    result = subprocess.run(
        audit_cmd, cwd=repo_root, check=False, text=True, capture_output=True
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
        base_config_path=fixture_root / "base_config.json",
        override_path=fixture_root / "override.json",
    )


def _fixture_arg(repo_root: Path, fixture_root: Path) -> str:
    try:
        return str(fixture_root.relative_to(repo_root))
    except ValueError:
        return str(fixture_root)


def _run_hermes(
    *, profile: str, fixture_root: Path, prompt: str, timeout_seconds: int
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
                and _success_transition_seen(output)
            ):
                process.stdin.write(b"/exit\r")
                process.stdin.flush()
                exit_submitted = True

        if process.poll() is not None:
            break

    if process.poll() is None:
        process.terminate()
        raise TimeoutError(f"Hermes smoke run timed out after {timeout_seconds} seconds")
    if process.returncode != 0:
        raise RuntimeError(f"Hermes exited with code {process.returncode}")
    return session_id


def _success_transition_seen(output: str) -> bool:
    normalized = output.replace("*", "").replace("`", "")
    if "RESULTS_COLLECTED" not in normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "Success transition",
            "completed successfully",
            "No forbidden actions",
            "Final state",
            "Final state_hint",
        )
    )


def _extract_session_id(output: str) -> str | None:
    for line in output.splitlines():
        if line.startswith("Session:"):
            return line.split(":", 1)[1].strip()
    return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1)
