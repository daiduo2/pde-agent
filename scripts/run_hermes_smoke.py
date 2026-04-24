#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pty
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
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        ["hermes", "-p", profile],
        cwd=fixture_root,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        close_fds=True,
    )
    os.close(slave_fd)
    session_id: str | None = None
    output = ""
    deadline = time.time() + timeout_seconds
    prompt_submitted = False
    exit_submitted = False

    try:
        while time.time() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.5)
            if readable:
                chunk = os.read(master_fd, 4096).decode("utf-8", errors="replace")
                output += chunk
                print(chunk, end="")
                session_id = session_id or _extract_session_id(output)
                if not prompt_submitted and "❯" in output:
                    os.write(master_fd, (prompt + "\r").encode("utf-8"))
                    prompt_submitted = True
                elif (
                    prompt_submitted
                    and not exit_submitted
                    and "Success transition reached: RESULTS_COLLECTED" in output
                    and "❯" in chunk
                ):
                    os.write(master_fd, b"/exit\r")
                    exit_submitted = True

            if process.poll() is not None:
                break

        if process.poll() is None:
            process.terminate()
            raise TimeoutError(f"Hermes smoke run timed out after {timeout_seconds} seconds")
    finally:
        os.close(master_fd)

    if process.returncode not in (0, None):
        raise RuntimeError(f"Hermes exited with code {process.returncode}")
    return session_id


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
