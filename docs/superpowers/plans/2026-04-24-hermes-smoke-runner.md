# Hermes Smoke Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repo-local Hermes smoke runner that drives a real Hermes CLI session by default, then deterministically audits the session log and generated artifacts.

**Architecture:** Keep provider-specific orchestration in `scripts/` and keep reusable deterministic audit logic in `src/pde_agent/core/hermes_smoke_audit.py`. The runner renders a prompt template, launches Hermes in a PTY-friendly subprocess, exits the session, then calls the same audit code used by the audit-only script.

**Tech Stack:** Python 3.12+, standard library only for the new runner/auditor (`argparse`, `json`, `pathlib`, `pty`, `select`, `subprocess`, `time`, `dataclasses`), existing `pytest` test suite.

---

## File Structure

- Create `src/pde_agent/core/hermes_smoke_audit.py`: deterministic audit functions and report data structures. No Hermes process launching here.
- Create `tests/test_hermes_smoke_audit.py`: unit tests for pass/fail audit cases using small synthetic session and manifest JSON files.
- Create `scripts/audit_hermes_smoke.py`: CLI wrapper around the audit module. Reads latest or specified session and writes report JSON.
- Create `scripts/run_hermes_smoke.py`: real Hermes runner. Renders prompt template, launches `hermes -p <profile>`, submits prompt, exits, and invokes audit.
- Create `integration_tests/hermes_protocol_smoke/prompt_template.md`: reusable prompt with explicit placeholders.
- Create `integration_tests/hermes_protocol_smoke/reports/.gitkeep`: stable reports directory, while generated reports remain ignored if covered by existing ignore rules.

## Task 1: Deterministic Audit Core

**Files:**
- Create: `src/pde_agent/core/hermes_smoke_audit.py`
- Test: `tests/test_hermes_smoke_audit.py`

- [ ] **Step 1: Write passing audit test**

Add this test file with helper builders and the first passing case:

```python
import json
from pathlib import Path

from pde_agent.core.hermes_smoke_audit import audit_hermes_smoke_session


REQUIRED_READS = [
    "/repo/integration_tests/hermes_protocol_smoke/run_state.yaml",
    "/repo/integration_tests/hermes_protocol_smoke/bootstrap_prompt.md",
    "/repo/integration_tests/hermes_protocol_smoke/task_spec.json",
    "/repo/integration_tests/hermes_protocol_smoke/base_config.json",
    "/repo/integration_tests/hermes_protocol_smoke/override.json",
]

ALLOWED_COMMANDS = [
    "/repo/.venv/bin/pde-agent init-run --run-id hermes_protocol_smoke_003 --runs-root /repo/integration_tests/hermes_protocol_smoke/runs",
    "/repo/.venv/bin/pde-agent resolve-config --run-id hermes_protocol_smoke_003 --base-config-file /repo/integration_tests/hermes_protocol_smoke/base_config.json --override-file /repo/integration_tests/hermes_protocol_smoke/override.json --task-spec-file /repo/integration_tests/hermes_protocol_smoke/task_spec.json --run-root /repo/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003",
    "/repo/.venv/bin/pde-agent eval-rollout --run-id hermes_protocol_smoke_003 --run-root /repo/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003",
    "/repo/.venv/bin/pde-agent record-timing --run-id hermes_protocol_smoke_003 --run-root /repo/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003",
    "/repo/.venv/bin/pde-agent collect-artifacts --run-id hermes_protocol_smoke_003 --run-root /repo/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003 --config-hash synthetic-smoke-003",
]


def _tool_call(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _terminal_result(command: str, *, status: str = "ok", state_hint: str = "RESULTS_COLLECTED", exit_code: int = 0) -> dict:
    payload = {
        "run_id": "hermes_protocol_smoke_003",
        "command": command.split()[1],
        "artifacts": {},
        "metrics": {},
        "errors": [],
        "next_hint": None,
        "status": status,
        "state_hint": state_hint,
    }
    return {"role": "tool", "content": json.dumps({"output": json.dumps(payload), "exit_code": exit_code, "error": None})}


def _session(command_sequence: list[str] | None = None) -> dict:
    commands = command_sequence or ALLOWED_COMMANDS
    messages = [{"role": "user", "content": "smoke", "tool_calls": None}]
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("read_file", {"path": path}) for path in REQUIRED_READS],
        }
    )
    for command in commands:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("terminal", {"command": command})],
            }
        )
        messages.append(_terminal_result(command))
    return {"messages": messages}


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "run_id": "hermes_protocol_smoke_003",
                "phase": "results_collected",
                "config": {},
                "artifacts": {
                    "rollout_metrics_path": "/repo/run/metrics/rollout.json",
                    "timing_metrics_path": "/repo/run/metrics/timing.json",
                    "prediction_hdf5_path": "/repo/run/artifacts/prediction.hdf5",
                    "error_curve_path": "/repo/run/plots/error_curve.png",
                    "time_csv_path": "/repo/run/submission/time.csv",
                },
                "summary": {},
                "completeness": {},
                "errors": [],
            }
        ),
        encoding="utf-8",
    )


def test_audit_passes_for_expected_session(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_121355_f1cb57.json"
    manifest_path = tmp_path / "runs" / "hermes_protocol_smoke_003" / "artifacts" / "manifest.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_smoke_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "passed"
    assert report["tool_call_summary"] == {"read_file": 5, "terminal": 5}
    assert report["terminal_commands"] == ALLOWED_COMMANDS
    assert report["errors"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hermes_smoke_audit.py::test_audit_passes_for_expected_session -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pde_agent.core.hermes_smoke_audit'`.

- [ ] **Step 3: Implement minimal audit module**

Create `src/pde_agent/core/hermes_smoke_audit.py`:

```python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def audit_hermes_smoke_session(
    *,
    session_path: Path,
    manifest_path: Path,
    required_reads: list[str],
    allowed_commands: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    if not session_path.exists():
        return _error_report(session_path, manifest_path, f"session file not found: {session_path}")

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
    _add_check(
        checks,
        errors,
        "forbidden_actions",
        not _forbidden_calls(tool_calls),
        {"offenders": _forbidden_calls(tool_calls)},
    )
    _add_check(
        checks,
        errors,
        "terminal_exit_codes",
        _terminal_exit_codes_ok(messages),
        {"terminal_result_count": len(_terminal_results(messages))},
    )
    _add_check(
        checks,
        errors,
        "final_state",
        _final_terminal_payload(messages).get("status") == "ok"
        and _final_terminal_payload(messages).get("state_hint") == "RESULTS_COLLECTED",
        {"final_payload": _final_terminal_payload(messages)},
    )
    _add_check(
        checks,
        errors,
        "manifest",
        _manifest_ok(manifest_path),
        {"manifest_path": str(manifest_path)},
    )

    return {
        "status": "passed" if not errors else "failed",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "manifest_path": str(manifest_path),
        "checks": checks,
        "tool_call_summary": tool_summary,
        "terminal_commands": terminal_commands,
        "errors": errors,
    }


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
            results.append(json.loads(message.get("content") or "{}"))
        except json.JSONDecodeError:
            results.append({"parse_error": message.get("content")})
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


def _manifest_ok(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    required_artifacts = [
        "rollout_metrics_path",
        "timing_metrics_path",
        "prediction_hdf5_path",
        "error_curve_path",
        "time_csv_path",
    ]
    return (
        manifest.get("status") == "ok"
        and manifest.get("phase") == "results_collected"
        and manifest.get("errors") == []
        and all(artifacts.get(key) for key in required_artifacts)
    )


def _forbidden_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    forbidden_terms = [
        " train-baseline",
        " analyze-spectrum",
        " check-physics",
        " build-submission",
        " --help",
    ]
    for call in tool_calls:
        name = call["name"]
        command = call["arguments"].get("command", "")
        if name not in {"read_file", "terminal"}:
            offenders.append(call)
        elif any(term in command for term in forbidden_terms):
            offenders.append(call)
    return offenders


def _add_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    passed: bool,
    details: dict[str, Any],
) -> None:
    checks.append({"name": name, "status": "passed" if passed else "failed", "details": details})
    if not passed:
        errors.append(name)


def _error_report(session_path: Path, manifest_path: Path, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "manifest_path": str(manifest_path),
        "checks": [],
        "tool_call_summary": {},
        "terminal_commands": [],
        "errors": [message],
    }


def _session_id_from_path(path: Path) -> str:
    name = path.stem
    return name.removeprefix("session_")
```

- [ ] **Step 4: Run passing audit test**

Run: `uv run pytest tests/test_hermes_smoke_audit.py::test_audit_passes_for_expected_session -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/pde_agent/core/hermes_smoke_audit.py tests/test_hermes_smoke_audit.py
git commit -m "feat: add hermes smoke audit core"
```

## Task 2: Audit Failure Cases

**Files:**
- Modify: `tests/test_hermes_smoke_audit.py`
- Modify: `src/pde_agent/core/hermes_smoke_audit.py`

- [ ] **Step 1: Add missing required read test**

Append:

```python
def test_audit_fails_when_required_read_is_missing(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_121355_f1cb57.json"
    manifest_path = tmp_path / "runs" / "hermes_protocol_smoke_003" / "artifacts" / "manifest.json"
    session = _session()
    session["messages"][1]["tool_calls"] = session["messages"][1]["tool_calls"][:-1]
    session_path.write_text(json.dumps(session), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_smoke_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "required_reads" in report["errors"]
```

- [ ] **Step 2: Add forbidden command test**

Append:

```python
def test_audit_fails_on_forbidden_command(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_121355_f1cb57.json"
    manifest_path = tmp_path / "runs" / "hermes_protocol_smoke_003" / "artifacts" / "manifest.json"
    commands = [*ALLOWED_COMMANDS, "/repo/.venv/bin/pde-agent train-baseline --run-id x --run-root /tmp/x"]
    session_path.write_text(json.dumps(_session(commands)), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_smoke_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]
    assert "forbidden_actions" in report["errors"]
```

- [ ] **Step 3: Add command sequence mismatch test**

Append:

```python
def test_audit_fails_on_command_order_mismatch(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_121355_f1cb57.json"
    manifest_path = tmp_path / "runs" / "hermes_protocol_smoke_003" / "artifacts" / "manifest.json"
    commands = [ALLOWED_COMMANDS[1], ALLOWED_COMMANDS[0], *ALLOWED_COMMANDS[2:]]
    session_path.write_text(json.dumps(_session(commands)), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_smoke_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]
```

- [ ] **Step 4: Add missing manifest test**

Append:

```python
def test_audit_fails_when_manifest_is_missing(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_121355_f1cb57.json"
    manifest_path = tmp_path / "runs" / "hermes_protocol_smoke_003" / "artifacts" / "manifest.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")

    report = audit_hermes_smoke_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "manifest" in report["errors"]
```

- [ ] **Step 5: Add malformed terminal output test**

Append:

```python
def test_audit_fails_on_malformed_terminal_output_json(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_121355_f1cb57.json"
    manifest_path = tmp_path / "runs" / "hermes_protocol_smoke_003" / "artifacts" / "manifest.json"
    session = _session()
    session["messages"][-1] = {"role": "tool", "content": json.dumps({"output": "{not-json", "exit_code": 0, "error": None})}
    session_path.write_text(json.dumps(session), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_smoke_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "final_state" in report["errors"]
```

- [ ] **Step 6: Run failure-case tests**

Run: `uv run pytest tests/test_hermes_smoke_audit.py -v`

Expected: all tests PASS. If any failure-case test unexpectedly passes before implementation hardening, update `hermes_smoke_audit.py` so failed checks append their check name to `errors`.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add src/pde_agent/core/hermes_smoke_audit.py tests/test_hermes_smoke_audit.py
git commit -m "test: cover hermes smoke audit failures"
```

## Task 3: Audit CLI and Prompt Template

**Files:**
- Create: `scripts/audit_hermes_smoke.py`
- Create: `integration_tests/hermes_protocol_smoke/prompt_template.md`
- Create: `integration_tests/hermes_protocol_smoke/reports/.gitkeep`
- Modify: `integration_tests/hermes_protocol_smoke/README.md`

- [ ] **Step 1: Add prompt template**

Create `integration_tests/hermes_protocol_smoke/prompt_template.md`:

```markdown
This is a protocol-compliance smoke test for pde-agent.

Before taking any action, read these files in order using the exact absolute paths below:
1. {run_state_path}
2. {bootstrap_prompt_path}
3. {task_spec_path}
4. {base_config_path}
5. {override_path}

Hard rules:
- Treat {run_state_path} as the source of truth.
- Do not rely on the current working directory.
- Do not search for alternate file locations.
- Do not guess missing paths.
- Do not skip required reads.
- Do not patch code.
- Do not run training.
- Do not inspect command help.
- Do not probe command surfaces.
- Do not call commands outside the allowed set for the current state.
- Use only the exact command strings declared in run_state.yaml.
- Read command outputs as JSON only.
- Do not stop after resolve_config or any intermediate command output.
- Continue until the allowed success transition RESULTS_COLLECTED is reached.
- If required information is missing, stop and report the missing file or field instead of inferring it.

After reading the files, briefly state:
- current_state
- allowed next actions
- forbidden actions

Then continue by following the allowed action sequence only.
```

- [ ] **Step 2: Add audit CLI script**

Create `scripts/audit_hermes_smoke.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pde_agent.core.hermes_smoke_audit import audit_hermes_smoke_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a saved Hermes PDE smoke-test session.")
    parser.add_argument("--fixture-root", default="integration_tests/hermes_protocol_smoke")
    parser.add_argument("--session-id")
    parser.add_argument("--session-path")
    parser.add_argument("--report-path")
    args = parser.parse_args()

    repo_root = Path.cwd()
    fixture_root = (repo_root / args.fixture_root).resolve()
    session_path = _resolve_session_path(fixture_root, args.session_id, args.session_path)
    run_state = _load_run_state_text(fixture_root / "run_state.yaml")
    required_reads = _yaml_list(run_state, "required_reads")
    allowed_commands = _yaml_command_list(run_state)
    required_artifacts = _yaml_list(run_state, "required_artifacts")
    manifest_path = Path(required_artifacts[0])

    report = audit_hermes_smoke_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=required_reads,
        allowed_commands=allowed_commands,
    )

    report_path = Path(args.report_path) if args.report_path else _default_report_path(fixture_root, report["session_id"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report_path": str(report_path)}, indent=2))
    return 0 if report["status"] == "passed" else 1


def _resolve_session_path(fixture_root: Path, session_id: str | None, session_path: str | None) -> Path:
    if session_path:
        return Path(session_path).resolve()
    sessions_dir = fixture_root / "hermes_home" / "sessions"
    if session_id:
        return sessions_dir / f"session_{session_id}.json"
    sessions = sorted(sessions_dir.glob("session_*.json"), key=lambda path: path.stat().st_mtime)
    if not sessions:
        return sessions_dir / "session_missing.json"
    return sessions[-1]


def _load_run_state_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def _default_report_path(fixture_root: Path, session_id: str) -> Path:
    return fixture_root / "reports" / f"audit_{session_id}.json"


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Add reports directory marker**

Run: `mkdir -p integration_tests/hermes_protocol_smoke/reports`

Create `integration_tests/hermes_protocol_smoke/reports/.gitkeep` as an empty file.

- [ ] **Step 4: Update README manual audit docs**

Append to `integration_tests/hermes_protocol_smoke/README.md`:

```markdown

## Audit saved Hermes session

After a Hermes run, audit the latest saved session:

```bash
uv run python scripts/audit_hermes_smoke.py
```

Audit a specific session:

```bash
uv run python scripts/audit_hermes_smoke.py --session-id 20260424_121355_f1cb57
```

The audit writes a JSON report under `reports/` and exits non-zero if protocol compliance fails.
```

- [ ] **Step 5: Run audit CLI against existing session**

Run:

`uv run python scripts/audit_hermes_smoke.py --session-id 20260424_121355_f1cb57`

Expected: prints JSON with `"status": "passed"` and a `report_path` under `integration_tests/hermes_protocol_smoke/reports/`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add scripts/audit_hermes_smoke.py integration_tests/hermes_protocol_smoke/prompt_template.md integration_tests/hermes_protocol_smoke/reports/.gitkeep integration_tests/hermes_protocol_smoke/README.md
git commit -m "feat: add hermes smoke audit cli"
```

## Task 4: Real Hermes Runner

**Files:**
- Create: `scripts/run_hermes_smoke.py`
- Modify: `integration_tests/hermes_protocol_smoke/README.md`

- [ ] **Step 1: Add real Hermes runner script**

Create `scripts/run_hermes_smoke.py`:

```python
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


PROMPT_SENTINEL = "\n❯"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real Hermes protocol smoke test and audit it.")
    parser.add_argument("--fixture-root", default="integration_tests/hermes_protocol_smoke")
    parser.add_argument("--profile", default="hermes-protocol-smoke")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--report-path")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--session-id")
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
        str(fixture_root.relative_to(repo_root)),
    ]
    if session_id:
        audit_cmd.extend(["--session-id", session_id])
    if args.report_path:
        audit_cmd.extend(["--report-path", args.report_path])

    result = subprocess.run(audit_cmd, cwd=repo_root, check=False, text=True, capture_output=True)
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


def _run_hermes(*, profile: str, fixture_root: Path, prompt: str, timeout_seconds: int) -> str | None:
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
                elif prompt_submitted and not exit_submitted and "Success transition reached: RESULTS_COLLECTED" in output and "❯" in chunk:
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
```

- [ ] **Step 2: Update README runner docs**

Append to `integration_tests/hermes_protocol_smoke/README.md`:

```markdown

## Run real Hermes smoke test

Run Hermes with the default `hermes-protocol-smoke` profile and audit the resulting session:

```bash
uv run python scripts/run_hermes_smoke.py --timeout-seconds 300
```

Use audit-only mode through the same entry point:

```bash
uv run python scripts/run_hermes_smoke.py --audit-only --session-id 20260424_121355_f1cb57
```

The default path calls the real Hermes CLI and may consume provider quota.
```

- [ ] **Step 3: Run audit-only through runner**

Run:

`uv run python scripts/run_hermes_smoke.py --audit-only --session-id 20260424_121355_f1cb57`

Expected: prints JSON with `"status": "passed"`.

- [ ] **Step 4: Run real Hermes smoke**

Run:

`uv run python scripts/run_hermes_smoke.py --timeout-seconds 300`

Expected: Hermes launches, executes the five-command smoke sequence, exits, and the final audit JSON prints `"status": "passed"`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add scripts/run_hermes_smoke.py integration_tests/hermes_protocol_smoke/README.md
git commit -m "feat: add real hermes smoke runner"
```

## Task 5: Final Verification

**Files:**
- Review all changed files.

- [ ] **Step 1: Run full deterministic test suite**

Run: `uv run pytest tests -v`

Expected: all tests PASS.

- [ ] **Step 2: Run audit-only verification**

Run: `uv run python scripts/audit_hermes_smoke.py --session-id 20260424_121355_f1cb57`

Expected: exits 0 and prints `"status": "passed"`.

- [ ] **Step 3: Check git state**

Run: `git status --short`

Expected: only expected generated report files may appear. If report JSON is not ignored, either add an ignore rule for generated reports or remove generated reports before final commit.

- [ ] **Step 4: Summarize implementation**

Report:

- deterministic tests run and result
- audit-only command result
- real Hermes runner command result if executed
- any residual caveat about PTY behavior or local credentials

