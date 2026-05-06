from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_infrastructure_audit_script_parses_run_state_lists() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "audit_hermes_hypothesis_lab_infrastructure",
        repo_root / "scripts" / "audit_hermes_hypothesis_lab_infrastructure.py",
    )
    run_state = """
required_reads:
  - /repo/run_state.yaml
  - /repo/bootstrap_prompt.md
required_artifacts:
  - /tmp/hypothesis_lab_infrastructure_audit.json
allowed_commands:
  - name: audit_hypothesis_lab_infrastructure
    command: cd /repo && uv run --frozen pde-agent audit-hermes-hypothesis-lab-infrastructure --acceptance-smoke-path /repo/acceptance.json --preflight-chain-report-path /repo/chain.json --runbook-path /repo/docs/runbook.md --output-path /tmp/audit.json --min-completed-iterations 2 --force
success_transition:
  on_complete: HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED
"""

    assert module._yaml_list(run_state, "required_reads") == [
        "/repo/run_state.yaml",
        "/repo/bootstrap_prompt.md",
    ]
    assert module._yaml_list(run_state, "required_artifacts") == [
        "/tmp/hypothesis_lab_infrastructure_audit.json",
    ]
    assert module._yaml_command_list(run_state) == [
        "cd /repo && uv run --frozen pde-agent audit-hermes-hypothesis-lab-infrastructure --acceptance-smoke-path /repo/acceptance.json --preflight-chain-report-path /repo/chain.json --runbook-path /repo/docs/runbook.md --output-path /tmp/audit.json --min-completed-iterations 2 --force"
    ]


def test_run_script_renders_hypothesis_lab_infrastructure_prompt() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "run_hermes_hypothesis_lab_infrastructure",
        repo_root / "scripts" / "run_hermes_hypothesis_lab_infrastructure.py",
    )
    fixture_root = (
        repo_root / "integration_tests" / "hermes_hypothesis_lab_infrastructure"
    )

    prompt = module._render_prompt(fixture_root)

    assert str(fixture_root / "run_state.yaml") in prompt
    assert str(fixture_root / "bootstrap_prompt.md") in prompt
    assert str(fixture_root / "task_spec.json") in prompt
    assert "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED" in prompt


def test_run_script_detects_state_transition_achieved_output() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "run_hermes_hypothesis_lab_infrastructure",
        repo_root / "scripts" / "run_hermes_hypothesis_lab_infrastructure.py",
    )

    assert module._success_transition_seen(
        "Command executed successfully. State transition achieved: "
        "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED."
    )


def test_infrastructure_session_audit_accepts_exact_single_command(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_hypothesis_lab_infrastructure_fixture_audit import (
        audit_hermes_hypothesis_lab_infrastructure_session,
    )

    read_a = str(tmp_path / "run_state.yaml")
    read_b = str(tmp_path / "bootstrap_prompt.md")
    command = "cd /repo && uv run --frozen pde-agent audit-hermes-hypothesis-lab-infrastructure --force"
    artifact = tmp_path / "audit.json"
    artifact.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED",
                "decision": "passed",
                "summary": {"check_count": 13, "failed_check_count": 0},
            }
        ),
        encoding="utf-8",
    )
    session_path = tmp_path / "session_infra.json"
    session_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "run fixture"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": read_a}),
                                }
                            },
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": read_b}),
                                }
                            },
                            {
                                "function": {
                                    "name": "terminal",
                                    "arguments": json.dumps({"command": command}),
                                }
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "exit_code": 0,
                                "output": json.dumps(
                                    {
                                        "status": "ok",
                                        "state_hint": (
                                            "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED"
                                        ),
                                    }
                                ),
                            }
                        ),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_hermes_hypothesis_lab_infrastructure_session(
        session_path=session_path,
        required_artifacts=[str(artifact)],
        required_reads=[read_a, read_b],
        allowed_commands=[command],
    )

    assert report["status"] == "passed"
    assert report["infrastructure"]["failed_check_count"] == 0
    assert report["tool_call_summary"] == {"read_file": 2, "terminal": 1}
