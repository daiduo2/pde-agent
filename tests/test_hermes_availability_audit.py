from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_availability_audit_classifies_catalog_and_canary_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_availability_audit import audit_hermes_availability

    tool_catalog_path = tmp_path / "tool_catalog.json"
    _write_json(
        tool_catalog_path,
        {
            "tools": [
                {
                    "name": "build-hypothesis-algorithm-design",
                    "command": "build-hypothesis-algorithm-design",
                    "risk_level": "artifact_write",
                    "state_hints": ["HYPOTHESIS_ALGORITHM_DESIGN_BUILT"],
                },
                {
                    "name": "build-fno-training-harness-contract",
                    "command": "build-fno-training-harness-contract",
                    "risk_level": "artifact_write",
                    "state_hints": ["FNO_TRAINING_HARNESS_CONTRACT_BUILT"],
                },
            ],
        },
    )
    canary_path = tmp_path / "audit.json"
    _write_json(
        canary_path,
        {
            "status": "passed",
            "session_id": "20260504_121519_9bc312",
            "terminal_commands": ["uv run --frozen pde-agent audit"],
            "tool_call_summary": {"read_file": 3, "terminal": 1},
        },
    )

    report = audit_hermes_availability(
        output_path=tmp_path / "availability.json",
        tool_catalog_path=tool_catalog_path,
        canary_report_paths=[canary_path],
        cli_commands=[
            "build-hypothesis-algorithm-design",
            "build-fno-training-harness-contract",
            "run-fno-training-optimization-loop",
        ],
        required_commands=[
            "build-hypothesis-algorithm-design",
            "run-fno-training-optimization-loop",
        ],
        force=False,
    )

    assert report["state_hint"] == "AGENT_RUNTIME_AVAILABILITY_AUDITED"
    assert report["runtime"]["name"] == "hermes"
    assert report["runtime"]["primary"] is False
    entries = {entry["command"]: entry for entry in report["matrix"]}

    assert entries["build-hypothesis-algorithm-design"]["readiness"] == (
        "hermes_discoverable"
    )
    assert entries["build-hypothesis-algorithm-design"]["cli_exposed"] is True
    assert entries["build-hypothesis-algorithm-design"]["tool_catalog_exposed"] is True
    assert entries["build-hypothesis-algorithm-design"]["risk_level"] == (
        "artifact_write"
    )

    assert entries["run-fno-training-optimization-loop"]["readiness"] == "cli_only"
    assert entries["run-fno-training-optimization-loop"]["tool_catalog_exposed"] is False
    assert "not_in_tool_catalog" in entries["run-fno-training-optimization-loop"][
        "gaps"
    ]

    assert report["metrics"]["required_command_count"] == 2
    assert report["metrics"]["required_ready_count"] == 1
    assert report["metrics"]["passed_canary_count"] == 1
    assert report["decision"] == "needs_work"


def test_availability_audit_marks_missing_required_command(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_availability_audit import audit_hermes_availability

    tool_catalog_path = tmp_path / "tool_catalog.json"
    _write_json(tool_catalog_path, {"tools": []})

    report = audit_hermes_availability(
        output_path=tmp_path / "availability.json",
        tool_catalog_path=tool_catalog_path,
        canary_report_paths=[],
        cli_commands=["run-hermes-heartbeat"],
        required_commands=["missing-command"],
        force=False,
    )

    entries = {entry["command"]: entry for entry in report["matrix"]}
    assert entries["missing-command"]["readiness"] == "missing"
    assert entries["missing-command"]["cli_exposed"] is False
    assert "not_cli_exposed" in entries["missing-command"]["gaps"]
    assert report["decision"] == "needs_work"


def test_availability_audit_command_returns_json_contract(tmp_path: Path) -> None:
    from pde_agent.commands.hypothesis_lab_runtime import (
        audit_hermes_availability_command,
    )
    from pde_agent.commands.hypothesis_lab_runtime import (
        audit_agent_runtime_availability_command,
    )

    tool_catalog_path = tmp_path / "tool_catalog.json"
    _write_json(
        tool_catalog_path,
        {
            "tools": [
                {
                    "name": "run-hermes-heartbeat",
                    "command": "run-hermes-heartbeat",
                    "risk_level": "long_running",
                    "state_hints": ["HERMES_HEARTBEAT_RAN"],
                }
            ]
        },
    )
    canary_path = tmp_path / "canary.json"
    _write_json(canary_path, {"status": "passed", "terminal_commands": []})

    payload = audit_hermes_availability_command(
        output_path=str(tmp_path / "availability.json"),
        tool_catalog_path=str(tool_catalog_path),
        canary_report_paths=[str(canary_path)],
        cli_commands=["run-hermes-heartbeat"],
        required_commands=["run-hermes-heartbeat"],
        force=False,
    )

    assert payload["command"] == "audit-hermes-availability"
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "AGENT_RUNTIME_AVAILABILITY_AUDITED"
    assert payload["metrics"]["required_ready_count"] == 1
    assert payload["metrics"]["passed_canary_count"] == 1
    assert payload["artifacts"]["missing_required_commands"] == []

    runtime_payload = audit_agent_runtime_availability_command(
        output_path=str(tmp_path / "claude_availability.json"),
        runtime_name="claude_agent_sdk",
        tool_catalog_path=str(tool_catalog_path),
        canary_report_paths=[str(canary_path)],
        cli_commands=["run-hermes-heartbeat"],
        required_commands=["run-hermes-heartbeat"],
        force=False,
    )
    assert runtime_payload["command"] == "audit-agent-runtime-availability"
    assert runtime_payload["state_hint"] == "AGENT_RUNTIME_AVAILABILITY_AUDITED"
    assert runtime_payload["metrics"]["runtime_primary"] is True


def test_availability_audit_treats_completed_claude_session_as_passed(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_availability_audit import (
        audit_agent_runtime_availability,
    )

    tool_catalog_path = tmp_path / "tool_catalog.json"
    _write_json(
        tool_catalog_path,
        {
            "tools": [
                {
                    "name": "run-claude-agent-sdk-runtime",
                    "command": "run-claude-agent-sdk-runtime",
                    "risk_level": "sandbox",
                    "state_hints": ["CLAUDE_AGENT_SDK_RUNTIME_RAN"],
                }
            ]
        },
    )
    canary_path = tmp_path / "claude_session.json"
    _write_json(
        canary_path,
        {
            "status": "completed",
            "state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN",
            "sdk_message_count": 3,
        },
    )

    report = audit_agent_runtime_availability(
        runtime_name="claude_agent_sdk",
        output_path=tmp_path / "availability.json",
        tool_catalog_path=tool_catalog_path,
        canary_report_paths=[canary_path],
        cli_commands=["run-claude-agent-sdk-runtime"],
        required_commands=["run-claude-agent-sdk-runtime"],
        force=False,
    )

    assert report["decision"] == "passed"
    assert report["metrics"]["passed_canary_count"] == 1
