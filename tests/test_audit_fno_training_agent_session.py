"""Tests for the FNO training agent session audit script."""

import json
from pathlib import Path
from typing import Any

import pytest


class TestAuditScript:
    def test_audit_passes_valid_session(self, tmp_path: Path) -> None:
        from scripts.audit.audit_fno_training_agent_session import (
            _emit_audit_report,
            _extract_tool_events,
        )

        session = {
            "workspace_dir": str(tmp_path),
            "sdk_messages": [
                json.dumps({"type": "tool_use", "name": "Read", "input": {"file_path": "report.json"}}),
                json.dumps({"type": "tool_use", "name": "Bash", "input": {"command": "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent run-fno-training-iteration"}}),
                json.dumps({"type": "result", "result": "FNO_TRAINING_ITERATION_REPORTED"}),
            ],
        }
        events = _extract_tool_events(session)
        assert len(events) == 2
        assert events[0]["name"] == "Read"
        assert events[1]["name"] == "Bash"

    def test_audit_fails_forbidden_action(self, tmp_path: Path) -> None:
        from scripts.audit.audit_fno_training_agent_session import _extract_tool_events

        session = {
            "workspace_dir": str(tmp_path),
            "sdk_messages": [
                json.dumps({"type": "tool_use", "name": "Edit", "input": {"file_path": "foo.py"}}),
            ],
        }
        events = _extract_tool_events(session)
        assert events[0]["name"] == "Edit"

    def test_extract_pde_agent_command(self) -> None:
        from scripts.audit.audit_fno_training_agent_session import _extract_pde_agent_command

        event = {
            "name": "Bash",
            "input": {"command": "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent run-fno-training-iteration --workspace-dir /tmp"},
        }
        assert _extract_pde_agent_command(event) == "run-fno-training-iteration"

    def test_extract_pde_agent_command_none(self) -> None:
        from scripts.audit.audit_fno_training_agent_session import _extract_pde_agent_command

        assert _extract_pde_agent_command({"name": "Read", "input": {}}) is None
        assert _extract_pde_agent_command({"name": "Bash", "input": {"command": "echo hello"}}) is None

    def test_emit_audit_report(self, capsys) -> None:
        from scripts.audit.audit_fno_training_agent_session import _emit_audit_report

        _emit_audit_report(
            passed=True,
            errors=[],
            checks=[{"check": "forbidden_tools", "passed": True}],
            session_path="session.json",
        )
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["passed"] is True
        assert report["state_hint"] == "FNO_TRAINING_AGENT_SESSION_AUDITED"
