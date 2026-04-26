import json
from pathlib import Path

import pytest


def test_record_autonomy_decision_writes_valid_artifact(tmp_path: Path) -> None:
    from pde_agent.core.autonomy_decision import record_autonomy_decision

    evidence_path = tmp_path / "fno_iteration_report.json"
    output_path = tmp_path / "autonomy_decision.json"
    evidence_path.write_text('{"state_hint": "FNO_ITERATION_COMPLETED"}\n', encoding="utf-8")

    report = record_autonomy_decision(
        decision="request_evidence_sweep",
        reason_code="single_window_improvement_needs_validation",
        evidence_source_path=evidence_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert report == json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["state_hint"] == "HERMES_AUTONOMY_DECISION_RECORDED"
    assert report["decision"] == "request_evidence_sweep"
    assert report["reason_code"] == "single_window_improvement_needs_validation"
    assert report["evidence_source_path"] == str(evidence_path)
    assert report["autonomy_decision_path"] == str(output_path)
    assert report["next_command"] == "run-fno-evidence-sweep"


def test_record_autonomy_decision_rejects_unsupported_decision(
    tmp_path: Path,
) -> None:
    from pde_agent.core.autonomy_decision import (
        AutonomyDecisionError,
        record_autonomy_decision,
    )

    evidence_path = tmp_path / "fno_iteration_report.json"
    evidence_path.write_text("{}", encoding="utf-8")

    with pytest.raises(AutonomyDecisionError, match="unsupported decision"):
        record_autonomy_decision(
            decision="invent_new_training_loop",
            reason_code="single_window_improvement_needs_validation",
            evidence_source_path=evidence_path,
            output_path=tmp_path / "autonomy_decision.json",
        )


def test_record_autonomy_decision_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    evidence_path = tmp_path / "fno_iteration_report.json"
    output_path = tmp_path / "autonomy_decision.json"
    evidence_path.write_text('{"state_hint": "FNO_ITERATION_COMPLETED"}\n', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "record-autonomy-decision",
            "--decision",
            "accept_current_winner",
            "--reason-code",
            "clear_improvement_accept_for_now",
            "--evidence-source-path",
            str(evidence_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "record-autonomy-decision"
    assert payload["state_hint"] == "HERMES_AUTONOMY_DECISION_RECORDED"
    assert payload["artifacts"]["autonomy_decision_path"] == str(output_path)
    assert payload["artifacts"]["evidence_source_path"] == str(evidence_path)
    assert payload["metrics"]["decision"] == "accept_current_winner"
    assert payload["metrics"]["reason_code"] == "clear_improvement_accept_for_now"
    assert payload["metrics"]["next_command"] == "run-fno-iteration"
    assert payload["next_hint"] == "run-fno-iteration"
