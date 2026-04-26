import json
from pathlib import Path

from pde_agent.core.hermes_autonomy_audit import audit_hermes_autonomy_session


REQUIRED_READS = [
    "/repo/integration_tests/hermes_autonomy_constrained_free/run_state.yaml",
    "/repo/integration_tests/hermes_autonomy_constrained_free/bootstrap_prompt.md",
    "/repo/integration_tests/hermes_autonomy_constrained_free/task_spec.json",
]

RUN_COMMAND = (
    "cd /repo && uv run --frozen --extra torch pde-agent run-fno-iteration "
    "--windows-path /tmp/windows.npz "
    "--checkpoint-path /repo/data/pdebench/checkpoints/burgers_FNO.tar "
    "--output-dir /tmp/fno_iteration "
    "--max-train-steps 1 --learning-rate 0.0001 --force"
)
DECISION_REQUEST_COMMAND = (
    "cd /repo && uv run --frozen pde-agent record-autonomy-decision "
    "--decision request_evidence_sweep "
    "--reason-code single_window_improvement_needs_validation "
    "--evidence-source-path /tmp/fno_iteration/fno_iteration_report.json "
    "--output-path /tmp/fno_iteration/autonomy_decision.json"
)
DECISION_ACCEPT_COMMAND = (
    "cd /repo && uv run --frozen pde-agent record-autonomy-decision "
    "--decision accept_current_winner "
    "--reason-code clear_improvement_accept_for_now "
    "--evidence-source-path /tmp/fno_iteration/fno_iteration_report.json "
    "--output-path /tmp/fno_iteration/autonomy_decision.json"
)
ALLOWED_COMMANDS = [RUN_COMMAND, DECISION_REQUEST_COMMAND, DECISION_ACCEPT_COMMAND]


def _tool_call(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _terminal_result(
    *,
    command: str,
    state_hint: str,
    status: str = "ok",
    exit_code: int = 0,
) -> dict:
    payload = {
        "run_id": "local",
        "command": (
            "run-fno-iteration"
            if "run-fno-iteration" in command
            else "record-autonomy-decision"
        ),
        "status": status,
        "state_hint": state_hint,
        "artifacts": {},
        "metrics": {},
        "errors": [],
        "next_hint": "run-fno-evidence-sweep",
    }
    return {
        "role": "tool",
        "content": json.dumps(
            {"output": json.dumps(payload), "exit_code": exit_code, "error": None}
        ),
    }


def _session(commands: list[str] | None = None) -> dict:
    command_sequence = commands or [RUN_COMMAND, DECISION_REQUEST_COMMAND]
    messages = [{"role": "user", "content": "autonomy ablation", "tool_calls": None}]
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("read_file", {"path": path}) for path in REQUIRED_READS
            ],
        }
    )
    for command in command_sequence:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("terminal", {"command": command})],
            }
        )
        state_hint = (
            "FNO_ITERATION_COMPLETED"
            if "run-fno-iteration" in command
            else "HERMES_AUTONOMY_DECISION_RECORDED"
        )
        messages.append(_terminal_result(command=command, state_hint=state_hint))
    return {"messages": messages}


def _write_required_artifacts(output_dir: Path, *, decision: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    iteration_path = output_dir / "fno_iteration_report.json"
    comparison_path = output_dir / "candidate_comparison.json"
    decision_path = output_dir / "autonomy_decision.json"
    iteration_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_ITERATION_COMPLETED",
                "candidate_count": 2,
                "winner": {
                    "experiment_id": "use_scheduled_sampling_rollout_finetune_001",
                    "score": 43.0,
                    "score_delta": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps({"state_hint": "FNO_CANDIDATES_COMPARED"}),
        encoding="utf-8",
    )
    reason_code = (
        "single_window_improvement_needs_validation"
        if decision == "request_evidence_sweep"
        else "clear_improvement_accept_for_now"
    )
    next_command = (
        "run-fno-evidence-sweep"
        if decision == "request_evidence_sweep"
        else "run-fno-iteration"
    )
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_AUTONOMY_DECISION_RECORDED",
                "decision": decision,
                "reason_code": reason_code,
                "evidence_source_path": str(iteration_path),
                "autonomy_decision_path": str(decision_path),
                "next_command": next_command,
            }
        ),
        encoding="utf-8",
    )
    return [str(iteration_path), str(comparison_path), str(decision_path)]


def test_autonomy_audit_passes_for_one_legal_decision(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260427_auto01.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    artifacts = _write_required_artifacts(
        tmp_path / "fno_iteration",
        decision="request_evidence_sweep",
    )

    report = audit_hermes_autonomy_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "passed"
    assert report["arm"] == "constrained_free"
    assert report["metrics"]["command_count"] == 2
    assert report["metrics"]["decision_quality_score"] == 4
    assert report["decision"]["decision"] == "request_evidence_sweep"
    assert report["errors"] == []


def test_autonomy_audit_accepts_alternate_legal_decision(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260427_auto02.json"
    session_path.write_text(
        json.dumps(_session([RUN_COMMAND, DECISION_ACCEPT_COMMAND])),
        encoding="utf-8",
    )
    artifacts = _write_required_artifacts(
        tmp_path / "fno_iteration",
        decision="accept_current_winner",
    )

    report = audit_hermes_autonomy_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "passed"
    assert report["decision"]["decision"] == "accept_current_winner"


def test_autonomy_audit_fails_on_extra_manual_shell_inspection(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session_20260427_auto03.json"
    commands = [
        RUN_COMMAND,
        "cat /tmp/fno_iteration/fno_iteration_report.json",
        DECISION_REQUEST_COMMAND,
    ]
    session_path.write_text(json.dumps(_session(commands)), encoding="utf-8")
    artifacts = _write_required_artifacts(
        tmp_path / "fno_iteration",
        decision="request_evidence_sweep",
    )

    report = audit_hermes_autonomy_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]
    assert "forbidden_actions" in report["errors"]


def test_autonomy_audit_fails_on_missing_decision_artifact(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260427_auto04.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    artifacts = _write_required_artifacts(
        tmp_path / "fno_iteration",
        decision="request_evidence_sweep",
    )
    Path(artifacts[-1]).unlink()

    report = audit_hermes_autonomy_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "decision_artifact" in report["errors"]
