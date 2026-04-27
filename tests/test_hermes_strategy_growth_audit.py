import json
from pathlib import Path

from pde_agent.core.hermes_strategy_growth_audit import (
    audit_hermes_strategy_growth_session,
)


REQUIRED_READS = [
    "/repo/integration_tests/hermes_fno_strategy_growth/run_state.yaml",
    "/repo/integration_tests/hermes_fno_strategy_growth/bootstrap_prompt.md",
    "/repo/integration_tests/hermes_fno_strategy_growth/task_spec.json",
]

COMMANDS = [
    "cd /repo && uv run --frozen pde-agent convert-fno-evidence-summary --sweep-summary-path /tmp/fno_evidence_summary.json --output-path /tmp/out/strategy_evidence_summary.json --long-window-threshold 31 --force",
    'cd /repo && uv run --frozen pde-agent record-strategy-proposal --strategy-id fno_scheduled_sampling_evidence_v0 --title "FNO scheduled sampling evidence gate" --source-type local_library --problem-context "1D Burgers FNO evidence sweep strategy-growth gate" --hypothesis "Scheduled sampling rollout fine-tuning should only be promoted when multi-window evidence is stable." --mechanism "Use FNO evidence sweep winners as quantitative evidence before skill promotion." --implementation-plan-json "{\\"model_family\\":\\"FNO\\"}" --expected-observable "positive mean score delta across multiple windows" --known-risk "single-window improvement may not generalize" --output-path /tmp/out/strategy_proposal.json',
    "cd /repo && uv run --frozen pde-agent build-strategy-rubric --proposal-path /tmp/out/strategy_proposal.json --output-path /tmp/out/experiment_rubric.json --min-evaluation-windows 3 --max-inference-time-regression 0.10",
    "cd /repo && uv run --frozen pde-agent evaluate-strategy-promotion --proposal-path /tmp/out/strategy_proposal.json --rubric-path /tmp/out/experiment_rubric.json --evidence-summary-path /tmp/out/strategy_evidence_summary.json --output-path /tmp/out/promotion_report.json",
    "cd /repo && uv run --frozen pde-agent append-strategy-ledger --proposal-path /tmp/out/strategy_proposal.json --rubric-path /tmp/out/experiment_rubric.json --evidence-summary-path /tmp/out/strategy_evidence_summary.json --promotion-report-path /tmp/out/promotion_report.json --ledger-path /tmp/out/strategy_ledger.jsonl",
]


def _tool_call(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _terminal_result(command: str, *, exit_code: int = 0) -> dict:
    command_name = (
        command.split(" pde-agent ", 1)[1].split(" ", 1)[0]
        if " pde-agent " in command
        else "manual-shell-command"
    )
    state_by_command = {
        "convert-fno-evidence-summary": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "record-strategy-proposal": "STRATEGY_PROPOSAL_RECORDED",
        "build-strategy-rubric": "STRATEGY_RUBRIC_BUILT",
        "evaluate-strategy-promotion": "STRATEGY_PROMOTION_EVALUATED",
        "append-strategy-ledger": "STRATEGY_LEDGER_APPENDED",
        "manual-shell-command": "MANUAL_SHELL_COMMAND",
    }
    payload = {
        "run_id": "local",
        "command": command_name,
        "status": "ok",
        "state_hint": state_by_command[command_name],
        "artifacts": {},
        "metrics": {},
        "errors": [],
        "next_hint": None,
    }
    return {
        "role": "tool",
        "content": json.dumps(
            {"output": json.dumps(payload), "exit_code": exit_code, "error": None}
        ),
    }


def _session(commands: list[str] | None = None) -> dict:
    command_sequence = commands or COMMANDS
    messages = [{"role": "user", "content": "strategy growth", "tool_calls": None}]
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
        messages.append(_terminal_result(command))
    return {"messages": messages}


def _write_required_artifacts(output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_evidence = output_dir / "strategy_evidence_summary.json"
    proposal = output_dir / "strategy_proposal.json"
    rubric = output_dir / "experiment_rubric.json"
    promotion = output_dir / "promotion_report.json"
    ledger = output_dir / "strategy_ledger.jsonl"
    strategy_evidence.write_text(
        json.dumps({"state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT"}),
        encoding="utf-8",
    )
    proposal.write_text(
        json.dumps({"state_hint": "STRATEGY_PROPOSAL_RECORDED"}),
        encoding="utf-8",
    )
    rubric.write_text(
        json.dumps({"state_hint": "STRATEGY_RUBRIC_BUILT"}),
        encoding="utf-8",
    )
    promotion.write_text(
        json.dumps({"state_hint": "STRATEGY_PROMOTION_EVALUATED"}),
        encoding="utf-8",
    )
    ledger.write_text(
        json.dumps(
            {
                "state_hint": "STRATEGY_LEDGER_APPENDED",
                "strategy_id": "fno_scheduled_sampling_evidence_v0",
                "decision": "promote",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return [str(strategy_evidence), str(proposal), str(rubric), str(promotion), str(ledger)]


def test_strategy_growth_audit_passes_for_exact_gate_sequence(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260427_strategy01.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out")

    report = audit_hermes_strategy_growth_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=COMMANDS,
    )

    assert report["status"] == "passed"
    assert report["terminal_commands"] == COMMANDS
    assert report["ledger"]["strategy_id"] == "fno_scheduled_sampling_evidence_v0"
    assert report["metrics"]["command_count"] == 5
    assert report["errors"] == []


def test_strategy_growth_audit_uses_supplied_arm_label(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260427_strategy_deeponet.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out")

    report = audit_hermes_strategy_growth_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=COMMANDS,
        arm="deeponet_strategy_growth",
    )

    assert report["status"] == "passed"
    assert report["arm"] == "deeponet_strategy_growth"


def test_strategy_growth_audit_fails_if_raw_fno_summary_is_promoted_directly(
    tmp_path: Path,
) -> None:
    bad_commands = COMMANDS.copy()
    bad_commands[3] = bad_commands[3].replace(
        "/tmp/out/strategy_evidence_summary.json",
        "/tmp/fno_evidence_summary.json",
    )
    session_path = tmp_path / "session_20260427_strategy02.json"
    session_path.write_text(json.dumps(_session(bad_commands)), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out")

    report = audit_hermes_strategy_growth_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=COMMANDS,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]
    assert "forbidden_actions" in report["errors"]


def test_strategy_growth_audit_fails_on_manual_artifact_inspection(
    tmp_path: Path,
) -> None:
    bad_commands = [
        *COMMANDS[:1],
        "cat /tmp/out/strategy_evidence_summary.json",
        *COMMANDS[1:],
    ]
    session_path = tmp_path / "session_20260427_strategy03.json"
    session_path.write_text(json.dumps(_session(bad_commands)), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out")

    report = audit_hermes_strategy_growth_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=COMMANDS,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]
    assert "forbidden_actions" in report["errors"]


def test_strategy_growth_audit_fails_on_missing_ledger(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260427_strategy04.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out")
    Path(artifacts[-1]).unlink()

    report = audit_hermes_strategy_growth_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=COMMANDS,
    )

    assert report["status"] == "failed"
    assert "ledger_artifact" in report["errors"]
