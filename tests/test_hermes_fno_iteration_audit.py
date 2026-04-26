import json
from pathlib import Path

from pde_agent.core.hermes_fno_iteration_audit import (
    audit_hermes_fno_iteration_session,
)


REQUIRED_READS = [
    "/repo/integration_tests/hermes_fno_iteration_smoke/run_state.yaml",
    "/repo/integration_tests/hermes_fno_iteration_smoke/bootstrap_prompt.md",
    "/repo/integration_tests/hermes_fno_iteration_smoke/task_spec.json",
]

ALLOWED_COMMANDS = [
    "cd /repo && uv run --frozen --extra torch pde-agent run-fno-iteration --windows-path /tmp/windows.npz --checkpoint-path /repo/data/pdebench/checkpoints/burgers_FNO.tar --output-dir /tmp/fno_iteration --max-train-steps 1 --learning-rate 0.0001 --force",
]


def _tool_call(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _terminal_result(
    *,
    command: str = ALLOWED_COMMANDS[0],
    state_hint: str = "FNO_ITERATION_COMPLETED",
    status: str = "ok",
    exit_code: int = 0,
) -> dict:
    payload = {
        "run_id": "local",
        "command": "run-fno-iteration",
        "status": status,
        "state_hint": state_hint,
        "artifacts": {
            "iteration_report_path": "/tmp/fno_iteration/fno_iteration_report.json",
            "comparison_report_path": "/tmp/fno_iteration/candidate_comparison.json",
        },
        "metrics": {
            "baseline_score": 40.0,
            "candidate_count": 2,
            "winner_experiment_id": "use_scheduled_sampling_rollout_finetune_001",
            "winner_score": 43.0,
            "winner_score_delta": 3.0,
        },
        "errors": [],
        "next_hint": "run-fno-iteration",
    }
    return {
        "role": "tool",
        "content": json.dumps(
            {"output": json.dumps(payload), "exit_code": exit_code, "error": None}
        ),
    }


def _session(commands: list[str] | None = None) -> dict:
    command_sequence = commands or ALLOWED_COMMANDS
    messages = [{"role": "user", "content": "fno iteration", "tool_calls": None}]
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
        messages.append(_terminal_result(command=command))
    return {"messages": messages}


def _write_iteration_report(path: Path, *, candidate_count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_ITERATION_COMPLETED",
                "candidate_count": candidate_count,
                "winner": {
                    "experiment_id": "use_scheduled_sampling_rollout_finetune_001",
                    "score": 43.0,
                    "score_delta": 3.0,
                },
                "candidates": [{}, {}],
                "artifacts": {
                    "comparison_report_path": str(path.parent / "candidate_comparison.json"),
                    "iteration_report_path": str(path),
                },
            }
        ),
        encoding="utf-8",
    )
    (path.parent / "candidate_comparison.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_CANDIDATES_COMPARED",
                "candidate_count": candidate_count,
                "winner": {
                    "experiment_id": "use_scheduled_sampling_rollout_finetune_001"
                },
            }
        ),
        encoding="utf-8",
    )


def test_fno_iteration_audit_passes_for_expected_session(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260427_fno01.json"
    iteration_report_path = tmp_path / "fno_iteration" / "fno_iteration_report.json"
    comparison_path = tmp_path / "fno_iteration" / "candidate_comparison.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    _write_iteration_report(iteration_report_path)

    report = audit_hermes_fno_iteration_session(
        session_path=session_path,
        required_artifacts=[str(iteration_report_path), str(comparison_path)],
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "passed"
    assert report["iteration"] == {
        "candidate_count": 2,
        "winner_experiment_id": "use_scheduled_sampling_rollout_finetune_001",
        "winner_score": 43.0,
        "winner_score_delta": 3.0,
    }
    assert report["errors"] == []


def test_fno_iteration_audit_fails_on_lower_level_fno_command(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session_20260427_fno01.json"
    iteration_report_path = tmp_path / "fno_iteration" / "fno_iteration_report.json"
    comparison_path = tmp_path / "fno_iteration" / "candidate_comparison.json"
    commands = [
        *ALLOWED_COMMANDS,
        "cd /repo && uv run --frozen --extra torch pde-agent evaluate-fno-candidate --candidate-report-path x --target-path y --output-dir z",
    ]
    session_path.write_text(json.dumps(_session(commands)), encoding="utf-8")
    _write_iteration_report(iteration_report_path)

    report = audit_hermes_fno_iteration_session(
        session_path=session_path,
        required_artifacts=[str(iteration_report_path), str(comparison_path)],
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]
    assert "forbidden_actions" in report["errors"]


def test_fno_iteration_audit_fails_on_missing_iteration_report(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session_20260427_fno01.json"
    iteration_report_path = tmp_path / "fno_iteration" / "fno_iteration_report.json"
    comparison_path = tmp_path / "fno_iteration" / "candidate_comparison.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")

    report = audit_hermes_fno_iteration_session(
        session_path=session_path,
        required_artifacts=[str(iteration_report_path), str(comparison_path)],
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "iteration_report" in report["errors"]
