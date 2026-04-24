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


def _terminal_result(
    command: str,
    *,
    status: str = "ok",
    state_hint: str = "RESULTS_COLLECTED",
    exit_code: int = 0,
) -> dict:
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
    return {
        "role": "tool",
        "content": json.dumps(
            {"output": json.dumps(payload), "exit_code": exit_code, "error": None}
        ),
    }


def _session(command_sequence: list[str] | None = None) -> dict:
    commands = command_sequence or ALLOWED_COMMANDS
    messages = [{"role": "user", "content": "smoke", "tool_calls": None}]
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("read_file", {"path": path}) for path in REQUIRED_READS
            ],
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
    manifest_path = (
        tmp_path / "runs" / "hermes_protocol_smoke_003" / "artifacts" / "manifest.json"
    )
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
