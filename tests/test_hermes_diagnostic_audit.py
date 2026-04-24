import json
from pathlib import Path

from pde_agent.core.hermes_diagnostic_audit import audit_hermes_diagnostic_session


REQUIRED_READS = [
    "/repo/integration_tests/hermes_diagnostic_smoke/run_state.yaml",
    "/repo/integration_tests/hermes_diagnostic_smoke/bootstrap_prompt.md",
    "/repo/integration_tests/hermes_diagnostic_smoke/task_spec.json",
    "/repo/integration_tests/hermes_diagnostic_smoke/base_config.json",
    "/repo/integration_tests/hermes_diagnostic_smoke/override.json",
]

ALLOWED_COMMANDS = [
    "/repo/.venv/bin/pde-agent init-run --run-id diagnostic_smoke_001 --runs-root /repo/integration_tests/hermes_diagnostic_smoke/runs",
    "/repo/.venv/bin/pde-agent resolve-config --run-id diagnostic_smoke_001 --base-config-file /repo/integration_tests/hermes_diagnostic_smoke/base_config.json --override-file /repo/integration_tests/hermes_diagnostic_smoke/override.json --task-spec-file /repo/integration_tests/hermes_diagnostic_smoke/task_spec.json --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent eval-rollout --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent analyze-spectrum --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent check-physics --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent interpret-error --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent select-hypothesis --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent collect-artifacts --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001 --config-hash diagnostic-smoke-001",
]


def _tool_call(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _terminal_result(command: str, state_hint: str) -> dict:
    payload = {
        "run_id": "diagnostic_smoke_001",
        "command": command.split()[1],
        "status": "ok",
        "state_hint": state_hint,
        "artifacts": {},
        "metrics": {},
        "errors": [],
        "next_hint": None,
    }
    return {
        "role": "tool",
        "content": json.dumps(
            {"output": json.dumps(payload), "exit_code": 0, "error": None}
        ),
    }


def _session(
    commands: list[str] | None = None, final_hint: str = "HYPOTHESIS_SELECTED"
) -> dict:
    command_sequence = commands or ALLOWED_COMMANDS
    messages = [{"role": "user", "content": "diagnostic smoke", "tool_calls": None}]
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("read_file", {"path": path}) for path in REQUIRED_READS
            ],
        }
    )
    hints = [
        "RUN_INITIALIZED",
        "CONFIG_RESOLVED",
        "RESULTS_COLLECTED",
        "SPECTRUM_ANALYZED",
        "PHYSICS_CHECKED",
        "ERROR_INTERPRETED",
        "HYPOTHESIS_SELECTED",
        final_hint,
    ]
    for index, command in enumerate(command_sequence):
        hint = hints[index] if index < len(hints) else final_hint
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("terminal", {"command": command})],
            }
        )
        messages.append(_terminal_result(command, hint))
    return {"messages": messages}


def _write_manifest(path: Path, *, phase: str = "hypothesis_selected") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "run_id": "diagnostic_smoke_001",
                "phase": phase,
                "config": {},
                "artifacts": {
                    "rollout_metrics_path": "/repo/run/metrics/rollout.json",
                    "spectrum_metrics_path": "/repo/run/metrics/spectrum.json",
                    "physics_check_path": "/repo/run/metrics/physics_check.json",
                    "error_interpretation_path": "/repo/run/diagnostics/error_interpretation.json",
                    "selected_hypothesis_path": "/repo/run/diagnostics/selected_hypothesis.json",
                },
                "summary": {
                    "dominant_spectral_issue": "excessive_smoothing",
                    "physics_issue": "overdissipation",
                    "dominant_failure_mode": "excessive_smoothing",
                    "selected_hypothesis": "add_spectral_consistency_loss",
                },
                "completeness": {},
                "errors": [],
            }
        ),
        encoding="utf-8",
    )


def test_diagnostic_audit_passes_for_expected_session(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_130000_diag01.json"
    manifest_path = (
        tmp_path / "runs" / "diagnostic_smoke_001" / "artifacts" / "manifest.json"
    )
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "passed"
    assert report["diagnosis"] == {
        "dominant_failure_mode": "excessive_smoothing",
        "selected_hypothesis": "add_spectral_consistency_loss",
    }
    assert report["errors"] == []


def test_diagnostic_audit_fails_on_missing_diagnostic_command(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_130000_diag01.json"
    manifest_path = (
        tmp_path / "runs" / "diagnostic_smoke_001" / "artifacts" / "manifest.json"
    )
    commands = [command for command in ALLOWED_COMMANDS if "analyze-spectrum" not in command]
    session_path.write_text(json.dumps(_session(commands)), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]


def test_diagnostic_audit_fails_on_forbidden_training_command(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_130000_diag01.json"
    manifest_path = (
        tmp_path / "runs" / "diagnostic_smoke_001" / "artifacts" / "manifest.json"
    )
    commands = [
        *ALLOWED_COMMANDS,
        "/repo/.venv/bin/pde-agent train-baseline --run-id diagnostic_smoke_001 --run-root /repo/run",
    ]
    session_path.write_text(json.dumps(_session(commands)), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "forbidden_actions" in report["errors"]


def test_diagnostic_audit_fails_on_manifest_phase_mismatch(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_130000_diag01.json"
    manifest_path = (
        tmp_path / "runs" / "diagnostic_smoke_001" / "artifacts" / "manifest.json"
    )
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    _write_manifest(manifest_path, phase="results_collected")

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "manifest" in report["errors"]
