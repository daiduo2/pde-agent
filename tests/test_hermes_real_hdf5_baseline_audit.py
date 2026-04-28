from __future__ import annotations

import json
from pathlib import Path

from pde_agent.core.hermes_real_hdf5_baseline_audit import (
    audit_hermes_real_hdf5_baseline_session,
)


REQUIRED_READS = [
    "/repo/integration_tests/hermes_real_hdf5_baseline_smoke/run_state.yaml",
    "/repo/integration_tests/hermes_real_hdf5_baseline_smoke/bootstrap_prompt.md",
    "/repo/integration_tests/hermes_real_hdf5_baseline_smoke/task_spec.json",
]

ALLOWED_COMMAND = (
    "cd /repo && uv run --frozen pde-agent run-real-hdf5-baseline-smoke "
    "--source-path /repo/data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 "
    "--output-dir /tmp/real_hdf5_baseline_smoke --dataset-key tensor "
    "--start-index 0 --sample-limit 2 --context-steps 10 --forecast-steps 31 "
    "--time-reduction 5 --space-reduction 4 --nu 0.001 --dt 1 --dx 1 --force"
)


def _tool_call(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _terminal_result(command: str = ALLOWED_COMMAND, exit_code: int = 0) -> dict:
    payload = {
        "run_id": "local",
        "command": "run-real-hdf5-baseline-smoke",
        "status": "ok",
        "state_hint": "REAL_HDF5_BASELINE_SMOKE_COMPLETED",
        "artifacts": {
            "summary_path": "/tmp/real_hdf5_baseline_smoke/real_hdf5_baseline_smoke_summary.json",
        },
        "metrics": {
            "sample_count": 2,
            "forecast_steps": 31,
            "competition_like_score_short": 39.1,
            "rel_mse": 0.06,
            "residual_rmse": 0.02,
            "dominant_error_pattern": "overdamped_rollout",
            "physics_issue": "nonphysical_oscillation",
            "residual_issue": "residual_mismatch",
        },
        "errors": [],
        "next_hint": "inspect-real-baseline-smoke-summary",
    }
    return {
        "role": "tool",
        "content": json.dumps(
            {"output": json.dumps(payload), "exit_code": exit_code, "error": None}
        ),
    }


def _session(commands: list[str] | None = None) -> dict:
    command_sequence = commands or [ALLOWED_COMMAND]
    messages = [{"role": "user", "content": "real hdf5 smoke", "tool_calls": None}]
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


def _write_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "state_hint": "REAL_HDF5_BASELINE_SMOKE_COMPLETED",
                "metrics": {
                    "sample_count": 2,
                    "forecast_steps": 31,
                    "competition_like_score_short": 39.1,
                    "rel_mse": 0.06,
                    "residual_rmse": 0.02,
                    "dominant_error_pattern": "overdamped_rollout",
                    "physics_issue": "nonphysical_oscillation",
                    "residual_issue": "residual_mismatch",
                },
                "artifacts": {
                    "summary_path": str(path),
                    "windows_npz_path": str(path.parent / "windows_f031.npz"),
                    "prediction_npz_path": str(path.parent / "persistence_prediction.npz"),
                    "score_path": str(path.parent / "persistence_score.json"),
                    "burgers_residual_path": str(path.parent / "persistence_burgers_residual.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    for name in [
        "windows_f031.npz",
        "persistence_prediction.npz",
        "persistence_score.json",
        "persistence_burgers_residual.json",
    ]:
        (path.parent / name).write_text("artifact", encoding="utf-8")


def test_real_hdf5_baseline_audit_passes_for_expected_session(tmp_path: Path) -> None:
    session_path = tmp_path / "session_real.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    summary_path = tmp_path / "real_hdf5_baseline_smoke_summary.json"
    _write_summary(summary_path)

    report = audit_hermes_real_hdf5_baseline_session(
        session_path=session_path,
        required_artifacts=[str(summary_path)],
        required_reads=REQUIRED_READS,
        allowed_commands=[ALLOWED_COMMAND],
    )

    assert report["status"] == "passed"
    assert report["checks"][-1]["name"] == "summary_artifact"
    assert report["summary"]["competition_like_score_short"] == 39.1


def test_real_hdf5_baseline_audit_rejects_extra_terminal_command(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session_real.json"
    session_path.write_text(
        json.dumps(_session(commands=[ALLOWED_COMMAND, "cd /repo && ls"])),
        encoding="utf-8",
    )
    summary_path = tmp_path / "real_hdf5_baseline_smoke_summary.json"
    _write_summary(summary_path)

    report = audit_hermes_real_hdf5_baseline_session(
        session_path=session_path,
        required_artifacts=[str(summary_path)],
        required_reads=REQUIRED_READS,
        allowed_commands=[ALLOWED_COMMAND],
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]
