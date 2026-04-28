import json
from pathlib import Path

from pde_agent.core.hermes_burgers_spectral_viscosity_benchmark_audit import (
    audit_hermes_burgers_spectral_viscosity_benchmark_session,
)


REQUIRED_READS = [
    "/repo/integration_tests/hermes_burgers_spectral_viscosity_benchmark/run_state.yaml",
    "/repo/integration_tests/hermes_burgers_spectral_viscosity_benchmark/bootstrap_prompt.md",
    "/repo/integration_tests/hermes_burgers_spectral_viscosity_benchmark/task_spec.json",
]

COMMAND = (
    "cd /repo && uv run --frozen --extra data pde-agent "
    "run-burgers-spectral-viscosity-benchmark --source-path /data/burgers.hdf5 "
    "--output-dir /tmp/out --dataset-key tensor --sample-limit 4 "
    "--min-evaluation-windows 4 --force"
)


def _tool_call(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _terminal_result(command: str, *, exit_code: int = 0) -> dict:
    payload = {
        "run_id": "local",
        "command": "run-burgers-spectral-viscosity-benchmark",
        "status": "ok",
        "state_hint": "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED",
        "artifacts": {},
        "metrics": {},
        "errors": [],
        "next_hint": "record-strategy-proposal",
    }
    return {
        "role": "tool",
        "content": json.dumps(
            {"output": json.dumps(payload), "exit_code": exit_code, "error": None}
        ),
    }


def _session(commands: list[str] | None = None) -> dict:
    command_sequence = commands or [COMMAND]
    messages = [{"role": "user", "content": "benchmark", "tool_calls": None}]
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


def _write_required_artifacts(
    output_dir: Path,
    *,
    evaluation_window_count: int = 4,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "burgers_spectral_viscosity_benchmark_report.json"
    decision = output_dir / "strategy_decision.json"
    evidence = output_dir / "strategy_evidence_summary.json"
    baseline = output_dir / "baseline" / "real_hdf5_baseline_smoke_summary.json"
    sweep = output_dir / "strategy_sweep" / "deeponet_strategy_sweep_summary.json"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    sweep.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "state_hint": "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED",
                "proposal_id": "burgers_spectral_viscosity_real_hdf5_v0",
                "evaluation_window_count": evaluation_window_count,
                "metrics": {
                    "mean_score_delta": -1.0,
                    "long_window_score_delta": -1.0,
                    "failure_rate": 1.0,
                },
                "decision": {
                    "decision": "retrieve_external_strategy",
                    "next_command": "record-strategy-proposal",
                    "reason_codes": ["negative_mean_score_delta"],
                },
                "artifacts": {
                    "strategy_decision_path": str(decision),
                    "strategy_evidence_summary_path": str(evidence),
                },
            }
        ),
        encoding="utf-8",
    )
    decision.write_text(json.dumps({"state_hint": "STRATEGY_NEXT_ACTION_RECOMMENDED"}))
    evidence.write_text(json.dumps({"state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT"}))
    baseline.write_text(json.dumps({"state_hint": "REAL_HDF5_BASELINE_SMOKE_COMPLETED"}))
    sweep.write_text(json.dumps({"state_hint": "DEEPONET_STRATEGY_SWEEP_COMPLETED"}))
    return [str(report), str(decision), str(evidence), str(baseline), str(sweep)]


def test_burgers_benchmark_audit_passes_for_exact_one_command_session(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session_20260428_burgers01.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out")

    report = audit_hermes_burgers_spectral_viscosity_benchmark_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=[COMMAND],
        min_evaluation_windows=4,
    )

    assert report["status"] == "passed"
    assert report["terminal_commands"] == [COMMAND]
    assert report["benchmark_report"]["proposal_id"] == (
        "burgers_spectral_viscosity_real_hdf5_v0"
    )
    assert report["metrics"]["command_count"] == 1
    assert report["metrics"]["decision"] == "retrieve_external_strategy"
    assert report["errors"] == []


def test_burgers_benchmark_audit_fails_on_manual_artifact_inspection(
    tmp_path: Path,
) -> None:
    bad_commands = [COMMAND, "cat /tmp/out/burgers_spectral_viscosity_benchmark_report.json"]
    session_path = tmp_path / "session_20260428_burgers02.json"
    session_path.write_text(json.dumps(_session(bad_commands)), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out")

    report = audit_hermes_burgers_spectral_viscosity_benchmark_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=[COMMAND],
        min_evaluation_windows=4,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]
    assert "forbidden_actions" in report["errors"]


def test_burgers_benchmark_audit_fails_on_missing_report(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260428_burgers03.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out")
    Path(artifacts[0]).unlink()

    report = audit_hermes_burgers_spectral_viscosity_benchmark_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=[COMMAND],
        min_evaluation_windows=4,
    )

    assert report["status"] == "failed"
    assert "required_artifacts" in report["errors"]
    assert "benchmark_report_semantics" in report["errors"]


def test_burgers_benchmark_audit_fails_on_insufficient_report_windows(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session_20260428_burgers04.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    artifacts = _write_required_artifacts(tmp_path / "out", evaluation_window_count=2)

    report = audit_hermes_burgers_spectral_viscosity_benchmark_session(
        session_path=session_path,
        required_artifacts=artifacts,
        required_reads=REQUIRED_READS,
        allowed_commands=[COMMAND],
        min_evaluation_windows=4,
    )

    assert report["status"] == "failed"
    assert "benchmark_report_semantics" in report["errors"]
