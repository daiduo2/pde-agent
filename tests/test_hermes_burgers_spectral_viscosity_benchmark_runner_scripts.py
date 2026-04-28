from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "run_hermes_burgers_spectral_viscosity_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "run_hermes_burgers_spectral_viscosity_benchmark",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_burgers_benchmark_runner_renders_prompt(tmp_path: Path) -> None:
    module = _load_runner_module()
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    (fixture_root / "prompt_template.md").write_text(
        "{run_state_path}\n{bootstrap_prompt_path}\n{task_spec_path}\n",
        encoding="utf-8",
    )

    prompt = module._render_prompt(fixture_root)

    assert str(fixture_root / "run_state.yaml") in prompt
    assert str(fixture_root / "bootstrap_prompt.md") in prompt
    assert str(fixture_root / "task_spec.json") in prompt


def test_burgers_benchmark_runner_reads_success_transition(tmp_path: Path) -> None:
    module = _load_runner_module()
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    (fixture_root / "run_state.yaml").write_text(
        "success_transition:\n"
        "  on_complete: BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED\n",
        encoding="utf-8",
    )

    assert module._success_transition_from_run_state(fixture_root) == (
        "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
    )


def test_burgers_benchmark_runner_extracts_session_footer() -> None:
    module = _load_runner_module()

    assert (
        module._extract_session_id(
            "Resume this session with:\n"
            "  hermes --resume 20260428_213103_c7e027\n\n"
            "Session:        20260428_213103_c7e027\n"
        )
        == "20260428_213103_c7e027"
    )


def test_burgers_benchmark_audit_script_writes_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_root = tmp_path / "fixture"
    sessions_dir = fixture_root / "hermes_home" / "sessions"
    sessions_dir.mkdir(parents=True)
    report_dir = fixture_root / "reports"
    report_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    benchmark_report = output_dir / "burgers_spectral_viscosity_benchmark_report.json"
    decision = output_dir / "strategy_decision.json"
    evidence = output_dir / "strategy_evidence_summary.json"
    baseline = output_dir / "baseline" / "real_hdf5_baseline_smoke_summary.json"
    sweep = output_dir / "strategy_sweep" / "deeponet_strategy_sweep_summary.json"
    baseline.parent.mkdir()
    sweep.parent.mkdir()
    command = (
        "cd /repo && uv run --frozen --extra data pde-agent "
        "run-burgers-spectral-viscosity-benchmark --source-path /data/burgers.hdf5 "
        "--output-dir /tmp/out --dataset-key tensor --sample-limit 4 "
        "--min-evaluation-windows 4 --force"
    )
    required_reads = [
        str(fixture_root / "run_state.yaml"),
        str(fixture_root / "bootstrap_prompt.md"),
        str(fixture_root / "task_spec.json"),
    ]
    (fixture_root / "run_state.yaml").write_text(
        "min_evaluation_windows: 4\n"
        "required_reads:\n"
        f"  - {required_reads[0]}\n"
        f"  - {required_reads[1]}\n"
        f"  - {required_reads[2]}\n"
        "required_artifacts:\n"
        f"  - {benchmark_report}\n"
        f"  - {decision}\n"
        f"  - {evidence}\n"
        f"  - {baseline}\n"
        f"  - {sweep}\n"
        "allowed_commands:\n"
        "  - name: run_burgers_spectral_viscosity_benchmark\n"
        f"    command: {command}\n"
        "success_transition:\n"
        "  on_complete: BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED\n",
        encoding="utf-8",
    )
    benchmark_report.write_text(
        json.dumps(
            {
                "state_hint": "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED",
                "proposal_id": "burgers_spectral_viscosity_real_hdf5_v0",
                "evaluation_window_count": 4,
                "metrics": {
                    "mean_score_delta": -1.0,
                    "long_window_score_delta": -1.0,
                    "failure_rate": 1.0,
                },
                "decision": {
                    "decision": "retrieve_external_strategy",
                    "next_command": "record-strategy-proposal",
                },
                "artifacts": {
                    "strategy_decision_path": str(decision),
                    "strategy_evidence_summary_path": str(evidence),
                },
            }
        ),
        encoding="utf-8",
    )
    for path in [decision, evidence, baseline, sweep]:
        path.write_text("{}", encoding="utf-8")
    session_path = sessions_dir / "session_20260428_burgers.json"
    session_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "benchmark", "tool_calls": None},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": path}),
                                },
                            }
                            for path in required_reads
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": json.dumps({"command": command}),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "output": json.dumps(
                                    {
                                        "status": "ok",
                                        "state_hint": (
                                            "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
                                        ),
                                    }
                                ),
                                "exit_code": 0,
                            }
                        ),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "audit.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_hermes_burgers_spectral_viscosity_benchmark.py",
            "--fixture-root",
            str(fixture_root),
            "--session-path",
            str(session_path),
            "--report-path",
            str(report_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["status"] == "passed"
