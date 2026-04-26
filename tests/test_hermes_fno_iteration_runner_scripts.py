from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_script_parses_run_state_lists(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "audit_hermes_fno_iteration",
        repo_root / "scripts" / "audit_hermes_fno_iteration.py",
    )
    run_state = """
required_reads:
  - /repo/run_state.yaml
  - /repo/bootstrap_prompt.md
required_artifacts:
  - /tmp/fno_iteration_report.json
  - /tmp/candidate_comparison.json
allowed_commands:
  - name: run_fno_iteration
    command: cd /repo && uv run --frozen --extra torch pde-agent run-fno-iteration --windows-path /tmp/windows.npz --checkpoint-path /repo/c.pt --output-dir /tmp/out --max-train-steps 1 --learning-rate 0.0001 --force
success_transition:
  on_complete: FNO_ITERATION_COMPLETED
"""

    assert module._yaml_list(run_state, "required_reads") == [
        "/repo/run_state.yaml",
        "/repo/bootstrap_prompt.md",
    ]
    assert module._yaml_list(run_state, "required_artifacts") == [
        "/tmp/fno_iteration_report.json",
        "/tmp/candidate_comparison.json",
    ]
    assert module._yaml_command_list(run_state) == [
        "cd /repo && uv run --frozen --extra torch pde-agent run-fno-iteration --windows-path /tmp/windows.npz --checkpoint-path /repo/c.pt --output-dir /tmp/out --max-train-steps 1 --learning-rate 0.0001 --force"
    ]


def test_run_script_renders_fno_iteration_prompt() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "run_hermes_fno_iteration",
        repo_root / "scripts" / "run_hermes_fno_iteration.py",
    )
    fixture_root = repo_root / "integration_tests" / "hermes_fno_iteration_smoke"

    prompt = module._render_prompt(fixture_root)

    assert str(fixture_root / "run_state.yaml") in prompt
    assert str(fixture_root / "bootstrap_prompt.md") in prompt
    assert str(fixture_root / "task_spec.json") in prompt
    assert "FNO_ITERATION_COMPLETED" in prompt
