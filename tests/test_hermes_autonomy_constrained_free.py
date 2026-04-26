from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "prepare_autonomy_constrained_free.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_autonomy_constrained_free", path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _allowed_commands(run_state: str) -> list[str]:
    commands: list[str] = []
    in_allowed_commands = False
    for line in run_state.splitlines():
        stripped = line.strip()
        if stripped == "allowed_commands:":
            in_allowed_commands = True
            continue
        if not in_allowed_commands:
            continue
        if line and not line.startswith("  "):
            break
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))
    return commands


def test_prepare_autonomy_constrained_free_renders_choice_protocol(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_autonomy_constrained_free"
    fixture_root.mkdir(parents=True)
    for name in [
        "run_state.template.yaml",
        "bootstrap_prompt.md",
        "prompt_template.md",
        "task_spec.json",
    ]:
        source = (
            Path(__file__).resolve().parents[1]
            / "integration_tests"
            / "hermes_autonomy_constrained_free"
            / name
        )
        (fixture_root / name).write_text(source.read_text(encoding="utf-8"))
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "checkpoint.tar"
    output_dir = tmp_path / "runs" / "autonomy_001"
    windows_path.write_bytes(b"windows")
    checkpoint_path.write_bytes(b"checkpoint")

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        max_train_steps=1,
        learning_rate=1e-4,
    )

    run_state = outputs["run_state"].read_text(encoding="utf-8")
    commands = _allowed_commands(run_state)
    assert "{{" not in run_state
    assert len(commands) == 3
    assert "run-fno-iteration" in commands[0]
    assert commands[1].endswith(
        "--decision request_evidence_sweep "
        "--reason-code single_window_improvement_needs_validation "
        f"--evidence-source-path {output_dir}/fno_iteration_report.json "
        f"--output-path {output_dir}/autonomy_decision.json"
    )
    assert commands[2].endswith(
        "--decision accept_current_winner "
        "--reason-code clear_improvement_accept_for_now "
        f"--evidence-source-path {output_dir}/fno_iteration_report.json "
        f"--output-path {output_dir}/autonomy_decision.json"
    )
    assert str(output_dir / "fno_iteration_report.json") in run_state
    assert str(output_dir / "candidate_comparison.json") in run_state
    assert str(output_dir / "autonomy_decision.json") in run_state
    assert "execute-fno-finetune-candidate" not in run_state
    assert "evaluate-fno-candidate" not in run_state
    assert "compare-fno-candidates" not in run_state


def test_prepare_autonomy_constrained_free_rejects_missing_inputs(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_autonomy_constrained_free"
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.template.yaml").write_text("run_id: {{RUN_ID}}\n")

    with pytest.raises(SystemExit, match="windows file does not exist"):
        module.prepare_runtime(
            repo_root=repo_root,
            windows_path=tmp_path / "missing_windows.npz",
            checkpoint_path=tmp_path / "missing_checkpoint.tar",
            output_dir=tmp_path / "runs" / "autonomy_001",
            max_train_steps=1,
            learning_rate=1e-4,
        )

    assert not (fixture_root / "run_state.yaml").exists()
