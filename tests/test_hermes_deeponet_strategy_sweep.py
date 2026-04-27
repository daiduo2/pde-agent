from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "prepare_deeponet_strategy_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_deeponet_strategy_sweep",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _allowed_commands(run_state: str) -> list[str]:
    commands: list[str] = []
    for line in run_state.splitlines():
        stripped = line.strip()
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))
    return commands


def test_prepare_deeponet_strategy_sweep_renders_protocol(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_deeponet_strategy_sweep"
    fixture_root.mkdir(parents=True)
    source_fixture = (
        Path(__file__).resolve().parents[1]
        / "integration_tests"
        / "hermes_deeponet_strategy_sweep"
    )
    for name in [
        "run_state.template.yaml",
        "bootstrap_prompt.md",
        "prompt_template.md",
        "task_spec.json",
    ]:
        (fixture_root / name).write_text(
            (source_fixture / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        output_dir=tmp_path / "runs" / "deeponet_strategy_sweep_001",
    )

    run_state = outputs["run_state"].read_text(encoding="utf-8")
    commands = _allowed_commands(run_state)
    assert "{{" not in run_state
    assert len(commands) == 7
    assert "run-deeponet-strategy-sweep" in commands[0]
    assert commands[0].count("--start-index") == 2
    assert commands[0].count("--training-strategy") == 3
    assert commands[0].count("--spectral-weight") == 2
    assert commands[0].count("--physics-weight") == 2
    assert "branch_trunk_physics_residual" in commands[0]
    assert "convert-operator-evidence-summary" in commands[1]
    assert "recommend-strategy-next-action" in commands[2]
    assert "record-strategy-proposal" in commands[3]
    assert "append-strategy-ledger" in commands[6]
    assert str(outputs["windows_path"].resolve()) in commands[0]
    assert str(outputs["baseline_score_path"].resolve()) in commands[0]
    assert outputs["windows_path"].exists()
    assert outputs["baseline_score_path"].exists()
