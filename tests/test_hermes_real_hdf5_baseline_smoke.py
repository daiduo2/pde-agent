from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "prepare_real_hdf5_baseline_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_real_hdf5_baseline_smoke",
        path,
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


def test_prepare_real_hdf5_baseline_smoke_renders_single_command_protocol(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_real_hdf5_baseline_smoke"
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
            / "hermes_real_hdf5_baseline_smoke"
            / name
        )
        (fixture_root / name).write_text(source.read_text(encoding="utf-8"))
    hdf5_path = tmp_path / "burgers.hdf5"
    hdf5_path.write_bytes(b"hdf5")

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        source_path=hdf5_path,
        output_dir=tmp_path / "runs" / "real_hdf5_baseline_smoke_001",
        dataset_key="tensor",
        start_index=0,
        sample_limit=2,
        context_steps=10,
        forecast_steps=31,
        time_reduction=5,
        space_reduction=4,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    run_state = outputs["run_state"].read_text(encoding="utf-8")
    commands = _allowed_commands(run_state)
    assert "{{" not in run_state
    assert len(commands) == 1
    assert "run-real-hdf5-baseline-smoke" in commands[0]
    assert str(hdf5_path.resolve()) in commands[0]
    assert "--sample-limit 2" in commands[0]
    assert "--forecast-steps 31" in commands[0]
    assert "run-fno-iteration" not in run_state
    assert "train-baseline" not in run_state
    assert str(tmp_path / "runs" / "real_hdf5_baseline_smoke_001") in run_state
    assert "REAL_HDF5_BASELINE_SMOKE_COMPLETED" in run_state


def test_prepare_real_hdf5_baseline_smoke_rejects_missing_source(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_real_hdf5_baseline_smoke"
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.template.yaml").write_text("run_id: {{RUN_ID}}\n")

    with pytest.raises(SystemExit, match="source HDF5 file does not exist"):
        module.prepare_runtime(
            repo_root=repo_root,
            source_path=tmp_path / "missing.hdf5",
            output_dir=tmp_path / "runs" / "real_hdf5_baseline_smoke_001",
            dataset_key="tensor",
            start_index=0,
            sample_limit=2,
            context_steps=10,
            forecast_steps=31,
            time_reduction=5,
            space_reduction=4,
            nu=0.001,
            dt=1.0,
            dx=1.0,
        )

    assert not (fixture_root / "run_state.yaml").exists()
