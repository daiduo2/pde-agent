from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "prepare_burgers_spectral_viscosity_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_burgers_spectral_viscosity_benchmark",
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


def _yaml_list(run_state: str, key: str) -> list[str]:
    values: list[str] = []
    inside = False
    for line in run_state.splitlines():
        if line.strip() == f"{key}:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            break
        if inside and line.strip().startswith("- "):
            values.append(line.strip()[2:])
    return values


def test_prepare_burgers_spectral_viscosity_benchmark_renders_one_command_protocol(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = (
        repo_root
        / "integration_tests"
        / "hermes_burgers_spectral_viscosity_benchmark"
    )
    fixture_root.mkdir(parents=True)
    source_fixture = (
        Path(__file__).resolve().parents[1]
        / "integration_tests"
        / "hermes_burgers_spectral_viscosity_benchmark"
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
    hdf5_path = tmp_path / "burgers.hdf5"
    hdf5_path.write_bytes(b"hdf5")
    output_dir = tmp_path / "runs" / "burgers_spectral_viscosity_benchmark_001"

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        source_path=hdf5_path,
        output_dir=output_dir,
        dataset_key="tensor",
        sample_limit=4,
        min_evaluation_windows=4,
        forecast_steps=31,
    )

    run_state = outputs["run_state"].read_text(encoding="utf-8")
    commands = _allowed_commands(run_state)
    required_artifacts = _yaml_list(run_state, "required_artifacts")

    assert "{{" not in run_state
    assert len(commands) == 1
    command = commands[0]
    assert "run-burgers-spectral-viscosity-benchmark" in command
    assert str(hdf5_path.resolve()) in command
    assert str(output_dir.resolve()) in command
    assert "--sample-limit 4" in command
    assert "--min-evaluation-windows 4" in command
    assert "--forecast-steps 31" in command
    assert "--force" in command
    assert "run-real-hdf5-baseline-smoke" not in command
    assert "run-deeponet-strategy-sweep" not in command
    assert len(required_artifacts) == 5
    assert str(output_dir.resolve() / "burgers_spectral_viscosity_benchmark_report.json") in required_artifacts
    assert str(output_dir.resolve() / "strategy_decision.json") in required_artifacts
    assert str(output_dir.resolve() / "strategy_evidence_summary.json") in required_artifacts
    assert outputs["benchmark_report_path"] == (
        output_dir.resolve() / "burgers_spectral_viscosity_benchmark_report.json"
    )


def test_prepare_burgers_spectral_viscosity_benchmark_rejects_missing_source(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = (
        repo_root
        / "integration_tests"
        / "hermes_burgers_spectral_viscosity_benchmark"
    )
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.template.yaml").write_text("run_id: {{RUN_ID}}\n")

    with pytest.raises(SystemExit, match="source HDF5 file does not exist"):
        module.prepare_runtime(
            repo_root=repo_root,
            source_path=tmp_path / "missing.hdf5",
            output_dir=tmp_path / "runs" / "burgers_spectral_viscosity_benchmark_001",
            dataset_key="tensor",
            sample_limit=4,
            min_evaluation_windows=4,
            forecast_steps=31,
        )

    assert not (fixture_root / "run_state.yaml").exists()
