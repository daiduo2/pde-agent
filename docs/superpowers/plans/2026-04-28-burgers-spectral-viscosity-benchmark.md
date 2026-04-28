# Burgers Spectral Viscosity Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic real-HDF5 Burgers benchmark command that evaluates the retrieved spectral-viscosity DeepONet strategy against persistence and DeepONet controls across multiple local windows.

**Architecture:** Add a small orchestration core that composes existing baseline smoke, DeepONet strategy sweep, operator evidence conversion, and strategy decision routing. Add a thin CLI wrapper that returns the standard JSON protocol payload. Keep Hermes fixture work deferred until the local quantitative chain is stable.

**Tech Stack:** Python 3.14, Typer, NumPy, optional h5py in tests, existing pde-agent JSON artifact contracts, pytest.

---

## File Structure

- Create `src/pde_agent/core/burgers_spectral_viscosity_benchmark.py`
  - Owns validation, default strategy configuration, artifact paths, orchestration, and final benchmark report writing.
- Create `src/pde_agent/commands/run_burgers_spectral_viscosity_benchmark.py`
  - Wraps the core report in `success_payload`.
- Modify `src/pde_agent/cli.py`
  - Imports the command wrapper and exposes `run-burgers-spectral-viscosity-benchmark`.
- Create `tests/test_burgers_spectral_viscosity_benchmark.py`
  - Covers core behavior, CLI payload, and validation failure behavior using synthetic HDF5 fixtures.

## Task 1: Core Benchmark Report

**Files:**
- Create: `tests/test_burgers_spectral_viscosity_benchmark.py`
- Create: `src/pde_agent/core/burgers_spectral_viscosity_benchmark.py`

- [ ] **Step 1: Write the failing core test**

Add `tests/test_burgers_spectral_viscosity_benchmark.py` with a synthetic HDF5 writer and this test:

```python
import json
from pathlib import Path

import numpy as np
import pytest


def _write_hdf5(path: Path, *, sample_count: int = 8) -> Path:
    h5py = pytest.importorskip("h5py")
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
    tensor = np.zeros((sample_count, 50, 16), dtype=np.float32)
    for sample in range(tensor.shape[0]):
        for time in range(tensor.shape[1]):
            tensor[sample, time, :] = (
                np.sin(x + 0.07 * sample + 0.04 * time)
                + 0.08 * np.sin(3.0 * x + 0.01 * time)
            )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)
    return path


def test_run_burgers_spectral_viscosity_benchmark_writes_quantitative_report(
    tmp_path: Path,
) -> None:
    from pde_agent.core.burgers_spectral_viscosity_benchmark import (
        run_burgers_spectral_viscosity_benchmark,
    )

    source_path = _write_hdf5(tmp_path / "burgers.hdf5", sample_count=8)
    output_dir = tmp_path / "benchmark"

    report = run_burgers_spectral_viscosity_benchmark(
        source_path=source_path,
        output_dir=output_dir,
        dataset_key="tensor",
        sample_limit=4,
        min_evaluation_windows=3,
        context_steps=10,
        forecast_steps=31,
        time_reduction=1,
        space_reduction=2,
        train_steps=[2],
        hidden_widths=[8],
        latent_dims=[4],
        learning_rates=[0.01],
        seeds=[5],
        spectral_weights=[0.05, 0.1],
        physics_weights=[0.005],
        proposal_id="burgers_spectral_viscosity_real_hdf5_v0",
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert report["state_hint"] == "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
    assert report["proposal_id"] == "burgers_spectral_viscosity_real_hdf5_v0"
    assert report["evaluation_window_count"] == 4
    assert report["metrics"]["baseline_score"] > 0.0
    assert isinstance(report["metrics"]["mean_score_delta"], float)
    assert isinstance(report["metrics"]["long_window_score_delta"], float)
    assert report["decision"]["decision"] in {
        "evaluate_promotion",
        "expand_evidence",
        "retrieve_external_strategy",
    }
    assert report["decision"]["next_command"]
    assert set(report["inputs"]["training_strategies"]) == {
        "branch_trunk_supervised",
        "branch_trunk_physics_residual",
        "branch_trunk_spectral",
    }
    artifact_paths = report["artifacts"]
    for key in [
        "baseline_summary_path",
        "baseline_windows_path",
        "baseline_score_path",
        "deeponet_strategy_sweep_summary_path",
        "deeponet_evidence_summary_path",
        "strategy_evidence_summary_path",
        "strategy_decision_path",
        "benchmark_report_path",
    ]:
        assert Path(artifact_paths[key]).exists(), key
    written = json.loads(
        (output_dir / "burgers_spectral_viscosity_benchmark_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == report
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run --frozen pytest tests/test_burgers_spectral_viscosity_benchmark.py::test_run_burgers_spectral_viscosity_benchmark_writes_quantitative_report -q
```

Expected: fail with `ModuleNotFoundError: No module named 'pde_agent.core.burgers_spectral_viscosity_benchmark'`.

- [ ] **Step 3: Implement minimal core orchestration**

Create `src/pde_agent/core/burgers_spectral_viscosity_benchmark.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.operator_strategy_evidence import convert_operator_evidence_summary
from pde_agent.core.real_hdf5_baseline_smoke import run_real_hdf5_baseline_smoke
from pde_agent.core.strategy_decision import recommend_strategy_next_action
from pde_agent.core.deeponet_strategy_sweep import run_deeponet_strategy_sweep


STATE_HINT = "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
DEFAULT_PROPOSAL_ID = "burgers_spectral_viscosity_real_hdf5_v0"
DEFAULT_TRAINING_STRATEGIES = [
    "branch_trunk_supervised",
    "branch_trunk_physics_residual",
    "branch_trunk_spectral",
]


class BurgersSpectralViscosityBenchmarkError(RuntimeError):
    pass


def run_burgers_spectral_viscosity_benchmark(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    sample_limit: int,
    min_evaluation_windows: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: list[int],
    hidden_widths: list[int],
    latent_dims: list[int],
    learning_rates: list[float],
    seeds: list[int],
    spectral_weights: list[float],
    physics_weights: list[float],
    proposal_id: str = DEFAULT_PROPOSAL_ID,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    source_file = Path(source_path)
    output_root = Path(output_dir)
    _validate_inputs(
        source_file=source_file,
        output_root=output_root,
        dataset_key=dataset_key,
        sample_limit=sample_limit,
        min_evaluation_windows=min_evaluation_windows,
        proposal_id=proposal_id,
    )
    paths = _paths(output_root, forecast_steps)
    if paths["benchmark_report_path"].exists() and not force:
        raise BurgersSpectralViscosityBenchmarkError(
            f"benchmark report already exists: {paths['benchmark_report_path']}"
        )

    baseline = run_real_hdf5_baseline_smoke(
        source_path=source_file,
        output_dir=paths["baseline_dir"],
        dataset_key=dataset_key,
        start_index=0,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    start_indices = list(range(sample_limit))
    sweep = run_deeponet_strategy_sweep(
        windows_path=paths["baseline_windows_path"],
        output_dir=paths["strategy_sweep_dir"],
        baseline_score_path=paths["baseline_score_path"],
        start_indices=start_indices,
        training_strategies=DEFAULT_TRAINING_STRATEGIES,
        train_steps=train_steps,
        hidden_widths=hidden_widths,
        latent_dims=latent_dims,
        learning_rates=learning_rates,
        seeds=seeds,
        spectral_weights=spectral_weights,
        physics_weights=physics_weights,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    evidence = convert_operator_evidence_summary(
        sweep_summary_path=paths["deeponet_evidence_summary_path"],
        output_path=paths["strategy_evidence_summary_path"],
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=forecast_steps,
        force=force,
    )
    decision = recommend_strategy_next_action(
        evidence_summary_path=paths["strategy_evidence_summary_path"],
        output_path=paths["strategy_decision_path"],
        min_evaluation_windows=min_evaluation_windows,
    )
    report = _report(
        source_file=source_file,
        output_root=output_root,
        dataset_key=dataset_key,
        sample_limit=sample_limit,
        min_evaluation_windows=min_evaluation_windows,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        train_steps=train_steps,
        hidden_widths=hidden_widths,
        latent_dims=latent_dims,
        learning_rates=learning_rates,
        seeds=seeds,
        spectral_weights=spectral_weights,
        physics_weights=physics_weights,
        proposal_id=proposal_id,
        nu=nu,
        dt=dt,
        dx=dx,
        baseline=baseline,
        sweep=sweep,
        evidence=evidence,
        decision=decision,
        paths=paths,
    )
    paths["benchmark_report_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["benchmark_report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
```

Then add helper functions `_validate_inputs`, `_paths`, `_report`, `_optional_float`, and `_ranking_metric`. Keep them private and deterministic.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_burgers_spectral_viscosity_benchmark.py::test_run_burgers_spectral_viscosity_benchmark_writes_quantitative_report -q
```

Expected: pass.

## Task 2: Validation Behavior

**Files:**
- Modify: `tests/test_burgers_spectral_viscosity_benchmark.py`
- Modify: `src/pde_agent/core/burgers_spectral_viscosity_benchmark.py`

- [ ] **Step 1: Write the failing validation test**

Append:

```python
def test_run_burgers_spectral_viscosity_benchmark_rejects_insufficient_windows(
    tmp_path: Path,
) -> None:
    from pde_agent.core.burgers_spectral_viscosity_benchmark import (
        BurgersSpectralViscosityBenchmarkError,
        run_burgers_spectral_viscosity_benchmark,
    )

    source_path = _write_hdf5(tmp_path / "burgers.hdf5", sample_count=4)
    output_dir = tmp_path / "benchmark"

    with pytest.raises(
        BurgersSpectralViscosityBenchmarkError,
        match="sample_limit must be at least min_evaluation_windows",
    ):
        run_burgers_spectral_viscosity_benchmark(
            source_path=source_path,
            output_dir=output_dir,
            dataset_key="tensor",
            sample_limit=2,
            min_evaluation_windows=3,
            context_steps=10,
            forecast_steps=31,
            time_reduction=1,
            space_reduction=2,
            train_steps=[2],
            hidden_widths=[8],
            latent_dims=[4],
            learning_rates=[0.01],
            seeds=[5],
            spectral_weights=[0.05],
            physics_weights=[0.005],
        )

    assert not (output_dir / "burgers_spectral_viscosity_benchmark_report.json").exists()
```

- [ ] **Step 2: Run the validation test and verify RED**

Run:

```bash
uv run --frozen pytest tests/test_burgers_spectral_viscosity_benchmark.py::test_run_burgers_spectral_viscosity_benchmark_rejects_insufficient_windows -q
```

Expected: fail with no `BurgersSpectralViscosityBenchmarkError` or with a message mismatch until validation enforces the sample/window invariant.

- [ ] **Step 3: Implement validation**

In `_validate_inputs`, enforce:

```python
if not source_file.exists():
    raise BurgersSpectralViscosityBenchmarkError(
        f"source HDF5 file does not exist: {source_file}"
    )
if not dataset_key:
    raise BurgersSpectralViscosityBenchmarkError("dataset_key must be non-empty")
if sample_limit < 2:
    raise BurgersSpectralViscosityBenchmarkError("sample_limit must be at least 2")
if min_evaluation_windows < 2:
    raise BurgersSpectralViscosityBenchmarkError(
        "min_evaluation_windows must be at least 2"
    )
if sample_limit < min_evaluation_windows:
    raise BurgersSpectralViscosityBenchmarkError(
        "sample_limit must be at least min_evaluation_windows"
    )
if not proposal_id:
    raise BurgersSpectralViscosityBenchmarkError("proposal_id must be non-empty")
if output_root.exists() and not output_root.is_dir():
    raise BurgersSpectralViscosityBenchmarkError(
        f"output_dir exists and is not a directory: {output_root}"
    )
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --frozen pytest tests/test_burgers_spectral_viscosity_benchmark.py -q
```

Expected: all benchmark tests pass.

## Task 3: CLI Command

**Files:**
- Modify: `tests/test_burgers_spectral_viscosity_benchmark.py`
- Create: `src/pde_agent/commands/run_burgers_spectral_viscosity_benchmark.py`
- Modify: `src/pde_agent/cli.py`

- [ ] **Step 1: Write the failing CLI test**

Append:

```python
def test_run_burgers_spectral_viscosity_benchmark_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    source_path = _write_hdf5(tmp_path / "burgers.hdf5", sample_count=8)
    output_dir = tmp_path / "benchmark"

    result = CliRunner().invoke(
        app,
        [
            "run-burgers-spectral-viscosity-benchmark",
            "--source-path",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--dataset-key",
            "tensor",
            "--sample-limit",
            "4",
            "--min-evaluation-windows",
            "3",
            "--context-steps",
            "10",
            "--forecast-steps",
            "31",
            "--time-reduction",
            "1",
            "--space-reduction",
            "2",
            "--train-steps",
            "2",
            "--hidden-width",
            "8",
            "--latent-dim",
            "4",
            "--learning-rate",
            "0.01",
            "--seed",
            "5",
            "--spectral-weight",
            "0.05",
            "--spectral-weight",
            "0.1",
            "--physics-weight",
            "0.005",
            "--proposal-id",
            "burgers_spectral_viscosity_real_hdf5_v0",
            "--nu",
            "0.001",
            "--dt",
            "1.0",
            "--dx",
            "1.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-burgers-spectral-viscosity-benchmark"
    assert payload["state_hint"] == "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
    assert payload["artifacts"]["benchmark_report_path"] == str(
        output_dir / "burgers_spectral_viscosity_benchmark_report.json"
    )
    assert payload["metrics"]["evaluation_window_count"] == 4
    assert payload["metrics"]["proposal_id"] == "burgers_spectral_viscosity_real_hdf5_v0"
    assert payload["metrics"]["decision"] in {
        "evaluate_promotion",
        "expand_evidence",
        "retrieve_external_strategy",
    }
    assert payload["next_hint"]
```

- [ ] **Step 2: Run CLI test and verify RED**

Run:

```bash
uv run --frozen pytest tests/test_burgers_spectral_viscosity_benchmark.py::test_run_burgers_spectral_viscosity_benchmark_cli_returns_protocol_payload -q
```

Expected: fail because the command is not registered.

- [ ] **Step 3: Add command wrapper**

Create `src/pde_agent/commands/run_burgers_spectral_viscosity_benchmark.py`:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.burgers_spectral_viscosity_benchmark import (
    STATE_HINT,
    run_burgers_spectral_viscosity_benchmark,
)
from pde_agent.core.json_output import success_payload


def run_burgers_spectral_viscosity_benchmark_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    sample_limit: int,
    min_evaluation_windows: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: list[int],
    hidden_widths: list[int],
    latent_dims: list[int],
    learning_rates: list[float],
    seeds: list[int],
    spectral_weights: list[float],
    physics_weights: list[float],
    proposal_id: str,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    report = run_burgers_spectral_viscosity_benchmark(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        sample_limit=sample_limit,
        min_evaluation_windows=min_evaluation_windows,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        train_steps=train_steps,
        hidden_widths=hidden_widths,
        latent_dims=latent_dims,
        learning_rates=learning_rates,
        seeds=seeds,
        spectral_weights=spectral_weights,
        physics_weights=physics_weights,
        proposal_id=proposal_id,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-burgers-spectral-viscosity-benchmark",
        state_hint=STATE_HINT,
        artifacts={
            "benchmark_report_path": report["artifacts"]["benchmark_report_path"],
            "strategy_decision_path": report["artifacts"]["strategy_decision_path"],
            "strategy_evidence_summary_path": report["artifacts"][
                "strategy_evidence_summary_path"
            ],
        },
        metrics={
            "proposal_id": report["proposal_id"],
            "evaluation_window_count": report["evaluation_window_count"],
            "best_strategy_id": report["metrics"]["best_strategy_id"],
            "mean_score_delta": report["metrics"]["mean_score_delta"],
            "long_window_score_delta": report["metrics"]["long_window_score_delta"],
            "failure_rate": report["metrics"]["failure_rate"],
            "decision": report["decision"]["decision"],
        },
        next_hint=report["decision"]["next_command"],
    )
```

- [ ] **Step 4: Register CLI command**

In `src/pde_agent/cli.py`, import the wrapper and add:

```python
@app.command("run-burgers-spectral-viscosity-benchmark")
def run_burgers_spectral_viscosity_benchmark_cmd(
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    sample_limit: int = typer.Option(8),
    min_evaluation_windows: int = typer.Option(4),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(31),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    train_steps: list[int] = typer.Option([4], "--train-steps"),
    hidden_widths: list[int] = typer.Option([8], "--hidden-width"),
    latent_dims: list[int] = typer.Option([4], "--latent-dim"),
    learning_rates: list[float] = typer.Option([0.01], "--learning-rate"),
    seeds: list[int] = typer.Option([5], "--seed"),
    spectral_weights: list[float] = typer.Option([0.05, 0.1], "--spectral-weight"),
    physics_weights: list[float] = typer.Option([0.005], "--physics-weight"),
    proposal_id: str = typer.Option("burgers_spectral_viscosity_real_hdf5_v0"),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_burgers_spectral_viscosity_benchmark_command(
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                sample_limit=sample_limit,
                min_evaluation_windows=min_evaluation_windows,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                train_steps=train_steps,
                hidden_widths=hidden_widths,
                latent_dims=latent_dims,
                learning_rates=learning_rates,
                seeds=seeds,
                spectral_weights=spectral_weights,
                physics_weights=physics_weights,
                proposal_id=proposal_id,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run --frozen pytest tests/test_burgers_spectral_viscosity_benchmark.py -q
```

Expected: all benchmark tests pass.

## Task 4: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README section**

Add a short `Burgers Spectral Viscosity Benchmark` section after `Lightweight Spectral Baseline` explaining:

- it is a real-HDF5 Burgers local benchmark for the retrieved strategy
- it compares spectral-viscosity DeepONet proxy against controls
- it records negative evidence instead of forcing promotion
- it is not the final contest model

- [ ] **Step 2: Run focused and full verification**

Run:

```bash
uv run --frozen pytest tests/test_burgers_spectral_viscosity_benchmark.py -q
uv run --frozen pytest -q
git diff --check
```

Expected:

- benchmark tests pass
- full suite passes
- no whitespace errors

- [ ] **Step 3: Commit implementation**

Run:

```bash
git add README.md src/pde_agent/core/burgers_spectral_viscosity_benchmark.py src/pde_agent/commands/run_burgers_spectral_viscosity_benchmark.py src/pde_agent/cli.py tests/test_burgers_spectral_viscosity_benchmark.py docs/superpowers/plans/2026-04-28-burgers-spectral-viscosity-benchmark.md
git commit -m "feat: add burgers spectral viscosity benchmark"
```
