# Lightweight Spectral Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a NumPy-only trainable spectral residual baseline that produces checkpoints and prediction NPZ files consumable by the existing local scoring and PDE diagnostic chain.

**Architecture:** Implement one focused core module for training/prediction, two thin CLI command wrappers, and tests that prove the model beats persistence on a deterministic trend fixture. Keep PyTorch out of the dependency graph while preserving a clean future extension path.

**Tech Stack:** Python 3.12+, NumPy, Typer, pytest, existing `success_payload`, existing local scoring and PDE diagnostic commands.

---

## File Structure

- Create `src/pde_agent/core/spectral_residual_baseline.py`: load windows, build Fourier/trend features, train ridge regression, save/load checkpoint, predict residuals.
- Create `src/pde_agent/commands/train_spectral_residual_baseline.py`: CLI wrapper for training.
- Create `src/pde_agent/commands/predict_local_model.py`: CLI wrapper for prediction.
- Modify `src/pde_agent/cli.py`: register `train-spectral-residual-baseline` and `predict-local-model`.
- Modify `src/pde_agent/core/state_hints.py`: add `LOCAL_MODEL_TRAINED` and `LOCAL_MODEL_PREDICTED`.
- Create `tests/test_lightweight_spectral_baseline.py`: unit and CLI tests.
- Modify `README.md`: add lightweight spectral model chain.

## Design Rules

- Do not add PyTorch or any new dependency.
- Do not modify existing scoring/diagnostic behavior.
- Do not require real HDF5 in tests.
- Write NPZ artifacts to the exact requested path by opening file handles; do not let `np.savez` append `.npz`.
- Use `allow_pickle=False` for NPZ reads.
- Use `allow_nan=False` for JSON serialization or validate finite values before writing.
- Keep model deterministic; no random initialization.

---

### Task 1: Core Spectral Residual Model

**Files:**
- Create: `src/pde_agent/core/spectral_residual_baseline.py`
- Test: `tests/test_lightweight_spectral_baseline.py`

- [ ] **Step 1: Write failing core tests**

Create `tests/test_lightweight_spectral_baseline.py` with this content:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.core.local_scoring import score_prediction_arrays
from pde_agent.core.spectral_residual_baseline import (
    SpectralResidualBaselineError,
    predict_spectral_residual_baseline,
    train_spectral_residual_baseline,
)


def _write_windows(path: Path, context: np.ndarray, target: np.ndarray) -> None:
    full_reference = np.concatenate([context, target], axis=1)
    with path.open("wb") as handle:
        np.savez(
            handle,
            context=context.astype(np.float32),
            target=target.astype(np.float32),
            full_reference=full_reference.astype(np.float32),
            metadata_json=json.dumps({"fixture": True}),
        )


def _linear_trend_windows(
    *,
    sample_count: int = 6,
    context_steps: int = 4,
    forecast_steps: int = 5,
    space_points: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 2.0 * np.pi, space_points, endpoint=False)
    context = np.zeros((sample_count, context_steps, space_points), dtype=np.float32)
    target = np.zeros((sample_count, forecast_steps, space_points), dtype=np.float32)
    for sample in range(sample_count):
        base = np.sin(x + sample * 0.1)
        velocity = 0.05 * np.cos(x) + 0.02 * (sample + 1)
        for t in range(context_steps):
            context[sample, t, :] = base + velocity * t
        for k in range(forecast_steps):
            target[sample, k, :] = base + velocity * (context_steps + k)
    return context, target


def test_train_spectral_residual_baseline_writes_checkpoint(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "checkpoint"
    context, target = _linear_trend_windows()
    _write_windows(windows_path, context, target)

    summary = train_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        modes=6,
        ridge=1e-6,
        force=False,
    )

    assert checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".npz").exists()
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        assert {"weights_real", "weights_imag", "feature_mean", "feature_scale", "metadata_json"} <= set(archive.files)
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["model_type"] == "spectral_residual_ridge"
    assert metadata["modes"] == 6
    assert metadata["forecast_steps"] == target.shape[1]
    assert summary["checkpoint_path"] == str(checkpoint_path)
    assert summary["train_sample_count"] == context.shape[0]


def test_predict_spectral_residual_baseline_beats_persistence_on_linear_trend(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "checkpoint.npz"
    prediction_path = tmp_path / "prediction"
    context, target = _linear_trend_windows()
    _write_windows(windows_path, context, target)
    train_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        modes=8,
        ridge=1e-8,
        force=False,
    )

    summary = predict_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_path=prediction_path,
        force=False,
    )

    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
    persistence = np.repeat(context[:, -1:, :], repeats=target.shape[1], axis=1)
    model_score = score_prediction_arrays(prediction, target)
    persistence_score = score_prediction_arrays(persistence, target)
    assert prediction.shape == target.shape
    assert model_score["rmse"] < persistence_score["rmse"]
    assert summary["prediction_shape"] == list(target.shape)


def test_train_spectral_residual_baseline_rejects_invalid_modes(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    context, target = _linear_trend_windows(space_points=8)
    _write_windows(windows_path, context, target)

    with pytest.raises(SpectralResidualBaselineError, match="modes"):
        train_spectral_residual_baseline(
            windows_path=windows_path,
            checkpoint_path=tmp_path / "checkpoint.npz",
            modes=0,
            ridge=1e-3,
            force=False,
        )


def test_predict_spectral_residual_baseline_rejects_incompatible_windows(
    tmp_path: Path,
) -> None:
    train_windows = tmp_path / "train_windows.npz"
    predict_windows = tmp_path / "predict_windows.npz"
    checkpoint_path = tmp_path / "checkpoint.npz"
    context, target = _linear_trend_windows(space_points=16)
    _write_windows(train_windows, context, target)
    train_spectral_residual_baseline(
        windows_path=train_windows,
        checkpoint_path=checkpoint_path,
        modes=6,
        ridge=1e-6,
        force=False,
    )
    bad_context, bad_target = _linear_trend_windows(space_points=12)
    _write_windows(predict_windows, bad_context, bad_target)

    with pytest.raises(SpectralResidualBaselineError, match="incompatible"):
        predict_spectral_residual_baseline(
            windows_path=predict_windows,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "prediction.npz",
            force=False,
        )
```

- [ ] **Step 2: Run core tests to verify they fail**

Run:

```bash
uv run --frozen pytest tests/test_lightweight_spectral_baseline.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pde_agent.core.spectral_residual_baseline'`.

- [ ] **Step 3: Implement core model module**

Create `src/pde_agent/core/spectral_residual_baseline.py` with this content:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_TYPE = "spectral_residual_ridge"


class SpectralResidualBaselineError(RuntimeError):
    pass


def train_spectral_residual_baseline(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    modes: int = 16,
    ridge: float = 1e-3,
    force: bool = False,
) -> dict[str, Any]:
    windows = Path(windows_path)
    checkpoint = Path(checkpoint_path)
    context, target = _load_windows(windows)
    modes = _validate_modes(modes, context.shape[2])
    ridge = _validate_ridge(ridge)
    if checkpoint.exists() and not force:
        raise SpectralResidualBaselineError(f"checkpoint path already exists: {checkpoint}")

    features = _feature_matrix(context, modes=modes)
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
    features_scaled = (features - feature_mean) / feature_scale
    design = np.concatenate(
        [np.ones((features_scaled.shape[0], 1), dtype=np.float64), features_scaled],
        axis=1,
    )
    persistence = np.repeat(context[:, -1:, :], repeats=target.shape[1], axis=1)
    residual = target - persistence
    residual_coeff = np.fft.rfft(residual, axis=2)[:, :, :modes]
    targets_real = residual_coeff.real.reshape(context.shape[0], -1)
    targets_imag = residual_coeff.imag.reshape(context.shape[0], -1)
    weights_real = _ridge_solve(design, targets_real, ridge)
    weights_imag = _ridge_solve(design, targets_imag, ridge)
    metadata = {
        "model_type": MODEL_TYPE,
        "modes": modes,
        "ridge": ridge,
        "context_steps": int(context.shape[1]),
        "forecast_steps": int(target.shape[1]),
        "space_points": int(context.shape[2]),
        "feature_dim": int(features.shape[1]),
        "train_sample_count": int(context.shape[0]),
        "windows_path": str(windows),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as handle:
        np.savez(
            handle,
            weights_real=weights_real,
            weights_imag=weights_imag,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            metadata_json=json.dumps(metadata, allow_nan=False, sort_keys=True),
        )
    return {"checkpoint_path": str(checkpoint), **metadata}


def predict_spectral_residual_baseline(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    windows = Path(windows_path)
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    context, target = _load_windows(windows)
    checkpoint_data = _load_checkpoint(checkpoint)
    metadata = checkpoint_data["metadata"]
    _validate_checkpoint_compatible(metadata, context=context, target=target)
    if output.exists() and not force:
        raise SpectralResidualBaselineError(f"output path already exists: {output}")

    modes = int(metadata["modes"])
    features = _feature_matrix(context, modes=modes)
    features_scaled = (features - checkpoint_data["feature_mean"]) / checkpoint_data["feature_scale"]
    design = np.concatenate(
        [np.ones((features_scaled.shape[0], 1), dtype=np.float64), features_scaled],
        axis=1,
    )
    coeff_real = design @ checkpoint_data["weights_real"]
    coeff_imag = design @ checkpoint_data["weights_imag"]
    coeff = coeff_real + 1j * coeff_imag
    coeff = coeff.reshape(context.shape[0], int(metadata["forecast_steps"]), modes)
    full_coeff = np.zeros(
        (context.shape[0], int(metadata["forecast_steps"]), context.shape[2] // 2 + 1),
        dtype=np.complex128,
    )
    full_coeff[:, :, :modes] = coeff
    residual_prediction = np.fft.irfft(full_coeff, n=context.shape[2], axis=2)
    persistence = np.repeat(context[:, -1:, :], repeats=target.shape[1], axis=1)
    prediction = (persistence + residual_prediction).astype(np.float32)
    if not np.all(np.isfinite(prediction)):
        raise SpectralResidualBaselineError("prediction contains non-finite values")
    prediction_metadata = {
        "model_type": MODEL_TYPE,
        "checkpoint_path": str(checkpoint),
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction,
            metadata_json=json.dumps(prediction_metadata, allow_nan=False, sort_keys=True),
        )
    return {
        "prediction_path": str(output),
        "prediction_shape": [int(axis) for axis in prediction.shape],
        **prediction_metadata,
    }


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SpectralResidualBaselineError(f"missing windows path: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in ("context", "target"):
                if key not in archive.files:
                    raise SpectralResidualBaselineError(f"{path} missing key '{key}'")
            context = np.asarray(archive["context"], dtype=np.float64)
            target = np.asarray(archive["target"], dtype=np.float64)
    except SpectralResidualBaselineError:
        raise
    except OSError as exc:
        raise SpectralResidualBaselineError(f"failed to load windows NPZ {path}: {exc}") from exc
    _validate_windows(context, target)
    return context, target


def _validate_windows(context: np.ndarray, target: np.ndarray) -> None:
    for name, array in (("context", context), ("target", target)):
        if array.ndim != 3:
            raise SpectralResidualBaselineError(f"{name} must be rank 3 [N, T, X]")
        if any(axis <= 0 for axis in array.shape):
            raise SpectralResidualBaselineError(f"{name} dimensions must be non-empty")
        if not np.all(np.isfinite(array)):
            raise SpectralResidualBaselineError(f"{name} contains non-finite values")
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise SpectralResidualBaselineError(
            f"context/target shape mismatch: {context.shape} vs {target.shape}"
        )
    if context.shape[1] < 2:
        raise SpectralResidualBaselineError("context requires at least two time steps")


def _validate_modes(modes: int, space_points: int) -> int:
    if not isinstance(modes, int) or isinstance(modes, bool):
        raise SpectralResidualBaselineError("modes must be an integer")
    max_modes = space_points // 2 + 1
    if modes <= 0 or modes > max_modes:
        raise SpectralResidualBaselineError(f"modes must be in [1, {max_modes}]")
    return modes


def _validate_ridge(ridge: float) -> float:
    if not isinstance(ridge, int | float) or not np.isfinite(ridge) or ridge < 0:
        raise SpectralResidualBaselineError("ridge must be a finite non-negative number")
    return float(ridge)


def _feature_matrix(context: np.ndarray, *, modes: int) -> np.ndarray:
    last = context[:, -1, :]
    last_delta = context[:, -1, :] - context[:, -2, :]
    last_coeff = np.fft.rfft(last, axis=1)[:, :modes]
    delta_coeff = np.fft.rfft(last_delta, axis=1)[:, :modes]
    return np.concatenate(
        [
            np.mean(last, axis=1, keepdims=True),
            np.mean(last_delta, axis=1, keepdims=True),
            last_coeff.real,
            last_coeff.imag,
            delta_coeff.real,
            delta_coeff.imag,
        ],
        axis=1,
    )


def _ridge_solve(design: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ targets
    return np.linalg.solve(lhs, rhs)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SpectralResidualBaselineError(f"missing checkpoint path: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"weights_real", "weights_imag", "feature_mean", "feature_scale", "metadata_json"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise SpectralResidualBaselineError(f"checkpoint missing keys: {', '.join(missing)}")
            metadata = json.loads(str(archive["metadata_json"]))
            return {
                "weights_real": np.asarray(archive["weights_real"], dtype=np.float64),
                "weights_imag": np.asarray(archive["weights_imag"], dtype=np.float64),
                "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
                "feature_scale": np.asarray(archive["feature_scale"], dtype=np.float64),
                "metadata": metadata,
            }
    except SpectralResidualBaselineError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise SpectralResidualBaselineError(f"failed to load checkpoint {path}: {exc}") from exc


def _validate_checkpoint_compatible(
    metadata: dict[str, Any],
    *,
    context: np.ndarray,
    target: np.ndarray,
) -> None:
    expected = {
        "model_type": MODEL_TYPE,
        "context_steps": int(context.shape[1]),
        "forecast_steps": int(target.shape[1]),
        "space_points": int(context.shape[2]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SpectralResidualBaselineError(
                f"incompatible checkpoint metadata for {key}: {metadata.get(key)!r} != {value!r}"
            )
```

- [ ] **Step 4: Run core tests**

Run:

```bash
uv run --frozen pytest tests/test_lightweight_spectral_baseline.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/pde_agent/core/spectral_residual_baseline.py tests/test_lightweight_spectral_baseline.py
git commit -m "feat: add spectral residual baseline core"
```

Expected: commit succeeds.

---

### Task 2: Train And Predict CLI Commands

**Files:**
- Create: `src/pde_agent/commands/train_spectral_residual_baseline.py`
- Create: `src/pde_agent/commands/predict_local_model.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_lightweight_spectral_baseline.py`

- [ ] **Step 1: Add failing CLI tests**

Append this content to `tests/test_lightweight_spectral_baseline.py`:

```python
from typer.testing import CliRunner

from pde_agent.cli import app


def test_train_spectral_residual_baseline_cli_writes_payload(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "checkpoint.npz"
    context, target = _linear_trend_windows()
    _write_windows(windows_path, context, target)

    result = CliRunner().invoke(
        app,
        [
            "train-spectral-residual-baseline",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--modes",
            "6",
            "--ridge",
            "1e-6",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "LOCAL_MODEL_TRAINED"
    assert payload["artifacts"]["checkpoint_path"] == str(checkpoint_path)
    assert payload["metrics"]["model_type"] == "spectral_residual_ridge"
    assert checkpoint_path.exists()


def test_predict_local_model_cli_writes_payload(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "checkpoint.npz"
    prediction_path = tmp_path / "prediction.npz"
    context, target = _linear_trend_windows()
    _write_windows(windows_path, context, target)
    train_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        modes=6,
        ridge=1e-6,
        force=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "predict-local-model",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-path",
            str(prediction_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "LOCAL_MODEL_PREDICTED"
    assert payload["artifacts"]["prediction_npz_path"] == str(prediction_path)
    assert payload["metrics"]["forecast_steps"] == target.shape[1]
    assert prediction_path.exists()
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
uv run --frozen pytest tests/test_lightweight_spectral_baseline.py::test_train_spectral_residual_baseline_cli_writes_payload tests/test_lightweight_spectral_baseline.py::test_predict_local_model_cli_writes_payload -q
```

Expected: FAIL because commands are not registered.

- [ ] **Step 3: Implement command wrappers and state hints**

Create `src/pde_agent/commands/train_spectral_residual_baseline.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_baseline import train_spectral_residual_baseline as train_model


def train_spectral_residual_baseline(
    *,
    windows_path: str,
    checkpoint_path: str,
    modes: int,
    ridge: float,
    force: bool,
) -> dict[str, object]:
    summary = train_model(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        modes=modes,
        ridge=ridge,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="train-spectral-residual-baseline",
        state_hint="LOCAL_MODEL_TRAINED",
        artifacts={"checkpoint_path": summary["checkpoint_path"]},
        metrics={
            "model_type": summary["model_type"],
            "modes": summary["modes"],
            "ridge": summary["ridge"],
            "train_sample_count": summary["train_sample_count"],
            "forecast_steps": summary["forecast_steps"],
            "space_points": summary["space_points"],
        },
        next_hint="predict-local-model",
    )
```

Create `src/pde_agent/commands/predict_local_model.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_baseline import predict_spectral_residual_baseline


def predict_local_model(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    summary = predict_spectral_residual_baseline(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="predict-local-model",
        state_hint="LOCAL_MODEL_PREDICTED",
        artifacts={"prediction_npz_path": summary["prediction_path"]},
        metrics={
            "model_type": summary["model_type"],
            "sample_count": summary["sample_count"],
            "forecast_steps": summary["forecast_steps"],
            "space_points": summary["space_points"],
        },
        next_hint="score-local-rollout",
    )
```

Modify `src/pde_agent/core/state_hints.py` to include:

```python
"LOCAL_MODEL_TRAINED",
"LOCAL_MODEL_PREDICTED",
```

- [ ] **Step 4: Register CLI commands**

Modify `src/pde_agent/cli.py` imports:

```python
from pde_agent.commands.predict_local_model import predict_local_model
from pde_agent.commands.train_spectral_residual_baseline import train_spectral_residual_baseline
```

Add commands near other local-model commands:

```python
@app.command("train-spectral-residual-baseline")
def train_spectral_residual_baseline_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    modes: int = typer.Option(16),
    ridge: float = typer.Option(1e-3),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            train_spectral_residual_baseline(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                modes=modes,
                ridge=ridge,
                force=force,
            )
        )
    )


@app.command("predict-local-model")
def predict_local_model_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            predict_local_model(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                force=force,
            )
        )
    )
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
uv run --frozen pytest tests/test_lightweight_spectral_baseline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/pde_agent/commands/train_spectral_residual_baseline.py src/pde_agent/commands/predict_local_model.py src/pde_agent/cli.py src/pde_agent/core/state_hints.py tests/test_lightweight_spectral_baseline.py
git commit -m "feat: expose spectral baseline cli"
```

Expected: commit succeeds.

---

### Task 3: End-To-End Model Diagnostic Chain And README

**Files:**
- Modify: `tests/test_lightweight_spectral_baseline.py`
- Modify: `README.md`

- [ ] **Step 1: Add end-to-end CLI chain test**

Append this content to `tests/test_lightweight_spectral_baseline.py`:

```python
def test_spectral_baseline_cli_chain_reaches_pde_hypothesis(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "checkpoint.npz"
    prediction_path = tmp_path / "prediction.npz"
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    context, target = _linear_trend_windows(sample_count=5)
    _write_windows(windows_path, context, target)
    runner = CliRunner()

    commands = [
        [
            "train-spectral-residual-baseline",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--modes",
            "8",
            "--ridge",
            "1e-8",
        ],
        [
            "predict-local-model",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-path",
            str(prediction_path),
        ],
        [
            "score-local-rollout",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(windows_path),
            "--output-path",
            str(score_path),
        ],
        [
            "diagnose-rollout-error",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(windows_path),
            "--score-path",
            str(score_path),
            "--output-path",
            str(error_path),
        ],
        [
            "check-burgers-physics-local",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(windows_path),
            "--output-path",
            str(physics_path),
        ],
        [
            "select-pde-hypothesis",
            "--score-path",
            str(score_path),
            "--error-diagnosis-path",
            str(error_path),
            "--physics-check-path",
            str(physics_path),
            "--output-path",
            str(hypothesis_path),
        ],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.stdout
        assert json.loads(result.stdout)["status"] == "ok"

    assert checkpoint_path.exists()
    assert prediction_path.exists()
    assert score_path.exists()
    assert error_path.exists()
    assert physics_path.exists()
    assert hypothesis_path.exists()
```

- [ ] **Step 2: Add README model chain section**

Append this section to `README.md`:

````markdown
## Lightweight Spectral Baseline

Train a NumPy-only spectral residual ridge model on local windows:

```bash
uv run --frozen pde-agent train-spectral-residual-baseline \
  --windows-path data/derived/local_windows_task1_val.npz \
  --checkpoint-path runs/spectral_residual_checkpoint.npz \
  --modes 16 \
  --ridge 1e-3

uv run --frozen pde-agent predict-local-model \
  --windows-path data/derived/local_windows_task1_val.npz \
  --checkpoint-path runs/spectral_residual_checkpoint.npz \
  --output-path data/derived/local_spectral_prediction.npz
```

The prediction NPZ can then be passed to `score-local-rollout`, `diagnose-rollout-error`, `check-burgers-physics-local`, and `select-pde-hypothesis`.

This baseline intentionally avoids PyTorch. If neural models become necessary, add PyTorch under a separate optional dependency group and keep this NumPy model as the fast smoke baseline.
````

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run --frozen pytest tests/test_lightweight_spectral_baseline.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full tests and diff check**

Run:

```bash
git diff --check
uv run --frozen --extra data pytest tests -q
```

Expected: `git diff --check` has no output and tests PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add tests/test_lightweight_spectral_baseline.py README.md
git commit -m "docs: add spectral baseline model chain"
```

Expected: commit succeeds.

---

### Task 4: Optional Real Local Window Smoke

**Files:**
- Modify: none unless a bug is found; if a bug is found, stop and report it before editing.

- [ ] **Step 1: Run guarded real-window smoke**

Run:

```bash
if test -f data/derived/local_windows_task1_smoke.npz; then
  uv run --frozen pde-agent train-spectral-residual-baseline \
    --windows-path data/derived/local_windows_task1_smoke.npz \
    --checkpoint-path runs/spectral_residual_smoke_checkpoint.npz \
    --modes 8 \
    --ridge 1e-3 \
    --force
  uv run --frozen pde-agent predict-local-model \
    --windows-path data/derived/local_windows_task1_smoke.npz \
    --checkpoint-path runs/spectral_residual_smoke_checkpoint.npz \
    --output-path data/derived/local_spectral_prediction_smoke.npz \
    --force
  uv run --frozen pde-agent score-local-rollout \
    --prediction-path data/derived/local_spectral_prediction_smoke.npz \
    --target-path data/derived/local_windows_task1_smoke.npz \
    --output-path runs/spectral_residual_smoke_score.json
fi
```

Expected if fixture exists: three commands exit 0 and print JSON payloads with state hints `LOCAL_MODEL_TRAINED`, `LOCAL_MODEL_PREDICTED`, `LOCAL_ROLLOUT_SCORED`. If fixture does not exist, command exits 0 without output.

- [ ] **Step 2: Verify CLI help**

Run:

```bash
uv run --frozen pde-agent train-spectral-residual-baseline --help
uv run --frozen pde-agent predict-local-model --help
```

Expected: both commands print help and exit 0.

- [ ] **Step 3: Verify final worktree and tests**

Run:

```bash
git diff --check
uv run --frozen pytest tests/test_lightweight_spectral_baseline.py -q
uv run --frozen --extra data pytest tests -q
git status --short --branch
```

Expected: no diff-check output, tests PASS, tracked worktree clean.

---

## Final Verification

- [ ] Run focused spectral baseline tests:

```bash
uv run --frozen pytest tests/test_lightweight_spectral_baseline.py -q
```

Expected: PASS.

- [ ] Run full suite:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected: PASS.

- [ ] Verify command availability:

```bash
uv run --frozen pde-agent train-spectral-residual-baseline --help
uv run --frozen pde-agent predict-local-model --help
```

Expected: both commands exit 0.

- [ ] Inspect worktree:

```bash
git status --short --branch
```

Expected: clean tracked worktree on `master`; ignored generated data may exist locally.

## Spec Coverage

- NumPy-only trainable spectral residual model is implemented by Task 1.
- Train and predict CLI commands are implemented by Task 2.
- Diagnostic-chain integration and README documentation are implemented by Task 3.
- Optional real-window smoke is handled by Task 4.
- PyTorch remains deferred and is not added to dependencies.

