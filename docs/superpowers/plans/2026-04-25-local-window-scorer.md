# Local Window Builder And Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local validation layer that slices real PDEBench Burgers HDF5 data into reproducible short rollout windows and scores model predictions with competition-shaped metrics.

**Architecture:** Add two focused core modules: one creates canonical NPZ windows from HDF5 data, and one scores NPZ predictions against those windows. Add thin Typer CLI commands so Hermes can invoke the tools through the existing `pde-agent` command surface without learning internal Python APIs.

**Tech Stack:** Python 3.12+, NumPy, optional `h5py` via `--extra data`, Typer, pytest, existing `success_payload` JSON contract.

---

## File Structure

- Create `src/pde_agent/core/local_windows.py`: validates HDF5 inputs, applies sample/time/space slicing, writes canonical NPZ windows, and returns a summary dictionary.
- Create `src/pde_agent/commands/build_local_windows.py`: wraps the core window builder in a machine-readable success payload.
- Create `src/pde_agent/core/local_scoring.py`: loads prediction/target arrays, validates shape, computes RMSE, normalized error, segmented rel-MSE scores, and writes optional JSON.
- Create `src/pde_agent/commands/score_local_rollout.py`: wraps the scorer in a machine-readable success payload.
- Modify `src/pde_agent/cli.py`: register `build-local-windows` and `score-local-rollout`.
- Create `tests/test_local_windows.py`: unit and CLI tests for HDF5 window construction.
- Create `tests/test_local_scoring.py`: unit and CLI tests for scoring behavior.
- Modify `src/pde_agent/core/state_hints.py`: include new state hints for the two commands.

## Design Rules

- Do not use the full 7.7GB PDEBench file in tests; all tests create tiny temporary HDF5/NPZ files.
- Do not change the existing diagnostic `eval-rollout` behavior in this plan.
- Do not add model training, FNO loading, or final submission generation here.
- Keep generated `.npz` and score JSON outputs under user-provided paths; `.gitignore` already ignores `data/**/*.npz` and `runs/`.
- Store `metadata_json` as a scalar JSON string in NPZ to match existing fixture conventions.
- Use `float32` for saved arrays to keep derived data compact.

---

### Task 1: Core Local Window Builder

**Files:**
- Create: `src/pde_agent/core/local_windows.py`
- Test: `tests/test_local_windows.py`

- [ ] **Step 1: Write failing tests for HDF5 window construction**

Create `tests/test_local_windows.py` with this content:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.core.local_windows import LocalWindowError, build_local_windows


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)


def test_build_local_windows_slices_reduced_hdf5(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    raw = np.arange(3 * 12 * 8, dtype=np.float32).reshape(3, 12, 8)
    _write_hdf5(source, raw)
    output = tmp_path / "windows.npz"

    summary = build_local_windows(
        source_path=source,
        output_path=output,
        dataset_key="tensor",
        start_index=1,
        sample_limit=2,
        context_steps=2,
        forecast_steps=3,
        time_reduction=2,
        space_reduction=2,
        force=False,
    )

    reduced = raw[1:3, ::2, ::2]
    with np.load(output, allow_pickle=False) as archive:
        assert np.array_equal(archive["context"], reduced[:, :2, :].astype(np.float32))
        assert np.array_equal(archive["target"], reduced[:, 2:5, :].astype(np.float32))
        assert np.array_equal(
            archive["full_reference"], reduced[:, :5, :].astype(np.float32)
        )
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["source_path"] == str(source)
    assert metadata["dataset_key"] == "tensor"
    assert metadata["raw_shape"] == [3, 12, 8]
    assert metadata["reduced_shape"] == [2, 6, 4]
    assert metadata["start_index"] == 1
    assert metadata["sample_limit"] == 2
    assert metadata["context_steps"] == 2
    assert metadata["forecast_steps"] == 3
    assert summary["context_shape"] == [2, 2, 4]
    assert summary["target_shape"] == [2, 3, 4]
    assert summary["full_reference_shape"] == [2, 5, 4]


def test_build_local_windows_refuses_overwrite_without_force(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((1, 8, 4), dtype=np.float32))
    output = tmp_path / "windows.npz"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(LocalWindowError, match="already exists"):
        build_local_windows(
            source_path=source,
            output_path=output,
            dataset_key="tensor",
            start_index=0,
            sample_limit=1,
            context_steps=2,
            forecast_steps=2,
            time_reduction=1,
            space_reduction=1,
            force=False,
        )


def test_build_local_windows_rejects_short_reduced_time(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((1, 9, 4), dtype=np.float32))
    output = tmp_path / "windows.npz"

    with pytest.raises(LocalWindowError, match="reduced time dimension"):
        build_local_windows(
            source_path=source,
            output_path=output,
            dataset_key="tensor",
            start_index=0,
            sample_limit=1,
            context_steps=3,
            forecast_steps=3,
            time_reduction=2,
            space_reduction=1,
            force=False,
        )


def test_build_local_windows_rejects_missing_dataset_key(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((1, 8, 4), dtype=np.float32), key="other")

    with pytest.raises(LocalWindowError, match="missing dataset key"):
        build_local_windows(
            source_path=source,
            output_path=tmp_path / "windows.npz",
            dataset_key="tensor",
            start_index=0,
            sample_limit=1,
            context_steps=2,
            forecast_steps=2,
            time_reduction=1,
            space_reduction=1,
            force=False,
        )
```

- [ ] **Step 2: Run tests to verify they fail because the module does not exist**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_windows.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pde_agent.core.local_windows'`.

- [ ] **Step 3: Implement the core window builder**

Create `src/pde_agent/core/local_windows.py` with this content:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class LocalWindowError(RuntimeError):
    pass


def build_local_windows(
    *,
    source_path: str | Path,
    output_path: str | Path,
    dataset_key: str,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(source_path)
    output = Path(output_path)
    _validate_positive_int("start_index", start_index, allow_zero=True)
    if sample_limit is not None:
        _validate_positive_int("sample_limit", sample_limit, allow_zero=False)
    _validate_positive_int("context_steps", context_steps, allow_zero=False)
    _validate_positive_int("forecast_steps", forecast_steps, allow_zero=False)
    _validate_positive_int("time_reduction", time_reduction, allow_zero=False)
    _validate_positive_int("space_reduction", space_reduction, allow_zero=False)
    if not dataset_key:
        raise LocalWindowError("dataset_key must be a non-empty string")
    if not source.exists():
        raise LocalWindowError(f"source HDF5 file does not exist: {source}")
    if output.exists() and not force:
        raise LocalWindowError(f"output file already exists: {output}")

    h5py = _import_h5py()
    with h5py.File(source, "r") as handle:
        if dataset_key not in handle:
            available = ", ".join(sorted(handle.keys()))
            raise LocalWindowError(
                f"missing dataset key {dataset_key!r}; available keys: {available}"
            )
        dataset = handle[dataset_key]
        raw_shape = tuple(int(axis) for axis in dataset.shape)
        dtype = np.dtype(dataset.dtype)
        if len(raw_shape) != 3:
            raise LocalWindowError(f"dataset {dataset_key!r} must be rank 3")
        if not np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.complexfloating
        ):
            raise LocalWindowError(f"dataset {dataset_key!r} must be real-valued")
        stop_index = raw_shape[0] if sample_limit is None else start_index + sample_limit
        if start_index >= raw_shape[0]:
            raise LocalWindowError(
                f"start_index {start_index} is outside sample count {raw_shape[0]}"
            )
        stop_index = min(stop_index, raw_shape[0])
        if stop_index <= start_index:
            raise LocalWindowError("sample slice is empty")
        reduced_time = _reduced_length(raw_shape[1], time_reduction)
        reduced_space = _reduced_length(raw_shape[2], space_reduction)
        required_time = context_steps + forecast_steps
        if reduced_time < required_time:
            raise LocalWindowError(
                "reduced time dimension is shorter than context_steps + forecast_steps"
            )
        reduced = np.asarray(
            dataset[start_index:stop_index, ::time_reduction, ::space_reduction],
            dtype=np.float32,
        )

    full_reference = reduced[:, : context_steps + forecast_steps, :]
    context = full_reference[:, :context_steps, :]
    target = full_reference[:, context_steps : context_steps + forecast_steps, :]
    metadata = {
        "source_path": str(source),
        "dataset_key": dataset_key,
        "raw_shape": list(raw_shape),
        "raw_dtype": str(dtype),
        "reduced_shape": [int(reduced.shape[0]), int(reduced_time), int(reduced_space)],
        "start_index": start_index,
        "sample_limit": sample_limit,
        "actual_sample_count": int(context.shape[0]),
        "context_steps": context_steps,
        "forecast_steps": forecast_steps,
        "time_reduction": time_reduction,
        "space_reduction": space_reduction,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        context=context.astype(np.float32, copy=False),
        target=target.astype(np.float32, copy=False),
        full_reference=full_reference.astype(np.float32, copy=False),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return {
        "output_path": str(output),
        "context_shape": list(context.shape),
        "target_shape": list(target.shape),
        "full_reference_shape": list(full_reference.shape),
        "metadata": metadata,
    }


def _validate_positive_int(name: str, value: int, *, allow_zero: bool) -> None:
    if not isinstance(value, int):
        raise LocalWindowError(f"{name} must be an integer")
    if allow_zero:
        if value < 0:
            raise LocalWindowError(f"{name} must be >= 0")
    elif value <= 0:
        raise LocalWindowError(f"{name} must be > 0")


def _reduced_length(length: int, stride: int) -> int:
    return ((length - 1) // stride) + 1


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise LocalWindowError(
            "h5py is required for build-local-windows; install with `uv sync --extra data`"
        ) from exc
    return h5py
```

- [ ] **Step 4: Run window builder tests**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_windows.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/pde_agent/core/local_windows.py tests/test_local_windows.py
git commit -m "feat: add local hdf5 window builder"
```

Expected: commit succeeds.

---

### Task 2: `build-local-windows` CLI

**Files:**
- Create: `src/pde_agent/commands/build_local_windows.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_local_windows.py`

- [ ] **Step 1: Add failing CLI test**

Append this test to `tests/test_local_windows.py`:

```python
from typer.testing import CliRunner

from pde_agent.cli import app


def test_build_local_windows_cli_writes_json_payload(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((2, 10, 8), dtype=np.float32))
    output = tmp_path / "derived" / "windows.npz"

    result = CliRunner().invoke(
        app,
        [
            "build-local-windows",
            "--source-path",
            str(source),
            "--output-path",
            str(output),
            "--dataset-key",
            "tensor",
            "--start-index",
            "0",
            "--sample-limit",
            "2",
            "--context-steps",
            "2",
            "--forecast-steps",
            "3",
            "--time-reduction",
            "2",
            "--space-reduction",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "build-local-windows"
    assert payload["state_hint"] == "LOCAL_WINDOWS_BUILT"
    assert payload["artifacts"]["windows_npz_path"] == str(output)
    assert payload["metrics"]["sample_count"] == 2
    assert payload["metrics"]["context_steps"] == 2
    assert payload["metrics"]["forecast_steps"] == 3
    assert payload["metrics"]["space_points"] == 4
    assert output.exists()
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_windows.py::test_build_local_windows_cli_writes_json_payload -q
```

Expected: FAIL because `build-local-windows` is not registered.

- [ ] **Step 3: Add command wrapper**

Create `src/pde_agent/commands/build_local_windows.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_windows import build_local_windows as build_windows


def build_local_windows(
    *,
    source_path: str,
    output_path: str,
    dataset_key: str,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    force: bool,
) -> dict[str, object]:
    summary = build_windows(
        source_path=Path(source_path),
        output_path=Path(output_path),
        dataset_key=dataset_key,
        start_index=start_index,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        force=force,
    )
    metadata = summary["metadata"]
    return success_payload(
        run_id="local",
        command="build-local-windows",
        state_hint="LOCAL_WINDOWS_BUILT",
        artifacts={"windows_npz_path": summary["output_path"]},
        metrics={
            "sample_count": metadata["actual_sample_count"],
            "context_steps": metadata["context_steps"],
            "forecast_steps": metadata["forecast_steps"],
            "space_points": summary["target_shape"][2],
        },
        next_hint="score-local-rollout once a prediction NPZ is available",
    )
```

- [ ] **Step 4: Register CLI command and state hint**

Modify `src/pde_agent/core/state_hints.py` so `ALLOWED_STATE_HINTS` becomes:

```python
ALLOWED_STATE_HINTS = {
    "TASK_PARSED",
    "BASELINE_READY",
    "RESULTS_COLLECTED",
    "ERROR_ATTRIBUTED",
    "SUBMISSION_READY",
    "DATA_INSPECTED",
    "LOCAL_WINDOWS_BUILT",
    "LOCAL_ROLLOUT_SCORED",
}
```

Modify `src/pde_agent/cli.py` by adding this import near the other command imports:

```python
from pde_agent.commands.build_local_windows import build_local_windows
```

Add this command before `eval-rollout`:

```python
@app.command("build-local-windows")
def build_local_windows_cmd(
    source_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    start_index: int = typer.Option(0),
    sample_limit: int | None = typer.Option(None),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(31),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_local_windows(
                source_path=source_path,
                output_path=output_path,
                dataset_key=dataset_key,
                start_index=start_index,
                sample_limit=sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                force=force,
            )
        )
    )
```

- [ ] **Step 5: Run window tests**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_windows.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/pde_agent/commands/build_local_windows.py src/pde_agent/cli.py src/pde_agent/core/state_hints.py tests/test_local_windows.py
git commit -m "feat: expose local window builder cli"
```

Expected: commit succeeds.

---

### Task 3: Core Local Rollout Scorer

**Files:**
- Create: `src/pde_agent/core/local_scoring.py`
- Test: `tests/test_local_scoring.py`

- [ ] **Step 1: Write failing scorer tests**

Create `tests/test_local_scoring.py` with this content:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.core.local_scoring import (
    LocalScoreError,
    score_prediction_arrays,
    score_prediction_npz,
)


def test_score_prediction_arrays_perfect_prediction_scores_100() -> None:
    target = np.ones((2, 8, 4), dtype=np.float32)
    result = score_prediction_arrays(target.copy(), target)

    assert result["sample_count"] == 2
    assert result["time_steps"] == 8
    assert result["space_points"] == 4
    assert result["rmse"] == 0.0
    assert result["rel_mse"] == 0.0
    assert result["competition_like_score_short"] == pytest.approx(100.0)
    assert result["segments"]["early"]["score"] == pytest.approx(100.0)
    assert result["segments"]["middle"]["score"] == pytest.approx(100.0)
    assert result["segments"]["late"]["score"] == pytest.approx(100.0)


def test_score_prediction_arrays_noisy_prediction_lowers_score() -> None:
    target = np.ones((1, 8, 4), dtype=np.float32)
    prediction = target + 0.5

    result = score_prediction_arrays(prediction, target)

    assert result["rmse"] > 0.0
    assert result["rel_mse"] > 0.0
    assert result["competition_like_score_short"] < 100.0


def test_score_prediction_arrays_rejects_shape_mismatch() -> None:
    with pytest.raises(LocalScoreError, match="shape mismatch"):
        score_prediction_arrays(
            np.ones((1, 8, 4), dtype=np.float32),
            np.ones((1, 7, 4), dtype=np.float32),
        )


def test_score_prediction_npz_loads_prediction_and_target_files(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    output_path = tmp_path / "score.json"
    target = np.ones((2, 8, 4), dtype=np.float32)
    np.savez(prediction_path, prediction=target.copy())
    np.savez(target_path, target=target)

    result = score_prediction_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=output_path,
    )

    assert result["competition_like_score_short"] == pytest.approx(100.0)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["competition_like_score_short"] == pytest.approx(100.0)


def test_score_prediction_npz_accepts_combined_file(tmp_path: Path) -> None:
    combined_path = tmp_path / "combined.npz"
    target = np.ones((1, 8, 4), dtype=np.float32)
    np.savez(combined_path, prediction=target.copy(), target=target)

    result = score_prediction_npz(
        prediction_path=combined_path,
        target_path=combined_path,
        output_path=None,
    )

    assert result["sample_count"] == 1
    assert result["competition_like_score_short"] == pytest.approx(100.0)


def test_score_prediction_npz_rejects_missing_prediction_key(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    np.savez(prediction_path, pred=np.ones((1, 8, 4), dtype=np.float32))
    np.savez(target_path, target=np.ones((1, 8, 4), dtype=np.float32))

    with pytest.raises(LocalScoreError, match="missing key 'prediction'"):
        score_prediction_npz(
            prediction_path=prediction_path,
            target_path=target_path,
            output_path=None,
        )
```

- [ ] **Step 2: Run tests to verify they fail because the module does not exist**

Run:

```bash
uv run --frozen pytest tests/test_local_scoring.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pde_agent.core.local_scoring'`.

- [ ] **Step 3: Implement scorer core**

Create `src/pde_agent/core/local_scoring.py` with this content:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class LocalScoreError(RuntimeError):
    pass


def score_prediction_npz(
    *,
    prediction_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    prediction_file = Path(prediction_path)
    target_file = Path(target_path)
    prediction = _load_npz_array(prediction_file, "prediction")
    target = _load_npz_array(target_file, "target")
    result = score_prediction_arrays(prediction, target)
    result["prediction_path"] = str(prediction_file)
    result["target_path"] = str(target_file)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        result["score_path"] = str(output)
    return result


def score_prediction_arrays(
    prediction: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]],
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise LocalScoreError(
            f"shape mismatch: prediction shape {pred.shape} != target shape {truth.shape}"
        )
    if pred.ndim != 3:
        raise LocalScoreError("prediction and target must have shape [N, T, X]")
    if pred.shape[0] <= 0 or pred.shape[1] <= 0 or pred.shape[2] <= 0:
        raise LocalScoreError("prediction and target dimensions must be non-empty")
    error = pred - truth
    rmse = float(np.sqrt(np.mean(error**2)))
    target_rms = float(np.sqrt(np.mean(truth**2)))
    nrmse = float(rmse / (target_rms + 1e-12))
    rel_by_time = _relative_mse_by_sample_time(error, truth)
    rel_capped = np.minimum(rel_by_time, 5.0)
    segments = _score_segments(error, rel_capped)
    total_score = (
        0.25 * segments["early"]["score"]
        + 0.25 * segments["middle"]["score"]
        + 0.5 * segments["late"]["score"]
    )
    return {
        "sample_count": int(pred.shape[0]),
        "time_steps": int(pred.shape[1]),
        "space_points": int(pred.shape[2]),
        "rmse": rmse,
        "nrmse": nrmse,
        "rel_mse": float(np.mean(rel_by_time)),
        "rel_mse_capped": float(np.mean(rel_capped)),
        "competition_like_score_short": float(total_score),
        "segments": segments,
        "notes": [
            "Segments are proportional to the available local horizon.",
            "Late segment uses Lorentzian RMSE proxy; Frechet component is reserved for the hidden long-horizon evaluator.",
        ],
    }


def _load_npz_array(path: Path, key: str) -> NDArray[np.floating[Any]]:
    if not path.exists():
        raise LocalScoreError(f"NPZ file does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise LocalScoreError(f"NPZ file {path} missing key {key!r}")
        return np.asarray(archive[key], dtype=np.float64)


def _relative_mse_by_sample_time(
    error: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    numerator = np.sum(error**2, axis=2)
    denominator = np.sum(target**2, axis=2) + 1e-12
    return numerator / denominator


def _score_segments(
    error: NDArray[np.float64],
    rel_capped: NDArray[np.float64],
) -> dict[str, dict[str, float | list[int]]]:
    time_steps = rel_capped.shape[1]
    early_end = max(1, int(round(time_steps * 0.25)))
    middle_end = max(early_end + 1, int(round(time_steps * 0.50)))
    middle_end = min(middle_end, time_steps)
    if middle_end == time_steps and time_steps > 1:
        middle_end = time_steps - 1

    early_rel = float(np.mean(rel_capped[:, :early_end]))
    middle_rel = float(np.mean(rel_capped[:, early_end:middle_end]))
    late_error = error[:, middle_end:, :]
    late_rmse = float(np.sqrt(np.mean(late_error**2))) if late_error.size else 0.0
    return {
        "early": {
            "range": [0, int(early_end)],
            "rel_mse_capped": early_rel,
            "score": float(100.0 * np.exp(-20.0 * early_rel)),
        },
        "middle": {
            "range": [int(early_end), int(middle_end)],
            "rel_mse_capped": middle_rel,
            "score": float(100.0 * np.exp(-10.0 * middle_rel)),
        },
        "late": {
            "range": [int(middle_end), int(time_steps)],
            "rmse": late_rmse,
            "score": float(100.0 / (1.0 + 10.0 * late_rmse)),
        },
    }
```

- [ ] **Step 4: Run scorer tests**

Run:

```bash
uv run --frozen pytest tests/test_local_scoring.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add src/pde_agent/core/local_scoring.py tests/test_local_scoring.py
git commit -m "feat: add local rollout scorer"
```

Expected: commit succeeds.

---

### Task 4: `score-local-rollout` CLI

**Files:**
- Create: `src/pde_agent/commands/score_local_rollout.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_local_scoring.py`

- [ ] **Step 1: Add failing CLI test**

Append this test to `tests/test_local_scoring.py`:

```python
from typer.testing import CliRunner

from pde_agent.cli import app


def test_score_local_rollout_cli_writes_json_payload(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    output_path = tmp_path / "metrics" / "score.json"
    target = np.ones((2, 8, 4), dtype=np.float32)
    np.savez(prediction_path, prediction=target.copy())
    np.savez(target_path, target=target)

    result = CliRunner().invoke(
        app,
        [
            "score-local-rollout",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(target_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "score-local-rollout"
    assert payload["state_hint"] == "LOCAL_ROLLOUT_SCORED"
    assert payload["artifacts"]["score_json_path"] == str(output_path)
    assert payload["metrics"]["competition_like_score_short"] == pytest.approx(100.0)
    assert payload["metrics"]["rel_mse"] == pytest.approx(0.0)
    assert payload["metrics"]["rmse"] == pytest.approx(0.0)
    assert output_path.exists()
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run:

```bash
uv run --frozen pytest tests/test_local_scoring.py::test_score_local_rollout_cli_writes_json_payload -q
```

Expected: FAIL because `score-local-rollout` is not registered.

- [ ] **Step 3: Add command wrapper**

Create `src/pde_agent/commands/score_local_rollout.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_scoring import score_prediction_npz


def score_local_rollout(
    *,
    prediction_path: str,
    target_path: str,
    output_path: str,
) -> dict[str, object]:
    score = score_prediction_npz(
        prediction_path=Path(prediction_path),
        target_path=Path(target_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="score-local-rollout",
        state_hint="LOCAL_ROLLOUT_SCORED",
        artifacts={"score_json_path": str(output_path)},
        metrics={
            "competition_like_score_short": score["competition_like_score_short"],
            "rel_mse": score["rel_mse"],
            "rmse": score["rmse"],
        },
        next_hint="use this score to compare local model variants before hidden long-horizon submission",
    )
```

- [ ] **Step 4: Register CLI command**

Modify `src/pde_agent/cli.py` by adding this import near the other command imports:

```python
from pde_agent.commands.score_local_rollout import score_local_rollout
```

Add this command before `eval-rollout`:

```python
@app.command("score-local-rollout")
def score_local_rollout_cmd(
    prediction_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            score_local_rollout(
                prediction_path=prediction_path,
                target_path=target_path,
                output_path=output_path,
            )
        )
    )
```

- [ ] **Step 5: Run scorer tests**

Run:

```bash
uv run --frozen pytest tests/test_local_scoring.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add src/pde_agent/commands/score_local_rollout.py src/pde_agent/cli.py tests/test_local_scoring.py
git commit -m "feat: expose local rollout scorer cli"
```

Expected: commit succeeds.

---

### Task 5: End-To-End Smoke And Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README usage section**

Append this section to `README.md`:

````markdown
## Local PDEBench Window Scoring

Create a short local validation window from the public Burgers HDF5 file:

```bash
uv run --frozen --extra data pde-agent build-local-windows \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-path data/derived/local_windows_task1_val.npz \
  --dataset-key tensor \
  --start-index 0 \
  --sample-limit 128 \
  --context-steps 10 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4
```

Score a prediction NPZ containing a `prediction` array with shape `[N, T, X]`:

```bash
uv run --frozen pde-agent score-local-rollout \
  --prediction-path data/derived/local_predictions_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_score.json
```

This score is a local proxy for short-horizon model iteration. It is not the hidden 190-step competition evaluator.
````

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_windows.py tests/test_local_scoring.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected: PASS.

- [ ] **Step 4: Run real HDF5 command smoke if the downloaded data exists**

Run:

```bash
test -f data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 && \
uv run --frozen --extra data pde-agent build-local-windows \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-path data/derived/local_windows_task1_smoke.npz \
  --dataset-key tensor \
  --start-index 0 \
  --sample-limit 4 \
  --context-steps 10 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4 \
  --force
```

Expected: command prints JSON with `"state_hint": "LOCAL_WINDOWS_BUILT"` and `data/derived/local_windows_task1_smoke.npz` exists. If the data file is absent, the `test -f` guard exits without running the smoke.

- [ ] **Step 5: Validate README command with a synthetic perfect prediction**

Run:

```bash
uv run --frozen python - <<'PY'
import numpy as np
from pathlib import Path

windows = Path("data/derived/local_windows_task1_smoke.npz")
if not windows.exists():
    raise SystemExit(0)
with np.load(windows, allow_pickle=False) as archive:
    target = archive["target"]
prediction_path = Path("data/derived/local_predictions_task1_smoke.npz")
prediction_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(prediction_path, prediction=target)
PY

test -f data/derived/local_predictions_task1_smoke.npz && \
uv run --frozen pde-agent score-local-rollout \
  --prediction-path data/derived/local_predictions_task1_smoke.npz \
  --target-path data/derived/local_windows_task1_smoke.npz \
  --output-path runs/local_score_smoke.json
```

Expected: command prints JSON with `"state_hint": "LOCAL_ROLLOUT_SCORED"` and score near `100.0`.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add README.md
git commit -m "docs: document local window scoring"
```

Expected: commit succeeds.

---

## Final Verification

- [ ] Run formatting check:

```bash
git diff --check
```

Expected: no output.

- [ ] Run focused tests:

```bash
uv run --frozen --extra data pytest tests/test_local_windows.py tests/test_local_scoring.py -q
```

Expected: PASS.

- [ ] Run full tests:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected: PASS.

- [ ] Inspect git status:

```bash
git status --short
```

Expected: no uncommitted source, test, or documentation changes. Ignored generated data under `data/derived/` and `runs/` may exist locally.

## Spec Coverage

- `build-local-windows` is covered by Task 1 and Task 2.
- `score-local-rollout` is covered by Task 3 and Task 4.
- HDF5 shape/key/time validation is covered by Task 1.
- NPZ prediction/target shape validation is covered by Task 3.
- Hermes-readable JSON payloads are covered by Task 2 and Task 4.
- Local real-data smoke usage is covered by Task 5.
