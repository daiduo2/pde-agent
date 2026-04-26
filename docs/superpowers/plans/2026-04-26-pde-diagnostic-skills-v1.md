# PDE Diagnostic Skills V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an executable PDE diagnostic loop that turns local Burgers windows into a baseline prediction, score, error diagnosis, physics diagnosis, and evidence-backed next hypothesis.

**Architecture:** Keep the existing CLI-first harness pattern. Add small core modules for each science operation and thin command wrappers that emit stable JSON payloads for Hermes/Kimi. Tests use tiny NPZ fixtures; real HDF5 remains optional smoke only.

**Tech Stack:** Python 3.12+, NumPy, Typer, pytest, existing `success_payload` JSON contract, existing `local_scoring` and `local_windows` artifacts.

---

## File Structure

- Create `src/pde_agent/core/baseline_prediction.py`: persistence baseline generator from local window NPZ.
- Create `src/pde_agent/commands/generate_baseline_prediction.py`: CLI wrapper for baseline generation.
- Create `src/pde_agent/core/rollout_error_diagnosis.py`: deterministic prediction-vs-target error pattern diagnosis.
- Create `src/pde_agent/commands/diagnose_rollout_error.py`: CLI wrapper for rollout diagnosis.
- Create `src/pde_agent/core/burgers_physics_local.py`: finite-difference Burgers residual and stability diagnostics.
- Create `src/pde_agent/commands/check_burgers_physics_local.py`: CLI wrapper for physics diagnosis.
- Create `src/pde_agent/core/pde_hypothesis.py`: evidence-to-hypothesis mapping.
- Create `src/pde_agent/commands/select_pde_hypothesis.py`: CLI wrapper for hypothesis selection.
- Modify `src/pde_agent/cli.py`: register four new commands.
- Modify `src/pde_agent/core/state_hints.py`: add `BASELINE_PREDICTION_READY`, `ROLLOUT_ERROR_DIAGNOSED`, `BURGERS_PHYSICS_CHECKED`, `PDE_HYPOTHESIS_SELECTED`.
- Create `tests/test_pde_diagnostic_skills_v1.py`: unit and CLI coverage for the full new loop.
- Create command templates under `src/pde_agent/templates/command/`.
- Modify `README.md`: add compact end-to-end local diagnostic chain.

## Design Rules

- Do not modify existing synthetic diagnostic commands.
- Do not add training, checkpoint loading, or submission generation.
- Do not require real HDF5 in unit tests.
- All JSON writes must use `allow_nan=False` after finite-value validation.
- All NPZ reads must use `allow_pickle=False`.
- All generated command payloads must use `success_payload`.
- Command wrappers may let domain exceptions bubble, matching the current CLI style.

---

### Task 1: Persistence Baseline Prediction

**Files:**
- Create: `src/pde_agent/core/baseline_prediction.py`
- Create: `src/pde_agent/commands/generate_baseline_prediction.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Write failing baseline tests**

Create `tests/test_pde_diagnostic_skills_v1.py` with this content:

```python
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pde_agent.cli import app
from pde_agent.core.baseline_prediction import (
    BaselinePredictionError,
    generate_baseline_prediction,
)


def _write_windows(path: Path, context: np.ndarray, target: np.ndarray) -> None:
    full_reference = np.concatenate([context, target], axis=1)
    np.savez(
        path,
        context=context.astype(np.float32),
        target=target.astype(np.float32),
        full_reference=full_reference.astype(np.float32),
        metadata_json=json.dumps({"fixture": True}),
    )


def _write_prediction(path: Path, prediction: np.ndarray) -> None:
    np.savez(path, prediction=prediction.astype(np.float32))


def test_generate_baseline_prediction_repeats_last_context_frame(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    context = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    target = np.zeros((2, 5, 4), dtype=np.float32)
    _write_windows(windows_path, context, target)

    summary = generate_baseline_prediction(
        windows_path=windows_path,
        output_path=output_path,
        method="persistence",
        force=False,
    )

    expected = np.repeat(context[:, -1:, :], repeats=5, axis=1)
    with np.load(output_path, allow_pickle=False) as archive:
        assert np.array_equal(archive["prediction"], expected)
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["method"] == "persistence"
    assert summary["prediction_shape"] == [2, 5, 4]
    assert summary["method"] == "persistence"


def test_generate_baseline_prediction_refuses_overwrite_without_force(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    _write_windows(
        windows_path,
        np.ones((1, 2, 3), dtype=np.float32),
        np.ones((1, 4, 3), dtype=np.float32),
    )
    output_path.write_text("exists", encoding="utf-8")

    with pytest.raises(BaselinePredictionError, match="already exists"):
        generate_baseline_prediction(
            windows_path=windows_path,
            output_path=output_path,
            method="persistence",
            force=False,
        )


def test_generate_baseline_prediction_cli_writes_payload(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    _write_windows(
        windows_path,
        np.ones((2, 3, 4), dtype=np.float32),
        np.zeros((2, 5, 4), dtype=np.float32),
    )

    result = CliRunner().invoke(
        app,
        [
            "generate-baseline-prediction",
            "--windows-path",
            str(windows_path),
            "--output-path",
            str(output_path),
            "--method",
            "persistence",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "generate-baseline-prediction"
    assert payload["state_hint"] == "BASELINE_PREDICTION_READY"
    assert payload["artifacts"]["prediction_npz_path"] == str(output_path)
    assert payload["metrics"] == {
        "method": "persistence",
        "sample_count": 2,
        "forecast_steps": 5,
        "space_points": 4,
    }
    assert output_path.exists()
```

- [ ] **Step 2: Run baseline tests to verify they fail**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pde_agent.core.baseline_prediction'`.

- [ ] **Step 3: Implement baseline core**

Create `src/pde_agent/core/baseline_prediction.py` with this content:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class BaselinePredictionError(RuntimeError):
    pass


def generate_baseline_prediction(
    *,
    windows_path: str | Path,
    output_path: str | Path,
    method: str = "persistence",
    force: bool = False,
) -> dict[str, Any]:
    windows = Path(windows_path)
    output = Path(output_path)
    if method != "persistence":
        raise BaselinePredictionError(f"unknown baseline method: {method}")
    if not windows.exists():
        raise BaselinePredictionError(f"missing windows path: {windows}")
    if output.exists() and not force:
        raise BaselinePredictionError(f"output path already exists: {output}")

    context, target = _load_windows(windows)
    prediction = np.repeat(context[:, -1:, :], repeats=target.shape[1], axis=1)
    prediction = prediction.astype(np.float32, copy=False)
    metadata = {
        "method": method,
        "windows_path": str(windows),
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        prediction=prediction,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return {
        "prediction_path": str(output),
        "prediction_shape": [int(axis) for axis in prediction.shape],
        **metadata,
    }


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in ("context", "target"):
                if key not in archive.files:
                    raise BaselinePredictionError(f"{path} missing key '{key}'")
            context = np.asarray(archive["context"], dtype=np.float64)
            target = np.asarray(archive["target"], dtype=np.float64)
    except BaselinePredictionError:
        raise
    except OSError as exc:
        raise BaselinePredictionError(f"failed to load windows NPZ {path}: {exc}") from exc
    _validate_rank3("context", context)
    _validate_rank3("target", target)
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise BaselinePredictionError(
            f"context/target shape mismatch: {context.shape} vs {target.shape}"
        )
    if not np.all(np.isfinite(context)) or not np.all(np.isfinite(target)):
        raise BaselinePredictionError("windows arrays contain non-finite values")
    return context, target


def _validate_rank3(name: str, array: np.ndarray) -> None:
    if array.ndim != 3:
        raise BaselinePredictionError(f"{name} must be rank 3 [N, T, X]")
    if any(axis <= 0 for axis in array.shape):
        raise BaselinePredictionError(f"{name} dimensions must be non-empty")
```

- [ ] **Step 4: Implement baseline command and CLI registration**

Create `src/pde_agent/commands/generate_baseline_prediction.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.baseline_prediction import (
    generate_baseline_prediction as generate_prediction,
)
from pde_agent.core.json_output import success_payload


def generate_baseline_prediction(
    *,
    windows_path: str,
    output_path: str,
    method: str,
    force: bool,
) -> dict[str, object]:
    summary = generate_prediction(
        windows_path=Path(windows_path),
        output_path=Path(output_path),
        method=method,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="generate-baseline-prediction",
        state_hint="BASELINE_PREDICTION_READY",
        artifacts={"prediction_npz_path": summary["prediction_path"]},
        metrics={
            "method": summary["method"],
            "sample_count": summary["sample_count"],
            "forecast_steps": summary["forecast_steps"],
            "space_points": summary["space_points"],
        },
        next_hint="score-local-rollout",
    )
```

Modify `src/pde_agent/core/state_hints.py` so `ALLOWED_STATE_HINTS` includes these additional strings while preserving all existing hints:

```python
"BASELINE_PREDICTION_READY",
"ROLLOUT_ERROR_DIAGNOSED",
"BURGERS_PHYSICS_CHECKED",
"PDE_HYPOTHESIS_SELECTED",
```

Modify `src/pde_agent/cli.py`:

```python
from pde_agent.commands.generate_baseline_prediction import generate_baseline_prediction
```

Add this command near the local window commands:

```python
@app.command("generate-baseline-prediction")
def generate_baseline_prediction_cmd(
    windows_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    method: str = typer.Option("persistence"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            generate_baseline_prediction(
                windows_path=windows_path,
                output_path=output_path,
                method=method,
                force=force,
            )
        )
    )
```

- [ ] **Step 5: Run baseline tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/pde_agent/core/baseline_prediction.py src/pde_agent/commands/generate_baseline_prediction.py src/pde_agent/cli.py src/pde_agent/core/state_hints.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: add persistence baseline prediction"
```

Expected: commit succeeds.

---

### Task 2: Rollout Error Diagnosis

**Files:**
- Create: `src/pde_agent/core/rollout_error_diagnosis.py`
- Create: `src/pde_agent/commands/diagnose_rollout_error.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Add failing rollout diagnosis tests**

Append this content to `tests/test_pde_diagnostic_skills_v1.py`:

```python
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz


def _write_score(path: Path, score: float = 80.0, rmse: float = 0.1) -> None:
    path.write_text(
        json.dumps(
            {
                "competition_like_score_short": score,
                "rmse": rmse,
                "rel_mse": rmse * rmse,
                "segments": {
                    "early": {"rel_mse_capped": rmse * rmse, "score": score},
                    "middle": {"rel_mse_capped": rmse * rmse, "score": score},
                    "late": {"rmse": rmse, "score": score},
                },
            }
        ),
        encoding="utf-8",
    )


def test_diagnose_rollout_error_low_error_for_perfect_prediction(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    score_path = tmp_path / "score.json"
    output_path = tmp_path / "diagnosis.json"
    target = np.ones((1, 8, 4), dtype=np.float32)
    _write_prediction(prediction_path, target)
    np.savez(target_path, target=target)
    _write_score(score_path, score=100.0, rmse=0.0)

    diagnosis = diagnose_rollout_error_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        score_path=score_path,
        output_path=output_path,
    )

    assert diagnosis["dominant_error_pattern"] == "low_error"
    assert diagnosis["global_rmse"] == pytest.approx(0.0)
    assert output_path.exists()


def test_diagnose_rollout_error_long_horizon_drift(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    score_path = tmp_path / "score.json"
    target = np.ones((1, 8, 4), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 4:, :] += 1.0
    _write_prediction(prediction_path, prediction)
    np.savez(target_path, target=target)
    _write_score(score_path, score=20.0, rmse=0.5)

    diagnosis = diagnose_rollout_error_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        score_path=score_path,
        output_path=None,
    )

    assert diagnosis["dominant_error_pattern"] == "long_horizon_drift"
    assert diagnosis["late_rmse"] > diagnosis["early_rmse"]


def test_diagnose_rollout_error_overdamped_rollout(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    score_path = tmp_path / "score.json"
    x = np.linspace(0, 4 * np.pi, 16, dtype=np.float32)
    target = np.sin(x)[None, None, :] * np.ones((1, 8, 1), dtype=np.float32)
    prediction = np.zeros_like(target)
    _write_prediction(prediction_path, prediction)
    np.savez(target_path, target=target)
    _write_score(score_path, score=30.0, rmse=0.5)

    diagnosis = diagnose_rollout_error_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        score_path=score_path,
        output_path=None,
    )

    assert diagnosis["dominant_error_pattern"] == "overdamped_rollout"
    assert diagnosis["prediction_total_variation"] < diagnosis["target_total_variation"]


def test_diagnose_rollout_error_cli_writes_payload(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    score_path = tmp_path / "score.json"
    output_path = tmp_path / "diagnostics" / "rollout_error.json"
    target = np.ones((1, 8, 4), dtype=np.float32)
    _write_prediction(prediction_path, target)
    np.savez(target_path, target=target)
    _write_score(score_path, score=100.0, rmse=0.0)

    result = CliRunner().invoke(
        app,
        [
            "diagnose-rollout-error",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(target_path),
            "--score-path",
            str(score_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "ROLLOUT_ERROR_DIAGNOSED"
    assert payload["artifacts"]["rollout_error_diagnosis_path"] == str(output_path)
    assert payload["metrics"]["dominant_error_pattern"] == "low_error"
```

- [ ] **Step 2: Run diagnosis tests to verify they fail**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pde_agent.core.rollout_error_diagnosis'`.

- [ ] **Step 3: Implement rollout diagnosis core**

Create `src/pde_agent/core/rollout_error_diagnosis.py` with this content:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class RolloutDiagnosisError(RuntimeError):
    pass


def diagnose_rollout_error_npz(
    *,
    prediction_path: str | Path,
    target_path: str | Path,
    score_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    prediction = _load_npz_array(Path(prediction_path), "prediction")
    target = _load_npz_array(Path(target_path), "target")
    score = _load_json(Path(score_path))
    diagnosis = diagnose_rollout_error_arrays(
        prediction=prediction,
        target=target,
        score=score,
    )
    diagnosis["prediction_path"] = str(prediction_path)
    diagnosis["target_path"] = str(target_path)
    diagnosis["score_path"] = str(score_path)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        diagnosis["diagnosis_path"] = str(output)
        output.write_text(
            json.dumps(diagnosis, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return diagnosis


def diagnose_rollout_error_arrays(
    *,
    prediction: Any,
    target: Any,
    score: dict[str, Any],
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _validate_arrays(pred, truth)
    error = pred - truth
    squared = error**2
    rmse_by_time = np.sqrt(np.mean(squared, axis=(0, 2)))
    rel_by_time = np.sum(squared, axis=2) / (np.sum(truth**2, axis=2) + 1e-12)
    rel_by_time_mean = np.mean(rel_by_time, axis=0)
    early, middle, late = _segment_slices(pred.shape[1])
    early_rmse = float(np.mean(rmse_by_time[early]))
    middle_rmse = float(np.mean(rmse_by_time[middle]))
    late_rmse = float(np.mean(rmse_by_time[late]))
    early_rel = float(np.mean(rel_by_time_mean[early]))
    middle_rel = float(np.mean(rel_by_time_mean[middle]))
    late_rel = float(np.mean(rel_by_time_mean[late]))
    time_error_slope = float(np.polyfit(np.arange(pred.shape[1]), rmse_by_time, deg=1)[0])
    mean_spatial_error = np.mean(np.abs(error), axis=(0, 1))
    max_spatial_error = float(np.max(mean_spatial_error))
    max_error_time_index = int(np.argmax(rmse_by_time))
    pred_tv = _total_variation(pred)
    target_tv = _total_variation(truth)
    shock_ratio = _shock_region_error_ratio(error, truth)
    sign_change_ratio = _sign_change_ratio(error)
    global_rmse = float(np.sqrt(np.mean(squared)))
    global_rel_mse = float(np.mean(rel_by_time))
    total_score = float(score.get("competition_like_score_short", 0.0))
    pattern, evidence = _classify(
        total_score=total_score,
        global_rmse=global_rmse,
        early_rmse=early_rmse,
        late_rmse=late_rmse,
        pred_tv=pred_tv,
        target_tv=target_tv,
        shock_ratio=shock_ratio,
        sign_change_ratio=sign_change_ratio,
    )
    result = {
        "global_rmse": global_rmse,
        "global_rel_mse": global_rel_mse,
        "early_rel_mse": early_rel,
        "middle_rel_mse": middle_rel,
        "late_rel_mse": late_rel,
        "early_rmse": early_rmse,
        "middle_rmse": middle_rmse,
        "late_rmse": late_rmse,
        "time_error_slope": time_error_slope,
        "max_error_time_index": max_error_time_index,
        "mean_spatial_error": float(np.mean(mean_spatial_error)),
        "max_spatial_error": max_spatial_error,
        "prediction_total_variation": pred_tv,
        "target_total_variation": target_tv,
        "shock_region_error_ratio": shock_ratio,
        "sign_change_ratio": sign_change_ratio,
        "dominant_error_pattern": pattern,
        "evidence": evidence,
    }
    _ensure_finite_json(result)
    return result


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise RolloutDiagnosisError(f"missing path: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise RolloutDiagnosisError(f"{path} missing key '{key}'")
            return np.asarray(archive[key], dtype=np.float64)
    except RolloutDiagnosisError:
        raise
    except OSError as exc:
        raise RolloutDiagnosisError(f"failed to load NPZ file {path}: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RolloutDiagnosisError(f"missing score path: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RolloutDiagnosisError("score JSON must be an object")
    if "competition_like_score_short" not in payload:
        raise RolloutDiagnosisError("score JSON missing competition_like_score_short")
    return payload


def _validate_arrays(pred: np.ndarray, truth: np.ndarray) -> None:
    if pred.shape != truth.shape:
        raise RolloutDiagnosisError(f"shape mismatch: {pred.shape} vs {truth.shape}")
    if pred.ndim != 3:
        raise RolloutDiagnosisError("prediction and target must be rank 3 [N, T, X]")
    if any(axis <= 0 for axis in pred.shape):
        raise RolloutDiagnosisError("prediction and target dimensions must be non-empty")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise RolloutDiagnosisError("prediction and target must be finite")


def _segment_slices(time_steps: int) -> tuple[slice, slice, slice]:
    if time_steps == 1:
        return slice(0, 1), slice(0, 1), slice(0, 1)
    if time_steps == 2:
        return slice(0, 1), slice(0, 1), slice(1, 2)
    early_stop = max(1, time_steps // 4)
    middle_stop = max(early_stop + 1, time_steps // 2)
    return slice(0, early_stop), slice(early_stop, middle_stop), slice(middle_stop, time_steps)


def _total_variation(array: np.ndarray) -> float:
    return float(np.mean(np.sum(np.abs(np.diff(array, axis=2)), axis=2)))


def _shock_region_error_ratio(error: np.ndarray, target: np.ndarray) -> float:
    gradient = np.abs(np.diff(target, axis=2, prepend=target[:, :, :1]))
    threshold = float(np.quantile(gradient, 0.8))
    mask = gradient >= threshold
    if not np.any(mask):
        return 1.0
    shock_error = float(np.mean(np.abs(error)[mask]))
    global_error = float(np.mean(np.abs(error)) + 1e-12)
    return shock_error / global_error


def _sign_change_ratio(error: np.ndarray) -> float:
    signs = np.sign(error)
    changes = np.diff(signs, axis=2) != 0
    return float(np.mean(changes))


def _classify(
    *,
    total_score: float,
    global_rmse: float,
    early_rmse: float,
    late_rmse: float,
    pred_tv: float,
    target_tv: float,
    shock_ratio: float,
    sign_change_ratio: float,
) -> tuple[str, list[str]]:
    evidence = [
        f"score={total_score:.6g}",
        f"global_rmse={global_rmse:.6g}",
        f"early_rmse={early_rmse:.6g}",
        f"late_rmse={late_rmse:.6g}",
        f"prediction_total_variation={pred_tv:.6g}",
        f"target_total_variation={target_tv:.6g}",
        f"shock_region_error_ratio={shock_ratio:.6g}",
        f"sign_change_ratio={sign_change_ratio:.6g}",
    ]
    if total_score >= 99.0 and global_rmse <= 1e-8:
        return "low_error", evidence
    if late_rmse > max(early_rmse * 2.0, early_rmse + 1e-8):
        return "long_horizon_drift", evidence
    if target_tv > 1e-12 and pred_tv / target_tv < 0.5:
        return "overdamped_rollout", evidence
    if sign_change_ratio > 0.45 and global_rmse > 1e-6:
        return "oscillatory_error", evidence
    if shock_ratio > 1.5 and global_rmse > 1e-6:
        return "shock_region_error", evidence
    return "short_horizon_bias", evidence


def _ensure_finite_json(payload: dict[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)
```

- [ ] **Step 4: Implement rollout diagnosis command and CLI**

Create `src/pde_agent/commands/diagnose_rollout_error.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz


def diagnose_rollout_error(
    *,
    prediction_path: str,
    target_path: str,
    score_path: str,
    output_path: str,
) -> dict[str, object]:
    diagnosis = diagnose_rollout_error_npz(
        prediction_path=Path(prediction_path),
        target_path=Path(target_path),
        score_path=Path(score_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="diagnose-rollout-error",
        state_hint="ROLLOUT_ERROR_DIAGNOSED",
        artifacts={"rollout_error_diagnosis_path": str(output_path)},
        metrics={
            "dominant_error_pattern": diagnosis["dominant_error_pattern"],
            "global_rmse": diagnosis["global_rmse"],
            "time_error_slope": diagnosis["time_error_slope"],
        },
        next_hint="check-burgers-physics-local",
    )
```

Modify `src/pde_agent/cli.py`:

```python
from pde_agent.commands.diagnose_rollout_error import diagnose_rollout_error
```

Add:

```python
@app.command("diagnose-rollout-error")
def diagnose_rollout_error_cmd(
    prediction_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    score_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            diagnose_rollout_error(
                prediction_path=prediction_path,
                target_path=target_path,
                score_path=score_path,
                output_path=output_path,
            )
        )
    )
```

- [ ] **Step 5: Run diagnosis tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/pde_agent/core/rollout_error_diagnosis.py src/pde_agent/commands/diagnose_rollout_error.py src/pde_agent/cli.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: add local rollout error diagnosis"
```

Expected: commit succeeds.

---

### Task 3: Local Burgers Physics Check

**Files:**
- Create: `src/pde_agent/core/burgers_physics_local.py`
- Create: `src/pde_agent/commands/check_burgers_physics_local.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Add failing Burgers physics tests**

Append this content to `tests/test_pde_diagnostic_skills_v1.py`:

```python
from pde_agent.core.burgers_physics_local import check_burgers_physics_npz


def test_check_burgers_physics_returns_finite_metrics(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    output_path = tmp_path / "physics.json"
    x = np.linspace(0, 2 * np.pi, 16, dtype=np.float32)
    t = np.linspace(0, 1, 8, dtype=np.float32)
    target = np.sin(x[None, None, :] + t[None, :, None]).astype(np.float32)
    _write_prediction(prediction_path, target.copy())
    np.savez(target_path, target=target)

    result = check_burgers_physics_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=output_path,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["physics_issue"] in {
        "stable_low_error",
        "residual_mismatch",
        "inconclusive",
    }
    assert result["prediction_residual_rmse"] >= 0.0
    assert output_path.exists()


def test_check_burgers_physics_flags_overdissipation(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    x = np.linspace(0, 4 * np.pi, 32, dtype=np.float32)
    target = np.sin(x)[None, None, :] * np.ones((1, 8, 1), dtype=np.float32)
    prediction = np.zeros_like(target)
    _write_prediction(prediction_path, prediction)
    np.savez(target_path, target=target)

    result = check_burgers_physics_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=None,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["physics_issue"] == "overdissipation"
    assert result["total_variation_ratio"] < 0.5


def test_check_burgers_physics_cli_writes_payload(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    output_path = tmp_path / "diagnostics" / "physics.json"
    target = np.ones((1, 8, 4), dtype=np.float32)
    _write_prediction(prediction_path, target)
    np.savez(target_path, target=target)

    result = CliRunner().invoke(
        app,
        [
            "check-burgers-physics-local",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(target_path),
            "--output-path",
            str(output_path),
            "--nu",
            "0.001",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "BURGERS_PHYSICS_CHECKED"
    assert payload["artifacts"]["burgers_physics_check_path"] == str(output_path)
    assert "physics_issue" in payload["metrics"]
```

- [ ] **Step 2: Run physics tests to verify they fail**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pde_agent.core.burgers_physics_local'`.

- [ ] **Step 3: Implement Burgers physics core**

Create `src/pde_agent/core/burgers_physics_local.py` with this content:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class BurgersPhysicsError(RuntimeError):
    pass


def check_burgers_physics_npz(
    *,
    prediction_path: str | Path,
    target_path: str | Path,
    output_path: str | Path | None,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
) -> dict[str, Any]:
    prediction = _load_npz_array(Path(prediction_path), "prediction")
    target = _load_npz_array(Path(target_path), "target")
    result = check_burgers_physics_arrays(
        prediction=prediction,
        target=target,
        nu=nu,
        dt=dt,
        dx=dx,
    )
    result["prediction_path"] = str(prediction_path)
    result["target_path"] = str(target_path)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result["physics_check_path"] = str(output)
        output.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def check_burgers_physics_arrays(
    *,
    prediction: Any,
    target: Any,
    nu: float,
    dt: float,
    dx: float,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _validate_arrays(pred, truth)
    if nu < 0 or dt <= 0 or dx <= 0:
        raise BurgersPhysicsError("nu must be >= 0 and dt/dx must be > 0")

    pred_residual = _burgers_residual(pred, nu=nu, dt=dt, dx=dx)
    target_residual = _burgers_residual(truth, nu=nu, dt=dt, dx=dx)
    pred_residual_rmse = float(np.sqrt(np.mean(pred_residual**2)))
    target_residual_rmse = float(np.sqrt(np.mean(target_residual**2)))
    residual_ratio = pred_residual_rmse / (target_residual_rmse + 1e-12)
    mass_drift = _mass_drift(pred)
    energy_decay_ratio = _energy_decay_ratio(pred, truth)
    total_variation_ratio = _total_variation(pred) / (_total_variation(truth) + 1e-12)
    gradient_error_ratio = _gradient_error_ratio(pred, truth)
    shock_region_error_ratio = _shock_region_error_ratio(pred, truth)
    physics_issue = _classify_physics_issue(
        residual_ratio=residual_ratio,
        mass_drift=mass_drift,
        energy_decay_ratio=energy_decay_ratio,
        total_variation_ratio=total_variation_ratio,
        gradient_error_ratio=gradient_error_ratio,
    )
    result = {
        "nu": float(nu),
        "dt": float(dt),
        "dx": float(dx),
        "prediction_residual_rmse": pred_residual_rmse,
        "target_residual_rmse": target_residual_rmse,
        "residual_ratio": float(residual_ratio),
        "mass_drift": float(mass_drift),
        "energy_decay_ratio": float(energy_decay_ratio),
        "total_variation_ratio": float(total_variation_ratio),
        "gradient_error_ratio": float(gradient_error_ratio),
        "shock_region_error_ratio": float(shock_region_error_ratio),
        "physics_issue": physics_issue,
        "notes": "Finite-difference diagnostics are relative proxies on downsampled local windows.",
    }
    json.dumps(result, allow_nan=False)
    return result


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise BurgersPhysicsError(f"missing path: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise BurgersPhysicsError(f"{path} missing key '{key}'")
            return np.asarray(archive[key], dtype=np.float64)
    except BurgersPhysicsError:
        raise
    except OSError as exc:
        raise BurgersPhysicsError(f"failed to load NPZ file {path}: {exc}") from exc


def _validate_arrays(pred: np.ndarray, truth: np.ndarray) -> None:
    if pred.shape != truth.shape:
        raise BurgersPhysicsError(f"shape mismatch: {pred.shape} vs {truth.shape}")
    if pred.ndim != 3:
        raise BurgersPhysicsError("prediction and target must be rank 3 [N, T, X]")
    if any(axis <= 0 for axis in pred.shape):
        raise BurgersPhysicsError("prediction and target dimensions must be non-empty")
    if pred.shape[1] < 2 or pred.shape[2] < 3:
        raise BurgersPhysicsError("physics check requires at least T>=2 and X>=3")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise BurgersPhysicsError("prediction and target must be finite")


def _burgers_residual(array: np.ndarray, *, nu: float, dt: float, dx: float) -> np.ndarray:
    u = array[:, :-1, :]
    u_next = array[:, 1:, :]
    u_t = (u_next - u) / dt
    u_x = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2.0 * dx)
    u_xx = (np.roll(u, -1, axis=2) - 2.0 * u + np.roll(u, 1, axis=2)) / (dx * dx)
    return u_t + u * u_x - nu * u_xx


def _mass_drift(array: np.ndarray) -> float:
    mass = np.mean(array, axis=2)
    return float(np.mean(np.abs(mass[:, -1] - mass[:, 0])))


def _energy_decay_ratio(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_energy = np.mean(pred[:, -1, :] ** 2)
    target_energy = np.mean(truth[:, -1, :] ** 2)
    return float(pred_energy / (target_energy + 1e-12))


def _total_variation(array: np.ndarray) -> float:
    return float(np.mean(np.sum(np.abs(np.diff(array, axis=2)), axis=2)))


def _gradient_error_ratio(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_grad = np.diff(pred, axis=2)
    truth_grad = np.diff(truth, axis=2)
    return float(np.sqrt(np.mean((pred_grad - truth_grad) ** 2)) / (np.sqrt(np.mean(truth_grad**2)) + 1e-12))


def _shock_region_error_ratio(pred: np.ndarray, truth: np.ndarray) -> float:
    grad = np.abs(np.diff(truth, axis=2, prepend=truth[:, :, :1]))
    threshold = float(np.quantile(grad, 0.8))
    mask = grad >= threshold
    if not np.any(mask):
        return 1.0
    local_error = float(np.mean(np.abs(pred - truth)[mask]))
    global_error = float(np.mean(np.abs(pred - truth)) + 1e-12)
    return local_error / global_error


def _classify_physics_issue(
    *,
    residual_ratio: float,
    mass_drift: float,
    energy_decay_ratio: float,
    total_variation_ratio: float,
    gradient_error_ratio: float,
) -> str:
    if total_variation_ratio < 0.5 or energy_decay_ratio < 0.5:
        return "overdissipation"
    if gradient_error_ratio > 2.0 and total_variation_ratio > 1.5:
        return "nonphysical_oscillation"
    if residual_ratio > 2.0:
        return "residual_mismatch"
    if mass_drift > 0.1:
        return "mass_drift"
    if residual_ratio < 1.25 and gradient_error_ratio < 0.25:
        return "stable_low_error"
    return "inconclusive"
```

- [ ] **Step 4: Implement Burgers physics command and CLI**

Create `src/pde_agent/commands/check_burgers_physics_local.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.burgers_physics_local import check_burgers_physics_npz
from pde_agent.core.json_output import success_payload


def check_burgers_physics_local(
    *,
    prediction_path: str,
    target_path: str,
    output_path: str,
    nu: float,
    dt: float,
    dx: float,
) -> dict[str, object]:
    result = check_burgers_physics_npz(
        prediction_path=Path(prediction_path),
        target_path=Path(target_path),
        output_path=Path(output_path),
        nu=nu,
        dt=dt,
        dx=dx,
    )
    return success_payload(
        run_id="local",
        command="check-burgers-physics-local",
        state_hint="BURGERS_PHYSICS_CHECKED",
        artifacts={"burgers_physics_check_path": str(output_path)},
        metrics={
            "physics_issue": result["physics_issue"],
            "residual_ratio": result["residual_ratio"],
            "total_variation_ratio": result["total_variation_ratio"],
        },
        next_hint="select-pde-hypothesis",
    )
```

Modify `src/pde_agent/cli.py`:

```python
from pde_agent.commands.check_burgers_physics_local import check_burgers_physics_local
```

Add:

```python
@app.command("check-burgers-physics-local")
def check_burgers_physics_local_cmd(
    prediction_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            check_burgers_physics_local(
                prediction_path=prediction_path,
                target_path=target_path,
                output_path=output_path,
                nu=nu,
                dt=dt,
                dx=dx,
            )
        )
    )
```

- [ ] **Step 5: Run physics tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add src/pde_agent/core/burgers_physics_local.py src/pde_agent/commands/check_burgers_physics_local.py src/pde_agent/cli.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: add local burgers physics check"
```

Expected: commit succeeds.

---

### Task 4: PDE Hypothesis Selection

**Files:**
- Create: `src/pde_agent/core/pde_hypothesis.py`
- Create: `src/pde_agent/commands/select_pde_hypothesis.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Add failing hypothesis tests**

Append this content to `tests/test_pde_diagnostic_skills_v1.py`:

```python
from pde_agent.core.pde_hypothesis import select_pde_hypothesis_from_files


def _write_error_diagnosis(path: Path, pattern: str) -> None:
    path.write_text(
        json.dumps(
            {
                "dominant_error_pattern": pattern,
                "evidence": [f"dominant_error_pattern={pattern}"],
                "global_rmse": 0.5,
            }
        ),
        encoding="utf-8",
    )


def _write_physics_check(path: Path, issue: str) -> None:
    path.write_text(
        json.dumps(
            {
                "physics_issue": issue,
                "total_variation_ratio": 0.25,
                "residual_ratio": 1.0,
            }
        ),
        encoding="utf-8",
    )


def test_select_pde_hypothesis_maps_overdamping(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    output_path = tmp_path / "hypothesis.json"
    _write_score(score_path, score=40.0, rmse=0.5)
    _write_error_diagnosis(error_path, "overdamped_rollout")
    _write_physics_check(physics_path, "overdissipation")

    result = select_pde_hypothesis_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        output_path=output_path,
    )

    assert result["selected_hypothesis"] in {
        "add_spectral_consistency_loss",
        "add_gradient_weighted_loss",
    }
    assert result["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
    assert output_path.exists()


def test_select_pde_hypothesis_maps_long_horizon_drift(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    _write_score(score_path, score=35.0, rmse=0.6)
    _write_error_diagnosis(error_path, "long_horizon_drift")
    _write_physics_check(physics_path, "inconclusive")

    result = select_pde_hypothesis_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        output_path=None,
    )

    assert result["selected_hypothesis"] == "use_scheduled_sampling_rollout_finetune"


def test_select_pde_hypothesis_cli_writes_payload(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    output_path = tmp_path / "diagnostics" / "hypothesis.json"
    _write_score(score_path, score=40.0, rmse=0.5)
    _write_error_diagnosis(error_path, "overdamped_rollout")
    _write_physics_check(physics_path, "overdissipation")

    result = CliRunner().invoke(
        app,
        [
            "select-pde-hypothesis",
            "--score-path",
            str(score_path),
            "--error-diagnosis-path",
            str(error_path),
            "--physics-check-path",
            str(physics_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
    assert payload["artifacts"]["selected_pde_hypothesis_path"] == str(output_path)
    assert "selected_hypothesis" in payload["metrics"]
```

- [ ] **Step 2: Run hypothesis tests to verify they fail**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pde_agent.core.pde_hypothesis'`.

- [ ] **Step 3: Implement hypothesis core**

Create `src/pde_agent/core/pde_hypothesis.py` with this content:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PDEHypothesisError(RuntimeError):
    pass


def select_pde_hypothesis_from_files(
    *,
    score_path: str | Path,
    error_diagnosis_path: str | Path,
    physics_check_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    score = _load_json(Path(score_path), "score")
    error = _load_json(Path(error_diagnosis_path), "error diagnosis")
    physics = _load_json(Path(physics_check_path), "physics check")
    result = select_pde_hypothesis(score=score, error=error, physics=physics)
    result["score_path"] = str(score_path)
    result["error_diagnosis_path"] = str(error_diagnosis_path)
    result["physics_check_path"] = str(physics_check_path)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result["hypothesis_path"] = str(output)
        output.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def select_pde_hypothesis(
    *,
    score: dict[str, Any],
    error: dict[str, Any],
    physics: dict[str, Any],
) -> dict[str, Any]:
    pattern = _required_string(error, "dominant_error_pattern")
    issue = _required_string(physics, "physics_issue")
    total_score = float(score.get("competition_like_score_short", 0.0))
    if pattern == "overdamped_rollout" or issue == "overdissipation":
        selected = "add_spectral_consistency_loss"
        expected = "reduce high-frequency damping while preserving rollout stability"
        risk = "may amplify oscillations near shocks if overweighted"
        rejected = ["apply_low_pass_stabilization", "keep_baseline_and_collect_stronger_evidence"]
    elif pattern == "long_horizon_drift":
        selected = "use_scheduled_sampling_rollout_finetune"
        expected = "reduce exposure-bias accumulation during autoregressive rollout"
        risk = "may increase training cost and destabilize short-term accuracy"
        rejected = ["add_spectral_consistency_loss"]
    elif issue == "residual_mismatch":
        selected = "add_residual_correction_head"
        expected = "reduce PDE residual mismatch without replacing the base rollout"
        risk = "may overfit finite-difference residual proxies"
        rejected = ["apply_low_pass_stabilization"]
    elif issue == "nonphysical_oscillation" or pattern == "oscillatory_error":
        selected = "apply_low_pass_stabilization"
        expected = "suppress unstable high-frequency oscillations"
        risk = "may worsen overdissipation"
        rejected = ["add_gradient_weighted_loss"]
    elif pattern == "shock_region_error":
        selected = "add_gradient_weighted_loss"
        expected = "focus training signal near steep gradients and shock-like regions"
        risk = "may reduce smooth-region accuracy"
        rejected = ["apply_low_pass_stabilization"]
    else:
        selected = "keep_baseline_and_collect_stronger_evidence"
        expected = "avoid model changes when diagnostics are weak or contradictory"
        risk = "delays optimization if the diagnosis is actually sufficient"
        rejected = ["add_residual_correction_head", "use_scheduled_sampling_rollout_finetune"]

    rationale = (
        f"Selected {selected} because dominant_error_pattern={pattern}, "
        f"physics_issue={issue}, score={total_score:.6g}."
    )
    return {
        "selected_hypothesis": selected,
        "rationale": rationale,
        "expected_effect": expected,
        "risk": risk,
        "next_experiment_scope": "bounded_proposal_no_model_code_changes",
        "required_artifacts": [
            "score_json",
            "rollout_error_diagnosis_json",
            "burgers_physics_check_json",
        ],
        "rejected_hypotheses": rejected,
        "state_hint": "PDE_HYPOTHESIS_SELECTED",
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise PDEHypothesisError(f"missing {label} path: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PDEHypothesisError(f"{label} JSON must be an object")
    return payload


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PDEHypothesisError(f"missing string field: {key}")
    return value
```

- [ ] **Step 4: Implement hypothesis command and CLI**

Create `src/pde_agent/commands/select_pde_hypothesis.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.pde_hypothesis import select_pde_hypothesis_from_files


def select_pde_hypothesis(
    *,
    score_path: str,
    error_diagnosis_path: str,
    physics_check_path: str,
    output_path: str,
) -> dict[str, object]:
    result = select_pde_hypothesis_from_files(
        score_path=Path(score_path),
        error_diagnosis_path=Path(error_diagnosis_path),
        physics_check_path=Path(physics_check_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="select-pde-hypothesis",
        state_hint="PDE_HYPOTHESIS_SELECTED",
        artifacts={"selected_pde_hypothesis_path": str(output_path)},
        metrics={
            "selected_hypothesis": result["selected_hypothesis"],
            "next_experiment_scope": result["next_experiment_scope"],
        },
        next_hint="use selected hypothesis to design the next bounded model experiment",
    )
```

Modify `src/pde_agent/cli.py`:

```python
from pde_agent.commands.select_pde_hypothesis import select_pde_hypothesis
```

Add:

```python
@app.command("select-pde-hypothesis")
def select_pde_hypothesis_cmd(
    score_path: str = typer.Option(...),
    error_diagnosis_path: str = typer.Option(...),
    physics_check_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            select_pde_hypothesis(
                score_path=score_path,
                error_diagnosis_path=error_diagnosis_path,
                physics_check_path=physics_check_path,
                output_path=output_path,
            )
        )
    )
```

- [ ] **Step 5: Run hypothesis tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add src/pde_agent/core/pde_hypothesis.py src/pde_agent/commands/select_pde_hypothesis.py src/pde_agent/cli.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: add pde hypothesis selector"
```

Expected: commit succeeds.

---

### Task 5: Templates, End-To-End Smoke, And Documentation

**Files:**
- Create: `src/pde_agent/templates/command/call_generate_baseline_prediction.md`
- Create: `src/pde_agent/templates/command/call_diagnose_rollout_error.md`
- Create: `src/pde_agent/templates/command/call_check_burgers_physics_local.md`
- Create: `src/pde_agent/templates/command/call_select_pde_hypothesis.md`
- Modify: `README.md`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Add end-to-end smoke test**

Append this content to `tests/test_pde_diagnostic_skills_v1.py`:

```python
def test_local_pde_diagnostic_chain_cli_end_to_end(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    prediction_path = tmp_path / "prediction.npz"
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    context = np.ones((1, 3, 8), dtype=np.float32)
    target = np.zeros((1, 6, 8), dtype=np.float32)
    _write_windows(windows_path, context, target)

    commands = [
        [
            "generate-baseline-prediction",
            "--windows-path",
            str(windows_path),
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
        result = CliRunner().invoke(app, command)
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["status"] == "ok"

    for artifact in (prediction_path, score_path, error_path, physics_path, hypothesis_path):
        assert artifact.exists()
    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    assert hypothesis["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
```

- [ ] **Step 2: Create command templates**

Create `src/pde_agent/templates/command/call_generate_baseline_prediction.md`:

```markdown
# Command: generate-baseline-prediction

Use this when a local windows NPZ exists and no prediction NPZ exists yet.

Required inputs:
- `windows_path`: NPZ containing `context` and `target`
- `output_path`: desired prediction NPZ

Command:

```bash
uv run --frozen pde-agent generate-baseline-prediction \
  --windows-path <windows_path> \
  --output-path <output_path> \
  --method persistence
```

Expected artifact:
- NPZ at `output_path` with key `prediction`

Forbidden shortcuts:
- Do not invent a prediction file path that was not requested.
- Do not read target values to construct the prediction.

Research log summary:
- Record method, sample count, forecast steps, and output path.
```

Create `src/pde_agent/templates/command/call_diagnose_rollout_error.md`:

```markdown
# Command: diagnose-rollout-error

Use this after `score-local-rollout` has produced a score JSON.

Required inputs:
- `prediction_path`: NPZ containing `prediction`
- `target_path`: windows NPZ containing `target`
- `score_path`: JSON produced by `score-local-rollout`
- `output_path`: rollout diagnosis JSON path

Command:

```bash
uv run --frozen pde-agent diagnose-rollout-error \
  --prediction-path <prediction_path> \
  --target-path <target_path> \
  --score-path <score_path> \
  --output-path <output_path>
```

Expected artifact:
- JSON at `output_path` with `dominant_error_pattern` and evidence.

Forbidden shortcuts:
- Do not infer the dominant error pattern without running this command.
- Do not ignore the score JSON.

Research log summary:
- Record dominant error pattern, global RMSE, time error slope, and two evidence strings.
```

Create `src/pde_agent/templates/command/call_check_burgers_physics_local.md`:

```markdown
# Command: check-burgers-physics-local

Use this after a prediction NPZ and target windows NPZ exist.

Required inputs:
- `prediction_path`: NPZ containing `prediction`
- `target_path`: windows NPZ containing `target`
- `output_path`: physics check JSON path
- `nu`: Burgers viscosity, default `0.001`

Command:

```bash
uv run --frozen pde-agent check-burgers-physics-local \
  --prediction-path <prediction_path> \
  --target-path <target_path> \
  --output-path <output_path> \
  --nu 0.001
```

Expected artifact:
- JSON at `output_path` with `physics_issue`, residual metrics, and variation metrics.

Forbidden shortcuts:
- Do not claim physical consistency from NRMSE alone.
- Do not treat this finite-difference proxy as an exact PDE proof.

Research log summary:
- Record physics issue, residual ratio, total variation ratio, and shock-region error ratio.
```

Create `src/pde_agent/templates/command/call_select_pde_hypothesis.md`:

```markdown
# Command: select-pde-hypothesis

Use this after score, rollout error diagnosis, and Burgers physics check artifacts exist.

Required inputs:
- `score_path`
- `error_diagnosis_path`
- `physics_check_path`
- `output_path`

Command:

```bash
uv run --frozen pde-agent select-pde-hypothesis \
  --score-path <score_path> \
  --error-diagnosis-path <error_diagnosis_path> \
  --physics-check-path <physics_check_path> \
  --output-path <output_path>
```

Expected artifact:
- JSON at `output_path` with `selected_hypothesis`, rationale, expected effect, risk, and rejected hypotheses.

Forbidden shortcuts:
- Do not choose a hypothesis before both diagnostics exist.
- Do not edit model code in this step.

Research log summary:
- Record selected hypothesis, rationale, expected effect, and risk.
```

- [ ] **Step 3: Add README local diagnostic chain**

Append this section to `README.md`:

````markdown
## Local PDE Diagnostic Chain

After building a local windows NPZ, generate a weak persistence baseline and run the first PDE diagnostic loop:

```bash
uv run --frozen pde-agent generate-baseline-prediction \
  --windows-path data/derived/local_windows_task1_val.npz \
  --output-path data/derived/local_persistence_task1_val.npz \
  --method persistence

uv run --frozen pde-agent score-local-rollout \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_score.json

uv run --frozen pde-agent diagnose-rollout-error \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --score-path runs/local_score.json \
  --output-path runs/rollout_error_diagnosis.json

uv run --frozen pde-agent check-burgers-physics-local \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/burgers_physics_check.json \
  --nu 0.001

uv run --frozen pde-agent select-pde-hypothesis \
  --score-path runs/local_score.json \
  --error-diagnosis-path runs/rollout_error_diagnosis.json \
  --physics-check-path runs/burgers_physics_check.json \
  --output-path runs/selected_pde_hypothesis.json
```

This chain is intended to give Hermes a reproducible failed baseline, numerical error diagnosis, physics proxy diagnosis, and bounded next-experiment proposal.
````

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: PASS.

- [ ] **Step 5: Run full tests and formatting check**

Run:

```bash
git diff --check
uv run --frozen --extra data pytest tests -q
```

Expected: `git diff --check` has no output and tests PASS.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add src/pde_agent/templates/command/call_generate_baseline_prediction.md src/pde_agent/templates/command/call_diagnose_rollout_error.md src/pde_agent/templates/command/call_check_burgers_physics_local.md src/pde_agent/templates/command/call_select_pde_hypothesis.md README.md tests/test_pde_diagnostic_skills_v1.py
git commit -m "docs: add pde diagnostic command templates"
```

Expected: commit succeeds.

---

## Final Verification

- [ ] Run focused test:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected: PASS.

- [ ] Run full test suite:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected: PASS.

- [ ] Verify CLI command availability:

```bash
uv run --frozen pde-agent generate-baseline-prediction --help
uv run --frozen pde-agent diagnose-rollout-error --help
uv run --frozen pde-agent check-burgers-physics-local --help
uv run --frozen pde-agent select-pde-hypothesis --help
```

Expected: each command prints help and exits 0.

- [ ] Inspect worktree:

```bash
git status --short --branch
```

Expected: clean tracked worktree on `master`; ignored generated data may exist locally.

## Spec Coverage

- Baseline generator is implemented by Task 1.
- Rollout error diagnosis is implemented by Task 2.
- Burgers physics local check is implemented by Task 3.
- PDE hypothesis selection is implemented by Task 4.
- Hermes/Kimi command templates and end-to-end smoke are implemented by Task 5.
- Training, checkpoint loading, final submission generation, and exact hidden Frechet scoring remain deferred.

