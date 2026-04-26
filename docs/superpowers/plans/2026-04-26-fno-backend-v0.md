# FNO Backend V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal optional FNO backend path that can inspect FNO-style torch checkpoints and produce local prediction NPZ artifacts for fixture checkpoints without making torch a default dependency.

**Architecture:** Add an optional `torch` extra, implement a focused `fno_backend` core module with optional torch import, checkpoint inspection, fixture-compatible prediction, and typed unsupported-checkpoint errors. Add thin command wrappers and CLI commands for `inspect-fno-checkpoint` and `predict-fno-local`, plus torch-skipping tests and README docs. This phase does not train, finetune, execute long-horizon submission, or implement DeepONet/PI-DeepONet.

**Tech Stack:** Python standard library, NumPy for local windows/prediction NPZ, optional PyTorch via `project.optional-dependencies.torch`, Typer CLI, pytest.

---

## File Structure

- Modify `pyproject.toml`: add optional `torch` dependency group.
- Create `src/pde_agent/core/fno_backend.py`: optional torch import, checkpoint inspection, fixture prediction path, local windows validation, exact output writes.
- Create `src/pde_agent/commands/inspect_fno_checkpoint.py`: command wrapper.
- Create `src/pde_agent/commands/predict_fno_local.py`: command wrapper.
- Modify `src/pde_agent/cli.py`: register `inspect-fno-checkpoint` and `predict-fno-local`.
- Modify `src/pde_agent/core/state_hints.py`: add `FNO_CHECKPOINT_INSPECTED` and `FNO_PREDICTION_READY`.
- Create `tests/test_fno_backend_v0.py`: torch-skipping core and CLI tests.
- Modify `README.md`: document optional FNO backend commands and limits.

---

### Task 1: Optional Torch Dependency And Core Inspect

**Files:**
- Modify: `pyproject.toml`
- Create: `src/pde_agent/core/fno_backend.py`
- Create: `tests/test_fno_backend_v0.py`

- [ ] **Step 1: Add failing inspect tests**

Create `tests/test_fno_backend_v0.py`:

```python
import json
from pathlib import Path

import pytest

from pde_agent.core.fno_backend import (
    FNOBackendError,
    inspect_fno_checkpoint,
)

torch = pytest.importorskip("torch", reason="FNO backend tests require torch extra")


def _write_fixture_checkpoint(path: Path) -> None:
    torch.save(
        {
            "model_type": "fno_v0_fixture",
            "prediction_rule": "delta_extrapolation",
            "config": {
                "modes": 8,
                "width": 16,
            },
            "model_state_dict": {
                "weight": torch.ones(2, 3),
                "bias": torch.zeros(3),
            },
        },
        path,
    )


def test_inspect_fno_checkpoint_fixture(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_path = tmp_path / "metadata.json"
    _write_fixture_checkpoint(checkpoint_path)

    metadata = inspect_fno_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert metadata["state_hint"] == "FNO_CHECKPOINT_INSPECTED"
    assert metadata["checkpoint_path"] == str(checkpoint_path)
    assert metadata["checkpoint_format"] == "torch_serialized"
    assert metadata["has_model_state"] is True
    assert metadata["has_config"] is True
    assert metadata["tensor_summary"]["tensor_key_count"] == 2
    assert metadata["tensor_summary"]["parameter_count"] == 9
    assert "model_state_dict" in metadata["top_level_keys"]
```

- [ ] **Step 2: Run inspect test and verify import failure**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py::test_inspect_fno_checkpoint_fixture -q
```

Expected result before implementation: fail because `pde_agent.core.fno_backend` does not exist.

- [ ] **Step 3: Add optional torch dependency**

Modify `pyproject.toml`:

```toml
[project.optional-dependencies]
data = [
  "h5py>=3.10",
]
torch = [
  "torch>=2.2",
]
```

Keep existing default dependencies unchanged.

- [ ] **Step 4: Create core inspect implementation**

Create `src/pde_agent/core/fno_backend.py`:

```python
"""Optional FNO backend helpers for local PDE agent loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FNOBackendError(RuntimeError):
    pass


def inspect_fno_checkpoint(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    torch = _require_torch()
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise FNOBackendError(
            f"failed to load FNO checkpoint {checkpoint_file}: {exc}"
        ) from exc

    metadata = _inspect_loaded_checkpoint(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_file,
        torch=torch,
    )
    if output_path is not None:
        output_file = Path(output_path)
        metadata["metadata_path"] = str(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return metadata


def _inspect_loaded_checkpoint(
    *,
    checkpoint: Any,
    checkpoint_path: Path,
    torch: Any,
) -> dict[str, Any]:
    top_level_keys: list[str] = []
    has_model_state = False
    has_config = False
    tensor_key_count = 0
    parameter_count = 0
    load_warnings: list[str] = []

    if isinstance(checkpoint, dict):
        top_level_keys = sorted(str(key) for key in checkpoint.keys())
        has_model_state = any(
            key in checkpoint
            for key in ("model_state_dict", "state_dict", "model")
        )
        has_config = any(key in checkpoint for key in ("config", "args", "model_config"))
        tensors = _collect_tensors(checkpoint, torch=torch)
        tensor_key_count = len(tensors)
        parameter_count = int(sum(tensor.numel() for tensor in tensors))
    else:
        load_warnings.append(
            f"unsupported top-level checkpoint object: {type(checkpoint).__name__}"
        )

    return {
        "schema_version": 1,
        "state_hint": "FNO_CHECKPOINT_INSPECTED",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_format": "torch_serialized",
        "top_level_keys": top_level_keys,
        "has_model_state": has_model_state,
        "has_config": has_config,
        "tensor_summary": {
            "parameter_count": parameter_count,
            "tensor_key_count": tensor_key_count,
        },
        "load_warnings": load_warnings,
    }


def _collect_tensors(value: Any, *, torch: Any) -> list[Any]:
    tensors: list[Any] = []
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        for child in value.values():
            tensors.extend(_collect_tensors(child, torch=torch))
    elif isinstance(value, list | tuple):
        for child in value:
            tensors.extend(_collect_tensors(child, torch=torch))
    return tensors


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise FNOBackendError(
            "FNO backend requires the optional torch extra; run with --extra torch"
        ) from exc
    return torch


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FNOBackendError(f"missing {label} path: {path}")
    if not path.is_file():
        raise FNOBackendError(f"{label} path is not a file: {path}")
    return path
```

- [ ] **Step 5: Run inspect test**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py::test_inspect_fno_checkpoint_fixture -q
```

Expected result: pass, or skip only if torch cannot be installed in the current environment. If torch is unavailable, also run `uv run --frozen pytest tests -q` to ensure non-FNO tests still pass.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add pyproject.toml src/pde_agent/core/fno_backend.py tests/test_fno_backend_v0.py
git commit -m "feat: add fno checkpoint inspection core"
```

---

### Task 2: Fixture Prediction Core

**Files:**
- Modify: `src/pde_agent/core/fno_backend.py`
- Modify: `tests/test_fno_backend_v0.py`

- [ ] **Step 1: Add fixture windows and prediction tests**

Append helpers to `tests/test_fno_backend_v0.py`:

```python
def _write_windows(path: Path) -> tuple[object, object]:
    np = pytest.importorskip("numpy")
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    context = np.zeros((2, 4, 8), dtype=np.float32)
    target = np.zeros((2, 3, 8), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.1)
        delta = 0.05 * np.cos(x) + 0.01 * sample
        for t in range(4):
            context[sample, t, :] = base + t * delta
        for t in range(3):
            target[sample, t, :] = base + (4 + t) * delta
    with path.open("wb") as handle:
        np.savez(
            handle,
            context=context,
            target=target,
            metadata_json=json.dumps({"fixture": True}),
        )
    return context, target
```

Add tests:

```python
def test_predict_fno_local_fixture_writes_prediction(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_path = tmp_path / "prediction"
    context, target = _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    from pde_agent.core.fno_backend import predict_fno_local

    summary = predict_fno_local(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        metadata_path=None,
        force=False,
    )

    np = pytest.importorskip("numpy")
    assert output_path.exists()
    assert not output_path.with_suffix(".npz").exists()
    with np.load(output_path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
        metadata = json.loads(str(archive["metadata_json"]))
    assert prediction.shape == target.shape
    assert prediction.dtype == np.float32
    assert metadata["model_type"] == "fno"
    assert metadata["prediction_rule"] == "delta_extrapolation"
    assert summary["prediction_shape"] == list(target.shape)
    assert summary["model_type"] == "fno"


def test_predict_fno_local_rejects_unsupported_checkpoint(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unsupported.pt"
    _write_windows(windows_path)
    torch.save({"model_type": "unknown"}, checkpoint_path)

    from pde_agent.core.fno_backend import predict_fno_local

    with pytest.raises(FNOBackendError, match="unsupported"):
        predict_fno_local(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "prediction.npz",
            metadata_path=None,
            force=False,
        )
```

- [ ] **Step 2: Run prediction tests and verify failure**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py::test_predict_fno_local_fixture_writes_prediction tests/test_fno_backend_v0.py::test_predict_fno_local_rejects_unsupported_checkpoint -q
```

Expected result before implementation: fail because `predict_fno_local` is missing.

- [ ] **Step 3: Implement fixture prediction**

Modify `src/pde_agent/core/fno_backend.py`.

Add imports if not already present:

```python
import json
import numpy as np
```

Add public function:

```python
def predict_fno_local(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    torch = _require_torch()
    windows_file = _require_file(Path(windows_path), "windows")
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise FNOBackendError(f"prediction output already exists: {output_file}")
    context, target = _load_windows(windows_file)
    checkpoint = _load_checkpoint(checkpoint_file, torch=torch)
    if metadata_path is not None:
        _require_file(Path(metadata_path), "metadata")
    prediction_rule = _fixture_prediction_rule(checkpoint)
    prediction = _predict_fixture_rule(
        context=context,
        forecast_steps=target.shape[1],
        prediction_rule=prediction_rule,
    )
    metadata = {
        "model_type": "fno",
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_format": "torch_serialized",
        "prediction_rule": prediction_rule,
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction.astype(np.float32),
            metadata_json=json.dumps(metadata, allow_nan=False, sort_keys=True),
        )
    return {
        "prediction_path": str(output_file),
        "prediction_shape": [int(axis) for axis in prediction.shape],
        **metadata,
    }
```

Add helpers:

```python
def _load_checkpoint(path: Path, *, torch: Any) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise FNOBackendError(f"failed to load FNO checkpoint {path}: {exc}") from exc


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "context" not in archive or "target" not in archive:
                raise FNOBackendError("windows NPZ must include context and target")
            context = np.asarray(archive["context"], dtype=np.float32)
            target = np.asarray(archive["target"], dtype=np.float32)
    except FNOBackendError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise FNOBackendError(f"failed to load windows NPZ {path}: {exc}") from exc
    _validate_windows(context=context, target=target)
    return context, target


def _validate_windows(*, context: np.ndarray, target: np.ndarray) -> None:
    if context.ndim != 3 or target.ndim != 3:
        raise FNOBackendError("context and target must be rank-3 arrays [N, T, X]")
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise FNOBackendError("context and target must share sample and space axes")
    if context.shape[1] < 2:
        raise FNOBackendError("context must include at least two time steps")
    if target.shape[1] < 1:
        raise FNOBackendError("target must include at least one forecast step")
    if not np.all(np.isfinite(context)) or not np.all(np.isfinite(target)):
        raise FNOBackendError("context and target must contain finite values")


def _fixture_prediction_rule(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        raise FNOBackendError("unsupported FNO checkpoint schema")
    if checkpoint.get("model_type") != "fno_v0_fixture":
        raise FNOBackendError("unsupported FNO checkpoint schema")
    prediction_rule = checkpoint.get("prediction_rule")
    if prediction_rule != "delta_extrapolation":
        raise FNOBackendError(f"unsupported FNO prediction rule: {prediction_rule}")
    return prediction_rule


def _predict_fixture_rule(
    *,
    context: np.ndarray,
    forecast_steps: int,
    prediction_rule: str,
) -> np.ndarray:
    if prediction_rule != "delta_extrapolation":
        raise FNOBackendError(f"unsupported FNO prediction rule: {prediction_rule}")
    last = context[:, -1:, :]
    delta = context[:, -1:, :] - context[:, -2:-1, :]
    steps = np.arange(1, forecast_steps + 1, dtype=np.float32).reshape(1, -1, 1)
    prediction = last + steps * delta
    if not np.all(np.isfinite(prediction)):
        raise FNOBackendError("prediction contains non-finite values")
    return prediction.astype(np.float32)
```

- [ ] **Step 4: Run prediction tests**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py -q
```

Expected result: all current FNO tests pass, or skip cleanly if torch unavailable.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/pde_agent/core/fno_backend.py tests/test_fno_backend_v0.py
git commit -m "feat: add fixture fno local prediction"
```

---

### Task 3: CLI Commands And State Hints

**Files:**
- Create: `src/pde_agent/commands/inspect_fno_checkpoint.py`
- Create: `src/pde_agent/commands/predict_fno_local.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Modify: `tests/test_fno_backend_v0.py`

- [ ] **Step 1: Add CLI tests**

Append tests to `tests/test_fno_backend_v0.py`:

```python
from pde_agent.cli import app
from typer.testing import CliRunner


def test_inspect_fno_checkpoint_cli_writes_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_path = tmp_path / "metadata.json"
    _write_fixture_checkpoint(checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "inspect-fno-checkpoint",
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "inspect-fno-checkpoint"
    assert payload["state_hint"] == "FNO_CHECKPOINT_INSPECTED"
    assert payload["artifacts"]["fno_checkpoint_metadata_path"] == str(output_path)
    assert payload["metrics"]["checkpoint_format"] == "torch_serialized"
    assert payload["next_hint"] == "predict-fno-local"


def test_predict_fno_local_cli_writes_payload(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_path = tmp_path / "prediction.npz"
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "predict-fno-local",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "predict-fno-local"
    assert payload["state_hint"] == "FNO_PREDICTION_READY"
    assert payload["artifacts"]["prediction_npz_path"] == str(output_path)
    assert payload["metrics"]["model_type"] == "fno"
    assert payload["next_hint"] == "score-local-rollout"
    assert output_path.exists()
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py::test_inspect_fno_checkpoint_cli_writes_payload tests/test_fno_backend_v0.py::test_predict_fno_local_cli_writes_payload -q
```

Expected result before implementation: fail because commands are not registered.

- [ ] **Step 3: Add command wrappers**

Create `src/pde_agent/commands/inspect_fno_checkpoint.py`:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.fno_backend import inspect_fno_checkpoint as inspect_core
from pde_agent.core.json_output import success_payload


def inspect_fno_checkpoint(
    *,
    checkpoint_path: str,
    output_path: str,
) -> dict[str, object]:
    metadata = inspect_core(
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="inspect-fno-checkpoint",
        state_hint="FNO_CHECKPOINT_INSPECTED",
        artifacts={"fno_checkpoint_metadata_path": metadata["metadata_path"]},
        metrics={
            "checkpoint_format": metadata["checkpoint_format"],
            "top_level_key_count": len(metadata["top_level_keys"]),
            "has_model_state": metadata["has_model_state"],
            "has_config": metadata["has_config"],
        },
        next_hint="predict-fno-local",
    )
```

Create `src/pde_agent/commands/predict_fno_local.py`:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.fno_backend import predict_fno_local as predict_core
from pde_agent.core.json_output import success_payload


def predict_fno_local(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_path: str,
    metadata_path: str | None,
    force: bool,
) -> dict[str, object]:
    summary = predict_core(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        metadata_path=Path(metadata_path) if metadata_path is not None else None,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="predict-fno-local",
        state_hint="FNO_PREDICTION_READY",
        artifacts={"prediction_npz_path": summary["prediction_path"]},
        metrics={
            "model_type": summary["model_type"],
            "sample_count": summary["sample_count"],
            "forecast_steps": summary["forecast_steps"],
            "space_points": summary["space_points"],
            "checkpoint_format": summary["checkpoint_format"],
        },
        next_hint="score-local-rollout",
    )
```

- [ ] **Step 4: Add state hints**

Modify `src/pde_agent/core/state_hints.py`:

```python
    "FNO_CHECKPOINT_INSPECTED",
    "FNO_PREDICTION_READY",
```

Place them after `"EXPERIMENT_CONFIG_GENERATED",`.

- [ ] **Step 5: Register CLI commands**

Modify `src/pde_agent/cli.py`.

Add imports:

```python
from pde_agent.commands.inspect_fno_checkpoint import inspect_fno_checkpoint
from pde_agent.commands.predict_fno_local import predict_fno_local
```

Add commands near the other local model commands:

```python
@app.command("inspect-fno-checkpoint")
def inspect_fno_checkpoint_cmd(
    checkpoint_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            inspect_fno_checkpoint(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
            )
        )
    )


@app.command("predict-fno-local")
def predict_fno_local_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    metadata_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            predict_fno_local(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                metadata_path=metadata_path,
                force=force,
            )
        )
    )
```

- [ ] **Step 6: Run CLI tests**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py -q
```

Expected result: pass, or skip cleanly if torch unavailable.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add src/pde_agent/commands/inspect_fno_checkpoint.py src/pde_agent/commands/predict_fno_local.py src/pde_agent/cli.py src/pde_agent/core/state_hints.py tests/test_fno_backend_v0.py
git commit -m "feat: expose fno backend cli"
```

---

### Task 4: README And Optional Smoke

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add after the Lightweight Spectral Baseline section:

```markdown
## FNO Backend V0

FNO support is optional and requires the torch extra. V0 can inspect
torch-serialized FNO-style checkpoints and can run fixture-compatible local
prediction. Official checkpoint prediction may still fail with `FNOBackendError`
if the checkpoint schema is not supported by the V0 adapter.

```bash
uv run --frozen --extra torch pde-agent inspect-fno-checkpoint \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-path runs/fno_checkpoint_metadata.json

uv run --frozen --extra torch pde-agent predict-fno-local \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --metadata-path runs/fno_checkpoint_metadata.json \
  --output-path data/derived/local_fno_prediction_task1_smoke.npz \
  --force
```

The prediction command writes the same `prediction` NPZ contract consumed by
`score-local-rollout`. It never falls back to persistence or the spectral
baseline while claiming to be FNO.
```

Ensure Markdown fences are balanced.

- [ ] **Step 2: Run optional real checkpoint inspect smoke**

Run:

```bash
if test -f data/pdebench/checkpoints/burgers_FNO.tar; then
  uv run --frozen --extra torch pde-agent inspect-fno-checkpoint \
    --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
    --output-path /tmp/pde-agent-fno-metadata.json
fi
```

Expected result: metadata JSON is written if torch and checkpoint are available. If torch cannot be installed or the file is absent, record that the smoke was skipped. Do not require official checkpoint prediction in V0.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py -q
```

Expected result: pass, or skip cleanly if torch unavailable.

- [ ] **Step 4: Commit Task 4**

Run:

```bash
git add README.md
git commit -m "docs: document fno backend v0"
```

If the only action was a skipped smoke with no README changes, no commit is required.

---

### Task 5: Final Verification

**Files:**
- No expected source edits. If this task finds a bug, fix the smallest relevant file and add a regression test before committing.

- [ ] **Step 1: Verify non-FNO suite without torch extra**

Run:

```bash
uv run --frozen pytest tests -q
```

Expected result: non-FNO tests pass and FNO tests skip cleanly if torch is absent.

- [ ] **Step 2: Verify full suite with extras**

Run:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected result: pass.

If torch is available, also run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py -q
```

Expected result: pass.

- [ ] **Step 3: Verify CLI help**

Run:

```bash
uv run --frozen pde-agent inspect-fno-checkpoint --help
uv run --frozen pde-agent predict-fno-local --help
```

Expected output includes required options:

```text
inspect-fno-checkpoint: --checkpoint-path, --output-path
predict-fno-local: --windows-path, --checkpoint-path, --output-path, --metadata-path, --force
```

- [ ] **Step 4: Run diff/status checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected result: `git diff --check` prints nothing. `git status --short --branch` shows only the current branch unless a deliberate committed-state check is being done.

- [ ] **Step 5: Final checklist**

Verify directly:

- `torch` is optional and not in default dependencies.
- FNO tests skip cleanly without torch.
- No DeepONet or PI-DeepONet code was added.
- No training or finetuning command was added.
- Unsupported checkpoint schemas raise `FNOBackendError`.
- FNO fixture prediction writes exact output path and prediction NPZ.
- Existing score/diagnose/physics/plan/config commands are unchanged.

If no source edits are required, no commit is needed for Task 5.
