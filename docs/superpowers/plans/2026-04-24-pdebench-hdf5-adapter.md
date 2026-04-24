# PDEBench HDF5 Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only PDEBench/Burgers data adapter so existing diagnostic commands can inspect HDF5/NPZ arrays and compute real rollout, spectral, and physics metrics without training.

**Architecture:** Keep the current CLI protocol stable and add a data layer beneath it. Commands read `config/resolved.json`, load a canonical array bundle through a small adapter, compute deterministic metrics, and write JSON artifacts. Synthetic mode remains unchanged for Hermes protocol smoke tests; NPZ and HDF5 modes add real-array coverage.

**Tech Stack:** Python 3.12, Typer, pytest, numpy, optional h5py, JSON artifacts, local git-ignored data directory.

---

## File Structure

- Create `src/pde_agent/core/array_contract.py`: dataclasses, shape validation, source-mode constants, error types for canonical Burgers arrays.
- Create `src/pde_agent/core/data_config.py`: read and normalize the `data` section from `config/resolved.json`.
- Create `src/pde_agent/core/pdebench_adapter.py`: load synthetic, NPZ, and HDF5-backed canonical bundles; inspect data sources.
- Create `src/pde_agent/core/real_metrics.py`: NRMSE, error curve, FFT band errors, and physics proxy metrics from canonical arrays.
- Create `src/pde_agent/commands/inspect_data.py`: CLI command implementation for `inspect-data`.
- Modify `src/pde_agent/commands/eval_rollout.py`: route to synthetic or real-array rollout metrics based on resolved config.
- Modify `src/pde_agent/commands/analyze_spectrum.py`: route to synthetic or real-array spectrum metrics.
- Modify `src/pde_agent/commands/check_physics.py`: route to synthetic or real-array physics metrics.
- Modify `src/pde_agent/commands/collect_artifacts.py`: include data inspection artifacts and summary fields when present.
- Modify `src/pde_agent/cli.py`: expose `inspect-data`.
- Modify `pyproject.toml`: add `numpy` runtime dependency and optional `data` extra for `h5py`.
- Modify `.gitignore`: ignore local PDEBench files and generated data products.
- Create `data/README.md`: document local data placement.
- Create `tests/test_array_contract.py`: canonical array validation tests.
- Create `tests/test_pdebench_adapter.py`: NPZ and tiny-HDF5 loader/inspection tests.
- Create `tests/test_real_metrics.py`: deterministic metric tests.
- Create `tests/test_inspect_data_command.py`: command artifact tests.
- Modify `tests/test_phase2_synthetic_harness.py`: preserve synthetic command behavior.
- Modify `tests/test_phase3_diagnostic_skills.py`: preserve synthetic diagnostic behavior and add real-array branches where focused.

---

### Task 1: Dependencies And Data Directory Contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `data/README.md`
- Test: no dedicated test file; verify with `uv lock` and `git status --short`

- [ ] **Step 1: Add numpy and optional h5py packaging**

Edit `pyproject.toml` so dependencies become:

```toml
dependencies = [
  "numpy>=2.0",
  "typer>=0.12",
]

[project.optional-dependencies]
data = [
  "h5py>=3.10",
]
```

- [ ] **Step 2: Ignore local PDEBench data and generated products**

Add these lines to `.gitignore`:

```gitignore
# Local PDEBench data and generated experiment outputs
data/**/*.h5
data/**/*.hdf5
data/**/*.npz
data/**/*.pt
data/**/*.tar
integration_tests/**/fixture.npz
runs/
```

- [ ] **Step 3: Document local data placement**

Create `data/README.md`:

````markdown
# Local PDEBench Data

Place local PDEBench/Burgers files under:

```text
data/pdebench/burgers/
```

Expected competition files may include:

```text
1D_Burgers_Sols_Nu0.001.hdf5
1D_Burgers_Sols_Nu0.001_pred.hdf5
1D_Burgers_Sols_Nu0.001_val.hdf5
1D_Burgers_Sols_Nu0.001_submit.hdf5
```

Large data files, checkpoints, generated NPZ files, and generated HDF5 files are ignored by git. The harness reads these files only when a run config explicitly points to them.
````

- [ ] **Step 4: Verify dependency metadata**

Run:

```bash
uv lock
uv run python -c "import numpy; print(numpy.__version__)"
```

Expected:

```text
Resolved ...
<installed numpy version>
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore data/README.md
git commit -m "chore: add pdebench data dependencies"
```

---

### Task 2: Canonical Array Contract

**Files:**
- Create: `src/pde_agent/core/array_contract.py`
- Create: `tests/test_array_contract.py`

- [ ] **Step 1: Write failing tests for canonical shape validation**

Create `tests/test_array_contract.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from pde_agent.core.array_contract import (
    ArrayContractError,
    BurgersArrayBundle,
    validate_burgers_bundle,
)


def _metadata() -> dict[str, object]:
    return {
        "problem_id": "PDE_Burgers",
        "source_mode": "npz_fixture",
        "nu": 0.001,
        "dt": 5,
        "dx": 4,
        "context_time": 2,
        "forecast_time": 3,
        "space_points": 4,
        "time_reduction": 5,
        "space_reduction": 4,
        "sample_count": 2,
    }


def test_validate_burgers_bundle_accepts_matching_arrays() -> None:
    bundle = BurgersArrayBundle(
        context=np.zeros((2, 2, 4), dtype=np.float32),
        prediction=np.ones((2, 3, 4), dtype=np.float32),
        target=np.ones((2, 3, 4), dtype=np.float32) * 1.5,
        full_reference=None,
        metadata=_metadata(),
    )

    result = validate_burgers_bundle(bundle)

    assert result.context.shape == (2, 2, 4)
    assert result.prediction.shape == (2, 3, 4)
    assert result.target.shape == (2, 3, 4)
    assert result.metadata["source_mode"] == "npz_fixture"


def test_validate_burgers_bundle_rejects_prediction_target_shape_mismatch() -> None:
    bundle = BurgersArrayBundle(
        context=np.zeros((2, 2, 4), dtype=np.float32),
        prediction=np.ones((2, 3, 4), dtype=np.float32),
        target=np.ones((2, 4, 4), dtype=np.float32),
        full_reference=None,
        metadata=_metadata(),
    )

    with pytest.raises(ArrayContractError, match="prediction and target shapes"):
        validate_burgers_bundle(bundle)


def test_validate_burgers_bundle_rejects_non_3d_arrays() -> None:
    bundle = BurgersArrayBundle(
        context=np.zeros((2, 4), dtype=np.float32),
        prediction=np.ones((2, 3, 4), dtype=np.float32),
        target=np.ones((2, 3, 4), dtype=np.float32),
        full_reference=None,
        metadata=_metadata(),
    )

    with pytest.raises(ArrayContractError, match="context must have rank 3"):
        validate_burgers_bundle(bundle)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_array_contract.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'pde_agent.core.array_contract'
```

- [ ] **Step 3: Implement the array contract**

Create `src/pde_agent/core/array_contract.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


SourceMode = Literal["synthetic", "npz_fixture", "pdebench_hdf5"]


class ArrayContractError(ValueError):
    """Raised when an array bundle cannot satisfy the Burgers contract."""


@dataclass(frozen=True)
class BurgersArrayBundle:
    context: NDArray[np.floating]
    prediction: NDArray[np.floating]
    target: NDArray[np.floating]
    full_reference: NDArray[np.floating] | None
    metadata: dict[str, object]


def as_float_array(name: str, value: object) -> NDArray[np.floating]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ArrayContractError(f"{name} dtype cannot be converted to float safely")
    return array.astype(np.float64, copy=False)


def validate_burgers_bundle(bundle: BurgersArrayBundle) -> BurgersArrayBundle:
    context = as_float_array("context", bundle.context)
    prediction = as_float_array("prediction", bundle.prediction)
    target = as_float_array("target", bundle.target)
    full_reference = (
        None
        if bundle.full_reference is None
        else as_float_array("full_reference", bundle.full_reference)
    )

    _require_rank("context", context, 3)
    _require_rank("prediction", prediction, 3)
    _require_rank("target", target, 3)
    if full_reference is not None:
        _require_rank("full_reference", full_reference, 3)

    if prediction.shape != target.shape:
        raise ArrayContractError(
            f"prediction and target shapes must match: {prediction.shape} != {target.shape}"
        )
    if context.shape[0] != prediction.shape[0]:
        raise ArrayContractError(
            f"context and prediction sample counts must match: {context.shape[0]} != {prediction.shape[0]}"
        )
    if context.shape[2] != prediction.shape[2]:
        raise ArrayContractError(
            f"context and prediction space dimensions must match: {context.shape[2]} != {prediction.shape[2]}"
        )

    metadata = dict(bundle.metadata)
    metadata["sample_count"] = int(prediction.shape[0])
    metadata["context_time"] = int(context.shape[1])
    metadata["forecast_time"] = int(prediction.shape[1])
    metadata["space_points"] = int(prediction.shape[2])

    return BurgersArrayBundle(
        context=context,
        prediction=prediction,
        target=target,
        full_reference=full_reference,
        metadata=metadata,
    )


def _require_rank(name: str, array: NDArray[np.floating], rank: int) -> None:
    if array.ndim != rank:
        raise ArrayContractError(f"{name} must have rank {rank}; got rank {array.ndim}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_array_contract.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/array_contract.py tests/test_array_contract.py
git commit -m "feat: add burgers array contract"
```

---

### Task 3: Data Config And Source Inspection

**Files:**
- Create: `src/pde_agent/core/data_config.py`
- Create: `src/pde_agent/core/pdebench_adapter.py`
- Create: `tests/test_pdebench_adapter.py`

- [ ] **Step 1: Write failing tests for config, NPZ loading, and HDF5 inspection**

Create `tests/test_pdebench_adapter.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.core.data_config import DataConfig, load_data_config
from pde_agent.core.pdebench_adapter import (
    DataInspectionError,
    inspect_data_source,
    load_array_bundle,
)


def test_load_data_config_defaults_to_synthetic(tmp_path: Path) -> None:
    resolved = tmp_path / "resolved.json"
    resolved.write_text("{}", encoding="utf-8")

    config = load_data_config(resolved)

    assert config.mode == "synthetic"
    assert config.problem_id == "PDE_Burgers"
    assert config.context_steps == 10
    assert config.forecast_steps == 190
    assert config.time_reduction == 5
    assert config.space_reduction == 4


def test_npz_fixture_loads_canonical_bundle(tmp_path: Path) -> None:
    npz_path = tmp_path / "fixture.npz"
    np.savez(
        npz_path,
        context=np.zeros((2, 2, 4), dtype=np.float32),
        prediction=np.ones((2, 3, 4), dtype=np.float32),
        target=np.ones((2, 3, 4), dtype=np.float32) * 2,
        metadata_json=json.dumps({"nu": 0.001}),
    )
    config = DataConfig(
        mode="npz_fixture",
        problem_id="PDE_Burgers",
        source_path=str(npz_path),
        dataset_key=None,
        context_steps=2,
        forecast_steps=3,
        time_reduction=5,
        space_reduction=4,
        sample_limit=None,
    )

    bundle = load_array_bundle(config)

    assert bundle.context.shape == (2, 2, 4)
    assert bundle.prediction.shape == (2, 3, 4)
    assert bundle.target.shape == (2, 3, 4)
    assert bundle.metadata["source_mode"] == "npz_fixture"
    assert bundle.metadata["nu"] == 0.001


def test_npz_fixture_missing_source_path_fails() -> None:
    config = DataConfig(
        mode="npz_fixture",
        problem_id="PDE_Burgers",
        source_path=None,
        dataset_key=None,
        context_steps=2,
        forecast_steps=3,
        time_reduction=5,
        space_reduction=4,
        sample_limit=None,
    )

    with pytest.raises(DataInspectionError, match="source_path is required"):
        load_array_bundle(config)


def test_hdf5_inspection_reports_missing_h5py_or_shape(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    hdf5_path = tmp_path / "tiny.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 25, 16), dtype=np.float32))

    config = DataConfig(
        mode="pdebench_hdf5",
        problem_id="PDE_Burgers",
        source_path=str(hdf5_path),
        dataset_key="tensor",
        context_steps=2,
        forecast_steps=3,
        time_reduction=5,
        space_reduction=4,
        sample_limit=None,
    )

    inspection = inspect_data_source(config)

    assert inspection["source_mode"] == "pdebench_hdf5"
    assert inspection["dataset_key"] == "tensor"
    assert inspection["raw_shape"] == [2, 25, 16]
    assert inspection["reduced_shape"] == [2, 5, 4]
    assert inspection["contract_shape"]["context"] == [2, 2, 4]
    assert inspection["contract_shape"]["target"] == [2, 3, 4]
    assert inspection["is_compatible"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_pdebench_adapter.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'pde_agent.core.data_config'
```

- [ ] **Step 3: Implement data config**

Create `src/pde_agent/core/data_config.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    mode: str
    problem_id: str
    source_path: str | None
    dataset_key: str | None
    context_steps: int
    forecast_steps: int
    time_reduction: int
    space_reduction: int
    sample_limit: int | None


def load_data_config(resolved_config_path: Path) -> DataConfig:
    payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    raw = payload.get("data", {})
    if not isinstance(raw, dict):
        raw = {}
    return DataConfig(
        mode=str(raw.get("mode", "synthetic")),
        problem_id=str(raw.get("problem_id", "PDE_Burgers")),
        source_path=_optional_str(raw.get("source_path")),
        dataset_key=_optional_str(raw.get("dataset_key")),
        context_steps=int(raw.get("context_steps", 10)),
        forecast_steps=int(raw.get("forecast_steps", 190)),
        time_reduction=int(raw.get("time_reduction", 5)),
        space_reduction=int(raw.get("space_reduction", 4)),
        sample_limit=_optional_int(raw.get("sample_limit")),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
```

- [ ] **Step 4: Implement adapter inspection and loading**

Create `src/pde_agent/core/pdebench_adapter.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.array_contract import BurgersArrayBundle, validate_burgers_bundle
from pde_agent.core.data_config import DataConfig


class DataInspectionError(RuntimeError):
    """Raised when a configured data source cannot be inspected or loaded."""


def inspect_data_source(config: DataConfig) -> dict[str, object]:
    if config.mode == "synthetic":
        return _synthetic_inspection(config)
    if config.mode == "npz_fixture":
        bundle = load_array_bundle(config)
        return _bundle_inspection(config, bundle)
    if config.mode == "pdebench_hdf5":
        return _inspect_hdf5(config)
    raise DataInspectionError(f"unsupported data mode: {config.mode}")


def load_array_bundle(config: DataConfig) -> BurgersArrayBundle:
    if config.mode == "npz_fixture":
        return _load_npz_bundle(config)
    if config.mode == "pdebench_hdf5":
        return _load_hdf5_bundle(config)
    raise DataInspectionError(f"array bundle loading is not supported for mode: {config.mode}")


def _synthetic_inspection(config: DataConfig) -> dict[str, object]:
    return {
        "source_mode": "synthetic",
        "source_path": None,
        "dataset_key": None,
        "available_keys": [],
        "raw_shape": None,
        "raw_dtype": None,
        "reduced_shape": None,
        "contract_shape": None,
        "sample_count": None,
        "context_time": config.context_steps,
        "forecast_time": config.forecast_steps,
        "space_points": 256,
        "is_compatible": True,
        "warnings": ["synthetic mode does not inspect external data"],
    }


def _load_npz_bundle(config: DataConfig) -> BurgersArrayBundle:
    path = _required_existing_path(config)
    with np.load(path, allow_pickle=False) as handle:
        metadata = _metadata(config, source_mode="npz_fixture")
        if "metadata_json" in handle.files:
            metadata.update(json.loads(str(handle["metadata_json"])))
        bundle = BurgersArrayBundle(
            context=handle["context"],
            prediction=handle["prediction"],
            target=handle["target"],
            full_reference=handle["full_reference"] if "full_reference" in handle.files else None,
            metadata=metadata,
        )
    return validate_burgers_bundle(bundle)


def _inspect_hdf5(config: DataConfig) -> dict[str, object]:
    path = _required_existing_path(config)
    h5py = _import_h5py()
    with h5py.File(path, "r") as handle:
        keys = _dataset_keys(handle)
        key = _select_dataset_key(config.dataset_key, keys)
        dataset = handle[key]
        raw_shape = list(dataset.shape)
        raw_dtype = str(dataset.dtype)
    reduced_shape = _reduced_shape(raw_shape, config)
    compatible = len(reduced_shape) == 3 and reduced_shape[1] >= (
        config.context_steps + config.forecast_steps
    )
    return {
        "source_mode": "pdebench_hdf5",
        "source_path": str(path),
        "dataset_key": key,
        "available_keys": keys,
        "raw_shape": raw_shape,
        "raw_dtype": raw_dtype,
        "reduced_shape": reduced_shape,
        "contract_shape": _contract_shape(reduced_shape, config) if compatible else None,
        "sample_count": reduced_shape[0] if len(reduced_shape) == 3 else None,
        "context_time": config.context_steps,
        "forecast_time": config.forecast_steps,
        "space_points": reduced_shape[2] if len(reduced_shape) == 3 else None,
        "is_compatible": compatible,
        "warnings": [],
    }


def _load_hdf5_bundle(config: DataConfig) -> BurgersArrayBundle:
    path = _required_existing_path(config)
    h5py = _import_h5py()
    with h5py.File(path, "r") as handle:
        keys = _dataset_keys(handle)
        key = _select_dataset_key(config.dataset_key, keys)
        array = np.asarray(handle[key], dtype=np.float64)
    if array.ndim != 3:
        raise DataInspectionError(f"HDF5 dataset must have rank 3; got rank {array.ndim}")
    if config.sample_limit is not None:
        array = array[: config.sample_limit]
    reduced = array[:, :: config.time_reduction, :: config.space_reduction]
    needed = config.context_steps + config.forecast_steps
    if reduced.shape[1] < needed:
        raise DataInspectionError(
            f"reduced array has {reduced.shape[1]} time steps; need at least {needed}"
        )
    context = reduced[:, : config.context_steps, :]
    target = reduced[:, config.context_steps : needed, :]
    prediction = target.copy()
    metadata = _metadata(config, source_mode="pdebench_hdf5")
    metadata["dataset_key"] = key
    return validate_burgers_bundle(
        BurgersArrayBundle(
            context=context,
            prediction=prediction,
            target=target,
            full_reference=reduced,
            metadata=metadata,
        )
    )


def _bundle_inspection(config: DataConfig, bundle: BurgersArrayBundle) -> dict[str, object]:
    return {
        "source_mode": bundle.metadata["source_mode"],
        "source_path": config.source_path,
        "dataset_key": config.dataset_key,
        "available_keys": [],
        "raw_shape": None,
        "raw_dtype": str(bundle.target.dtype),
        "reduced_shape": list(bundle.full_reference.shape) if bundle.full_reference is not None else None,
        "contract_shape": {
            "context": list(bundle.context.shape),
            "prediction": list(bundle.prediction.shape),
            "target": list(bundle.target.shape),
        },
        "sample_count": int(bundle.target.shape[0]),
        "context_time": int(bundle.context.shape[1]),
        "forecast_time": int(bundle.target.shape[1]),
        "space_points": int(bundle.target.shape[2]),
        "is_compatible": True,
        "warnings": [],
    }


def _required_existing_path(config: DataConfig) -> Path:
    if config.source_path is None:
        raise DataInspectionError("source_path is required for non-synthetic data mode")
    path = Path(config.source_path)
    if not path.exists():
        raise DataInspectionError(f"source_path does not exist: {path}")
    return path


def _metadata(config: DataConfig, source_mode: str) -> dict[str, object]:
    return {
        "problem_id": config.problem_id,
        "source_mode": source_mode,
        "nu": None,
        "dt": config.time_reduction,
        "dx": config.space_reduction,
        "context_time": config.context_steps,
        "forecast_time": config.forecast_steps,
        "space_points": None,
        "time_reduction": config.time_reduction,
        "space_reduction": config.space_reduction,
        "sample_count": None,
    }


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise DataInspectionError(
            "h5py is required for pdebench_hdf5 mode; install pde-agent[data]"
        ) from exc
    return h5py


def _dataset_keys(handle: Any) -> list[str]:
    keys: list[str] = []
    handle.visititems(
        lambda name, obj: keys.append(name)
        if hasattr(obj, "shape") and hasattr(obj, "dtype")
        else None
    )
    return keys


def _select_dataset_key(configured: str | None, keys: list[str]) -> str:
    if configured is not None:
        if configured not in keys:
            raise DataInspectionError(f"configured dataset_key is absent: {configured}")
        return configured
    candidates = [key for key in keys if key.lower() in {"tensor", "u", "solution", "data"}]
    if len(candidates) == 1:
        return candidates[0]
    if len(keys) == 1:
        return keys[0]
    raise DataInspectionError(f"dataset key is ambiguous; candidates: {keys}")


def _reduced_shape(raw_shape: list[int], config: DataConfig) -> list[int]:
    if len(raw_shape) != 3:
        return raw_shape
    return [
        raw_shape[0],
        len(range(0, raw_shape[1], config.time_reduction)),
        len(range(0, raw_shape[2], config.space_reduction)),
    ]


def _contract_shape(reduced_shape: list[int], config: DataConfig) -> dict[str, list[int]]:
    return {
        "context": [reduced_shape[0], config.context_steps, reduced_shape[2]],
        "target": [reduced_shape[0], config.forecast_steps, reduced_shape[2]],
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_pdebench_adapter.py -v
```

Expected:

```text
4 passed
```

If `h5py` is not installed, expected:

```text
3 passed, 1 skipped
```

- [ ] **Step 6: Commit**

```bash
git add src/pde_agent/core/data_config.py src/pde_agent/core/pdebench_adapter.py tests/test_pdebench_adapter.py
git commit -m "feat: add pdebench data adapter"
```

---

### Task 4: Real Metric Computation

**Files:**
- Create: `src/pde_agent/core/real_metrics.py`
- Create: `tests/test_real_metrics.py`

- [ ] **Step 1: Write failing tests for rollout, spectrum, and physics metrics**

Create `tests/test_real_metrics.py`:

```python
from __future__ import annotations

import numpy as np

from pde_agent.core.array_contract import BurgersArrayBundle, validate_burgers_bundle
from pde_agent.core.real_metrics import (
    physics_metrics,
    rollout_metrics,
    spectrum_metrics,
)


def _bundle() -> BurgersArrayBundle:
    target = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 1.5, 2.5, 4.0],
                [1.0, 1.0, 2.0, 4.0],
            ]
        ],
        dtype=np.float64,
    )
    prediction = target * 0.9
    context = np.zeros((1, 2, 4), dtype=np.float64)
    return validate_burgers_bundle(
        BurgersArrayBundle(
            context=context,
            prediction=prediction,
            target=target,
            full_reference=None,
            metadata={
                "problem_id": "PDE_Burgers",
                "source_mode": "npz_fixture",
                "context_time": 2,
                "forecast_time": 3,
                "space_points": 4,
                "time_reduction": 5,
                "space_reduction": 4,
                "sample_count": 1,
                "nu": 0.001,
                "dt": 5,
                "dx": 4,
            },
        )
    )


def test_rollout_metrics_compute_nrmse_and_curve() -> None:
    metrics = rollout_metrics(_bundle())

    assert metrics["mode"] == "real_array_diagnostic"
    assert metrics["source_mode"] == "npz_fixture"
    assert metrics["one_step"]["nrmse"] > 0
    assert metrics["long_horizon"]["nrmse"] > 0
    assert metrics["segments"]["early"]["nrmse"] > 0
    assert len(metrics["curve"]["time_index"]) == 3
    assert len(metrics["curve"]["nrmse"]) == 3


def test_spectrum_metrics_identify_deterministic_band_errors() -> None:
    metrics = spectrum_metrics(_bundle())

    assert metrics["mode"] == "real_array_diagnostic"
    assert metrics["low_frequency_error"] >= 0
    assert metrics["mid_frequency_error"] >= 0
    assert metrics["high_frequency_error"] >= 0
    assert metrics["dominant_spectral_issue"] in {
        "low_frequency_bias",
        "mid_frequency_error",
        "excessive_smoothing",
        "balanced_spectral_error",
    }


def test_physics_metrics_compute_proxy_flags() -> None:
    metrics = physics_metrics(_bundle())

    assert metrics["mode"] == "real_array_diagnostic"
    assert "mass_drift" in metrics
    assert "energy_decay_ratio" in metrics
    assert "total_variation_ratio" in metrics
    assert metrics["proxy_notice"] == "diagnostic proxies, not formal PDE residuals"
    assert metrics["physics_issue"] in {
        "overdissipation",
        "unstable_growth",
        "oscillatory_artifact",
        "no_major_proxy_issue",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_real_metrics.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'pde_agent.core.real_metrics'
```

- [ ] **Step 3: Implement real metrics**

Create `src/pde_agent/core/real_metrics.py`:

```python
from __future__ import annotations

import numpy as np

from pde_agent.core.array_contract import BurgersArrayBundle


EPS = 1e-12
REAL_DIAGNOSTIC_MODE = "real_array_diagnostic"


def rollout_metrics(bundle: BurgersArrayBundle) -> dict[str, object]:
    curve = _per_time_nrmse(bundle.prediction, bundle.target)
    return {
        "run_id": None,
        "mode": REAL_DIAGNOSTIC_MODE,
        "source_mode": bundle.metadata["source_mode"],
        "one_step": {"nrmse": float(curve[0])},
        "long_horizon": {"nrmse": float(_nrmse(bundle.prediction, bundle.target))},
        "segments": _segment_metrics(curve),
        "curve": {
            "time_index": list(range(len(curve))),
            "nrmse": [float(value) for value in curve],
        },
        "prediction": {"path": None},
    }


def spectrum_metrics(bundle: BurgersArrayBundle) -> dict[str, object]:
    error_fft = np.abs(np.fft.rfft(bundle.prediction - bundle.target, axis=-1))
    target_fft = np.abs(np.fft.rfft(bundle.target, axis=-1))
    normalized = error_fft / (target_fft + EPS)
    band_values = _band_means(normalized)
    high_to_low = band_values["high"] / (band_values["low"] + EPS)
    return {
        "mode": REAL_DIAGNOSTIC_MODE,
        "source_mode": bundle.metadata["source_mode"],
        "low_frequency_error": float(band_values["low"]),
        "mid_frequency_error": float(band_values["mid"]),
        "high_frequency_error": float(band_values["high"]),
        "high_to_low_error_ratio": float(high_to_low),
        "spectral_drift_score": float(np.std(normalized)),
        "dominant_spectral_issue": _dominant_spectral_issue(band_values),
        "metric_definition": "mean abs rfft error normalized by target rfft magnitude over low/mid/high bands",
    }


def physics_metrics(bundle: BurgersArrayBundle) -> dict[str, object]:
    pred = bundle.prediction
    target = bundle.target
    mass_drift = abs(float(np.mean(pred) - np.mean(target))) / (abs(float(np.mean(target))) + EPS)
    pred_energy = float(np.mean(pred**2))
    target_energy = float(np.mean(target**2))
    energy_ratio = pred_energy / (target_energy + EPS)
    pred_tv = _total_variation(pred)
    target_tv = _total_variation(target)
    tv_ratio = pred_tv / (target_tv + EPS)
    blowup = bool(energy_ratio > 1.5)
    oscillatory = bool(tv_ratio > 1.4)
    smoothing = bool(tv_ratio < 0.8)
    return {
        "mode": REAL_DIAGNOSTIC_MODE,
        "source_mode": bundle.metadata["source_mode"],
        "mass_drift": float(mass_drift),
        "energy_decay_ratio": float(energy_ratio),
        "total_variation_ratio": float(tv_ratio),
        "nonphysical_oscillation": oscillatory,
        "blowup_like_behavior": blowup,
        "excessive_smoothing": smoothing,
        "physics_issue": _physics_issue(blowup, oscillatory, smoothing),
        "proxy_notice": "diagnostic proxies, not formal PDE residuals",
    }


def _nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((prediction - target) ** 2))
    scale = np.sqrt(np.mean(target**2)) + EPS
    return float(rmse / scale)


def _per_time_nrmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = prediction - target
    rmse = np.sqrt(np.mean(error**2, axis=(0, 2)))
    scale = np.sqrt(np.mean(target**2, axis=(0, 2))) + EPS
    return rmse / scale


def _segment_metrics(curve: np.ndarray) -> dict[str, dict[str, float]]:
    splits = np.array_split(curve, 3)
    names = ["early", "middle", "late"]
    return {
        name: {"nrmse": float(np.mean(split))}
        for name, split in zip(names, splits, strict=True)
    }


def _band_means(values: np.ndarray) -> dict[str, float]:
    bands = np.array_split(values, 3, axis=-1)
    return {
        "low": float(np.mean(bands[0])),
        "mid": float(np.mean(bands[1])),
        "high": float(np.mean(bands[2])),
    }


def _dominant_spectral_issue(bands: dict[str, float]) -> str:
    low = bands["low"]
    mid = bands["mid"]
    high = bands["high"]
    if high > low * 1.25 and high >= mid:
        return "excessive_smoothing"
    if low > mid and low > high:
        return "low_frequency_bias"
    if mid > low and mid > high:
        return "mid_frequency_error"
    return "balanced_spectral_error"


def _total_variation(array: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(array, axis=-1))))


def _physics_issue(blowup: bool, oscillatory: bool, smoothing: bool) -> str:
    if blowup:
        return "unstable_growth"
    if oscillatory:
        return "oscillatory_artifact"
    if smoothing:
        return "overdissipation"
    return "no_major_proxy_issue"
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_real_metrics.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/real_metrics.py tests/test_real_metrics.py
git commit -m "feat: add real array diagnostics"
```

---

### Task 5: Inspect Data Command

**Files:**
- Create: `src/pde_agent/commands/inspect_data.py`
- Modify: `src/pde_agent/cli.py`
- Create: `tests/test_inspect_data_command.py`

- [ ] **Step 1: Write failing command tests**

Create `tests/test_inspect_data_command.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pde_agent.commands.inspect_data import inspect_data


def test_inspect_data_writes_synthetic_inspection_by_default(tmp_path: Path) -> None:
    run_root = tmp_path / "run_001"
    (run_root / "config").mkdir(parents=True)
    (run_root / "config" / "resolved.json").write_text("{}", encoding="utf-8")

    result = inspect_data(run_id="run_001", run_root=str(run_root))

    inspection_path = run_root / "metrics" / "data_inspection.json"
    payload = json.loads(inspection_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["state_hint"] == "DATA_INSPECTED"
    assert result["artifacts"]["data_inspection_path"] == str(inspection_path)
    assert payload["source_mode"] == "synthetic"
    assert payload["is_compatible"] is True


def test_inspect_data_writes_npz_fixture_inspection(tmp_path: Path) -> None:
    npz_path = tmp_path / "fixture.npz"
    np.savez(
        npz_path,
        context=np.zeros((1, 2, 4), dtype=np.float32),
        prediction=np.ones((1, 3, 4), dtype=np.float32),
        target=np.ones((1, 3, 4), dtype=np.float32),
    )
    run_root = tmp_path / "run_001"
    (run_root / "config").mkdir(parents=True)
    (run_root / "config" / "resolved.json").write_text(
        json.dumps(
            {
                "data": {
                    "mode": "npz_fixture",
                    "source_path": str(npz_path),
                    "context_steps": 2,
                    "forecast_steps": 3,
                }
            }
        ),
        encoding="utf-8",
    )

    result = inspect_data(run_id="run_001", run_root=str(run_root))

    payload = json.loads((run_root / "metrics" / "data_inspection.json").read_text())
    assert result["metrics"]["source_mode"] == "npz_fixture"
    assert payload["contract_shape"]["target"] == [1, 3, 4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_inspect_data_command.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'pde_agent.commands.inspect_data'
```

- [ ] **Step 3: Implement inspect-data command**

Create `src/pde_agent/commands/inspect_data.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from pde_agent.core.data_config import load_data_config
from pde_agent.core.json_output import success_payload
from pde_agent.core.pdebench_adapter import inspect_data_source


def inspect_data(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    config = load_data_config(root / "config" / "resolved.json")
    payload = inspect_data_source(config)
    inspection_path = root / "metrics" / "data_inspection.json"
    inspection_path.parent.mkdir(parents=True, exist_ok=True)
    inspection_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="inspect-data",
        state_hint="DATA_INSPECTED",
        artifacts={"data_inspection_path": str(inspection_path)},
        metrics={
            "source_mode": payload["source_mode"],
            "is_compatible": payload["is_compatible"],
            "sample_count": payload["sample_count"],
        },
    )
```

- [ ] **Step 4: Expose CLI command**

Modify `src/pde_agent/cli.py` imports:

```python
from pde_agent.commands.inspect_data import inspect_data
```

Add this command after `resolve-config`:

```python
@app.command("inspect-data")
def inspect_data_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(inspect_data(run_id=run_id, run_root=run_root)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_inspect_data_command.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/pde_agent/commands/inspect_data.py src/pde_agent/cli.py tests/test_inspect_data_command.py
git commit -m "feat: add inspect data command"
```

---

### Task 6: Wire Real Array Modes Into Existing Diagnostics

**Files:**
- Modify: `src/pde_agent/commands/eval_rollout.py`
- Modify: `src/pde_agent/commands/analyze_spectrum.py`
- Modify: `src/pde_agent/commands/check_physics.py`
- Modify: `src/pde_agent/commands/collect_artifacts.py`
- Modify: `tests/test_phase2_synthetic_harness.py`
- Modify: `tests/test_phase3_diagnostic_skills.py`

- [ ] **Step 1: Add failing real-array eval-rollout test**

Append to `tests/test_phase2_synthetic_harness.py`:

```python
def test_eval_rollout_uses_npz_fixture_when_configured(tmp_path: Path) -> None:
    npz_path = tmp_path / "fixture.npz"
    np.savez(
        npz_path,
        context=np.zeros((1, 2, 4), dtype=np.float32),
        prediction=np.ones((1, 3, 4), dtype=np.float32) * 0.9,
        target=np.ones((1, 3, 4), dtype=np.float32),
    )
    run_root = tmp_path / "run_001"
    (run_root / "config").mkdir(parents=True)
    (run_root / "config" / "resolved.json").write_text(
        json.dumps(
            {
                "data": {
                    "mode": "npz_fixture",
                    "source_path": str(npz_path),
                    "context_steps": 2,
                    "forecast_steps": 3,
                }
            }
        ),
        encoding="utf-8",
    )

    result = eval_rollout(run_id="run_001", run_root=str(run_root))

    rollout = json.loads((run_root / "metrics" / "rollout.json").read_text())
    assert result["metrics"]["source_mode"] == "npz_fixture"
    assert rollout["mode"] == "real_array_diagnostic"
    assert rollout["curve"]["path"] == str(run_root / "metrics" / "error_curve.json")
    assert rollout["prediction"]["path"] == str(npz_path)
```

Also add `import numpy as np` near the top of this test file.

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
uv run pytest tests/test_phase2_synthetic_harness.py::test_eval_rollout_uses_npz_fixture_when_configured -v
```

Expected:

```text
KeyError or assertion failure because eval-rollout still uses synthetic payload
```

- [ ] **Step 3: Update eval-rollout routing**

Modify `src/pde_agent/commands/eval_rollout.py` so it becomes:

```python
import json
from pathlib import Path

from pde_agent.core.data_config import load_data_config
from pde_agent.core.json_output import success_payload
from pde_agent.core.pdebench_adapter import load_array_bundle
from pde_agent.core.real_metrics import rollout_metrics
from pde_agent.core.synthetic_pde_fixture import (
    SYNTHETIC_LONG_HORIZON_NRMSE,
    SYNTHETIC_ONE_STEP_NRMSE,
    synthetic_rollout_payload,
)


def eval_rollout(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    config_path = root / "config" / "resolved.json"
    if config_path.exists():
        data_config = load_data_config(config_path)
        if data_config.mode != "synthetic":
            return _eval_real_array_rollout(run_id=run_id, run_root=root)
    return _eval_synthetic_rollout(run_id=run_id, run_root=root)


def _eval_synthetic_rollout(run_id: str, run_root: Path) -> dict[str, object]:
    rollout_path = run_root / "metrics" / "rollout.json"
    rollout_payload = synthetic_rollout_payload(run_id=run_id, run_root=run_root)
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(json.dumps(rollout_payload, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="eval-rollout",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "rollout_metrics_path": str(rollout_path),
            "error_curve_path": rollout_payload["curve"]["path"],
            "prediction_hdf5_path": rollout_payload["prediction"]["path"],
        },
        metrics={
            "one_step_nrmse": SYNTHETIC_ONE_STEP_NRMSE,
            "long_horizon_nrmse": SYNTHETIC_LONG_HORIZON_NRMSE,
        },
    )


def _eval_real_array_rollout(run_id: str, run_root: Path) -> dict[str, object]:
    data_config = load_data_config(run_root / "config" / "resolved.json")
    bundle = load_array_bundle(data_config)
    payload = rollout_metrics(bundle)
    payload["run_id"] = run_id
    payload["curve"]["path"] = str(run_root / "metrics" / "error_curve.json")
    payload["prediction"]["path"] = data_config.source_path
    rollout_path = run_root / "metrics" / "rollout.json"
    error_curve_path = run_root / "metrics" / "error_curve.json"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    error_curve_path.write_text(json.dumps(payload["curve"], indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="eval-rollout",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "rollout_metrics_path": str(rollout_path),
            "error_curve_path": str(error_curve_path),
            "prediction_hdf5_path": str(data_config.source_path),
        },
        metrics={
            "source_mode": payload["source_mode"],
            "one_step_nrmse": payload["one_step"]["nrmse"],
            "long_horizon_nrmse": payload["long_horizon"]["nrmse"],
        },
    )
```

- [ ] **Step 4: Run eval-rollout tests**

Run:

```bash
uv run pytest tests/test_phase2_synthetic_harness.py -v
```

Expected:

```text
all tests in tests/test_phase2_synthetic_harness.py passed
```

- [ ] **Step 5: Add failing real-array spectrum and physics tests**

Append to `tests/test_phase3_diagnostic_skills.py`:

```python
def test_real_array_spectrum_and_physics_follow_npz_config(tmp_path: Path) -> None:
    npz_path = tmp_path / "fixture.npz"
    np.savez(
        npz_path,
        context=np.zeros((1, 2, 8), dtype=np.float32),
        prediction=np.ones((1, 3, 8), dtype=np.float32) * 0.8,
        target=np.ones((1, 3, 8), dtype=np.float32),
    )
    run_root = tmp_path / "run_001"
    (run_root / "config").mkdir(parents=True)
    (run_root / "config" / "resolved.json").write_text(
        json.dumps(
            {
                "data": {
                    "mode": "npz_fixture",
                    "source_path": str(npz_path),
                    "context_steps": 2,
                    "forecast_steps": 3,
                }
            }
        ),
        encoding="utf-8",
    )
    eval_rollout(run_id="run_001", run_root=str(run_root))

    spectrum_result = analyze_spectrum(run_id="run_001", run_root=str(run_root))
    physics_result = check_physics(run_id="run_001", run_root=str(run_root))

    spectrum = json.loads((run_root / "metrics" / "spectrum.json").read_text())
    physics = json.loads((run_root / "metrics" / "physics_check.json").read_text())
    assert spectrum_result["metrics"]["source_mode"] == "npz_fixture"
    assert physics_result["metrics"]["source_mode"] == "npz_fixture"
    assert spectrum["mode"] == "real_array_diagnostic"
    assert physics["mode"] == "real_array_diagnostic"
```

Also add `import numpy as np` near the top of this test file.

- [ ] **Step 6: Run the new diagnostic test to verify it fails**

Run:

```bash
uv run pytest tests/test_phase3_diagnostic_skills.py::test_real_array_spectrum_and_physics_follow_npz_config -v
```

Expected:

```text
assertion failure because analyze-spectrum and check-physics still write synthetic diagnostics
```

- [ ] **Step 7: Update analyze-spectrum routing**

Modify `src/pde_agent/commands/analyze_spectrum.py`:

```python
from pathlib import Path

from pde_agent.core.data_config import load_data_config
from pde_agent.core.diagnostic_synthesis import read_json, spectrum_payload, write_json
from pde_agent.core.json_output import success_payload
from pde_agent.core.pdebench_adapter import load_array_bundle
from pde_agent.core.real_metrics import spectrum_metrics


def analyze_spectrum(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    data_config_path = root / "config" / "resolved.json"
    if data_config_path.exists():
        data_config = load_data_config(data_config_path)
        if data_config.mode != "synthetic":
            return _analyze_real_spectrum(run_id=run_id, run_root=root)
    return _analyze_synthetic_spectrum(run_id=run_id, run_root=root)


def _analyze_synthetic_spectrum(run_id: str, run_root: Path) -> dict[str, object]:
    rollout = read_json(run_root / "metrics" / "rollout.json")
    payload = spectrum_payload(run_id=run_id, rollout=rollout)
    spectrum_path = run_root / "metrics" / "spectrum.json"
    write_json(spectrum_path, payload)
    return success_payload(
        run_id=run_id,
        command="analyze-spectrum",
        state_hint="SPECTRUM_ANALYZED",
        artifacts={"spectrum_metrics_path": str(spectrum_path)},
        metrics={
            "low_frequency_error": payload["low_frequency_error"],
            "high_frequency_error": payload["high_frequency_error"],
            "spectral_drift_score": payload["spectral_drift_score"],
            "dominant_spectral_issue": payload["dominant_spectral_issue"],
        },
    )


def _analyze_real_spectrum(run_id: str, run_root: Path) -> dict[str, object]:
    data_config = load_data_config(run_root / "config" / "resolved.json")
    payload = spectrum_metrics(load_array_bundle(data_config))
    payload["run_id"] = run_id
    spectrum_path = run_root / "metrics" / "spectrum.json"
    write_json(spectrum_path, payload)
    return success_payload(
        run_id=run_id,
        command="analyze-spectrum",
        state_hint="SPECTRUM_ANALYZED",
        artifacts={"spectrum_metrics_path": str(spectrum_path)},
        metrics={
            "source_mode": payload["source_mode"],
            "low_frequency_error": payload["low_frequency_error"],
            "high_frequency_error": payload["high_frequency_error"],
            "spectral_drift_score": payload["spectral_drift_score"],
            "dominant_spectral_issue": payload["dominant_spectral_issue"],
        },
    )
```

- [ ] **Step 8: Update check-physics routing**

Modify `src/pde_agent/commands/check_physics.py`:

```python
from pathlib import Path

from pde_agent.core.data_config import load_data_config
from pde_agent.core.diagnostic_synthesis import physics_payload, read_json, write_json
from pde_agent.core.json_output import success_payload
from pde_agent.core.pdebench_adapter import load_array_bundle
from pde_agent.core.real_metrics import physics_metrics


def check_physics(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    data_config_path = root / "config" / "resolved.json"
    if data_config_path.exists():
        data_config = load_data_config(data_config_path)
        if data_config.mode != "synthetic":
            return _check_real_physics(run_id=run_id, run_root=root)
    return _check_synthetic_physics(run_id=run_id, run_root=root)


def _check_synthetic_physics(run_id: str, run_root: Path) -> dict[str, object]:
    rollout = read_json(run_root / "metrics" / "rollout.json")
    payload = physics_payload(run_id=run_id, rollout=rollout)
    physics_path = run_root / "metrics" / "physics_check.json"
    write_json(physics_path, payload)
    return success_payload(
        run_id=run_id,
        command="check-physics",
        state_hint="PHYSICS_CHECKED",
        artifacts={"physics_check_path": str(physics_path)},
        metrics={
            "mass_drift": payload["mass_drift"],
            "energy_decay_ratio": payload["energy_decay_ratio"],
            "total_variation_ratio": payload["total_variation_ratio"],
            "physics_issue": payload["physics_issue"],
        },
    )


def _check_real_physics(run_id: str, run_root: Path) -> dict[str, object]:
    data_config = load_data_config(run_root / "config" / "resolved.json")
    payload = physics_metrics(load_array_bundle(data_config))
    payload["run_id"] = run_id
    physics_path = run_root / "metrics" / "physics_check.json"
    write_json(physics_path, payload)
    return success_payload(
        run_id=run_id,
        command="check-physics",
        state_hint="PHYSICS_CHECKED",
        artifacts={"physics_check_path": str(physics_path)},
        metrics={
            "source_mode": payload["source_mode"],
            "mass_drift": payload["mass_drift"],
            "energy_decay_ratio": payload["energy_decay_ratio"],
            "total_variation_ratio": payload["total_variation_ratio"],
            "physics_issue": payload["physics_issue"],
        },
    )
```

- [ ] **Step 9: Include data inspection in collect-artifacts**

Modify `src/pde_agent/commands/collect_artifacts.py` so `_load_diagnostic_payloads` also checks:

```python
data_inspection_path = root / "metrics" / "data_inspection.json"
```

and includes:

```python
if data_inspection_path.exists():
    data_inspection = json.loads(data_inspection_path.read_text(encoding="utf-8"))
    artifacts["data_inspection_path"] = str(data_inspection_path)
    summary["source_mode"] = data_inspection["source_mode"]
    summary["data_is_compatible"] = data_inspection["is_compatible"]
```

Also add this completeness flag to the return object:

```python
"has_data_inspection": data_inspection_path.exists(),
```

- [ ] **Step 10: Run diagnostic tests**

Run:

```bash
uv run pytest tests/test_phase2_synthetic_harness.py tests/test_phase3_diagnostic_skills.py -v
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 11: Commit**

```bash
git add src/pde_agent/commands/eval_rollout.py src/pde_agent/commands/analyze_spectrum.py src/pde_agent/commands/check_physics.py src/pde_agent/commands/collect_artifacts.py tests/test_phase2_synthetic_harness.py tests/test_phase3_diagnostic_skills.py
git commit -m "feat: route diagnostics through real array modes"
```

---

### Task 7: Fixture Integration And Full Verification

**Files:**
- Create: `integration_tests/hermes_array_fixture_smoke/README.md`
- Create: `integration_tests/hermes_array_fixture_smoke/base_config.json`
- Create: `integration_tests/hermes_array_fixture_smoke/override.json`
- Create: `integration_tests/hermes_array_fixture_smoke/task_spec.json`
- Create: `integration_tests/hermes_array_fixture_smoke/run_state.yaml`
- Create: `integration_tests/hermes_array_fixture_smoke/bootstrap_prompt.md`
- Create: `integration_tests/hermes_array_fixture_smoke/prompt_template.md`
- Create: `tests/test_array_fixture_flow.py`

- [ ] **Step 1: Write an end-to-end local fixture flow test**

Create `tests/test_array_fixture_flow.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.inspect_data import inspect_data
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.select_hypothesis import select_hypothesis


def test_array_fixture_flow_reaches_hypothesis_selected_manifest(tmp_path: Path) -> None:
    npz_path = tmp_path / "fixture.npz"
    target = np.ones((2, 5, 16), dtype=np.float32)
    prediction = target * 0.85
    np.savez(
        npz_path,
        context=np.zeros((2, 2, 16), dtype=np.float32),
        prediction=prediction,
        target=target,
    )
    run_root = tmp_path / "run_001"
    (run_root / "config").mkdir(parents=True)
    (run_root / "config" / "resolved.json").write_text(
        json.dumps(
            {
                "data": {
                    "mode": "npz_fixture",
                    "source_path": str(npz_path),
                    "context_steps": 2,
                    "forecast_steps": 5,
                }
            }
        ),
        encoding="utf-8",
    )

    inspect_data(run_id="run_001", run_root=str(run_root))
    eval_rollout(run_id="run_001", run_root=str(run_root))
    analyze_spectrum(run_id="run_001", run_root=str(run_root))
    check_physics(run_id="run_001", run_root=str(run_root))
    interpret_error(run_id="run_001", run_root=str(run_root))
    select_hypothesis(run_id="run_001", run_root=str(run_root))
    record_timing(run_id="run_001", run_root=str(run_root))
    collect_artifacts(run_id="run_001", run_root=str(run_root), config_hash="fixture")

    manifest = json.loads((run_root / "artifacts" / "manifest.json").read_text())
    assert manifest["phase"] == "hypothesis_selected"
    assert manifest["summary"]["source_mode"] == "npz_fixture"
    assert manifest["summary"]["data_is_compatible"] is True
    assert manifest["completeness"]["has_data_inspection"] is True
    assert manifest["completeness"]["has_selected_hypothesis"] is True
```

- [ ] **Step 2: Run the flow test to verify it fails if previous task missed manifest wiring**

Run:

```bash
uv run pytest tests/test_array_fixture_flow.py -v
```

Expected:

```text
1 passed
```

If it fails, fix only the missing manifest or command-routing issue revealed by the assertion.

- [ ] **Step 3: Add fixture directory skeleton**

Create `integration_tests/hermes_array_fixture_smoke/README.md`:

```markdown
# Hermes Array Fixture Smoke

This fixture validates the real-array diagnostic command chain with an explicit NPZ fixture path.

It is not a training run and does not use full PDEBench HDF5 files. It exists to prove that Hermes can follow the same protocol after `inspect-data` is introduced.
```

Create `integration_tests/hermes_array_fixture_smoke/base_config.json`:

```json
{
  "model": {
    "name": "none",
    "mode": "read_only_diagnostic"
  },
  "data": {
    "mode": "npz_fixture",
    "problem_id": "PDE_Burgers",
    "source_path": "./fixture.npz",
    "dataset_key": null,
    "context_steps": 2,
    "forecast_steps": 5,
    "time_reduction": 5,
    "space_reduction": 4,
    "sample_limit": null
  }
}
```

Create `integration_tests/hermes_array_fixture_smoke/override.json`:

```json
{
  "run": {
    "phase": "array_fixture_diagnostic"
  }
}
```

Create `integration_tests/hermes_array_fixture_smoke/task_spec.json`:

```json
{
  "problem_id": "PDE_Burgers",
  "task": "run read-only array fixture diagnostics",
  "forbidden": ["training", "code_patch", "submission_build"]
}
```

Create `integration_tests/hermes_array_fixture_smoke/bootstrap_prompt.md`:

```markdown
# Array Fixture Diagnostic Protocol

Read the required files in the current directory. Then execute only the commands listed in `run_state.yaml`, in order.

Do not inspect command help. Do not search outside the current directory. Do not patch code. Do not run training.
```

Create `integration_tests/hermes_array_fixture_smoke/prompt_template.md`:

```markdown
You are running a bounded PDE Agent array fixture smoke test.

Required behavior:

- Read `run_state.yaml`.
- Read `bootstrap_prompt.md`.
- Read `task_spec.json`.
- Read `base_config.json`.
- Read `override.json`.
- Execute the exact allowed commands from `run_state.yaml`.
- Stop after `collect-artifacts` succeeds.

Forbidden behavior:

- Do not patch code.
- Do not run training.
- Do not inspect command help.
- Do not search outside the current directory.
- Do not guess missing paths.
```

Create `integration_tests/hermes_array_fixture_smoke/run_state.yaml` using the same structure as `integration_tests/hermes_diagnostic_smoke/run_state.yaml`, with:

```yaml
run_id: array_fixture_smoke_001
run_root: /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
working_directory: /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke
command_prefix: /Users/daiduo2/pde-agent/.venv/bin/pde-agent

current_state: TASK_PARSED

allowed_actions:
  - read_run_state
  - read_bootstrap_prompt
  - read_task_spec
  - read_base_config
  - read_override
  - call_init_run
  - call_resolve_config
  - call_inspect_data
  - call_eval_rollout
  - call_analyze_spectrum
  - call_check_physics
  - call_interpret_error
  - call_select_hypothesis
  - call_record_timing
  - call_collect_artifacts

forbidden_actions:
  - patch_code
  - run_training
  - call_build_submission
  - guess_missing_paths
  - skip_required_reads
  - search_outside_current_directory
  - inspect_command_help
  - probe_command_surface

required_reads:
  - /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/run_state.yaml
  - /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/bootstrap_prompt.md
  - /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/task_spec.json
  - /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/base_config.json
  - /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/override.json

required_artifacts:
  - /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001/artifacts/manifest.json

allowed_commands:
  - name: init_run
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent init-run --run-id array_fixture_smoke_001 --runs-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs
  - name: resolve_config
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent resolve-config --run-id array_fixture_smoke_001 --base-config-file /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/base_config.json --override-file /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/override.json --task-spec-file /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/task_spec.json --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
  - name: inspect_data
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent inspect-data --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
  - name: eval_rollout
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent eval-rollout --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
  - name: analyze_spectrum
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent analyze-spectrum --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
  - name: check_physics
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent check-physics --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
  - name: interpret_error
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent interpret-error --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
  - name: select_hypothesis
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent select-hypothesis --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
  - name: record_timing
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent record-timing --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
  - name: collect_artifacts
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent collect-artifacts --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001 --config-hash array-fixture-smoke-001

success_transition:
  on_complete: HYPOTHESIS_SELECTED
```

- [ ] **Step 4: Generate fixture NPZ through a small deterministic command**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np

root = Path("integration_tests/hermes_array_fixture_smoke")
target = np.ones((2, 5, 16), dtype=np.float32)
prediction = target * 0.85
np.savez(
    root / "fixture.npz",
    context=np.zeros((2, 2, 16), dtype=np.float32),
    prediction=prediction,
    target=target,
)
PY
```

Expected:

```text
integration_tests/hermes_array_fixture_smoke/fixture.npz exists locally and is ignored by git
```

- [ ] **Step 5: Run full test suite**

Run:

```bash
uv run pytest tests -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Run CLI smoke manually for array fixture**

Run:

```bash
cd integration_tests/hermes_array_fixture_smoke
/Users/daiduo2/pde-agent/.venv/bin/pde-agent init-run --run-id array_fixture_smoke_001 --runs-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs
/Users/daiduo2/pde-agent/.venv/bin/pde-agent resolve-config --run-id array_fixture_smoke_001 --base-config-file /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/base_config.json --override-file /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/override.json --task-spec-file /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/task_spec.json --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
/Users/daiduo2/pde-agent/.venv/bin/pde-agent inspect-data --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
/Users/daiduo2/pde-agent/.venv/bin/pde-agent eval-rollout --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
/Users/daiduo2/pde-agent/.venv/bin/pde-agent analyze-spectrum --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
/Users/daiduo2/pde-agent/.venv/bin/pde-agent check-physics --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
/Users/daiduo2/pde-agent/.venv/bin/pde-agent interpret-error --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
/Users/daiduo2/pde-agent/.venv/bin/pde-agent select-hypothesis --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
/Users/daiduo2/pde-agent/.venv/bin/pde-agent record-timing --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001
/Users/daiduo2/pde-agent/.venv/bin/pde-agent collect-artifacts --run-id array_fixture_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_array_fixture_smoke/runs/array_fixture_smoke_001 --config-hash array-fixture-smoke-001
```

Expected final command output includes:

```json
{"status": "ok", "state_hint": "HYPOTHESIS_SELECTED"}
```

- [ ] **Step 7: Commit**

```bash
git add integration_tests/hermes_array_fixture_smoke tests/test_array_fixture_flow.py
git commit -m "test: add array fixture diagnostic flow"
```

- [ ] **Step 8: Final verification**

Run:

```bash
uv run pytest tests -v
git status --short
```

Expected:

```text
all tests passed
```

and:

```text
<no output from git status --short>
```
