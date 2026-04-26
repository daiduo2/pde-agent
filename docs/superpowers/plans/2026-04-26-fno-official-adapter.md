# FNO Official Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real inference adapter for official PDEBench 1D Burgers FNO checkpoints while keeping training out of scope.

**Architecture:** Keep `predict_fno_local` as the public entry point and dispatch between the existing fixture path and a new official PDEBench FNO1d adapter. Reconstruct the model from checkpoint tensor shapes, run CPU autoregressive rollout from local windows, and write the existing prediction NPZ contract.

**Tech Stack:** Python 3.12+, NumPy, optional PyTorch via `--extra torch`, Typer CLI wrappers already present, pytest.

---

## File Structure

- Modify `src/pde_agent/core/fno_backend.py`: add official FNO1d config inference, model classes, rollout helpers, and official checkpoint prediction dispatch.
- Modify `tests/test_fno_backend_v0.py`: add synthetic official-schema checkpoint tests and keep existing fixture tests passing.
- Modify `README.md`: document that `predict-fno-local` now supports the official PDEBench FNO1d checkpoint adapter, while training is still out of scope.

No new command files are required. Do not add training commands, DeepONet, PI-DeepONet, UNet, external PDEBench dependency, or neuraloperator dependency.

---

### Task 1: Official FNO1d Schema And Model Core

**Files:**
- Modify: `src/pde_agent/core/fno_backend.py`
- Modify: `tests/test_fno_backend_v0.py`

- [ ] **Step 1: Add failing schema inference test**

Append this helper and test to `tests/test_fno_backend_v0.py` after `_write_fixture_checkpoint`:

```python
def _official_fno_state_dict(
    *,
    torch: object,
    history_steps: int = 3,
    width: int = 4,
    modes: int = 2,
    fc_hidden: int = 5,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(7)
    state: dict[str, object] = {
        "fc0.weight": torch.randn(width, history_steps + 1, generator=generator),
        "fc0.bias": torch.randn(width, generator=generator),
        "fc1.weight": torch.randn(fc_hidden, width, generator=generator),
        "fc1.bias": torch.randn(fc_hidden, generator=generator),
        "fc2.weight": torch.randn(1, fc_hidden, generator=generator),
        "fc2.bias": torch.randn(1, generator=generator),
    }
    for index in range(4):
        state[f"conv{index}.weights1"] = torch.randn(
            width,
            width,
            modes,
            dtype=torch.complex64,
            generator=generator,
        )
        state[f"w{index}.weight"] = torch.randn(
            width,
            width,
            1,
            generator=generator,
        )
        state[f"w{index}.bias"] = torch.randn(width, generator=generator)
    return state
```

Append this test:

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_infer_pdebench_fno1d_config_from_state_dict() -> None:
    import torch

    from pde_agent.core.fno_backend import _infer_pdebench_fno1d_config

    state = _official_fno_state_dict(
        torch=torch,
        history_steps=3,
        width=4,
        modes=2,
        fc_hidden=5,
    )

    config = _infer_pdebench_fno1d_config(state)

    assert config.history_steps == 3
    assert config.width == 4
    assert config.modes == 2
    assert config.fc_hidden == 5
    assert config.layers == 4
```

- [ ] **Step 2: Run schema inference test and verify RED**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py::test_infer_pdebench_fno1d_config_from_state_dict -q
```

Expected before implementation: fail with `ImportError` or `AttributeError` because `_infer_pdebench_fno1d_config` does not exist.

- [ ] **Step 3: Implement config inference and model classes**

Modify imports in `src/pde_agent/core/fno_backend.py`:

```python
from dataclasses import dataclass
```

Add this code before `_inspect_loaded_checkpoint`:

```python
@dataclass(frozen=True)
class PDEBenchFNO1dConfig:
    history_steps: int
    width: int
    modes: int
    fc_hidden: int
    layers: int = 4


def _extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _looks_like_pdebench_fno1d(checkpoint: Any, *, torch: Any) -> bool:
    state = _extract_state_dict(checkpoint)
    if not isinstance(state, dict):
        return False
    required_keys = _pdebench_fno1d_required_keys()
    return all(key in state and torch.is_tensor(state[key]) for key in required_keys)


def _pdebench_fno1d_required_keys() -> tuple[str, ...]:
    keys: list[str] = ["fc0.weight", "fc0.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    for index in range(4):
        keys.extend(
            [
                f"conv{index}.weights1",
                f"w{index}.weight",
                f"w{index}.bias",
            ]
        )
    return tuple(keys)


def _infer_pdebench_fno1d_config(state: Any) -> PDEBenchFNO1dConfig:
    if not isinstance(state, dict):
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: not a dict")
    missing = [key for key in _pdebench_fno1d_required_keys() if key not in state]
    if missing:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: missing keys "
            + ", ".join(missing)
        )

    fc0_weight = state["fc0.weight"]
    fc1_weight = state["fc1.weight"]
    fc2_weight = state["fc2.weight"]
    conv0_weight = state["conv0.weights1"]
    if fc0_weight.ndim != 2:
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: fc0.weight must be rank 2")
    if conv0_weight.ndim != 3:
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: conv0.weights1 must be rank 3")
    width = int(fc0_weight.shape[0])
    history_steps = int(fc0_weight.shape[1]) - 1
    modes = int(conv0_weight.shape[-1])
    if fc1_weight.ndim != 2 or fc2_weight.ndim != 2:
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: fc weights must be rank 2")
    fc_hidden = int(fc1_weight.shape[0])
    config = PDEBenchFNO1dConfig(
        history_steps=history_steps,
        width=width,
        modes=modes,
        fc_hidden=fc_hidden,
    )
    _validate_pdebench_fno1d_shapes(state=state, config=config)
    return config


def _validate_pdebench_fno1d_shapes(
    *,
    state: dict[str, Any],
    config: PDEBenchFNO1dConfig,
) -> None:
    if config.history_steps < 1:
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: history_steps must be >= 1")
    if config.width < 1:
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: width must be >= 1")
    if config.modes < 1:
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: modes must be >= 1")
    if tuple(state["fc0.weight"].shape) != (config.width, config.history_steps + 1):
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: invalid fc0.weight shape")
    if tuple(state["fc0.bias"].shape) != (config.width,):
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: invalid fc0.bias shape")
    for index in range(config.layers):
        if tuple(state[f"conv{index}.weights1"].shape) != (
            config.width,
            config.width,
            config.modes,
        ):
            raise FNOBackendError(
                f"unsupported PDEBench FNO1d state dict: invalid conv{index}.weights1 shape"
            )
        if tuple(state[f"w{index}.weight"].shape) != (config.width, config.width, 1):
            raise FNOBackendError(
                f"unsupported PDEBench FNO1d state dict: invalid w{index}.weight shape"
            )
        if tuple(state[f"w{index}.bias"].shape) != (config.width,):
            raise FNOBackendError(
                f"unsupported PDEBench FNO1d state dict: invalid w{index}.bias shape"
            )
    if tuple(state["fc1.weight"].shape) != (config.fc_hidden, config.width):
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: invalid fc1.weight shape")
    if tuple(state["fc1.bias"].shape) != (config.fc_hidden,):
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: invalid fc1.bias shape")
    if tuple(state["fc2.weight"].shape) != (1, config.fc_hidden):
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: invalid fc2.weight shape")
    if tuple(state["fc2.bias"].shape) != (1,):
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: invalid fc2.bias shape")
```

Add model classes after the config helpers:

```python
def _make_spectral_conv1d(torch: Any) -> type:
    class SpectralConv1d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes
            scale = 1.0 / (in_channels * out_channels)
            self.weights1 = torch.nn.Parameter(
                scale
                * torch.randn(
                    in_channels,
                    out_channels,
                    modes,
                    dtype=torch.cfloat,
                )
            )

        def forward(self, x: Any) -> Any:
            batch_size = x.shape[0]
            space_points = x.shape[-1]
            x_ft = torch.fft.rfft(x)
            if self.modes > x_ft.shape[-1]:
                raise FNOBackendError(
                    "PDEBench FNO1d modes exceed available Fourier modes for input grid"
                )
            out_ft = torch.zeros(
                batch_size,
                self.out_channels,
                x_ft.shape[-1],
                device=x.device,
                dtype=torch.cfloat,
            )
            out_ft[:, :, : self.modes] = torch.einsum(
                "bix,iox->box",
                x_ft[:, :, : self.modes],
                self.weights1,
            )
            return torch.fft.irfft(out_ft, n=space_points)

    return SpectralConv1d


def _make_pdebench_fno1d_class(torch: Any) -> type:
    SpectralConv1d = _make_spectral_conv1d(torch)

    class PDEBenchFNO1d(torch.nn.Module):
        def __init__(self, config: PDEBenchFNO1dConfig) -> None:
            super().__init__()
            self.config = config
            self.fc0 = torch.nn.Linear(config.history_steps + 1, config.width)
            for index in range(config.layers):
                setattr(
                    self,
                    f"conv{index}",
                    SpectralConv1d(config.width, config.width, config.modes),
                )
                setattr(
                    self,
                    f"w{index}",
                    torch.nn.Conv1d(config.width, config.width, 1),
                )
            self.fc1 = torch.nn.Linear(config.width, config.fc_hidden)
            self.fc2 = torch.nn.Linear(config.fc_hidden, 1)

        def forward(self, x: Any) -> Any:
            x = self.fc0(x)
            x = x.permute(0, 2, 1)
            for index in range(self.config.layers):
                x1 = getattr(self, f"conv{index}")(x)
                x2 = getattr(self, f"w{index}")(x)
                x = x1 + x2
                if index != self.config.layers - 1:
                    x = torch.nn.functional.gelu(x)
            x = x.permute(0, 2, 1)
            x = torch.nn.functional.gelu(self.fc1(x))
            x = self.fc2(x)
            return x.squeeze(-1)

    return PDEBenchFNO1d
```

- [ ] **Step 4: Run schema inference test and FNO file tests**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py::test_infer_pdebench_fno1d_config_from_state_dict -q
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py -q
```

Expected result: schema inference test passes and existing FNO tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/pde_agent/core/fno_backend.py tests/test_fno_backend_v0.py
git commit -m "feat: infer official fno checkpoint schema"
```

---

### Task 2: Official FNO1d Rollout Prediction

**Files:**
- Modify: `src/pde_agent/core/fno_backend.py`
- Modify: `tests/test_fno_backend_v0.py`

- [ ] **Step 1: Add failing official prediction and validation tests**

Append these tests to `tests/test_fno_backend_v0.py` after `test_predict_fno_local_fixture_writes_prediction`:

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_predict_fno_local_official_schema_writes_prediction(tmp_path: Path) -> None:
    import torch

    np = pytest.importorskip("numpy")
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    output_path = tmp_path / "official_prediction.npz"
    context, target = _write_windows(windows_path)
    torch.save(
        {"model_state_dict": _official_fno_state_dict(torch=torch, history_steps=3, width=4, modes=2, fc_hidden=5)},
        checkpoint_path,
    )

    from pde_agent.core.fno_backend import predict_fno_local

    summary = predict_fno_local(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        metadata_path=None,
        force=False,
    )

    assert output_path.exists()
    with np.load(output_path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
        metadata = json.loads(str(archive["metadata_json"]))
    assert prediction.shape == target.shape
    assert prediction.dtype == np.float32
    assert np.all(np.isfinite(prediction))
    assert metadata["adapter_type"] == "pdebench_fno1d"
    assert metadata["history_steps"] == 3
    assert metadata["modes"] == 2
    assert metadata["width"] == 4
    assert summary["adapter_type"] == "pdebench_fno1d"
    assert summary["prediction_shape"] == list(target.shape)
    assert context.shape[0] == prediction.shape[0]
```

Append short-context test:

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_predict_fno_local_official_schema_rejects_short_context(tmp_path: Path) -> None:
    import torch

    np = pytest.importorskip("numpy")
    windows_path = tmp_path / "short_windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    context = np.zeros((2, 2, 8), dtype=np.float32)
    target = np.zeros((2, 3, 8), dtype=np.float32)
    np.savez(windows_path, context=context, target=target)
    torch.save(
        {"model_state_dict": _official_fno_state_dict(torch=torch, history_steps=3, width=4, modes=2, fc_hidden=5)},
        checkpoint_path,
    )

    from pde_agent.core.fno_backend import predict_fno_local

    with pytest.raises(FNOBackendError, match="context_steps|history"):
        predict_fno_local(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "prediction.npz",
            metadata_path=None,
            force=False,
        )
```

Append large-modes test:

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_predict_fno_local_official_schema_rejects_large_modes_for_grid(tmp_path: Path) -> None:
    import torch

    np = pytest.importorskip("numpy")
    windows_path = tmp_path / "small_grid_windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    context = np.zeros((2, 4, 4), dtype=np.float32)
    target = np.zeros((2, 3, 4), dtype=np.float32)
    np.savez(windows_path, context=context, target=target)
    torch.save(
        {"model_state_dict": _official_fno_state_dict(torch=torch, history_steps=3, width=4, modes=4, fc_hidden=5)},
        checkpoint_path,
    )

    from pde_agent.core.fno_backend import predict_fno_local

    with pytest.raises(FNOBackendError, match="modes|Fourier"):
        predict_fno_local(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "prediction.npz",
            metadata_path=None,
            force=False,
        )
```

- [ ] **Step 2: Run official prediction tests and verify RED**

Run:

```bash
uv run --frozen --extra torch pytest \
  tests/test_fno_backend_v0.py::test_predict_fno_local_official_schema_writes_prediction \
  tests/test_fno_backend_v0.py::test_predict_fno_local_official_schema_rejects_short_context \
  tests/test_fno_backend_v0.py::test_predict_fno_local_official_schema_rejects_large_modes_for_grid \
  -q
```

Expected before implementation: fail because `predict_fno_local` still rejects official schema as unsupported or lacks official rollout dispatch.

- [ ] **Step 3: Implement official prediction dispatch**

Modify `predict_fno_local` in `src/pde_agent/core/fno_backend.py`. Replace:

```python
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
```

with:

```python
    loaded = _load_checkpoint_with_metadata(checkpoint_file, torch=torch)
    checkpoint = loaded["checkpoint"]
    if metadata_path is not None:
        _require_file(Path(metadata_path), "metadata")

    prediction, metadata = _predict_checkpoint(
        checkpoint=checkpoint,
        context=context,
        forecast_steps=target.shape[1],
        checkpoint_file=checkpoint_file,
        archive_member=loaded["archive_member"],
        torch=torch,
    )
```

Remove the earlier line in the function that loaded checkpoint directly:

```python
    checkpoint = _load_checkpoint(checkpoint_file, torch=torch)
```

Add this helper after `_predict_fixture_rule`:

```python
def _predict_checkpoint(
    *,
    checkpoint: Any,
    context: np.ndarray,
    forecast_steps: int,
    checkpoint_file: Path,
    archive_member: str | None,
    torch: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(checkpoint, dict) and checkpoint.get("model_type") == "fno_v0_fixture":
        prediction_rule = _fixture_prediction_rule(checkpoint)
        prediction = _predict_fixture_rule(
            context=context,
            forecast_steps=forecast_steps,
            prediction_rule=prediction_rule,
        )
        metadata = {
            "model_type": "fno",
            "adapter_type": "fixture_delta_extrapolation",
            "checkpoint_path": str(checkpoint_file),
            "checkpoint_format": "torch_serialized",
            "prediction_rule": prediction_rule,
            "sample_count": int(prediction.shape[0]),
            "forecast_steps": int(prediction.shape[1]),
            "space_points": int(prediction.shape[2]),
        }
        if archive_member is not None:
            metadata["archive_member"] = archive_member
        return prediction, metadata

    if _looks_like_pdebench_fno1d(checkpoint, torch=torch):
        return _predict_pdebench_fno1d_checkpoint(
            checkpoint=checkpoint,
            context=context,
            forecast_steps=forecast_steps,
            checkpoint_file=checkpoint_file,
            archive_member=archive_member,
            torch=torch,
        )

    raise FNOBackendError("unsupported FNO checkpoint schema")
```

Add official rollout helpers after `_predict_checkpoint`:

```python
def _predict_pdebench_fno1d_checkpoint(
    *,
    checkpoint: Any,
    context: np.ndarray,
    forecast_steps: int,
    checkpoint_file: Path,
    archive_member: str | None,
    torch: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    state = _extract_state_dict(checkpoint)
    config = _infer_pdebench_fno1d_config(state)
    prediction = _rollout_pdebench_fno1d(
        state=state,
        config=config,
        context=context,
        forecast_steps=forecast_steps,
        torch=torch,
        device="cpu",
    )
    metadata: dict[str, Any] = {
        "model_type": "fno",
        "adapter_type": "pdebench_fno1d",
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_format": "torch_serialized",
        "history_steps": config.history_steps,
        "modes": config.modes,
        "width": config.width,
        "fc_hidden": config.fc_hidden,
        "rollout_steps": int(forecast_steps),
        "device": "cpu",
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
    }
    if archive_member is not None:
        metadata["archive_member"] = archive_member
    return prediction, metadata


def _rollout_pdebench_fno1d(
    *,
    state: dict[str, Any],
    config: PDEBenchFNO1dConfig,
    context: np.ndarray,
    forecast_steps: int,
    torch: Any,
    device: str,
) -> np.ndarray:
    if context.shape[1] < config.history_steps:
        raise FNOBackendError(
            "context_steps must be >= official FNO history steps "
            f"({context.shape[1]} < {config.history_steps})"
        )
    available_modes = (context.shape[2] // 2) + 1
    if config.modes > available_modes:
        raise FNOBackendError(
            "PDEBench FNO1d modes exceed available Fourier modes for input grid "
            f"({config.modes} > {available_modes})"
        )
    PDEBenchFNO1d = _make_pdebench_fno1d_class(torch)
    model = PDEBenchFNO1d(config).to(device)
    model.load_state_dict(state)
    model.eval()
    history = torch.as_tensor(
        context[:, -config.history_steps :, :],
        dtype=torch.float32,
        device=device,
    )
    grid = torch.linspace(
        0.0,
        1.0,
        context.shape[2],
        dtype=torch.float32,
        device=device,
    ).view(1, context.shape[2], 1)
    grid = grid.repeat(context.shape[0], 1, 1)

    predictions: list[Any] = []
    with torch.no_grad():
        for _ in range(forecast_steps):
            model_input = torch.cat([history.permute(0, 2, 1), grid], dim=-1)
            next_step = model(model_input)
            if not torch.isfinite(next_step).all():
                raise FNOBackendError("prediction contains non-finite values")
            predictions.append(next_step)
            history = torch.cat([history[:, 1:, :], next_step.unsqueeze(1)], dim=1)
    prediction_tensor = torch.stack(predictions, dim=1)
    prediction = prediction_tensor.detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(prediction)):
        raise FNOBackendError("prediction contains non-finite values")
    return prediction
```

- [ ] **Step 4: Run official prediction tests and FNO tests**

Run:

```bash
uv run --frozen --extra torch pytest \
  tests/test_fno_backend_v0.py::test_predict_fno_local_official_schema_writes_prediction \
  tests/test_fno_backend_v0.py::test_predict_fno_local_official_schema_rejects_short_context \
  tests/test_fno_backend_v0.py::test_predict_fno_local_official_schema_rejects_large_modes_for_grid \
  -q

uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py -q
```

Expected result: official prediction tests pass and existing fixture/CLI tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/pde_agent/core/fno_backend.py tests/test_fno_backend_v0.py
git commit -m "feat: predict with official fno checkpoint adapter"
```

---

### Task 3: CLI Metrics, README, And Real Smoke

**Files:**
- Modify: `src/pde_agent/commands/predict_fno_local.py`
- Modify: `tests/test_fno_backend_v0.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing CLI metrics assertion**

In `test_predict_fno_local_cli_writes_payload`, after:

```python
    assert payload["metrics"]["model_type"] == "fno"
```

add:

```python
    assert payload["metrics"]["adapter_type"] == "fixture_delta_extrapolation"
```

Append this official CLI test:

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO backend tests require torch extra")
def test_predict_fno_local_official_schema_cli_reports_adapter_metrics(
    tmp_path: Path,
) -> None:
    import torch
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    output_path = tmp_path / "official_prediction.npz"
    _write_windows(windows_path)
    torch.save(
        {"model_state_dict": _official_fno_state_dict(torch=torch, history_steps=3, width=4, modes=2, fc_hidden=5)},
        checkpoint_path,
    )

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
    assert payload["metrics"]["adapter_type"] == "pdebench_fno1d"
    assert payload["metrics"]["history_steps"] == 3
    assert payload["metrics"]["modes"] == 2
    assert payload["metrics"]["width"] == 4
```

- [ ] **Step 2: Run CLI metrics tests and verify RED**

Run:

```bash
uv run --frozen --extra torch pytest \
  tests/test_fno_backend_v0.py::test_predict_fno_local_cli_writes_payload \
  tests/test_fno_backend_v0.py::test_predict_fno_local_official_schema_cli_reports_adapter_metrics \
  -q
```

Expected before implementation: fail because CLI metrics do not include adapter fields.

- [ ] **Step 3: Add adapter metrics to command wrapper**

Modify `src/pde_agent/commands/predict_fno_local.py`.

Replace the inline `metrics={...}` argument with a local dict:

```python
    metrics = {
        "model_type": summary["model_type"],
        "sample_count": summary["sample_count"],
        "forecast_steps": summary["forecast_steps"],
        "space_points": summary["space_points"],
        "checkpoint_format": summary["checkpoint_format"],
    }
    for key in ("adapter_type", "history_steps", "modes", "width"):
        if key in summary:
            metrics[key] = summary[key]
```

Then pass:

```python
        metrics=metrics,
```

The full function body should be:

```python
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
    metrics = {
        "model_type": summary["model_type"],
        "sample_count": summary["sample_count"],
        "forecast_steps": summary["forecast_steps"],
        "space_points": summary["space_points"],
        "checkpoint_format": summary["checkpoint_format"],
    }
    for key in ("adapter_type", "history_steps", "modes", "width"):
        if key in summary:
            metrics[key] = summary[key]
    return success_payload(
        run_id="local",
        command="predict-fno-local",
        state_hint="FNO_PREDICTION_READY",
        artifacts={"prediction_npz_path": summary["prediction_path"]},
        metrics=metrics,
        next_hint="score-local-rollout",
    )
```

- [ ] **Step 4: Update README FNO Backend V0 section**

Modify the `README.md` `FNO Backend V0` section to state:

```markdown
`predict-fno-local` supports the official PDEBench 1D Burgers FNO checkpoint
schema used by `burgers_FNO.tar`. It reconstructs the FNO1d model from
checkpoint tensor shapes and performs autoregressive rollout from the local
windows context. Training and finetuning are intentionally not included in this
adapter phase.
```

Keep the existing commands. Keep the statement that the command never falls
back to persistence or the spectral baseline while claiming to be FNO.

- [ ] **Step 5: Run tests and optional real smoke**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py -q
uv run --frozen pytest tests -q
```

If both files exist:

```bash
test -f data/pdebench/checkpoints/burgers_FNO.tar && \
test -f data/derived/local_windows_task1_smoke.npz && \
uv run --frozen --extra torch pde-agent predict-fno-local \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-path /tmp/pde-agent-fno-prediction-smoke.npz \
  --force
```

Expected smoke result: command exits 0 and writes `/tmp/pde-agent-fno-prediction-smoke.npz`.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add src/pde_agent/commands/predict_fno_local.py tests/test_fno_backend_v0.py README.md
git commit -m "docs: expose official fno adapter metrics"
```

---

### Task 4: Final Verification

**Files:**
- No expected source edits. If a bug is found, fix the smallest relevant file and add a regression test before committing.

- [ ] **Step 1: Verify non-FNO suite without torch extra**

Run:

```bash
uv run --frozen pytest tests -q
```

Expected result: all tests pass in the current environment. If torch is absent in another environment, FNO tests must skip cleanly.

- [ ] **Step 2: Verify data extra suite**

Run:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected result: all tests pass.

- [ ] **Step 3: Verify torch FNO suite**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py -q
```

Expected result: all FNO tests pass.

- [ ] **Step 4: Verify CLI help and real smoke**

Run:

```bash
uv run --frozen pde-agent predict-fno-local --help
```

Expected output includes:

```text
--windows-path
--checkpoint-path
--output-path
--metadata-path
--force
```

If files exist, run:

```bash
test -f data/pdebench/checkpoints/burgers_FNO.tar && \
test -f data/derived/local_windows_task1_smoke.npz && \
uv run --frozen --extra torch pde-agent predict-fno-local \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-path /tmp/pde-agent-fno-prediction-final.npz \
  --force
```

Expected result: command exits 0 and writes prediction NPZ.

- [ ] **Step 5: Inspect smoke prediction contract**

If `/tmp/pde-agent-fno-prediction-final.npz` exists, run:

```bash
uv run --frozen python - <<'PY'
import json
import numpy as np

path = "/tmp/pde-agent-fno-prediction-final.npz"
with np.load(path, allow_pickle=False) as archive:
    prediction = archive["prediction"]
    metadata = json.loads(str(archive["metadata_json"]))
print(prediction.shape, prediction.dtype)
print(metadata["adapter_type"], metadata["history_steps"], metadata["modes"], metadata["width"])
assert prediction.ndim == 3
assert prediction.dtype == np.float32
assert metadata["adapter_type"] == "pdebench_fno1d"
PY
```

Expected output contains a rank-3 `float32` prediction and `pdebench_fno1d` metadata.

- [ ] **Step 6: Run diff/status checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected result: `git diff --check` prints nothing. `git status --short --branch` shows a clean branch.

- [ ] **Step 7: Final checklist**

Verify directly:

- Torch remains optional in `pyproject.toml`.
- No training command was added.
- No DeepONet, PI-DeepONet, or UNet adapter was added.
- No external PDEBench or neuraloperator dependency was added.
- Existing fixture prediction still works.
- Official FNO checkpoint adapter writes prediction NPZ.
- Unsupported schemas still raise `FNOBackendError`.

No commit is needed for this task unless a fix is required.
