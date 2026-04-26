# FNO Official Adapter Design

Date: 2026-04-26

## Purpose

Add a real inference adapter for the official PDEBench 1D Burgers FNO
checkpoint. The adapter should consume the existing local windows NPZ contract,
reconstruct the official FNO1d architecture from checkpoint state, run
autoregressive rollout, and write the same `prediction` NPZ consumed by
`score-local-rollout`.

This phase turns FNO Backend V0 from a fixture-only prediction path into a real
checkpoint inference path. It does not add training.

## Current Evidence

The local official checkpoint archive is:

```text
data/pdebench/checkpoints/burgers_FNO.tar
```

It is a tar archive containing multiple `.pt` checkpoints. The Task-1
`Nu0.001` member has top-level keys:

```text
epoch
loss
model_state_dict
optimizer_state_dict
```

The selected `model_state_dict` has 18 tensors:

```text
fc0.weight       (20, 11)
fc0.bias         (20,)
conv0.weights1   (20, 20, 12)
conv1.weights1   (20, 20, 12)
conv2.weights1   (20, 20, 12)
conv3.weights1   (20, 20, 12)
w0.weight        (20, 20, 1)
w0.bias          (20,)
w1.weight        (20, 20, 1)
w1.bias          (20,)
w2.weight        (20, 20, 1)
w2.bias          (20,)
w3.weight        (20, 20, 1)
w3.bias          (20,)
fc1.weight       (128, 20)
fc1.bias         (128,)
fc2.weight       (1, 128)
fc2.bias         (1,)
```

This implies a standard 1D FNO with:

- `history_steps = 10`
- `grid_channels = 1`
- `width = 20`
- `modes = 12`
- four Fourier blocks
- one-step output

## Scope

In scope:

- detect official PDEBench FNO1d checkpoints,
- reconstruct a minimal compatible FNO1d PyTorch module,
- load official checkpoint state dict,
- run local autoregressive rollout from `context` to `target.shape[1]`,
- write existing `prediction` NPZ contract,
- expose adapter metadata in prediction summary,
- keep fixture checkpoint prediction working.

Out of scope:

- FNO training or finetuning,
- scheduled sampling or rollout-loss training,
- DeepONet, PI-DeepONet, or UNet adapters,
- final competition HDF5 submission generation,
- hidden 190-step competition evaluation,
- automatic fallback to persistence or spectral baselines,
- depending on external PDEBench or neuraloperator packages.

## Architecture

Use an internal minimal implementation instead of importing PDEBench or
neuraloperator. The checkpoint schema is compact enough to reconstruct directly,
and keeping the adapter local preserves the harness property: small, auditable,
and easy for Hermes to reason about.

The existing `predict_fno_local` remains the public entry point:

```python
def predict_fno_local(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    ...
```

Internally it should dispatch by checkpoint schema:

1. `model_type == "fno_v0_fixture"`: keep existing fixture prediction rule.
2. official PDEBench FNO1d state dict: use the new official adapter.
3. anything else: raise `FNOBackendError`.

## Model Adapter

Create a focused internal model implementation in `pde_agent.core.fno_backend`.
If the file becomes too large during implementation, split model code into
`pde_agent.core.fno_model`; otherwise keep it local to avoid premature
abstraction.

Required components:

- `PDEBenchFNO1dConfig`
  - `history_steps: int`
  - `width: int`
  - `modes: int`
  - `fc_hidden: int`
  - `layers: int = 4`
- `SpectralConv1d`
  - parameter name must be `weights1` to match checkpoint keys,
  - forward uses `torch.fft.rfft` and `torch.fft.irfft`,
  - only first `modes` Fourier modes are multiplied.
- `PDEBenchFNO1d`
  - modules named `fc0`, `conv0..conv3`, `w0..w3`, `fc1`, `fc2`,
  - input shape `[N, X, history_steps + 1]`,
  - output shape `[N, X]`.

Expected forward structure:

```text
input [N, X, history + grid]
  -> fc0
  -> permute to [N, width, X]
  -> four blocks: spectral conv + 1x1 conv + GELU except after final block
  -> permute to [N, X, width]
  -> fc1 + GELU
  -> fc2
  -> squeeze final channel to [N, X]
```

No padding is included in this phase. If local score reveals obvious boundary
artifacts or mismatch against known PDEBench behavior, padding should be a later
adapter refinement with its own test.

## Schema Inference

Detect official FNO1d using state dict keys and tensor shapes. Accept either:

- top-level checkpoint dict with `model_state_dict`, or
- a raw state dict containing the required keys.

Required keys:

```text
fc0.weight
fc0.bias
conv0.weights1
conv1.weights1
conv2.weights1
conv3.weights1
w0.weight
w0.bias
w1.weight
w1.bias
w2.weight
w2.bias
w3.weight
w3.bias
fc1.weight
fc1.bias
fc2.weight
fc2.bias
```

Infer:

- `width = fc0.weight.shape[0]`
- `history_steps = fc0.weight.shape[1] - 1`
- `modes = conv0.weights1.shape[-1]`
- `fc_hidden = fc1.weight.shape[0]`

Validation:

- `history_steps >= 1`
- all four `conv*.weights1` have shape `[width, width, modes]`
- all four `w*.weight` have shape `[width, width, 1]`
- `fc1.weight.shape == [fc_hidden, width]`
- `fc2.weight.shape == [1, fc_hidden]`
- `fc0.weight` input channels equal `history_steps + 1`

Invalid schema raises `FNOBackendError` with a concise explanation.

## Rollout Data Flow

Input windows:

```text
context: float32 [N, context_steps, X]
target:  float32 [N, forecast_steps, X]
```

Rollout:

1. Validate `context_steps >= history_steps`.
2. Keep only the latest `history_steps` context frames.
3. Create grid `linspace(0, 1, X)` with shape `[N, X, 1]`.
4. For each forecast step:
   - transpose history to `[N, X, history_steps]`,
   - concatenate grid to `[N, X, history_steps + 1]`,
   - run model to get `[N, X]`,
   - append as next step,
   - slide the history window.
5. Stack predictions into `[N, forecast_steps, X]`.

Device:

- default to CPU,
- expose an internal `device` parameter for tests/future training,
- CLI does not need a device flag in this phase.

Inference:

- `model.eval()`
- `torch.no_grad()`
- convert output to `float32` NumPy.

## Metadata

Prediction NPZ metadata should include:

```json
{
  "model_type": "fno",
  "adapter_type": "pdebench_fno1d",
  "checkpoint_path": "...",
  "checkpoint_format": "torch_serialized",
  "archive_member": "1D_Burgers_Sols_Nu0.001_FNO.pt",
  "history_steps": 10,
  "modes": 12,
  "width": 20,
  "fc_hidden": 128,
  "rollout_steps": 31,
  "device": "cpu",
  "sample_count": 4,
  "forecast_steps": 31,
  "space_points": 256
}
```

For fixture prediction, keep existing metadata but add
`adapter_type = "fixture_delta_extrapolation"` if this can be done without
breaking existing tests.

The CLI payload should continue to expose stable top-level metrics:

- `model_type`
- `sample_count`
- `forecast_steps`
- `space_points`
- `checkpoint_format`

It may add:

- `adapter_type`
- `history_steps`
- `modes`
- `width`

## Error Handling

The adapter must fail fast and explicitly:

- missing checkpoint or windows path: existing `FNOBackendError`,
- unsupported checkpoint schema: `FNOBackendError`,
- `context_steps < history_steps`: `FNOBackendError`,
- `modes > X // 2 + 1`: `FNOBackendError`,
- non-finite prediction: `FNOBackendError`.

It must never silently use persistence, spectral residual, fixture prediction,
or random output while claiming to be official FNO.

## Tests

Tests should not require real downloaded PDEBench files.

Required synthetic tests:

1. `test_infer_pdebench_fno1d_config_from_state_dict`
   - create a tiny official-like state dict,
   - infer config,
   - assert `history_steps`, `width`, `modes`, and `fc_hidden`.

2. `test_predict_fno_local_official_schema_writes_prediction`
   - create tiny local windows,
   - create tiny official-like FNO checkpoint with deterministic-compatible
     random weights,
   - run `predict_fno_local`,
   - assert output path, shape, dtype, metadata, and adapter type.

3. `test_predict_fno_local_official_schema_rejects_short_context`
   - create official-like checkpoint with `history_steps = 3`,
   - create windows with `context_steps = 2`,
   - assert `FNOBackendError`.

4. `test_predict_fno_local_official_schema_rejects_large_modes_for_grid`
   - create checkpoint with `modes` larger than `X // 2 + 1`,
   - assert `FNOBackendError`.

5. Existing fixture tests continue to pass.

Optional real smoke:

```bash
uv run --frozen --extra torch pde-agent predict-fno-local \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --metadata-path /tmp/pde-agent-fno-metadata-final.json \
  --output-path /tmp/pde-agent-fno-prediction-smoke.npz \
  --force
```

If the local files are present, this should complete and write a prediction NPZ.
The smoke is not an accuracy guarantee.

## Training Extension Point

Training is intentionally excluded from this phase, but the implementation
should make later training straightforward:

- keep model construction separate from checkpoint loading,
- keep schema inference pure and testable,
- keep rollout as a reusable function,
- avoid CLI-only logic in model code,
- preserve enough metadata to reproduce model dimensions.

The future training phase can then add:

- dataset minibatching from HDF5/local windows,
- optimizer/loss configuration,
- one-step and rollout-loss modes,
- checkpoint writing compatible with this adapter,
- timing and experiment tracking integration.

## Success Criteria

- Existing fixture FNO tests still pass.
- Official-like synthetic checkpoint prediction writes valid prediction NPZ.
- Real `burgers_FNO.tar` prediction smoke completes when local files are
  present.
- Torch remains optional.
- Non-FNO tests still pass without extra setup.
- No training command is added.
- No external PDEBench/neuraloperator dependency is added.
