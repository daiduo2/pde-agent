# Lightweight Spectral Baseline Design

Date: 2026-04-26

## Purpose

Add the first trainable local model to `pde-agent` without introducing a deep-learning framework dependency.

The current system can build local Burgers windows, generate a persistence baseline, score predictions, diagnose rollout errors, check Burgers physics proxies, and select a PDE hypothesis. The missing step is a real model experiment that trains on local windows and produces predictions that can flow through the diagnostic chain.

This phase introduces a NumPy-only spectral residual baseline:

```text
build-local-windows
  -> train-spectral-residual-baseline
  -> predict-local-model
  -> score-local-rollout
  -> diagnose-rollout-error
  -> check-burgers-physics-local
  -> select-pde-hypothesis
```

## Design Position

This is not intended to compete with FNO. It is a controlled first model loop.

The baseline should be stronger than persistence but much simpler than a neural operator. It gives Hermes a trainable artifact, a checkpoint, a prediction file, and diagnostic evidence. That is enough to validate the model-experiment loop before adding PyTorch, checkpoint loading, or FNO/UNet code.

PyTorch should remain an explicit future extension. If later experiments show that convolutional or neural-operator baselines are necessary, the project should add a separate optional dependency group such as `torch` or `models`, and isolate torch code behind separate commands. This phase must not require PyTorch.

## Model

The model is a ridge-regression residual forecaster in Fourier-feature space.

Inputs:

- `context`: `[N, context_steps, X]`
- `target`: `[N, forecast_steps, X]`

Baseline:

```text
persistence = repeat(context[:, -1:, :], forecast_steps, axis=1)
residual_target = target - persistence
```

Features:

- recent temporal differences from context,
- final context frame,
- low Fourier modes of final context frame,
- optional low Fourier modes of the last temporal difference.

Target:

- low Fourier coefficients of residual_target for each forecast step.

Training:

- solve ridge regression in closed form with NumPy,
- save checkpoint as NPZ,
- include scalar `metadata_json`.

Prediction:

- load checkpoint,
- compute features from new context,
- predict low Fourier residual coefficients,
- inverse transform to physical space,
- output `prediction = persistence + residual_prediction`.

This should be deterministic, CPU-friendly, and fast on small sample limits.

## Commands

### `train-spectral-residual-baseline`

Input:

- `--windows-path`: local windows NPZ containing `context` and `target`
- `--checkpoint-path`: checkpoint NPZ path
- `--modes`: number of low Fourier modes, default `16`
- `--ridge`: ridge penalty, default `1e-3`
- `--force`: overwrite checkpoint

Output:

- checkpoint NPZ
- JSON payload with `state_hint`: `LOCAL_MODEL_TRAINED`

Checkpoint keys:

- `weights_real`
- `weights_imag`
- `feature_mean`
- `feature_scale`
- `metadata_json`

Metadata:

- model type: `spectral_residual_ridge`
- modes
- ridge
- context_steps
- forecast_steps
- space_points
- feature_dim
- train_sample_count

### `predict-local-model`

Input:

- `--windows-path`: local windows NPZ containing `context` and `target`
- `--checkpoint-path`: checkpoint NPZ from training
- `--output-path`: prediction NPZ path
- `--force`: overwrite prediction

Output:

- prediction NPZ containing `prediction`
- JSON payload with `state_hint`: `LOCAL_MODEL_PREDICTED`

Prediction metadata:

- model type
- checkpoint path
- sample count
- forecast steps
- space points

## Error Handling

Raise command-specific `RuntimeError` subclasses for:

- missing input path,
- missing NPZ keys,
- non-rank-3 arrays,
- non-finite arrays,
- shape mismatch between context and target,
- invalid `modes`,
- invalid `ridge`,
- output exists without `--force`,
- checkpoint metadata incompatible with windows.

All JSON writes must use `allow_nan=False`, or validate finite values before writing.

## Testing Strategy

Unit tests should use tiny NPZ windows and should not require real HDF5.

Required tests:

- training writes checkpoint with expected keys and metadata,
- prediction writes exact requested output path, including no-suffix paths,
- prediction shape matches target shape,
- model improves over persistence on a simple synthetic linear-trend fixture,
- invalid modes/ridge fail clearly,
- incompatible checkpoint/windows fail clearly,
- CLI smoke tests for train and predict,
- end-to-end local model chain: train, predict, score, diagnose, physics check, select hypothesis.

The model-improvement test should use a deterministic fixture where persistence is visibly wrong:

```text
context[t, x] = base[x] + velocity[x] * t
target[k, x] = base[x] + velocity[x] * (context_steps + k)
```

The residual ridge model should learn the trend from temporal differences and beat persistence on RMSE.

## PyTorch Extension Policy

PyTorch is not part of this phase.

If PyTorch becomes necessary later:

- add it under an optional dependency group, not as a mandatory base dependency,
- create separate commands such as `train-torch-local-model`,
- keep NumPy spectral baseline as a fast smoke baseline,
- keep checkpoints and predictions compatible with existing scoring/diagnostic commands,
- avoid mixing torch-specific device logic into the NumPy baseline modules.

This keeps the current harness lightweight while leaving a clean path to Mini-FNO or convolutional baselines.

## Success Criteria

- A local windows NPZ can produce a trained checkpoint and prediction NPZ without PyTorch.
- The prediction NPZ can be consumed by existing `score-local-rollout` and PDE diagnostic skills.
- The model beats persistence on a deterministic synthetic trend test.
- The full test suite remains fast.
- The design leaves a clear future path for PyTorch without making it a current dependency.

## Deferred Work

- PyTorch installation and device management.
- Mini-FNO or Conv1D autoregressive baseline.
- Official PDEBench checkpoint loading.
- Long-horizon `10 -> 190` rollout generation.
- Task 2 multi-nu conditioning.

