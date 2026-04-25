# Local Window Builder And Scorer Design

## Purpose

Build the first real modeling feedback layer for the PDE Agent harness.

The public PDEBench Task-1 training file only supports supervised local validation over approximately 41 reduced time steps. The competition final inference target is 200 reduced time steps, but the public file cannot directly supervise that horizon. This feature therefore adds:

1. A local window builder for `10 -> 31` supervised training/validation.
2. A local scorer that approximates the competition scoring style on available local windows.

This does not implement model training or official checkpoint inference.

## Scope

In scope:

- Read PDEBench Burgers HDF5 files.
- Downsample by `time_reduction=5` and `space_reduction=4`.
- Create deterministic NPZ windows.
- Score predictions against local targets.
- Expose both operations as `pde-agent` CLI commands.
- Write machine-readable JSON outputs for Hermes/Agent consumption.

Out of scope:

- Loading FNO/U-Net checkpoints.
- Training neural networks.
- Generating final `task{N}_pred.hdf5`.
- Implementing the official Frechet score exactly.
- Assuming access to hidden `task1_test.hdf5` or `task2_test.hdf5`.

## Command 1: `build-local-windows`

### CLI

```bash
pde-agent build-local-windows \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output data/derived/local_windows_task1_val.npz \
  --dataset-key tensor \
  --start-index 0 \
  --sample-limit 128 \
  --context-steps 10 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4
```

### Input

Expected HDF5 dataset:

```text
tensor: [sample, raw_time, raw_space]
```

For the public Task-1 file:

```text
raw shape: [10000, 201, 1024]
reduced shape: [10000, 41, 256]
```

### Output NPZ

The command writes:

```text
context:        float32 [N, context_steps, 256]
target:         float32 [N, forecast_steps, 256]
full_reference: float32 [N, context_steps + forecast_steps, 256]
metadata_json:  scalar JSON string
```

For the default public Task-1 window:

```text
context: [N, 10, 256]
target:  [N, 31, 256]
full_reference: [N, 41, 256]
```

`metadata_json` includes:

```json
{
  "problem_id": "PDE_Burgers",
  "source_mode": "pdebench_hdf5",
  "source_path": "...",
  "dataset_key": "tensor",
  "start_index": 0,
  "sample_count": 128,
  "context_steps": 10,
  "forecast_steps": 31,
  "time_reduction": 5,
  "space_reduction": 4,
  "raw_shape": [10000, 201, 1024],
  "reduced_shape": [128, 41, 256]
}
```

### Behavior

- Fail if the dataset key is missing.
- Fail if the source dataset is not rank 3.
- Fail if reduced time is shorter than `context_steps + forecast_steps`.
- Fail if `start_index + sample_limit` exceeds available samples.
- Do not load the entire HDF5 file when `sample_limit` is provided.
- Create parent directories for the output path.
- Refuse to overwrite existing output unless `--force` is provided.

## Command 2: `score-local-rollout`

### CLI

```bash
pde-agent score-local-rollout \
  --prediction data/derived/local_predictions_val.npz \
  --target data/derived/local_windows_task1_val.npz \
  --output runs/local_score.json
```

### Accepted Prediction Formats

Initial implementation supports NPZ prediction files:

```text
prediction: float array [N, T, X]
```

The target NPZ must contain:

```text
target: float array [N, T, X]
```

The scorer may also accept a file where prediction and target are in the same NPZ.

### Metrics

The scorer writes:

```json
{
  "sample_count": 128,
  "time_steps": 31,
  "space_points": 256,
  "rmse": 0.0,
  "nrmse": 0.0,
  "rel_mse": 0.0,
  "rel_mse_capped": 0.0,
  "segments": {
    "early": {"start": 0, "stop": 8, "rel_mse": 0.0, "score": 100.0},
    "middle": {"start": 8, "stop": 16, "rel_mse": 0.0, "score": 100.0},
    "late": {"start": 16, "stop": 31, "rmse": 0.0, "lorentzian_score": 100.0}
  },
  "competition_like_score_short": 100.0,
  "notes": [
    "Short-window proxy score; final competition uses 190 predicted steps.",
    "Frechet component is not implemented because the official feature space is unspecified."
  ]
}
```

### Formula

For each sample and time step:

```text
rel_t = sum_x (pred - gt)^2 / (sum_x gt^2 + eps)
```

Then:

```text
Rel-MSE = mean_t(mean_sample(min(rel_t, 5.0)))
```

For short local windows, use proportional segment boundaries:

```text
early:  first 25% of T
middle: next 25% of T
late:   remaining 50% of T
```

Scores:

```text
early_score  = 100 * exp(-20 * early_rel_mse)
middle_score = 100 * exp(-10 * middle_rel_mse)
late_score   = 100 / (1 + 10 * late_rmse)
total        = 0.25 * early_score + 0.25 * middle_score + 0.5 * late_score
```

This intentionally mirrors the competition structure while remaining honest that it is a short-window proxy.

## Data Flow

```text
PDEBench HDF5
  -> build-local-windows
  -> local_windows.npz
  -> model or baseline prediction
  -> prediction.npz
  -> score-local-rollout
  -> score.json
  -> Hermes reads score.json
  -> Agent selects next experiment
```

## Integration With Current Harness

The new NPZ window output is compatible with the existing `npz_fixture` adapter shape:

```text
context: [N, C, X]
prediction: [N, T, X]
target: [N, T, X]
```

However, `build-local-windows` should not invent predictions. It only writes context/target/full_reference. Predictions will be produced by future model commands or by test fixtures.

`score-local-rollout` is independent from `eval-rollout` for now. `eval-rollout` currently performs diagnostic target-copy behavior for real arrays and should not be overloaded with model scoring until a real model inference path exists.

## Tests

Unit tests must use tiny synthetic HDF5/NPZ files and must not require the 7.7GB PDEBench file.

Required tests:

- Build windows from tiny HDF5 with known values.
- Verify downsampling and slicing.
- Verify metadata JSON content.
- Verify overwrite protection.
- Verify failure on insufficient reduced time.
- Verify perfect prediction score is 100.
- Verify nonzero prediction error reduces score.
- Verify shape mismatch fails clearly.
- Verify CLI commands write expected files.

## Success Criteria

- `uv run --frozen pytest tests -q` passes.
- `uv run --frozen --extra data pytest tests -q` passes.
- A tiny HDF5 can be converted to local windows.
- A perfect prediction NPZ scores 100.
- A noisy prediction NPZ scores below 100.
- Outputs are JSON/NPZ files that Hermes can inspect without special parsing.
