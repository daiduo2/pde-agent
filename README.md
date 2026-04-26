# pde-agent

CLI-first diagnostic harness for PDE agent experiments.

Initial MVP goals:

- strict JSON-only CLI contract
- explicit `run_id` and run directory tree
- resolved config generation
- manifest-centered artifact contract
- Thin-Hermes integration hooks

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

## Lightweight Spectral Baseline

The lightweight spectral residual baseline is a NumPy-only local smoke model for fast diagnostic loops. It trains a small ridge-regression residual model from a local windows NPZ and writes checkpoint/prediction NPZ artifacts that are compatible with the existing scoring and PDE diagnostic commands. It is intended for deterministic local sanity checks, not as the final contest model.

```bash
uv run --frozen pde-agent train-spectral-residual-baseline \
  --windows-path data/derived/local_windows_task1_val.npz \
  --checkpoint-path runs/spectral_residual_checkpoint.npz \
  --modes 16 \
  --ridge 1e-3

uv run --frozen pde-agent predict-local-model \
  --windows-path data/derived/local_windows_task1_val.npz \
  --checkpoint-path runs/spectral_residual_checkpoint.npz \
  --output-path data/derived/local_spectral_prediction_task1_val.npz

uv run --frozen pde-agent score-local-rollout \
  --prediction-path data/derived/local_spectral_prediction_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_spectral_score.json

uv run --frozen pde-agent diagnose-rollout-error \
  --prediction-path data/derived/local_spectral_prediction_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --score-path runs/local_spectral_score.json \
  --output-path runs/local_spectral_rollout_error.json

uv run --frozen pde-agent check-burgers-physics-local \
  --prediction-path data/derived/local_spectral_prediction_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_spectral_burgers_physics.json \
  --nu 0.001

uv run --frozen pde-agent select-pde-hypothesis \
  --score-path runs/local_spectral_score.json \
  --error-diagnosis-path runs/local_spectral_rollout_error.json \
  --physics-check-path runs/local_spectral_burgers_physics.json \
  --output-path runs/local_spectral_pde_hypothesis.json

uv run --frozen pde-agent plan-next-experiment \
  --score-path runs/local_spectral_score.json \
  --error-diagnosis-path runs/local_spectral_rollout_error.json \
  --physics-check-path runs/local_spectral_burgers_physics.json \
  --hypothesis-path runs/local_spectral_pde_hypothesis.json \
  --output-path runs/local_spectral_next_experiment_plan.json
```

`plan-next-experiment` writes machine-readable JSON only. It does not modify model
code, execute experiments, or write human-readable research notes. The planned
future next step is `generate-experiment-config`, which will convert the JSON plan
into an executable experiment configuration.
