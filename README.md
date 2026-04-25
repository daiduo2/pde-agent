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
