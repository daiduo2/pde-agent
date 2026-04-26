# Local Window Suite V0 Design

## Goal

Build a small suite of real-HDF5 local validation windows from the same PDEBench
source file, instead of relying on one smoke NPZ. This lets the FNO loop test
candidate behavior across multiple horizons and larger sample counts while
keeping the existing `context/target/full_reference` NPZ contract unchanged.

For the public Task-1 file currently present in this repo, the raw shape is
`[10000, 201, 1024]`. With `time_reduction = 5` and `space_reduction = 4`, that
becomes `41` reduced time steps and `256` space points. With `context_steps = 10`,
the maximum local forecast horizon is therefore `31`. The 190-step hidden
competition horizon needs the longer validation/test data; this suite is the
public-data stress bridge, not a substitute for hidden evaluation.

## Command

```bash
uv run --frozen --extra data pde-agent build-local-window-suite \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-dir data/derived/window_suites/task1_public \
  --dataset-key tensor \
  --start-index 0 \
  --sample-limit 32 \
  --context-steps 10 \
  --forecast-steps 8 \
  --forecast-steps 16 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4 \
  --force
```

## Artifacts

The command writes:

- `windows_f008.npz`
- `windows_f016.npz`
- `windows_f031.npz`
- `local_window_suite_manifest.json`

Each NPZ is produced by the existing `build_local_windows` core helper, so it is
compatible with current FNO prediction, scoring, diagnosis, and candidate
evaluation commands.

## Manifest Contract

`local_window_suite_manifest.json` includes:

- `state_hint = LOCAL_WINDOW_SUITE_BUILT`
- source path and dataset key
- sample range and sample count
- context steps
- requested forecast steps
- built window records, each with profile id, forecast steps, output path, and
  shapes
- reduced data limits, including `max_forecast_steps`
- `next_command = run-fno-iteration-dryrun`

## Guardrails

- Reject empty `forecast_steps`.
- Reject duplicate forecast horizons.
- Reject horizons that exceed `reduced_time - context_steps` before writing
  artifacts.
- Reject output overwrite unless `--force`.
- Do not change the single-window NPZ schema.
- Do not infer hidden 190-step performance from public 31-step windows.

## Acceptance

- Fixture test proves multiple NPZ files and a manifest are written.
- Fixture test proves impossible horizons are rejected before partial writes.
- CLI returns `LOCAL_WINDOW_SUITE_BUILT`.
- Existing tests still pass.
