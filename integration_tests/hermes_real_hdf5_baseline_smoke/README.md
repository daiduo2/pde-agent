# Hermes Real HDF5 Baseline Smoke

This fixture checks whether Hermes can run the bounded real-HDF5 baseline smoke
protocol without drifting into unrelated commands.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_real_hdf5_baseline_smoke.py \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --sample-limit 2 \
  --forecast-steps 31
```

Run and audit Hermes:

```bash
uv run --frozen python scripts/run_hermes_real_hdf5_baseline.py --timeout-seconds 300
```

Audit the latest saved session without rerunning Hermes:

```bash
uv run --frozen python scripts/run_hermes_real_hdf5_baseline.py --audit-only
```

Hermes must read `run_state.yaml`, `bootstrap_prompt.md`, and `task_spec.json`,
then execute the single exact `run-real-hdf5-baseline-smoke` command.
