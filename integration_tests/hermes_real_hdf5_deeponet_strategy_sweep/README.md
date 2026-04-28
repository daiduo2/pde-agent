# Hermes Real HDF5 DeepONet Strategy Sweep

This fixture runs a real-data mini loop:

1. Build a bounded real-HDF5 persistence baseline.
2. Run a tiny DeepONet strategy sweep against the generated windows and baseline
   score.
3. Convert the DeepONet evidence into the shared strategy-growth interface.
4. Route failed evidence into an external-retrieval strategy proposal.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_real_hdf5_deeponet_strategy_sweep.py \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5
```

Run and audit Hermes:

```bash
uv run --frozen python scripts/run_hermes_strategy_growth.py \
  --fixture-root integration_tests/hermes_real_hdf5_deeponet_strategy_sweep \
  --timeout-seconds 600
```
