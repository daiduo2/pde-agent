# Hermes Burgers Spectral Viscosity Benchmark

This fixture checks whether Hermes can run the real-HDF5 Burgers spectral
viscosity benchmark through one constrained command and stop without inspecting
generated artifacts.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_burgers_spectral_viscosity_benchmark.py
```

Run and audit Hermes:

```bash
uv run --frozen python scripts/run_hermes_burgers_spectral_viscosity_benchmark.py \
  --timeout-seconds 600
```
