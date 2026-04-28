# Hermes Real HDF5 Baseline Smoke Protocol

You are running a bounded real-HDF5 baseline smoke test for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Stop after `run-real-hdf5-baseline-smoke` returns `REAL_HDF5_BASELINE_SMOKE_COMPLETED`.
- Do not inspect generated artifacts manually.
- Do not patch source code.
- Do not download data.
- Do not inspect help output.
- Do not probe command surfaces.
- Do not search outside the current directory.
- Do not train a model.
- Do not call FNO, DeepONet, or submission-building commands.
