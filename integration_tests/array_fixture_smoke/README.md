# Hermes Array Fixture Smoke

This fixture validates the real-array diagnostic command chain with a generated NPZ fixture and checkout-local runtime state.

It is bounded to a tiny local NPZ array fixture and does not use full PDEBench HDF5 data, checkpoints, training, submission builds, or code patching. It exists to prove that Hermes can follow the diagnostic protocol after `inspect-data` is introduced.

Before launching Hermes, generate the ignored runtime files for the current checkout:

```bash
uv run python scripts/prepare_array_fixture_smoke.py
```

The preparation step writes:

- `integration_tests/array_fixture_smoke/fixture.npz`
- `integration_tests/array_fixture_smoke/base_config.runtime.json`
- `integration_tests/array_fixture_smoke/run_state.yaml`

Use the generated `run_state.yaml` as the Hermes source of truth. The committed `run_state.template.yaml` is portable and contains placeholders only.
