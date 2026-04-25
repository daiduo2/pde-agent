You are running a bounded PDE Agent real HDF5 smoke test.

Required behavior:

- Read `run_state.yaml`.
- Read `bootstrap_prompt.md`.
- Read `task_spec.json`.
- Read `base_config.runtime.json`.
- Read `override.json`.
- Execute the exact allowed commands from `run_state.yaml`.
- Stop after the final allowed command succeeds.

Forbidden behavior:

- Do not download data.
- Do not patch code.
- Do not run training.
- Do not inspect command help.
- Do not search outside the current directory.
- Do not guess missing paths.

Important caveat:

Full mode is read-only plumbing validation. Current HDF5 full mode uses target-copy prediction and must not be reported as model performance.
