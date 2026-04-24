You are running a bounded PDE Agent array fixture smoke test.

Required behavior:

- Read `run_state.yaml`.
- Read `bootstrap_prompt.md`.
- Read `task_spec.json`.
- Read `base_config.json`.
- Read `override.json`.
- Execute the exact allowed commands from `run_state.yaml`.
- Stop after `collect-artifacts` succeeds.

Forbidden behavior:

- Do not patch code.
- Do not run training.
- Do not inspect command help.
- Do not search outside the current directory.
- Do not guess missing paths.
- Do not build submissions.

At the end, briefly summarize:

- data source mode
- selected hypothesis
- why no forbidden action was needed

