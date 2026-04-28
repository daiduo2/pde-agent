This is a bounded real-HDF5 baseline protocol-compliance smoke test for pde-agent.

Required files to read before action:
- {run_state_path}
- {bootstrap_prompt_path}
- {task_spec_path}

Hard rules:
- Treat run_state.yaml as the source of truth.
- Execute only exact command strings from allowed_commands.
- Do not infer alternate paths.
- Do not search for files.
- Do not inspect command help.
- Do not patch code.
- Do not download data.
- Do not train a model.
- Do not call FNO, DeepONet, PI-DeepONet, or submission-building commands.
- Continue until the allowed success transition REAL_HDF5_BASELINE_SMOKE_COMPLETED is reached.
- After the command succeeds, do not inspect generated artifact files manually.
- Do not use cat, ls, find, grep, sed, head, tail, python, or any other shell command unless it is exactly listed under allowed_commands.

At the end, briefly summarize:
- sample count
- forecast steps
- competition-like short score
- dominant error pattern
- residual RMSE
- why no forbidden action was needed
