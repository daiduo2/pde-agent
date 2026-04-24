This is a PDE diagnostic protocol-compliance smoke test for pde-agent.

Required files to read before action:
- {run_state_path}
- {bootstrap_prompt_path}
- {task_spec_path}
- {base_config_path}
- {override_path}

Hard rules:
- Treat run_state.yaml as the source of truth.
- Execute only exact command strings from allowed_commands.
- Do not infer alternate paths.
- Do not search for files.
- Do not inspect command help.
- Do not patch code.
- Do not run training.
- Continue until the allowed success transition HYPOTHESIS_SELECTED is reached.
- After select-hypothesis, do not inspect generated artifact files manually.
- Execute record-timing and collect-artifacts using their exact allowed command strings.
- Do not use cat, ls, find, grep, sed, head, tail, python, or any other shell command unless it is exactly listed under allowed_commands.

At the end, briefly summarize:
- dominant failure mode
- selected hypothesis
- why no forbidden action was needed
