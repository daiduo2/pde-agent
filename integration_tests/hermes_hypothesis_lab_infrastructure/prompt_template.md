This is a bounded Hypothesis Lab infrastructure canary for pde-agent.

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
- Do not call lower-level Hypothesis Lab or Hermes runtime commands directly.
- Continue until the allowed success transition HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED is reached.
- After the command succeeds, do not inspect generated artifact files manually.
- Do not use cat, ls, find, grep, rg, sed, head, tail, python, or any other shell command unless it is exactly listed under allowed_commands.

At the end, briefly summarize:
- infrastructure audit decision
- check count and failed check count
- why no forbidden action was needed
