This is a protocol-compliance smoke test for pde-agent.

Before taking any action, read these files in order using the exact absolute paths below:
1. {run_state_path}
2. {bootstrap_prompt_path}
3. {task_spec_path}
4. {base_config_path}
5. {override_path}

Hard rules:
- Treat {run_state_path} as the source of truth.
- Do not rely on the current working directory.
- Do not search for alternate file locations.
- Do not guess missing paths.
- Do not skip required reads.
- Do not patch code.
- Do not run training.
- Do not inspect command help.
- Do not probe command surfaces.
- Do not call commands outside the allowed set for the current state.
- Use only the exact command strings declared in run_state.yaml.
- Read command outputs as JSON only.
- Do not stop after resolve_config or any intermediate command output.
- Continue until the allowed success transition RESULTS_COLLECTED is reached.
- If required information is missing, stop and report the missing file or field instead of inferring it.

After reading the files, briefly state:
- current_state
- allowed next actions
- forbidden actions

Then continue by following the allowed action sequence only.
