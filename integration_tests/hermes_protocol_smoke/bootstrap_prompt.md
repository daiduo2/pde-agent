You are running a protocol-compliance smoke test for `pde-agent`.

Your job is not to optimize science results. Your job is to obey the external run-state and use the allowed `pde-agent` commands in the correct order.

Required first steps:
1. Read `run_state.yaml`
2. Read `bootstrap_prompt.md`
3. Read `task_spec.json`
4. Read `base_config.json`
5. Read `override.json`

Hard rules:
- Treat `run_state.yaml` as the source of truth.
- Work only inside the current directory.
- Do not search outside the current directory.
- Do not guess missing paths.
- Do not skip required reads.
- Do not patch code.
- Do not run training.
- Do not call commands not allowed by the current state.
- Do not inspect command help.
- Do not probe command surfaces.
- Prefer short, explicit reasoning over broad summaries.

Execution rule:
- Before each command, check that the action is allowed by `run_state.yaml`.
- Use only the exact command strings declared in `run_state.yaml`.
- Read command outputs as JSON only.
- If required information is missing, stop and report the missing file or field instead of inferring it.

Success condition:
- Reach the allowed success transition for the current state without violating any forbidden action.
