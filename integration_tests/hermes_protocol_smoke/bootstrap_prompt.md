You are running a protocol-compliance smoke test for `pde-agent`.

Your job is not to optimize science results. Your job is to obey the external run-state and use the allowed `pde-agent` commands in the correct order.

Required first steps:
1. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/run_state.yaml`
2. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/bootstrap_prompt.md`
3. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/task_spec.json`
4. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/base_config.json`
5. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/override.json`

Hard rules:
- Treat `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/run_state.yaml` as the source of truth.
- Do not rely on the current working directory.
- Do not search for alternate file locations.
- Do not guess missing paths.
- Do not skip required reads.
- Do not patch code.
- Do not run training.
- Do not call commands not allowed by the current state.
- Do not inspect command help.
- Do not probe command surfaces.
- Prefer short, explicit reasoning over broad summaries.

Execution rule:
- Before each command, check that the action is allowed by `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/run_state.yaml`.
- Use only the exact command strings declared in `run_state.yaml`.
- Read command outputs as JSON only.
- If required information is missing, stop and report the missing file or field instead of inferring it.
- Do not stop after `BASELINE_READY`.
- Continue until the allowed success transition `RESULTS_COLLECTED` is reached.

Success condition:
- Reach the allowed success transition for the current state without violating any forbidden action.
