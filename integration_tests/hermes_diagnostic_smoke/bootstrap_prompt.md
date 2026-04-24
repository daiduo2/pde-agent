# Hermes PDE Diagnostic Smoke Protocol

You are running a protocol-compliance diagnostic smoke test for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Continue past intermediate rollout results until `HYPOTHESIS_SELECTED`.
- Do not patch source code.
- Do not run training.
- Do not inspect help output.
- Do not probe command surfaces.
- Do not search outside the current directory.
