# Hermes Hypothesis Lab Infrastructure Protocol

You are running a bounded Hypothesis Lab infrastructure canary for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Stop after the command returns `HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED`.
- Do not manually inspect generated artifacts with shell commands.
- Do not patch source code.
- Do not download data.
- Do not inspect help output.
- Do not probe command surfaces.
- Do not search outside the current directory.
- Do not call lower-level Hypothesis Lab or Hermes commands directly.
