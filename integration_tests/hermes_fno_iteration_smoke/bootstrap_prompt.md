# Hermes FNO Iteration Smoke Protocol

You are running a bounded FNO iteration smoke test for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Stop after `run-fno-iteration` returns `FNO_ITERATION_COMPLETED`.
- Do not manually inspect generated artifacts with shell commands.
- Do not patch source code.
- Do not download data.
- Do not inspect help output.
- Do not probe command surfaces.
- Do not search outside the current directory.
- Do not call lower-level FNO candidate commands directly.
