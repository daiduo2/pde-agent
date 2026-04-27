# Hermes DeepONet Strategy Sweep Protocol

You are running a DeepONet strategy-sweep evidence loop for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Execute commands in the listed order.
- Do not patch source code or input data.
- Do not download data.
- Do not inspect help output.
- Do not manually inspect generated artifacts with shell commands.
- Do not run cleanup, move, delete, or retry commands if an allowed command fails.
- Stop after `append-strategy-ledger` returns `STRATEGY_LEDGER_APPENDED`.
