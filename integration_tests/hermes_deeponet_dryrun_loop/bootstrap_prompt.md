# Hermes DeepONet Dryrun Loop Protocol

You are running a DeepONet dry-run evidence loop for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Execute commands in the listed order.
- Do not patch source code or input data.
- Do not download data.
- Do not inspect help output.
- Do not manually inspect generated artifacts with shell commands.
- Do not skip candidate execution, candidate evaluation, evidence conversion, or ledger append.
- Stop after `append-strategy-ledger` returns `STRATEGY_LEDGER_APPENDED`.
