# Hermes DeepONet Strategy Growth Protocol

You are running the DeepONet strategy-growth gate for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Execute commands in the listed order.
- Do not pass raw DeepONet evaluation reports directly to `evaluate-strategy-promotion`.
- Do not skip `build-deeponet-evidence-sweep`.
- Do not skip `convert-operator-evidence-summary`.
- Do not patch source code or input data.
- Do not download data.
- Do not inspect help output.
- Do not manually inspect generated artifacts with shell commands.
- Do not call lower-level training or submission commands directly.
- Stop after `append-strategy-ledger` returns `STRATEGY_LEDGER_APPENDED`.
