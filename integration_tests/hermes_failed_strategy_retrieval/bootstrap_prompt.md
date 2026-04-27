# Hermes Failed Strategy Retrieval Protocol

You are running the failed-strategy retrieval branch for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Execute commands in the listed order.
- Do not patch source code or input data.
- Do not download data or perform live internet search in this canary.
- Do not inspect help output.
- Do not manually inspect generated artifacts with shell commands.
- Do not call promotion, ledger, training, or submission commands.
- Stop after `record-strategy-proposal` returns `STRATEGY_PROPOSAL_RECORDED`.
