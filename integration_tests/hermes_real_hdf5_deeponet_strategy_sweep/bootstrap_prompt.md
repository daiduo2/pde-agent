# Hermes Real HDF5 DeepONet Strategy Sweep Protocol

You are running a bounded real-HDF5 DeepONet strategy-sweep evidence loop for
`pde-agent`.

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
- Do not call `evaluate-strategy-promotion` in this protocol.
- Do not call `append-strategy-ledger` in this protocol.
- Treat `strategy_decision.json` as the promotion gate routing artifact.
- Stop after `record-strategy-proposal` returns `STRATEGY_PROPOSAL_RECORDED`.
