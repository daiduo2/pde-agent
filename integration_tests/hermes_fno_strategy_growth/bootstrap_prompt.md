# Hermes FNO Strategy Growth Protocol

You are running the FNO strategy-growth gate for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Execute commands in the listed order.
- Do not pass raw FNO sweep output directly to `evaluate-strategy-promotion`.
- Do not skip `convert-fno-evidence-summary`.
- Do not skip `recommend-strategy-next-action`.
- Do not call `evaluate-strategy-promotion` before
  `recommend-strategy-next-action`.
- Treat `strategy_decision.json` as the promotion gate routing artifact.
- Do not patch source code.
- Do not download data.
- Do not inspect help output.
- Do not manually inspect generated artifacts with shell commands.
- Do not call lower-level FNO candidate commands directly.
- Stop after `append-strategy-ledger` returns `STRATEGY_LEDGER_APPENDED`.
