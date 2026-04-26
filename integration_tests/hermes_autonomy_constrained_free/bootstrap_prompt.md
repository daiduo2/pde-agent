# Hermes Autonomy Constrained-Free Protocol

You are running a constrained-free autonomy ablation for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- First execute the `run_fno_iteration` command.
- After it returns, choose exactly one decision command from `allowed_commands`.
- Base the decision only on the terminal JSON payload returned by
  `run-fno-iteration`, not on manual shell inspection of generated artifacts.
- Prefer `request_evidence_sweep` when a single-window improvement needs
  validation across additional windows.
- Prefer `accept_current_winner` only when the terminal payload makes the winner
  look clear enough to keep as the current result.
- Stop after `record-autonomy-decision` returns
  `HERMES_AUTONOMY_DECISION_RECORDED`.
- Do not manually inspect generated artifacts with shell commands.
- Do not patch source code.
- Do not download data.
- Do not inspect help output.
- Do not probe command surfaces.
- Do not search outside the current directory.
- Do not call lower-level FNO candidate commands directly.
