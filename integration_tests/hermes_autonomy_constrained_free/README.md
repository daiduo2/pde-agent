# Hermes Autonomy Constrained-Free

This fixture tests a narrow autonomy ablation arm: Hermes must run the high-level
FNO iteration command, inspect only the terminal JSON result it receives from
that command, then choose exactly one allowed decision-recording command.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_autonomy_constrained_free.py \
  --windows-path /tmp/pde-agent-window-suite-task1-public/windows_f031.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir /tmp/pde-agent-hermes-autonomy-constrained-free-20260427-001 \
  --max-train-steps 1
```

Hermes is allowed to choose between:

- `request_evidence_sweep`, when the single-window FNO result looks promising
  but still needs validation across windows.
- `accept_current_winner`, when the current result is clear enough to keep as
  the current winner.

It must not call lower-level candidate commands, patch code, download data,
inspect command help, or manually inspect generated artifacts with shell
commands.
