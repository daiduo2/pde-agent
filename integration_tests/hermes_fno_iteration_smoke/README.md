# Hermes FNO Iteration Smoke

This fixture is the Hermes-facing protocol for the V1 single-window FNO
iteration loop.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_fno_iteration_smoke.py \
  --windows-path /tmp/pde-agent-window-suite-task1-public/windows_f031.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --max-train-steps 1
```

Then run Hermes from this fixture directory using the rendered prompt. Hermes
must read `run_state.yaml`, `bootstrap_prompt.md`, and `task_spec.json`, then
execute the single allowed `run-fno-iteration` command.

The fixture intentionally exposes only the high-level iteration command. Lower
level candidate execution, evaluation, and comparison commands remain internal
to the harness for this protocol.
