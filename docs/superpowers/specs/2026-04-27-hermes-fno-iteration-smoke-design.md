# Hermes FNO Iteration Smoke Design

## Goal

Add a Hermes-facing protocol fixture for the V1 FNO iteration loop. Hermes
should read a bounded `run_state.yaml`, execute exactly one allowed command,
and leave reviewable artifacts under a fixed output directory.

## Design

Create a separate fixture under `integration_tests/hermes_fno_iteration_smoke`
instead of reusing diagnostic smoke fixtures. This loop performs a bounded
finetune through `run-fno-iteration`, so it is not a read-only diagnostic test.

The fixture contains:

- `run_state.template.yaml` with placeholders for repo root, window file,
  checkpoint, output directory, and train budget.
- `bootstrap_prompt.md` with strict protocol rules.
- `prompt_template.md` for runner scripts and manual Hermes invocation.
- `task_spec.json` describing the single-window FNO iteration task.
- `README.md` with preparation and execution instructions.

Add `scripts/prepare_fno_iteration_smoke.py` to render `run_state.yaml` from
local paths. The script validates that the windows NPZ and checkpoint exist
before writing runtime state, because Hermes should not infer or search for
missing paths.

## Command Contract

The rendered `run_state.yaml` permits exactly one command:

```bash
cd <repo-root> && uv run --frozen --extra torch pde-agent run-fno-iteration ...
```

Required artifacts:

- `<output-dir>/fno_iteration_report.json`
- `<output-dir>/candidate_comparison.json`

Forbidden behavior includes patching code, downloading data, inspecting command
help, probing command surfaces, calling lower-level candidate commands, and
guessing paths.

## Testing

Add tests that render the fixture into a temporary repo layout and assert:

- placeholders are fully resolved,
- `run-fno-iteration` is the only allowed command,
- lower-level FNO commands are not exposed,
- required artifacts point to the rendered output directory,
- missing input files are rejected before `run_state.yaml` is written.
