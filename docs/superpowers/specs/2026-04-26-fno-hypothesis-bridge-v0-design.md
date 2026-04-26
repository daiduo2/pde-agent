# FNO Hypothesis Bridge V0 Design

## Goal

Add a small bridge from `diagnose-fno-rollout` to the existing PDE hypothesis
selector so Hermes can move from one FNO summary artifact to a bounded next
scientific hypothesis without manually passing three artifact paths.

## Current Context

The harness already has:

- `diagnose-fno-rollout`, which writes `fno_rollout_summary.json`.
- `select-pde-hypothesis`, which reads score, rollout-error, and Burgers physics
  JSON artifacts.
- `plan-next-experiment`, which consumes the selected hypothesis.

The gap is operational, not scientific: the FNO summary already contains the
three artifact paths needed by `select-pde-hypothesis`, but Hermes still has to
extract and pass them manually.

## Design

Create one FNO-specific bridge:

```text
fno_rollout_summary.json
  -> select-pde-hypothesis-from-fno-summary
  -> selected_pde_hypothesis.json
  -> plan-next-experiment
```

The bridge must:

- read a `fno_rollout_summary.json` object,
- require `state_hint == FNO_ROLLOUT_DIAGNOSED`,
- require artifact paths for score, rollout error, and Burgers physics checks,
- delegate to the existing `select_pde_hypothesis_from_files` function,
- write the same hypothesis JSON schema as `select-pde-hypothesis`,
- return a normal success payload with `state_hint = PDE_HYPOTHESIS_SELECTED`.

## Non-Goals

- No new hypothesis labels.
- No training or finetuning.
- No duplicated scoring, rollout-error, or physics formulas.
- No change to the existing three-file `select-pde-hypothesis` command.

## CLI Contract

```bash
uv run --frozen pde-agent select-pde-hypothesis-from-fno-summary \
  --summary-path runs/fno/fno_rollout_summary.json \
  --output-path runs/fno/selected_pde_hypothesis.json
```

Expected command payload:

```json
{
  "status": "ok",
  "command": "select-pde-hypothesis-from-fno-summary",
  "state_hint": "PDE_HYPOTHESIS_SELECTED",
  "artifacts": {
    "selected_pde_hypothesis_path": "runs/fno/selected_pde_hypothesis.json"
  },
  "metrics": {
    "selected_hypothesis": "use_scheduled_sampling_rollout_finetune",
    "next_experiment_scope": "short rollout finetune with scheduled sampling ramp"
  },
  "next_hint": "plan-next-experiment"
}
```

## Error Handling

The bridge fails fast if:

- the summary file is missing or is not JSON object,
- `state_hint` is not `FNO_ROLLOUT_DIAGNOSED`,
- `artifacts` is missing or not an object,
- any required artifact path is missing or not a string,
- any delegated artifact cannot be loaded by the existing selector.

## Acceptance

- Unit tests prove valid FNO summary selects the same hypothesis as the
  existing selector.
- Unit tests prove malformed summary fails before writing output.
- CLI test proves the command writes the hypothesis artifact and returns the
  expected command/state hints.
- Existing full tests still pass.
