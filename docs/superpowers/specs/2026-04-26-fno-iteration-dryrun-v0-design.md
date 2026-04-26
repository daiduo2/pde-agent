# FNO Iteration Dry-Run V0 Design

## Goal

Compress the current FNO diagnostic and planning chain into one deterministic
agent-facing command:

```text
diagnose FNO rollout
  -> select PDE hypothesis
  -> plan next experiment
  -> generate experiment config
  -> write iteration manifest
```

The command validates the harness loop before adding real FNO finetuning.

## Command

```bash
uv run --frozen --extra torch pde-agent run-fno-iteration-dryrun \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir runs/fno_iteration_001 \
  --nu 0.001 \
  --dt 1.0 \
  --dx 1.0 \
  --force
```

## Artifacts

The output directory contains:

- `fno_prediction.npz`
- `fno_score.json`
- `fno_rollout_error.json`
- `fno_burgers_physics.json`
- `fno_rollout_summary.json`
- `selected_pde_hypothesis.json`
- `next_experiment_plan.json`
- `experiment_config.json`
- `iteration_manifest.json`

## Design Rules

- Reuse existing core functions:
  - `diagnose_fno_rollout`
  - `select_pde_hypothesis_from_fno_summary`
  - `plan_next_experiment_from_fno_summary`
  - `generate_experiment_config_from_file`
- Do not add training or finetuning.
- Do not duplicate score, physics, hypothesis, or planning formulas.
- Protect existing target artifacts unless `--force` is set.
- On failure, remove artifacts created by the runner but preserve preexisting
  artifacts.

## Manifest Contract

`iteration_manifest.json` contains:

- `schema_version`
- `state_hint = FNO_ITERATION_DRYRUN_COMPLETED`
- `artifacts`
- `metrics`
- `stages`
- `next_command = execute-candidate-experiment`

The CLI returns the same state hint and points to the manifest.

## Acceptance

- Unit test proves a fixture FNO checkpoint creates all expected artifacts.
- Unit test proves existing artifacts are rejected without `--force`.
- CLI test proves command payload includes the manifest path and high-level
  metrics.
- Existing full tests still pass.
