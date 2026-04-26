# FNO Evidence Sweep V0 Design

## Goal

Turn the manual evidence-expansion procedure into a reproducible harness command
that runs a bounded FNO candidate sweep and writes one summary plus one
recommendation.

V0 answers two questions:

- Is `train_steps` a stronger variable than the newly added training strategy?
- Does `scheduled_sampling` remain stable across multiple public-data windows?

## Command

```bash
uv run --frozen --extra torch --extra data pde-agent run-fno-evidence-sweep \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir runs/fno_evidence_sweep_001 \
  --dataset-key tensor \
  --start-index 0 \
  --start-index 128 \
  --start-index 512 \
  --sample-limit 8 \
  --context-steps 10 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4 \
  --train-steps 1 \
  --train-steps 3 \
  --train-steps 5 \
  --train-steps 8 \
  --learning-rate 1e-4 \
  --force
```

V0 accepts one forecast horizon. Multi-horizon evidence can be run as multiple
commands or added in a later version.

## Orchestration

For each `start_index`:

1. Build a local window suite with one `forecast_steps` value.
2. Run `run_fno_iteration_dryrun` for the generated window.
3. Audit the generated experiment config.

For each `train_steps` and each audited candidate:

1. Prepare an isolated candidate workspace.
2. Execute `execute_fno_finetune_candidate`.
3. Evaluate the candidate.
4. Compare candidates for that `(start_index, train_steps)` cell.

## Artifacts

The output directory contains:

- `windows/start_<N>/...`
- `baselines/start_<N>/...`
- `sweeps/start_<N>/steps_<K>/...`
- `evidence_summary.json`
- `evidence_summary.csv`

## Summary Contract

`evidence_summary.json` includes:

- `state_hint = FNO_EVIDENCE_SWEEP_COMPLETED`
- inputs and sweep dimensions
- flat candidate records
- per-cell winners
- aggregate strategy statistics
- recommendation
- `next_command = run-fno-evidence-sweep` or `run-fno-iteration-dryrun`

The recommendation says whether to add more strategies. V0 recommends against
adding more strategies when one strategy wins every completed comparison and all
strategies improve over baseline.

## Guardrails

- Reject empty `start_indices` or `train_steps`.
- Reject duplicate `start_indices` or `train_steps`.
- Reject output overwrite unless `--force`.
- Keep every run under `output_dir`; do not write into repo data directories.
- Do not add new training algorithms.

## Acceptance

- Fixture test proves a small official-like FNO checkpoint can run a one-cell
  sweep and write JSON/CSV summaries.
- CLI returns `FNO_EVIDENCE_SWEEP_COMPLETED`.
- Real smoke can run on the public Task-1 HDF5 using the official checkpoint.
