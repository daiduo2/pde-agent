# Official Operator Comparison Design

## Context

The project now has two different UNet/FNO lines:

- trainable local operator-family sweeps for strategy discovery
- official PDEBench checkpoint adapters for competition-provided FNO and UNet

These should not be mixed in the same implementation path. The trainable sweep
answers "what strategy should the agent try next?" The official checkpoint
comparison answers "how do the provided models behave under our local diagnostic
chain?"

## Goal

Build a small official-model comparison chain for Burgers real-HDF5 windows. It
should evaluate persistence, spectral residual ridge, official FNO, and official
UNet on the same windows file, then write a ranked, machine-readable report.

## Non-Goals

- Do not train or finetune FNO or UNet.
- Do not add a Hermes fixture in this step.
- Do not replace `run-burgers-spectral-viscosity-benchmark`; that benchmark
  remains the strategy-sweep benchmark.
- Do not hide failed official models behind fallback predictions.

## Architecture

Create `official_operator_comparison.py` as a lightweight orchestrator:

1. Build a real-HDF5 baseline window and persistence diagnostic bundle by calling
   `run_real_hdf5_baseline_smoke`.
2. Train and score the existing spectral residual ridge baseline on the same
   windows file.
3. Run `diagnose_fno_rollout` against the official FNO checkpoint.
4. Run `diagnose_unet_rollout` against the official UNet checkpoint.
5. Normalize all four summaries into `model_records`.
6. Rank by `competition_like_score_short` descending, then lower `rel_mse`, then
   stable model id.
7. Write `official_operator_comparison_report.json` with state hint
   `OFFICIAL_OPERATOR_COMPARISON_COMPLETED`.

The report should include artifact paths for every sub-run, the leaderboard, the
best model, and a recommendation. Recommendation is intentionally narrow:

- `promote_official_model_candidate` when the best official checkpoint beats
  persistence.
- `continue_strategy_search` when persistence or spectral residual wins.

## CLI

Add `run-official-operator-comparison` with explicit paths:

- `source_path`, `output_dir`, `dataset_key`
- `fno_checkpoint_path`
- `unet_checkpoint_path`
- window parameters: `sample_limit`, `context_steps`, `forecast_steps`,
  `time_reduction`, `space_reduction`
- spectral parameters: `spectral_modes`, `spectral_ridge`
- physics parameters: `nu`, `dt`, `dx`
- `force`

The command returns the standard `success_payload` envelope.

## Testing

Use fixture-friendly tests:

- monkeypatch baseline/spectral/FNO/UNet functions to prove orchestration,
  ranking, report writing, and CLI payload without requiring torch checkpoints.
- test overwrite protection for the report.
- keep real checkpoint verification as a smoke command, not as a unit test.

## Success Criteria

- A single command produces a four-model leaderboard on one windows file.
- The report clearly separates official checkpoint evidence from trainable
  strategy-sweep evidence.
- Existing full tests remain green.
