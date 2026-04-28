# Burgers Spectral Viscosity Benchmark Design

## Purpose

This project serves an agent competition now and an ai4pde research workflow later. The next step is to turn the retrieved strategy `burgers_spectral_viscosity_real_hdf5_v0` from a proposal artifact into a quantified, repeatable Burgers benchmark chain.

The chain must answer one narrow question: does a spectral-viscosity DeepONet candidate improve real-HDF5 Burgers rollout evidence against the existing persistence and DeepONet controls across multiple local windows?

It must not claim final contest performance. It is a bounded local benchmark for agent decision-making, strategy ledger evidence, and future ai4pde workflow reuse.

## Scope

The first version targets the existing public Burgers file:

- `data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5`
- dataset key `tensor`
- local reduced windows with `context_steps=10`, `forecast_steps=31`, `time_reduction=5`, `space_reduction=4`
- `nu=0.001`, `dt=1.0`, `dx=1.0`

The benchmark uses a small but meaningful multi-window setup, not a two-window canary. The default evaluates eight start indices when the source file has enough samples. Tests may use synthetic HDF5 fixtures with fewer samples to keep pytest lightweight.

Out of scope for V0:

- hidden competition submission generation
- long 190-step scoring on unavailable test labels
- full torch/FNO training
- web retrieval
- automatic skill installation
- promotion without quantitative evidence

## Architecture

Add a deterministic orchestration layer that composes existing CLI/core pieces instead of duplicating their internals:

1. Run the real-HDF5 persistence baseline smoke to create local windows and baseline artifacts.
2. Run a DeepONet strategy sweep on the generated windows.
3. Include controls and candidate strategies in the same sweep:
   - `branch_trunk_supervised`
   - `branch_trunk_physics_residual`
   - `branch_trunk_spectral`
4. Use spectral weights as the V0 spectral-viscosity proxy. The candidate strategy is identified by `branch_trunk_spectral` plus explicit spectral-weight values and the proposal id `burgers_spectral_viscosity_real_hdf5_v0`.
5. Convert the DeepONet evidence into the canonical operator evidence summary.
6. Run the strategy next-action router with a benchmark-grade minimum window count.
7. If evidence is strong, prepare the strategy-growth promotion artifacts. If evidence fails, record the decision and keep the negative evidence as benchmark output.

The orchestration unit produces one top-level benchmark report that records every artifact path and the decision outcome.

## Data Flow

Input:

- real HDF5 path
- dataset key
- sample limit / start indices
- benchmark output directory
- DeepONet sweep hyperparameters
- proposal id

Intermediate artifacts:

- `baseline/windows_f031.npz`
- `baseline/persistence_score.json`
- `baseline/real_hdf5_baseline_smoke_summary.json`
- `strategy_sweep/deeponet_strategy_sweep_summary.json`
- `strategy_sweep/deeponet_evidence/evidence_summary.json`
- `strategy_evidence_summary.json`
- `strategy_decision.json`

Final artifact:

- `burgers_spectral_viscosity_benchmark_report.json`

The final report includes:

- `state_hint: BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED`
- source data and parameter summary
- candidate strategy id
- evaluation window count
- baseline score
- best strategy id
- mean score delta
- long-window score delta
- failure rate
- high-frequency/residual metrics when available from the evidence summary
- strategy decision and reason codes
- next command suggested by the decision router
- all artifact paths

## Evaluation Rules

The benchmark must treat a candidate as promotion-ready only when the existing strategy router returns `evaluate_promotion`. For V0 that requires:

- evaluation window count meets the configured minimum
- mean score delta is non-negative
- long-window score delta is non-negative
- failure rate is not high
- ranking data exists

The benchmark also exposes physics-related metrics, but V0 does not invent a separate promotion rule outside `recommend-strategy-next-action`. Physics metrics are supporting evidence for ai4pde review and future rubrics.

Negative evidence is useful. If the spectral-viscosity proxy still loses against persistence, the correct final report is a completed benchmark with `decision: retrieve_external_strategy` or `decision: expand_evidence`, not a failed benchmark run.

## CLI Shape

Add one high-level command:

```bash
uv run --frozen --extra data pde-agent run-burgers-spectral-viscosity-benchmark \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-dir runs/burgers_spectral_viscosity_benchmark_001 \
  --dataset-key tensor \
  --sample-limit 8 \
  --min-evaluation-windows 4 \
  --train-steps 4 \
  --hidden-width 8 \
  --latent-dim 4 \
  --learning-rate 0.01 \
  --seed 5 \
  --spectral-weight 0.05 \
  --spectral-weight 0.1 \
  --physics-weight 0.005 \
  --force
```

The command returns the standard JSON protocol payload with:

- `command: run-burgers-spectral-viscosity-benchmark`
- `state_hint: BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED`
- artifact path for the final report
- metrics copied from the report
- `next_hint` from the strategy decision

## Testing

Tests stay deterministic and lightweight:

- use a synthetic HDF5 Burgers-like fixture in `tmp_path`
- skip only when `h5py` is unavailable, matching existing real-HDF5 tests
- avoid torch requirements
- verify the benchmark invokes the existing chain through artifact outputs, not through mocks
- assert the final report exists and contains all expected artifact paths
- assert the CLI returns the JSON protocol payload
- assert invalid sample/window settings fail before writing a partial report

Focused tests:

- `tests/test_burgers_spectral_viscosity_benchmark.py`

Regression suite:

```bash
uv run --frozen pytest tests/test_burgers_spectral_viscosity_benchmark.py -q
uv run --frozen pytest -q
git diff --check
```

## Hermes Follow-Up

V0 does not need to run Hermes inside the unit tests. After the local benchmark command is stable, add a Hermes fixture in a separate step:

- prepare script renders exact command strings
- run state allows the benchmark command, strategy-growth follow-up commands, and no artifact inspection
- audit checks required reads, exact terminal commands, final state, and benchmark report semantics

This keeps the first implementation focused on the quantitative execution chain while preserving the project rule that Hermes-facing protocols require real Hermes audit before being treated as stable.
