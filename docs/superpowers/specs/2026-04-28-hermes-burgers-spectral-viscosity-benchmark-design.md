# Hermes Burgers Spectral Viscosity Benchmark Design

## Purpose

The local Burgers spectral-viscosity benchmark is now a working CLI chain. The next step is to make it Hermes-facing so the agent competition path can prove that Hermes can execute the quantitative benchmark under a narrow run-state contract.

The fixture must validate protocol behavior, not model success. If the benchmark report says the spectral-viscosity proxy still loses and routes to `retrieve_external_strategy`, that is an accepted scientific outcome.

## Scope

Add a new fixture:

- `integration_tests/hermes_burgers_spectral_viscosity_benchmark`

The fixture exposes exactly one terminal command:

- `run-burgers-spectral-viscosity-benchmark`

The fixture requires Hermes to read only:

- `run_state.yaml`
- `bootstrap_prompt.md`
- `task_spec.json`

Hermes must not inspect generated artifacts, run command help, patch code, download data, or run lower-level component commands directly.

## Runtime Contract

The prepare script renders `run_state.yaml` from `run_state.template.yaml` with:

- repo root
- fixture root
- real HDF5 source path
- output directory
- dataset key
- sample limit
- minimum evaluation windows
- forecast steps

Default runtime values:

- source path: `data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5`
- sample limit: `4`
- minimum evaluation windows: `4`
- forecast steps: `31`
- output directory: `integration_tests/hermes_burgers_spectral_viscosity_benchmark/runs/burgers_spectral_viscosity_benchmark_001`

The single allowed command uses a small but meaningful benchmark:

```bash
cd <repo> && uv run --frozen --extra data pde-agent run-burgers-spectral-viscosity-benchmark ...
```

The command includes `--force` so repeated fixture runs are deterministic.

## Required Artifacts

The fixture requires at least:

- `burgers_spectral_viscosity_benchmark_report.json`
- `strategy_decision.json`
- `strategy_evidence_summary.json`
- `baseline/real_hdf5_baseline_smoke_summary.json`
- `strategy_sweep/deeponet_strategy_sweep_summary.json`

The audit reads artifacts after Hermes finishes. Hermes itself must not read them.

## Audit Rules

Add a dedicated audit module for this fixture. It checks:

- session file exists and has a user message
- first `read_file` tool calls exactly match required reads
- terminal command sequence exactly matches the one allowed command
- no forbidden tool calls
- terminal exit codes are zero
- final terminal payload has `state_hint: BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED`
- required artifacts exist
- final benchmark report has:
  - `state_hint: BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED`
  - `proposal_id: burgers_spectral_viscosity_real_hdf5_v0`
  - `evaluation_window_count >= min_evaluation_windows`
  - numeric `mean_score_delta`, `long_window_score_delta`, and `failure_rate`
  - decision in `evaluate_promotion`, `expand_evidence`, or `retrieve_external_strategy`
  - artifact paths for strategy decision and strategy evidence

The audit must pass for a negative-evidence benchmark if the protocol and report are valid.

## Runner

Add scripts matching existing Hermes patterns:

- `scripts/prepare_burgers_spectral_viscosity_benchmark.py`
- `scripts/audit_hermes_burgers_spectral_viscosity_benchmark.py`
- `scripts/run_hermes_burgers_spectral_viscosity_benchmark.py`

The runner submits the prompt to Hermes, exits once the success state appears, then runs the audit. It also supports `--audit-only` with `--session-id` or `--session-path`.

## Tests

Add tests for:

- prepare script renders exact one-command protocol and rejects missing source files
- audit passes for a synthetic valid saved session and benchmark report
- audit fails on manual artifact inspection such as `cat report.json`
- audit fails when the benchmark report is missing or semantically invalid
- runner/audit script parses run-state command lists and writes report JSON

The local test suite must not invoke real Hermes or require the real 7.7 GB HDF5 file.

## Real Hermes Verification

After tests pass, run a real Hermes canary when the local environment has Hermes available and the real HDF5 file exists:

```bash
uv run --frozen python scripts/prepare_burgers_spectral_viscosity_benchmark.py
uv run --frozen python scripts/run_hermes_burgers_spectral_viscosity_benchmark.py --timeout-seconds 600
```

If Hermes is unavailable or blocks, report the local tests and the blocker explicitly instead of claiming the Hermes canary passed.
