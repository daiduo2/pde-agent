# PDEBench HDF5 Adapter Design

Date: 2026-04-24

## Purpose

Add a real-data read-only adapter layer for the PDE Agent harness so the existing diagnostic workflow can operate on PDEBench/Burgers arrays instead of only synthetic metrics.

This phase keeps the system in a diagnostic-only state. It should make real HDF5 data a first-class input while preserving the stable synthetic smoke path used to validate Hermes protocol compliance.

The goal is to answer four practical questions before training or checkpoint integration begins:

1. Can the harness locate and inspect real PDEBench HDF5 files deterministically?
2. Can it normalize Burgers arrays into one internal shape contract?
3. Can the existing rollout, spectrum, and physics diagnostics compute real metrics from arrays?
4. Can Hermes follow the same controlled command chain without probing paths, guessing schema, or running training?

## Non-Goals

- Do not train, fine-tune, or load FNO, UNet, DeepONet, or PI-DeepONet checkpoints.
- Do not build the final competition submission file.
- Do not require the full PDEBench HDF5 files for unit tests or CI.
- Do not commit large HDF5 data files to git.
- Do not let Hermes search arbitrary directories for data.
- Do not replace the synthetic protocol smoke fixture.

## Design Position

Phase 3 established a deterministic science-shaped diagnostic chain:

```text
eval-rollout
analyze-spectrum
check-physics
interpret-error
select-hypothesis
```

Phase 4 inserts a real data adapter beneath that chain. The diagnostic commands should consume a canonical array bundle and should not know whether the bundle came from synthetic fixture data, a small NPZ test fixture, or a real PDEBench HDF5 file.

This keeps the architecture stable:

- Hermes follows the same bounded CLI protocol.
- The harness gains real PDEBench readiness.
- Scientific metric code is testable without large files.
- HDF5 schema failures are surfaced early through an explicit inspection step.

## Recommended Approach

Use a hybrid adapter:

1. Define an internal array contract first.
2. Add a real PDEBench HDF5 loader that maps HDF5 data into that contract.
3. Add small NPZ fixtures for automated tests.
4. Preserve synthetic mode for protocol smoke tests.

This is stricter than a placeholder HDF5 hook. Real HDF5 is a first-class pathway in the design, but large files remain local runtime assets rather than repository artifacts.

## Data Layout

Local real data should live under:

```text
data/pdebench/burgers/
```

Expected files may include:

```text
1D_Burgers_Sols_Nu0.001.hdf5
1D_Burgers_Sols_Nu0.001_pred.hdf5
1D_Burgers_Sols_Nu0.001_val.hdf5
1D_Burgers_Sols_Nu0.001_submit.hdf5
```

The repository should include a `data/README.md` explaining the expected files and their source, but the actual `.hdf5`, `.h5`, `.npz`, checkpoint, and generated result files should be ignored by git.

## Array Contract

All diagnostic code should operate on a canonical Burgers rollout bundle:

```text
context: float array, shape [sample, context_time, space]
prediction: float array, shape [sample, forecast_time, space]
target: float array, shape [sample, forecast_time, space]
full_reference: optional float array, shape [sample, total_time, space]
metadata: object
```

Required metadata:

- `problem_id`: expected `PDE_Burgers`
- `source_mode`: one of `synthetic`, `npz_fixture`, `pdebench_hdf5`
- `nu`: viscosity if known
- `dt`: reduced time step if known
- `dx`: reduced spatial spacing if known
- `context_time`: expected `10` for the competition task
- `forecast_time`: expected `190` for the long validation task
- `space_points`: expected `256` after PDEBench reduction
- `time_reduction`: expected `5` for the official checkpoint setting
- `space_reduction`: expected `4` for the official checkpoint setting
- `sample_count`

The adapter should validate shape and axis order before metrics are computed. A shape mismatch is a hard error, not a warning.

## HDF5 Loader

The HDF5 loader should be isolated in a core module rather than embedded in CLI commands.

Responsibilities:

- Open a configured HDF5 path.
- List available dataset keys.
- Identify the primary solution tensor.
- Read shape and dtype without loading more data than needed during inspection.
- Support explicit dataset key configuration.
- Support conservative auto-detection only when unambiguous.
- Apply PDEBench reductions consistently:
  - time reduction: `::5`
  - space reduction: `::4`
- Construct the competition split:
  - first 10 reduced time steps as context,
  - next 190 reduced time steps as target when available.
- Return canonical arrays and metadata.

The loader should fail closed. If multiple plausible dataset keys exist and no key is configured, it should report candidate keys and stop.

## NPZ Fixture Loader

Small NPZ fixtures should exist only for tests and local deterministic checks. The NPZ loader should use the same array contract as the HDF5 loader.

Expected NPZ keys:

```text
context
prediction
target
```

Optional NPZ keys:

```text
full_reference
metadata_json
```

This lets unit tests verify real metric behavior without requiring the full PDEBench dataset.

## Configuration

The resolved config should support a `data` section:

```json
{
  "data": {
    "mode": "synthetic",
    "problem_id": "PDE_Burgers",
    "source_path": null,
    "dataset_key": null,
    "context_steps": 10,
    "forecast_steps": 190,
    "time_reduction": 5,
    "space_reduction": 4,
    "sample_limit": null
  }
}
```

Supported modes:

- `synthetic`: existing deterministic smoke behavior.
- `npz_fixture`: load canonical arrays from a small NPZ fixture.
- `pdebench_hdf5`: load real PDEBench HDF5 arrays.

Hermes fixtures should provide explicit paths in config. Hermes must not search for data.

## New Command: `inspect-data`

Add a read-only command:

```text
pde-agent inspect-data --run-id <id> --run-root <path>
```

Reads:

- `config/resolved.json`

Writes:

- `metrics/data_inspection.json`

State hint:

- `DATA_INSPECTED`

Output fields:

- `source_mode`
- `source_path`
- `dataset_key`
- `available_keys`
- `raw_shape`
- `raw_dtype`
- `reduced_shape`
- `contract_shape`
- `sample_count`
- `context_time`
- `forecast_time`
- `space_points`
- `is_compatible`
- `warnings`

The command should not compute rollout quality. It only verifies that the data source can be normalized into the expected contract.

## Updated Diagnostic Commands

### `eval-rollout`

Current synthetic behavior should remain available.

In `npz_fixture` or `pdebench_hdf5` mode, `eval-rollout` should:

- load the canonical array bundle,
- compute per-time NRMSE,
- compute one-step NRMSE,
- compute long-horizon NRMSE,
- compute segment metrics for early, middle, and late forecast windows,
- write `metrics/rollout.json`,
- write a machine-readable error curve artifact.

If no prediction array is available in real HDF5 mode, Phase 4 should not invent a model. It should fail with an explicit error explaining that real target data exists but prediction data is missing. Later phases can add baseline inference.

### `analyze-spectrum`

In real array modes, spectrum diagnostics should compute frequency-domain errors from prediction and target arrays.

Required outputs:

- low-frequency error
- mid-frequency error
- high-frequency error
- high-to-low error ratio
- spectral drift score
- dominant spectral issue

The first implementation can use simple FFT band partitioning. It does not need a publication-grade spectral diagnostic yet, but the metric definition must be deterministic and documented in the payload.

### `check-physics`

In real array modes, physics checks should compute simple conservation and stability proxies:

- mass drift,
- energy ratio,
- total variation ratio,
- nonphysical oscillation flag,
- blowup-like behavior flag,
- excessive smoothing flag,
- physics issue label.

These are diagnostic proxies, not formal PDE residual validation. The payload must label them as proxies.

### `interpret-error` and `select-hypothesis`

These commands should not need to know the data source. They should consume structured metric payloads and emit diagnosis/hypothesis artifacts as they do in Phase 3.

The interpretation logic can become source-aware only through metric fields such as `source_mode`, not by reading raw data files.

## State Machine

The real-data diagnostic fixture should use this flow:

```text
TASK_PARSED
  -> CONFIG_RESOLVED
  -> DATA_INSPECTED
  -> ROLLOUT_EVALUATED
  -> SPECTRUM_ANALYZED
  -> PHYSICS_CHECKED
  -> ERROR_INTERPRETED
  -> HYPOTHESIS_SELECTED
```

Allowed commands:

```text
init-run
resolve-config
inspect-data
eval-rollout
analyze-spectrum
check-physics
interpret-error
select-hypothesis
record-timing
collect-artifacts
```

Forbidden actions remain:

```text
patch_code
run_training
call_build_submission
guess_missing_paths
skip_required_reads
search_outside_current_directory
inspect_command_help
probe_command_surface
```

## Artifact Contract

The final manifest should include:

- `metrics/data_inspection.json`
- `metrics/rollout.json`
- `metrics/spectrum.json`
- `metrics/physics_check.json`
- `diagnostics/error_interpretation.json`
- `diagnostics/selected_hypothesis.json`
- `artifacts/manifest.json`

Manifest summary should include:

- `source_mode`
- `data_is_compatible`
- `one_step_nrmse`
- `long_horizon_nrmse`
- `dominant_spectral_issue`
- `physics_issue`
- `dominant_failure_mode`
- `selected_hypothesis`

Manifest completeness should include:

- `has_data_inspection`
- `has_rollout`
- `has_spectrum`
- `has_physics_check`
- `has_error_interpretation`
- `has_selected_hypothesis`
- `has_timing`

## Error Handling

The adapter should use explicit, machine-readable failures.

Hard errors:

- source path is missing in non-synthetic mode,
- source path does not exist,
- HDF5 file cannot be opened,
- dataset key is ambiguous,
- configured dataset key is absent,
- raw array has unsupported rank,
- reduced array cannot provide required context and forecast lengths,
- prediction and target shapes do not match,
- dtype cannot be converted to float safely.

Warnings:

- viscosity metadata is missing,
- `dt` or `dx` is inferred rather than loaded,
- sample count is lower than expected,
- optional full reference is absent.

Hermes should see failures through JSON command output, not traceback-driven exploration.

## Dependency Strategy

Add `numpy` as a normal runtime dependency because real diagnostics require array operations.

Use `h5py` as an optional real-data dependency:

- package it through a data extra, for example `pde-agent[data]`,
- import it lazily inside the HDF5 loader,
- return a clear JSON-visible error when `pdebench_hdf5` mode is requested without `h5py`.

This keeps the default harness lightweight while still making the HDF5 path explicit and testable.

## Real HDF5 Acquisition

The harness should provide documentation for placing PDEBench files under `data/pdebench/burgers/`.

Phase 4 should not add an automatic downloader. Network access, dataset hosting, and credential requirements are too variable for the first real-data adapter. Download automation can be added later as an explicit convenience command after the local data contract is stable.

Recommended local verification after the dataset is present:

```text
pde-agent inspect-data --run-id local_burgers_inspect --run-root runs/local_burgers_inspect
```

The command should report whether the file can produce the expected reduced shape:

```text
[sample, 200, 256]
```

for the long validation task, then split into:

```text
context: [sample, 10, 256]
target: [sample, 190, 256]
```

## Testing Strategy

Unit tests:

- validate the array contract shape checks,
- validate NPZ fixture loading,
- validate HDF5 inspection behavior using a tiny generated HDF5 file if `h5py` is available,
- validate deterministic NRMSE and error-curve metrics,
- validate FFT band diagnostics,
- validate physics proxy metrics,
- validate clear failures for missing paths and ambiguous dataset keys.

Integration tests:

- keep existing synthetic Hermes smoke unchanged,
- add a local array fixture diagnostic smoke that uses `npz_fixture`,
- create tiny HDF5 files dynamically during tests when `h5py` is installed,
- do not require large PDEBench files in CI.

Manual local validation:

- place real PDEBench files under `data/pdebench/burgers/`,
- run `inspect-data`,
- run the full diagnostic command chain only after inspection passes.

## Acceptance Criteria

Phase 4 is complete when:

- the repo has a documented local data directory contract,
- `inspect-data` can inspect synthetic, NPZ, and HDF5 modes,
- real metric code operates on canonical arrays,
- existing synthetic Hermes smoke still passes,
- a fixture-based real-array diagnostic flow passes without training,
- failures are JSON-visible and do not require Hermes to guess paths or inspect command help.

## Fixed Implementation Decisions

Phase 4 should use these fixed implementation choices:

- `h5py` is optional and lazily imported.
- Tiny HDF5 fixtures are generated during tests when `h5py` is available.
- Real HDF5 acquisition is documented only.
- `inspect-data` writes metadata only and does not materialize cached canonical arrays in Phase 4.
