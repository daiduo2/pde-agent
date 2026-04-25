# Real PDEBench HDF5 Smoke Design

Date: 2026-04-25

## Purpose

Add a controlled local smoke harness for real PDEBench/Burgers HDF5 files, including an explicit single-file download/preflight path.

This phase is the bridge between the Phase 4 array adapter and later real PDEBench experiments. It should prove that the harness can download or accept one configured Burgers HDF5 file, verify its integrity, inspect it from disk, identify the correct dataset, validate the reduced Burgers shape, and produce a machine-readable inspection artifact without training or committing large files.

The first success target is intentionally narrow:

```text
init-run
resolve-config
inspect-data
```

Only after this passes should the same fixture support the full read-only diagnostic chain:

```text
eval-rollout
analyze-spectrum
check-physics
interpret-error
select-hypothesis
record-timing
collect-artifacts
```

## Non-Goals

- Do not download the whole PDEBench dataset.
- Do not download data implicitly during tests, imports, or normal CLI command execution.
- Do not commit `.hdf5`, `.h5`, `.npz`, checkpoint, or generated run artifacts.
- Do not train, fine-tune, or load model checkpoints.
- Do not build a final competition submission file.
- Do not infer missing data paths by searching the filesystem.
- Do not let Hermes guess HDF5 dataset keys.
- Do not treat target-copy prediction as model performance.

## Design Position

Phase 4 added:

- canonical Burgers array contracts,
- optional HDF5 loading,
- `inspect-data`,
- real-array diagnostic metrics,
- portable generated smoke state for NPZ fixtures.

The real HDF5 smoke should reuse those pieces rather than adding new metric logic. The missing pieces are:

- a deterministic runtime-preparation script for a local real file,
- a controlled single-file download/preflight script for the official Burgers file,
- documentation that separates data acquisition from schema inspection.

The scripts should generate concrete runtime files for the user's checkout, while committed files stay portable.

## Official File Target

Primary official file:

```text
1D_Burgers_Sols_Nu0.001.hdf5
```

Official dataset:

```text
PDEBench Datasets, DaRUS, doi:10.18419/DARUS-2986
```

Observed current metadata from DaRUS V8:

```text
path: 1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5
size: 7.7 GB
md5: 44cb784d5a07aa2b1c864cabdcf625f9
published: 2023-11-30
```

The DaRUS page also states the general HDF5 convention:

```text
[b,t,x1,...,xd,v]
```

For 1D Burgers this may expose a channel dimension, for example `[sample,time,space,variable]`. If the real file is rank 4 with a singleton variable/channel dimension, the adapter should not silently fail. It should either support an explicit safe squeeze or report a clear shape-contract failure that drives the next adapter update.

## Local Data Contract

Default expected real file:

```text
data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5
```

The script should allow overriding the path:

```text
--hdf5-path <path>
```

If this file is absent, the smoke preparation script should fail with a concise message explaining where to place the file, how to pass `--hdf5-path`, or how to run the controlled downloader.

## Controlled Download And Preflight

Add a separate opt-in script:

```text
scripts/download_pdebench_file.py
```

Responsibilities:

- Download exactly one configured file, not the entire dataset.
- Default target is the official Burgers file above.
- Write to:

```text
data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5
```

- Refuse to overwrite an existing file unless `--force` is provided.
- Support `--url`, `--output`, `--expected-md5`, and `--expected-size-gb` overrides.
- Stream to a temporary `.part` file and atomically rename on success.
- Compute MD5 after download.
- Fail if the MD5 does not match the configured expected MD5.
- Print a concise preflight report:
  - output path,
  - byte size,
  - expected MD5,
  - actual MD5,
  - elapsed time.

The downloader should be explicit and user-invoked. Tests must not hit the network.

The implementation plan should verify the current DaRUS direct download API before hard-coding a URL. If DaRUS requires a file ID or API endpoint, the script should use that endpoint and document the source URL in the README. If the API cannot be made stable without manual browser interaction, the script should fall back to generating a clear manual-download instruction rather than pretending download automation is reliable.

## Preflight Script

Add or reuse a local preflight path after the file exists:

```text
scripts/preflight_pdebench_file.py
```

This can be a separate script or a subcommand path inside `download_pdebench_file.py`, but it must be invokable without downloading.

Responsibilities:

- Check file existence.
- Compute size and MD5.
- Compare against expected MD5 when provided.
- Optionally run lightweight HDF5 metadata inspection if `h5py` is installed:
  - list dataset keys,
  - report raw shapes and dtypes without loading full arrays.

The preflight should not run the PDE Agent command chain. It validates the file artifact; `inspect-data` validates the harness contract.

## Runtime Fixture Directory

Add a new fixture directory:

```text
integration_tests/real_hdf5_smoke/
```

Committed files:

- `README.md`
- `base_config.template.json`
- `override.json`
- `task_spec.json`
- `bootstrap_prompt.md`
- `prompt_template.md`
- `run_state.inspect.template.yaml`
- `run_state.full.template.yaml`

Generated ignored files:

- `base_config.runtime.json`
- `run_state.yaml`
- `runs/`

No generated runtime file should be required in git.

## Preparation Script

Add:

```text
scripts/prepare_real_hdf5_smoke.py
```

Responsibilities:

- Resolve repository root from the script location.
- Accept optional CLI args:
  - `--hdf5-path`
  - `--dataset-key`
  - `--sample-limit`, default `1`
  - `--mode`, either `inspect` or `full`, default `inspect`
- Validate that the HDF5 file path exists.
- Write `base_config.runtime.json` with:
  - `data.mode = "pdebench_hdf5"`
  - `data.problem_id = "PDE_Burgers"`
  - absolute `data.source_path`
  - explicit `data.dataset_key` if provided, otherwise `null`
  - `context_steps = 10`
  - `forecast_steps = 190`
  - `time_reduction = 5`
  - `space_reduction = 4`
  - configured `sample_limit`
- Write `run_state.yaml` from the selected template using absolute paths for the current checkout.
- Print generated file paths and the selected HDF5 path.

The script should not inspect HDF5 contents itself. The source of truth for schema inspection remains the harness command:

```text
pde-agent inspect-data
```

## Inspect Mode

Default mode should be `inspect`.

Generated command chain:

```text
init-run
resolve-config
inspect-data
```

Success criteria:

- `metrics/data_inspection.json` exists.
- Command output has `status: ok`.
- `state_hint` is `DATA_INSPECTED`.
- Inspection payload includes:
  - `source_mode = "pdebench_hdf5"`
  - `source_path`
  - `dataset_key`
  - `available_keys`
  - `raw_shape`
  - `raw_dtype`
  - `reduced_shape`
  - `contract_shape`
  - `is_compatible = true`

This mode is the first gate before any full diagnostic chain.

## Full Mode

Full mode is opt-in:

```text
--mode full
```

Generated command chain:

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

This remains read-only. In current Phase 4 behavior, HDF5 loading uses `prediction = target.copy()` for plumbing validation. The README and generated prompt must explicitly state that this is not model performance.

Success criteria:

- final `collect-artifacts` returns `state_hint = HYPOTHESIS_SELECTED`,
- manifest phase is `hypothesis_selected`,
- manifest summary includes `source_mode = "pdebench_hdf5"`,
- manifest summary includes `data_is_compatible = true`.

## Dataset Key Policy

The smoke must be fail-closed:

- If `--dataset-key` is provided, the loader must use it.
- If no key is provided and the HDF5 file contains exactly one dataset, auto-selection is acceptable.
- If multiple datasets exist, `inspect-data` should fail with a clear ambiguity error and the user should rerun the preparation script with `--dataset-key`.

The smoke script must not search or infer keys outside the configured HDF5 file.

## Hermes Protocol

The generated `run_state.yaml` should follow existing fixture conventions:

- required reads are explicit absolute paths,
- allowed commands are explicit full commands,
- forbidden actions include:
  - `patch_code`
  - `run_training`
  - `call_build_submission`
  - `guess_missing_paths`
  - `skip_required_reads`
  - `search_outside_current_directory`
  - `inspect_command_help`
  - `probe_command_surface`

The prompt should instruct Hermes to stop if a required file is missing.

## Testing Strategy

Unit tests should cover the preparation script without requiring a real PDEBench file:

- missing default HDF5 path fails with a clear message,
- script generates inspect-mode runtime files when given a tiny temporary HDF5 path,
- generated inspect `run_state.yaml` contains `inspect-data` and does not contain full diagnostic commands,
- generated full `run_state.yaml` contains the full chain,
- generated `base_config.runtime.json` points to the provided HDF5 path and uses `pdebench_hdf5`.

Unit tests for download/preflight should not hit the network:

- existing output file is not overwritten unless `--force` is set,
- MD5 computation passes for a tiny local file,
- MD5 mismatch fails with a clear message,
- URL/download logic is isolated behind a function that can be tested with local `file://` or a fake stream,
- preflight reports HDF5 dataset keys for a tiny generated HDF5 file when `h5py` is available.

Integration-level manual validation should be local-only:

```text
uv run --frozen python scripts/prepare_real_hdf5_smoke.py --hdf5-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 --sample-limit 1
```

Then execute the commands listed in:

```text
integration_tests/real_hdf5_smoke/run_state.yaml
```

Automated tests should use tiny generated HDF5 files, not the real PDEBench dataset.

## Git Hygiene

Add ignore rules for:

```text
integration_tests/real_hdf5_smoke/base_config.runtime.json
integration_tests/real_hdf5_smoke/run_state.yaml
integration_tests/real_hdf5_smoke/runs/
data/pdebench/burgers/*.part
```

The existing data ignore rules already cover real `.hdf5` files under `data/`.

## Acceptance Criteria

This phase is complete when:

- the real HDF5 smoke fixture directory exists with portable templates,
- the controlled downloader or manual-download helper exists and is documented,
- downloaded files are written only under ignored `data/pdebench/burgers/`,
- MD5 preflight exists and is test-covered without network access,
- `scripts/prepare_real_hdf5_smoke.py` generates inspect and full runtime state,
- tests validate the generated runtime files using tiny HDF5 fixtures,
- no real data or generated runtime state is committed,
- `inspect-data` can be run against a configured local HDF5 path,
- the README documents the exact command sequence and the target-copy prediction caveat.

## Residual Risks

- DaRUS direct download behavior may change; the downloader should fail with clear manual instructions if automation is blocked.
- The 7.7 GB file may take significant time and disk space; the script should make this explicit before download.
- Until the actual PDEBench file is downloaded locally, tests only prove behavior on tiny HDF5 fixtures.
- If the real PDEBench dataset uses a rank or axis order different from the current `[sample, time, space]` expectation, `inspect-data` should fail and the adapter contract must be updated deliberately.
- Full mode remains plumbing validation until real prediction ingestion or baseline model inference is added.
