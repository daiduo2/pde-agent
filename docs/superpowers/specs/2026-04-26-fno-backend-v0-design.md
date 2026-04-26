# FNO Backend V0 Design

Date: 2026-04-26

## Purpose

Add the first real neural-operator backend to the PDE agent harness: a minimal
FNO path for local Burgers evaluation.

The harness already supports:

```text
prediction NPZ
  -> score-local-rollout
  -> diagnose-rollout-error
  -> check-burgers-physics-local
  -> select-pde-hypothesis
  -> plan-next-experiment
  -> generate-experiment-config
```

This phase should make FNO produce the same prediction NPZ contract so it can
enter the existing scientific iteration loop.

## Non-Goals

- Do not implement DeepONet or PI-DeepONet.
- Do not train or finetune FNO.
- Do not build final competition HDF5 submissions.
- Do not make PyTorch a default dependency.
- Do not replace the lightweight spectral baseline.
- Do not change scoring, diagnosis, physics, planning, or config-generation
  commands.
- Do not implement hidden 190-step evaluation.

## Design Position

FNO is the right first real backend because it matches the current Burgers
setting and the existing spectral diagnostics.

V0 should be deliberately narrow:

1. Add optional PyTorch dependency support.
2. Inspect local official-style FNO checkpoint archives.
3. Provide a minimal FNO prediction adapter that consumes local windows NPZ and
   emits prediction NPZ.
4. Keep the output compatible with all existing diagnostic skills.

If the official checkpoint format is not immediately loadable without importing
external PDEBench training code, V0 should still provide useful inspection and a
clear typed error. The adapter must fail fast rather than silently generating
fake FNO predictions.

## Dependency Strategy

PyTorch must be optional.

Add a project optional dependency group:

```toml
[project.optional-dependencies]
torch = [
  "torch>=2.2",
]
```

Existing commands and tests must continue to work without installing the torch
extra. Torch-specific commands should fail with a clear `FNOBackendError` if
torch is unavailable.

## Local Data Assumptions

The current repo may contain:

```text
data/pdebench/checkpoints/burgers_FNO.tar
data/derived/local_windows_task1_smoke.npz
```

V0 tests must not require those real files. Tests should use tiny synthetic
checkpoint fixtures and synthetic local windows.

Manual smoke commands can use the real files if present.

## Commands

### `inspect-fno-checkpoint`

Purpose:

Inspect a local FNO checkpoint/archive and write machine-readable metadata.

Inputs:

- `--checkpoint-path`: local `.pt`, `.pth`, or `.tar` checkpoint path
- `--output-path`: JSON metadata artifact

Output payload:

- `run_id`: `local`
- `command`: `inspect-fno-checkpoint`
- `state_hint`: `FNO_CHECKPOINT_INSPECTED`
- artifact: `fno_checkpoint_metadata_path`
- metrics:
  - `checkpoint_format`
  - `top_level_key_count`
  - `has_model_state`
  - `has_config`
- `next_hint`: `predict-fno-local`

Metadata JSON should include:

```json
{
  "schema_version": 1,
  "state_hint": "FNO_CHECKPOINT_INSPECTED",
  "checkpoint_path": "data/pdebench/checkpoints/burgers_FNO.tar",
  "checkpoint_format": "torch_serialized",
  "top_level_keys": ["model_state_dict", "config"],
  "has_model_state": true,
  "has_config": true,
  "tensor_summary": {
    "parameter_count": 123456,
    "tensor_key_count": 48
  },
  "load_warnings": []
}
```

V0 does not need to fully understand every possible PDEBench checkpoint schema.
It should inspect safely and report what it can.

### `predict-fno-local`

Purpose:

Run an FNO backend on local windows and write prediction NPZ compatible with
`score-local-rollout`.

Inputs:

- `--windows-path`: local windows NPZ with `context` and `target`
- `--checkpoint-path`: FNO checkpoint path
- `--output-path`: prediction NPZ
- `--metadata-path`: optional checkpoint metadata JSON from
  `inspect-fno-checkpoint`
- `--force`: overwrite existing output

Output payload:

- `run_id`: `local`
- `command`: `predict-fno-local`
- `state_hint`: `FNO_PREDICTION_READY`
- artifact: `prediction_npz_path`
- metrics:
  - `model_type`: `fno`
  - `sample_count`
  - `forecast_steps`
  - `space_points`
  - `checkpoint_format`
- `next_hint`: `score-local-rollout`

Prediction NPZ contract:

```text
prediction: float32 array [N, forecast_steps, X]
metadata_json: JSON string
```

If the checkpoint cannot be mapped to the minimal V0 FNO wrapper, the command
must raise `FNOBackendError` with a concise message. It must not fall back to
persistence or spectral baseline while claiming to be FNO.

## Core Modules

### `pde_agent.core.fno_backend`

Responsibilities:

- optional torch import helper,
- checkpoint loading/inspection,
- local windows validation,
- minimal prediction adapter,
- exact output-path writing.

Public API:

```python
class FNOBackendError(RuntimeError):
    pass

def inspect_fno_checkpoint(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    ...

def predict_fno_local(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    ...
```

V0 may support a deliberately small checkpoint schema:

- a torch checkpoint dict with `model_type == "fno_v0_fixture"`,
- a `prediction_rule` fixture used in tests,
- or a directly loadable minimal state dict if a small internal FNO wrapper is
  implemented.

The official checkpoint smoke can be inspect-only until the exact model schema
is confirmed.

## Minimal Prediction Strategy

V0 has two acceptable prediction paths:

1. **Fixture-compatible path for tests**
   - tiny torch checkpoint with explicit fixture metadata,
   - deterministic prediction such as learned delta extrapolation,
   - used only to test the FNO command contract.

2. **Real checkpoint path**
   - inspect official checkpoint,
   - if loadable by the V0 wrapper, predict,
   - otherwise raise `FNOBackendError` explaining unsupported checkpoint schema.

This avoids fake success on official checkpoints while still making CLI and
artifact contracts testable.

## State Hints

Add:

- `FNO_CHECKPOINT_INSPECTED`
- `FNO_PREDICTION_READY`

## Tests

Tests must be skipped cleanly if torch is not installed.

Required tests:

1. `test_inspect_fno_checkpoint_fixture`
   - creates tiny torch checkpoint fixture,
   - runs core inspect,
   - asserts metadata JSON exists and includes expected keys.

2. `test_inspect_fno_checkpoint_cli_writes_payload`
   - invokes `pde-agent inspect-fno-checkpoint`,
   - asserts state hint, artifact path, and metrics.

3. `test_predict_fno_local_fixture_writes_prediction`
   - creates tiny local windows NPZ,
   - creates fixture checkpoint,
   - runs `predict_fno_local`,
   - asserts prediction shape and exact output path.

4. `test_predict_fno_local_cli_writes_payload`
   - invokes `pde-agent predict-fno-local`,
   - asserts payload and output file.

5. `test_predict_fno_local_rejects_unsupported_checkpoint`
   - creates torch checkpoint with unsupported schema,
   - asserts `FNOBackendError`.

Non-torch environments:

- All FNO tests should skip with a clear message if torch is unavailable.
- Non-FNO tests must still run without torch.

## README

Add a short section:

```text
## FNO Backend V0

FNO support is optional and requires the torch extra.
```

Include commands:

```bash
uv run --frozen --extra torch pde-agent inspect-fno-checkpoint \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-path runs/fno_checkpoint_metadata.json

uv run --frozen --extra torch pde-agent predict-fno-local \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --metadata-path runs/fno_checkpoint_metadata.json \
  --output-path data/derived/local_fno_prediction_task1_smoke.npz
```

The README must state that V0 may inspect official checkpoints even when the
prediction adapter does not yet support their exact schema.

## Success Criteria

- Torch remains optional.
- Existing non-FNO test suite passes without torch.
- FNO tests skip cleanly without torch.
- `inspect-fno-checkpoint` exists and writes metadata for fixture checkpoints.
- `predict-fno-local` exists and writes prediction NPZ for fixture checkpoints.
- Unsupported checkpoint schemas fail with `FNOBackendError`.
- FNO prediction output can be passed to existing local scoring and diagnostic
  commands.
