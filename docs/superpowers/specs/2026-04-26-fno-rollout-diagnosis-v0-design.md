# FNO Rollout Diagnosis V0 Design

Date: 2026-04-26

## Purpose

Add a thin orchestration layer that runs the full local diagnostic chain for an
official FNO rollout and writes one agent-readable summary artifact.

The current harness can already run each piece separately:

```text
predict-fno-local
  -> score-local-rollout
  -> diagnose-rollout-error
  -> check-burgers-physics-local
```

V0 should make this chain reliable for Hermes by standardizing paths, enforcing
overwrite rules, and aggregating the most important evidence into
`fno_rollout_summary.json`.

## Scope

In scope:

- predict local FNO rollout from existing local windows,
- score the rollout using existing local scoring,
- diagnose rollout error using existing error diagnosis,
- run Burgers physics checks using existing physics utilities,
- write all intermediate artifacts under one output directory,
- write a compact summary JSON for downstream agent decisions,
- expose one CLI command: `diagnose-fno-rollout`.

Out of scope:

- changing FNO model architecture,
- training or finetuning,
- comparing against spectral or persistence baselines,
- selecting PDE hypotheses,
- planning experiments,
- generating final competition submissions,
- changing scoring, error diagnosis, or physics formulas.

## Command

Add:

```bash
uv run --frozen --extra torch pde-agent diagnose-fno-rollout \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir runs/fno_rollout_diagnosis \
  --nu 0.001 \
  --dt 1.0 \
  --dx 1.0 \
  --force
```

Inputs:

- `--windows-path`: NPZ with `context` and `target`
- `--checkpoint-path`: official FNO checkpoint `.pt` or `.tar`
- `--output-dir`: directory for generated artifacts
- `--nu`: Burgers viscosity, default `0.001`
- `--dt`: local diagnostic time step, default `1.0`
- `--dx`: local diagnostic spatial step, default `1.0`
- `--force`: allow overwriting generated artifacts
- `--json`: existing CLI style, default `True`

Output payload:

```json
{
  "run_id": "local",
  "command": "diagnose-fno-rollout",
  "state_hint": "FNO_ROLLOUT_DIAGNOSED",
  "artifacts": {
    "fno_prediction_npz_path": ".../fno_prediction.npz",
    "fno_score_json_path": ".../fno_score.json",
    "fno_rollout_error_json_path": ".../fno_rollout_error.json",
    "fno_burgers_physics_json_path": ".../fno_burgers_physics.json",
    "fno_rollout_summary_json_path": ".../fno_rollout_summary.json"
  },
  "metrics": {
    "adapter_type": "pdebench_fno1d",
    "competition_like_score_short": 49.3,
    "dominant_error_pattern": "long_horizon_drift",
    "physics_issue": "residual_mismatch"
  },
  "next_hint": "select-pde-hypothesis"
}
```

## Artifact Contract

The command writes exactly these files under `output_dir`:

```text
fno_prediction.npz
fno_score.json
fno_rollout_error.json
fno_burgers_physics.json
fno_rollout_summary.json
```

If any target artifact exists and `force=False`, the command must fail before
writing any output. This prevents partial overwrite of previous runs.

If `force=True`, existing target artifacts may be overwritten.

## Summary JSON

`fno_rollout_summary.json` is the main agent-facing artifact.

Required fields:

```json
{
  "schema_version": 1,
  "state_hint": "FNO_ROLLOUT_DIAGNOSED",
  "model": {
    "model_type": "fno",
    "adapter_type": "pdebench_fno1d",
    "history_steps": 10,
    "modes": 12,
    "width": 20,
    "checkpoint_path": "...",
    "archive_member": "1D_Burgers_Sols_Nu0.001_FNO.pt"
  },
  "artifacts": {
    "prediction_npz_path": "...",
    "score_json_path": "...",
    "rollout_error_json_path": "...",
    "burgers_physics_json_path": "..."
  },
  "score": {
    "competition_like_score_short": 49.3,
    "rel_mse": 0.069,
    "rmse": 0.161
  },
  "error_diagnosis": {
    "dominant_error_pattern": "long_horizon_drift",
    "late_rel_mse": 0.12,
    "time_error_slope": 0.01,
    "shock_region_error_ratio": 1.8
  },
  "physics": {
    "physics_issue": "residual_mismatch",
    "residual_ratio": 2.4,
    "total_variation_ratio": 0.7,
    "gradient_error_ratio": 1.3
  },
  "recommendation": {
    "next_command": "select-pde-hypothesis",
    "rationale": "Use existing hypothesis selection with this score, error diagnosis, and physics check."
  }
}
```

The summary should keep a compact subset of fields, not duplicate entire
intermediate JSON files.

## Core API

Create `pde_agent.core.fno_rollout_diagnosis`.

Public API:

```python
class FNORolloutDiagnosisError(RuntimeError):
    pass

def diagnose_fno_rollout(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    ...
```

Implementation should call existing core functions directly:

- `predict_fno_local` from `pde_agent.core.fno_backend`
- `score_prediction_npz` from `pde_agent.core.local_scoring`
- `diagnose_rollout_error_npz` from `pde_agent.core.rollout_error_diagnosis`
- `check_burgers_physics_npz` from `pde_agent.core.burgers_physics_local`

The core API returns the same summary dictionary that it writes to
`fno_rollout_summary.json`, with `summary_path` included.

## Command Wrapper

Create `pde_agent.commands.diagnose_fno_rollout`.

It should call the core API and wrap it with `success_payload`.

Metrics should include:

- `adapter_type`
- `competition_like_score_short`
- `rel_mse`
- `rmse`
- `dominant_error_pattern`
- `physics_issue`
- `residual_ratio`
- `total_variation_ratio`

Artifacts should include all five generated artifact paths.

## State Hint

Add:

```text
FNO_ROLLOUT_DIAGNOSED
```

to `ALLOWED_STATE_HINTS`.

## Tests

Tests should use synthetic fixture FNO checkpoint data and existing small windows
helpers where possible. They must not require real PDEBench files.

Required tests:

1. Core orchestration writes all five artifacts.
2. Core summary contains model, score, error diagnosis, physics, artifacts, and
   recommendation sections.
3. `force=False` rejects pre-existing target artifact before overwriting.
4. CLI command returns `FNO_ROLLOUT_DIAGNOSED`, artifact paths, and key metrics.
5. Existing FNO backend and non-FNO tests still pass.

Optional real smoke:

If local files exist, run `diagnose-fno-rollout` against:

```text
data/derived/local_windows_task1_smoke.npz
data/pdebench/checkpoints/burgers_FNO.tar
```

The smoke should complete and write a score-consumable prediction.

## Success Criteria

- `diagnose-fno-rollout` exists and writes a complete artifact bundle.
- Summary JSON is sufficient for an agent to decide whether to run
  `select-pde-hypothesis`.
- No model/training behavior is changed.
- Existing FNO adapter remains valid.
- Existing test suites pass with and without extras.
