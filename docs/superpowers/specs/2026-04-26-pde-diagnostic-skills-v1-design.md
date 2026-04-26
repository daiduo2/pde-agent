# PDE Diagnostic Skills V1 Design

Date: 2026-04-26

## Purpose

Upgrade `pde-agent` from a scoreable local harness to an executable PDE experiment interpretation loop.

The current harness can slice real PDEBench Burgers HDF5 data into local windows and score a prediction NPZ. The next step is to give Hermes a minimal failed prediction and deterministic PDE diagnostics so it can reason over evidence rather than jump directly into model-code changes.

This phase introduces a small baseline generator plus the first real-local PDE diagnostic skills:

```text
build-local-windows
  -> generate-baseline-prediction
  -> score-local-rollout
  -> diagnose-rollout-error
  -> check-burgers-physics
  -> select-pde-hypothesis
```

The output should be machine-readable for automation and concise enough to become part of an Agent research log.

## Non-Goals

- Do not train FNO, UNet, DeepONet, or PI-DeepONet in this phase.
- Do not load official checkpoints in this phase.
- Do not build final `task1_pred.hdf5` or `submission.zip`.
- Do not implement exact hidden 190-step Frechet scoring.
- Do not give Hermes open-ended shell exploration for this loop.
- Do not make the PDE skills pure prompt templates with no executable evidence.

## Design Position

This is the first bridge between the CLI harness and actual PDE scientific reasoning.

The earlier synthetic diagnostic commands prove that Hermes can follow a constrained command protocol. The new layer should work on local NPZ windows derived from the real Burgers HDF5 file. It should still be simple and deterministic enough to test without the large file, but the same commands must run on real local windows.

The key design rule is that each PDE skill consumes explicit artifacts and emits structured JSON:

- prediction arrays,
- target arrays,
- local score JSON,
- diagnostic JSON,
- selected hypothesis JSON.

This keeps the agent loop auditable and prevents the model from replacing scientific diagnosis with unconstrained prose.

## Components

### 1. `generate-baseline-prediction`

Purpose:

Create a minimal prediction NPZ from a local window NPZ so the diagnostic skills have a real failure case.

Input:

- `--windows-path`: NPZ produced by `build-local-windows`
- `--output-path`: prediction NPZ path
- `--method`: initially only `persistence`
- `--force`: overwrite existing output

Reads:

- `context` from the windows NPZ
- `target` only for shape metadata, not for prediction values

Writes:

- NPZ with key `prediction`
- optional scalar `metadata_json`

Persistence method:

```text
last_context = context[:, -1:, :]
prediction = repeat(last_context, forecast_steps, axis=1)
```

The baseline is intentionally weak. Its purpose is not leaderboard performance; it establishes a reproducible lower bound and produces meaningful rollout error.

Output payload:

- `state_hint`: `BASELINE_PREDICTION_READY`
- artifact: `prediction_npz_path`
- metrics:
  - `method`
  - `sample_count`
  - `forecast_steps`
  - `space_points`

### 2. `diagnose-rollout-error`

Purpose:

Convert prediction-vs-target arrays and local score JSON into a compact numerical error diagnosis.

Input:

- `--prediction-path`
- `--target-path`
- `--score-path`
- `--output-path`

Reads:

- prediction NPZ key `prediction`
- target/window NPZ key `target`
- score JSON from `score-local-rollout`

Writes:

- `diagnostics/rollout_error_diagnosis.json`

Metrics:

- `global_rmse`
- `global_rel_mse`
- `early_rel_mse`
- `middle_rel_mse`
- `late_rmse`
- `time_error_slope`
- `max_error_time_index`
- `mean_spatial_error`
- `max_spatial_error`
- `dominant_error_pattern`

Required pattern labels:

- `short_horizon_bias`
- `long_horizon_drift`
- `overdamped_rollout`
- `oscillatory_error`
- `shock_region_error`
- `low_error`

Initial classification rules:

- If total score is near perfect and RMSE is tiny, label `low_error`.
- If late error dominates early error, label `long_horizon_drift`.
- If prediction total variation is much lower than target total variation, label `overdamped_rollout`.
- If high-frequency error proxy is large and sign changes are excessive, label `oscillatory_error`.
- If error concentrates near large target gradients, label `shock_region_error`.
- Otherwise label `short_horizon_bias`.

The rules should be deterministic and documented in JSON evidence fields so Hermes can cite them.

### 3. `check-burgers-physics-local`

Purpose:

Estimate simple Burgers physics consistency indicators from predicted and target local trajectories.

Input:

- `--prediction-path`
- `--target-path`
- `--output-path`
- `--nu`: default `0.001`
- `--dt`: default `1.0`
- `--dx`: default `1.0`

Reads:

- prediction NPZ key `prediction`
- target/window NPZ key `target`

Writes:

- `diagnostics/burgers_physics_check.json`

Finite-difference indicators:

- periodic or edge-padded spatial first derivative `u_x`
- spatial second derivative `u_xx`
- forward time derivative `u_t`
- residual proxy `u_t + u * u_x - nu * u_xx`

Metrics:

- `prediction_residual_rmse`
- `target_residual_rmse`
- `residual_ratio`
- `mass_drift`
- `energy_decay_ratio`
- `total_variation_ratio`
- `gradient_error_ratio`
- `shock_region_error_ratio`
- `physics_issue`

Required issue labels:

- `overdissipation`
- `nonphysical_oscillation`
- `residual_mismatch`
- `mass_drift`
- `stable_low_error`
- `inconclusive`

This is a diagnostic proxy, not a proof of PDE correctness. Because the local windows are downsampled, the default `dt` and `dx` are relative units unless metadata provides physical spacing.

### 4. `select-pde-hypothesis`

Purpose:

Map local score, rollout error diagnosis, and physics diagnosis into a bounded next experimental hypothesis.

Input:

- `--score-path`
- `--error-diagnosis-path`
- `--physics-check-path`
- `--output-path`

Writes:

- `diagnostics/selected_pde_hypothesis.json`

Output fields:

- `selected_hypothesis`
- `rationale`
- `expected_effect`
- `risk`
- `next_experiment_scope`
- `required_artifacts`
- `rejected_hypotheses`
- `state_hint`

Initial hypothesis labels:

- `add_spectral_consistency_loss`
- `add_residual_correction_head`
- `use_scheduled_sampling_rollout_finetune`
- `add_gradient_weighted_loss`
- `apply_low_pass_stabilization`
- `keep_baseline_and_collect_stronger_evidence`

Mapping rules:

- `overdamped_rollout` or `overdissipation` -> `add_spectral_consistency_loss` or `add_gradient_weighted_loss`.
- `long_horizon_drift` -> `use_scheduled_sampling_rollout_finetune`.
- `residual_mismatch` with moderate score -> `add_residual_correction_head`.
- `nonphysical_oscillation` -> `apply_low_pass_stabilization`.
- `low_error` or contradictory diagnostics -> `keep_baseline_and_collect_stronger_evidence`.

This command should not edit model code. It only produces the next bounded experiment proposal.

## Templates And Skills

Add prompt templates that tell Hermes when and how to call the new commands.

Recommended templates:

- `src/pde_agent/templates/command/call_generate_baseline_prediction.md`
- `src/pde_agent/templates/command/call_diagnose_rollout_error.md`
- `src/pde_agent/templates/command/call_check_burgers_physics_local.md`
- `src/pde_agent/templates/command/call_select_pde_hypothesis.md`

Each template should include:

- required input files,
- exact CLI command shape,
- expected output artifact,
- forbidden shortcuts,
- how to summarize the JSON result in the research log.

The templates should be concrete enough for Kimi/Hermes. Avoid abstract instructions such as "analyze the PDE carefully" without a required command and artifact.

## Data Flow

Example local run:

```bash
uv run --frozen --extra data pde-agent build-local-windows \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-path data/derived/local_windows_task1_val.npz \
  --dataset-key tensor \
  --sample-limit 128 \
  --context-steps 10 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4

uv run --frozen pde-agent generate-baseline-prediction \
  --windows-path data/derived/local_windows_task1_val.npz \
  --output-path data/derived/local_persistence_task1_val.npz \
  --method persistence

uv run --frozen pde-agent score-local-rollout \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_score.json

uv run --frozen pde-agent diagnose-rollout-error \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --score-path runs/local_score.json \
  --output-path runs/rollout_error_diagnosis.json

uv run --frozen pde-agent check-burgers-physics-local \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/burgers_physics_check.json \
  --nu 0.001

uv run --frozen pde-agent select-pde-hypothesis \
  --score-path runs/local_score.json \
  --error-diagnosis-path runs/rollout_error_diagnosis.json \
  --physics-check-path runs/burgers_physics_check.json \
  --output-path runs/selected_pde_hypothesis.json
```

## Error Handling

All new core functions should raise command-specific `RuntimeError` subclasses with concise messages.

Required validation:

- missing input path,
- missing NPZ key,
- non-rank-3 arrays,
- prediction and target shape mismatch,
- non-finite arrays,
- output exists without `--force` where applicable,
- unknown baseline method,
- unknown or incomplete diagnostic JSON.

CLI wrappers may let exceptions surface as command failures, consistent with existing harness commands.

All JSON files must be written with `allow_nan=False` when possible, or otherwise explicitly reject non-finite values before writing.

## Testing Strategy

Unit tests should use tiny NPZ fixtures and avoid the large HDF5 file.

Required tests:

- Persistence baseline repeats the last context frame for the requested horizon.
- Baseline generator refuses overwrite unless `--force`.
- Rollout diagnosis returns `low_error` for perfect prediction.
- Rollout diagnosis returns `long_horizon_drift` when late error dominates.
- Rollout diagnosis returns `overdamped_rollout` when prediction total variation is much lower than target total variation.
- Physics check returns finite residual metrics for smooth synthetic trajectories.
- Physics check flags `overdissipation` when prediction total variation and energy are too low.
- Hypothesis selector maps overdamping to a spectral or gradient-preserving hypothesis.
- CLI smoke tests write JSON payloads and artifacts for each command.

End-to-end smoke:

- Build tiny local windows from fixture NPZ or generated arrays.
- Generate persistence prediction.
- Score it.
- Run both diagnostics.
- Select a hypothesis.
- Assert all expected artifacts exist and JSON is parseable.

Real HDF5 smoke may be added as a guarded command in documentation or integration scripts, but it should not be required for unit tests.

## Success Criteria

- Hermes can run the full local diagnostic chain without human interpretation between commands.
- The chain produces a baseline prediction, a local score, an error diagnosis, a physics diagnosis, and a selected PDE hypothesis.
- Every artifact is JSON or NPZ with explicit keys.
- The selected hypothesis is evidence-backed by diagnostics, not free-form LLM opinion.
- Full test suite remains fast and does not require the 7.7GB HDF5 file.

## Deferred Work

- FNO checkpoint loading.
- Train/fine-tune loop.
- Long-horizon `10 -> 190` submission generation.
- Task 2 multi-nu conditioning.
- Exact Frechet component implementation.
- Automatic model-code patching from selected hypotheses.

