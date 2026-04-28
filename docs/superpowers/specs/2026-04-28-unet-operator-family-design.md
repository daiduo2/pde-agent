# UNet Operator Family Design

## Context

The Burgers real-HDF5 benchmark now proves a full local and Hermes-audited loop for
the retrieved strategy `burgers_spectral_viscosity_real_hdf5_v0`. The observed
DeepONet strategy sweep is useful but weak: the best branch/trunk spectral arm is
still below persistence, with negative mean and long-window score deltas. The
strategy gate correctly routes this to external strategy retrieval.

The next step is to add UNet as a third operator-family baseline alongside FNO and
DeepONet. This supports the agent competition by increasing the agent's available
strategy space, and it supports future AI4PDE work by adding a local convolutional
inductive bias that is distinct from Fourier and branch/trunk operator models.

## Goal

Build a minimal trainable 1D UNet candidate path that can be quantified with the
same local rollout, Burgers residual, spectrum, and operator-evidence contracts
already used by DeepONet and FNO.

This design intentionally stops before a Hermes fixture. The first deliverable
must be a reliable local UNet evidence chain; Hermes canary/audit should be added
after the local report schema and CLI are stable.

## Non-Goals

- Do not replace the existing Burgers spectral-viscosity benchmark.
- Do not add a large research-grade UNet implementation with schedulers,
  checkpoint resumes, mixed precision, or multi-GPU training.
- Do not require external datasets beyond the existing local window `.npz` files.
- Do not add a Hermes fixture in this step.
- Do not tune UNet until it beats persistence. The first success criterion is
  evidence quality and protocol compatibility.

## Architecture

The implementation mirrors the DeepONet trainable path:

1. `execute_unet_candidate` trains a small torch UNet on an existing local window
   file and writes a prediction `.npz`, checkpoint, and execution report.
2. `evaluate_unet_candidate` runs the existing local diagnostics against the
   prediction and writes an evaluation report with the same score, rollout-error,
   Burgers-physics, residual, and hypothesis sections used by DeepONet.
3. `run_unet_strategy_sweep` runs a bounded Cartesian sweep across windows,
   strategies, model widths, train steps, learning rates, seeds, and loss weights.
4. `build_unet_evidence_sweep` normalizes UNet evaluation reports into a sweep
   summary with comparison winners and strategy statistics.
5. `convert-operator-evidence-summary` remains the shared bridge into
   strategy-growth evidence by using `model_family=UNet` and
   `source_state_hint=UNET_EVIDENCE_SWEEP_COMPLETED`.

The core implementation should live in small UNet-specific modules rather than
generalizing DeepONet modules prematurely. Duplication is acceptable where it keeps
contracts explicit. Generalization can come later if FNO, DeepONet, and UNet all
share enough stable behavior.

## Candidate Execution Contract

Create `src/pde_agent/core/unet_candidate_execution.py` with:

- public function: `execute_unet_candidate(...) -> dict[str, Any]`
- state hint: `UNET_CANDIDATE_EXECUTED`
- output files:
  - `candidate_prediction.npz`
  - `candidate_unet_checkpoint.pt`
  - `candidate_execution_report.json`
- model metadata:
  - `model_type = "unet"`
  - `adapter_type = "torch_unet1d_minimal"`
  - `experiment_id`
  - `training_strategy`
  - `strategy_label`
  - `base_channels`
  - `depth`
  - `learning_rate`
  - `seed`

The model input is `context` with shape `(sample, context_time, space)`. The model
should treat context time as channels and produce `forecast_steps` output channels,
then transpose to prediction shape `(sample, forecast_time, space)`.

The minimal architecture is a 1D encoder/decoder with skip connections:

- input projection from `context_steps` channels to `base_channels`
- `depth` down blocks using Conv1d/ReLU/Conv1d/ReLU plus average pooling
- bottleneck Conv1d/ReLU/Conv1d/ReLU
- decoder upsampling by interpolation, concatenating skip activations
- output projection to `forecast_steps` channels

The architecture must preserve the spatial length for the existing local scoring
contracts. If pooling changes length because of odd spatial sizes, the decoder
should interpolate back to the skip length before concatenation and finally to the
target spatial length.

## Training Strategies

Support three initial strategies:

- `unet_supervised`: MSE only.
- `unet_spectral`: MSE plus spectral magnitude MSE along the space dimension.
- `unet_physics_residual`: MSE plus Burgers residual energy on the predicted
  rollout.

Use the same loss-weight semantics as DeepONet:

- `spectral_weight`, default `0.1`
- `physics_weight`, default `0.01`
- `nu`, `dt`, `dx` for residual calculation

The execution report must include `train.loss_terms`, `loss_trace`,
`initial_loss`, `final_loss`, and `loss_weights`.

## Evaluation Contract

Create `src/pde_agent/core/unet_candidate_evaluation.py` with:

- public function: `evaluate_unet_candidate(...) -> dict[str, Any]`
- accepted execution state hint: `UNET_CANDIDATE_EXECUTED`
- output state hint: `UNET_CANDIDATE_EVALUATED`
- output files:
  - `candidate_score.json`
  - `candidate_rollout_error.json`
  - `candidate_burgers_physics.json`
  - `candidate_burgers_residual.json`
  - `candidate_selected_pde_hypothesis.json`
  - `candidate_evaluation_report.json`

This module should call the existing scoring and diagnostic helpers exactly as
DeepONet does. The report shape should stay close enough for operator evidence to
consume the same fields: candidate identifiers, score, error diagnosis, physics,
physics residual, and optional baseline comparison.

## Evidence Sweep Contract

Create `src/pde_agent/core/unet_evidence_sweep.py` with:

- state hint: `UNET_EVIDENCE_SWEEP_COMPLETED`
- evaluation state hint: `UNET_CANDIDATE_EVALUATED`
- `model_family = "UNet"`
- records containing:
  - `start_index`
  - `train_steps`
  - `experiment_id`
  - `training_strategy`
  - `forecast_steps`
  - baseline and candidate scores
  - residual metrics when present
  - `evaluation_report_path`
- comparison grouping by `(start_index, train_steps)`
- `strategy_statistics`
- `next_command = "convert-operator-evidence-summary"`

This keeps UNet compatible with the existing generic operator evidence converter.

## Strategy Sweep Contract

Create `src/pde_agent/core/unet_strategy_sweep.py` with:

- public function: `run_unet_strategy_sweep(...) -> dict[str, Any]`
- state hint: `UNET_STRATEGY_SWEEP_COMPLETED`
- output files:
  - `unet_strategy_sweep_summary.json`
  - `unet_strategy_sweep_summary.csv`
  - `unet_evidence/evidence_summary.json`
  - `unet_evidence/evidence_summary.csv`

Sweep parameters:

- `start_index` or `start_indices`
- `training_strategies`
- `train_steps`
- `base_channels`
- `depths`
- `learning_rates`
- `seeds`
- `spectral_weights`
- `physics_weights`
- `nu`, `dt`, `dx`
- `force`

Strategy labels should mirror the DeepONet label style, replacing hidden/latent
parts with UNet-specific hyperparameters:

- one strategy: `c8_d2_lr1e2_seed5`
- multiple strategies: `spectral_c8_d2_lr1e2_seed5`
- spectral weight sweep: `spectral_sw5e2_c8_d2_lr1e2_seed5`
- physics weight sweep: `physics_residual_pw5e3_c8_d2_lr1e2_seed5`

The sweep recommendation is deliberately modest: it should recommend more UNet
strategy sweep work unless a single tested strategy is stable and every candidate
improves over baseline.

## CLI Surface

Add commands that match existing naming patterns:

- `execute-unet-candidate`
- `evaluate-unet-candidate`
- `build-unet-evidence-sweep`
- `run-unet-strategy-sweep`

Each command returns the standard `success_payload` JSON envelope and exposes only
the parameters needed by the corresponding core function.

## Testing

Use TDD and keep tests small:

- `tests/test_unet_candidate_execution.py`
  - trains a tiny UNet on synthetic local windows and writes all artifacts
  - CLI returns `UNET_CANDIDATE_EXECUTED`
  - strategy labels make experiment IDs unique
  - spectral and physics-residual strategies expose the correct loss terms
- `tests/test_unet_candidate_evaluation.py`
  - evaluates a generated candidate and writes the full diagnostic bundle
  - validates the CLI payload
- `tests/test_unet_evidence_sweep.py`
  - converts evaluation reports into UNet evidence records
  - rejects duplicate experiment IDs
- `tests/test_unet_strategy_sweep.py`
  - writes strategy summary and UNet evidence
  - accepts multiple start indices
  - accepts all three training strategies
  - expands loss weights by strategy
  - CLI returns the expected summary payload
- `tests/test_operator_strategy_evidence.py`
  - add or reuse coverage proving `model_family=UNet` and
    `UNET_EVIDENCE_SWEEP_COMPLETED` convert into generic strategy evidence

Final verification must run the targeted UNet tests and the full suite.

## Success Criteria

- A tiny local UNet can train and write protocol-compatible artifacts.
- UNet evaluation reports use the same quantitative diagnostic sections as
  DeepONet.
- UNet evidence summaries convert through the existing operator evidence bridge.
- The CLI can run a bounded UNet strategy sweep from local windows and baseline
  score.
- The implementation adds no Hermes fixture yet, leaving the next step clean:
  integrate UNet into the Burgers real-HDF5 benchmark or create a dedicated Hermes
  UNet canary after local behavior is stable.
