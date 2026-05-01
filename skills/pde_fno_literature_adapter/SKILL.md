---
name: pde-fno-literature-adapter
description: Use when extracting AlgorithmCard v2 records from Fourier Neural Operator/FNO literature context for Hypothesis Lab, especially FNO_LITERATURE_ADAPTATION_TASK_PREPARED artifacts.
---

# PDE FNO Literature Adapter

Use this skill with `pde-algorithm-card-extractor` after reading a `FNO_LITERATURE_ADAPTATION_TASK_PREPARED` artifact.

This skill is a domain reading guide, not a rule-based extractor. Use your understanding of neural operators and PDE rollouts, but ground every card in the supplied context units.

## What Counts As An FNO Algorithm Card

Extract an AlgorithmCard only when the paper text supports an actionable mechanism such as:

- Fourier/spectral convolution: learned weights on low Fourier modes.
- Mode truncation or spectral filtering: finite retained modes, high-frequency damping, anti-aliasing, or stability control.
- Lifting/projection stack: pointwise maps around spectral layers that change channel representation.
- Resolution generalization: mechanisms that claim discretization-invariant or mesh-independent operator learning.
- Autoregressive rollout or long-horizon stabilization: training or inference changes that affect temporal propagation.
- Checkpoint or finetuning adaptation: using pretrained FNO weights, calibration, or small supervised updates.
- Physics or residual constraints around FNO: PDE residual loss, conservation penalty, or spectral residual objective.

Do not extract generic motivation, benchmark claims, or architecture names unless the text implies a runnable experiment.

## Hook Guidance

Prefer these hooks when they fit the grounded mechanism:

- `fourier_neural_operator_architecture`: full FNO architecture or spectral convolution implementation. This requires bounded executor design before sandboxing.
- `spectral_rollout_stabilization`: inference-time spectral filtering, mode damping, or high-frequency stabilization.
- `official_checkpoint_calibration`: official/pretrained FNO checkpoint adaptation or calibration.
- `scheduled_rollout_training`: autoregressive rollout training or teacher-forcing schedule.
- `physics_residual_regularization`: PDE residual, conservation, or physics-informed loss.

Use a new descriptive hook only when none of these match. New hooks must be marked as needing bounded executor design.

## Burgers/PDEBench Lens

For this project, translate claims into 1D Burgers Task 1 experiments when possible:

- Inputs are first 10 time steps, output is 200 time steps at 256 spatial points.
- First 10 predicted steps must match the context exactly.
- Useful observables include segment score, late RMSE, high-frequency energy growth, Burgers residual, runtime, and context preservation.
- A full FNO architecture claim should not be promoted without bounded executor or checkpoint-family sandbox evidence.

## Card Quality Bar

Each card must include:

- A claim grounded in one or more context units.
- A mechanism that explains what changes in model, loss, rollout, or postprocess.
- A minimal sandbox experiment that can reject the mechanism.
- Failure modes that would make the mechanism invalid for Burgers rollouts.
- Evidence spans with `unit_id`, `section_title`, `line_start`, and `line_end`.

If the FNO context is only descriptive and has no actionable mechanism, return an empty extraction report with `failure_code: "no_algorithmic_claim"`.
