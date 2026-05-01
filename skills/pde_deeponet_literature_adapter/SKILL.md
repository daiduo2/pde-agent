---
name: pde-deeponet-literature-adapter
description: Use when extracting AlgorithmCard v2 records from DeepONet or PI-DeepONet literature context for Hypothesis Lab, especially DEEPONET_LITERATURE_ADAPTATION_TASK_PREPARED artifacts.
---

# PDE DeepONet Literature Adapter

Use this skill with `pde-algorithm-card-extractor` after reading a `DEEPONET_LITERATURE_ADAPTATION_TASK_PREPARED` artifact.

This skill is a domain reading guide, not a rule-based extractor. Use your understanding of branch/trunk neural operators and physics-informed DeepONet, but ground every card in the supplied context units.

## What Counts As A DeepONet Algorithm Card

Extract an AlgorithmCard only when the paper text supports an actionable mechanism such as:

- Branch/trunk architecture: branch encodes an input function or sensors, trunk encodes coordinates, and the merge rule produces solution values.
- Sensor or context encoding: how input samples, initial conditions, or context windows become branch features.
- Coordinate basis or evaluation grid: how time/space coordinates are represented by trunk features.
- PI-DeepONet residual training: Burgers residual, boundary/initial penalties, conservation terms, or automatic differentiation residuals.
- Rollout construction: pointwise operator evaluation assembled into a full trajectory, or autoregressive use of DeepONet predictions.
- Hybrid correction: DeepONet as a residual or calibration layer on top of another rollout.

Do not extract generic universal approximation claims, benchmark tables, or high-level motivation unless the text implies a runnable experiment.

## Hook Guidance

Prefer these hooks when they fit the grounded mechanism:

- `deeponet_branch_trunk_architecture`: full branch/trunk operator implementation. This requires bounded executor design before sandboxing.
- `physics_residual_regularization`: PDE residual, PI-DeepONet loss, conservation, or boundary/initial penalties.
- `scheduled_rollout_training`: autoregressive rollout training, teacher forcing, or scheduled sampling.
- `operator_correction_postprocess`: DeepONet used as a residual corrector or calibration operator.

Use a new descriptive hook only when none of these match. New hooks must be marked as needing bounded executor design.

## Burgers/PDEBench Lens

For this project, translate claims into 1D Burgers Task 1 experiments when possible:

- Inputs are first 10 time steps, output is 200 time steps at 256 spatial points.
- First 10 predicted steps must match the context exactly.
- Useful observables include segment score, coordinate-grid reconstruction error, late RMSE, Burgers residual, runtime, and context preservation.
- A full branch/trunk architecture claim should not be promoted without bounded executor or trainable DeepONet sandbox evidence.

## Card Quality Bar

Each card must include:

- A claim grounded in one or more context units.
- A mechanism that explains what changes in model, loss, rollout, conditioning, or postprocess.
- A minimal sandbox experiment that can reject the mechanism.
- Failure modes that would make the mechanism invalid for Burgers rollouts.
- Evidence spans with `unit_id`, `section_title`, `line_start`, and `line_end`.

If the DeepONet context is only descriptive and has no actionable mechanism, return an empty extraction report with `failure_code: "no_algorithmic_claim"`.
