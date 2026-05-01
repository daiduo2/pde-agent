---
name: pde-unet-literature-adapter
description: Use when extracting AlgorithmCard v2 records from UNet/U-Net PDE rollout literature context for Hypothesis Lab, especially UNET_LITERATURE_ADAPTATION_TASK_PREPARED artifacts.
---

# PDE UNet Literature Adapter

Use this skill with `pde-algorithm-card-extractor` after reading a `UNET_LITERATURE_ADAPTATION_TASK_PREPARED` artifact.

This skill is a domain reading guide, not a rule-based extractor. Use your understanding of convolutional encoder-decoder rollout models, but ground every card in the supplied context units.

## What Counts As A UNet Algorithm Card

Extract an AlgorithmCard only when the paper text supports an actionable mechanism such as:

- Encoder-decoder architecture: multi-scale convolutional representation, downsampling/upsampling, and skip connections.
- Push-forward or frame rollout: predicting future frames from a context window or rolling predictions autoregressively.
- Checkpoint or finetuning adaptation: official/pretrained UNet weights, calibration, small supervised updates, or model selection.
- Stability controls: smoothing, residual updates, spectral filters, or loss terms that reduce long-horizon drift.
- Hybrid correction: UNet output blended with persistence, checkpoint rollout, retrieval prior, or another operator family.

Do not extract generic image segmentation background, benchmark claims, or architecture names unless the text implies a runnable PDE rollout experiment.

## Hook Guidance

Prefer these hooks when they fit the grounded mechanism:

- `unet_encoder_decoder_architecture`: full UNet architecture or push-forward predictor. This requires bounded executor design before sandboxing.
- `official_checkpoint_calibration`: official/pretrained UNet checkpoint adaptation or calibration.
- `scheduled_rollout_training`: push-forward, teacher forcing, scheduled sampling, or autoregressive rollout training.
- `operator_correction_postprocess`: UNet used as a residual corrector, denoiser, or calibration postprocess.

Use a new descriptive hook only when none of these match. New hooks must be marked as needing bounded executor design.

## Burgers/PDEBench Lens

For this project, translate claims into 1D Burgers Task 1 experiments when possible:

- Inputs are first 10 time steps, output is 200 time steps at 256 spatial points.
- First 10 predicted steps must match the context exactly.
- Useful observables include segment score, late RMSE, drift, high-frequency artifacts, runtime, and context preservation.
- A full UNet architecture claim should not be promoted without bounded executor, official checkpoint, or trainable UNet sandbox evidence.

## Card Quality Bar

Each card must include:

- A claim grounded in one or more context units.
- A mechanism that explains what changes in model, loss, rollout, checkpoint use, or postprocess.
- A minimal sandbox experiment that can reject the mechanism.
- Failure modes that would make the mechanism invalid for Burgers rollouts.
- Evidence spans with `unit_id`, `section_title`, `line_start`, and `line_end`.

If the UNet context is only descriptive and has no actionable mechanism, return an empty extraction report with `failure_code: "no_algorithmic_claim"`.
