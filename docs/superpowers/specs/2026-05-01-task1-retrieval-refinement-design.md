# Task1 Retrieval Refinement Design

## Purpose

Hypothesis Lab can already route local strategy cards into a bounded sandbox and promote the Task 1 retrieval prior. The current winning retrieval prior is still narrow: it sweeps only `k` over `[8, 12, 20]` with `mode="affine"`. On Task 1 validation it reaches about `109.22` proxy score, with strong late-window performance but weak early and middle scores.

This spec adds one candidate family: `task1_retrieval_refinement`. The goal is to let Hermes explore controlled retrieval variants that can improve early and middle windows without giving up the late-window stability that made retrieval useful.

## Scope

In scope:

- Add a Hypothesis Lab family named `task1_retrieval_refinement`.
- Generate this family from the existing Task 1 retrieval prior card.
- Execute it inside the existing numpy sandbox.
- Sweep bounded parameters for neighbor count, retrieval mode, horizon blending, and optional smoothing.
- Record enough metadata to explain which refinement won and how it affected early, middle, and late segments.

Out of scope:

- No external literature retrieval.
- No arbitrary generated code execution.
- No neural network training or checkpoint fine-tuning.
- No Task 2 optimization in this step.
- No replacement of the existing `task1_retrieval_prior`; this is an additional candidate family.

## Candidate Family

`task1_retrieval_refinement` builds a full prediction in three stages:

1. Produce a retrieval future using the existing `predict_retrieval_prior` kernel.
2. Produce a stable fallback future using persistence from the 10-frame context.
3. Blend the two by forecast horizon and optionally smooth only the predicted future window.

The intended mechanism is:

- Early future steps may benefit from persistence/context anchoring because nearest-neighbor futures can be phase-misaligned immediately after the context window.
- Middle and late future steps should increasingly trust retrieval because the validation evidence shows retrieval dominates persistence on long-horizon Burgers rollouts.
- Optional spectral smoothing may suppress retrieval-induced high-frequency mismatch, but it must not touch the first 10 context frames.

## Parameter Space

The first bounded grid is:

- `k`: `[12, 20, 32, 48]`
- `mode`: `["affine", "delta", "raw"]`
- `early_blend_steps`: `[0, 20, 40]`
- `persistence_weight_start`: `[0.0, 0.25, 0.5]`
- `spectral_strength`: `[0.0, 0.5, 1.0]`

Blend rule:

- `persistence_weight_start=0.0` means pure retrieval.
- For forecast step `h`, if `early_blend_steps > 0`, persistence weight decays linearly from `persistence_weight_start` at `h=0` to `0.0` at `h=early_blend_steps`.
- Retrieval weight is `1 - persistence_weight`.
- After blending, optional spectral smoothing applies to the full prediction array, then the first 10 context frames are restored exactly.

The grid is intentionally small enough for local validation. If it is too slow on the full 100-sample validation set, the implementation may cap the first smoke to a subset, but promotion evidence must be generated on the full Task 1 validation tensor.

## Sandbox And Verification

The family uses a new allowed executor label:

- `controlled_numpy_retrieval_refinement`

Verification remains the existing Task 1 local judge and segment scorer:

- Primary metric: `task1_total_score_proxy`
- Required invariant: first 10 context frames preserved within existing tolerance
- Required metadata: `leave_one_out`, `reference_count`, `k`, `mode`, `early_blend_steps`, `persistence_weight_start`, `spectral_strength`
- Promotion threshold: score delta over persistence baseline must be at least `0.01`

The sandbox report must keep top candidates with full early/middle/late segment scores so Hermes can learn whether the refinement improved the actual weak windows or only traded runtime for the same score.

## Integration Points

Changes should stay localized to:

- `src/pde_agent/core/hypothesis_lab.py`
- `tests/test_hypothesis_lab.py`

Expected additions:

- A hypothesis generated from retrieval prior cards with `candidate_family="task1_retrieval_refinement"`.
- A blueprint with `blueprint_id="bp_task1_retrieval_refinement_001"`.
- A parameter-grid iterator for refinement params.
- A prediction helper that composes retrieval, persistence blending, optional smoothing, and context restoration.
- Candidate IDs that encode the main parameters.

Heartbeat and batch loop should not need changes because they already call Hypothesis Lab as a candidate generator.

## Testing Plan

Tests should be added before implementation:

- Card to hypothesis to blueprint: retrieval cards generate both the existing retrieval prior blueprint and the new refinement blueprint.
- Sandbox execution: a synthetic leave-one-out fixture can execute `task1_retrieval_refinement`, preserve context frames, and report refinement metadata.
- Parameter grid: the refinement grid expands the expected cross product and rejects malformed grids.
- Regression: existing retrieval prior tests continue to pass.

After unit tests pass, run a real Task 1 validation smoke through `run-hypothesis-lab` and compare the best refinement candidate against the current `task1_retrieval_prior_k20_affine` score.
