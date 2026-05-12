# FNO Training Protocol v2 Design

## Goal

Upgrade the current FNO training sprint from a smoke-level loop into a bounded, agent-usable training protocol that can produce meaningful early/middle/late evidence and a better next-round strategy.

## Current State

The Claude SDK agent can already execute the FNO harness, plan candidates, run training, record evidence, and validate a hypothesis design. The latest horizon-coverage sprint produced candidates at `forecast_steps = 47, 95, 190`, but all candidates were very weak: early rel-MSE stayed around `1.22`, and middle/late rel-MSE stayed above `2.5`.

The main bottleneck is not orchestration. It is the training protocol:

- the candidate budget uses smoke-level `train_steps`;
- input and target tensors are not normalized;
- loss functions do not explicitly stabilize full-horizon behavior;
- FNO selection still favors the best early-only candidate even when middle/late guardrails are informative.

## Scope

This v2 protocol stays inside the existing `task1_short_window_training.py` harness and preserves the Claude SDK command surface. It does not introduce a full official FNO implementation, new datasets, Task2, or checkpoint fine-tuning.

## Design

### 1. Normalized Training

Add a `normalization` option for FNO training candidates:

- `none`: current behavior.
- `sample_standard`: normalize each sample using its context mean/std over time and space, train in normalized space, and denormalize predictions before writing metrics/artifacts.

The first implementation should support the option in `run_task1_short_window_training_loop`, candidate configs, reports, and evidence. The default remains `none` for backward compatibility.

### 2. Horizon-Aware Loss

Add a `horizon_weighted_mse` loss type. It should weight early steps strongly but still keep non-zero middle/late pressure across the full forecast horizon. A simple deterministic schedule is enough:

- first 25% of horizon: weight `1.5`
- middle 25%: weight `1.0`
- final 50%: weight `0.75`

This is intentionally conservative. It gives the agent a loss candidate that is different from `early_weighted_mse` without introducing fragile physics code.

### 3. Multi-Objective FNO Selection

Add a candidate aggregate score used only for FNO optimization selection. It should combine:

- early score as primary signal;
- middle guardrail score when evaluated;
- late guardrail score when evaluated;
- runtime as a tie-breaker.

The selected candidate record should include `fno_selection_score` and `selection_components`. If middle/late are missing, the score should explicitly record that rather than silently treating them as success.

Initial weighting:

- early: `0.60`
- middle: `0.25`
- late: `0.15`

This still prioritizes early segment, but it prevents the agent from ignoring all horizon-coverage evidence.

### 4. Candidate Planning Surface

Extend FNO candidate planning and harness contracts so Claude SDK agent can vary:

- `normalization`;
- `loss_type`, including `horizon_weighted_mse`;
- `forecast_steps`;
- existing `modes`, `width`, `depth`, and `train_steps`.

The planner should preserve horizon coverage behavior from the previous fix.

## Success Criteria

1. Tests prove `sample_standard` normalization trains and reports denormalized metrics.
2. Tests prove `horizon_weighted_mse` is accepted and appears in candidate planning.
3. Tests prove FNO selection uses a multi-objective score that can select a candidate with slightly worse early score but evaluated middle/late guardrails.
4. A Claude SDK sprint can run with v2 candidates and produce evidence containing:
   - `normalization`;
   - `horizon_weighted_mse` or another v2 loss;
   - `fno_selection_score`;
   - early/middle/late metrics;
   - next-generation constraints.
5. Full test suite passes.

## Non-Goals

- Do not optimize leaderboard score in this step.
- Do not introduce Task2.
- Do not replace the lightweight FNO model with a large external implementation.
- Do not require GPU-specific code.
