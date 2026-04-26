# FNO Training Strategy V0 Design

## Goal

Extend the FNO finetune executor from a single hard-coded scheduled-sampling
loss into a small strategy switch that the agent can compare through the
existing candidate/evaluation/comparison loop.

V0 adds one new candidate parameter:

```json
"training_strategy": "scheduled_sampling"
```

Supported values:

- `scheduled_sampling`: current uniform rollout MSE behavior.
- `late_weighted_scheduled_sampling`: same rollout schedule, but later rollout
  steps receive larger loss weights.

## Rationale

The competition scoring gives substantial weight to later rollout behavior.
Before changing architecture, the agent should be able to ask a narrower
scientific question: does emphasizing late-step rollout loss improve local
long-horizon stability without damaging early steps?

This is a training strategy extension, not a new model family.

## Execution Contract

`execute-fno-finetune-candidate` remains the executor. It reads
`candidate.parameters.training_strategy` if present and defaults to
`scheduled_sampling` for backward compatibility.

The execution report adds:

- `train.training_strategy`
- `train.loss_weighting`

The prediction and checkpoint artifact names stay unchanged.

## Planner Contract

The `use_scheduled_sampling_rollout_finetune` intervention policy adds:

```json
"training_strategy": [
  "scheduled_sampling",
  "late_weighted_scheduled_sampling"
]
```

Because experiment-config expansion sorts parameter names, the first two
candidates compare the two strategies under the same rollout and teacher-forcing
settings. This keeps the multi-candidate loop interpretable.

## Guardrails

- Reject unknown training strategies.
- Keep existing candidates valid when `training_strategy` is missing.
- Keep all training CPU-only in V0.
- Do not add new model layers or physics losses in this step.

## Acceptance

- Fixture test proves the late-weighted strategy executes and records strategy
  metadata.
- Fixture test proves unknown strategies are rejected.
- Planning test proves the scheduled-sampling policy exposes both strategies.
- Existing FNO finetune, candidate evaluation, and comparison tests still pass.
