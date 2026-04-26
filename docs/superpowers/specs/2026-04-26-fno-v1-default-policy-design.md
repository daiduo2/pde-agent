# FNO V1 Default Policy Design

## Goal

Promote the evidence-backed FNO finetune configuration into the default agent
policy so future runs start from the current best local setting instead of the
earlier exploratory grid.

Evidence used:

- `scheduled_sampling` won all strategy comparisons.
- `rollout_steps=10` was more robust than longer rollout training.
- `teacher_forcing_decay=0.85` had the strongest average score across windows.
- `learning_rate=1e-4` was the most robust learning rate.
- `train_steps=16` gave the best average local score, with one-window overfit
  risk that should be monitored by evidence sweeps.

## Policy Changes

Scheduled-sampling planner policy becomes:

```json
{
  "training_strategy": ["scheduled_sampling"],
  "rollout_steps": [10],
  "teacher_forcing_decay": [0.85, 0.9]
}
```

The first candidate is the V1 default. The second candidate is a nearby fallback
that is still scientifically meaningful and keeps the comparison pipeline
non-degenerate.

Executor defaults become:

- `max_train_steps = 16`
- `learning_rate = 1e-4`

`learning_rate` remains an executor-level argument rather than candidate
parameter in V1. This avoids changing the experiment-config schema before there
is a need to compare learning rates inside ordinary agent loops.

## Guardrails

- Keep older candidate plans valid.
- Keep explicit CLI values authoritative over defaults.
- Do not add new training strategies.
- Do not change artifact names.
- Keep `run-fno-evidence-sweep` usable with one or more generated candidates.

## Acceptance

- Planner test proves V1 policy parameter ordering.
- CLI test proves `execute-fno-finetune-candidate` help/default path uses
  `max_train_steps=16`.
- Real smoke proves generated candidate 0 uses V1 default parameters and
  executor report records `train_steps=16`.
