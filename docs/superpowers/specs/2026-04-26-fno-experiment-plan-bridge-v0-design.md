# FNO Experiment Plan Bridge V0 Design

## Goal

Let Hermes plan the next bounded experiment from a single FNO rollout summary
plus the selected hypothesis, without manually passing score, rollout-error, and
physics artifact paths.

## Current Context

The current FNO chain is:

```text
diagnose-fno-rollout
  -> fno_rollout_summary.json
select-pde-hypothesis-from-fno-summary
  -> selected_pde_hypothesis.json
plan-next-experiment
  -> next_experiment_plan.json
```

`plan-next-experiment` still requires four explicit paths. The FNO summary
already contains three of them, so Hermes only needs a bridge that accepts:

- `summary_path`
- `hypothesis_path`
- `output_path`

## Design

Add:

```text
plan-next-experiment-from-fno-summary
```

The command must:

- read `fno_rollout_summary.json`,
- require `state_hint == FNO_ROLLOUT_DIAGNOSED`,
- extract score, rollout-error, and Burgers-physics artifact paths,
- read the selected PDE hypothesis from `hypothesis_path`,
- delegate to `plan_next_experiment_from_files`,
- write the same `next_experiment_plan.json` schema as `plan-next-experiment`,
- return `state_hint = NEXT_EXPERIMENT_PLANNED`.

## Non-Goals

- No new intervention policies.
- No duplicated experiment planning logic.
- No training execution.
- No generated experiment config in this step.

## CLI Contract

```bash
uv run --frozen pde-agent plan-next-experiment-from-fno-summary \
  --summary-path runs/fno_rollout_diagnosis/fno_rollout_summary.json \
  --hypothesis-path runs/fno_rollout_diagnosis/selected_pde_hypothesis.json \
  --output-path runs/fno_rollout_diagnosis/next_experiment_plan.json
```

Expected payload:

```json
{
  "status": "ok",
  "command": "plan-next-experiment-from-fno-summary",
  "state_hint": "NEXT_EXPERIMENT_PLANNED",
  "artifacts": {
    "next_experiment_plan_path": "runs/fno_rollout_diagnosis/next_experiment_plan.json"
  },
  "metrics": {
    "selected_intervention": "use_scheduled_sampling_rollout_finetune",
    "change_scope": "training_schedule_only",
    "max_experiment_count": 2,
    "next_command": "generate-experiment-config"
  },
  "next_hint": "generate-experiment-config"
}
```

## Acceptance

- Unit tests prove the summary bridge delegates to the existing planner.
- Unit tests prove wrong summary state hint is rejected before writing output.
- CLI test proves the command writes the plan artifact and returns the expected
  command/state hints.
- Existing full tests still pass.
