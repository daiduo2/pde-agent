# Experiment Config Audit V0 Design

## Goal

Add a deterministic audit step between `generate-experiment-config` and the
future `execute-candidate-experiment` hook.

The audit checks whether `experiment_config.json` is consistent with its source
`next_experiment_plan.json`, within budget, and safe to hand to an executor.

## Command

```bash
uv run --frozen pde-agent audit-experiment-config \
  --config-path runs/fno_iteration_001/experiment_config.json \
  --output-path runs/fno_iteration_001/experiment_config_audit.json
```

`--plan-path` is optional. If omitted, the command uses `source_plan_path` from
the experiment config.

## Audit Decision

The audit writes a report instead of failing the process for semantic issues:

- `decision = approved_for_execution` when no issues are found.
- `decision = rejected` when one or more consistency or boundary issues are
  found.

Malformed inputs that prevent reading the config or plan still raise an
exception.

## Checks

The audit checks:

- config `state_hint == EXPERIMENT_CONFIG_GENERATED`
- plan `state_hint == NEXT_EXPERIMENT_PLANNED`
- config `source_plan_path` matches the plan path being audited
- selected intervention, change scope, requires-training flag, and budget match
  the plan
- candidate count is at least one and no greater than plan budget
- each candidate:
  - uses the selected intervention
  - uses the expected change scope
  - has the expected requires-training flag
  - has parameters drawn from the plan's candidate parameter grid
  - has expected artifacts matching `required_next_artifacts`
  - has forbidden actions matching the plan's forbidden actions
- config `next_command == execute-candidate-experiment`

## Output Contract

```json
{
  "schema_version": 1,
  "state_hint": "EXPERIMENT_CONFIG_AUDITED",
  "decision": "approved_for_execution",
  "issue_count": 0,
  "issues": [],
  "config_path": ".../experiment_config.json",
  "plan_path": ".../next_experiment_plan.json",
  "summary": {
    "selected_intervention": "use_scheduled_sampling_rollout_finetune",
    "candidate_count": 2,
    "requires_training": true,
    "next_command": "execute-candidate-experiment"
  }
}
```

## Non-Goals

- Do not execute candidates.
- Do not score candidate results.
- Do not mutate config or plan artifacts.
- Do not infer missing values.

## Acceptance

- Valid generated config audits to `approved_for_execution`.
- Tampered candidate parameters audit to `rejected` with an issue.
- CLI writes an audit artifact and returns `EXPERIMENT_CONFIG_AUDITED`.
- Existing full tests still pass.
