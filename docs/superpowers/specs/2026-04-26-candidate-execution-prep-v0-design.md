# Candidate Execution Prep V0 Design

## Goal

Add the first executor-facing step after `audit-experiment-config` without
running training. The command selects one approved candidate experiment, creates
a candidate workspace, and writes a machine-readable execution plan.

This is the final boundary check before implementing actual FNO finetuning.

## Command

```bash
uv run --frozen pde-agent prepare-candidate-execution \
  --audit-path runs/fno_iteration_001/experiment_config_audit.json \
  --candidate-index 0 \
  --output-dir runs/fno_iteration_001/candidates \
  --force
```

## Inputs

- `audit-path`: JSON produced by `audit-experiment-config`
- `candidate-index`: zero-based index into `config.candidate_experiments`
- `output-dir`: root under which the selected candidate workspace is created
- `force`: allow replacing an existing candidate execution plan

The command reads `config_path` from the audit report. It does not accept an
unaudited config directly.

## Output

For candidate `use_scheduled_sampling_rollout_finetune_001`, the command writes:

```text
<output-dir>/use_scheduled_sampling_rollout_finetune_001/
├── artifacts/
├── logs/
└── candidate_execution_plan.json
```

The plan contains:

- `state_hint = CANDIDATE_EXECUTION_PREPARED`
- selected candidate fields copied from the config
- audit/config paths
- workspace paths
- `execution_status = not_started`
- `next_command = execute-candidate-experiment`

## Guardrails

- Reject audit reports whose decision is not `approved_for_execution`.
- Reject audit reports whose state hint is not `EXPERIMENT_CONFIG_AUDITED`.
- Reject configs whose state hint is not `EXPERIMENT_CONFIG_GENERATED`.
- Reject out-of-range candidate indexes.
- Do not execute training, inference, or scoring.

## Acceptance

- Valid approved audit creates workspace directories and candidate execution
  plan.
- Rejected audit is refused before writing a workspace.
- CLI returns `CANDIDATE_EXECUTION_PREPARED`.
- Existing full tests still pass.
