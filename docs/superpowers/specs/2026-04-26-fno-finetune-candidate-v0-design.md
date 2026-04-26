# FNO Finetune Candidate V0 Design

## Goal

Execute the first real training step in the harness: a minimal CPU FNO
scheduled-sampling finetune for an audited and prepared candidate.

This V0 is deliberately narrow. It supports only:

```text
candidate.intervention == use_scheduled_sampling_rollout_finetune
```

It does not implement the generic `execute-candidate-experiment` hook.

## Command

```bash
uv run --frozen --extra torch pde-agent execute-fno-finetune-candidate \
  --candidate-plan-path runs/fno_iteration_001/candidates/use_scheduled_sampling_rollout_finetune_001/candidate_execution_plan.json \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --max-train-steps 1 \
  --learning-rate 1e-4 \
  --force
```

`windows-path` and `checkpoint-path` are explicit executor inputs. They are not
inferred from the candidate plan because data and weight provenance are
high-risk execution inputs.

## Outputs

Under the candidate workspace `artifacts/` directory:

- `finetuned_fno_checkpoint.pt`
- `candidate_prediction.npz`
- `candidate_execution_report.json`

The report includes:

- `state_hint = FNO_FINETUNE_CANDIDATE_EXECUTED`
- candidate id and intervention
- train step count
- loss trace
- output artifact paths
- `next_command = score-local-rollout`

## Training Semantics

V0 uses the existing PDEBench FNO1d reconstruction code:

- load checkpoint,
- infer official FNO1d config,
- reconstruct model,
- load state dict,
- run a small scheduled-sampling rollout loss on local windows,
- save the finetuned state dict,
- write a prediction NPZ using the finetuned model.

Candidate parameters:

- `rollout_steps`: capped by the available local target horizon.
- `teacher_forcing_decay`: used as the deterministic teacher-forcing blend
  ratio for the next rollout history state.

## Guardrails

- Reject unprepared candidate plans.
- Reject unsupported interventions.
- Reject existing output artifacts unless `--force`.
- Reject unsupported checkpoint schemas.
- Do not claim score improvement.
- Do not submit results.

## Acceptance

- Unit test proves a tiny official-like FNO checkpoint can be finetuned for one
  step and emits checkpoint, prediction, and report.
- Unit test proves unsupported intervention is rejected.
- CLI test proves command returns `FNO_FINETUNE_CANDIDATE_EXECUTED`.
- Existing non-torch and torch tests still pass.
