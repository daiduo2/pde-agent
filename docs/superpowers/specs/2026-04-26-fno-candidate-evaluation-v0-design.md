# FNO Candidate Evaluation V0 Design

## Goal

Evaluate a finetuned FNO candidate prediction with the same local diagnostic
chain used for the baseline:

```text
candidate_prediction.npz
  -> score-local-rollout
  -> diagnose-rollout-error
  -> check-burgers-physics-local
  -> select-pde-hypothesis
  -> candidate_evaluation_report.json
```

This closes the candidate feedback loop after `execute-fno-finetune-candidate`.

## Command

```bash
uv run --frozen pde-agent evaluate-fno-candidate \
  --candidate-report-path runs/fno_iteration_001/candidates/use_scheduled_sampling_rollout_finetune_001/artifacts/candidate_execution_report.json \
  --target-path data/derived/local_windows_task1_smoke.npz \
  --output-dir runs/fno_iteration_001/candidates/use_scheduled_sampling_rollout_finetune_001/artifacts/evaluation \
  --baseline-score-path runs/fno_iteration_001/fno_score.json \
  --nu 0.001 \
  --dt 1.0 \
  --dx 1.0 \
  --force
```

## Artifacts

The output directory contains:

- `candidate_score.json`
- `candidate_rollout_error.json`
- `candidate_burgers_physics.json`
- `candidate_selected_pde_hypothesis.json`
- `candidate_evaluation_report.json`

## Report Contract

`candidate_evaluation_report.json` includes:

- `state_hint = FNO_CANDIDATE_EVALUATED`
- candidate id and intervention
- score metrics
- error diagnosis
- physics diagnosis
- selected follow-up hypothesis
- optional baseline comparison if `baseline-score-path` is provided
- `next_command = compare-fno-candidates`

## Guardrails

- Reject candidate reports whose state hint is not
  `FNO_FINETUNE_CANDIDATE_EXECUTED`.
- Reject missing prediction artifacts.
- Protect output artifacts unless `--force`.
- Do not train or mutate model outputs.

## Acceptance

- Fixture test proves a candidate report and prediction produce all evaluation
  artifacts.
- Baseline comparison reports score delta when baseline score is provided.
- CLI returns `FNO_CANDIDATE_EVALUATED`.
- Existing tests still pass.
