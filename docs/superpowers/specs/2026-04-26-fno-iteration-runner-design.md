# FNO Iteration Runner Design

## Goal

Add a single-command FNO iteration runner that executes the current V1
agent loop on one local window file:

1. baseline dry-run and diagnostic planning,
2. experiment-config audit,
3. candidate workspace preparation,
4. candidate finetune execution,
5. candidate evaluation,
6. candidate comparison,
7. machine-readable iteration report.

This is the stable command Hermes should call during ordinary iterative
development after a window suite already exists.

## Scope

The runner operates on an existing `.npz` local-window file and an existing FNO
checkpoint. It does not build HDF5 windows, expand multi-window evidence, write
submission files, or introduce new training strategies.

The command should run all candidates emitted by the planner. With the current
V1 policy that means two candidates: `teacher_forcing_decay=0.85` and `0.9`.

## Command

```bash
uv run --frozen --extra torch pde-agent run-fno-iteration \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir runs/fno_iteration_001 \
  --max-train-steps 16 \
  --learning-rate 1e-4 \
  --force
```

`--max-train-steps` and `--learning-rate` use the same defaults as
`execute-fno-finetune-candidate`.

## Artifacts

The output directory contains:

- baseline dry-run artifacts from `run_fno_iteration_dryrun`,
- `experiment_config_audit.json`,
- `candidates/<experiment_id>/candidate_execution_plan.json`,
- candidate execution/evaluation artifacts under each candidate workspace,
- `candidate_comparison.json`,
- `fno_iteration_report.json`.

The iteration report has `state_hint = FNO_ITERATION_COMPLETED` and includes:

- input paths and numeric settings,
- baseline score and selected intervention,
- audit decision and issue count,
- one summary per candidate,
- comparison winner,
- next command hint.

## Error Handling

If `output_dir` already contains files and `--force` is not set, the runner
fails before writing anything new. Existing lower-level functions keep their
own validation and cleanup behavior.

If no candidates are generated or audit rejects the config, the runner raises a
specific error rather than silently continuing.

## Testing

Tests should use the synthetic tiny FNO checkpoint/window helpers already used
by `test_fno_finetune_candidate.py`. The first implementation test should
prove the runner writes an iteration report with two candidate summaries and a
comparison winner. A CLI test should prove command output exposes the report
path, winner experiment id, and candidate count.
