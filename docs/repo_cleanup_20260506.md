# Repository Cleanup 2026-05-06

## Summary

The historical `runs/` experiment artifacts were moved out of the repository
workspace to keep the active worktree small while preserving reproducibility.

## Active In-Repo Artifacts

- `runs/agent_runtime_migration_20260506/`

This directory remains in the project because it contains the current Claude
Agent SDK runtime migration evidence:

- runtime contract
- FNO training smoke session
- promotion smoke session
- combined runtime availability audit

## Archived Artifacts

Archived root:

- `/Users/daiduo2/pde-agent-archive/runs_20260506/`

This directory contains the older Hermes, Hypothesis Lab, Task1, Task2,
literature, and operator-tuning run artifacts that previously lived under
`runs/`.

## Restore

To restore one archived run:

```bash
mkdir -p runs
mv /Users/daiduo2/pde-agent-archive/runs_20260506/<run_name> runs/
```

To inspect archive size:

```bash
du -sh /Users/daiduo2/pde-agent-archive/runs_20260506
```

## Cleanup Policy

- Do not delete archived runs without an explicit review.
- Keep current runtime migration evidence in `runs/`.
- Cache directories such as `__pycache__` and `.pytest_cache` can be deleted
  without review.
