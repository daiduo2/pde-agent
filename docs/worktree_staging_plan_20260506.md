# Worktree Staging Plan 2026-05-06

## Goal

Prepare the repository to continue the FNO early/middle training mainline
without losing the previous Hermes and Hypothesis Lab work.

## Current State

- `runs/` is clean and only contains `agent_runtime_migration_20260506/`.
- Historical runs are archived at
  `/Users/daiduo2/pde-agent-archive/runs_20260506/`.
- Cache directories were removed.
- `data/` remains in place and is about 13G.
- The worktree still contains a large set of source, docs, and test changes
  from multiple related development threads.

## Recommended Commit Groups

### 1. Runtime Migration Foundation

Purpose: preserve the Claude Agent SDK-first runtime switch as the new agent
execution base.

Files:

- `pyproject.toml`
- `uv.lock`
- `src/pde_agent/core/agent_runtime.py`
- `src/pde_agent/commands/agent_runtime.py`
- `src/pde_agent/core/hermes_availability_audit.py`
- `src/pde_agent/core/scientific_tool_router.py`
- `src/pde_agent/commands/hypothesis_lab_runtime.py`
- `src/pde_agent/cli.py`
- `tests/test_agent_runtime.py`
- `tests/test_hermes_availability_audit.py`
- `tests/test_hypothesis_lab_runtime.py`
- `docs/repo_cleanup_20260506.md`
- `docs/worktree_inventory_20260506.md`
- `docs/worktree_staging_plan_20260506.md`

Verification:

```bash
uv run --frozen pytest -q tests/test_agent_runtime.py tests/test_hermes_availability_audit.py tests/test_hypothesis_lab_runtime.py::test_scientific_tool_catalog_exposes_risk_and_state_hints
uv run --frozen --extra agent pde-agent audit-agent-runtime-availability --output-path /tmp/pde_agent_runtime_audit.json --runtime-name claude_agent_sdk --tool-catalog-path runs/agent_runtime_migration_20260506/tool_catalog.json --canary-report-path runs/agent_runtime_migration_20260506/fno_training_smoke_task/claude_agent_sdk_session.json --canary-report-path runs/agent_runtime_migration_20260506/promotion_smoke_task/claude_agent_sdk_session.json --cli-command build-fno-training-harness-contract --cli-command run-fno-training-optimization-loop --cli-command record-fno-training-optimization-evidence --cli-command build-hypothesis-algorithm-design --cli-command validate-hypothesis-algorithm-design --cli-command record-promoted-algorithm-strategy-proposal --cli-command convert-candidate-promotion-to-strategy-evidence --cli-command audit-agent-runtime-availability --required-command build-fno-training-harness-contract --required-command run-fno-training-optimization-loop --required-command record-fno-training-optimization-evidence --required-command build-hypothesis-algorithm-design --required-command validate-hypothesis-algorithm-design --required-command record-promoted-algorithm-strategy-proposal --required-command convert-candidate-promotion-to-strategy-evidence --required-command audit-agent-runtime-availability --force
```

### 2. FNO Training And Hypothesis Lab Mainline

Purpose: preserve the trainable dynamics source work needed for the next FNO
early/middle sprint.

Files:

- `src/pde_agent/core/task1_short_window_training.py`
- `src/pde_agent/core/task1_short_window_corrector.py`
- `src/pde_agent/core/task1_candidate_factory.py`
- `src/pde_agent/core/hypothesis_algorithm_designer.py`
- `src/pde_agent/core/hypothesis_algorithm_designer_stability.py`
- `src/pde_agent/core/hypothesis_candidate_planner.py`
- `src/pde_agent/core/hypothesis_doom_loop.py`
- `src/pde_agent/core/hypothesis_kpi.py`
- `src/pde_agent/core/hypothesis_lab_trace.py`
- matching command wrappers and tests.

Verification:

```bash
uv run --frozen pytest -q tests/test_task1_short_window_training.py tests/test_hypothesis_algorithm_designer.py tests/test_hypothesis_algorithm_designer_stability.py tests/test_hypothesis_candidate_planner.py tests/test_task1_candidate_factory.py
```

### 3. Hermes Compatibility And Memory Layer

Purpose: keep Hermes available as the historical backend and comparison loop
while Claude SDK becomes primary.

Files:

- `src/pde_agent/core/hermes_*.py`
- `src/pde_agent/commands/hermes_*.py`
- `tests/test_hermes_*.py`
- `docs/hermes_hypothesis_lab_runbook.md`
- `docs/hermes_integration_discipline.md`

Verification:

```bash
uv run --frozen pytest -q tests/test_hermes_heartbeat.py tests/test_hermes_literature_expansion.py tests/test_hermes_batch_optimization_loop.py tests/test_hermes_iteration_memory.py
```

### 4. Competition Timing And Submission Support

Purpose: retain timing-factor, materialization, and Task2 source-policy work
without mixing it into the FNO training sprint.

Files:

- `src/pde_agent/core/competition_timing.py`
- `src/pde_agent/core/competition_submission.py`
- `src/pde_agent/core/competition_scoring.py`
- `src/pde_agent/core/competition_task_pipeline.py`
- `src/pde_agent/core/task2_source_policy.py`
- matching command wrappers and tests.

Verification:

```bash
uv run --frozen pytest -q tests/test_competition_timing.py tests/test_competition_scoring.py tests/test_task2_source_policy.py
```

## Do Not Delete

- `data/`
- `runs/agent_runtime_migration_20260506/`
- `/Users/daiduo2/pde-agent-archive/runs_20260506/`
- untracked source and test files listed above

## Safe To Delete Later

- `__pycache__/`
- `.pytest_cache/`
- generated temporary audit files under `/tmp`
