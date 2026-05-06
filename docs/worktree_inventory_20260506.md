# Worktree Inventory 2026-05-06

## Cleanup State

- Historical run artifacts were moved to
  `/Users/daiduo2/pde-agent-archive/runs_20260506/`.
- Active in-repo run artifacts are limited to
  `runs/agent_runtime_migration_20260506/`.
- Python and pytest cache directories were removed.

## Keep For Current Mainline

These files support the current Claude Agent SDK-first runtime migration and
should remain in the active worktree:

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

## Keep For Hypothesis Lab And Training Mainline

These files are part of the current Hypothesis Lab, FNO training loop, evidence,
candidate planning, and Task1 capability work. They should be reviewed as a
coherent follow-up change set rather than deleted:

- `src/pde_agent/core/hypothesis_algorithm_designer.py`
- `src/pde_agent/core/hypothesis_algorithm_designer_stability.py`
- `src/pde_agent/core/hypothesis_candidate_planner.py`
- `src/pde_agent/core/hypothesis_doom_loop.py`
- `src/pde_agent/core/hypothesis_kpi.py`
- `src/pde_agent/core/hypothesis_lab_trace.py`
- `src/pde_agent/core/task1_short_window_training.py`
- `src/pde_agent/core/task1_candidate_factory.py`
- `src/pde_agent/commands/task1_candidate_factory.py`
- `tests/test_hypothesis_algorithm_designer.py`
- `tests/test_hypothesis_algorithm_designer_stability.py`
- `tests/test_hypothesis_candidate_planner.py`
- `tests/test_task1_short_window_training.py`
- `tests/test_task1_candidate_factory.py`

## Keep As Hermes Compatibility Or Historical Backend

These files preserve Hermes as a compatibility backend and keep the previous
agent loop available for comparison. They should not be deleted until Claude SDK
has fully replaced every operational path:

- `src/pde_agent/core/hermes_runtime_closed_loop.py`
- `src/pde_agent/core/hermes_runtime_audit_chain.py`
- `src/pde_agent/core/hermes_runtime_closed_loop_audit.py`
- `src/pde_agent/core/hermes_runtime_memory_bridge_smoke.py`
- `src/pde_agent/core/hermes_research_directive.py`
- `src/pde_agent/core/hermes_meta_research_controller.py`
- `src/pde_agent/core/hermes_exploration_scheduler.py`
- `src/pde_agent/core/hermes_repair_policy_summary.py`
- `src/pde_agent/core/hermes_task1_agent_capability_benchmark.py`
- related `src/pde_agent/commands/hermes_*.py`
- related `tests/test_hermes_*.py`

## Candidate For Later Archival Review

These are useful but less central to the current runtime migration. Review
before keeping in the main branch:

- `integration_tests/hermes_hypothesis_lab_infrastructure/`
- `scripts/audit_hermes_hypothesis_lab_infrastructure.py`
- `scripts/run_hermes_hypothesis_lab_infrastructure.py`
- `scripts/run_task1_scalar_source_gate.py`
- `docs/hermes_hypothesis_lab_runbook.md`
- `docs/superpowers/plans/2026-05-04-hermes-research-directive.md`
- `docs/superpowers/plans/2026-05-04-task1-candidate-factory.md`
- `docs/superpowers/specs/2026-05-04-task1-agent-capability-benchmark-design.md`
- `docs/superpowers/specs/2026-05-04-task1-candidate-factory-design.md`

## Large Directories

- `data/` remains in place and is about 13G. Do not clean it without a separate
  dataset policy review.
- `external/` is small, about 8.6M, and currently includes local research
  references such as `ml-intern`.
- `runs/` is now about 2.5M and contains only current runtime migration
  artifacts.

## Verification Commands Used

```bash
du -sh runs data external integration_tests docs scripts src tests
find . -maxdepth 4 -type d -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache -o -name .ipynb_checkpoints
git status --short
```
