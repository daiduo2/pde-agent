# FNO Iteration Dry-Run V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a one-command dry-run for the FNO diagnostic-to-config chain.

**Architecture:** Add a focused orchestration module that calls existing FNO diagnosis, hypothesis selection, experiment planning, and config generation helpers. Add a thin CLI wrapper, a new state hint, tests, README docs, and a manifest artifact.

**Tech Stack:** Python, Typer CLI, pytest, optional torch extra for FNO fixture tests.

---

### Task 1: Core Runner

**Files:**
- Create: `src/pde_agent/core/fno_iteration_dryrun.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_fno_iteration_dryrun.py`

- [ ] **Step 1: Write failing tests**

Create tests for artifact generation and existing artifact rejection.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_iteration_dryrun.py -q
```

Expected: failure because `run_fno_iteration_dryrun` does not exist.

- [ ] **Step 3: Implement core runner**

Implement `run_fno_iteration_dryrun(windows_path, checkpoint_path, output_dir,
nu, dt, dx, force)` and write `iteration_manifest.json`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_iteration_dryrun.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/fno_iteration_dryrun.py src/pde_agent/core/state_hints.py tests/test_fno_iteration_dryrun.py
git commit -m "feat: add fno iteration dryrun core"
```

### Task 2: CLI Wrapper

**Files:**
- Create: `src/pde_agent/commands/run_fno_iteration_dryrun.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_fno_iteration_dryrun.py`

- [ ] **Step 1: Write failing CLI test**

Invoke `run-fno-iteration-dryrun` through Typer.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_iteration_dryrun.py -q
```

Expected: CLI test fails because command is not registered.

- [ ] **Step 3: Implement wrapper**

Return `success_payload` with `state_hint = FNO_ITERATION_DRYRUN_COMPLETED` and
`next_hint = execute-candidate-experiment`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_iteration_dryrun.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/run_fno_iteration_dryrun.py src/pde_agent/cli.py tests/test_fno_iteration_dryrun.py
git commit -m "feat: expose fno iteration dryrun cli"
```

### Task 3: Docs and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document usage**

Add a short section after FNO planning bridge.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_iteration_dryrun.py tests/test_fno_rollout_diagnosis.py tests/test_fno_backend_v0.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent run-fno-iteration-dryrun --help
git diff --check
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-fno-iteration-dryrun-v0-design.md docs/superpowers/plans/2026-04-26-fno-iteration-dryrun-v0.md
git commit -m "docs: document fno iteration dryrun"
```
