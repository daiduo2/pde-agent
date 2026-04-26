# FNO Experiment Plan Bridge V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Hermes call one FNO-specific planning command after selecting a hypothesis from the FNO summary.

**Architecture:** Add a summary-to-planner helper in `next_experiment_plan.py`, then expose it through a thin Typer command. The helper validates the FNO summary and delegates to the existing `plan_next_experiment_from_files` implementation.

**Tech Stack:** Python, Typer CLI, pytest, JSON artifact contracts.

---

### Task 1: Core Planning Bridge

**Files:**
- Modify: `src/pde_agent/core/next_experiment_plan.py`
- Test: `tests/test_fno_experiment_plan_bridge.py`

- [ ] **Step 1: Write failing tests**

Add tests for valid FNO summary delegation and invalid state-hint rejection.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_fno_experiment_plan_bridge.py -q
```

Expected: tests fail because `plan_next_experiment_from_fno_summary` does not exist.

- [ ] **Step 3: Implement minimal helper**

Add `plan_next_experiment_from_fno_summary(summary_path, hypothesis_path, output_path)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_fno_experiment_plan_bridge.py -q
```

Expected: all bridge tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/next_experiment_plan.py tests/test_fno_experiment_plan_bridge.py
git commit -m "feat: plan experiment from fno summary"
```

### Task 2: CLI Bridge

**Files:**
- Create: `src/pde_agent/commands/plan_next_experiment_from_fno_summary.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_fno_experiment_plan_bridge.py`

- [ ] **Step 1: Write failing CLI test**

Invoke `plan-next-experiment-from-fno-summary` through Typer.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_fno_experiment_plan_bridge.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement command wrapper**

Return a standard `success_payload` with `state_hint = NEXT_EXPERIMENT_PLANNED`
and `next_hint = generate-experiment-config`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_fno_experiment_plan_bridge.py -q
```

Expected: all bridge tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/plan_next_experiment_from_fno_summary.py src/pde_agent/cli.py tests/test_fno_experiment_plan_bridge.py
git commit -m "feat: expose fno summary planning cli"
```

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document command usage**

Add the command after `select-pde-hypothesis-from-fno-summary`.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen pytest tests/test_fno_experiment_plan_bridge.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent plan-next-experiment-from-fno-summary --help
git diff --check
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-fno-experiment-plan-bridge-v0-design.md docs/superpowers/plans/2026-04-26-fno-experiment-plan-bridge-v0.md
git commit -m "docs: document fno planning bridge"
```
