# FNO Hypothesis Bridge V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Hermes select a PDE hypothesis directly from `fno_rollout_summary.json`.

**Architecture:** Add a small core helper that validates the FNO summary and delegates to the existing PDE hypothesis selector. Add a thin command wrapper, CLI registration, focused tests, and README usage docs. Do not duplicate scientific decision logic.

**Tech Stack:** Python, Typer CLI, pytest, JSON artifact contracts.

---

### Task 1: Core Bridge

**Files:**
- Modify: `src/pde_agent/core/pde_hypothesis.py`
- Test: `tests/test_fno_hypothesis_bridge.py`

- [ ] **Step 1: Write failing tests**

Add tests for valid FNO summary delegation and invalid state-hint rejection.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_fno_hypothesis_bridge.py -q
```

Expected: tests fail because `select_pde_hypothesis_from_fno_summary` does not exist.

- [ ] **Step 3: Implement minimal core helper**

Add `select_pde_hypothesis_from_fno_summary(summary_path, output_path)` to `pde_hypothesis.py`. It must load summary JSON, validate `FNO_ROLLOUT_DIAGNOSED`, extract the three artifact paths, and call `select_pde_hypothesis_from_files`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_fno_hypothesis_bridge.py -q
```

Expected: all bridge tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/pde_hypothesis.py tests/test_fno_hypothesis_bridge.py
git commit -m "feat: select hypothesis from fno summary"
```

### Task 2: CLI Bridge

**Files:**
- Create: `src/pde_agent/commands/select_pde_hypothesis_from_fno_summary.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_fno_hypothesis_bridge.py`

- [ ] **Step 1: Write failing CLI test**

Add a Typer CLI test invoking `select-pde-hypothesis-from-fno-summary`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_fno_hypothesis_bridge.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement command wrapper and registration**

Create a command wrapper returning `success_payload` with `command =
select-pde-hypothesis-from-fno-summary`, `state_hint =
PDE_HYPOTHESIS_SELECTED`, and `next_hint = plan-next-experiment`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_fno_hypothesis_bridge.py -q
```

Expected: all bridge tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/select_pde_hypothesis_from_fno_summary.py src/pde_agent/cli.py tests/test_fno_hypothesis_bridge.py
git commit -m "feat: expose fno summary hypothesis cli"
```

### Task 3: Docs and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document usage**

Add a short command example after the FNO rollout diagnosis section.

- [ ] **Step 2: Run verification**

Run:

```bash
uv run --frozen pytest tests/test_fno_hypothesis_bridge.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent select-pde-hypothesis-from-fno-summary --help
git diff --check
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-04-26-fno-hypothesis-bridge-v0-design.md docs/superpowers/plans/2026-04-26-fno-hypothesis-bridge-v0.md
git commit -m "docs: document fno hypothesis bridge"
```
