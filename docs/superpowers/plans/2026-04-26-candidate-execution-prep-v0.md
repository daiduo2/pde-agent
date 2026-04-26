# Candidate Execution Prep V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare an audited candidate experiment workspace without executing it.

**Architecture:** Create one core module that reads an audit report, loads the audited experiment config, selects a candidate, creates workspace directories, and writes `candidate_execution_plan.json`. Add a CLI wrapper, state hint, tests, and README docs.

**Tech Stack:** Python, Typer CLI, pytest, JSON artifact contracts.

---

### Task 1: Core Preparation

**Files:**
- Create: `src/pde_agent/core/candidate_execution_prep.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_candidate_execution_prep.py`

- [ ] **Step 1: Write failing tests**

Add tests for successful workspace creation and rejected audit refusal.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_candidate_execution_prep.py -q
```

Expected: failure because `prepare_candidate_execution_from_audit` does not exist.

- [ ] **Step 3: Implement core**

Implement `prepare_candidate_execution_from_audit(audit_path, output_dir,
candidate_index=0, force=False)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_candidate_execution_prep.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/candidate_execution_prep.py src/pde_agent/core/state_hints.py tests/test_candidate_execution_prep.py
git commit -m "feat: prepare candidate execution core"
```

### Task 2: CLI

**Files:**
- Create: `src/pde_agent/commands/prepare_candidate_execution.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_candidate_execution_prep.py`

- [ ] **Step 1: Write failing CLI test**

Invoke `prepare-candidate-execution`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_candidate_execution_prep.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement wrapper**

Return `success_payload` with `state_hint = CANDIDATE_EXECUTION_PREPARED`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_candidate_execution_prep.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/prepare_candidate_execution.py src/pde_agent/cli.py tests/test_candidate_execution_prep.py
git commit -m "feat: expose candidate execution prep cli"
```

### Task 3: Docs and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document usage**

Add the command after `audit-experiment-config`.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen pytest tests/test_candidate_execution_prep.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent prepare-candidate-execution --help
git diff --check
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-candidate-execution-prep-v0-design.md docs/superpowers/plans/2026-04-26-candidate-execution-prep-v0.md
git commit -m "docs: document candidate execution prep"
```
