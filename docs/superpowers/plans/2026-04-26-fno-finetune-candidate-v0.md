# FNO Finetune Candidate V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal executor for prepared FNO scheduled-sampling finetune candidates.

**Architecture:** Create a focused torch-dependent core module that consumes `candidate_execution_plan.json`, explicit windows and checkpoint paths, and writes finetuned checkpoint, prediction NPZ, and execution report. Add a thin CLI wrapper, state hint, tests, and README docs.

**Tech Stack:** Python, PyTorch optional extra, NumPy, Typer CLI, pytest.

---

### Task 1: Core Executor

**Files:**
- Create: `src/pde_agent/core/fno_finetune_candidate.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_fno_finetune_candidate.py`

- [ ] **Step 1: Write failing tests**

Add tests for one-step finetune output artifacts and unsupported intervention rejection.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py -q
```

Expected: failure because `execute_fno_finetune_candidate` does not exist.

- [ ] **Step 3: Implement core executor**

Implement `execute_fno_finetune_candidate(candidate_plan_path, windows_path,
checkpoint_path, max_train_steps=1, learning_rate=1e-4, force=False)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/fno_finetune_candidate.py src/pde_agent/core/state_hints.py tests/test_fno_finetune_candidate.py
git commit -m "feat: execute fno finetune candidate core"
```

### Task 2: CLI

**Files:**
- Create: `src/pde_agent/commands/execute_fno_finetune_candidate.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_fno_finetune_candidate.py`

- [ ] **Step 1: Write failing CLI test**

Invoke `execute-fno-finetune-candidate`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement wrapper**

Return `success_payload` with `state_hint = FNO_FINETUNE_CANDIDATE_EXECUTED`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/execute_fno_finetune_candidate.py src/pde_agent/cli.py tests/test_fno_finetune_candidate.py
git commit -m "feat: expose fno finetune candidate cli"
```

### Task 3: Docs and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document command**

Add usage after `prepare-candidate-execution`.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py tests/test_fno_backend_v0.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent execute-fno-finetune-candidate --help
git diff --check
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-fno-finetune-candidate-v0-design.md docs/superpowers/plans/2026-04-26-fno-finetune-candidate-v0.md
git commit -m "docs: document fno finetune candidate"
```
