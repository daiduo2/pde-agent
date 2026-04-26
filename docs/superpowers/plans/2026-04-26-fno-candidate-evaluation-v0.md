# FNO Candidate Evaluation V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-command evaluation loop for finetuned FNO candidate predictions.

**Architecture:** Create a focused orchestration module that reads `candidate_execution_report.json`, reuses existing score/error/physics/hypothesis helpers, writes a candidate evaluation bundle, and optionally compares against a baseline score. Add CLI, state hint, tests, and README docs.

**Tech Stack:** Python, NumPy, Typer CLI, pytest, JSON artifact contracts.

---

### Task 1: Core Evaluation

**Files:**
- Create: `src/pde_agent/core/fno_candidate_evaluation.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_fno_candidate_evaluation.py`

- [ ] **Step 1: Write failing tests**

Add tests for evaluation artifact bundle and baseline score delta.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_evaluation.py -q
```

Expected: failure because `evaluate_fno_candidate` does not exist.

- [ ] **Step 3: Implement core**

Implement `evaluate_fno_candidate(candidate_report_path, target_path, output_dir,
baseline_score_path=None, nu=0.001, dt=1.0, dx=1.0, force=False)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_evaluation.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/fno_candidate_evaluation.py src/pde_agent/core/state_hints.py tests/test_fno_candidate_evaluation.py
git commit -m "feat: evaluate fno candidate core"
```

### Task 2: CLI

**Files:**
- Create: `src/pde_agent/commands/evaluate_fno_candidate.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_fno_candidate_evaluation.py`

- [ ] **Step 1: Write failing CLI test**

Invoke `evaluate-fno-candidate`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_evaluation.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement wrapper**

Return `success_payload` with `state_hint = FNO_CANDIDATE_EVALUATED`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_evaluation.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/evaluate_fno_candidate.py src/pde_agent/cli.py tests/test_fno_candidate_evaluation.py
git commit -m "feat: expose fno candidate evaluation cli"
```

### Task 3: Docs and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document command**

Add usage after `execute-fno-finetune-candidate`.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_evaluation.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent evaluate-fno-candidate --help
git diff --check
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-fno-candidate-evaluation-v0-design.md docs/superpowers/plans/2026-04-26-fno-candidate-evaluation-v0.md
git commit -m "docs: document fno candidate evaluation"
```
