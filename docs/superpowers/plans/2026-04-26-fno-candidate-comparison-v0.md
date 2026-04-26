# FNO Candidate Comparison V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic comparison command for evaluated FNO candidates.

**Architecture:** Create a focused core comparator that validates one or more `candidate_evaluation_report.json` files, normalizes candidate summaries, sorts by score/error tie-breakers, writes a comparison report, and exposes it through a Typer CLI wrapper. The comparator is read-only with respect to candidate workspaces.

**Tech Stack:** Python, Typer CLI, pytest, JSON artifact contracts.

---

### Task 1: Core Comparator

**Files:**
- Create: `src/pde_agent/core/fno_candidate_comparison.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_fno_candidate_comparison.py`

- [ ] **Step 1: Write failing tests**

Create fixture reports with `state_hint = FNO_CANDIDATE_EVALUATED`, distinct candidate ids, score metrics, baseline comparisons, error diagnoses, physics diagnoses, and selected hypotheses. Test that higher score wins and that a single candidate still produces a winner.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_comparison.py -q
```

Expected: failure because `pde_agent.core.fno_candidate_comparison` does not exist.

- [ ] **Step 3: Implement core comparator**

Implement:

```python
compare_fno_candidates(
    *,
    evaluation_report_paths: list[str | Path],
    output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]
```

Validation rules:

- at least one evaluation report path
- file exists and contains JSON object
- `state_hint == "FNO_CANDIDATE_EVALUATED"`
- `candidate.experiment_id` is a unique string
- `score.competition_like_score_short`, `score.rel_mse`, and `score.rmse` are numeric

Sorting key:

```python
(-score, rel_mse, rmse, experiment_id)
```

Report state hint:

```python
"FNO_CANDIDATES_COMPARED"
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_comparison.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/fno_candidate_comparison.py src/pde_agent/core/state_hints.py tests/test_fno_candidate_comparison.py
git commit -m "feat: compare fno candidates core"
```

### Task 2: CLI

**Files:**
- Create: `src/pde_agent/commands/compare_fno_candidates.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_fno_candidate_comparison.py`

- [ ] **Step 1: Write failing CLI test**

Invoke:

```bash
pde-agent compare-fno-candidates \
  --evaluation-report-path candidate_a.json \
  --evaluation-report-path candidate_b.json \
  --output-path comparison.json
```

Assert `status = ok`, `state_hint = FNO_CANDIDATES_COMPARED`, winner id, candidate count, and `next_hint = prepare-next-fno-iteration`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_comparison.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement CLI wrapper**

Create `compare_fno_candidates_command(...)` and register `compare-fno-candidates` in `src/pde_agent/cli.py`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_comparison.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/compare_fno_candidates.py src/pde_agent/cli.py tests/test_fno_candidate_comparison.py
git commit -m "feat: expose fno candidate comparison cli"
```

### Task 3: Docs and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document command**

Add usage after `evaluate-fno-candidate`, explaining that the command consumes candidate evaluation reports and writes the next decision boundary.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen pytest tests/test_fno_candidate_comparison.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent compare-fno-candidates --help
git diff --check
```

Also run a real smoke comparison against the existing evaluated FNO candidate report if present under `/tmp/pde-agent-fno-iteration-dryrun-final-20260426`.

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-fno-candidate-comparison-v0-design.md docs/superpowers/plans/2026-04-26-fno-candidate-comparison-v0.md
git commit -m "docs: plan fno candidate comparison"
```
