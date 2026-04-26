# FNO Evidence Sweep V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible command that sweeps FNO finetune evidence across start windows and train-step budgets, then writes a recommendation.

**Architecture:** Create a core orchestration module that reuses existing core helpers instead of shelling out: local window suite, FNO dry-run, experiment audit, candidate prep, candidate execution, candidate evaluation, and candidate comparison. Add a thin CLI wrapper returning a standard JSON payload.

**Tech Stack:** Python, PyTorch for FNO execution, h5py/NumPy for HDF5 windows, Typer CLI, pytest, JSON/CSV artifacts.

---

### Task 1: Core Evidence Sweep

**Files:**
- Create: `src/pde_agent/core/fno_evidence_sweep.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_fno_evidence_sweep.py`

- [ ] **Step 1: Write failing tests**

Create a small HDF5 fixture and official-like FNO checkpoint. Run one-cell sweep with `start_indices=[0]`, `train_steps=[1]`, `forecast_steps=3`, and assert:

- `state_hint == "FNO_EVIDENCE_SWEEP_COMPLETED"`
- `evidence_summary.json` and `evidence_summary.csv` exist
- there are two candidate records
- there is one comparison winner
- recommendation includes `add_more_training_strategies`

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra torch --extra data pytest tests/test_fno_evidence_sweep.py -q
```

Expected: failure because `pde_agent.core.fno_evidence_sweep` does not exist.

- [ ] **Step 3: Implement core**

Implement:

```python
run_fno_evidence_sweep(
    *,
    source_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    start_indices: list[int],
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: list[int],
    learning_rate: float,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]
```

The function must call existing core functions directly:

- `build_local_window_suite`
- `run_fno_iteration_dryrun`
- `audit_experiment_config_from_file`
- `prepare_candidate_execution_from_audit`
- `execute_fno_finetune_candidate`
- `evaluate_fno_candidate`
- `compare_fno_candidates`

Recommendation rule:

- if one strategy wins all completed comparison cells, set `add_more_training_strategies = False`
- otherwise set `add_more_training_strategies = True`

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra torch --extra data pytest tests/test_fno_evidence_sweep.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/fno_evidence_sweep.py src/pde_agent/core/state_hints.py tests/test_fno_evidence_sweep.py
git commit -m "feat: run fno evidence sweep core"
```

### Task 2: CLI

**Files:**
- Create: `src/pde_agent/commands/run_fno_evidence_sweep.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_fno_evidence_sweep.py`

- [ ] **Step 1: Write failing CLI test**

Invoke `run-fno-evidence-sweep` with repeatable `--start-index` and `--train-steps`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra torch --extra data pytest tests/test_fno_evidence_sweep.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement CLI wrapper**

Return `success_payload` with summary path, CSV path, record count, winner count,
best score, and recommendation flag.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra torch --extra data pytest tests/test_fno_evidence_sweep.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/run_fno_evidence_sweep.py src/pde_agent/cli.py tests/test_fno_evidence_sweep.py
git commit -m "feat: expose fno evidence sweep cli"
```

### Task 3: Docs and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document command**

Add a concise usage block after candidate comparison docs.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen --extra torch --extra data pytest tests/test_fno_evidence_sweep.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen --extra torch pytest tests/test_fno_evidence_sweep.py tests/test_fno_finetune_candidate.py -q
uv run --frozen pde-agent run-fno-evidence-sweep --help
git diff --check
```

If public Task-1 HDF5 and official FNO checkpoint are present, run a one-window
real smoke with `start_indices=[0]` and `train_steps=[1,3]`.

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-fno-evidence-sweep-v0-design.md docs/superpowers/plans/2026-04-26-fno-evidence-sweep-v0.md
git commit -m "docs: plan fno evidence sweep"
```
