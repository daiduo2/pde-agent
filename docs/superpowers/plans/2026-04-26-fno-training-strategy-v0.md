# FNO Training Strategy V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded FNO finetune strategy switch so candidates can compare uniform scheduled sampling against late-weighted scheduled sampling.

**Architecture:** Extend the existing `execute_fno_finetune_candidate` path without changing artifact names. Candidate plans may now include `parameters.training_strategy`; missing values default to `scheduled_sampling`. The planner exposes both strategies for the scheduled-sampling intervention so multi-candidate comparison can evaluate them.

**Tech Stack:** Python, PyTorch, NumPy, pytest, JSON artifact contracts.

---

### Task 1: Executor Strategy Support

**Files:**
- Modify: `tests/test_fno_finetune_candidate.py`
- Modify: `src/pde_agent/core/fno_finetune_candidate.py`
- Modify: `src/pde_agent/commands/execute_fno_finetune_candidate.py`

- [ ] **Step 1: Write failing tests**

Add a candidate-plan fixture override for `training_strategy`. Test that `late_weighted_scheduled_sampling` executes, writes finite predictions, and records `train.training_strategy`. Test that an unknown strategy raises `FNOFinetuneCandidateError`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py -q
```

Expected: failure because reports do not include strategy metadata and unknown strategies are not validated.

- [ ] **Step 3: Implement strategy support**

In `src/pde_agent/core/fno_finetune_candidate.py`:

- add `_training_strategy(parameters)` returning `scheduled_sampling` by default
- allow `scheduled_sampling` and `late_weighted_scheduled_sampling`
- pass `training_strategy` into `_finetune_model`
- compute rollout loss weights as all ones for `scheduled_sampling`
- compute linearly increasing weights from `1.0` to `2.0` for `late_weighted_scheduled_sampling`
- normalize weights by their mean before applying them
- record `train.training_strategy` and `train.loss_weighting`

In `src/pde_agent/commands/execute_fno_finetune_candidate.py`, include strategy metadata in the payload metrics.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_fno_finetune_candidate.py src/pde_agent/core/fno_finetune_candidate.py src/pde_agent/commands/execute_fno_finetune_candidate.py
git commit -m "feat: add fno finetune training strategies"
```

### Task 2: Planner Policy Support

**Files:**
- Modify: `src/pde_agent/core/next_experiment_plan.py`
- Modify: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Write failing planner test**

Update `test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune` to assert that `training_strategy` equals:

```python
["scheduled_sampling", "late_weighted_scheduled_sampling"]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune -q
```

Expected: failure because the policy does not expose `training_strategy`.

- [ ] **Step 3: Update policy**

Add `training_strategy` to `_INTERVENTION_POLICIES["use_scheduled_sampling_rollout_finetune"]["candidate_parameters"]`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune -q
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_from_plan_writes_candidate_grid -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/next_experiment_plan.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: expose fno training strategy candidates"
```

### Task 3: Docs and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document strategy parameter**

Update the FNO finetune section to describe `training_strategy`, the default behavior, and the late-weighted option.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py -q
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen --extra torch pytest tests/test_fno_backend_v0.py tests/test_fno_finetune_candidate.py -q
git diff --check
```

If the official FNO smoke artifacts are present, run one `late_weighted_scheduled_sampling` candidate by editing a temporary candidate plan under `/tmp` and executing `execute-fno-finetune-candidate`.

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-fno-training-strategy-v0-design.md docs/superpowers/plans/2026-04-26-fno-training-strategy-v0.md
git commit -m "docs: plan fno training strategies"
```
