# FNO Training Protocol v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the FNO training sprint with normalization, horizon-aware loss, and multi-objective selection so Claude SDK agent can produce stronger early/middle/late training evidence.

**Architecture:** Keep the implementation inside `src/pde_agent/core/task1_short_window_training.py` and the existing CLI wrappers. Extend existing config/evidence schemas conservatively so older artifacts remain readable.

**Tech Stack:** Python, NumPy, PyTorch via optional runtime, Typer CLI, pytest.

---

### Task 1: Normalization Protocol

**Files:**
- Modify: `src/pde_agent/core/task1_short_window_training.py`
- Modify: `src/pde_agent/commands/hypothesis_lab_runtime.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_task1_short_window_training.py`

- [ ] Add a failing test that calls `run_task1_short_window_training_loop(..., normalization="sample_standard")` and asserts the report/evidence/model metadata record `normalization`, predictions remain denormalized, and metrics are finite.
- [ ] Implement `SUPPORTED_NORMALIZATION_TYPES = {"none", "sample_standard"}`.
- [ ] Add `normalization: str = "none"` to the training loop signature, command wrapper, and CLI command.
- [ ] Implement context-derived mean/std normalization for context and target tensors before training, then denormalize predictions before saving metrics/artifacts.
- [ ] Store normalization metadata in config, checkpoint metadata, report model config, and evidence.
- [ ] Run `uv run --frozen pytest -q tests/test_task1_short_window_training.py::test_task1_short_window_training_supports_sample_standard_normalization`.

### Task 2: Horizon-Weighted Loss

**Files:**
- Modify: `src/pde_agent/core/task1_short_window_training.py`
- Test: `tests/test_task1_short_window_training.py`

- [ ] Add a failing test that plans candidates with `loss_types=["horizon_weighted_mse"]`, runs one candidate, and asserts the config/evidence records the loss.
- [ ] Add `"horizon_weighted_mse"` to `SUPPORTED_FNO_LOSS_TYPES`.
- [ ] Implement deterministic horizon weights: first 25% `1.5`, second 25% `1.0`, final 50% `0.75`.
- [ ] Ensure CLI validation accepts the new loss through the existing loss list path.
- [ ] Run `uv run --frozen pytest -q tests/test_task1_short_window_training.py::test_fno_training_accepts_horizon_weighted_mse`.

### Task 3: Candidate Planning Surface

**Files:**
- Modify: `src/pde_agent/core/task1_short_window_training.py`
- Modify: `src/pde_agent/commands/hypothesis_lab_runtime.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_task1_short_window_training.py`

- [ ] Add a failing test that calls `plan_fno_training_candidates(..., normalizations=["none", "sample_standard"])` and asserts planned configs include `normalization`.
- [ ] Add `normalizations` to `plan_fno_training_candidates` and `build_fno_training_harness_contract`.
- [ ] Include `normalization` in FNO config signatures to avoid duplicate/repeated candidate ambiguity.
- [ ] Add `--normalization` Typer options to candidate planning and harness contract commands.
- [ ] Pass candidate normalization into `run_task1_short_window_training_loop` during optimization.
- [ ] Run the planner and optimization tests for this file.

### Task 4: Multi-Objective FNO Selection

**Files:**
- Modify: `src/pde_agent/core/task1_short_window_training.py`
- Test: `tests/test_task1_short_window_training.py`

- [ ] Add a failing unit test with synthetic candidate records where candidate A has slightly better early score but no guardrails, and candidate B has evaluated middle/late guardrails with better aggregate score.
- [ ] Implement `_fno_selection_components(record)` returning early/middle/late score components and missing flags.
- [ ] Implement `_fno_selection_score(record)` with weights early `0.60`, middle `0.25`, late `0.15`.
- [ ] Update `_select_fno_candidate` to sort by `fno_selection_score`, then early score, val loss, runtime.
- [ ] Store `fno_selection_score` and `selection_components` in the selected candidate.
- [ ] Add selection reason code `multi_objective_selection`.
- [ ] Run the new selection test.

### Task 5: Claude SDK v2 Sprint

**Files:**
- Create: `runs/fno_training_protocol_v2_20260506/claude_fno_training_v2_prompt.md`
- Generated artifacts under: `runs/fno_training_protocol_v2_20260506/`

- [ ] Build a v2 harness contract with `normalization=none/sample_standard`, `loss_type=early_weighted_mse/horizon_weighted_mse`, `forecast_steps=47/95/190`, and bounded `train_steps`.
- [ ] Run Claude SDK runtime with the prompt and `uv run --frozen --extra agent pde-agent run-claude-agent-sdk-runtime ...`.
- [ ] Audit the produced plan/evidence/validation JSON for v2 fields.
- [ ] Run `uv run --frozen pytest -q`.

### Task 6: Completion Audit

**Files:**
- Inspect: `runs/fno_training_protocol_v2_20260506/*.json`
- Inspect: `src/pde_agent/core/task1_short_window_training.py`
- Inspect: `tests/test_task1_short_window_training.py`

- [ ] Verify `normalization` appears in candidate configs and training evidence.
- [ ] Verify `horizon_weighted_mse` appears in at least one candidate or accepted planning surface.
- [ ] Verify selected candidate contains `fno_selection_score` and `selection_components`.
- [ ] Verify middle/late guardrails are preserved in next-generation constraints when evaluated by any candidate.
- [ ] Verify full test suite passes.
