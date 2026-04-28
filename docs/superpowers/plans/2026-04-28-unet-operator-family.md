# UNet Operator Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal trainable UNet operator family that can produce local Burgers rollout evidence compatible with the existing operator strategy gate.

**Architecture:** Implement UNet-specific execution, evaluation, evidence, and strategy-sweep modules that mirror the existing DeepONet contracts without prematurely generalizing shared code. CLI commands return the existing `success_payload` envelope and `convert-operator-evidence-summary` remains the bridge into strategy-growth evidence.

**Tech Stack:** Python, NumPy, torch optional extra, Typer CLI, pytest, existing local scoring/Burgers diagnostic helpers.

---

## File Structure

- Create `src/pde_agent/core/unet_candidate_execution.py`: train tiny torch Conv1d UNet, write prediction/checkpoint/execution report.
- Create `src/pde_agent/core/unet_candidate_evaluation.py`: run existing score/error/physics/residual/hypothesis diagnostics for UNet predictions.
- Create `src/pde_agent/core/unet_evidence_sweep.py`: normalize UNet evaluation reports into records, comparison winners, CSV, and JSON evidence summary.
- Create `src/pde_agent/core/unet_strategy_sweep.py`: execute/evaluate bounded UNet strategy grids and build UNet evidence.
- Create `src/pde_agent/commands/execute_unet_candidate.py`: CLI payload wrapper for execution.
- Create `src/pde_agent/commands/evaluate_unet_candidate.py`: CLI payload wrapper for evaluation.
- Create `src/pde_agent/commands/build_unet_evidence_sweep.py`: CLI payload wrapper for evidence.
- Create `src/pde_agent/commands/run_unet_strategy_sweep.py`: CLI payload wrapper for strategy sweep.
- Modify `src/pde_agent/cli.py`: register four UNet commands.
- Create tests:
  - `tests/test_unet_candidate_execution.py`
  - `tests/test_unet_candidate_evaluation.py`
  - `tests/test_unet_evidence_sweep.py`
  - `tests/test_unet_strategy_sweep.py`
- Modify `tests/test_operator_strategy_evidence.py`: cover UNet conversion through generic operator evidence.

### Task 1: UNet Candidate Execution

**Files:**
- Create: `src/pde_agent/core/unet_candidate_execution.py`
- Create: `src/pde_agent/commands/execute_unet_candidate.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_unet_candidate_execution.py`

- [ ] **Step 1: Write failing execution tests**

Create `tests/test_unet_candidate_execution.py` with synthetic local windows and tests for:

```python
def test_execute_unet_candidate_trains_torch_model_and_writes_report(tmp_path: Path) -> None:
    report = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=output_dir,
        start_index=2,
        train_steps=6,
        training_strategy="unet_supervised",
        base_channels=8,
        depth=2,
        learning_rate=1e-2,
        seed=123,
        force=False,
    )
    assert report["state_hint"] == "UNET_CANDIDATE_EXECUTED"
    assert report["candidate"]["experiment_id"] == "unet_unet_supervised_s0002_steps0006"
    assert report["train"]["mode"] == "torch_unet1d"
    assert report["train"]["loss_terms"] == ["mse"]
    assert len(report["train"]["loss_trace"]) == 6
    assert Path(report["artifacts"]["candidate_prediction_path"]).exists()
```

Also test CLI payload, strategy labels, and the `unet_spectral` / `unet_physics_residual` loss terms.

- [ ] **Step 2: Run execution tests and verify RED**

Run: `uv run --frozen pytest tests/test_unet_candidate_execution.py -q`

Expected: fail because `pde_agent.core.unet_candidate_execution` does not exist.

- [ ] **Step 3: Implement minimal UNet execution**

Implement `execute_unet_candidate` with:

- `_MinimalUNet1D` Conv1d encoder/decoder preserving spatial length.
- `_training_loss` with `unet_supervised`, `unet_spectral`, `unet_physics_residual`.
- `_burgers_residual` copied in shape semantics from DeepONet.
- `_load_windows`, validation helpers, output path preparation.
- prediction `.npz` containing `prediction` and `metadata_json`.
- execution report state hint `UNET_CANDIDATE_EXECUTED`.

Add `execute_unet_candidate_command` returning:

```python
success_payload(
    run_id="local",
    command="execute-unet-candidate",
    state_hint="UNET_CANDIDATE_EXECUTED",
    artifacts={...},
    metrics={
        "experiment_id": report["candidate"]["experiment_id"],
        "training_strategy": report["candidate"]["training_strategy"],
        "final_loss": report["train"]["final_loss"],
    },
    next_hint="evaluate-unet-candidate",
)
```

Register `execute-unet-candidate` in `src/pde_agent/cli.py`.

- [ ] **Step 4: Run execution tests and verify GREEN**

Run: `uv run --frozen pytest tests/test_unet_candidate_execution.py -q`

Expected: all execution tests pass.

### Task 2: UNet Candidate Evaluation

**Files:**
- Create: `src/pde_agent/core/unet_candidate_evaluation.py`
- Create: `src/pde_agent/commands/evaluate_unet_candidate.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_unet_candidate_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create tests that execute a tiny UNet candidate, evaluate it against the same windows file, and assert:

```python
assert report["state_hint"] == "UNET_CANDIDATE_EVALUATED"
assert report["candidate"]["training_strategy"] == "unet_supervised"
assert report["score"]["competition_like_score_short"] > 0
assert report["baseline_comparison"]["score_delta"] == (
    report["score"]["competition_like_score_short"] - 1.0
)
assert report["next_command"] == "build-unet-evidence-sweep"
```

Add CLI coverage for `evaluate-unet-candidate`.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `uv run --frozen pytest tests/test_unet_candidate_evaluation.py -q`

Expected: fail because evaluation module/CLI do not exist.

- [ ] **Step 3: Implement UNet evaluation**

Mirror `evaluate_deeponet_candidate` but use:

- accepted input state hint `UNET_CANDIDATE_EXECUTED`
- output state hint `UNET_CANDIDATE_EVALUATED`
- `UnetCandidateEvaluationError`
- `next_command = "build-unet-evidence-sweep"`

Call existing helpers:

- `score_prediction_npz`
- `diagnose_rollout_error_npz`
- `check_burgers_physics_npz`
- `evaluate_burgers_residual_file`
- `select_pde_hypothesis_from_files`

Add CLI wrapper and register in `cli.py`.

- [ ] **Step 4: Run evaluation tests and verify GREEN**

Run: `uv run --frozen pytest tests/test_unet_candidate_evaluation.py -q`

Expected: all evaluation tests pass.

### Task 3: UNet Evidence Sweep

**Files:**
- Create: `src/pde_agent/core/unet_evidence_sweep.py`
- Create: `src/pde_agent/commands/build_unet_evidence_sweep.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `tests/test_operator_strategy_evidence.py`
- Test: `tests/test_unet_evidence_sweep.py`

- [ ] **Step 1: Write failing evidence tests**

Create tests that build two UNet evaluation reports and assert:

```python
summary = build_unet_evidence_sweep(
    evaluation_report_paths=[path_a, path_b],
    output_dir=output_dir,
    force=False,
)
assert summary["state_hint"] == "UNET_EVIDENCE_SWEEP_COMPLETED"
assert summary["model_family"] == "UNet"
assert summary["comparison_count"] == 1
assert summary["next_command"] == "convert-operator-evidence-summary"
```

Add duplicate `experiment_id` rejection. Add operator evidence coverage:

```python
evidence = convert_operator_evidence_summary(
    sweep_summary_path=summary_path,
    output_path=output_path,
    model_family="UNet",
    source_state_hint="UNET_EVIDENCE_SWEEP_COMPLETED",
    long_window_threshold=4,
    force=False,
)
assert evidence["model_family"] == "UNet"
assert evidence["source_unet_evidence_summary_path"] == str(summary_path)
```

- [ ] **Step 2: Run evidence tests and verify RED**

Run: `uv run --frozen pytest tests/test_unet_evidence_sweep.py tests/test_operator_strategy_evidence.py -q`

Expected: fail because UNet evidence module/CLI do not exist.

- [ ] **Step 3: Implement UNet evidence sweep**

Mirror `deeponet_evidence_sweep` with:

- `STATE_HINT = "UNET_EVIDENCE_SWEEP_COMPLETED"`
- `EVALUATION_STATE_HINT = "UNET_CANDIDATE_EVALUATED"`
- `model_family = "UNet"`
- CSV fields matching DeepONet plus residual metrics.
- grouping by `(start_index, train_steps)`.

Add CLI wrapper and register `build-unet-evidence-sweep`.

- [ ] **Step 4: Run evidence tests and verify GREEN**

Run: `uv run --frozen pytest tests/test_unet_evidence_sweep.py tests/test_operator_strategy_evidence.py -q`

Expected: all evidence tests pass.

### Task 4: UNet Strategy Sweep

**Files:**
- Create: `src/pde_agent/core/unet_strategy_sweep.py`
- Create: `src/pde_agent/commands/run_unet_strategy_sweep.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_unet_strategy_sweep.py`

- [ ] **Step 1: Write failing strategy sweep tests**

Create tests equivalent to DeepONet strategy sweep, asserting:

```python
summary = run_unet_strategy_sweep(
    windows_path=windows_path,
    output_dir=output_dir,
    baseline_score_path=baseline_score_path,
    start_indices=[1, 2],
    training_strategies=["unet_supervised", "unet_spectral", "unet_physics_residual"],
    train_steps=[3],
    base_channels=[8],
    depths=[2],
    learning_rates=[1e-2],
    seeds=[5],
    force=False,
)
assert summary["state_hint"] == "UNET_STRATEGY_SWEEP_COMPLETED"
assert summary["record_count"] == 6
assert summary["evidence_sweep"]["state_hint"] == "UNET_EVIDENCE_SWEEP_COMPLETED"
assert summary["evaluation_window_count"] == 2
```

Cover multiple start indices, loss-weight expansion, and CLI payload.

- [ ] **Step 2: Run strategy sweep tests and verify RED**

Run: `uv run --frozen pytest tests/test_unet_strategy_sweep.py -q`

Expected: fail because strategy sweep module/CLI do not exist.

- [ ] **Step 3: Implement UNet strategy sweep**

Mirror `deeponet_strategy_sweep` with UNet-specific hyperparameters:

- `base_channels`
- `depths`
- `training_strategies`
- loss-weight expansion only for relevant strategies.

Write:

- `unet_strategy_sweep_summary.json`
- `unet_strategy_sweep_summary.csv`
- `unet_evidence/evidence_summary.json`
- `unet_evidence/evidence_summary.csv`

Add CLI wrapper and register `run-unet-strategy-sweep`.

- [ ] **Step 4: Run strategy sweep tests and verify GREEN**

Run: `uv run --frozen pytest tests/test_unet_strategy_sweep.py -q`

Expected: all strategy sweep tests pass.

### Task 5: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Verify: all new and existing tests

- [ ] **Step 1: Add README command examples**

Add a short UNet section after the DeepONet/Burgers benchmark material with:

```bash
uv run --frozen --extra torch pde-agent run-unet-strategy-sweep \
  --windows-path runs/burgers_spectral_viscosity_benchmark_001/baseline/windows_f031.npz \
  --baseline-score-path runs/burgers_spectral_viscosity_benchmark_001/baseline/persistence_score.json \
  --output-dir runs/unet_strategy_sweep_001 \
  --start-index 0 \
  --training-strategy unet_supervised \
  --training-strategy unet_spectral \
  --training-strategy unet_physics_residual \
  --train-steps 4 \
  --base-channels 8 \
  --depth 2 \
  --learning-rate 0.01 \
  --seed 5 \
  --force
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
uv run --frozen pytest \
  tests/test_unet_candidate_execution.py \
  tests/test_unet_candidate_evaluation.py \
  tests/test_unet_evidence_sweep.py \
  tests/test_unet_strategy_sweep.py \
  tests/test_operator_strategy_evidence.py \
  -q
```

Expected: all targeted tests pass.

- [ ] **Step 3: Run full suite and diff check**

Run:

```bash
uv run --frozen pytest -q
git diff --check
```

Expected: full suite passes and diff check is clean.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add -- README.md src/pde_agent/cli.py src/pde_agent/core/unet_candidate_execution.py src/pde_agent/core/unet_candidate_evaluation.py src/pde_agent/core/unet_evidence_sweep.py src/pde_agent/core/unet_strategy_sweep.py src/pde_agent/commands/execute_unet_candidate.py src/pde_agent/commands/evaluate_unet_candidate.py src/pde_agent/commands/build_unet_evidence_sweep.py src/pde_agent/commands/run_unet_strategy_sweep.py tests/test_unet_candidate_execution.py tests/test_unet_candidate_evaluation.py tests/test_unet_evidence_sweep.py tests/test_unet_strategy_sweep.py tests/test_operator_strategy_evidence.py
git commit -m "feat: add unet operator family"
```

## Self-Review

- Spec coverage: candidate execution, evaluation, evidence, sweep, CLI, tests, and README are covered.
- Completeness scan: all steps include concrete file paths and commands.
- Type consistency: public function and command names match the design; state hints are consistent across tasks.
