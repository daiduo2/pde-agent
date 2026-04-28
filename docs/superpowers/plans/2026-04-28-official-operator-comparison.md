# Official Operator Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Burgers official-operator comparison command that evaluates persistence, spectral residual ridge, official FNO, and official UNet on the same local windows.

**Architecture:** Implement one focused core orchestrator plus one CLI wrapper. Reuse existing baseline, spectral, FNO, UNet, scoring, rollout-error, and Burgers-physics helpers; normalize their summaries into a leaderboard report.

**Tech Stack:** Python, NumPy, optional torch through existing FNO/UNet adapters, Typer CLI, pytest.

---

## File Structure

- Create `src/pde_agent/core/official_operator_comparison.py`: orchestrates sub-runs, ranks records, writes report.
- Create `src/pde_agent/commands/run_official_operator_comparison.py`: standard `success_payload` wrapper.
- Modify `src/pde_agent/cli.py`: register `run-official-operator-comparison`.
- Create `tests/test_official_operator_comparison.py`: unit tests with monkeypatched sub-runs and CLI coverage.
- Modify `README.md`: document the command and real-checkpoint usage.

### Task 1: Tests

- [ ] **Step 1: Write failing orchestration tests**

Create `tests/test_official_operator_comparison.py` with fixture summaries for
baseline, spectral, FNO, and UNet. Assert:

- report state hint is `OFFICIAL_OPERATOR_COMPARISON_COMPLETED`
- four model records are present
- leaderboard is sorted by score descending
- best model and recommendation are recorded
- report JSON is written

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest -q tests/test_official_operator_comparison.py
```

Expected: import failure for missing `pde_agent.core.official_operator_comparison`.

### Task 2: Core Implementation

- [ ] **Step 1: Implement core orchestrator**

Add `run_official_operator_comparison(...)` with explicit paths and parameters.
Use subdirectories:

- `baseline/`
- `spectral/`
- `official_fno/`
- `official_unet/`

Normalize all summaries into records with:

- `model_id`
- `model_family`
- `source_type`
- `score`, `rel_mse`, `rmse`
- `dominant_error_pattern`
- `physics_issue`
- `artifacts`

- [ ] **Step 2: Verify GREEN**

Run:

```bash
uv run --frozen pytest -q tests/test_official_operator_comparison.py
```

### Task 3: CLI and Docs

- [ ] **Step 1: Add CLI wrapper**

Create `run_official_operator_comparison_command(...)` and register
`run-official-operator-comparison` in `src/pde_agent/cli.py`.

- [ ] **Step 2: Add README usage**

Document official FNO/UNet comparison separately from trainable UNet and FNO
diagnosis sections.

- [ ] **Step 3: Verify targeted and full tests**

Run:

```bash
uv run --frozen pytest -q tests/test_official_operator_comparison.py
uv run --frozen pytest -q
```

### Task 4: Real Smoke and Commit

- [ ] **Step 1: Run real checkpoint smoke**

Run:

```bash
uv run --frozen --extra torch pde-agent run-official-operator-comparison \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-dir /tmp/pde-agent-official-operator-comparison-smoke \
  --dataset-key tensor \
  --fno-checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --unet-checkpoint-path data/pdebench/checkpoints/burgers_Unet.tar \
  --sample-limit 8 \
  --context-steps 10 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4 \
  --spectral-modes 16 \
  --spectral-ridge 0.001 \
  --nu 0.001 \
  --dt 1.0 \
  --dx 1.0 \
  --force
```

- [ ] **Step 2: Commit**

Run:

```bash
git add -- README.md docs/superpowers/specs/2026-04-28-official-operator-comparison-design.md docs/superpowers/plans/2026-04-28-official-operator-comparison.md src/pde_agent/cli.py src/pde_agent/core/official_operator_comparison.py src/pde_agent/commands/run_official_operator_comparison.py tests/test_official_operator_comparison.py
git commit -m "feat: add official operator comparison"
```
