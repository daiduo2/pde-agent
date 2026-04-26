# FNO V1 Default Policy Implementation Plan

**Goal:** Promote the evidence-backed FNO finetune setting into the default
planner/executor path without expanding the strategy surface.

**Evidence basis:** The local public Task-1 window suite shows
`scheduled_sampling` is the most reliable strategy family. The strongest
low-risk default is `rollout_steps=10`, `teacher_forcing_decay=0.85`,
`learning_rate=1e-4`, and `max_train_steps=16`.

## Step 1: Planner Policy Test

Add or update tests proving the long-horizon drift intervention exposes only
the V1 scheduled-sampling candidates:

- `training_strategy = ["scheduled_sampling"]`
- `rollout_steps = [10]`
- `teacher_forcing_decay = [0.85, 0.9]`

Run the focused planner test and confirm it fails against the current broader
grid.

## Step 2: Planner Policy Implementation

Update `use_scheduled_sampling_rollout_finetune` in
`src/pde_agent/core/next_experiment_plan.py` so the generated parameter grid
starts with the V1 default candidate and keeps only one nearby decay fallback.

Run the focused planner/config tests and confirm they pass.

## Step 3: Executor Default Test

Add a CLI test that omits `--max-train-steps` and asserts the reported
`train_steps` is `16`. Keep the explicit override test at `1` so quick smoke
paths remain available when requested.

Run the focused CLI test and confirm it fails with the current default.

## Step 4: Executor Default Implementation

Change the Typer CLI default for `execute-fno-finetune-candidate` from `1` to
`16`. Do not change the core function default or explicit command override
semantics in this step.

Run the focused FNO finetune tests with the torch extra.

## Step 5: Documentation And Real Smoke

Update README guidance so the default path is described as V1
scheduled-sampling, not a minimal one-step smoke. Then run a real public-window
smoke proving:

- generated candidate 0 is the V1 default,
- generated candidate 1 is the decay fallback,
- executor CLI default runs `16` train steps when the flag is omitted.

## Verification

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
uv run --frozen --extra torch pytest tests/test_fno_finetune_candidate.py -q
uv run --frozen --extra torch pytest tests/test_fno_evidence_sweep.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
git diff --check
```
