# Local Window Suite V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a command that builds multiple real-HDF5 local validation windows and a manifest for FNO stress testing.

**Architecture:** Reuse `build_local_windows` for each horizon, but add a suite-level preflight that validates the source HDF5 shape and all requested horizons before writing any NPZ files. The suite command writes a JSON manifest that downstream agents can consume without changing existing prediction/scoring contracts.

**Tech Stack:** Python, h5py through the existing data extra, NumPy, Typer CLI, pytest, JSON artifact contracts.

---

### Task 1: Core Window Suite Builder

**Files:**
- Create: `src/pde_agent/core/local_window_suite.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_local_window_suite.py`

- [ ] **Step 1: Write failing tests**

Create an HDF5 fixture with shape `(3, 21, 8)`. Build horizons `[3, 5]` with `context_steps = 4`, `time_reduction = 2`, `space_reduction = 2`, and assert that two NPZ files plus `local_window_suite_manifest.json` are written. Add a second test requesting an impossible horizon and assert no NPZ files are written.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_window_suite.py -q
```

Expected: failure because `pde_agent.core.local_window_suite` does not exist.

- [ ] **Step 3: Implement core**

Implement:

```python
build_local_window_suite(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    forecast_steps: list[int],
    time_reduction: int,
    space_reduction: int,
    force: bool = False,
) -> dict[str, Any]
```

The implementation must:

- validate non-empty unique `forecast_steps`
- inspect the HDF5 dataset before writing
- compute `reduced_time = ((raw_time - 1) // time_reduction) + 1`
- reject `max(forecast_steps) > reduced_time - context_steps`
- call existing `build_local_windows` once per horizon
- write `local_window_suite_manifest.json`
- return the manifest dictionary

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_window_suite.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/local_window_suite.py src/pde_agent/core/state_hints.py tests/test_local_window_suite.py
git commit -m "feat: build local window suite core"
```

### Task 2: CLI

**Files:**
- Create: `src/pde_agent/commands/build_local_window_suite.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_local_window_suite.py`

- [ ] **Step 1: Write failing CLI test**

Invoke:

```bash
pde-agent build-local-window-suite \
  --source-path burgers.hdf5 \
  --output-dir suite \
  --forecast-steps 3 \
  --forecast-steps 5
```

Assert `status = ok`, `state_hint = LOCAL_WINDOW_SUITE_BUILT`, profile count, max forecast, and manifest path.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_window_suite.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement CLI wrapper**

Create `build_local_window_suite_command(...)` and register `build-local-window-suite` in `src/pde_agent/cli.py`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_window_suite.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/build_local_window_suite.py src/pde_agent/cli.py tests/test_local_window_suite.py
git commit -m "feat: expose local window suite cli"
```

### Task 3: Docs and Real Smoke

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document command**

Add usage after the local PDEBench window scoring section. Explain that public Task-1 data supports at most 31 forecast steps after the configured reduction, while future longer data can use larger horizons through the same command.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen --extra data pytest tests/test_local_window_suite.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent build-local-window-suite --help
git diff --check
```

If `data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5` is present, run:

```bash
uv run --frozen --extra data pde-agent build-local-window-suite \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-dir /tmp/pde-agent-window-suite-task1-public \
  --dataset-key tensor \
  --start-index 0 \
  --sample-limit 8 \
  --context-steps 10 \
  --forecast-steps 8 \
  --forecast-steps 16 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4 \
  --force
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-local-window-suite-v0-design.md docs/superpowers/plans/2026-04-26-local-window-suite-v0.md
git commit -m "docs: plan local window suite"
```
