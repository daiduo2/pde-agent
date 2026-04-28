# Hermes Burgers Spectral Viscosity Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Hermes fixture and audit gate for the real-HDF5 Burgers spectral-viscosity benchmark command.

**Architecture:** Create a narrow fixture exposing exactly one benchmark command, a prepare script that renders run_state.yaml, and a dedicated audit that checks Hermes behavior plus final report semantics. Reuse existing Hermes runner patterns, but keep benchmark-specific semantic checks in a focused audit module.

**Tech Stack:** Python 3.14, JSON saved-session audit, Typer CLI command under test, pytest, existing Hermes fixture conventions.

---

## File Structure

- Create `integration_tests/hermes_burgers_spectral_viscosity_benchmark/`
  - `.gitignore`, `README.md`, `bootstrap_prompt.md`, `prompt_template.md`, `task_spec.json`, `run_state.template.yaml`, `.gitkeep` files.
- Create `scripts/prepare_burgers_spectral_viscosity_benchmark.py`
  - Renders `run_state.yaml` for the fixture.
- Create `src/pde_agent/core/hermes_burgers_spectral_viscosity_benchmark_audit.py`
  - Audits saved Hermes sessions and benchmark report semantics.
- Create `scripts/audit_hermes_burgers_spectral_viscosity_benchmark.py`
  - CLI wrapper around the audit module.
- Create `scripts/run_hermes_burgers_spectral_viscosity_benchmark.py`
  - Runs Hermes and then the audit, with `--audit-only` support.
- Add tests:
  - `tests/test_hermes_burgers_spectral_viscosity_benchmark.py`
  - `tests/test_hermes_burgers_spectral_viscosity_benchmark_audit.py`
  - `tests/test_hermes_burgers_spectral_viscosity_benchmark_runner_scripts.py`

## Task 1: Fixture And Prepare Script

**Files:**
- Create: `tests/test_hermes_burgers_spectral_viscosity_benchmark.py`
- Create: `integration_tests/hermes_burgers_spectral_viscosity_benchmark/*`
- Create: `scripts/prepare_burgers_spectral_viscosity_benchmark.py`

- [ ] **Step 1: Write failing prepare tests**

Create a test that imports `scripts/prepare_burgers_spectral_viscosity_benchmark.py`, copies fixture templates into a temp repo, calls `prepare_runtime`, and asserts:

- no unresolved `{{...}}`
- exactly one allowed command
- command contains `run-burgers-spectral-viscosity-benchmark`
- command includes `--sample-limit 4`, `--min-evaluation-windows 4`, and `--force`
- required artifacts include benchmark report, strategy decision, strategy evidence, baseline summary, and DeepONet sweep summary
- missing source path raises `SystemExit`

- [ ] **Step 2: Run prepare test and verify RED**

Run:

```bash
uv run --frozen pytest tests/test_hermes_burgers_spectral_viscosity_benchmark.py -q
```

Expected: fail because the prepare script and fixture do not exist.

- [ ] **Step 3: Add fixture files**

Create fixture files with this contract:

- `run_state.template.yaml` includes exact required reads, required artifacts, forbidden actions, and one allowed benchmark command.
- `bootstrap_prompt.md` tells Hermes to read required files first, run only the allowed command, not inspect artifacts, and stop at `BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED`.
- `prompt_template.md` points to `{run_state_path}`, `{bootstrap_prompt_path}`, and `{task_spec_path}`.
- `task_spec.json` records the benchmark purpose and success state.
- `.gitignore` ignores `runs/*`, `reports/*`, `hermes_home/*`, and `run_state.yaml` while keeping `.gitkeep`.

- [ ] **Step 4: Add prepare script**

Implement:

- `fixture_root(repo_root: Path) -> Path`
- `render_template(...) -> str`
- `prepare_runtime(...) -> dict[str, Path]`
- CLI args for `--source-path`, `--output-dir`, `--dataset-key`, `--sample-limit`, `--min-evaluation-windows`, and `--forecast-steps`

Validate source exists, dataset key is non-empty, sample limit is at least two, minimum evaluation windows is at least two, and sample limit is not below minimum windows.

- [ ] **Step 5: Run prepare tests**

Run:

```bash
uv run --frozen pytest tests/test_hermes_burgers_spectral_viscosity_benchmark.py -q
```

Expected: pass.

## Task 2: Dedicated Audit

**Files:**
- Create: `tests/test_hermes_burgers_spectral_viscosity_benchmark_audit.py`
- Create: `src/pde_agent/core/hermes_burgers_spectral_viscosity_benchmark_audit.py`
- Create: `scripts/audit_hermes_burgers_spectral_viscosity_benchmark.py`

- [ ] **Step 1: Write failing audit tests**

Add tests with synthetic saved Hermes session JSON. Cover:

- valid session passes with one benchmark command and valid report
- manual artifact inspection command such as `cat report.json` fails
- missing benchmark report fails
- invalid semantic report fails when `evaluation_window_count < min_evaluation_windows`

- [ ] **Step 2: Run audit tests and verify RED**

Run:

```bash
uv run --frozen pytest tests/test_hermes_burgers_spectral_viscosity_benchmark_audit.py -q
```

Expected: fail because audit module does not exist.

- [ ] **Step 3: Implement audit module**

Implement `audit_hermes_burgers_spectral_viscosity_benchmark_session(...)` with:

- required read prefix check
- exact terminal command sequence check
- forbidden tool call check
- terminal exit code check
- final payload `state_hint` check
- required artifact existence check
- benchmark report semantic check

Report shape:

```python
{
    "status": "passed" | "failed" | "error",
    "arm": "burgers_spectral_viscosity_benchmark",
    "session_id": "...",
    "session_path": "...",
    "required_artifacts": [...],
    "checks": [...],
    "tool_call_summary": {...},
    "terminal_commands": [...],
    "benchmark_report": {...},
    "metrics": {"command_count": 1, ...},
    "errors": [...],
}
```

- [ ] **Step 4: Implement audit script**

Follow `scripts/audit_hermes_strategy_growth.py` parsing helpers for:

- `_yaml_list`
- `_yaml_command_list`
- `_yaml_success_transition`

Add `_yaml_scalar_int` for `min_evaluation_windows`.

- [ ] **Step 5: Run audit tests**

Run:

```bash
uv run --frozen pytest tests/test_hermes_burgers_spectral_viscosity_benchmark_audit.py -q
```

Expected: pass.

## Task 3: Runner Script And Verification

**Files:**
- Create: `tests/test_hermes_burgers_spectral_viscosity_benchmark_runner_scripts.py`
- Create: `scripts/run_hermes_burgers_spectral_viscosity_benchmark.py`
- Modify: `README.md` or fixture README only if command docs need clarification.

- [ ] **Step 1: Write failing runner-script tests**

Test helper functions:

- prompt rendering includes run_state/bootstrap/task_spec paths
- success transition parser returns `BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED`
- latest session resolution works for fixture-local saved sessions

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
uv run --frozen pytest tests/test_hermes_burgers_spectral_viscosity_benchmark_runner_scripts.py -q
```

Expected: fail because runner script does not exist.

- [ ] **Step 3: Implement runner script**

Copy the safe structure from `scripts/run_hermes_strategy_growth.py`, but call:

- `scripts/audit_hermes_burgers_spectral_viscosity_benchmark.py`
- default fixture root `integration_tests/hermes_burgers_spectral_viscosity_benchmark`
- timeout default `600`

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --frozen pytest \
  tests/test_hermes_burgers_spectral_viscosity_benchmark.py \
  tests/test_hermes_burgers_spectral_viscosity_benchmark_audit.py \
  tests/test_hermes_burgers_spectral_viscosity_benchmark_runner_scripts.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Run full verification**

Run:

```bash
uv run --frozen pytest -q
git diff --check
```

Expected: pass.

- [ ] **Step 6: Real Hermes canary if available**

If `data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5` and `hermes` are available, run:

```bash
uv run --frozen python scripts/prepare_burgers_spectral_viscosity_benchmark.py
uv run --frozen python scripts/run_hermes_burgers_spectral_viscosity_benchmark.py --timeout-seconds 600
```

Record the actual audit status. If the command cannot run because Hermes or the data is unavailable, state that blocker explicitly.

- [ ] **Step 7: Commit**

Run:

```bash
git add integration_tests/hermes_burgers_spectral_viscosity_benchmark scripts/prepare_burgers_spectral_viscosity_benchmark.py scripts/audit_hermes_burgers_spectral_viscosity_benchmark.py scripts/run_hermes_burgers_spectral_viscosity_benchmark.py src/pde_agent/core/hermes_burgers_spectral_viscosity_benchmark_audit.py tests/test_hermes_burgers_spectral_viscosity_benchmark.py tests/test_hermes_burgers_spectral_viscosity_benchmark_audit.py tests/test_hermes_burgers_spectral_viscosity_benchmark_runner_scripts.py docs/superpowers/plans/2026-04-28-hermes-burgers-spectral-viscosity-benchmark.md
git commit -m "feat: add hermes burgers benchmark fixture"
```
