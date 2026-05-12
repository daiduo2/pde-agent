# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pde-agent` is a CLI-first diagnostic harness for PDE (Partial Differential Equation) agent experiments. It targets the PDEBench competition, specifically Burgers' equation (Task 1), using neural operators (FNO, UNet, DeepONet). The codebase supports both local model iteration (short-horizon scoring) and competition submission pipelines.

The active autonomous runtime is **Claude Code SDK** via the `claude-agent-sdk`
package. Historical Hermes/Kimi artifacts may exist in old documents or
archived files, but they are not the active execution substrate. New agent
workflows must use Claude SDK-facing commands and contracts.

Use [docs/repo_contract.md](docs/repo_contract.md) as the canonical layer
definition. If a capability looks reusable, it belongs in `src/pde_agent/core/`
first, with a thin command wrapper on top. `scripts/` are for operational
prep, audit, and smoke entrypoints, not a second business-logic layer.

## Build and Development Commands

All commands use `uv`. The project has a `uv.lock` file and uses `--frozen` in CI/fixtures.

```bash
# Install dependencies (dev group is automatic)
uv sync

# Run the CLI
uv run --frozen pde-agent <command>

# With optional extras
uv run --frozen --extra data  pde-agent <command>   # h5py for HDF5 I/O
uv run --frozen --extra torch pde-agent <command>   # PyTorch for FNO/UNet
uv run --frozen --extra agent pde-agent <command>   # claude-agent-sdk runtime

# Run tests
uv run --frozen pytest                           # all tests
uv run --frozen pytest tests/test_<file>.py      # single file
uv run --frozen pytest tests/test_<file>.py::test_<name>  # single test

# Run tests with torch/data extras
uv run --frozen --extra torch --extra data pytest tests/test_fno_backend_v0.py
```

**Note**: `external/` is excluded from pytest recursion (`norecursedirs` in `pyproject.toml`).

## High-Level Architecture

### Command / Core Split

Every CLI command follows the same two-layer pattern:

- **`src/pde_agent/commands/`** (~126 files) — Thin CLI wrappers. Parse `typer.Option`, call the core function, and echo JSON. These files are short (~50 lines) and contain no business logic.
- **`src/pde_agent/core/`** (~153 files) — Business logic, algorithms, state machines, and data contracts.

### Repository Layers

- `src/pde_agent/core/` — stable logic, contracts, and reusable algorithms.
- `src/pde_agent/commands/` — thin CLI wrappers only.
- `scripts/` — operational prep/audit/smoke entrypoints.
- `integration_tests/` — executable protocol fixtures and canaries.
- `runs/` — artifacts and evidence, not source.

The CLI entry point (`src/pde_agent/cli.py`, ~6600 lines) imports every command function and registers it with `typer.Typer`. When adding a new command, add both the core function and the thin wrapper, then wire the import and `@app.command(...)` registration in `cli.py`.

### JSON-Only CLI Contract

Every command outputs a single JSON object (echoed via `typer.echo(json.dumps(...))`). Do not print plain text or log to stdout. The standard shape is produced by helpers in `core/json_output.py`:

```python
{
  "run_id": "...",
  "command": "score-local-rollout",
  "status": "ok" | "error",
  "state_hint": "LOCAL_ROLLOUT_SCORED",   # machine-readable state token
  "artifacts": {"score_json_path": "runs/score.json"},
  "metrics": {"rel_mse": 0.05, "rmse": 0.12},
  "next_hint": "use this score to compare local model variants",
  "errors": []
}
```

- **`state_hint`**: A constant string (e.g., `LOCAL_ROLLOUT_SCORED`, `FNO_ITERATION_COMPLETED`) that drives the external state machine. Every core module defines its own `STATE_HINT` or `*_STATE_HINT` constant near the top.
- **`artifacts`**: Machine-readable paths to files the next command may consume.
- **`metrics`**: Numerical results for comparison and evidence tracking.
- **`next_hint`**: Human-readable guidance for the agent (not a command string).

### Array Contract for Burgers Data

All local window and prediction files are NPZ with rank-3 arrays of shape `[N, T, X]` (samples, time, space). The contract is enforced by `core/array_contract.py`:

- `context`: `[N, context_time, X]` — known history
- `prediction`/`target`: `[N, forecast_time, X]` — predicted / ground truth
- Metadata must include `problem_id` and `source_mode`

Use `validate_burgers_bundle()` before consuming arrays in new algorithms.

### Optional Backends (Torch-Guarded)

FNO and UNet support require PyTorch but must not break import when torch is absent. Every torch-dependent core module uses a `_require_torch()` helper that lazily imports torch and raises a clear error if missing. CLI commands that need torch should be documented with `--extra torch`.

### Run Directory and Manifest-Centered Artifact Contract

Every run gets a `run_id` and a directory tree under `runs/<run_id>/`. The manifest (`artifacts/manifest.json`) is the source of truth for what files exist. Commands reference each other through artifact paths, not hardcoded filenames. The `init-run` command bootstraps this structure.

### Agent Runtime Discipline

Any behavior meant for autonomous execution must pass a bounded runtime and
artifact audit before it is considered stable. The current path is Claude SDK
first. Key points:

1. Build deterministic CLI/core locally
2. Add pytest coverage
3. Run a real smoke test when touching real data, torch, or Claude SDK runtime code
4. Build/update a Claude SDK task or closed-loop contract with a narrow command surface
5. Add/update a session audit that checks required reads, exact allowed commands, forbidden actions, final state, and required artifacts
6. Run the real Claude SDK runtime when validating agent-owned behavior
7. Commit only after the audit report passes

Runtime fixtures live in `integration_tests/`. Helper scripts for preparing and
auditing them live in `scripts/`.

### Strategy Growth Gate

External retrieval, model proposals, and strategy discoveries must not bypass the evidence gate. The canonical sequence is:

1. `strategy_proposal.json` — records hypothesis, mechanism, plan, observables, risks
2. `experiment_rubric.json` — defines quantitative evaluation standard
3. `promotion_report.json` — evaluates evidence against rubric
4. `strategy_ledger.jsonl` — records the attempt (success, failure, or revise)

Promotion (`promote_to_skill_candidate = true`) does **not** install a skill or patch model code. Skill creation is a separate audited step.

### State Machine and Audit Patterns

- **Run State Contract**: `docs/run_state_contract.md`. The real state machine lives outside the CLI. The CLI only produces state hints in JSON output.
- **Audit Reports**: Every major agent-facing capability should have an audit script in `scripts/audit_*.py`. Audits check required reads, exact command sequences, forbidden actions, exit codes, artifact presence, and fixture-specific semantic checks.
- **Smoke Tests**: `scripts/smoke/prepare_*.py` set up fixture state; runtime-specific scripts execute and audit agent sessions.

### Test Organization

- `tests/test_*.py` — Unit and integration tests (1100+ total). Each test file mirrors a core or command module.
- Tests use `tmp_path` for filesystem isolation.
- Torch-dependent tests skip gracefully when torch is not installed.
- Integration test fixtures in `integration_tests/` are not run by pytest; they are driven by scripts.
