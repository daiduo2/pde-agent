# Experiment Config Audit V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic audit command for generated experiment configs.

**Architecture:** Create one core audit module that reads an experiment config and its source plan, records semantic issues without throwing, and writes an audit JSON. Add a thin command wrapper, Typer CLI registration, state hint, tests, and README docs.

**Tech Stack:** Python, Typer CLI, pytest, JSON artifact contracts.

---

### Task 1: Core Audit

**Files:**
- Create: `src/pde_agent/core/experiment_config_audit.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_experiment_config_audit.py`

- [ ] **Step 1: Write failing tests**

Create tests for approved audit and rejected audit after tampering with a
candidate parameter.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_experiment_config_audit.py -q
```

Expected: failure because `audit_experiment_config_from_file` does not exist.

- [ ] **Step 3: Implement core audit**

Implement `audit_experiment_config_from_file(config_path, output_path,
plan_path=None)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_experiment_config_audit.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/experiment_config_audit.py src/pde_agent/core/state_hints.py tests/test_experiment_config_audit.py
git commit -m "feat: audit experiment config core"
```

### Task 2: CLI

**Files:**
- Create: `src/pde_agent/commands/audit_experiment_config.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_experiment_config_audit.py`

- [ ] **Step 1: Write failing CLI test**

Invoke `audit-experiment-config` through Typer.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --frozen pytest tests/test_experiment_config_audit.py -q
```

Expected: CLI test fails because the command is not registered.

- [ ] **Step 3: Implement wrapper**

Return `success_payload` with `state_hint = EXPERIMENT_CONFIG_AUDITED`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_experiment_config_audit.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/audit_experiment_config.py src/pde_agent/cli.py tests/test_experiment_config_audit.py
git commit -m "feat: expose experiment config audit cli"
```

### Task 3: FNO Dry-Run Integration and Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document usage**

Add `audit-experiment-config` after `run-fno-iteration-dryrun`.

- [ ] **Step 2: Verify**

Run:

```bash
uv run --frozen pytest tests/test_experiment_config_audit.py -q
uv run --frozen pytest tests -q
uv run --frozen --extra data pytest tests -q
uv run --frozen pde-agent audit-experiment-config --help
git diff --check
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/superpowers/specs/2026-04-26-experiment-config-audit-v0-design.md docs/superpowers/plans/2026-04-26-experiment-config-audit-v0.md
git commit -m "docs: document experiment config audit"
```
