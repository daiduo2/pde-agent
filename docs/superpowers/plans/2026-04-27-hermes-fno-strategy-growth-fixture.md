# Hermes FNO Strategy Growth Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Hermes-facing fixture that runs the full FNO evidence-to-strategy-growth gate without allowing Hermes to bypass the adapter or promotion ledger.

**Architecture:** Create a new `hermes_fno_strategy_growth` fixture with exact command strings for six steps: convert FNO evidence, record proposal, build rubric, evaluate promotion, append ledger, and stop. Add a dedicated audit that verifies required reads, exact command sequence, forbidden actions, final ledger state, and required artifacts.

**Tech Stack:** Python 3.14, Typer CLI commands, YAML-like run_state templates, JSON/JSONL artifacts, pytest, Hermes saved-session audit.

---

## Files

- Create `integration_tests/hermes_fno_strategy_growth/` fixture files.
- Create `scripts/prepare_fno_strategy_growth.py`.
- Create `src/pde_agent/core/hermes_strategy_growth_audit.py`.
- Create `scripts/audit_hermes_strategy_growth.py`.
- Create `scripts/run_hermes_strategy_growth.py`.
- Add tests:
  - `tests/test_hermes_fno_strategy_growth.py`
  - `tests/test_hermes_strategy_growth_audit.py`
  - `tests/test_hermes_strategy_growth_runner_scripts.py`
- Update `docs/hermes_integration_discipline.md`.

## Task 1: Fixture And Prepare Script

- [ ] Write failing tests proving `prepare_fno_strategy_growth.py` renders:
  - no unresolved placeholders,
  - exactly five allowed commands,
  - required artifacts: strategy evidence summary, strategy proposal, rubric, promotion report, ledger,
  - no direct lower-level FNO candidate commands.
- [ ] Implement fixture files:
  - `.gitignore`
  - `README.md`
  - `bootstrap_prompt.md`
  - `prompt_template.md`
  - `task_spec.json`
  - `run_state.template.yaml`
  - `.gitkeep` files under `hermes_home`, `reports`, `runs`.
- [ ] Implement `scripts/prepare_fno_strategy_growth.py`:
  - input: `--fno-sweep-summary-path`,
  - output dir default: `/tmp/pde-agent-hermes-fno-strategy-growth-20260427-001`,
  - validates input file exists,
  - writes `run_state.yaml`.
- [ ] Run focused tests and commit `feat: add hermes fno strategy growth fixture`.

## Task 2: Strategy Growth Audit

- [ ] Write failing tests for `audit_hermes_strategy_growth_session`:
  - passes for exact five-command session ending in `STRATEGY_LEDGER_APPENDED`,
  - fails if raw FNO sweep summary is passed directly to promotion,
  - fails if generated artifacts are manually inspected with shell commands,
  - fails if ledger artifact is missing.
- [ ] Implement audit checks:
  - required reads prefix matches,
  - terminal command sequence exactly equals allowed commands,
  - no forbidden tool calls,
  - all terminal exit codes are zero,
  - final payload state is `STRATEGY_LEDGER_APPENDED`,
  - all required artifacts exist,
  - ledger has at least one JSONL row whose `state_hint` is `STRATEGY_LEDGER_APPENDED`.
- [ ] Implement `scripts/audit_hermes_strategy_growth.py`.
- [ ] Implement `scripts/run_hermes_strategy_growth.py`.
- [ ] Run focused tests and commit `feat: add hermes strategy growth audit`.

## Task 3: Verification And Documentation

- [ ] Update `docs/hermes_integration_discipline.md` with the canonical strategy-growth canary command sequence.
- [ ] Run:

```bash
uv run --frozen pytest tests/test_hermes_fno_strategy_growth.py tests/test_hermes_strategy_growth_audit.py tests/test_hermes_strategy_growth_runner_scripts.py -q
uv run --frozen pytest tests -q
git diff --check
```

- [ ] Commit `docs: document hermes fno strategy growth fixture`.

## Deferred

Do not run real Hermes in this implementation pass unless the fixture and audit tests are stable. The next canary run should use an existing FNO sweep summary artifact or a small synthetic summary generated into `/tmp`.
