# Hermes Autonomy Ablation V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable autonomy ablation comparing the existing hard-bound FNO protocol against a constrained-free Hermes protocol.

**Architecture:** Keep the existing `hermes_fno_iteration_smoke` as `hard_bound`. Add a constrained-free fixture where Hermes first runs `run-fno-iteration`, then chooses exactly one legal `record-autonomy-decision` command based on the JSON terminal output. Add a constrained-free audit that accepts choice among allowed decision commands while still rejecting forbidden commands and missing artifacts.

**Tech Stack:** Python, Typer, pytest, Hermes CLI fixture files, JSON artifact schemas.

---

## File Structure

- Create `src/pde_agent/core/autonomy_decision.py`: writes structured decision artifacts.
- Create `src/pde_agent/commands/record_autonomy_decision.py`: JSON payload wrapper.
- Modify `src/pde_agent/cli.py`: expose `record-autonomy-decision`.
- Create `tests/test_autonomy_decision.py`: core and CLI tests.
- Create `integration_tests/hermes_autonomy_constrained_free/*`: constrained-free Hermes fixture.
- Create `scripts/prepare_autonomy_constrained_free.py`: render runtime `run_state.yaml`.
- Create `src/pde_agent/core/hermes_autonomy_audit.py`: audit constrained-free sessions.
- Create `scripts/audit_hermes_autonomy.py` and `scripts/run_hermes_autonomy.py`: audit/run helpers.
- Create `tests/test_hermes_autonomy_constrained_free.py`: fixture render tests.
- Create `tests/test_hermes_autonomy_audit.py`: audit tests.
- Create `docs/superpowers/specs/2026-04-27-hermes-autonomy-ablation-design.md`: already exists.

## Task 1: Decision Artifact Command

**Files:**
- Create: `src/pde_agent/core/autonomy_decision.py`
- Create: `src/pde_agent/commands/record_autonomy_decision.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_autonomy_decision.py`

- [ ] **Step 1: Write failing tests**

Add tests proving `record_autonomy_decision` writes:

```json
{
  "schema_version": 1,
  "state_hint": "HERMES_AUTONOMY_DECISION_RECORDED",
  "decision": "request_evidence_sweep",
  "reason_code": "single_window_improvement_needs_validation",
  "evidence_source_path": ".../fno_iteration_report.json"
}
```

Also add a CLI test invoking:

```bash
pde-agent record-autonomy-decision --decision request_evidence_sweep --reason-code single_window_improvement_needs_validation --evidence-source-path report.json --output-path autonomy_decision.json
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run --frozen pytest tests/test_autonomy_decision.py -q
```

Expected: import or command failure.

- [ ] **Step 3: Implement minimal command**

Accept only these V0 decisions:

- `request_evidence_sweep`
- `accept_current_winner`

Accept only these V0 reason codes:

- `single_window_improvement_needs_validation`
- `clear_improvement_accept_for_now`

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_autonomy_decision.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/autonomy_decision.py src/pde_agent/commands/record_autonomy_decision.py src/pde_agent/cli.py tests/test_autonomy_decision.py
git commit -m "feat: add autonomy decision recorder"
```

## Task 2: Constrained-Free Fixture

**Files:**
- Create directory: `integration_tests/hermes_autonomy_constrained_free`
- Create: `integration_tests/hermes_autonomy_constrained_free/run_state.template.yaml`
- Create: `integration_tests/hermes_autonomy_constrained_free/bootstrap_prompt.md`
- Create: `integration_tests/hermes_autonomy_constrained_free/prompt_template.md`
- Create: `integration_tests/hermes_autonomy_constrained_free/task_spec.json`
- Create: `integration_tests/hermes_autonomy_constrained_free/README.md`
- Create: `integration_tests/hermes_autonomy_constrained_free/.gitignore`
- Create: `scripts/prepare_autonomy_constrained_free.py`
- Test: `tests/test_hermes_autonomy_constrained_free.py`

- [ ] **Step 1: Write failing fixture render tests**

Test that rendering produces a `run_state.yaml` with:

- one `run-fno-iteration` command,
- two legal `record-autonomy-decision` commands,
- required artifacts including `fno_iteration_report.json`, `candidate_comparison.json`, and `autonomy_decision.json`,
- no unresolved placeholders.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run --frozen pytest tests/test_hermes_autonomy_constrained_free.py -q
```

Expected: missing script/fixture failure.

- [ ] **Step 3: Implement fixture and prepare script**

The prompt must instruct Hermes:

1. read required files,
2. run exact `run-fno-iteration`,
3. use its terminal JSON metrics,
4. choose exactly one decision command,
5. stop at `HERMES_AUTONOMY_DECISION_RECORDED`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_hermes_autonomy_constrained_free.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add integration_tests/hermes_autonomy_constrained_free scripts/prepare_autonomy_constrained_free.py tests/test_hermes_autonomy_constrained_free.py
git commit -m "feat: add constrained-free autonomy fixture"
```

## Task 3: Constrained-Free Audit And Runner

**Files:**
- Create: `src/pde_agent/core/hermes_autonomy_audit.py`
- Create: `scripts/audit_hermes_autonomy.py`
- Create: `scripts/run_hermes_autonomy.py`
- Test: `tests/test_hermes_autonomy_audit.py`

- [ ] **Step 1: Write failing audit tests**

Synthetic session should pass when:

- required reads match,
- first terminal command is `run-fno-iteration`,
- second terminal command is one of the two decision commands,
- no forbidden commands appear,
- terminal exit codes are zero,
- final state is `HERMES_AUTONOMY_DECISION_RECORDED`,
- required artifacts exist,
- decision artifact has a valid decision and reason code.

Add failure tests for direct lower-level FNO command and missing decision artifact.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run --frozen pytest tests/test_hermes_autonomy_audit.py -q
```

Expected: missing module failure.

- [ ] **Step 3: Implement audit and runner scripts**

Audit should produce:

```json
{
  "status": "passed",
  "arm": "constrained_free",
  "decision": {
    "decision": "request_evidence_sweep",
    "reason_code": "single_window_improvement_needs_validation"
  },
  "metrics": {
    "command_count": 2,
    "decision_quality_score": 4
  }
}
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run --frozen pytest tests/test_hermes_autonomy_audit.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/core/hermes_autonomy_audit.py scripts/audit_hermes_autonomy.py scripts/run_hermes_autonomy.py tests/test_hermes_autonomy_audit.py
git commit -m "feat: add constrained-free autonomy audit"
```

## Task 4: Real Hermes Canary And Documentation

**Files:**
- Modify: `docs/hermes_integration_discipline.md`
- Modify: `integration_tests/hermes_autonomy_constrained_free/README.md`

- [ ] **Step 1: Run local tests**

Run:

```bash
uv run --frozen pytest tests/test_autonomy_decision.py tests/test_hermes_autonomy_constrained_free.py tests/test_hermes_autonomy_audit.py -q
```

- [ ] **Step 2: Run real constrained-free Hermes canary**

Prepare:

```bash
uv run --frozen python scripts/prepare_autonomy_constrained_free.py \
  --windows-path /tmp/pde-agent-window-suite-task1-public/windows_f031.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir /tmp/pde-agent-hermes-autonomy-constrained-free-20260427-001 \
  --max-train-steps 1
```

Run:

```bash
uv run --frozen python scripts/run_hermes_autonomy.py --timeout-seconds 420
```

Expected: audit status `passed`.

- [ ] **Step 3: Update docs with commands and result**

Document the constrained-free arm as the first autonomy ablation arm.

- [ ] **Step 4: Final verification**

Run:

```bash
uv run --frozen pytest tests -q
uv run --frozen --extra torch pytest tests/test_fno_iteration_runner.py tests/test_autonomy_decision.py tests/test_hermes_autonomy_constrained_free.py tests/test_hermes_autonomy_audit.py -q
git diff --check
```

- [ ] **Step 5: Commit docs**

```bash
git add docs/hermes_integration_discipline.md integration_tests/hermes_autonomy_constrained_free/README.md
git commit -m "docs: document constrained-free autonomy canary"
```
