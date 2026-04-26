# Hermes Integration Discipline

This project uses GPT-5.5/Codex as the builder, but Hermes/Kimi is part of the
target runtime. Any behavior meant to be executed by Hermes must be validated
with real Hermes before it is treated as stable.

## Core Rule

Do not let GPT-5.5's correction ability stand in for Hermes integration tests.

GPT-5.5 may build CLI tools, schemas, tests, docs, and deterministic harness
logic. Hermes must validate every Hermes-facing protocol boundary:

- `run_state.yaml` and `run_state.*.template.yaml`
- `bootstrap_prompt.md`
- `prompt_template.md`
- Hermes fixture README instructions
- allowed command lists
- required artifact lists
- session audit logic

## Required Development Sequence

For each new Hermes-facing capability:

1. Build the deterministic CLI/core function locally.
2. Add pytest coverage for the core function and CLI wrapper.
3. Run at least one non-Hermes real smoke when the command touches real data,
   checkpoints, or torch execution.
4. Create or update a Hermes fixture with a narrow `run_state.yaml` contract.
5. Add or update a session audit that checks required reads, exact allowed
   command sequence, forbidden actions, final state, and required artifacts.
6. Run the real Hermes CLI fixture with the target profile.
7. Commit only after the audit report passes or explicitly document the
   failure and its root cause.

If a capability is not exposed to Hermes, steps 4-6 are optional. Once a prompt,
template, or `run_state` exists for Hermes, steps 4-6 are mandatory.

## Prompt Discipline

Write prompts for Kimi/Hermes' failure modes, not GPT-5.5's strengths:

- Prefer explicit file paths over inferred paths.
- Prefer one high-level allowed command over a long command chain.
- Repeat forbidden actions in both `run_state.yaml` and `bootstrap_prompt.md`.
- Forbid command help, directory search, path guessing, downloads, and code
  patching unless they are the explicit purpose of that fixture.
- Treat generated artifacts as machine-reviewed by audit scripts, not manually
  inspected by Hermes.

## Run-State Discipline

Every Hermes fixture must make `run_state.yaml` the source of truth.

Required sections:

- `current_state`
- `allowed_actions`
- `forbidden_actions`
- `required_reads`
- `required_artifacts`
- `allowed_commands`
- `success_transition`

`allowed_commands` should contain exact command strings. If Hermes is allowed to
choose among commands, that choice must be encoded as separate explicit command
entries and covered by audit logic.

## Audit Gate

A Hermes session is not accepted because the terminal output "looks right".
Acceptance requires an audit report with:

- required reads in order,
- terminal commands exactly matching `allowed_commands`,
- no forbidden tool calls or forbidden command substrings,
- all terminal exit codes equal to zero,
- final payload state matching the fixture success transition,
- required artifacts present,
- fixture-specific semantic checks on the final report.

For FNO iteration, the semantic checks are:

- `fno_iteration_report.json` exists,
- `candidate_comparison.json` exists,
- `state_hint = FNO_ITERATION_COMPLETED`,
- `candidate_count > 0`,
- winner experiment id is non-empty,
- winner score and score delta are numeric.

For the constrained-free autonomy fixture, the semantic checks are:

- `fno_iteration_report.json` exists,
- `candidate_comparison.json` exists,
- `autonomy_decision.json` exists,
- Hermes reads only the required fixture files,
- Hermes runs exactly two terminal commands,
- the first command is the fixed `run-fno-iteration` command,
- the second command is one of the explicit `record-autonomy-decision` choices,
- the final state is `HERMES_AUTONOMY_DECISION_RECORDED`,
- the decision artifact uses one supported decision and reason code.

## When To Run Hermes

Run real Hermes immediately after any of these changes:

- a new Hermes fixture is added,
- a prompt or bootstrap rule changes,
- allowed or forbidden command lists change,
- a CLI command intended for Hermes changes its arguments or output schema,
- a session audit changes,
- an agent skill/harness boundary is introduced.

Do not wait until a full feature line is complete. Hermes canary runs should
happen at the first stable protocol boundary, even if the underlying model
strategy is still primitive.

## Current Canonical FNO Smoke

Prepare state:

```bash
uv run --frozen python scripts/prepare_fno_iteration_smoke.py \
  --windows-path /tmp/pde-agent-window-suite-task1-public/windows_f031.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --max-train-steps 1
```

Run and audit Hermes:

```bash
uv run --frozen python scripts/run_hermes_fno_iteration.py --timeout-seconds 420
```

Audit the latest saved session without rerunning Hermes:

```bash
uv run --frozen python scripts/run_hermes_fno_iteration.py --audit-only
```

## Current Constrained-Free Autonomy Canary

This canary checks whether Hermes/Kimi can make one bounded scientific workflow
decision without turning the harness into a fully static state machine. The
protocol still fixes the executable surface, but it leaves Hermes one meaningful
choice after seeing the `run-fno-iteration` terminal JSON:

- `request_evidence_sweep`: use when a single-window improvement needs
  validation across more windows.
- `accept_current_winner`: use when the current winner is clear enough to keep.

Prepare state:

```bash
uv run --frozen python scripts/prepare_autonomy_constrained_free.py \
  --windows-path /tmp/pde-agent-window-suite-task1-public/windows_f031.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir /tmp/pde-agent-hermes-autonomy-constrained-free-20260427-001 \
  --max-train-steps 1
```

Run and audit Hermes:

```bash
uv run --frozen python scripts/run_hermes_autonomy.py --timeout-seconds 420
```

Latest validated canary:

- Session: `20260427_005513_d44608`
- Audit: `integration_tests/hermes_autonomy_constrained_free/reports/audit_20260427_005513_d44608.json`
- Result: passed
- Decision: `request_evidence_sweep`
- Reason code: `single_window_improvement_needs_validation`
