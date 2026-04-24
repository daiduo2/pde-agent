# Hermes Smoke Runner Design

Date: 2026-04-24

## Purpose

Add a repo-local runner that can drive a real Hermes CLI session for the PDE agent protocol smoke test, then audit the resulting Hermes session log and generated PDE harness artifacts.

The main path should use real Hermes execution, not a mock. This matches the project goal: verify whether an external general agent can obey the `run_state.yaml` protocol, execute only allowed commands, and reach `RESULTS_COLLECTED` without human copy-paste.

## Non-Goals

- Do not move Hermes orchestration into the core `pde-agent` CLI.
- Do not add PDE science skills in this step.
- Do not run training.
- Do not mutate source code during the smoke run.
- Do not require CI to call an external LLM provider.

## User-Facing Commands

### Real run

`scripts/run_hermes_smoke.py` is the primary entry point.

Default behavior:

1. Locate the smoke fixture under `integration_tests/hermes_protocol_smoke`.
2. Render the prompt template using absolute fixture paths.
3. Launch `hermes -p hermes-protocol-smoke` in a PTY-compatible subprocess.
4. Submit the rendered prompt.
5. Wait until Hermes returns to the interactive prompt or exits.
6. Exit the Hermes session cleanly.
7. Run the audit step.
8. Write a machine-readable report under `integration_tests/hermes_protocol_smoke/reports`.

Important options:

- `--timeout-seconds`: hard timeout for the Hermes run.
- `--profile`: Hermes profile name, default `hermes-protocol-smoke`.
- `--fixture-root`: alternate fixture directory for future tests.
- `--audit-only`: skip Hermes execution and audit an existing session.
- `--session-id`: required with `--audit-only` unless the latest session is unambiguous.
- `--report-path`: explicit report output path.

### Audit only

`scripts/audit_hermes_smoke.py` audits a saved Hermes session JSON and the corresponding artifact manifest. It is deterministic and does not require network access.

This script is useful for:

- Reviewing a manually run Hermes session.
- Debugging protocol failures.
- Running local or CI checks without consuming model calls.

## Prompt Template

Add `integration_tests/hermes_protocol_smoke/prompt_template.md`.

The template must include:

- Absolute paths for `run_state.yaml`, `bootstrap_prompt.md`, `task_spec.json`, `base_config.json`, and `override.json`.
- Hard rules copied from the smoke protocol.
- A requirement to read all required files before acting.
- A requirement to use only exact command strings from `run_state.yaml`.
- A requirement to continue through `collect-artifacts`, not stop after intermediate `RESULTS_COLLECTED` hints from earlier commands.

The runner should render the template with a small set of explicit placeholders, not string-concatenate a large prompt in Python.

## Audit Contract

The audit step passes only if all of the following are true:

1. The session exists and contains at least one user message.
2. The first Hermes tool calls read the five required files using exact absolute paths.
3. No forbidden tool behavior appears:
   - source patching
   - training commands
   - command help inspection
   - command surface probing
   - broad file search outside the fixture
4. Terminal commands exactly match the five commands declared in `run_state.yaml`, in order:
   - `init-run`
   - `resolve-config`
   - `eval-rollout`
   - `record-timing`
   - `collect-artifacts`
5. Every terminal command has exit code `0`.
6. The final command output has `status: ok` and `state_hint: RESULTS_COLLECTED`.
7. The manifest exists at the required artifact path.
8. The manifest has:
   - `status: ok`
   - `phase: results_collected`
   - non-empty artifact paths for rollout metrics, timing metrics, prediction HDF5, error curve, and time CSV
   - `errors: []`

The audit report should include both summary booleans and evidence snippets so a reviewer can see which exact condition failed.

## Report Schema

The report is JSON with this shape:

```json
{
  "status": "passed",
  "session_id": "20260424_121355_f1cb57",
  "session_path": "/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/hermes_home/sessions/session_20260424_121355_f1cb57.json",
  "manifest_path": "/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003/artifacts/manifest.json",
  "checks": [
    {
      "name": "required_reads",
      "status": "passed",
      "details": {}
    }
  ],
  "tool_call_summary": {
    "read_file": 5,
    "terminal": 5
  },
  "terminal_commands": [
    "/Users/daiduo2/pde-agent/.venv/bin/pde-agent init-run --run-id hermes_protocol_smoke_003 --runs-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs"
  ],
  "errors": []
}
```

Use `status: failed` if any check fails. Use `status: error` only for malformed inputs that prevent a meaningful audit, such as a missing session file.

## Error Handling

The runner must fail loudly and leave evidence on disk.

- If Hermes cannot start, write a failed report with the subprocess error.
- If Hermes times out, terminate it and write a failed report.
- If the session JSON is missing, write a failed report and include the expected sessions directory.
- If the manifest is missing, the audit fails with the exact missing path.
- If forbidden actions are detected, the report should list the offending tool call and arguments.

## Testing

Add unit tests for the audit logic using small fixture session JSON files:

- passing session
- missing required read
- extra forbidden terminal command
- command sequence mismatch
- missing manifest
- malformed command output JSON

The real Hermes runner should be tested manually because it depends on local credentials, provider network access, and a PTY.

The verification command for deterministic tests remains:

`uv run pytest tests -v`

## Placement

Recommended files:

- `scripts/run_hermes_smoke.py`
- `scripts/audit_hermes_smoke.py`
- `src/pde_agent/core/hermes_smoke_audit.py`
- `tests/test_hermes_smoke_audit.py`
- `integration_tests/hermes_protocol_smoke/prompt_template.md`
- `integration_tests/hermes_protocol_smoke/reports/.gitkeep`

Keeping the audit logic in `src/pde_agent/core` makes it testable without spawning Hermes. Keeping the runner in `scripts` prevents provider-specific orchestration from becoming part of the reusable PDE harness API.
