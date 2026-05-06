# Hermes Hypothesis Lab Infrastructure Canary

This fixture verifies that Hermes can run the Hypothesis Lab infrastructure
audit through one bounded command.

Project-wide protocol rules are documented in
[`docs/hermes_integration_discipline.md`](../../docs/hermes_integration_discipline.md).
The gate sequence being audited is documented in
[`docs/hermes_hypothesis_lab_runbook.md`](../../docs/hermes_hypothesis_lab_runbook.md).

Hermes must read `run_state.yaml`, `bootstrap_prompt.md`, and `task_spec.json`,
then execute the single allowed
`audit-hermes-hypothesis-lab-infrastructure` command. It must not inspect
generated artifacts manually or call lower-level Hypothesis Lab commands.

Run and audit Hermes in one step:

```bash
uv run --frozen python scripts/run_hermes_hypothesis_lab_infrastructure.py --timeout-seconds 300
```

Audit the latest saved session without rerunning Hermes:

```bash
uv run --frozen python scripts/run_hermes_hypothesis_lab_infrastructure.py --audit-only
```
