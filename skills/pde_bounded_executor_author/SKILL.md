---
name: pde-bounded-executor-author
description: Use when an agent receives a BOUNDED_EXECUTOR_AUTHORING_TASK_PREPARED artifact and must implement a bounded PDE executor by editing only executor.py, then validate and run the sandbox.
---

# PDE Bounded Executor Author

Use this skill only with a `BOUNDED_EXECUTOR_AUTHORING_TASK_PREPARED` artifact.

Your job is to implement a small, auditable `predict(context, candidate, params)` function. The engineering system handles static audit, scoring, and registry updates.

## Contract

1. Read the authoring task JSON.
2. Read `design_spec_path`, `manifest_path`, and `executor_path`.
3. Edit only paths listed in `allowed_edit_paths`.
4. Do not manually read paths listed in `forbidden_read_paths`; the sandbox command will use them.
5. Preserve the required function signature:

```python
def predict(context, candidate, params):
    ...
```

6. Return a NumPy array with shape `(N, 200, X)` unless `params["horizon"]` requests another horizon.
7. Preserve the first context steps exactly: `prediction[:, :context.shape[1], :] = context`.

## Allowed Implementation Style

- Use only modules allowed by `design_spec.security_policy.allowed_imports`.
- Prefer simple NumPy operations over complex abstractions.
- Keep state local to `predict`; do not read or write files.
- Do not use network, shell commands, dynamic imports, `eval`, `exec`, or target labels.
- If the paper mechanism is too underspecified, implement the safest baseline or a small interpretable variant and let the sandbox evidence decide.

## Validation Loop

Run the commands listed in `validation_commands` in order:

1. `validate-bounded-executor-implementation`
2. `run-bounded-executor-sandbox`
3. `record-bounded-executor-sandbox-event`

If validation fails, revise only `executor_path` and rerun from step 1. Stop after producing the event artifact or after a clear failure that should be recorded.

## Output

When finished, report:

- edited executor path
- audit decision and issue count
- sandbox `rules_passed`
- score delta vs persistence
- registry event path
