# Repository Contract

`pde-agent` is organized around stable contracts, not around ad hoc scripts.
The goal is to keep the agent runtime, scientific logic, and experiment
artifacts separable enough that the system can be audited and resumed without
guesswork.

## Active Layers

1. `src/pde_agent/core/`
   - Business logic, data contracts, algorithms, and state transitions.
   - Reusable behavior belongs here first.
2. `src/pde_agent/commands/`
   - Thin CLI wrappers only.
   - Parse options, call core functions, emit JSON.
3. `scripts/`
   - Operational entrypoints for prep, audit, and smoke runs.
   - May orchestrate core commands, but should not become a second business-logic layer.
   - Current subdirectories:
     - `scripts/audit/` for session and protocol audit entrypoints.
     - `scripts/data/` for dataset acquisition and local preflight helpers.
     - `scripts/manual/` for explicitly manual diagnostic utilities.
     - `scripts/smoke/` for fixture preparation entrypoints.
4. `runs/`
   - Ephemeral and reproducible artifacts.
   - Never the source of truth for code behavior.
5. `integration_tests/`
   - Fixture contracts and protocol canaries.
   - These define what the runtime is allowed to do.

## Active Runtime

The primary autonomous runtime is Claude Code SDK via `claude-agent-sdk`.
Hermes-era artifacts remain only as compatibility or historical evidence unless a
fixture explicitly says otherwise.

## Contract Rules

- One capability should map to one core implementation and one thin command.
- If a behavior is reused in more than one place, it belongs in `core/`.
- If a task requires a stable agent surface, define the JSON contract first.
- If a task needs auditing, add a fixture and a session audit before treating it
  as stable.
- Do not create new one-off script modules when the same behavior can be
  expressed as a command wrapper over core logic.
- Do not place general-purpose data I/O inside experiment-specific modules.
- Do not treat `runs/` as durable source code.

## Agent Workflow Rules

- Keep the agent-centered workflow: the system constrains and audits the agent.
- Prefer reusable workspaces over throwaway run directories.
- Treat evidence, failure memory, and promotion reports as first-class artifacts.
- When a runtime boundary fails, isolate the boundary before adding another
  experimental branch.
