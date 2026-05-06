# Task1 Candidate Factory Design

## Goal

Give Hermes a bounded way to expand Task1 strategy space after current
candidate families saturate. The factory should convert bottleneck evidence into
structured candidate families, sandbox them, evaluate whether the factory added
usable search space, and feed the result back into the existing runtime chain.

## Scope

This layer is infrastructure, not a new Burgers model. It should not train large
models, inspect hidden targets, or directly promote untested strategies. It
creates and validates candidate-family artifacts that later Hermes runs can use.

## Artifacts

### CandidateFamilyCard

A card is the unit Hermes can author, validate, sandbox, and remember. Required
fields:

- `candidate_family_id`
- `family_name`
- `task`
- `target_segment`
- `mechanism`
- `hypothesis`
- `parameter_space`
- `implementation_hooks`
- `expected_observables`
- `risk_factors`
- `sandbox_budget`
- `source`
- `status`

The validator enforces bounded values, non-empty lists, and stable IDs. The
schema is intentionally semi-formal: agent-authored scientific text remains in
`hypothesis`, while execution-sensitive fields stay deterministic.

### CandidateFactoryReport

The factory reads current benchmark/audit artifacts and produces a list of
cards. It can seed bounded mechanism families such as front25 dynamics,
residual correctors, spectral physics corrections, checkpoint adapters, and
failure-conditioned ensemble gates. These seeds are scaffolds; Hermes can later
author richer cards through the same schema.

### CandidateFamilySandboxReport

The sandbox report expands one or more cards into bounded experiment plans and
marks whether each card is executable using existing hooks. It does not execute
arbitrary code. For supported hooks, it records the exact next command and
options; for unsupported hooks, it records a blocked status with a concrete
reason.

### CandidateFactoryCapabilityReport

The benchmark judges whether a factory run improved the agent's candidate-space
capability. It checks diversity, executability, sandbox coverage, non-duplicate
mechanisms, and evidence readiness. Score improvement is deliberately not
required here; this benchmark measures whether Hermes gained usable search
surface.

## Data Flow

1. `generate-task1-candidate-families` reads a Task1 capability benchmark,
   runtime audit-chain report, optional trace, and optional prior family
   registry.
2. It writes `task1_candidate_families.json`.
3. `run-candidate-family-sandbox` validates cards and writes bounded sandbox
   plans.
4. `evaluate-candidate-factory-capability` validates the family report and
   sandbox report.
5. Runtime audit-chain can cite the factory benchmark as expansion evidence when
   candidate space is exhausted.

## Failure Handling

Invalid cards fail validation before sandboxing. Unsupported hooks are recorded
as blocked, not silently skipped. Duplicate mechanisms are accepted only when
their parameter spaces differ enough to represent a new experiment surface.

## Testing

Focused tests should cover card validation, generation from stagnated Task1
benchmark artifacts, sandbox planning for supported and unsupported hooks,
factory benchmark decisions, CLI wiring, and tool-router visibility.
