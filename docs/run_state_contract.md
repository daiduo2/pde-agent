# Thin-Hermes Run State Contract

The real state machine is maintained outside Hermes.

See [Hermes Integration Discipline](hermes_integration_discipline.md) for the
required development and audit gate before treating Hermes-facing protocols as
stable.

Minimum shape:

```yaml
run_id: run_20260423_001
current_state: ERROR_ATTRIBUTED
allowed_actions:
  - read_manifest
  - read_rollout_metrics
forbidden_actions:
  - patch_code
  - run_training
required_artifacts:
  - artifacts/manifest.json
```
