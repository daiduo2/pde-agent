# Hermes Diagnostic Smoke Test

This fixture verifies that Hermes can execute the synthetic PDE diagnostic chain:

1. init-run
2. resolve-config
3. eval-rollout
4. analyze-spectrum
5. check-physics
6. interpret-error
7. select-hypothesis
8. collect-artifacts

The fixture is deterministic and does not use real PDEBench data, checkpoints,
training, or code patching.

Run audit only:

```bash
uv run python scripts/audit_hermes_diagnostic.py
```

Run real Hermes diagnostic smoke:

```bash
uv run python scripts/run_hermes_diagnostic.py --timeout-seconds 300
```
