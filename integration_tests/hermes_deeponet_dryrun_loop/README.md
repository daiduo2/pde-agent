# Hermes DeepONet Dryrun Loop

This fixture verifies that Hermes can run the lightweight DeepONet dry-run
candidate loop before routing evidence into the strategy-growth gate.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_deeponet_dryrun_loop.py \
  --output-dir /tmp/pde-agent-hermes-deeponet-dryrun-loop-001
```

The fixture uses synthetic local windows. It does not claim scientific model
quality; it validates protocol wiring and artifact contracts.
