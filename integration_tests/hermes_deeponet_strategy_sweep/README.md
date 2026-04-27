# Hermes DeepONet Strategy Sweep

This fixture verifies that Hermes can run the DeepONet strategy-sweep command
and route its evidence into the strategy-growth gate.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_deeponet_strategy_sweep.py \
  --output-dir /tmp/pde-agent-hermes-deeponet-strategy-sweep-001
```

The fixture uses synthetic local windows, two evaluation start indices, and a
tiny strategy matrix covering supervised, spectral, and Burgers-residual
DeepONet losses. It is an agent-protocol canary, not a scientific quality
benchmark.
