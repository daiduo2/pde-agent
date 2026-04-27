# Hermes DeepONet Trainable Loop

This fixture verifies that Hermes can run a minimal trainable DeepONet
candidate through evaluation and the strategy-growth gate.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_deeponet_trainable_loop.py \
  --output-dir /tmp/pde-agent-hermes-deeponet-trainable-loop-001
```

The fixture uses synthetic local windows and a tiny torch model. It validates
the real execution path without claiming scientific model quality.
