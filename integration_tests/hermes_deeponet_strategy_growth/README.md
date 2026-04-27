# Hermes DeepONet Strategy Growth

This fixture verifies that Hermes can route DeepONet candidate evaluation
reports through the generic operator evidence interface, the strategy
next-action router, and then the strategy-growth gate.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_deeponet_strategy_growth.py \
  --evaluation-report-path /tmp/deeponet_candidate_000.json \
  --evaluation-report-path /tmp/deeponet_candidate_001.json \
  --output-dir /tmp/pde-agent-hermes-deeponet-strategy-growth-001
```

The fixture does not train DeepONet. It starts from existing
`DEEPONET_CANDIDATE_EVALUATED` reports.
