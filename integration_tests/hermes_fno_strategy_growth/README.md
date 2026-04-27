# Hermes FNO Strategy Growth

This fixture verifies that Hermes routes FNO evidence through the
strategy-growth gate instead of promoting raw FNO sweep output directly.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_fno_strategy_growth.py \
  --fno-sweep-summary-path /tmp/pde-agent-hermes-fno-strategy-growth-20260427-001/fno_evidence_summary.json
```

The fixture expects an existing `FNO_EVIDENCE_SWEEP_COMPLETED` summary. It does
not run FNO training or evidence sweep itself.
