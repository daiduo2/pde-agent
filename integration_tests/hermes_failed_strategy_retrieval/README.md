# Hermes Failed Strategy Retrieval

This fixture verifies that Hermes can route failed operator evidence through the
strategy next-action router, formalize a retrieved external strategy candidate,
and record an `external_retrieval` strategy proposal.

Prepare runtime state:

```bash
uv run --frozen python scripts/prepare_failed_strategy_retrieval.py \
  --output-dir /tmp/pde-agent-hermes-failed-strategy-retrieval-001
```

The fixture uses a local synthetic failed evidence summary. It does not perform
live internet retrieval; it validates the reusable artifact boundary that a
future retrieval tool should feed.
