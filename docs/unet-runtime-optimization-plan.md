# UNet Runtime Optimization Plan

## Goal

Make the official PDEBench UNet path auditable and fast enough for the competition
inference budget. The current evidence shows UNet is close to the 120 second budget
at 4-8 sample shards, but larger batches regress sharply. Treat this as an execution
path problem before making model-level changes.

## Scope

The first optimization pass is limited to inference mechanics:

- Record hotspot timings for window loading, checkpoint loading, model rollout, and
  output serialization.
- Support chunked competition inference so the 1000 sample task can run through the
  empirically best batch window instead of one oversized batch.
- Add CPU thread control for profiling and competition prediction.
- Use inference-only PyTorch execution and preallocated rollout tensors.

This pass does not change checkpoint weights, network topology, loss functions, or
scientific prediction logic.

## Acceptance Criteria

- The agent can produce a JSON runtime profile with phase-level timing.
- The competition UNet command can run with explicit chunk size and thread count.
- The optimized path preserves prediction shape and finite output checks.
- Profiling recommends either full submission or timeout evidence from measured
  shard projections.
- The full test suite passes.
