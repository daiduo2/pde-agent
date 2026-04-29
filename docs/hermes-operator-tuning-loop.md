# Hermes Operator Tuning Loop

Hermes has permission to tune operator execution paths when an operator fails a
runtime or resource budget. This is separate from scientific model evolution.

## Permission Boundary

- Allowed by default: diagnostic profiling, shard scaling, batch/chunk sweeps,
  CPU thread sweeps, inference-only execution mode, tensor preallocation, and
  serialization timing.
- Not allowed in this loop: checkpoint weight changes, network topology changes,
  loss changes, physical correction logic, or new scientific hypotheses that change
  predicted dynamics.

Scientific changes must use the existing strategy evaluation and evidence ledger
path with score, baseline, failure analysis, and promotion review.

## UNet Runtime Flow

1. Run `run-hermes-operator-tuning-loop` for UNet with a bounded grid of chunk sizes
   and CPU thread counts.
2. Each grid point calls `profile-unet-runtime` on one or more validation shards.
3. Hermes selects the lowest projected full-sample inference time.
4. If the selected config is under the timeout budget, Hermes records a completed
   `LOCAL_EXPERIMENT_RESULT` and routes to `predict-unet-competition`.
5. If no config fits the budget, Hermes records timeout evidence and keeps the
   operator line open for further optimization.

## Current Task1 Evidence

On the local machine, UNet with chunked inference fits the 120 second budget. A
1000-sample direct profile completed in 98.28 seconds. A later Hermes-owned grid
over 64/128-sample shards selected `inference_batch_size=8` and
`torch_num_threads=1`, projecting 99.86 seconds. The exact best thread count can
vary, so the selected config should be treated as machine-specific evidence rather
than a universal constant.
