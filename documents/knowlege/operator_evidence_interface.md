# Operator Evidence Interface

This note defines the current reusable boundary between model-family experiments
and the strategy-growth gate.

## Scope

The shared interface is intentionally narrow:

1. A model-family runner produces an evidence sweep summary.
2. The summary is converted into a strategy-growth evidence artifact.
3. The strategy-growth gate evaluates the artifact with the common rubric.

This does not abstract model training, checkpoint loading, rollout mechanics, or
physics residual computation. Those remain model-family specific for now.

## Required Sweep Summary Contract

Any model family that wants to enter the strategy-growth gate should produce a
JSON object with:

```json
{
  "schema_version": 1,
  "state_hint": "DEEPONET_EVIDENCE_SWEEP_COMPLETED",
  "inputs": {"forecast_steps": 31},
  "comparison_winners": [
    {
      "start_index": 0,
      "winner_experiment_id": "candidate_000",
      "winner_training_strategy": "branch_trunk_rollout",
      "winner_score": 31.5,
      "winner_score_delta": 1.5
    }
  ],
  "comparison_count": 1,
  "record_count": 2,
  "physics_residual_delta": 0.0,
  "inference_time_regression": 0.0
}
```

Required fields:

- `state_hint`: Must match the model-family source state passed to the converter.
- `inputs.forecast_steps`: Used to decide whether evidence counts as long-window.
- `comparison_winners`: Non-empty list of per-window winners.
- `comparison_winners[].start_index`: Defines distinct evaluation windows.
- `comparison_winners[].winner_score_delta`: Numeric score delta of winner vs baseline.

Optional fields:

- `physics_residual_delta`: Defaults to `0.0` with provenance note when missing.
- `inference_time_regression`: Defaults to `0.0` with provenance note when missing.

## Conversion Command

Generic command:

```bash
uv run --frozen pde-agent convert-operator-evidence-summary \
  --sweep-summary-path /tmp/deeponet_evidence_summary.json \
  --output-path /tmp/strategy_evidence_summary.json \
  --model-family DeepONet \
  --source-state-hint DEEPONET_EVIDENCE_SWEEP_COMPLETED \
  --long-window-threshold 31
```

FNO compatibility command:

```bash
uv run --frozen pde-agent convert-fno-evidence-summary \
  --sweep-summary-path /tmp/fno_evidence_summary.json \
  --output-path /tmp/strategy_evidence_summary.json \
  --long-window-threshold 31
```

The FNO command is now a thin wrapper over the generic converter and keeps the
existing `source_fno_evidence_summary_path` artifact field for compatibility.

## Why This Boundary Comes First

DeepONet and PI-DeepONet will differ from FNO in architecture and training
mechanics. PI-DeepONet also changes the evidence semantics by adding PDE
residual and collocation-point choices. The stable shared part is not training;
it is the final evidence shape needed by the agent's strategy-growth gate.

This boundary lets new model families participate in the same promotion,
revision, and rejection protocol without forcing premature backend unification.
