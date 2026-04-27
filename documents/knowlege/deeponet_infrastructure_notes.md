# DeepONet Infrastructure Notes

This document records the current DeepONet integration boundary.

## Current Status

DeepONet is not yet implemented as a trainable backend in this harness. The
first committed layer is the evidence interface:

1. DeepONet candidate evaluation reports are collected.
2. Reports are grouped by `(start_index, train_steps)`.
3. The highest-score candidate in each group becomes a comparison winner.
4. The resulting `DEEPONET_EVIDENCE_SWEEP_COMPLETED` summary can be converted
   by `convert-operator-evidence-summary`.
5. The converted artifact can enter the existing strategy-growth gate.

This mirrors the stable part of the FNO loop without copying FNO-specific
checkpoint, rollout, or fine-tuning code.

## Candidate Evaluation Report Contract

Each DeepONet candidate evaluation report must be a JSON object:

```json
{
  "schema_version": 1,
  "state_hint": "DEEPONET_CANDIDATE_EVALUATED",
  "inputs": {"forecast_steps": 31},
  "candidate": {
    "experiment_id": "deeponet_candidate_000",
    "start_index": 0,
    "train_steps": 1,
    "training_strategy": "branch_trunk_rollout"
  },
  "baseline_comparison": {
    "baseline_score": 40.0,
    "score_delta": 1.0,
    "improved": true
  },
  "score": {
    "competition_like_score_short": 41.0,
    "rel_mse": 0.2,
    "rmse": 0.3
  },
  "error_diagnosis": {
    "dominant_error_pattern": "long_horizon_drift"
  },
  "physics": {
    "physics_issue": "inconclusive"
  }
}
```

## Commands

Build a DeepONet evidence sweep summary:

```bash
uv run --frozen pde-agent build-deeponet-evidence-sweep \
  --evaluation-report-path /tmp/deeponet_candidate_000.json \
  --evaluation-report-path /tmp/deeponet_candidate_001.json \
  --output-dir /tmp/deeponet_evidence
```

Convert it into generic strategy-growth evidence:

```bash
uv run --frozen pde-agent convert-operator-evidence-summary \
  --sweep-summary-path /tmp/deeponet_evidence/evidence_summary.json \
  --output-path /tmp/strategy_evidence_summary.json \
  --model-family DeepONet \
  --source-state-hint DEEPONET_EVIDENCE_SWEEP_COMPLETED \
  --long-window-threshold 31
```

## Next Development Step

The next useful layer is a small DeepONet candidate-evaluation producer. It can
initially be synthetic or lightweight, but it should emit the report contract
above. Only after this report-producing layer is stable should we add real
torch-based DeepONet training.
