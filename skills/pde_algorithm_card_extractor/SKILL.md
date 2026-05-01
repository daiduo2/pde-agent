---
name: pde-algorithm-card-extractor
description: Use when converting PDE literature extraction context artifacts into grounded AlgorithmCard v2 JSON for Hypothesis Lab. Trigger on LITERATURE_EXTRACTION_CONTEXT_PREPARED, agent-extract-algorithm-cards, neural operator paper extraction, or PDE literature-to-algorithm work.
---

# PDE AlgorithmCard Extractor

Use this skill after `prepare-literature-extraction-context` emits a `LITERATURE_EXTRACTION_CONTEXT_PREPARED` artifact.

The task is not keyword extraction. Use your scientific and neural-operator understanding to identify actionable algorithms, but keep every claim grounded in the provided context units.

## Workflow

1. Read the context artifact JSON.
2. Inspect `context_units`; prioritize method, architecture, loss, training, inference, algorithm, theorem-to-method, and experiment sections.
3. Extract only mechanisms that can plausibly become a PDE neural-operator algorithm, strategy, executor, loss, rollout rule, postprocess, checkpoint adapter, or evaluation procedure.
4. Emit one JSON object with `state_hint: "ALGORITHM_CARDS_EXTRACTED"`.
5. If no actionable algorithm exists, emit an empty extraction report with `extraction_status: "empty"` and `failure_code: "no_algorithmic_claim"`.

## Grounding Rules

- Every card must include `source_ids` from the context artifact.
- Every substantive field must be traceable to `evidence_spans` with `unit_id`, `section_title`, `line_start`, and `line_end`.
- You may use domain expertise to interpret a mechanism, but do not invent claims absent from the context.
- Preserve uncertainty in `failure_modes` and `minimal_experiment`; do not overstate readiness.
- Prefer existing hooks from `agent_contract.allowed_existing_hooks` when they fit.
- Use a new descriptive hook only when the mechanism needs executor design before sandbox testing.

## Output Shape

Return JSON only:

```json
{
  "schema_version": 1,
  "state_hint": "ALGORITHM_CARDS_EXTRACTED",
  "algorithm_cards_path": "<requested output path or null>",
  "algorithm_card_count": 1,
  "source_ids": ["source_id"],
  "extraction_status": "extracted",
  "algorithm_cards": [
    {
      "algorithm_card_id": "card_<source_id>_<short_hook>",
      "source_ids": ["source_id"],
      "claim": "Grounded actionable claim.",
      "core_mechanism": "Mechanism in implementation terms.",
      "operation": "What the algorithm does to data/model/training/inference.",
      "preconditions": ["When this mechanism is applicable."],
      "implementation_hooks": ["fourier_neural_operator_architecture"],
      "expected_observables": ["Metric or physical observable expected to improve."],
      "failure_modes": ["Likely failure or invalidation mode."],
      "minimal_experiment": "Small sandbox experiment that can validate or reject it.",
      "evidence_spans": {
        "claim": {
          "unit_id": "section_001_method",
          "section_title": "Method",
          "line_start": 10,
          "line_end": 20
        }
      },
      "evidence_status": "extracted",
      "operator_target": "ic_to_trajectory",
      "mechanism_type": "architecture",
      "mathematical_object": "fourier_modes",
      "implementation_requirements": {
        "requires_training": true,
        "requires_checkpoint": false,
        "requires_target_data": true,
        "requires_new_executor": true,
        "requires_external_code": false,
        "can_run_as_postprocess": false
      }
    }
  ],
  "extraction_audit": {
    "mode": "agent_skill",
    "context_unit_count": 0,
    "selected_unit_ids": [],
    "notes": []
  },
  "next_command": "validate-algorithm-cards-v2"
}
```

Allowed v2 values:

- `operator_target`: `ic_to_trajectory`, `coefficient_to_solution`, `residual_to_correction`, `context_to_rollout`, `postprocess_to_solution`.
- `mechanism_type`: `architecture`, `loss`, `rollout`, `postprocess`, `conditioning`, `checkpoint`, `hybrid`.
- `evidence_status`: `extracted`.

## Empty Output

If the context does not support an algorithm card:

```json
{
  "schema_version": 1,
  "state_hint": "ALGORITHM_CARDS_EXTRACTED",
  "algorithm_cards_path": "<requested output path or null>",
  "algorithm_card_count": 0,
  "source_ids": ["source_id"],
  "extraction_status": "empty",
  "failure_code": "no_algorithmic_claim",
  "algorithm_cards": [],
  "extraction_audit": {
    "mode": "agent_skill",
    "context_unit_count": 0,
    "selected_unit_ids": [],
    "notes": ["No grounded actionable PDE algorithm found."]
  },
  "next_command": "update-literature-registry"
}
```
