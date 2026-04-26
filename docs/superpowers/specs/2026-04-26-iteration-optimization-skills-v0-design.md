# Iteration Optimization Skills V0 Design

Date: 2026-04-26

## Purpose

Add the first experiment-iteration skill for the PDE agent harness.

The current harness can train a lightweight local model, predict a local rollout,
score it, diagnose rollout error, check Burgers physics proxies, and select a PDE
hypothesis. The missing layer is a deterministic planner that converts those
diagnostic artifacts into the next experiment decision.

This phase introduces `plan-next-experiment`:

```text
score-local-rollout
  -> diagnose-rollout-error
  -> check-burgers-physics-local
  -> select-pde-hypothesis
  -> plan-next-experiment
```

The output is machine-readable JSON only. Human-readable experiment narratives
can be generated later by another review agent from the JSON artifact.

## Non-Goals

- Do not implement FNO, UNet, DeepONet, or PI-DeepONet backends in this phase.
- Do not modify model code automatically.
- Do not run experiments automatically.
- Do not generate markdown research notes in this command.
- Do not implement hyperparameter grid execution in this phase.
- Do not depend on PyTorch or real HDF5 files.
- Do not let the LLM freely invent unconstrained experiment actions.

## Design Position

This skill should be a bounded scientific decision layer, not an AutoML system.

The planner consumes evidence that already exists:

- local rollout score,
- rollout error diagnosis,
- Burgers physics check,
- selected PDE hypothesis.

It emits an explicit next-experiment plan with:

- the failure mode being targeted,
- the recommended intervention,
- the allowed change scope,
- the experiment budget,
- acceptance and rejection criteria,
- required control baseline,
- required artifacts for the next round.

The plan must be deterministic for a given set of inputs. This matters because
Hermes/Kimi 2.6 needs a narrow action surface; the harness should guide it toward
scientific iteration rather than open-ended code exploration.

## Command

### `plan-next-experiment`

Inputs:

- `--score-path`: JSON produced by `score-local-rollout`
- `--error-diagnosis-path`: JSON produced by `diagnose-rollout-error`
- `--physics-check-path`: JSON produced by `check-burgers-physics-local`
- `--hypothesis-path`: JSON produced by `select-pde-hypothesis`
- `--output-path`: JSON artifact to write

Output payload:

- `run_id`: `local`
- `command`: `plan-next-experiment`
- `state_hint`: `NEXT_EXPERIMENT_PLANNED`
- artifact: `next_experiment_plan_path`
- metrics:
  - `selected_intervention`
  - `change_scope`
  - `max_experiment_count`
  - `stop_condition`
- `next_hint`: `generate-experiment-config`

## Core Artifact Schema

The planner writes `next_experiment_plan.json` with this schema:

```json
{
  "schema_version": 1,
  "state_hint": "NEXT_EXPERIMENT_PLANNED",
  "selected_intervention": "apply_low_pass_stabilization",
  "source_hypothesis": "apply_low_pass_stabilization",
  "failure_mode": {
    "dominant_error_pattern": "shock_region_error",
    "physics_issue": "nonphysical_oscillation",
    "score_band": "medium",
    "evidence": {
      "competition_like_score_short": 77.18,
      "rel_mse": 0.0086,
      "rmse": 0.0506,
      "residual_ratio": 0.9077,
      "total_variation_ratio": 2.7445
    }
  },
  "change_scope": "postprocess_or_inference_only",
  "intervention_plan": {
    "description": "Sweep low-pass thresholds to suppress nonphysical oscillations while auditing shock-region error.",
    "allowed_actions": [
      "add deterministic low-pass postprocessing",
      "sweep spectral cutoff",
      "compare against unchanged prediction"
    ],
    "forbidden_actions": [
      "change training data split",
      "submit without baseline comparison",
      "overwrite previous artifacts"
    ],
    "candidate_parameters": {
      "low_pass_cutoff_ratio": [0.5, 0.625, 0.75],
      "preserve_mean": [true]
    }
  },
  "budget": {
    "max_experiment_count": 3,
    "max_train_time_seconds": 0,
    "max_inference_time_seconds": 120,
    "requires_training": false
  },
  "controls": {
    "baseline_required": true,
    "baseline_artifacts": [
      "previous prediction NPZ",
      "previous score JSON",
      "previous physics check JSON"
    ]
  },
  "acceptance_criteria": [
    "competition_like_score_short improves by at least 1.0",
    "total_variation_ratio moves closer to 1.0",
    "shock-region error does not worsen by more than 5%"
  ],
  "rejection_criteria": [
    "score decreases by more than 0.5",
    "physics issue changes to overdissipation",
    "prediction contains non-finite values"
  ],
  "required_next_artifacts": [
    "candidate prediction NPZ",
    "candidate score JSON",
    "candidate rollout error diagnosis JSON",
    "candidate Burgers physics check JSON",
    "candidate selected hypothesis JSON"
  ],
  "next_command": "generate-experiment-config",
  "rationale": "Selected because the previous hypothesis and physics check indicate nonphysical oscillation."
}
```

The exact numerical values above are examples. The implemented output must follow
the same shape and use JSON-serializable primitive values only.

## Intervention Policy

The first version supports a small action table keyed by the selected PDE
hypothesis and diagnostic labels.

### `apply_low_pass_stabilization`

Use when `physics_issue` is `nonphysical_oscillation` or the hypothesis is
`apply_low_pass_stabilization`.

- `change_scope`: `postprocess_or_inference_only`
- `requires_training`: `false`
- candidate parameters:
  - `low_pass_cutoff_ratio`: `[0.5, 0.625, 0.75]`
  - `preserve_mean`: `[true]`
- stop condition:
  - accept if score improves and total variation ratio approaches 1.0

### `add_gradient_weighted_loss`

Use when the dominant error is `shock_region_error` and the physics issue is not
primarily nonphysical oscillation.

- `change_scope`: `training_loss_only`
- `requires_training`: `true`
- candidate parameters:
  - `gradient_weight`: `[0.5, 1.0, 2.0]`
  - `shock_quantile`: `[0.85, 0.9]`
- stop condition:
  - accept if shock-region evidence improves without broad score regression

### `use_scheduled_sampling_rollout_finetune`

Use when the dominant error is `long_horizon_drift`.

- `change_scope`: `training_schedule_only`
- `requires_training`: `true`
- candidate parameters:
  - `teacher_forcing_decay`: `[0.95, 0.9, 0.85]`
  - `rollout_steps`: `[10, 20, 31]`
- stop condition:
  - accept if late error improves without early error collapse

### `add_spectral_consistency_loss`

Use when the rollout is overdamped or high-frequency energy is excessively
suppressed.

- `change_scope`: `training_loss_only`
- `requires_training`: `true`
- candidate parameters:
  - `spectral_loss_weight`: `[0.01, 0.05, 0.1]`
  - `spectral_band`: `["mid", "high"]`
- stop condition:
  - accept if spectral energy alignment improves without oscillation growth

### `add_residual_correction_head`

Use when the physics check indicates residual mismatch without a simpler
postprocessing intervention.

- `change_scope`: `model_head_only`
- `requires_training`: `true`
- candidate parameters:
  - `residual_head_width`: `[16, 32]`
  - `residual_loss_weight`: `[0.01, 0.05]`
- stop condition:
  - accept if residual ratio improves and score does not regress

### `keep_baseline_and_collect_stronger_evidence`

Use when diagnostics are inconclusive or the local score is already near-perfect.

- `change_scope`: `data_diagnostic_only`
- `requires_training`: `false`
- candidate parameters:
  - `sample_limit`: `[16, 32, 64]`
  - `start_index`: `[0, 128, 512]`
- stop condition:
  - accept if the repeated diagnostics identify a stable failure mode

## Score Bands

The planner should classify local short-window score into simple bands:

- `excellent`: `competition_like_score_short >= 95`
- `high`: `85 <= score < 95`
- `medium`: `60 <= score < 85`
- `low`: `score < 60`
- `unknown`: score is absent or non-numeric

The score band affects budget but should not override hard physics warnings.

Default budget:

- postprocess-only interventions: at most 3 experiments, no training time
- training-loss or schedule interventions: at most 2 experiments
- data-diagnostic interventions: at most 3 diagnostic reruns

## Error Handling

The core planner must raise a dedicated `NextExperimentPlanError` when:

- an input file is missing,
- an input file is not valid JSON,
- an input JSON root is not an object,
- required string fields are absent:
  - `selected_hypothesis` from hypothesis JSON,
  - `dominant_error_pattern` from error diagnosis JSON,
  - `physics_issue` from physics check JSON,

The command wrapper should follow the existing pattern and let test-visible
exceptions fail fast. It should not silently write partial plans.

Like the current diagnostic JSON commands, v0 writes to the requested
`output_path` and creates parent directories as needed. It does not expose a
`--force` option.

## Tests

Add a dedicated test module or extend the PDE diagnostic skill tests with:

1. `test_plan_next_experiment_low_pass_for_nonphysical_oscillation`
   - score: medium,
   - error: shock-region or oscillatory error,
   - physics: nonphysical oscillation,
   - hypothesis: apply low-pass stabilization,
   - assert JSON plan selects `apply_low_pass_stabilization`,
   - assert `requires_training` is false,
   - assert state hint is `NEXT_EXPERIMENT_PLANNED`.

2. `test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune`
   - dominant error: `long_horizon_drift`,
   - assert scheduled sampling intervention and training schedule scope.

3. `test_plan_next_experiment_cli_writes_payload`
   - invoke `pde-agent plan-next-experiment`,
   - assert payload state hint, artifact path, metrics, and file existence.

4. `test_plan_next_experiment_rejects_missing_required_fields`
   - omit `selected_hypothesis` or `physics_issue`,
   - assert `NextExperimentPlanError`.

The tests must not require HDF5, PyTorch, or real model training.

## Future Phase: `generate-experiment-config`

The next phase should consume `next_experiment_plan.json` and emit a concrete
machine-readable experiment configuration. That command should be implemented
only after this planner is stable.

Expected future command:

```text
plan-next-experiment
  -> generate-experiment-config
  -> run candidate experiment
  -> score-local-rollout
```

The current design intentionally keeps `generate-experiment-config` out of scope
so that the planner schema can stabilize first.

## Success Criteria

- `plan-next-experiment` is available from the CLI.
- It writes deterministic JSON with `state_hint = NEXT_EXPERIMENT_PLANNED`.
- It emits only machine-readable output.
- It maps the current spectral smoke result toward low-pass stabilization.
- It provides clear budget, controls, acceptance criteria, rejection criteria,
  and required next artifacts.
- Existing tests continue to pass.
