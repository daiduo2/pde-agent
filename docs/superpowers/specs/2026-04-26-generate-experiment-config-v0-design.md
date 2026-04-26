# Generate Experiment Config V0 Design

Date: 2026-04-26

## Purpose

Add the second iteration-optimization skill for the PDE agent harness:
`generate-experiment-config`.

The previous skill, `plan-next-experiment`, converts diagnostic artifacts into a
bounded next-experiment plan. This phase converts that plan into a concrete,
machine-readable experiment configuration that downstream agents or commands can
execute later.

This phase completes the non-executing planning chain:

```text
select-pde-hypothesis
  -> plan-next-experiment
  -> generate-experiment-config
```

The command must not run experiments. It only materializes candidate experiment
configurations from a plan artifact.

## Non-Goals

- Do not implement `apply-low-pass-postprocess`.
- Do not train, finetune, or infer with any model.
- Do not modify model code automatically.
- Do not execute candidate experiments.
- Do not generate markdown research notes.
- Do not add PyTorch, NumPy, HDF5, FNO, UNet, DeepONet, or PI-DeepONet
  dependencies.
- Do not implement a general-purpose AutoML search engine.

## Design Position

`generate-experiment-config` is a protocol-stabilization step.

The planner says what kind of intervention should happen and gives a bounded
candidate parameter space. The config generator turns that into explicit
candidate experiment records:

- stable experiment IDs,
- selected intervention,
- change scope,
- candidate parameter values,
- budget inherited from the plan,
- controls,
- acceptance and rejection criteria,
- required output artifacts.

This keeps Hermes/Kimi 2.6 from inventing ad hoc experiment structures. Later
skills can consume the generated config and decide whether they know how to
execute a given intervention.

## Command

### `generate-experiment-config`

Inputs:

- `--plan-path`: JSON artifact produced by `plan-next-experiment`
- `--output-path`: output experiment configuration JSON

Output payload:

- `run_id`: `local`
- `command`: `generate-experiment-config`
- `state_hint`: `EXPERIMENT_CONFIG_GENERATED`
- artifact: `experiment_config_path`
- metrics:
  - `selected_intervention`
  - `candidate_count`
  - `requires_training`
  - `change_scope`
- `next_hint`: `execute-candidate-experiment`

The `next_hint` names a future execution step. This phase does not implement
that command.

## Core Artifact Schema

The command writes `experiment_config.json`:

```json
{
  "schema_version": 1,
  "state_hint": "EXPERIMENT_CONFIG_GENERATED",
  "source_plan_path": "runs/local_spectral_next_experiment_plan.json",
  "selected_intervention": "apply_low_pass_stabilization",
  "change_scope": "postprocess_or_inference_only",
  "requires_training": false,
  "budget": {
    "max_experiment_count": 3,
    "max_train_time_seconds": 0,
    "max_inference_time_seconds": 120,
    "requires_training": false
  },
  "candidate_experiments": [
    {
      "experiment_id": "apply_low_pass_stabilization_001",
      "intervention": "apply_low_pass_stabilization",
      "change_scope": "postprocess_or_inference_only",
      "requires_training": false,
      "parameters": {
        "low_pass_cutoff_ratio": 0.5,
        "preserve_mean": true
      },
      "expected_artifacts": [
        "candidate prediction NPZ",
        "candidate score JSON",
        "candidate rollout error diagnosis JSON",
        "candidate Burgers physics check JSON",
        "candidate selected hypothesis JSON"
      ],
      "forbidden_actions": [
        "change training data split",
        "submit without baseline comparison",
        "overwrite previous artifacts"
      ]
    }
  ],
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
  "next_command": "execute-candidate-experiment"
}
```

The exact values depend on the input plan. The schema shape should remain stable.

## Candidate Generation

Read candidate parameters from:

```text
plan["intervention_plan"]["candidate_parameters"]
```

Each candidate parameter field must be a non-empty list of JSON primitive values:

- string,
- number,
- boolean,
- null.

The generator performs deterministic Cartesian expansion in sorted key order,
then truncates to:

```text
plan["budget"]["max_experiment_count"]
```

Example:

```json
{
  "low_pass_cutoff_ratio": [0.5, 0.625, 0.75],
  "preserve_mean": [true]
}
```

with `max_experiment_count = 3` generates:

```json
[
  {
    "experiment_id": "apply_low_pass_stabilization_001",
    "parameters": {
      "low_pass_cutoff_ratio": 0.5,
      "preserve_mean": true
    }
  },
  {
    "experiment_id": "apply_low_pass_stabilization_002",
    "parameters": {
      "low_pass_cutoff_ratio": 0.625,
      "preserve_mean": true
    }
  },
  {
    "experiment_id": "apply_low_pass_stabilization_003",
    "parameters": {
      "low_pass_cutoff_ratio": 0.75,
      "preserve_mean": true
    }
  }
]
```

## Required Input Fields

The plan JSON must be an object and include:

- `selected_intervention`: string
- `change_scope`: string
- `intervention_plan`: object
- `intervention_plan.candidate_parameters`: object
- `budget`: object
- `budget.max_experiment_count`: positive integer
- `budget.requires_training`: boolean
- `controls`: object
- `acceptance_criteria`: list
- `rejection_criteria`: list
- `required_next_artifacts`: list

Optional fields copied when present:

- `source_hypothesis`
- `failure_mode`
- `rationale`

## Error Handling

The core generator raises `ExperimentConfigError` when:

- `plan_path` is missing,
- `plan_path` is not a file,
- plan JSON is invalid,
- plan JSON root is not an object,
- required fields are absent or wrong type,
- `candidate_parameters` is empty,
- any parameter value list is empty,
- a parameter value is not JSON-primitive,
- `max_experiment_count` is not a positive integer.

The command should fail fast and should not write a partial output artifact.

## Tests

Add tests covering:

1. `test_generate_experiment_config_expands_low_pass_candidates`
   - writes a low-pass next-experiment plan,
   - calls core generator,
   - asserts three candidate experiments,
   - asserts deterministic IDs and parameter order,
   - asserts `requires_training` is false.

2. `test_generate_experiment_config_respects_budget_limit`
   - plan has more parameter combinations than budget,
   - asserts candidate count is capped by `max_experiment_count`.

3. `test_generate_experiment_config_cli_writes_payload`
   - invokes `pde-agent generate-experiment-config`,
   - asserts payload state hint, artifact path, metrics, and file existence.

4. `test_generate_experiment_config_rejects_empty_candidate_parameters`
   - plan has empty candidate parameter object or empty parameter value list,
   - asserts `ExperimentConfigError`.

The tests must not require HDF5, NumPy, PyTorch, or model training.

## Future Phase

The next implementation phase can add the first executable intervention:
`apply-low-pass-postprocess`.

That future command should consume `experiment_config.json`, select a candidate
experiment, apply the candidate transformation, and produce a prediction NPZ for
rescoring. It is intentionally out of scope here.

## Success Criteria

- `generate-experiment-config` is available from the CLI.
- It consumes `next_experiment_plan.json` and writes deterministic
  `experiment_config.json`.
- It emits `state_hint = EXPERIMENT_CONFIG_GENERATED`.
- It expands candidate parameters deterministically and respects the experiment
  budget.
- It does not execute or modify experiments.
- Existing tests continue to pass.
