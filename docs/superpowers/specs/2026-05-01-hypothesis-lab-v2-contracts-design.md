# Hypothesis Lab V2 Contracts Design

## Context

Hypothesis Lab has crossed the first useful threshold: it can turn local
strategy knowledge and literature-derived candidates into controlled Burgers
sandbox experiments, record registry evidence, and guide Hermes away from
known bad repetitions. The current weakness is not lack of commands. The
weakness is that artifact semantics are still too loose for long-term
autonomous scientific iteration.

The next step is to treat Hypothesis Lab as an auditable algorithm
reproduction protocol. A paper, note, checkpoint, or local run should move
through explicit contracts before Hermes is allowed to internalize it as a
strategy. Failed attempts must be first-class records because they are the
main defense against cyclic search.

This design is PDE-focused. It should serve Burgers/PDEBench and neural
operator work first, while keeping the contracts clean enough to extract later
into a standalone literature-to-algorithm project.

## Goals

1. Define stricter artifact contracts for Hypothesis Lab V2.
2. Make `AlgorithmCard` a semi-formal algorithm statement, not a prose note.
3. Define a promotion rubric that uses score, runtime, physical diagnostics,
   novelty, cost, and reproducibility.
4. Standardize failure taxonomy and registry writeback.
5. Define the graph model around algorithm/strategy mechanisms rather than
   paper similarity.
6. Prepare, but not yet fully implement, a bounded blueprint DSL for executable
   algorithm reproduction.

## Non-Goals

- Do not build a generic paper-to-code system.
- Do not execute external repository code.
- Do not let LLM confidence or citation counts promote candidates.
- Do not require all old artifacts to be migrated immediately.
- Do not replace Hermes or the existing strategy-growth loop.
- Do not make the registry graph centered on papers as semantic nodes.

## Design Choice

Use staged contracts with explicit state transitions:

```text
SourceRecord
  -> ParsedDocument
  -> AlgorithmCardV2
  -> AlgorithmCandidate
  -> ExecutableBlueprint
  -> SandboxEvidence
  -> InternalizedStrategy
```

Each transition must either produce a valid next artifact or produce a
structured failure event. Hermes should consume summaries derived from these
contracts, not raw paper text.

## Artifact Contracts

### SourceRecord

Question answered: where did this source come from, and do we have a durable
local copy?

Required fields:

- `source_id`
- `source_type`: `openalex_work`, `arxiv_paper`, `github_repo`,
  `competition_resource`, or `manual_source`
- `title`
- `url`
- `retrieval_query`
- `retrieved_at`
- `license_note`
- `local_raw_path`
- `content_hash`
- `download_status`: `seen`, `downloaded`, `download_failed`
- `related_source_ids`

Notes:

- A failed download should still be recorded.
- `content_hash` is required once `download_status == downloaded`.
- Source records are provenance. They are not the center of the future graph.

### ParsedDocument

Question answered: can Hypothesis Lab reliably inspect the source text?

Required fields:

- `parse_id`
- `source_id`
- `parser`: initially `mineru`
- `parser_version`
- `markdown_path`
- `asset_dir`
- `parse_status`: `usable`, `partial`, `failed`
- `parse_quality`
- `section_index`
- `formula_count`
- `figure_count`
- `table_count`
- `quality_flags`

Quality flags:

- `missing_method_section`
- `missing_equation_context`
- `low_formula_density`
- `garbled_markdown`
- `empty_or_too_short`
- `asset_download_incomplete`

Rules:

- `partial` parses may produce cards, but every derived card must carry a
  `parse_quality_warning`.
- `failed` parses must write a registry event and stop.

### AlgorithmCardV2

Question answered: what algorithmic mechanism is claimed, and what would make
it experimentally meaningful for PDEBench?

Required fields:

- `algorithm_card_id`
- `source_ids`
- `claim`
- `operator_target`: `ic_to_trajectory`, `coefficient_to_solution`,
  `residual_to_correction`, `context_to_rollout`, or `postprocess_to_solution`
- `mechanism_type`: `architecture`, `loss`, `rollout`, `postprocess`,
  `conditioning`, `checkpoint`, or `hybrid`
- `mathematical_object`: examples include `fourier_modes`, `kernel_operator`,
  `burgers_residual`, `viscosity`, `time_integrator`, `spectral_filter`
- `operation`
- `implementation_hooks`
- `implementation_requirements`
- `preconditions`
- `expected_observables`
- `failure_modes`
- `minimal_experiment`
- `evidence_status`: `extracted`, `candidate_compiled`, `sandbox_planned`,
  `tested`, `promoted`, `rejected`, or `blocked`
- `provenance`

Implementation requirements:

- `requires_training`
- `requires_checkpoint`
- `requires_target_data`
- `requires_new_executor`
- `requires_external_code`
- `can_run_as_postprocess`

Rules:

- `operation` must be actionable. It must describe what Hermes should do, not
  merely why the paper is important.
- `implementation_hooks` can contain multiple hooks. Candidate compilation
  should expand them into distinct families.
- Cards requiring external code are allowed but must compile to a blocked
  candidate until an approved executor exists.

Validation entrypoint:

- CLI command: `validate-algorithm-cards-v2`
- Input artifact: `ALGORITHM_CARDS_EXTRACTED`
- Output artifact state: `ALGORITHM_CARDS_V2_VALIDATED`
- Approved cards route to `compile-algorithm-candidates`.
- Rejected cards route to `fix-algorithm-card-v2-schema`.

Validator issue records must include:

- `code`: machine-readable issue type, for example `missing_operation` or
  `invalid_operator_target`
- `path`: JSON path-like field location
- `failure_code`: stable recovery class
- `message`: short human-readable explanation

Important failure codes:

- `no_algorithmic_claim`: the card does not contain an actionable operation.
- `candidate_schema_invalid`: the card violates structural, enum, list, bool,
  provenance, or implementation requirement constraints.

### AlgorithmCandidate

Question answered: how does a card become a PDEBench/Burgers experiment?

Required fields:

- `algorithm_candidate_id`
- `parent_algorithm_card_id`
- `candidate_family`
- `target_problem`
- `target_task`
- `model_family`
- `hypothesis`
- `mechanism`
- `literature_hook`
- `normalized_hook`
- `executor_spec`
- `parameter_grid`
- `benchmark_plan`
- `ablation_plan`
- `promotion_criteria`
- `risk_controls`
- `source_grounding`

Candidate families in scope:

- `official_checkpoint_family`
- `lightweight_residual_corrector`
- `spectral_rollout_stabilization`
- `physics_residual_regularization`
- `scheduled_rollout_training`
- `operator_correction_postprocess`
- `local_strategy_skill`
- `unsupported`

Rules:

- Candidate families are normalized. Different paper phrases such as
  `scheduled_sampling` and `rollout_training` should map to the same family.
- Each family must declare whether it is executable now, blocked by executor
  design, or requires controlled training.
- Candidate IDs should be stable under repeated extraction from the same card.

### ExecutableBlueprint

Question answered: what exactly will the sandbox execute?

Required fields:

- `blueprint_id`
- `source_candidate_id`
- `candidate_family`
- `executor_id`
- `baseline`
- `transforms`
- `training_plan`
- `parameter_grid`
- `metrics`
- `safety_policy`
- `expected_artifacts`

DSL shape:

```json
{
  "baseline": "persistence|fno_checkpoint|retrieval_prior",
  "transforms": [
    {"name": "spectral_smooth", "params": {"strength": [0.0, 0.5]}},
    {"name": "residual_fit", "params": {"shrinkage": [0.25, 0.5]}}
  ],
  "training_plan": {"mode": "none|finetune|supervised|physics_loss"},
  "metrics": ["score", "runtime", "burgers_residual", "hf_energy"],
  "safety_policy": {
    "preserve_context_steps": 10,
    "forbid_target_leakage": true,
    "forbid_external_code": true
  }
}
```

V2 does not need a full interpreter for all DSL features. It needs a schema and
enough validation that later executors can be added safely.

Validation entrypoint:

- CLI command: `validate-executable-blueprints`
- Input artifact: either native `EXECUTABLE_BLUEPRINTS_DESIGNED` or existing
  `HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED`
- Output artifact state: `EXECUTABLE_BLUEPRINTS_VALIDATED`
- Approved blueprints route to `run-blueprint-sandbox`.
- Rejected blueprints route to `fix-executable-blueprint-dsl`.

The validator supports two shapes:

- Native DSL blueprints with all `ExecutableBlueprint` required fields.
- Legacy Hypothesis Lab blueprints, normalized into the native DSL by deriving
  `executor_id`, `baseline`, `transforms`, `training_plan`, `metrics`,
  `safety_policy`, and `expected_artifacts`.

Allowed baselines:

- `persistence`
- `fno_checkpoint`
- `retrieval_prior`

Allowed transforms:

- `linear_trend`
- `spectral_smooth`
- `residual_fit`
- `retrieval_prior`
- `retrieval_refinement`
- `checkpoint_calibration`

Allowed training modes:

- `none`
- `finetune`
- `supervised`
- `physics_loss`

Allowed executors:

- `controlled_numpy_operator`
- `controlled_numpy_residual_operator`
- `controlled_numpy_retrieval_prior`
- `controlled_numpy_retrieval_refinement`
- `official_checkpoint_candidate_pipeline`

Safety rules:

- `preserve_context_steps >= 10`
- `forbid_target_leakage == true`
- `forbid_external_code == true`

The validator must produce machine-readable `issues` with `code`, `path`,
`failure_code`, and `message`. Unsafe target access maps to
`target_leakage_risk`; external code maps to `requires_external_code`; schema
violations map to `blueprint_schema_invalid`.

### SandboxEvidence

Question answered: what happened when the candidate was tested?

Required fields:

- `evidence_id`
- `candidate_id`
- `blueprint_id`
- `sandbox_id`
- `dataset_id`
- `rules_passed`
- `score`
- `segment_scores`
- `runtime`
- `physics_metrics`
- `ablation_metrics`
- `score_delta_vs_current_best`
- `artifact_paths`
- `reproducibility`
- `decision`

Decisions:

- `promote`
- `refine`
- `deprioritize`
- `reject`
- `blocked`

Rules:

- Evidence without `rules_passed == true` cannot promote.
- Evidence below current best should normally become `refine` or `deprioritize`
  rather than a repeated sandbox run.
- Evidence must be linked back to both candidate and source provenance.

### InternalizedStrategy

Question answered: has Hermes learned a reusable strategy?

Required fields:

- `strategy_id`
- `source_candidate_id`
- `evidence_ids`
- `skill_candidate_path`
- `promotion_report_path`
- `allowed_commands`
- `known_risks`
- `regression_tests`
- `ledger_entry_path`

Rules:

- A strategy requires at least one sandbox evidence artifact.
- A strategy should preserve negative evidence in `known_risks`.
- Promotion must not erase the source chain.

## Promotion Rubric

Every candidate should be judged with a structured rubric:

| Dimension | Question | Example Signal |
| --- | --- | --- |
| validity | Did rules pass and context remain intact? | `rules_passed`, context tolerance |
| score | Did task score improve? | `score_delta_vs_current_best` |
| runtime | Is inference under limits? | seconds, timeout flag |
| physics | Did residual or spectral drift improve? | Burgers residual, HF energy |
| novelty | Is this meaningfully different from tried families? | family and blueprint hash |
| cost | Does it need training or new executor work? | sandbox level, executor gap |
| reproducibility | Can it be rerun deterministically? | repeated score variance |

Rubric outcomes:

- `promote`: improves or matches target criteria with valid evidence.
- `refine`: plausible but under current best, needs variant or ablation.
- `deprioritize`: valid but weak, avoid repeating without new evidence.
- `reject`: violates rules or repeatedly regresses.
- `blocked`: needs parser, executor, data, or resource work.

Candidate promotion entrypoint:

- CLI command: `evaluate-algorithm-candidate-promotion`
- Input artifacts:
  - `ALGORITHM_CANDIDATES_COMPILED`
  - one or more `SANDBOX_EVIDENCE_RECORDED` artifacts
  - `current_best_score`
- Output artifact state: `ALGORITHM_CANDIDATE_PROMOTION_EVALUATED`

The evaluator selects the best matching evidence for the requested
`algorithm_candidate_id` and writes:

- `decision`
- `promote_to_strategy_growth`
- `passed_checks`
- `failed_checks`
- `reason_codes`
- `metric_results`
- `ledger_event`
- `next_command`

Initial deterministic checks:

- `executor`: blocked or unsupported candidates cannot promote.
- `validity`: evidence with `rules_passed != true` is rejected.
- `score`: `score_delta_vs_current_best` must satisfy the candidate's
  `promotion_criteria.min_score_delta`.
- `runtime`: inference seconds must stay under
  `promotion_criteria.max_inference_seconds`.
- `physics`: residual delta must not regress.
- `ablation`: required ablation evidence must improve score.
- `reproducibility`: evidence must be marked deterministic before promotion.

Routing:

- `promote` -> `record-strategy-proposal`
- `refine` -> `refine-algorithm-candidate`
- `blocked` -> `create-executor-design-request`
- `deprioritize` or `reject` -> `record-literature-registry-event`

Registry event entrypoint:

- CLI command: `record-literature-registry-event`
- Input artifact: currently `ALGORITHM_CANDIDATE_PROMOTION_EVALUATED`
- Output artifact state: `LITERATURE_REGISTRY_EVENT_RECORDED`
- Follow-up command: `update-literature-registry`

The registry event normalizes candidate decisions into stable event types:

- `promote` -> `candidate_promoted`
- `refine` -> `candidate_refined`
- `deprioritize` -> `candidate_deprioritized`
- `reject` -> `candidate_rejected`
- `blocked` -> `candidate_blocked`

The event artifact must preserve:

- `event_source_path`
- `algorithm_candidate_id`
- `candidate_family`
- `evidence_id`
- `decision`
- validated and de-duplicated `failure_codes`
- `recommended_next`
- `retry_allowed`
- `metric_results`

`update-literature-registry` must consume `LITERATURE_REGISTRY_EVENT_RECORDED`
artifacts and write the result back to the candidate node:

- candidate `status`
- `failure_history`
- linked registry event with `reason_codes`
- parent algorithm `strategy_status` when the algorithm node is known

## Failure Taxonomy

Failure codes should be shared across registry, Hermes summaries, and logs:

- `source_download_failed`
- `parse_failed`
- `parse_quality_insufficient`
- `no_algorithmic_claim`
- `unsupported_executor`
- `requires_external_code`
- `candidate_schema_invalid`
- `blueprint_schema_invalid`
- `sandbox_rule_failed`
- `context_mismatch`
- `target_leakage_risk`
- `score_regressed`
- `runtime_budget_exceeded`
- `physics_regressed`
- `ablation_missing_or_weak`
- `insufficient_reproducibility`
- `runtime_exceeded`
- `duplicate_family`
- `insufficient_evidence`
- `reproducibility_failed`

Each failure event should include:

- `failure_code`
- `artifact_path`
- `source_ids`
- `algorithm_card_id` when available
- `candidate_id` when available
- `recommended_next`
- `retry_allowed`: boolean

## Registry Graph Model

The registry graph should center algorithmic objects, not papers.

Node types:

- `Source`
- `Claim`
- `Mechanism`
- `Candidate`
- `Blueprint`
- `Evidence`
- `Strategy`

Edges:

- `source_supports_claim`
- `claim_instantiates_mechanism`
- `mechanism_generates_candidate`
- `candidate_executed_as_blueprint`
- `evidence_supports_candidate`
- `evidence_rejects_candidate`
- `candidate_promoted_to_strategy`
- `candidate_blocked_by_gap`

Graph principles:

- Papers are provenance nodes.
- Mechanisms and candidates are the central reusable units.
- Failed evidence remains connected and queryable.
- Future knowledge graphs should compare mechanisms by tested behavior and
  executor requirements, not by text embedding similarity alone.

## Hermes Consumption

Hermes should not read full registries by default. It should read a compact
summary containing:

- `source_backlog`
- `candidate_action_queue`
- `family_summary`
- `learning_signals`
- `avoid_repetition`
- `executor_gaps`
- `next_command`

Hermes decision rules:

1. Run candidates with no sandbox evidence first when executable.
2. Refine candidates with valid but underperforming evidence only when a new
   variant or ablation is proposed.
3. Do not rerun candidates listed in `avoid_repetition` unchanged.
4. Escalate `unsupported_executor` families to bounded executor design.
5. Promote only candidates with evidence that passes the rubric.

## Implementation Phases

### Phase 1: Schema And Validation

Add lightweight validators for:

- `AlgorithmCardV2`
- `AlgorithmCandidate`
- `ExecutableBlueprint`
- `SandboxEvidence`

Keep them in core modules and expose CLI audit commands. Do not migrate all
existing artifacts at once.

### Phase 2: Failure Events

Add shared failure taxonomy helpers and registry writeback. Update parser,
candidate compilation, sandbox planning, and Hermes summary to emit the same
failure codes.

### Phase 3: Promotion Rubric

Add a rubric artifact that consumes sandbox evidence and current-best score.
Use it to produce `promote`, `refine`, `deprioritize`, `reject`, or `blocked`.

### Phase 4: Blueprint DSL

Introduce the bounded DSL and validate existing Hypothesis Lab blueprints
against it. Add interpreters only for already-controlled executor families.

### Phase 5: Graph Export

Export registry graph edges in a stable JSONL form. This enables future graph
analysis without changing the operational registry format.

## Acceptance Criteria

- A literature-derived algorithm can be traced from source to evidence.
- A failed parse, unsupported executor, or weak sandbox result creates a
  structured failure event.
- Hermes summary can say what to run, what to refine, and what not to repeat.
- At least one multi-hook algorithm card expands into multiple normalized
  candidate families.
- Candidate promotion depends on sandbox evidence and rubric outcome, not
  textual plausibility.
- The registry can support future graph construction around mechanisms,
  candidates, evidence, and strategies.

## Recommended First Slice

The first implementation slice should be Phase 1 plus enough Phase 2 to make
validation failures visible in the registry. That gives immediate value without
forcing a full DSL interpreter.
