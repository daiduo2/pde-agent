# Directory Cleanup 2026-05-12

## Completion Audit

Objective: split large monoliths into clearer submodules, preserve existing
behavior, and document the resulting ownership boundaries.

Checklist:

- Large module ownership clarified: partially complete. Batch optimization,
  literature algorithm lab, Task1 short-window training, agent runtime task
  construction, part of literature expansion, and part of Hypothesis Lab have
  been split into named ownership modules.
- Existing public command/import surfaces preserved: partially complete. The
  split modules keep compatibility facades for old public functions where tests
  and commands still import the original module path.
- Behavior verified after each split: complete for the covered scope. The
  latest broad regression command was:
  `uv run pytest tests/test_agent_runtime.py tests/test_literature_algorithm_lab.py tests/test_task1_short_window_training.py tests/test_fno_training_closed_loop.py tests/test_fno_training_escalation.py tests/test_hypothesis_lab.py tests/test_hypothesis_lab_policy.py tests/test_manual_operator_tuning.py tests/test_real_hdf5_smoke_scripts.py tests/test_array_fixture_flow.py tests/test_strategy_growth.py tests/test_hypothesis_algorithm_designer.py tests/test_task1_short_window_corrector.py`
  and passed with `368 passed`.
- Generated cache artifacts removed after verification: complete for the last
  run. `__pycache__` directories under `src/`, `tests/`, and `scripts/` were
  removed after the regression run.
- Remaining large monoliths eliminated: not complete. `src/pde_agent/cli.py`,
  `core/literature_expansion.py`, and `core/hypothesis_lab.py` remain large
  enough to be explicit follow-up risks.

## What Changed

- `scripts/` was split by responsibility:
  - `scripts/audit/`
  - `scripts/data/`
  - `scripts/manual/`
  - `scripts/smoke/`
- The broken `scripts/run_task1_scalar_source_gate.py` entrypoint was removed.
  It imported a deleted core module and was no longer a valid runtime surface.
- The template `pde_agent.factorial` module and its test were removed because
  they were unrelated to the PDEAgent domain and polluted package ownership.
- The obsolete `core/hermes_literature_expansion.py` duplicate was removed.
  Active literature expansion is now `core/literature_expansion.py`; old Hermes
  naming should not be used for new autonomous workflows.
- General Burgers HDF5 loading was moved from the experiment-specific
  `manual_operator_tuning` module into `core/burgers_hdf5_io.py`.
- Current script and core callers now depend on the neutral I/O module instead
  of using `manual_operator_tuning` as a utility grab bag.
- Claude SDK task construction was split from the runtime executor module:
  - `core/agent_runtime_fno_task.py`
  - `core/agent_runtime_hypothesis_task.py`
  - `core/agent_runtime_literature_task.py`
  - `core/agent_runtime_promotion_task.py`
- `core/agent_runtime.py` is now the compatibility facade plus runtime
  execution/log-export/issue-recording logic, not the owner of every task
  prompt template.
- Literature source acquisition was split out of the large literature lab:
  - `core/literature_algorithm_common.py` owns shared artifact/path/text helpers
    and the common `LiteratureAlgorithmLabError`.
  - `core/literature_source_acquisition.py` owns OpenAlex/arXiv/GitHub search
    normalization, source document download, arXiv-to-MinerU pipeline control,
    MinerU submit/poll/download, and MinerU HTTP access.
  - `core/literature_algorithm_lab.py` keeps compatibility imports for the old
    public function names while no longer owning the source/MinerU I/O
    implementation.
- AlgorithmCard v2 validation was split out of the large literature lab:
  - `core/literature_algorithm_card_validation.py` owns AlgorithmCard v2
    schema/rubric validation and issue reporting.
  - `core/literature_algorithm_lab.py` keeps the old public
    `validate_algorithm_cards_v2` import path.
- Candidate compilation and sandbox planning were split out of the large
  literature lab:
  - `core/literature_candidate_compilation.py` owns AlgorithmCard-to-candidate
    compilation, unsupported-hook blocking, executor spec generation,
    candidate parameter grids, candidate IDs, and initial sandbox planning.
  - `core/literature_algorithm_lab.py` keeps the old public
    `compile_algorithm_candidates` and `plan_candidate_sandbox` import paths.
- Candidate promotion and evidence-rubric evaluation were split out of the
  large literature lab:
  - `core/literature_candidate_promotion.py` owns promotion checks, best
    evidence selection, target-reference leakage refinement routing,
    promote/refine/reject/blocked decisions, metric summaries, and next-command
    mapping.
  - `core/literature_algorithm_lab.py` keeps the old public
    `evaluate_algorithm_candidate_promotion` import path.
- Literature registry ownership was split out of the large literature lab:
  - `core/literature_registry.py` owns source-record construction, registry
    updates, registry-event materialization, source/algorithm/candidate node
    upserts, candidate status transitions, failure-code normalization, and
    registry finalization.
  - `core/literature_algorithm_lab.py` keeps the old public
    `build_literature_source_records`, `update_literature_registry`, and
    `record_literature_registry_event` import paths.
- AlgorithmCard extraction was split out of the large literature lab:
  - `core/literature_algorithm_card_extraction.py` owns MinerU parse ingestion,
    deterministic markdown section scanning, AlgorithmCard field inference,
    evidence span construction, AlgorithmCard v2 defaults, and extraction audit
    reporting.
  - `core/literature_algorithm_lab.py` keeps the old public
    `ingest_mineru_parse` and `extract_algorithm_cards` import paths.
- Executable Blueprint DSL validation was split out of the large literature lab:
  - `core/literature_blueprint_validation.py` owns executable-blueprint
    normalization, legacy Hypothesis Lab blueprint conversion, DSL validation,
    issue reporting, candidate-to-blueprint materialization, and first-candidate
    selection.
  - `core/literature_algorithm_lab.py` keeps the old public
    `validate_executable_blueprints` and `audit_executable_blueprint_payload`
    import paths.
- Bounded executor authoring and sandbox execution were split out of the large
  literature lab:
  - `core/literature_bounded_executor.py` owns bounded executor design specs,
    template materialization, static security audit, sandbox execution,
    sandbox event/evidence recording, authoring-task preparation/audit, and
    authoring completion promotion/registry writeback.
  - `core/literature_algorithm_lab.py` keeps the old public import paths for
    bounded executor commands.
- Local asset AlgorithmCandidate construction was split out of literature
  expansion:
  - `core/literature_local_asset_candidates.py` owns conversion from local
    strategy-library entries and local checkpoint paths into
    AlgorithmCandidate records, including checkpoint-family classification and
    local strategy hook selection.
  - `core/literature_expansion.py` keeps the old public
    `build_local_asset_algorithm_candidates` import path as a compatibility
    wrapper and remains responsible for expansion orchestration, sandbox queue
    execution, registry summaries, refinement, and promoted-strategy writeback.
- Literature promotion materialization was split out of literature expansion:
  - `core/literature_promotion_materialization.py` owns conversion from
    AlgorithmCandidate promotion artifacts to strategy proposals, strategy
    evidence summaries, and evidence-accumulation plans.
  - `core/literature_expansion.py` keeps the old public import paths for
    `record_promoted_algorithm_strategy_proposal`,
    `convert_candidate_promotion_to_strategy_evidence`, and
    `plan_candidate_evidence_accumulation`.
- Batch optimization objective logic was split from the batch controller:
  - `core/batch_optimization_common.py` owns the shared
    `BatchOptimizationLoopError`.
  - `core/batch_optimization_objective.py` owns judge score extraction,
    objective signal extraction, `auto` objective selection, objective score
    computation, submit-threshold checks, and terminal decision mapping.
  - `core/batch_optimization_memory.py` owns memory JSON loading, avoid-set
    construction, search-guidance extraction, postprocess neighbor selection,
    skip recording, and repair-policy action extraction.
  - `core/batch_optimization_loop.py` remains the orchestration module.
- Batch optimization candidate generation was split from the controller:
  - `core/batch_optimization_candidates.py` owns candidate-generator routing
    and the remaining lightweight candidate families: postprocess scheduling,
    runtime config search, fast baselines, hybrid gates, and linear
    trend/spectral schedules.
  - `core/batch_optimization_physics_candidates.py` owns checkpoint-window
    gates, SEM/pseudospectral window blends, and physics-residual nudges.
  - `core/batch_optimization_hypothesis_candidates.py` owns Hypothesis Lab
    blueprint runners, early/middle/late patch runners, source-patch
    materialization, reference audits, and local source prediction loading.
  - The public batch loop still accepts the same `candidate_generator` values;
    the split is internal ownership cleanup.
- Task1 short-window training model primitives were split from the training
  orchestrator:
  - `core/task1_short_window_common.py` owns shared state hints, supported
    family/loss/normalization constants, and `Task1ShortWindowTrainingError`.
  - `core/task1_short_window_models.py` owns trainable model wrappers,
    spectral convolution, model construction, loss functions, normalization,
    denormalization, and parameter counting.
  - `core/task1_short_window_training.py` remains responsible for training-loop
    orchestration, candidate planning, evidence emission, and policy feedback.
- Task1 short-window family selection policy was split from the training
  orchestrator:
  - `core/task1_short_window_family_policy.py` owns family record
    normalization, family ranking, selection reason codes, training directive
    selection, and next-generation constraints.
  - `core/task1_short_window_training.py` remains the compatibility caller for
    family-selection loop artifacts.
- Task1/Hypothesis Lab artifact adapters were split from the training
  orchestrator:
  - `core/task1_short_window_hypothesis_bridge.py` owns training-window bundle
    materialization for Hypothesis Lab sandboxing and conversion of Hypothesis
    Lab evidence into algorithm-candidate/sandbox-evidence promotion artifacts.
  - `core/task1_short_window_training.py` keeps the old public import paths for
    those bridge functions.
- Task1 FNO training policy was split from the training orchestrator:
  - `core/task1_fno_training_policy.py` owns FNO candidate record
    normalization, early/middle selection scoring, directive/action selection,
    evidence-feedback preferences, regression repair candidates, candidate
    rationale helpers, training constraints, graft summaries, and
    short-window metric/decision helpers.
  - `core/task1_short_window_training.py` keeps training execution, candidate
    plan materialization, optimization-loop orchestration, harness/iteration
    contracts, and compatibility entrypoints.
- Hypothesis Lab sandbox support was split from the main Hypothesis Lab facade:
  - `core/hypothesis_lab_sandbox_reference.py` owns sandbox reference loading,
    rank/shape validation, time/space reduction normalization, and
    target-reference leakage audit metadata.
  - `core/hypothesis_lab_selection.py` owns context-preservation checks,
    blueprint selection-objective normalization, candidate ranking keys, and
    selection-objective metadata.
  - `core/hypothesis_lab_candidate_params.py` owns candidate parameter
    construction for retrieval refinement, early refinement, early transition
    expert, and front-25 dynamics families.
  - `core/hypothesis_lab.py` keeps the old private helper names importable for
    tests and command code while no longer owning those implementations.
- Generated `__pycache__` directories under `src/`, `tests/`, and `scripts/`
  were removed.

## Current Scripts Layout

```text
scripts/
├── audit/
│   └── audit_fno_training_agent_session.py
├── data/
│   └── download_pdebench_file.py
├── manual/
│   ├── materialize_manual_operator_candidate.py
│   └── run_manual_operator_tuning.py
└── smoke/
    ├── prepare_array_fixture_smoke.py
    └── prepare_real_hdf5_smoke.py
```

## Remaining Coupling Risks

These modules are still too large and should be split before more agent
features are added:

- `core/literature_algorithm_lab.py`
  - Source acquisition, AlgorithmCard v2 validation, candidate compilation,
    candidate promotion, registry writeback, and AlgorithmCard extraction are
    now split. Executable-blueprint validation, bounded executor authoring, and
    bounded executor sandbox execution are also split. This file is now mostly
    a compatibility facade plus high-level literature pipeline/task preparation
    and blocked-candidate routing.
- `core/literature_expansion.py`
  - Local asset candidate construction and promotion materialization are now
    split. Remaining split points are candidate handoff learning constraints,
    sandbox queue execution, queue-evidence materialization,
    registry-summary decision policy, and refinement planning.
- `core/batch_optimization_candidates.py`
  - Batch loop orchestration, objective scoring, memory, and physics candidate
    families are now split. Hypothesis Lab patch families are also split.
    Remaining split points are baseline/runtime candidate families and
    postprocess candidate scheduling if this module grows again.
- `core/task1_short_window_training.py`
  - Model primitives, family-selection policy, Hypothesis Lab bridges, and FNO
    policy logic are now split. Remaining split points are data preparation,
    candidate-plan materialization, optimization-loop orchestration,
    harness/iteration contracts, and evidence-file emission.
- `core/hypothesis_lab.py`
  - Sandbox reference helpers, sandbox selection helpers, and parameter
    candidate construction are now split. Remaining split points are full
    sandbox execution, AlgorithmBlueprint assembly, candidate metadata
    construction, and run orchestration.
- `core/agent_runtime.py`
  - Still contains SDK/CLI execution, bash guard hooks, competition log export,
    and runtime issue recording. The next split should separate low-level
    execution adapters from log/export utilities if this module grows again.

## Rule Going Forward

New behavior should not land as a standalone root script. It should first be a
core capability, then a thin command wrapper, and only then an operational
script if there is a real fixture or smoke-run need.
