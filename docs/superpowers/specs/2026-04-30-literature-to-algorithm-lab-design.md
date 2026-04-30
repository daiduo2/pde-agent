# Literature-To-Algorithm Lab Design

## Context

Hermes now has a useful but narrow scientific loop:

- local Burgers strategy library,
- Hypothesis Lab knowledge cards and blueprints,
- checkpoint-family sandbox,
- local competition judge,
- strategy proposal, rubric, promotion, and ledger protocols.

This is enough to prevent arbitrary search-to-code behavior, but the hypothesis
space is still mostly seeded by local hand-written mechanisms and prior runs.
The next capability should let Hermes stand on published work without turning
external retrieval into unverified prompt memory.

The goal is a PDE-focused literature-to-algorithm pipeline. Its output should
not be a paper summary. Its primary output should be an `Algorithm Candidate`:
a source-grounded, benchmark-ready algorithm/operator/strategy proposal that can
enter a controlled sandbox, produce quantitative evidence, and only then become
eligible for internalization.

## Goals

Build a narrow but real "paper-to-PDE-algorithm" lane for Hermes:

1. Retrieve PDE/neural-operator sources from open scholarly and code indexes.
2. Download and preserve original documents locally.
3. Parse papers through MinerU into markdown and structured parse artifacts.
4. Extract operational algorithm ideas from parsed papers and related repos.
5. Convert those ideas into benchmark-ready `Algorithm Candidate` artifacts.
6. Execute candidates only through controlled Hermes sandboxes.
7. Record positive and negative evidence in the existing strategy-growth ledger.
8. Promote only candidates with sandbox evidence, not textual plausibility.

The initial domain is PDEBench-style Burgers and neural operators. The design
should not attempt a universal paper-to-code system.

## Non-Goals

- Do not let external papers or GitHub repositories directly patch project code.
- Do not execute third-party repository code inside Hermes.
- Do not try to support arbitrary scientific papers in V0.
- Do not treat LLM confidence, citation count, or paper prestige as promotion
  evidence.
- Do not require internet access during ordinary sandbox execution.
- Do not replace the existing Hypothesis Lab; extend it with stronger external
  artifacts and algorithm-candidate handoff.

## Design Choice

Use a staged artifact pipeline:

```text
external source
  -> source_record
  -> paper_parse / repo_record / resource_record
  -> algorithm_card
  -> algorithm_candidate
  -> sandbox_result
  -> strategy_proposal + rubric + ledger + promotion gate
```

This keeps every transformation inspectable. The LLM may help extract and
translate ideas, but each output is bounded by a schema and later judged by
experiments.

## Rejected Alternatives

### Retrieval-To-Summary

Search arXiv, summarize the top papers, and feed summaries into Hermes memory.
This is easy, but it does not solve the project bottleneck. It expands text
context rather than executable strategy space.

### Retrieval-To-Code

Search a repo, copy or synthesize code, run it, and keep the best result. This
can move quickly but is hard to audit, unsafe, and likely to produce brittle
non-reproducible candidates.

### Fully Generic Paper Reproduction

Try to turn any paper into runnable code. This is a research project by itself.
The correct first target is narrower: PDE/neural-operator mechanisms that can be
translated into existing or clearly bounded Hermes executors.

## Source Types

### OpenAlex

OpenAlex is the discovery and metadata layer. It should be used for:

- keyword search over PDE/neural-operator topics,
- citation and related-work expansion,
- DOI and venue metadata,
- author/title deduplication,
- ranking sources before expensive PDF parsing.

OpenAlex should not be treated as the source of the full algorithm. It is a map.

### arXiv

arXiv is the preferred full-text source when available. It should be used for:

- PDF download,
- versioned source identity through arXiv ID and version,
- stable provenance records.

### GitHub

GitHub is a resource and implementation clue source. It may provide:

- official baseline code,
- checkpoint links,
- config values,
- algorithm details missing from a paper.

External code is evidence and reference material only. It should not be run
directly unless a separate sandbox policy is designed and approved.

### Competition Resources

Competition-provided files, baseline tarballs, and checkpoints are resource
records. They should enter the same registry as external sources because Hermes
needs provenance, hashes, local paths, and execution eligibility.

## Artifact Schemas

### Source Record

`source_record` is the canonical provenance artifact.

Required fields:

- `source_id`
- `source_type`: `openalex_work`, `arxiv_paper`, `github_repo`,
  `competition_resource`, or `manual_source`
- `title`
- `url`
- `retrieved_at`
- `license_note`
- `local_raw_path`
- `content_hash`
- `related_source_ids`
- `retrieval_query`

### Paper Parse

`paper_parse` stores MinerU output and parse quality metadata.

Required fields:

- `parse_id`
- `source_id`
- `parser`: `mineru`
- `parser_version`
- `markdown_path`
- `asset_dir`
- `parse_quality`
- `section_index`
- `formula_count`
- `figure_count`
- `table_count`
- `quality_flags`

Quality flags should catch missing abstract/method sections, near-empty formula
extraction, excessive garbling, or unusable markdown.

### Algorithm Card

`algorithm_card` is the first artifact that extracts operational value from a
paper or repo.

Required fields:

- `algorithm_card_id`
- `source_ids`
- `claim`
- `core_mechanism`
- `operation`
- `preconditions`
- `implementation_hooks`
- `expected_observables`
- `failure_modes`
- `minimal_experiment`
- `evidence_status`: `extracted`, `sandbox_planned`, `tested`, `promoted`,
  or `rejected`

The most important field is `operation`. It must describe an action Hermes can
attempt, not a vague statement such as "improves long-horizon stability".

### Algorithm Candidate

`algorithm_candidate` is the main product of the module.

Required fields:

- `algorithm_candidate_id`
- `parent_algorithm_card_id`
- `candidate_family`
- `target_problem`: initially `PDE_Burgers`
- `target_task`: `task1`, `task2`, or `both`
- `model_family`: `FNO`, `UNet`, `DeepONet`, `PI-DeepONet`,
  `postprocess`, `checkpoint`, or `hybrid`
- `hypothesis`
- `mechanism`
- `executor_spec`
- `parameter_grid`
- `benchmark_plan`
- `ablation_plan`
- `promotion_criteria`
- `risk_controls`
- `source_grounding`

The `executor_spec` must name an existing controlled executor or request a new
bounded executor. In V0, candidates that require unconstrained model-code
generation should be marked `blocked_requires_executor_design`.

## Candidate Families

V0 should support only families that can plausibly enter the current Hermes
benchmark stack:

- `official_checkpoint_family`: checkpoint discovery, calibration, and FNO
  short fine-tuning through `run-checkpoint-family-sandbox`.
- `spectral_rollout_stabilization`: filters, viscosity ramps, energy clamps, or
  frequency-domain residual controls.
- `physics_residual_regularization`: Burgers residual terms or conservative
  constraints used in training or calibration.
- `scheduled_rollout_training`: teacher forcing decay, scheduled sampling,
  late-window weighting, or curriculum rollout.
- `lightweight_residual_corrector`: low-capacity residual correction over
  persistence or checkpoint rollouts.
- `uncertainty_or_ensemble_arbitration`: model selection, score arbitration, or
  confidence-gated blending.

This list is intentionally small. Additional families should be added only when
there is a clear executor and benchmark plan.

## Sandbox Levels

Every candidate should be assigned the cheapest valid sandbox first.

### Level 0: Mechanism Proxy

Use numpy or existing prediction artifacts. Examples:

- spectral damping on a saved rollout,
- persistence/checkpoint blend,
- horizon-wise residual correction,
- late-window calibration.

This level can reject many weak ideas quickly.

### Level 1: Existing Model Adapter

Use existing audited model paths:

- official FNO/UNet checkpoint inference,
- checkpoint calibration,
- FNO short fine-tuning,
- existing local operator-family executors.

### Level 2: New Bounded Executor

Implement a new controlled executor only after an algorithm candidate cannot be
tested at Level 0 or Level 1 and has strong source grounding. The executor must
be separately designed, tested, and audited.

## Internalization Rule

No candidate is internalized because a paper says it works. Internalization
requires sandbox evidence.

Minimum promotion requirements:

- valid prediction shape and context preservation,
- local judge score improvement above a configured threshold,
- inference time within the competition budget,
- at least one ablation or contrast showing the mechanism matters,
- recorded negative risks and failure modes,
- reproducible artifact paths,
- ledger entry for both accepted and rejected outcomes.

Promotion means "eligible to become a Hermes strategy or skill candidate". It
does not mean automatic code generation or permanent default policy change.

## Data Flow

1. `search-external-sources`
   writes ranked source records from OpenAlex, arXiv, GitHub, and competition
   resource URLs.
2. `download-source-documents`
   downloads PDFs or resource files into a local source cache and records hashes.
3. `parse-paper-with-mineru`
   runs MinerU and writes markdown plus parse-quality metadata.
4. `extract-algorithm-cards`
   reads parsed markdown and extracts operational algorithm cards.
5. `compile-algorithm-candidates`
   maps algorithm cards to Hermes candidate families and executor specs.
6. `plan-candidate-sandbox`
   selects Level 0, Level 1, or Level 2 sandbox routes.
7. `run-algorithm-candidate-sandbox`
   executes only approved candidate routes.
8. `record-strategy-proposal`
   formalizes candidates with evidence for the existing promotion gate.
9. `append-strategy-ledger`
   records positive and negative outcomes.

## Integration With Existing Hermes

### Hypothesis Lab

Hypothesis Lab should keep its local knowledge-card behavior. The literature
pipeline should add richer external artifacts:

- source records become provenance cards,
- algorithm cards become higher-quality knowledge cards,
- algorithm candidates become executable blueprints or handoffs.

The existing checkpoint-family handoff should become one supported candidate
family, not a special case.

### Strategy Growth

The strategy-growth layer remains the gate for internalization. Literature
candidates should reuse:

- strategy proposal,
- experiment rubric,
- promotion report,
- strategy ledger,
- skill candidate creation.

### Local Judge

The local judge remains the primary quantitative proxy for competition progress.
If future PDE tasks require different metrics, they should be added as new judge
adapters rather than bypassing the ledger.

## Error Handling

External retrieval is expected to fail often. Failures should be explicit
artifacts, not silent skips.

Important failure states:

- `SOURCE_SEARCH_EMPTY`
- `SOURCE_DOWNLOAD_FAILED`
- `PAPER_PARSE_LOW_QUALITY`
- `ALGORITHM_EXTRACTION_EMPTY`
- `CANDIDATE_UNSUPPORTED_EXECUTOR`
- `SANDBOX_FAILED`
- `PROMOTION_REJECTED`

Each failure state should include recovery hints and enough provenance to avoid
repeating the same failed attempt blindly.

## Security And Reproducibility

- Store local copies and hashes for all downloaded documents.
- Record retrieval time and query.
- Prefer pinned source versions: arXiv version, Git commit, release tag, or
  content hash.
- Never execute third-party repository code in V0.
- Keep network retrieval separate from offline sandbox execution.
- Make every generated candidate reproducible from local artifacts.

## Testing Strategy

Unit tests should avoid live network calls.

V0 tests should cover:

- OpenAlex/arXiv/GitHub search result normalization from fixture JSON.
- source record hashing and duplicate handling.
- MinerU parse ingestion from fixture markdown.
- parse quality gate behavior.
- algorithm-card extraction from a small fixture paper note.
- algorithm-candidate compilation into supported families.
- sandbox planning for checkpoint, spectral, residual, and unsupported families.
- ledger/promotion handoff from a fake sandbox result.

Live retrieval and MinerU execution should be smoke tests gated behind explicit
commands or local configuration.

## Success Criteria

The first useful version is successful when:

- a known neural-operator paper can be registered, parsed, and converted into at
  least one `Algorithm Candidate`;
- the candidate is routed to an existing sandbox or explicitly blocked with a
  missing-executor reason;
- a checkpoint or spectral/residual candidate can run end-to-end through local
  judge and ledger;
- negative candidates are recorded and retrievable;
- no external code is executed directly;
- full repository tests remain green.

## Suggested Implementation Phases

### Phase 1: Offline Artifact Backbone

Implement schemas and commands for source records, MinerU parse ingestion,
algorithm cards, algorithm candidates, and sandbox planning. Use fixture data
only. This phase proves the shape of the system without network or MinerU
runtime dependencies.

### Phase 2: Resource Retrieval

Add retrieval for competition resources, GitHub releases, and checkpoint links.
Connect discovered checkpoint artifacts to `run-checkpoint-family-sandbox`.

### Phase 3: Literature Retrieval And MinerU

Add OpenAlex and arXiv source discovery, PDF download, and MinerU parse ingestion.
Keep live network and MinerU execution outside unit tests.

### Phase 4: Candidate Execution

Route supported candidates into Level 0 and Level 1 sandboxes. Add promotion
handoff and ledger integration.

### Phase 5: New Executor Design

Only after evidence shows repeated high-value unsupported candidates, design new
bounded executors for specific PDE algorithm families.

## Naming Decision

The module should be named around the durable product, not the intermediate
hypothesis. The recommended module name is:

`literature_algorithm_lab`

This keeps the emphasis on benchmark-ready algorithm candidates while still
allowing hypotheses, knowledge cards, and source records as internal artifacts.

## Phase 1 Implementation Note

Phase 1 is intentionally offline. It validates artifact contracts and candidate
handoff semantics with local fixture data before adding live source retrieval or
MinerU runtime execution. Network retrieval and parser invocation remain smoke
test concerns for later phases.

## Phase 2a Implementation Note

Phase 2a adds network-capable retrieval and MinerU task commands while keeping
ordinary tests fixture-only. OpenAlex, arXiv, and GitHub search results are
normalized into source candidates before any document is downloaded. MinerU
precise parsing reads credentials only from the `MINERU_API_TOKEN` environment
variable; the token must not be written into source records, task reports, logs,
or commits.
