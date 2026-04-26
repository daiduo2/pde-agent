# Strategy Growth Loop Design

## Problem

The current harness can make Hermes/Kimi follow bounded PDE workflows, but most
PDE strategy knowledge is still built into our code and prompts by humans. That
is useful for competition safety, but it risks turning the agent into a static
state machine: high local score, low scientific adaptability.

We need a growth layer that lets the agent propose, test, record, and eventually
internalize strategies without giving it unconstrained authority to search,
patch, or train arbitrarily.

## Design Choice

Build a deterministic `strategy-growth` layer before exposing web retrieval.
This layer treats every strategy, whether local or externally retrieved, as data
that must pass through the same protocol:

1. `strategy_proposal.json`: a formalized strategy proposal.
2. `experiment_rubric.json`: the quantitative standard required to evaluate it.
3. `strategy_ledger.jsonl`: append-only record of the attempt and outcome.
4. `promotion_report.json`: machine-readable decision on whether the strategy
   can become a reusable skill candidate.

This is not a skill generator yet. It is the gate in front of future skill
generation.

## Rejected Alternatives

### Direct Search-To-Code

Let Hermes search the internet, pick a method, and patch the model code. This
maximizes flexibility but makes failures hard to audit: unclear source quality,
unclear experiment standard, and no stable record of failed ideas.

### Fully Handwritten PDE Skill Library

We keep writing FNO, DeepONet, PI-DeepONet, and Burgers-specific skills by hand.
This is safe and competition-friendly, but it does not test agent intelligence;
it mainly tests our ability to build a state machine.

### Recommended Hybrid

Use hard protocol boundaries for execution, but allow agent-generated strategies
inside those boundaries. The agent may propose or retrieve a strategy, but it
cannot execute it until the strategy is formalized, assigned a rubric, evaluated
against evidence, and recorded in the ledger.

## Components

### Strategy Proposal

A proposal captures the scientific idea, not executable code. Required fields:

- `strategy_id`
- `title`
- `source_type`: `local_library`, `external_retrieval`, or `agent_hypothesis`
- `problem_context`
- `hypothesis`
- `mechanism`
- `implementation_plan`
- `expected_observables`
- `known_risks`

The proposal is intentionally method-agnostic. It may describe FNO curriculum
rollout, PINN residual regularization, spectral filtering, uncertainty
screening, or future inverse-problem strategies.

### Experiment Rubric

The rubric defines what evidence is required before a strategy can be accepted.
For the current PDEBench/Burgers setting, the default rubric requires:

- at least 3 evaluation windows,
- non-negative mean score delta,
- bounded score-delta variance,
- no increase in failure rate,
- no unacceptable inference-time regression,
- explicit long-window and physics-stability evidence fields.

The rubric is not the experiment itself. It is the standard future experiments
must satisfy.

### Strategy Ledger

The ledger is append-only JSONL. Each row records:

- proposal path,
- rubric path,
- evidence summary path,
- promotion report path,
- final decision,
- reason codes.

This makes failed strategies first-class artifacts. Future agents can avoid
retrying the same idea blindly or can revisit it when the context changes.

### Promotion Gate

The promotion gate consumes a proposal, rubric, and evidence summary. It returns:

- `promote_to_skill_candidate`: boolean,
- `decision`: `promote`, `revise`, or `reject`,
- `reason_codes`,
- metric-level pass/fail details,
- `next_command`.

In V0, `promote` only means “eligible to become a skill candidate”. It does not
automatically write a Codex/Hermes skill. That later step should be separate and
audited.

## Data Flow

1. Agent reaches a poor or uncertain local result.
2. Agent checks existing local strategy library.
3. If local strategies are insufficient, a future retrieval fixture may collect
   external candidate ideas.
4. Every candidate is converted into `strategy_proposal.json`.
5. The harness generates or validates `experiment_rubric.json`.
6. Existing experiment runners produce evidence summaries.
7. The promotion gate evaluates the evidence.
8. The ledger records the full attempt.
9. Only promoted strategies are eligible for future skill creation.

## V0 Scope

Implement only local deterministic commands:

- create a validated strategy proposal artifact,
- create a default experiment rubric artifact,
- evaluate a promotion report from synthetic or real evidence summaries,
- append one ledger row.

Do not add web search yet. Do not generate skills yet. Do not let Hermes patch
model code through this path.

## Success Criteria

- All artifacts are machine-readable JSON or JSONL.
- Invalid strategies fail before being recorded.
- Promotion requires multi-window evidence, not a single lucky result.
- Failed strategies are recorded with reason codes.
- The API is model-family agnostic and can later support FNO, DeepONet,
  PI-DeepONet, PINN, or inverse-problem strategies.
