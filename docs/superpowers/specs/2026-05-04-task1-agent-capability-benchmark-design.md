# Task1 Agent Capability Benchmark Design

## Goal

Build a read-only benchmark that judges whether Hermes is improving as a PDE
agent while optimizing Task1. The benchmark uses Task1 score as the hard task
signal, but it also measures whether the score came from reusable agent
capability: diverse candidate generation, sandbox evidence, promotion/rejection
discipline, and failure memory.

## Inputs

- `hermes_runtime_audit_chain_report.json`
- `hypothesis_lab_trace.jsonl`
- optional `baseline_score`
- thresholds for score delta, unique candidate families, and sandbox evidence

## Output

`task1_agent_capability_benchmark.json` with:

- `score_progress`: initial, final, best, delta, stagnation count
- `candidate_generation`: candidate event count, unique family count,
  non-postprocess ratio
- `sandbox_validation`: evidence event count, outcome counts, invalid promotion
  count
- `learning_memory`: failure-memory event count and taxonomy coverage
- `promotion_quality`: evidence-backed promotions, Pareto-gated promotions, and
  independent-slice evidence
- `decision`: `passed`, `score_only`, `infrastructure_only`, `stagnated`, or
  `failed`

## Decision Semantics

- `passed`: Task1 score improves and capability metrics pass.
- `score_only`: score improves, but capability evidence is too weak to claim
  reusable Hermes improvement.
- `infrastructure_only`: capability evidence passes, but score does not improve.
- `stagnated`: score and candidate space show plateau/exhaustion signals.
- `failed`: neither score nor capability evidence passes.

## Scope

This benchmark does not run Hermes, train models, inspect private targets, or
promote strategies. It only reads existing artifacts and writes a structured
evaluation report. Long-running Task1 optimization should use it after each
audited batch.
