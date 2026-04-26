# FNO Candidate Comparison V0 Design

## Goal

Compare multiple evaluated FNO candidates using their existing
`candidate_evaluation_report.json` artifacts and produce one deterministic
selection report for the next iteration.

This is the second step after FNO candidate evaluation:

```text
candidate_evaluation_report.json...
  -> compare-fno-candidates
  -> fno_candidate_comparison_report.json
```

The comparator is a decision boundary. It does not train models, rerun
diagnostics, or mutate candidate artifacts.

## Command

```bash
uv run --frozen pde-agent compare-fno-candidates \
  --evaluation-report-path runs/fno_iteration_001/candidates/candidate_a/artifacts/evaluation/candidate_evaluation_report.json \
  --evaluation-report-path runs/fno_iteration_001/candidates/candidate_b/artifacts/evaluation/candidate_evaluation_report.json \
  --output-path runs/fno_iteration_001/fno_candidate_comparison_report.json \
  --force
```

`--evaluation-report-path` is repeatable. V0 requires at least one report so it
also works for the current single-candidate smoke loop.

## Selection Rule

Candidates are sorted by:

1. Higher `score.competition_like_score_short`.
2. Lower `score.rel_mse`.
3. Lower `score.rmse`.
4. Lexicographically smaller `candidate.experiment_id` for deterministic ties.

The top candidate becomes `winner`. Each non-winner receives a machine-readable
`rejection_reason`:

- `lower_score`
- `tie_broken_by_rel_mse`
- `tie_broken_by_rmse`
- `tie_broken_by_experiment_id`

## Report Contract

`fno_candidate_comparison_report.json` includes:

- `state_hint = FNO_CANDIDATES_COMPARED`
- input report paths
- sorted candidate summaries
- winner candidate summary
- rejected candidate summaries
- aggregate counts
- `next_command = prepare-next-fno-iteration`

Each candidate summary includes:

- `rank`
- `experiment_id`
- `intervention`
- `score`
- `rel_mse`
- `rmse`
- `baseline_score`
- `score_delta`
- `improved`
- `dominant_error_pattern`
- `physics_issue`
- `selected_hypothesis`
- `evaluation_report_path`
- `rejection_reason`

## Guardrails

- Reject missing files.
- Reject invalid JSON.
- Reject reports whose state hint is not `FNO_CANDIDATE_EVALUATED`.
- Reject duplicate `experiment_id` values.
- Reject output overwrite unless `--force`.
- Do not infer missing scores or silently skip malformed reports.

## Acceptance

- Fixture test proves two evaluated candidates are sorted deterministically.
- Fixture test proves single-candidate comparison still writes a winner.
- CLI returns `FNO_CANDIDATES_COMPARED`.
- Existing tests still pass.
