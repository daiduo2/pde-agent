# FNO Evidence Strategy Summary Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `run-fno-evidence-sweep` outputs into the canonical `strategy-growth` evidence summary schema consumed by `evaluate-strategy-promotion`.

**Architecture:** Add a small FNO-specific adapter that reads an existing sweep summary JSON, derives the seven strategy-growth evidence metrics from comparison winners, and writes a separate strategy evidence artifact. The adapter does not rerun training, alter FNO strategy logic, or fabricate unavailable measurements; it emits provenance notes for neutral placeholder metrics.

**Tech Stack:** Python 3.14, Typer CLI, JSON artifacts, pytest, existing `success_payload` protocol.

---

## Files

- Create `src/pde_agent/core/fno_strategy_evidence.py`: parse FNO sweep summary and write canonical evidence summary.
- Create `src/pde_agent/commands/convert_fno_evidence_summary.py`: CLI wrapper.
- Modify `src/pde_agent/cli.py`: expose `convert-fno-evidence-summary`.
- Add `tests/test_fno_strategy_evidence.py`: core and CLI behavior.
- Update `docs/hermes_integration_discipline.md`: document FNO evidence adapter as the bridge into strategy growth.

## Metric Mapping

- `evaluation_window_count`: number of unique `start_index` values in `comparison_winners`.
- `mean_score_delta`: mean of `winner_score_delta`.
- `score_delta_std`: population standard deviation of `winner_score_delta`, zero for one cell.
- `long_window_score_delta`: mean winner delta when `inputs.forecast_steps >= long_window_threshold`; otherwise zero and marked as weak evidence.
- `physics_residual_delta`: numeric field from the sweep if present; otherwise neutral `0.0` with a provenance note.
- `inference_time_regression`: numeric field from the sweep if present; otherwise neutral `0.0` with a provenance note.
- `failure_rate`: fraction of comparison winners whose `winner_score_delta < 0`.

## Task 1: Core Adapter And CLI

- [ ] Write failing tests for:
  - strong multi-window FNO sweep conversion,
  - single-window conversion preserving `evaluation_window_count = 1`,
  - CLI protocol payload.
- [ ] Run `uv run --frozen pytest tests/test_fno_strategy_evidence.py -q` and confirm it fails because the module is missing.
- [ ] Implement `convert_fno_evidence_summary` with validation:
  - input file exists,
  - `state_hint == FNO_EVIDENCE_SWEEP_COMPLETED`,
  - `comparison_winners` is non-empty,
  - each winner has numeric `winner_score_delta`,
  - output path does not already exist unless `force=True`.
- [ ] Implement CLI command:
  - command name: `convert-fno-evidence-summary`,
  - options: `--sweep-summary-path`, `--output-path`, `--long-window-threshold`, `--force`,
  - state hint: `STRATEGY_EVIDENCE_SUMMARY_BUILT`,
  - next hint: `evaluate-strategy-promotion`.
- [ ] Run focused tests and commit `feat: add fno strategy evidence adapter`.

## Task 2: Documentation And Verification

- [ ] Update `docs/hermes_integration_discipline.md` to state that FNO evidence enters strategy-growth through `convert-fno-evidence-summary`.
- [ ] Run:

```bash
uv run --frozen pytest tests/test_fno_strategy_evidence.py tests/test_strategy_growth.py -q
uv run --frozen pytest tests -q
git diff --check
```

- [ ] Commit `docs: document fno strategy evidence bridge`.
