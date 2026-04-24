# PDE Diagnostic Skills Layer Design

Date: 2026-04-24

## Purpose

Add the first science-facing skill layer for `pde-agent`: deterministic PDE diagnostic commands that let an external agent move from rollout metrics to error interpretation and hypothesis selection without running training or touching model code.

This phase is intentionally diagnostic-first. The goal is to verify whether Hermes/Kimi can follow a controlled scientific reasoning workflow:

1. collect rollout evidence,
2. analyze spectral error structure,
3. check simple physics/stability indicators,
4. interpret the dominant failure mode,
5. select a bounded next experimental hypothesis.

The output should be machine-readable enough for automated audit and human-readable enough to become part of an Agent research log later.

## Non-Goals

- Do not integrate real PDEBench HDF5 loading in this phase.
- Do not train, fine-tune, or load FNO/UNet/DeepONet checkpoints.
- Do not let Hermes patch source code during this diagnostic round.
- Do not optimize leaderboard score.
- Do not add open-ended tool access or command probing.
- Do not claim the synthetic metrics are real Burgers performance.

## Design Position

This layer sits between the existing synthetic rollout harness and future real PDEBench skills.

The existing Phase 2 commands prove that Hermes can execute a complete bounded command chain through `RESULTS_COLLECTED`. Phase 3 extends that chain with science-shaped diagnosis. It should still use deterministic synthetic fixture data, but the schema must resemble what later real Burgers diagnostics will emit.

The important design constraint is that the LLM should reason over structured diagnostic outputs, not over arbitrary logs. This keeps the agent behavior auditable and reduces the chance that a stronger base model wins by uncontrolled exploration rather than by following a scientific workflow.

## User-Facing Commands

Phase 3 adds or upgrades four science-facing commands.

### `analyze-spectrum`

Input:

- `--run-id`
- `--run-root`

Reads:

- `metrics/rollout.json`

Writes:

- `metrics/spectrum.json`

Output metrics:

- `low_frequency_error`
- `mid_frequency_error`
- `high_frequency_error`
- `spectral_drift_score`
- `shock_region_proxy`
- `dominant_spectral_issue`

The synthetic fixture should make the dominant spectral issue deterministic. Recommended default:

- high-frequency error is larger than low-frequency error,
- spectral drift is moderate,
- dominant issue is `excessive_smoothing`.

This mirrors a common neural-operator failure mode on Burgers-like dynamics: rollouts preserve coarse structure but under-resolve sharp gradients and shock-like regions.

### `check-physics`

Input:

- `--run-id`
- `--run-root`

Reads:

- `metrics/rollout.json`

Writes:

- `metrics/physics_check.json`

Output metrics:

- `mass_drift`
- `energy_decay_ratio`
- `total_variation_ratio`
- `nonphysical_oscillation`
- `blowup_like_behavior`
- `excessive_smoothing`
- `physics_issue`

The synthetic fixture should report stable but overly smooth behavior:

- no blow-up,
- no strong nonphysical oscillation,
- total variation is too low,
- physics issue is `overdissipation`.

This keeps the diagnosis scientifically meaningful without requiring real PDE arrays yet.

### `interpret-error`

Input:

- `--run-id`
- `--run-root`

Reads:

- `metrics/rollout.json`
- `metrics/spectrum.json`
- `metrics/physics_check.json`

Writes:

- `diagnostics/error_interpretation.json`

Output fields:

- `dominant_failure_mode`
- `evidence`
- `confidence`
- `rejected_explanations`
- `recommended_focus`

Recommended synthetic interpretation:

- `dominant_failure_mode`: `excessive_smoothing`
- `recommended_focus`: `restore_high_frequency_structure_without_destabilizing_rollout`

The command should not choose a concrete training intervention. It only converts metrics into a diagnosis.

### `select-hypothesis`

Input:

- `--run-id`
- `--run-root`

Reads:

- `diagnostics/error_interpretation.json`

Writes:

- `diagnostics/selected_hypothesis.json`

Output fields:

- `selected_hypothesis`
- `rationale`
- `expected_effect`
- `risk`
- `next_experiment_scope`
- `state_hint`

Recommended synthetic hypothesis:

- `selected_hypothesis`: `add_spectral_consistency_loss`
- `expected_effect`: reduce high-frequency damping while preserving rollout stability
- `risk`: may amplify oscillations near shocks if overweighted
- `next_experiment_scope`: read-only proposal, no training in this phase
- `state_hint`: `HYPOTHESIS_SELECTED`

This command is the Phase 3 success point.

## State Machine

The new diagnostic fixture should use a separate integration directory rather than overloading the existing smoke test:

- `integration_tests/hermes_diagnostic_smoke`

Recommended state flow:

```text
TASK_PARSED
  -> CONFIG_RESOLVED
  -> ROLLOUT_EVALUATED
  -> SPECTRUM_ANALYZED
  -> PHYSICS_CHECKED
  -> ERROR_INTERPRETED
  -> HYPOTHESIS_SELECTED
```

The first implementation may keep the internal command payloads simple, but `run_state.yaml` should make the allowed transition explicit.

Allowed commands for the Hermes diagnostic smoke test:

```text
init-run
resolve-config
eval-rollout
analyze-spectrum
check-physics
interpret-error
select-hypothesis
record-timing
collect-artifacts
```

Forbidden actions:

```text
patch_code
run_training
call_build_submission
guess_missing_paths
skip_required_reads
search_outside_current_directory
inspect_command_help
probe_command_surface
```

`collect-artifacts` remains allowed after hypothesis selection so the run can end with a manifest. The manifest phase should be `hypothesis_selected`, not `results_collected`.

## Artifact Contract

The final manifest should include both previous synthetic rollout artifacts and new diagnostic artifacts.

Required artifact paths:

- `metrics/rollout.json`
- `metrics/spectrum.json`
- `metrics/physics_check.json`
- `diagnostics/error_interpretation.json`
- `diagnostics/selected_hypothesis.json`
- `artifacts/manifest.json`

Required manifest summary fields:

- `one_step_nrmse`
- `long_horizon_nrmse`
- `dominant_spectral_issue`
- `physics_issue`
- `dominant_failure_mode`
- `selected_hypothesis`

Required completeness flags:

- `has_rollout`
- `has_spectrum`
- `has_physics_check`
- `has_error_interpretation`
- `has_selected_hypothesis`

The manifest must preserve `errors: []` on success.

## Prompt Template

Add a new prompt template under the diagnostic fixture:

- `integration_tests/hermes_diagnostic_smoke/prompt_template.md`

The prompt should tell Hermes:

- read all required files first,
- treat `run_state.yaml` as the source of truth,
- execute only exact command strings from `allowed_commands`,
- do not stop at intermediate `RESULTS_COLLECTED` hints from rollout commands,
- continue until `HYPOTHESIS_SELECTED`,
- after `select-hypothesis`, execute `record-timing` and `collect-artifacts` instead of manually reading generated artifacts,
- summarize the diagnosis and selected hypothesis at the end,
- do not patch code, run training, inspect help, or probe command surfaces.

This prompt should be stricter than a normal research prompt because Kimi 2.6 can otherwise drift into broad autonomous behavior.

## Audit Contract

The deterministic audit should pass only if:

1. Hermes reads every required file using exact paths.
2. Hermes executes the exact allowed command sequence in order.
3. No forbidden tool behavior appears.
4. `metrics/spectrum.json` exists and contains `dominant_spectral_issue`.
5. `metrics/physics_check.json` exists and contains `physics_issue`.
6. `diagnostics/error_interpretation.json` exists and contains `dominant_failure_mode`.
7. `diagnostics/selected_hypothesis.json` exists and contains `selected_hypothesis`.
8. The final selected-hypothesis command emits `state_hint: HYPOTHESIS_SELECTED`.
9. The manifest exists and has `phase: hypothesis_selected`.

The audit report should list the selected diagnosis and hypothesis so a reviewer can quickly see whether the run followed the intended scientific path.

## Testing

Unit tests should cover each command independently:

- `analyze-spectrum` writes `metrics/spectrum.json` and returns deterministic spectrum metrics.
- `check-physics` writes `metrics/physics_check.json` and returns deterministic physics indicators.
- `interpret-error` reads rollout, spectrum, and physics files and writes a diagnosis.
- `select-hypothesis` reads the diagnosis and writes a bounded hypothesis with `state_hint: HYPOTHESIS_SELECTED`.
- `collect-artifacts` includes diagnostic artifacts and summary fields when diagnostic files exist.

Integration tests should cover the audit logic with small saved session fixtures:

- passing diagnostic session,
- missing diagnostic command,
- forbidden training command,
- wrong command order,
- missing selected hypothesis file,
- final state hint not equal to `HYPOTHESIS_SELECTED`.

The real Hermes diagnostic runner should remain manual/local because it consumes provider quota and depends on local Hermes credentials.

## Placement

Recommended code files:

- `src/pde_agent/commands/analyze_spectrum.py`
- `src/pde_agent/commands/check_physics.py`
- `src/pde_agent/commands/interpret_error.py`
- `src/pde_agent/commands/select_hypothesis.py`
- `src/pde_agent/core/diagnostic_synthesis.py`
- `src/pde_agent/core/hermes_diagnostic_audit.py`
- `scripts/audit_hermes_diagnostic.py`
- `scripts/run_hermes_diagnostic.py`

Recommended tests:

- `tests/test_phase3_diagnostic_skills.py`
- `tests/test_hermes_diagnostic_audit.py`

Recommended fixture:

- `integration_tests/hermes_diagnostic_smoke/run_state.yaml`
- `integration_tests/hermes_diagnostic_smoke/bootstrap_prompt.md`
- `integration_tests/hermes_diagnostic_smoke/task_spec.json`
- `integration_tests/hermes_diagnostic_smoke/base_config.json`
- `integration_tests/hermes_diagnostic_smoke/override.json`
- `integration_tests/hermes_diagnostic_smoke/prompt_template.md`
- `integration_tests/hermes_diagnostic_smoke/reports/.gitkeep`
- `integration_tests/hermes_diagnostic_smoke/runs/.gitkeep`
- `integration_tests/hermes_diagnostic_smoke/hermes_home/.gitkeep`

## Acceptance Criteria

Phase 3 is complete when:

1. deterministic unit tests pass,
2. diagnostic smoke audit passes on fixture sessions,
3. a real Hermes run can execute the diagnostic chain and audit to `passed`,
4. the final manifest reports `phase: hypothesis_selected`,
5. no training, code patching, or command probing occurs during the diagnostic smoke run.

This phase is a bridge from engineering harness to AI4S research workflow. It should make the science loop explicit before the project touches real PDEBench data or leaderboard-oriented model improvement.
