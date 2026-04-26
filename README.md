# pde-agent

CLI-first diagnostic harness for PDE agent experiments.

Initial MVP goals:

- strict JSON-only CLI contract
- explicit `run_id` and run directory tree
- resolved config generation
- manifest-centered artifact contract
- Thin-Hermes integration hooks

## Local PDEBench Window Scoring

Create a short local validation window from the public Burgers HDF5 file:

```bash
uv run --frozen --extra data pde-agent build-local-windows \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-path data/derived/local_windows_task1_val.npz \
  --dataset-key tensor \
  --start-index 0 \
  --sample-limit 128 \
  --context-steps 10 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4
```

Score a prediction NPZ containing a `prediction` array with shape `[N, T, X]`:

```bash
uv run --frozen pde-agent score-local-rollout \
  --prediction-path data/derived/local_predictions_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_score.json
```

This score is a local proxy for short-horizon model iteration. It is not the hidden 190-step competition evaluator.

Build a suite of real-HDF5 windows for horizon stress tests:

```bash
uv run --frozen --extra data pde-agent build-local-window-suite \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --output-dir data/derived/window_suites/task1_public \
  --dataset-key tensor \
  --start-index 0 \
  --sample-limit 32 \
  --context-steps 10 \
  --forecast-steps 8 \
  --forecast-steps 16 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4 \
  --force
```

For the public Task-1 file, `time_reduction=5` maps the raw 201 time samples to
41 reduced steps, so a 10-step context leaves at most 31 local forecast steps.
The same command can build 47/95/190-step suites once the longer validation or
test data is available.

## Local PDE Diagnostic Chain

After building a local windows NPZ, generate a weak persistence baseline and run the first PDE diagnostic loop:

```bash
uv run --frozen pde-agent generate-baseline-prediction \
  --windows-path data/derived/local_windows_task1_val.npz \
  --output-path data/derived/local_persistence_task1_val.npz \
  --method persistence

uv run --frozen pde-agent score-local-rollout \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_score.json

uv run --frozen pde-agent diagnose-rollout-error \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --score-path runs/local_score.json \
  --output-path runs/rollout_error_diagnosis.json

uv run --frozen pde-agent check-burgers-physics-local \
  --prediction-path data/derived/local_persistence_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/burgers_physics_check.json \
  --nu 0.001

uv run --frozen pde-agent select-pde-hypothesis \
  --score-path runs/local_score.json \
  --error-diagnosis-path runs/rollout_error_diagnosis.json \
  --physics-check-path runs/burgers_physics_check.json \
  --output-path runs/selected_pde_hypothesis.json
```

This chain is intended to give Hermes a reproducible failed baseline, numerical error diagnosis, physics proxy diagnosis, and bounded next-experiment proposal.

## Lightweight Spectral Baseline

The lightweight spectral residual baseline is a NumPy-only local smoke model for fast diagnostic loops. It trains a small ridge-regression residual model from a local windows NPZ and writes checkpoint/prediction NPZ artifacts that are compatible with the existing scoring and PDE diagnostic commands. It is intended for deterministic local sanity checks, not as the final contest model.

```bash
uv run --frozen pde-agent train-spectral-residual-baseline \
  --windows-path data/derived/local_windows_task1_val.npz \
  --checkpoint-path runs/spectral_residual_checkpoint.npz \
  --modes 16 \
  --ridge 1e-3

uv run --frozen pde-agent predict-local-model \
  --windows-path data/derived/local_windows_task1_val.npz \
  --checkpoint-path runs/spectral_residual_checkpoint.npz \
  --output-path data/derived/local_spectral_prediction_task1_val.npz

uv run --frozen pde-agent score-local-rollout \
  --prediction-path data/derived/local_spectral_prediction_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_spectral_score.json

uv run --frozen pde-agent diagnose-rollout-error \
  --prediction-path data/derived/local_spectral_prediction_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --score-path runs/local_spectral_score.json \
  --output-path runs/local_spectral_rollout_error.json

uv run --frozen pde-agent check-burgers-physics-local \
  --prediction-path data/derived/local_spectral_prediction_task1_val.npz \
  --target-path data/derived/local_windows_task1_val.npz \
  --output-path runs/local_spectral_burgers_physics.json \
  --nu 0.001

uv run --frozen pde-agent select-pde-hypothesis \
  --score-path runs/local_spectral_score.json \
  --error-diagnosis-path runs/local_spectral_rollout_error.json \
  --physics-check-path runs/local_spectral_burgers_physics.json \
  --output-path runs/local_spectral_pde_hypothesis.json

uv run --frozen pde-agent plan-next-experiment \
  --score-path runs/local_spectral_score.json \
  --error-diagnosis-path runs/local_spectral_rollout_error.json \
  --physics-check-path runs/local_spectral_burgers_physics.json \
  --hypothesis-path runs/local_spectral_pde_hypothesis.json \
  --output-path runs/local_spectral_next_experiment_plan.json

uv run --frozen pde-agent generate-experiment-config \
  --plan-path runs/local_spectral_next_experiment_plan.json \
  --output-path runs/local_spectral_experiment_config.json
```

`plan-next-experiment` writes machine-readable JSON only. It does not modify model
code, execute experiments, or write human-readable research notes. The planned
next step is `generate-experiment-config`, which expands bounded candidate
parameters into explicit candidate experiment records. It does not run candidates;
`execute-candidate-experiment` is a future hook.

## FNO Backend V0

FNO support is optional and requires the torch extra. V0 can inspect
torch-serialized FNO-style checkpoints and can run fixture-compatible local
prediction. `predict-fno-local` supports the official PDEBench 1D Burgers FNO
checkpoint schema used by `burgers_FNO.tar`. It reconstructs the FNO1d model
from checkpoint tensor shapes and performs autoregressive rollout from the local
windows context. Training and finetuning are intentionally not included in this
adapter phase.

```bash
uv run --frozen --extra torch pde-agent inspect-fno-checkpoint \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-path runs/fno_checkpoint_metadata.json

uv run --frozen --extra torch pde-agent predict-fno-local \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --metadata-path runs/fno_checkpoint_metadata.json \
  --output-path data/derived/local_fno_prediction_task1_smoke.npz \
  --force
```

The prediction command writes the same `prediction` NPZ contract consumed by
`score-local-rollout`. It never falls back to persistence or the spectral
baseline while claiming to be FNO.

## FNO Rollout Diagnosis

Run the official FNO adapter through the local diagnostic chain and collect one
agent-readable summary:

```bash
uv run --frozen --extra torch pde-agent diagnose-fno-rollout \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir runs/fno_rollout_diagnosis \
  --nu 0.001 \
  --dt 1.0 \
  --dx 1.0 \
  --force
```

The command writes `fno_prediction.npz`, `fno_score.json`,
`fno_rollout_error.json`, `fno_burgers_physics.json`, and
`fno_rollout_summary.json`. It does not train or finetune FNO; it only packages
the existing local prediction, scoring, error diagnosis, and physics checks into
one reproducible artifact bundle.

Select the next bounded PDE hypothesis directly from that FNO summary:

```bash
uv run --frozen pde-agent select-pde-hypothesis-from-fno-summary \
  --summary-path runs/fno_rollout_diagnosis/fno_rollout_summary.json \
  --output-path runs/fno_rollout_diagnosis/selected_pde_hypothesis.json
```

This bridge reuses `select-pde-hypothesis`; it only extracts the score,
rollout-error, and Burgers-physics artifact paths from the FNO summary so the
agent does not need to manually pass three separate paths.

Plan the next bounded experiment from the same FNO summary plus the selected
hypothesis:

```bash
uv run --frozen pde-agent plan-next-experiment-from-fno-summary \
  --summary-path runs/fno_rollout_diagnosis/fno_rollout_summary.json \
  --hypothesis-path runs/fno_rollout_diagnosis/selected_pde_hypothesis.json \
  --output-path runs/fno_rollout_diagnosis/next_experiment_plan.json
```

This produces the same `NEXT_EXPERIMENT_PLANNED` artifact as
`plan-next-experiment`, while keeping the FNO path contract compact for Hermes.

For agent runs, prefer the one-command dry-run wrapper:

```bash
uv run --frozen --extra torch pde-agent run-fno-iteration-dryrun \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir runs/fno_iteration_001 \
  --nu 0.001 \
  --dt 1.0 \
  --dx 1.0 \
  --force
```

The dry-run wrapper writes the FNO diagnosis artifacts, selected hypothesis,
next experiment plan, experiment config, and `iteration_manifest.json`. It does
not execute candidate experiments; its purpose is to stabilize the agent-facing
scientific loop before adding finetuning.

Run the default single-window FNO iteration loop:

```bash
uv run --frozen --extra torch pde-agent run-fno-iteration \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir runs/fno_iteration_001 \
  --max-train-steps 16 \
  --learning-rate 1e-4 \
  --force
```

This is the recommended Hermes entrypoint for ordinary V1 iterations after a
local window file exists. It runs the dry-run, audits the experiment config,
executes all generated candidates, evaluates them against the same window, and
writes `fno_iteration_report.json` plus `candidate_comparison.json`. The current
V1 planner emits two scheduled-sampling candidates with
`teacher_forcing_decay = 0.85` and `0.9`.

Before any future executor consumes the generated config, audit it:

```bash
uv run --frozen pde-agent audit-experiment-config \
  --config-path runs/fno_iteration_001/experiment_config.json \
  --output-path runs/fno_iteration_001/experiment_config_audit.json
```

The audit writes `EXPERIMENT_CONFIG_AUDITED`. Semantic problems produce a
`rejected` audit report with issue codes instead of silently continuing to
candidate execution.

Prepare an isolated workspace for the first approved candidate:

```bash
uv run --frozen pde-agent prepare-candidate-execution \
  --audit-path runs/fno_iteration_001/experiment_config_audit.json \
  --candidate-index 0 \
  --output-dir runs/fno_iteration_001/candidates \
  --force
```

This writes `candidate_execution_plan.json` plus `artifacts/` and `logs/`
directories under the selected candidate workspace. It still does not run
training; it only prepares the boundary that a future executor will consume.

Execute the V1 FNO scheduled-sampling finetune candidate:

```bash
uv run --frozen --extra torch pde-agent execute-fno-finetune-candidate \
  --candidate-plan-path runs/fno_iteration_001/candidates/use_scheduled_sampling_rollout_finetune_001/candidate_execution_plan.json \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --max-train-steps 16 \
  --learning-rate 1e-4 \
  --force
```

The evidence-backed V1 defaults are `training_strategy = scheduled_sampling`,
`rollout_steps = 10`, `teacher_forcing_decay = 0.85`,
`max_train_steps = 16`, and `learning_rate = 1e-4`. The planner keeps
`teacher_forcing_decay = 0.9` as the nearby fallback candidate. The
`--max-train-steps` flag is optional; omitting it uses the V1 default. Existing
candidate plans that omit `training_strategy` still default to
`scheduled_sampling`.

This executor supports `use_scheduled_sampling_rollout_finetune`. Candidate
plans may also set `training_strategy = late_weighted_scheduled_sampling` for
older experiments, which uses the same rollout finetune loop but linearly
increases loss weight for later rollout steps. It writes `finetuned_fno_checkpoint.pt`,
`candidate_prediction.npz`, and `candidate_execution_report.json`; it does not
claim score improvement.

Evaluate that candidate against the same local windows and optional baseline:

```bash
uv run --frozen --extra torch pde-agent evaluate-fno-candidate \
  --candidate-report-path runs/fno_iteration_001/candidates/use_scheduled_sampling_rollout_finetune_001/artifacts/candidate_execution_report.json \
  --target-path data/derived/local_windows_task1_smoke.npz \
  --output-dir runs/fno_iteration_001/candidates/use_scheduled_sampling_rollout_finetune_001/artifacts/evaluation \
  --baseline-score-path runs/fno_iteration_001/fno_score.json \
  --nu 0.001 \
  --dt 1.0 \
  --dx 1.0 \
  --force
```

The evaluator writes candidate-scoped score, rollout-error, Burgers-physics,
selected-hypothesis, and `candidate_evaluation_report.json` artifacts. This is
the comparison boundary for the next multi-candidate step; short-window
improvement is only a local proxy, not hidden 190-step evidence.

Compare one or more evaluated candidates:

```bash
uv run --frozen pde-agent compare-fno-candidates \
  --evaluation-report-path runs/fno_iteration_001/candidates/use_scheduled_sampling_rollout_finetune_001/artifacts/evaluation/candidate_evaluation_report.json \
  --output-path runs/fno_iteration_001/fno_candidate_comparison_report.json \
  --force
```

For multiple candidates, repeat `--evaluation-report-path`. The comparator sorts
by local score, then `rel_mse`, then `rmse`, then candidate id for deterministic
ties. It writes `fno_candidate_comparison_report.json` with the winner,
rejection reasons, and `next_command = prepare-next-fno-iteration`; it does not
rerun diagnostics or mutate candidate artifacts.

Run a bounded evidence sweep across windows and train-step budgets:

```bash
uv run --frozen --extra torch --extra data pde-agent run-fno-evidence-sweep \
  --source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir runs/fno_evidence_sweep_001 \
  --dataset-key tensor \
  --start-index 0 \
  --start-index 128 \
  --start-index 512 \
  --sample-limit 8 \
  --context-steps 10 \
  --forecast-steps 31 \
  --time-reduction 5 \
  --space-reduction 4 \
  --train-steps 1 \
  --train-steps 3 \
  --train-steps 5 \
  --train-steps 8 \
  --learning-rate 1e-4 \
  --force
```

The sweep writes `evidence_summary.json` and `evidence_summary.csv`, with
per-candidate scores, per-cell winners, aggregate strategy statistics, and a
recommendation on whether to add more training strategies. V0 sweeps one
forecast horizon at a time.
