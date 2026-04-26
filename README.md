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

Before any future executor consumes the generated config, audit it:

```bash
uv run --frozen pde-agent audit-experiment-config \
  --config-path runs/fno_iteration_001/experiment_config.json \
  --output-path runs/fno_iteration_001/experiment_config_audit.json
```

The audit writes `EXPERIMENT_CONFIG_AUDITED`. Semantic problems produce a
`rejected` audit report with issue codes instead of silently continuing to
candidate execution.
