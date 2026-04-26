# Command: diagnose-rollout-error

Use this after `score-local-rollout` has produced a score JSON.

Required inputs:
- `prediction_path`: NPZ containing `prediction`
- `target_path`: windows NPZ containing `target`
- `score_path`: JSON produced by `score-local-rollout`
- `output_path`: rollout diagnosis JSON path

Command:

```bash
uv run --frozen pde-agent diagnose-rollout-error \
  --prediction-path <prediction_path> \
  --target-path <target_path> \
  --score-path <score_path> \
  --output-path <output_path>
```

Expected artifact:
- JSON at `output_path` with `dominant_error_pattern` and evidence.

Forbidden shortcuts:
- Do not infer the dominant error pattern without running this command.
- Do not ignore the score JSON.

Research log summary:
- Record dominant error pattern, global RMSE, time error slope, and two evidence strings.
