# Command: generate-baseline-prediction

Use this when a local windows NPZ exists and no prediction NPZ exists yet.

Required inputs:
- `windows_path`: NPZ containing `context` and `target`
- `output_path`: desired prediction NPZ

Command:

```bash
uv run --frozen pde-agent generate-baseline-prediction \
  --windows-path <windows_path> \
  --output-path <output_path> \
  --method persistence
```

Expected artifact:
- NPZ at `output_path` with key `prediction`

Forbidden shortcuts:
- Do not invent a prediction file path that was not requested.
- Do not read target values to construct the prediction.

Research log summary:
- Record method, sample count, forecast steps, and output path.
