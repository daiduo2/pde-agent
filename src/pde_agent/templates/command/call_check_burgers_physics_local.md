# Command: check-burgers-physics-local

Use this after a prediction NPZ and target windows NPZ exist.

Required inputs:
- `prediction_path`: NPZ containing `prediction`
- `target_path`: windows NPZ containing `target`
- `output_path`: physics check JSON path
- `nu`: Burgers viscosity, default `0.001`

Command:

```bash
uv run --frozen pde-agent check-burgers-physics-local \
  --prediction-path <prediction_path> \
  --target-path <target_path> \
  --output-path <output_path> \
  --nu 0.001
```

Expected artifact:
- JSON at `output_path` with `physics_issue`, residual metrics, and variation metrics.

Forbidden shortcuts:
- Do not claim physical consistency from NRMSE alone.
- Do not treat this finite-difference proxy as an exact PDE proof.

Research log summary:
- Record physics issue, residual ratio, total variation ratio, and shock-region error ratio.
