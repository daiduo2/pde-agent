# Command: select-pde-hypothesis

Use this after score, rollout error diagnosis, and Burgers physics check artifacts exist.

Required inputs:
- `score_path`
- `error_diagnosis_path`
- `physics_check_path`
- `output_path`

Command:

```bash
uv run --frozen pde-agent select-pde-hypothesis \
  --score-path <score_path> \
  --error-diagnosis-path <error_diagnosis_path> \
  --physics-check-path <physics_check_path> \
  --output-path <output_path>
```

Expected artifact:
- JSON at `output_path` with `selected_hypothesis`, rationale, expected effect, risk, and rejected hypotheses.

Forbidden shortcuts:
- Do not choose a hypothesis before both diagnostics exist.
- Do not edit model code in this step.

Research log summary:
- Record selected hypothesis, rationale, expected effect, and risk.
