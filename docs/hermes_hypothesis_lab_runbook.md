# Hermes Hypothesis Lab Runbook

This runbook defines the standard gate sequence for treating Hypothesis Lab as
Hermes infrastructure. Use it before long autonomous Hermes optimization runs or
after changes to Hypothesis Lab memory, repair policy, research chronicle,
candidate planning, runtime audit-chain, or heartbeat plumbing.

## Goal

Hermes should not start long iterations just because a candidate generator can
produce artifacts. A run is allowed to continue only after the system proves
that:

- Hypothesis Lab trace memory can alter candidate planning.
- Audit repair failures can be converted into repair tasks.
- Auto-repair results are summarized into KPI and repair policy memory.
- Research chronicle records the repair experience.
- Candidate planner consumes the repaired memory.
- Runtime audit-chain refuses to start if the memory-internalization gate is
  stale or failed.

## Standard Gate Sequence

### 1. Run auto-repair memory acceptance smoke

Use a tiny Task 1 fixture or a known small submission package. This command
injects a `runtime_memory_bridge_smoke_gate` failure, runs auto-repair, then
checks KPI, repair policy, research chronicle, candidate planner, and strict
runtime audit internalization.

```bash
uv run --frozen pde-agent run-hermes-auto-repair-memory-acceptance-smoke \
  --output-dir runs/auto_repair_memory_acceptance_smoke_YYYYMMDD \
  --trace-path runs/auto_repair_memory_acceptance_smoke_YYYYMMDD/trace.jsonl \
  --initial-submission-dir runs/runtime_memory_bridge_smoke_YYYYMMDD/submission_package/submission \
  --task1-test-path runs/runtime_memory_bridge_smoke_YYYYMMDD/inputs/task1_test.hdf5 \
  --task1-target-path runs/runtime_memory_bridge_smoke_YYYYMMDD/inputs/task1_target.hdf5 \
  --candidate-generator postprocess \
  --force
```

Required report:

```text
runs/auto_repair_memory_acceptance_smoke_YYYYMMDD/hermes_auto_repair_memory_acceptance_smoke.json
```

Required checks:

- `decision == "passed"`
- `checks.auto_repair_passed == true`
- `checks.memory_bridge_passed == true`
- `checks.kpi_internalized == true`
- `checks.repair_policy_internalized == true`
- `checks.chronicle_internalized == true`
- `checks.planner_internalized == true`
- `checks.strict_audit_passed == true`
- `metrics.passed_check_count == metrics.check_count`

If any check fails, do not start long Hermes iterations. Inspect the failed
artifact named in the report and fix that boundary first.

### 2. Run runtime audit-chain with preflight gate

Pass the acceptance smoke report as `--preflight-acceptance-smoke-path`. The
audit chain must reject failed or stale acceptance reports before heartbeat
starts.

```bash
uv run --frozen pde-agent run-hermes-runtime-audit-chain \
  --output-dir runs/preflight_acceptance_audit_chain_YYYYMMDD \
  --trace-path runs/auto_repair_memory_acceptance_smoke_YYYYMMDD/trace.jsonl \
  --initial-submission-dir runs/runtime_memory_bridge_smoke_YYYYMMDD/submission_package/submission \
  --candidate-generator postprocess \
  --candidate-generator-fallback auto \
  --iterations 1 \
  --max-batches 1 \
  --max-rounds-per-batch 1 \
  --batch-deadline-seconds 30 \
  --submit-score-threshold 130 \
  --task1-test-path runs/runtime_memory_bridge_smoke_YYYYMMDD/inputs/task1_test.hdf5 \
  --task1-target-path runs/runtime_memory_bridge_smoke_YYYYMMDD/inputs/task1_target.hdf5 \
  --preflight-acceptance-smoke-path runs/auto_repair_memory_acceptance_smoke_YYYYMMDD/hermes_auto_repair_memory_acceptance_smoke.json \
  --force
```

Required report:

```text
runs/preflight_acceptance_audit_chain_YYYYMMDD/hermes_runtime_audit_chain_report.json
```

Required checks:

- `decision == "passed"` or `decision == "repaired"`
- `metrics.preflight_acceptance_smoke_passed == true`
- `artifacts.preflight_acceptance_smoke_path` points to the passed acceptance
  smoke report.
- The trace event includes `preflight_acceptance_smoke_path` and
  `preflight_acceptance_check_count`.

### 3. Run the Hermes infrastructure canary

After changing any Hermes-facing Hypothesis Lab protocol, run the real Hermes
fixture that proves Hermes can read the gate contract, execute the deterministic
infrastructure audit command, and stop at the expected success transition.

```bash
uv run --frozen python scripts/run_hermes_hypothesis_lab_infrastructure.py \
  --timeout-seconds 300
```

Audit the latest saved session without rerunning Hermes:

```bash
uv run --frozen python scripts/run_hermes_hypothesis_lab_infrastructure.py \
  --audit-only
```

Required audit result:

- `decision == "passed"`
- terminal command list exactly matches `run_state.yaml`
- final state is `HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED`
- `metrics.decision == "passed"`
- `metrics.failed_check_count == 0`

### 4. Start longer Hermes iterations

Only after steps 1, 2, and 3 pass, increase `--iterations`, `--max-batches`,
`--max-rounds-per-batch`, and the time budget. Keep both preflight gates enabled
for the long run:

```bash
uv run --frozen pde-agent run-hermes-runtime-audit-chain \
  --output-dir runs/hermes_runtime_audit_chain_long_YYYYMMDD \
  --trace-path runs/auto_repair_memory_acceptance_smoke_YYYYMMDD/trace.jsonl \
  --initial-submission-dir runs/runtime_memory_bridge_smoke_YYYYMMDD/submission_package/submission \
  --candidate-generator postprocess \
  --candidate-generator-fallback auto \
  --iterations 10 \
  --max-batches 5 \
  --max-rounds-per-batch 20 \
  --batch-deadline-seconds 3600 \
  --submit-score-threshold 130 \
  --task1-test-path runs/runtime_memory_bridge_smoke_YYYYMMDD/inputs/task1_test.hdf5 \
  --task1-target-path runs/runtime_memory_bridge_smoke_YYYYMMDD/inputs/task1_target.hdf5 \
  --preflight-acceptance-smoke-path runs/auto_repair_memory_acceptance_smoke_YYYYMMDD/hermes_auto_repair_memory_acceptance_smoke.json \
  --preflight-infrastructure-audit-path integration_tests/hermes_hypothesis_lab_infrastructure/reports/audit_YYYYMMDD_HHMMSS_SESSION.json \
  --force
```

Recommended long-run defaults:

- Keep `--candidate-generator-fallback auto`.
- Keep `--auto-repair` enabled only when the repair task command is supported
  and all required inputs are available.
- Keep `--trace-path` stable across batches so KPI and repair policy memory can
  accumulate.
- If the chain returns `audit_repair_tasks_json`, inspect or resolve it before
  rerunning.

## Stop Conditions

Stop the current run and diagnose instead of continuing when any of these occur:

- Acceptance smoke fails.
- Preflight audit-chain rejects the smoke report.
- `repair_convergence.decision == "nonconverged"` for the same error twice.
- Candidate planner schedules a family listed in `avoid_candidate_families`.
- Heartbeat reports candidate-space exhaustion without an expansion handoff.
- Runtime audit returns `failed_checks` after auto-repair.

## Real Verification Snapshot

The following local verification demonstrated the current gate sequence:

- Acceptance smoke report:
  `runs/auto_repair_memory_acceptance_smoke_20260504/hermes_auto_repair_memory_acceptance_smoke.json`
- Preflight audit-chain report:
  `runs/preflight_acceptance_audit_chain_20260504/hermes_runtime_audit_chain_report.json`
- Hypothesis Lab infrastructure canary:
  `integration_tests/hermes_hypothesis_lab_infrastructure/reports/audit_20260504_121519_9bc312.json`
- Dual-preflight audit-chain report:
  `runs/preflight_infrastructure_audit_chain_20260504/hermes_runtime_audit_chain_report.json`
- Dual-preflight stability chain report:
  `runs/dual_preflight_stability_chain_20260504/hermes_runtime_audit_chain_report.json`
- Dual-preflight candidate-pressure chain report:
  `runs/dual_preflight_candidate_pressure_chain_20260504/hermes_runtime_audit_chain_report.json`
- Readiness audit report:
  `runs/hermes_hypothesis_lab_readiness_audit_20260504/audit.json`
- Observed preflight metrics:
  `preflight_acceptance_smoke_passed=true`,
  `preflight_acceptance_check_count=7`,
  `preflight_infrastructure_audit_passed=true`,
  `preflight_infrastructure_check_count=13`
- Observed stability metrics:
  `completed_iteration_count=2`,
  `failed_audit_count=0`,
  `candidate_space_exhausted_count=0`,
  `chain_learning_signals=["score_stagnation"]`
- Observed candidate-pressure metrics:
  first iteration attempted `sem_window_gate_blend_multi_early_smoke`,
  second iteration guarded the repeated failed family and fell back to
  `postprocess`,
  `completed_iteration_count=2`,
  `failed_audit_count=0`,
  `chain_learning_signals=["score_stagnation"]`
- Readiness audit metrics:
  `check_count=20`,
  `failed_check_count=0`
- Test suite:
  `uv run --frozen pytest -q` -> `960 passed`

Generated run artifacts under `runs/` are evidence, not source. Re-run the gates
after changing any Hermes-facing protocol or Hypothesis Lab memory component.

The stability chain proves the dual-preflight gate survives repeated runtime
iterations. It does not prove the current candidate family can keep improving
Task 1 score; `score_stagnation` is an optimization signal, not an infrastructure
failure.

The candidate-pressure chain proves the gate also survives a non-trivial early
candidate family and records the resulting failure memory. It shows the current
small smoke fixture is already saturated, so future optimization should use a
larger Task 1 validation slice or a new candidate family rather than repeating
the same `sem_window_gate_blend_multi_early_smoke` smoke.

The readiness audit is the preferred single artifact to cite before a long
Hermes run because it checks acceptance memory internalization, dual-preflight
runtime stability, real Hermes infrastructure canary evidence, and a
candidate-pressure run in one report.

## Cluster Saturation And Expansion Gate

When KPI memory or prior expansion evidence marks a candidate cluster as
saturated, the candidate planner must apply heartbeat registry
`saturation_blocked_by` metadata before building a schedule. A saturated local
strategy cluster should exclude the sem-window family at planning time and
record `blocked_by_cluster_saturation`; it should not let heartbeat discover the
same block after scheduling.

Long runs should handle this state by recording `candidate_space_exhausted` and
writing expansion handoff artifacts. Do not replace auto fallback with an
explicit heavy fallback such as `sem_window_gate_blend_multi_search`: guarded
fallbacks are not auto-fallback eligible and should fail
`candidate_generator_registry_policy` if forced into the chain.

Current medium-run evidence after applying planner-side cluster saturation:

- Runtime audit-chain report:
  `runs/hermes_task1_agent_capability_medium_sprint_fix_20260504/chain_auto/hermes_runtime_audit_chain_report.json`
- Task 1 agent capability benchmark:
  `runs/hermes_task1_agent_capability_medium_sprint_fix_20260504/task1_agent_capability_benchmark.json`
- Observed audit-chain metrics:
  `requested_iteration_count=5`,
  `completed_iteration_count=5`,
  `failed_audit_count=0`,
  `audit_repair_task_count=0`,
  `candidate_space_exhausted_count=5`
- Observed benchmark metrics:
  `decision="infrastructure_only"`,
  `best_score=120.09092223481395`,
  `score_delta=0.0`,
  `evidence_event_count=43`,
  `invalid_promotion_count=0`

This evidence is long-run infrastructure evidence, not a score-improvement
claim. It proves that a saturated candidate space now enters the expansion path
cleanly for multiple iterations while preserving audit validity.

## Task1 Candidate Factory After Saturation

After a stable chain returns `candidate_space_exhausted`, run the Task1
candidate factory before starting another long heartbeat. This keeps candidate
space expansion explicit and auditable:

```bash
uv run --frozen pde-agent generate-task1-candidate-families \
  --benchmark-path runs/.../task1_agent_capability_benchmark.json \
  --chain-report-path runs/.../hermes_runtime_audit_chain_report.json \
  --output-path runs/.../task1_candidate_families.json \
  --max-families 5

uv run --frozen pde-agent run-candidate-family-sandbox \
  --family-report-path runs/.../task1_candidate_families.json \
  --output-path runs/.../task1_candidate_family_sandbox.json

uv run --frozen pde-agent evaluate-candidate-factory-capability \
  --family-report-path runs/.../task1_candidate_families.json \
  --sandbox-report-path runs/.../task1_candidate_family_sandbox.json \
  --output-path runs/.../task1_candidate_factory_capability.json
```

Required checks before returning to heartbeat:

- `TASK1_CANDIDATE_FAMILIES_GENERATED`
- `candidate_family_count >= 4`
- `TASK1_CANDIDATE_FAMILY_SANDBOX_PLANNED`
- `planned_count >= 1`
- `TASK1_CANDIDATE_FACTORY_CAPABILITY_EVALUATED`
- `decision == "passed"`
- `heartbeat_handoff.ready == true`

The factory is intentionally bounded. It validates CandidateFamilyCard artifacts
and emits sandbox plans; it does not execute arbitrary generated code or promote
untested strategies.
