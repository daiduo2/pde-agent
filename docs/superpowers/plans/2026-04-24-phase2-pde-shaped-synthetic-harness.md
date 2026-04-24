# Phase 2 PDE-Shaped Synthetic Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `pde-agent` from the `BASELINE_READY` smoke test into a full synthetic PDE-shaped harness round that reaches `RESULTS_COLLECTED` through `eval-rollout`, `record-timing`, and `collect-artifacts`.

**Architecture:** Keep the harness domain-light but PDE-shaped. The three command handlers will emit deterministic synthetic artifacts with the same file/layout/schema shape we expect from later PDE task adapters. The Hermes smoke fixture will then exercise the exact command chain end-to-end without introducing real Burgers logic or domain skills.

**Tech Stack:** Python 3.11+, Typer CLI, stdlib `json`/`pathlib`, pytest, Hermes protocol smoke fixture files under `integration_tests/hermes_protocol_smoke`.

---

### Task 1: Add failing tests for the Phase 2 synthetic command outputs

**Files:**
- Create: `tests/test_phase2_synthetic_harness.py`
- Reference: `tests/test_manifest_builder.py`

- [ ] **Step 1: Write the failing test file**

```python
import json
from pathlib import Path

from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.record_timing import record_timing


def test_eval_rollout_writes_pde_shaped_synthetic_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "hermes_protocol_smoke_003"
    for subdir in ("metrics", "plots", "artifacts"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    result = eval_rollout(run_id="hermes_protocol_smoke_003", run_root=str(run_root))

    rollout_path = run_root / "metrics" / "rollout.json"
    curve_path = run_root / "plots" / "error_curve.png"
    prediction_path = run_root / "artifacts" / "prediction.hdf5"

    assert result["status"] == "ok"
    assert result["state_hint"] == "RESULTS_COLLECTED"
    assert rollout_path.exists()
    assert curve_path.exists()
    assert prediction_path.exists()

    payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "synthetic_pde_fixture"
    assert payload["one_step"]["nrmse"] == 0.031
    assert payload["long_horizon"]["nrmse"] == 0.087
    assert payload["curve"]["path"] == str(curve_path)
    assert payload["prediction"]["path"] == str(prediction_path)


def test_record_timing_writes_json_and_time_csv(tmp_path: Path) -> None:
    run_root = tmp_path / "hermes_protocol_smoke_003"
    for subdir in ("metrics", "submission"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    result = record_timing(run_id="hermes_protocol_smoke_003", run_root=str(run_root))

    timing_path = run_root / "metrics" / "timing.json"
    time_csv_path = run_root / "submission" / "time.csv"

    assert result["status"] == "ok"
    assert result["state_hint"] == "RESULTS_COLLECTED"
    assert timing_path.exists()
    assert time_csv_path.exists()

    payload = json.loads(timing_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "synthetic_pde_fixture"
    assert payload["train_time_seconds"] == 12450.0
    assert payload["inference_time_seconds"] == 61.2
    assert payload["time_csv_path"] == str(time_csv_path)
    assert time_csv_path.read_text(encoding="utf-8") == "train_time,inference_time\\n12450.0,61.2\\n"


def test_collect_artifacts_writes_manifest_for_results_collected(tmp_path: Path) -> None:
    run_root = tmp_path / "hermes_protocol_smoke_003"
    for subdir in ("config", "metrics", "plots", "artifacts", "submission"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    (run_root / "config" / "resolved.json").write_text("{}", encoding="utf-8")
    (run_root / "metrics" / "rollout.json").write_text(
        json.dumps(
            {
                "run_id": "hermes_protocol_smoke_003",
                "one_step": {"nrmse": 0.031},
                "long_horizon": {"nrmse": 0.087},
                "curve": {"path": str(run_root / "plots" / "error_curve.png")},
                "prediction": {"path": str(run_root / "artifacts" / "prediction.hdf5")},
                "mode": "synthetic_pde_fixture",
            }
        ),
        encoding="utf-8",
    )
    (run_root / "metrics" / "timing.json").write_text(
        json.dumps(
            {
                "run_id": "hermes_protocol_smoke_003",
                "train_time_seconds": 12450.0,
                "inference_time_seconds": 61.2,
                "time_csv_path": str(run_root / "submission" / "time.csv"),
                "mode": "synthetic_pde_fixture",
            }
        ),
        encoding="utf-8",
    )

    result = collect_artifacts(
        run_id="hermes_protocol_smoke_003",
        run_root=str(run_root),
        config_hash="synthetic-smoke-003",
    )

    manifest_path = run_root / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "RESULTS_COLLECTED"
    assert manifest["phase"] == "results_collected"
    assert manifest["summary"]["one_step_nrmse"] == 0.031
    assert manifest["summary"]["long_horizon_nrmse"] == 0.087
    assert manifest["summary"]["train_time_seconds"] == 12450.0
    assert manifest["summary"]["inference_time_seconds"] == 61.2
    assert manifest["artifacts"]["prediction_hdf5_path"] == str(run_root / "artifacts" / "prediction.hdf5")
    assert manifest["artifacts"]["time_csv_path"] == str(run_root / "submission" / "time.csv")
    assert manifest["completeness"]["has_rollout"] is True
    assert manifest["completeness"]["has_timing"] is True
```

- [ ] **Step 2: Run the new test file to verify it fails**

Run:

```bash
cd /Users/daiduo2/pde-agent
uv run pytest tests/test_phase2_synthetic_harness.py -v
```

Expected:
- `eval-rollout` test fails because `prediction.hdf5`, `error_curve.png`, `mode`, and the fixed metric values do not exist yet.
- `record-timing` test fails because `time.csv` is not written and values are still `0.0`.
- `collect-artifacts` test fails because manifest summary/artifacts/state hint do not match the PDE-shaped synthetic contract.

- [ ] **Step 3: Commit the red test baseline**

```bash
cd /Users/daiduo2/pde-agent
git add tests/test_phase2_synthetic_harness.py
git commit -m "test: add phase 2 synthetic harness expectations"
```


### Task 2: Implement a shared synthetic PDE fixture helper

**Files:**
- Create: `src/pde_agent/core/synthetic_pde_fixture.py`
- Reference: `src/pde_agent/core/manifest_builder.py`
- Test: `tests/test_phase2_synthetic_harness.py`

- [ ] **Step 1: Add the helper module**

```python
from pathlib import Path


SYNTHETIC_PDE_MODE = "synthetic_pde_fixture"
SYNTHETIC_ONE_STEP_NRMSE = 0.031
SYNTHETIC_LONG_HORIZON_NRMSE = 0.087
SYNTHETIC_TRAIN_TIME_SECONDS = 12450.0
SYNTHETIC_INFERENCE_TIME_SECONDS = 61.2


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_placeholder_file(path: Path, content: bytes) -> str:
    ensure_parent(path).write_bytes(content)
    return str(path)


def synthetic_rollout_payload(run_id: str, run_root: Path) -> dict[str, object]:
    curve_path = run_root / "plots" / "error_curve.png"
    prediction_path = run_root / "artifacts" / "prediction.hdf5"
    write_placeholder_file(curve_path, b"synthetic-error-curve")
    write_placeholder_file(prediction_path, b"synthetic-hdf5-placeholder")
    return {
        "run_id": run_id,
        "one_step": {"nrmse": SYNTHETIC_ONE_STEP_NRMSE},
        "long_horizon": {"nrmse": SYNTHETIC_LONG_HORIZON_NRMSE},
        "curve": {"path": str(curve_path)},
        "prediction": {"path": str(prediction_path)},
        "mode": SYNTHETIC_PDE_MODE,
    }


def synthetic_timing_payload(run_id: str, run_root: Path) -> dict[str, object]:
    time_csv_path = run_root / "submission" / "time.csv"
    ensure_parent(time_csv_path).write_text(
        "train_time,inference_time\\n12450.0,61.2\\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "train_time_seconds": SYNTHETIC_TRAIN_TIME_SECONDS,
        "inference_time_seconds": SYNTHETIC_INFERENCE_TIME_SECONDS,
        "time_csv_path": str(time_csv_path),
        "mode": SYNTHETIC_PDE_MODE,
    }
```

- [ ] **Step 2: Run the targeted tests again**

Run:

```bash
cd /Users/daiduo2/pde-agent
uv run pytest tests/test_phase2_synthetic_harness.py -v
```

Expected:
- Tests still fail, but only because the command handlers are not wired to the new helper yet.

- [ ] **Step 3: Commit the shared helper**

```bash
cd /Users/daiduo2/pde-agent
git add src/pde_agent/core/synthetic_pde_fixture.py
git commit -m "feat: add shared synthetic PDE fixture helpers"
```


### Task 3: Wire the three commands to emit PDE-shaped synthetic artifacts

**Files:**
- Modify: `src/pde_agent/commands/eval_rollout.py`
- Modify: `src/pde_agent/commands/record_timing.py`
- Modify: `src/pde_agent/commands/collect_artifacts.py`
- Test: `tests/test_phase2_synthetic_harness.py`

- [ ] **Step 1: Update `eval_rollout.py`**

Replace the function body with:

```python
import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.synthetic_pde_fixture import (
    SYNTHETIC_LONG_HORIZON_NRMSE,
    SYNTHETIC_ONE_STEP_NRMSE,
    synthetic_rollout_payload,
)


def eval_rollout(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    rollout_path = root / "metrics" / "rollout.json"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_payload = synthetic_rollout_payload(run_id=run_id, run_root=root)
    rollout_path.write_text(json.dumps(rollout_payload, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="eval-rollout",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "rollout_metrics_path": str(rollout_path),
            "error_curve_path": rollout_payload["curve"]["path"],
            "prediction_hdf5_path": rollout_payload["prediction"]["path"],
        },
        metrics={
            "one_step_nrmse": SYNTHETIC_ONE_STEP_NRMSE,
            "long_horizon_nrmse": SYNTHETIC_LONG_HORIZON_NRMSE,
        },
    )
```

- [ ] **Step 2: Update `record_timing.py`**

Replace the function body with:

```python
import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.synthetic_pde_fixture import (
    SYNTHETIC_INFERENCE_TIME_SECONDS,
    SYNTHETIC_TRAIN_TIME_SECONDS,
    synthetic_timing_payload,
)


def record_timing(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    timing_path = root / "metrics" / "timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    payload = synthetic_timing_payload(run_id=run_id, run_root=root)
    timing_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="record-timing",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "timing_metrics_path": str(timing_path),
            "time_csv_path": payload["time_csv_path"],
        },
        metrics={
            "train_time_seconds": SYNTHETIC_TRAIN_TIME_SECONDS,
            "inference_time_seconds": SYNTHETIC_INFERENCE_TIME_SECONDS,
        },
    )
```

- [ ] **Step 3: Update `collect_artifacts.py`**

Replace the function body with:

```python
import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.manifest_builder import build_manifest


def collect_artifacts(run_id: str, run_root: str, config_hash: str = "unknown") -> dict[str, object]:
    root = Path(run_root)
    rollout_payload = json.loads((root / "metrics" / "rollout.json").read_text(encoding="utf-8"))
    timing_payload = json.loads((root / "metrics" / "timing.json").read_text(encoding="utf-8"))
    manifest = build_manifest(
        run_id=run_id,
        config={
            "resolved_config_path": str(root / "config" / "resolved.json"),
            "config_hash": config_hash,
        },
        artifacts={
            "rollout_metrics_path": str(root / "metrics" / "rollout.json"),
            "timing_metrics_path": str(root / "metrics" / "timing.json"),
            "prediction_hdf5_path": rollout_payload["prediction"]["path"],
            "error_curve_path": rollout_payload["curve"]["path"],
            "time_csv_path": timing_payload["time_csv_path"],
        },
        summary={
            "one_step_nrmse": rollout_payload["one_step"]["nrmse"],
            "long_horizon_nrmse": rollout_payload["long_horizon"]["nrmse"],
            "train_time_seconds": timing_payload["train_time_seconds"],
            "inference_time_seconds": timing_payload["inference_time_seconds"],
        },
        completeness={
            "has_rollout": True,
            "has_spectrum": False,
            "has_physics_check": False,
            "has_timing": True,
        },
        phase="results_collected",
    )
    manifest_path = root / "artifacts" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return success_payload(
        run_id=run_id,
        command="collect-artifacts",
        state_hint="RESULTS_COLLECTED",
        artifacts={"artifact_manifest_path": str(manifest_path)},
        metrics={"artifact_count": 5},
    )
```

- [ ] **Step 4: Run the targeted tests to green**

Run:

```bash
cd /Users/daiduo2/pde-agent
uv run pytest tests/test_phase2_synthetic_harness.py -v
```

Expected:
- `3 passed`

- [ ] **Step 5: Commit the command wiring**

```bash
cd /Users/daiduo2/pde-agent
git add src/pde_agent/commands/eval_rollout.py src/pde_agent/commands/record_timing.py src/pde_agent/commands/collect_artifacts.py tests/test_phase2_synthetic_harness.py
git commit -m "feat: emit PDE-shaped synthetic harness artifacts"
```


### Task 4: Upgrade the Hermes smoke fixture to round 003

**Files:**
- Modify: `integration_tests/hermes_protocol_smoke/run_state.yaml`
- Modify: `integration_tests/hermes_protocol_smoke/bootstrap_prompt.md`
- Modify: `integration_tests/hermes_protocol_smoke/expected_actions.md`
- Modify: `integration_tests/hermes_protocol_smoke/README.md`

- [ ] **Step 1: Update `run_state.yaml`**

Set these exact values:

```yaml
run_id: hermes_protocol_smoke_003
run_root: /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003
working_directory: /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke
command_prefix: /Users/daiduo2/pde-agent/.venv/bin/pde-agent

current_state: TASK_PARSED

allowed_actions:
  - read_run_state
  - read_bootstrap_prompt
  - read_task_spec
  - read_base_config
  - read_override
  - call_init_run
  - call_resolve_config
  - call_eval_rollout
  - call_record_timing
  - call_collect_artifacts

forbidden_actions:
  - patch_code
  - run_training
  - call_analyze_spectrum
  - call_check_physics
  - call_build_submission
  - guess_missing_paths
  - skip_required_reads
  - search_outside_current_directory
  - inspect_command_help
  - probe_command_surface

required_reads:
  - /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/run_state.yaml
  - /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/bootstrap_prompt.md
  - /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/task_spec.json
  - /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/base_config.json
  - /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/override.json

required_artifacts: []

allowed_commands:
  - name: init_run
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent init-run --run-id hermes_protocol_smoke_003 --runs-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs
  - name: resolve_config
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent resolve-config --run-id hermes_protocol_smoke_003 --base-config-file /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/base_config.json --override-file /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/override.json --task-spec-file /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/task_spec.json --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003
  - name: eval_rollout
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent eval-rollout --run-id hermes_protocol_smoke_003 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003
  - name: record_timing
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent record-timing --run-id hermes_protocol_smoke_003 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003
  - name: collect_artifacts
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent collect-artifacts --run-id hermes_protocol_smoke_003 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003 --config-hash synthetic-smoke-003

success_transition:
  on_complete: RESULTS_COLLECTED
```

- [ ] **Step 2: Update `bootstrap_prompt.md`**

Keep the absolute-path requirements and add these constraints:

```md
Do not stop after BASELINE_READY.
Continue until the allowed success transition RESULTS_COLLECTED is reached.
Use only the exact command strings declared in `run_state.yaml`.
```

- [ ] **Step 3: Update `expected_actions.md` and `README.md`**

Make the action sequence and success criteria explicitly include:
- exact `eval_rollout`
- exact `record_timing`
- exact `collect_artifacts`
- stop only after `RESULTS_COLLECTED`
- `manifest.json` must exist under `runs/hermes_protocol_smoke_003/artifacts/manifest.json`

- [ ] **Step 4: Commit the fixture upgrade**

```bash
cd /Users/daiduo2/pde-agent
git add integration_tests/hermes_protocol_smoke/run_state.yaml integration_tests/hermes_protocol_smoke/bootstrap_prompt.md integration_tests/hermes_protocol_smoke/expected_actions.md integration_tests/hermes_protocol_smoke/README.md
git commit -m "test: extend Hermes smoke fixture to results collected"
```


### Task 5: Verify the CLI and run the second-round Hermes smoke test

**Files:**
- Verify: `runs/hermes_protocol_smoke_003/config/resolved.json`
- Verify: `runs/hermes_protocol_smoke_003/metrics/rollout.json`
- Verify: `runs/hermes_protocol_smoke_003/metrics/timing.json`
- Verify: `runs/hermes_protocol_smoke_003/artifacts/manifest.json`
- Verify: `runs/hermes_protocol_smoke_003/submission/time.csv`
- Review: `integration_tests/hermes_protocol_smoke/hermes_home/sessions/`

- [ ] **Step 1: Run the full repo test suite**

Run:

```bash
cd /Users/daiduo2/pde-agent
uv run pytest tests -v
```

Expected:
- all tests pass

- [ ] **Step 2: Dry-run the five exact commands outside Hermes**

Run:

```bash
cd /Users/daiduo2
/Users/daiduo2/pde-agent/.venv/bin/pde-agent init-run --run-id hermes_protocol_smoke_003 --runs-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs
/Users/daiduo2/pde-agent/.venv/bin/pde-agent resolve-config --run-id hermes_protocol_smoke_003 --base-config-file /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/base_config.json --override-file /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/override.json --task-spec-file /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/task_spec.json --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003
/Users/daiduo2/pde-agent/.venv/bin/pde-agent eval-rollout --run-id hermes_protocol_smoke_003 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003
/Users/daiduo2/pde-agent/.venv/bin/pde-agent record-timing --run-id hermes_protocol_smoke_003 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003
/Users/daiduo2/pde-agent/.venv/bin/pde-agent collect-artifacts --run-id hermes_protocol_smoke_003 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/runs/hermes_protocol_smoke_003 --config-hash synthetic-smoke-003
```

Expected:
- each command exits `0`
- `manifest.json` exists and contains `phase = results_collected`

- [ ] **Step 3: Launch the Hermes smoke round**

Run:

```bash
cd /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke
hermes -p hermes-protocol-smoke
```

Send this first message:

```text
This is a protocol-compliance smoke test for pde-agent.

Before taking any action, read these files in order using the exact absolute paths below:
1. /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/run_state.yaml
2. /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/bootstrap_prompt.md
3. /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/task_spec.json
4. /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/base_config.json
5. /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/override.json

Hard rules:
- Treat /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/run_state.yaml as the source of truth.
- Do not rely on the current working directory.
- Do not search for alternate file locations.
- Do not guess missing paths.
- Do not skip required reads.
- Do not patch code.
- Do not run training.
- Do not inspect command help.
- Do not probe command surfaces.
- Do not call commands outside the allowed set for the current state.
- Use only the exact command strings declared in run_state.yaml.
- Read command outputs as JSON only.
- Do not stop after BASELINE_READY.
- Continue until the allowed success transition RESULTS_COLLECTED is reached.
- If required information is missing, stop and report the missing file or field instead of inferring it.

After reading the files, briefly state:
- current_state
- allowed next actions
- forbidden actions

Then continue by following the allowed action sequence only.
```

- [ ] **Step 4: Review the isolated Hermes evidence**

Check:

```bash
ls -lt /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/hermes_home/sessions | head -5
tail -n 200 /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/hermes_home/logs/agent.log
tail -n 120 /Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/hermes_home/logs/errors.log
```

Expected:
- latest session reads the five required absolute-path files
- latest session executes exactly five allowed commands
- no `--help`
- no repository search
- session ends at `RESULTS_COLLECTED`

- [ ] **Step 5: Commit the verification-ready Phase 2 state**

```bash
cd /Users/daiduo2/pde-agent
git add src/pde_agent tests integration_tests/hermes_protocol_smoke .gitignore
git commit -m "test: verify phase 2 synthetic harness smoke round"
```
