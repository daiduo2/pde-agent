# PDE Diagnostic Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic PDE diagnostic skills that let Hermes move from synthetic rollout evidence to error interpretation and hypothesis selection.

**Architecture:** Keep the Phase 3 layer CLI-first and fixture-driven. Command handlers write small JSON artifacts under a run directory, `collect-artifacts` upgrades the manifest when diagnostic artifacts exist, and a separate Hermes diagnostic smoke fixture audits the exact command chain through `HYPOTHESIS_SELECTED`.

**Tech Stack:** Python 3.11+, Typer, stdlib `json`/`pathlib`, pytest, Hermes CLI through the existing `script` PTY runner pattern.

---

## File Structure

- `src/pde_agent/core/diagnostic_synthesis.py`: deterministic payload builders for spectrum metrics, physics checks, error interpretation, and hypothesis selection.
- `src/pde_agent/commands/analyze_spectrum.py`: reads `metrics/rollout.json`, writes `metrics/spectrum.json`.
- `src/pde_agent/commands/check_physics.py`: reads `metrics/rollout.json`, writes `metrics/physics_check.json`.
- `src/pde_agent/commands/interpret_error.py`: reads rollout/spectrum/physics JSON, writes `diagnostics/error_interpretation.json`.
- `src/pde_agent/commands/select_hypothesis.py`: reads interpretation JSON, writes `diagnostics/selected_hypothesis.json`.
- `src/pde_agent/commands/collect_artifacts.py`: detects whether diagnostic artifacts exist and writes either `results_collected` or `hypothesis_selected` manifest.
- `src/pde_agent/cli.py`: exposes `interpret-error` and `select-hypothesis`; remove the duplicated `resolve-config` command definition while editing.
- `src/pde_agent/core/hermes_diagnostic_audit.py`: deterministic audit logic for saved Hermes diagnostic sessions.
- `scripts/audit_hermes_diagnostic.py`: CLI wrapper for diagnostic audit.
- `scripts/run_hermes_diagnostic.py`: real Hermes runner for the diagnostic fixture, using the same PTY approach as `run_hermes_smoke.py`.
- `integration_tests/hermes_diagnostic_smoke/*`: dedicated diagnostic fixture, separate from the existing protocol smoke test.
- `tests/test_phase3_diagnostic_skills.py`: command-level tests.
- `tests/test_hermes_diagnostic_audit.py`: audit-level tests with synthetic session JSON.

---

### Task 1: Add failing tests for diagnostic command outputs

**Files:**
- Create: `tests/test_phase3_diagnostic_skills.py`
- Reference: `tests/test_phase2_synthetic_harness.py`

- [ ] **Step 1: Write the failing test file**

```python
import json
from pathlib import Path

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.select_hypothesis import select_hypothesis


def _prepare_rollout(run_root: Path) -> None:
    eval_rollout(run_id="diagnostic_smoke_001", run_root=str(run_root))


def test_analyze_spectrum_writes_deterministic_spectrum_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)

    result = analyze_spectrum(run_id="diagnostic_smoke_001", run_root=str(run_root))

    spectrum_path = run_root / "metrics" / "spectrum.json"
    payload = json.loads(spectrum_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "SPECTRUM_ANALYZED"
    assert result["artifacts"]["spectrum_metrics_path"] == str(spectrum_path)
    assert payload["mode"] == "synthetic_pde_diagnostic"
    assert payload["low_frequency_error"] == 0.024
    assert payload["mid_frequency_error"] == 0.052
    assert payload["high_frequency_error"] == 0.141
    assert payload["spectral_drift_score"] == 0.37
    assert payload["shock_region_proxy"] == 0.68
    assert payload["dominant_spectral_issue"] == "excessive_smoothing"


def test_check_physics_writes_deterministic_physics_indicators(tmp_path: Path) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)

    result = check_physics(run_id="diagnostic_smoke_001", run_root=str(run_root))

    physics_path = run_root / "metrics" / "physics_check.json"
    payload = json.loads(physics_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "PHYSICS_CHECKED"
    assert result["artifacts"]["physics_check_path"] == str(physics_path)
    assert payload["mode"] == "synthetic_pde_diagnostic"
    assert payload["mass_drift"] == 0.006
    assert payload["energy_decay_ratio"] == 0.82
    assert payload["total_variation_ratio"] == 0.71
    assert payload["nonphysical_oscillation"] is False
    assert payload["blowup_like_behavior"] is False
    assert payload["excessive_smoothing"] is True
    assert payload["physics_issue"] == "overdissipation"


def test_interpret_error_writes_failure_mode_from_rollout_spectrum_and_physics(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)
    analyze_spectrum(run_id="diagnostic_smoke_001", run_root=str(run_root))
    check_physics(run_id="diagnostic_smoke_001", run_root=str(run_root))

    result = interpret_error(run_id="diagnostic_smoke_001", run_root=str(run_root))

    interpretation_path = run_root / "diagnostics" / "error_interpretation.json"
    payload = json.loads(interpretation_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "ERROR_INTERPRETED"
    assert result["artifacts"]["error_interpretation_path"] == str(interpretation_path)
    assert payload["dominant_failure_mode"] == "excessive_smoothing"
    assert payload["confidence"] == 0.82
    assert payload["recommended_focus"] == (
        "restore_high_frequency_structure_without_destabilizing_rollout"
    )
    assert "high_frequency_error=0.141" in payload["evidence"]
    assert "physics_issue=overdissipation" in payload["evidence"]
    assert payload["rejected_explanations"] == [
        "blowup_like_behavior",
        "nonphysical_oscillation",
    ]


def test_select_hypothesis_writes_bounded_next_experiment(tmp_path: Path) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)
    analyze_spectrum(run_id="diagnostic_smoke_001", run_root=str(run_root))
    check_physics(run_id="diagnostic_smoke_001", run_root=str(run_root))
    interpret_error(run_id="diagnostic_smoke_001", run_root=str(run_root))

    result = select_hypothesis(run_id="diagnostic_smoke_001", run_root=str(run_root))

    hypothesis_path = run_root / "diagnostics" / "selected_hypothesis.json"
    payload = json.loads(hypothesis_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "HYPOTHESIS_SELECTED"
    assert result["artifacts"]["selected_hypothesis_path"] == str(hypothesis_path)
    assert payload["selected_hypothesis"] == "add_spectral_consistency_loss"
    assert payload["expected_effect"] == (
        "reduce high-frequency damping while preserving rollout stability"
    )
    assert payload["risk"] == "may amplify oscillations near shocks if overweighted"
    assert payload["next_experiment_scope"] == "read_only_proposal_no_training"


def test_collect_artifacts_writes_hypothesis_selected_manifest_when_diagnostics_exist(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    (run_root / "config").mkdir(parents=True)
    (run_root / "config" / "resolved.json").write_text("{}", encoding="utf-8")
    _prepare_rollout(run_root)
    record_timing(run_id="diagnostic_smoke_001", run_root=str(run_root))
    analyze_spectrum(run_id="diagnostic_smoke_001", run_root=str(run_root))
    check_physics(run_id="diagnostic_smoke_001", run_root=str(run_root))
    interpret_error(run_id="diagnostic_smoke_001", run_root=str(run_root))
    select_hypothesis(run_id="diagnostic_smoke_001", run_root=str(run_root))

    result = collect_artifacts(
        run_id="diagnostic_smoke_001",
        run_root=str(run_root),
        config_hash="diagnostic-smoke-001",
    )

    manifest_path = run_root / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "HYPOTHESIS_SELECTED"
    assert manifest["phase"] == "hypothesis_selected"
    assert manifest["summary"]["dominant_spectral_issue"] == "excessive_smoothing"
    assert manifest["summary"]["physics_issue"] == "overdissipation"
    assert manifest["summary"]["dominant_failure_mode"] == "excessive_smoothing"
    assert manifest["summary"]["selected_hypothesis"] == "add_spectral_consistency_loss"
    assert manifest["completeness"]["has_rollout"] is True
    assert manifest["completeness"]["has_spectrum"] is True
    assert manifest["completeness"]["has_physics_check"] is True
    assert manifest["completeness"]["has_error_interpretation"] is True
    assert manifest["completeness"]["has_selected_hypothesis"] is True
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
uv run pytest tests/test_phase3_diagnostic_skills.py -v
```

Expected:

```text
ERROR tests/test_phase3_diagnostic_skills.py
ModuleNotFoundError: No module named 'pde_agent.commands.interpret_error'
```

- [ ] **Step 3: Commit the red test baseline**

```bash
git add tests/test_phase3_diagnostic_skills.py
git commit -m "test: add phase 3 diagnostic skill expectations"
```

---

### Task 2: Implement diagnostic synthesis helpers and command handlers

**Files:**
- Create: `src/pde_agent/core/diagnostic_synthesis.py`
- Modify: `src/pde_agent/commands/analyze_spectrum.py`
- Modify: `src/pde_agent/commands/check_physics.py`
- Create: `src/pde_agent/commands/interpret_error.py`
- Create: `src/pde_agent/commands/select_hypothesis.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_phase3_diagnostic_skills.py`

- [ ] **Step 1: Add the diagnostic synthesis helper**

Create `src/pde_agent/core/diagnostic_synthesis.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


DIAGNOSTIC_MODE = "synthetic_pde_diagnostic"


def write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def spectrum_payload(run_id: str, rollout: dict[str, object]) -> dict[str, object]:
    _ = rollout
    return {
        "run_id": run_id,
        "mode": DIAGNOSTIC_MODE,
        "low_frequency_error": 0.024,
        "mid_frequency_error": 0.052,
        "high_frequency_error": 0.141,
        "spectral_drift_score": 0.37,
        "shock_region_proxy": 0.68,
        "dominant_spectral_issue": "excessive_smoothing",
    }


def physics_payload(run_id: str, rollout: dict[str, object]) -> dict[str, object]:
    _ = rollout
    return {
        "run_id": run_id,
        "mode": DIAGNOSTIC_MODE,
        "mass_drift": 0.006,
        "energy_decay_ratio": 0.82,
        "total_variation_ratio": 0.71,
        "nonphysical_oscillation": False,
        "blowup_like_behavior": False,
        "excessive_smoothing": True,
        "physics_issue": "overdissipation",
    }


def interpretation_payload(
    *,
    run_id: str,
    rollout: dict[str, object],
    spectrum: dict[str, object],
    physics: dict[str, object],
) -> dict[str, object]:
    _ = rollout
    return {
        "run_id": run_id,
        "mode": DIAGNOSTIC_MODE,
        "dominant_failure_mode": "excessive_smoothing",
        "evidence": [
            f"high_frequency_error={spectrum['high_frequency_error']}",
            f"dominant_spectral_issue={spectrum['dominant_spectral_issue']}",
            f"physics_issue={physics['physics_issue']}",
            f"total_variation_ratio={physics['total_variation_ratio']}",
        ],
        "confidence": 0.82,
        "rejected_explanations": [
            "blowup_like_behavior",
            "nonphysical_oscillation",
        ],
        "recommended_focus": (
            "restore_high_frequency_structure_without_destabilizing_rollout"
        ),
    }


def hypothesis_payload(run_id: str, interpretation: dict[str, object]) -> dict[str, object]:
    _ = interpretation
    return {
        "run_id": run_id,
        "mode": DIAGNOSTIC_MODE,
        "selected_hypothesis": "add_spectral_consistency_loss",
        "rationale": (
            "The synthetic diagnostic indicates excessive smoothing: high-frequency "
            "error is dominant while physics checks show stable overdissipation."
        ),
        "expected_effect": (
            "reduce high-frequency damping while preserving rollout stability"
        ),
        "risk": "may amplify oscillations near shocks if overweighted",
        "next_experiment_scope": "read_only_proposal_no_training",
    }
```

- [ ] **Step 2: Replace `analyze_spectrum` with a file-writing implementation**

Replace `src/pde_agent/commands/analyze_spectrum.py`:

```python
from pathlib import Path

from pde_agent.core.diagnostic_synthesis import read_json, spectrum_payload, write_json
from pde_agent.core.json_output import success_payload


def analyze_spectrum(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    rollout = read_json(root / "metrics" / "rollout.json")
    payload = spectrum_payload(run_id=run_id, rollout=rollout)
    spectrum_path = root / "metrics" / "spectrum.json"
    write_json(spectrum_path, payload)
    return success_payload(
        run_id=run_id,
        command="analyze-spectrum",
        state_hint="SPECTRUM_ANALYZED",
        artifacts={"spectrum_metrics_path": str(spectrum_path)},
        metrics={
            "low_frequency_error": payload["low_frequency_error"],
            "high_frequency_error": payload["high_frequency_error"],
            "spectral_drift_score": payload["spectral_drift_score"],
            "dominant_spectral_issue": payload["dominant_spectral_issue"],
        },
    )
```

- [ ] **Step 3: Replace `check_physics` with a file-writing implementation**

Replace `src/pde_agent/commands/check_physics.py`:

```python
from pathlib import Path

from pde_agent.core.diagnostic_synthesis import physics_payload, read_json, write_json
from pde_agent.core.json_output import success_payload


def check_physics(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    rollout = read_json(root / "metrics" / "rollout.json")
    payload = physics_payload(run_id=run_id, rollout=rollout)
    physics_path = root / "metrics" / "physics_check.json"
    write_json(physics_path, payload)
    return success_payload(
        run_id=run_id,
        command="check-physics",
        state_hint="PHYSICS_CHECKED",
        artifacts={"physics_check_path": str(physics_path)},
        metrics={
            "mass_drift": payload["mass_drift"],
            "energy_decay_ratio": payload["energy_decay_ratio"],
            "total_variation_ratio": payload["total_variation_ratio"],
            "physics_issue": payload["physics_issue"],
        },
    )
```

- [ ] **Step 4: Add `interpret_error` command handler**

Create `src/pde_agent/commands/interpret_error.py`:

```python
from pathlib import Path

from pde_agent.core.diagnostic_synthesis import (
    interpretation_payload,
    read_json,
    write_json,
)
from pde_agent.core.json_output import success_payload


def interpret_error(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    rollout = read_json(root / "metrics" / "rollout.json")
    spectrum = read_json(root / "metrics" / "spectrum.json")
    physics = read_json(root / "metrics" / "physics_check.json")
    payload = interpretation_payload(
        run_id=run_id,
        rollout=rollout,
        spectrum=spectrum,
        physics=physics,
    )
    interpretation_path = root / "diagnostics" / "error_interpretation.json"
    write_json(interpretation_path, payload)
    return success_payload(
        run_id=run_id,
        command="interpret-error",
        state_hint="ERROR_INTERPRETED",
        artifacts={"error_interpretation_path": str(interpretation_path)},
        metrics={
            "dominant_failure_mode": payload["dominant_failure_mode"],
            "confidence": payload["confidence"],
        },
    )
```

- [ ] **Step 5: Add `select_hypothesis` command handler**

Create `src/pde_agent/commands/select_hypothesis.py`:

```python
from pathlib import Path

from pde_agent.core.diagnostic_synthesis import hypothesis_payload, read_json, write_json
from pde_agent.core.json_output import success_payload


def select_hypothesis(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    interpretation = read_json(root / "diagnostics" / "error_interpretation.json")
    payload = hypothesis_payload(run_id=run_id, interpretation=interpretation)
    hypothesis_path = root / "diagnostics" / "selected_hypothesis.json"
    write_json(hypothesis_path, payload)
    return success_payload(
        run_id=run_id,
        command="select-hypothesis",
        state_hint="HYPOTHESIS_SELECTED",
        artifacts={"selected_hypothesis_path": str(hypothesis_path)},
        metrics={
            "selected_hypothesis": payload["selected_hypothesis"],
            "next_experiment_scope": payload["next_experiment_scope"],
        },
    )
```

- [ ] **Step 6: Add CLI commands and remove duplicate `resolve-config`**

Modify `src/pde_agent/cli.py`:

```python
import json

import typer

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.build_submission import build_submission
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.init_run import init_run
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.resolve_config import resolve_config
from pde_agent.commands.select_hypothesis import select_hypothesis
from pde_agent.commands.train_baseline import train_baseline
```

Add these command functions near the existing diagnostic commands:

```python
@app.command("interpret-error")
def interpret_error_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(interpret_error(run_id=run_id, run_root=run_root)))


@app.command("select-hypothesis")
def select_hypothesis_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(select_hypothesis(run_id=run_id, run_root=run_root)))
```

Delete the second duplicate `@app.command("resolve-config")` block at the bottom of the file.

- [ ] **Step 7: Run targeted tests**

Run:

```bash
uv run pytest tests/test_phase3_diagnostic_skills.py -v
```

Expected:

```text
5 passed
```

If the collect-artifacts test fails because manifest support is not implemented yet, continue to Task 3 before committing the green state.

---

### Task 3: Upgrade `collect-artifacts` for diagnostic manifests

**Files:**
- Modify: `src/pde_agent/commands/collect_artifacts.py`
- Test: `tests/test_phase3_diagnostic_skills.py`
- Regression: `tests/test_phase2_synthetic_harness.py`

- [ ] **Step 1: Replace `collect_artifacts` with dual-mode manifest logic**

Replace `src/pde_agent/commands/collect_artifacts.py`:

```python
import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.manifest_builder import build_manifest


def collect_artifacts(
    run_id: str, run_root: str, config_hash: str = "unknown"
) -> dict[str, object]:
    root = Path(run_root)
    rollout_path = root / "metrics" / "rollout.json"
    timing_path = root / "metrics" / "timing.json"
    rollout_payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    diagnostic = _load_diagnostic_payloads(root)

    artifacts = _base_artifacts(root, rollout_path, timing_path, rollout_payload, timing_payload)
    artifacts.update(diagnostic["artifacts"])

    summary = {
        "one_step_nrmse": rollout_payload["one_step"]["nrmse"],
        "long_horizon_nrmse": rollout_payload["long_horizon"]["nrmse"],
        "train_time_seconds": timing_payload["train_time_seconds"],
        "inference_time_seconds": timing_payload["inference_time_seconds"],
    }
    summary.update(diagnostic["summary"])

    completeness = {
        "has_rollout": True,
        "has_spectrum": diagnostic["has_spectrum"],
        "has_physics_check": diagnostic["has_physics_check"],
        "has_error_interpretation": diagnostic["has_error_interpretation"],
        "has_selected_hypothesis": diagnostic["has_selected_hypothesis"],
        "has_timing": True,
    }
    phase = (
        "hypothesis_selected"
        if diagnostic["has_selected_hypothesis"]
        else "results_collected"
    )
    state_hint = (
        "HYPOTHESIS_SELECTED"
        if diagnostic["has_selected_hypothesis"]
        else "RESULTS_COLLECTED"
    )

    manifest = build_manifest(
        run_id=run_id,
        config={
            "resolved_config_path": str(root / "config" / "resolved.json"),
            "config_hash": config_hash,
        },
        artifacts=artifacts,
        summary=summary,
        completeness=completeness,
        phase=phase,
    )
    manifest_path = root / "artifacts" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return success_payload(
        run_id=run_id,
        command="collect-artifacts",
        state_hint=state_hint,
        artifacts={"artifact_manifest_path": str(manifest_path), **artifacts},
        metrics={"artifact_count": len(artifacts)},
    )


def _base_artifacts(
    root: Path,
    rollout_path: Path,
    timing_path: Path,
    rollout_payload: dict[str, object],
    timing_payload: dict[str, object],
) -> dict[str, object]:
    prediction_hdf5_path = rollout_payload["prediction"]["path"]  # type: ignore[index]
    error_curve_path = rollout_payload["curve"]["path"]  # type: ignore[index]
    time_csv_path = timing_payload["time_csv_path"]
    return {
        "rollout_metrics_path": str(rollout_path),
        "timing_metrics_path": str(timing_path),
        "prediction_hdf5_path": prediction_hdf5_path,
        "error_curve_path": error_curve_path,
        "time_csv_path": time_csv_path,
    }


def _load_diagnostic_payloads(root: Path) -> dict[str, object]:
    spectrum_path = root / "metrics" / "spectrum.json"
    physics_path = root / "metrics" / "physics_check.json"
    interpretation_path = root / "diagnostics" / "error_interpretation.json"
    hypothesis_path = root / "diagnostics" / "selected_hypothesis.json"

    artifacts: dict[str, object] = {}
    summary: dict[str, object] = {}

    if spectrum_path.exists():
        spectrum = json.loads(spectrum_path.read_text(encoding="utf-8"))
        artifacts["spectrum_metrics_path"] = str(spectrum_path)
        summary["dominant_spectral_issue"] = spectrum["dominant_spectral_issue"]
    if physics_path.exists():
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        artifacts["physics_check_path"] = str(physics_path)
        summary["physics_issue"] = physics["physics_issue"]
    if interpretation_path.exists():
        interpretation = json.loads(interpretation_path.read_text(encoding="utf-8"))
        artifacts["error_interpretation_path"] = str(interpretation_path)
        summary["dominant_failure_mode"] = interpretation["dominant_failure_mode"]
    if hypothesis_path.exists():
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        artifacts["selected_hypothesis_path"] = str(hypothesis_path)
        summary["selected_hypothesis"] = hypothesis["selected_hypothesis"]

    return {
        "artifacts": artifacts,
        "summary": summary,
        "has_spectrum": spectrum_path.exists(),
        "has_physics_check": physics_path.exists(),
        "has_error_interpretation": interpretation_path.exists(),
        "has_selected_hypothesis": hypothesis_path.exists(),
    }
```

- [ ] **Step 2: Run Phase 2 and Phase 3 tests**

Run:

```bash
uv run pytest tests/test_phase2_synthetic_harness.py tests/test_phase3_diagnostic_skills.py -v
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 3: Commit diagnostic command implementation**

```bash
git add src/pde_agent/core/diagnostic_synthesis.py \
  src/pde_agent/commands/analyze_spectrum.py \
  src/pde_agent/commands/check_physics.py \
  src/pde_agent/commands/interpret_error.py \
  src/pde_agent/commands/select_hypothesis.py \
  src/pde_agent/commands/collect_artifacts.py \
  src/pde_agent/cli.py \
  tests/test_phase3_diagnostic_skills.py
git commit -m "feat: add pde diagnostic skill commands"
```

---

### Task 4: Add deterministic diagnostic audit

**Files:**
- Create: `src/pde_agent/core/hermes_diagnostic_audit.py`
- Create: `scripts/audit_hermes_diagnostic.py`
- Create: `tests/test_hermes_diagnostic_audit.py`

- [ ] **Step 1: Write audit tests**

Create `tests/test_hermes_diagnostic_audit.py`:

```python
import json
from pathlib import Path

from pde_agent.core.hermes_diagnostic_audit import audit_hermes_diagnostic_session


REQUIRED_READS = [
    "/repo/integration_tests/hermes_diagnostic_smoke/run_state.yaml",
    "/repo/integration_tests/hermes_diagnostic_smoke/bootstrap_prompt.md",
    "/repo/integration_tests/hermes_diagnostic_smoke/task_spec.json",
    "/repo/integration_tests/hermes_diagnostic_smoke/base_config.json",
    "/repo/integration_tests/hermes_diagnostic_smoke/override.json",
]

ALLOWED_COMMANDS = [
    "/repo/.venv/bin/pde-agent init-run --run-id diagnostic_smoke_001 --runs-root /repo/integration_tests/hermes_diagnostic_smoke/runs",
    "/repo/.venv/bin/pde-agent resolve-config --run-id diagnostic_smoke_001 --base-config-file /repo/integration_tests/hermes_diagnostic_smoke/base_config.json --override-file /repo/integration_tests/hermes_diagnostic_smoke/override.json --task-spec-file /repo/integration_tests/hermes_diagnostic_smoke/task_spec.json --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent eval-rollout --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent analyze-spectrum --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent check-physics --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent interpret-error --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent select-hypothesis --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001",
    "/repo/.venv/bin/pde-agent collect-artifacts --run-id diagnostic_smoke_001 --run-root /repo/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001 --config-hash diagnostic-smoke-001",
]


def _tool_call(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _terminal_result(command: str, state_hint: str) -> dict:
    payload = {
        "run_id": "diagnostic_smoke_001",
        "command": command.split()[1],
        "status": "ok",
        "state_hint": state_hint,
        "artifacts": {},
        "metrics": {},
        "errors": [],
        "next_hint": None,
    }
    return {
        "role": "tool",
        "content": json.dumps(
            {"output": json.dumps(payload), "exit_code": 0, "error": None}
        ),
    }


def _session(commands: list[str] | None = None, final_hint: str = "HYPOTHESIS_SELECTED") -> dict:
    command_sequence = commands or ALLOWED_COMMANDS
    messages = [{"role": "user", "content": "diagnostic smoke", "tool_calls": None}]
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("read_file", {"path": path}) for path in REQUIRED_READS
            ],
        }
    )
    hints = [
        "RUN_INITIALIZED",
        "CONFIG_RESOLVED",
        "RESULTS_COLLECTED",
        "SPECTRUM_ANALYZED",
        "PHYSICS_CHECKED",
        "ERROR_INTERPRETED",
        "HYPOTHESIS_SELECTED",
        final_hint,
    ]
    for command, hint in zip(command_sequence, hints, strict=False):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("terminal", {"command": command})],
            }
        )
        messages.append(_terminal_result(command, hint))
    return {"messages": messages}


def _write_manifest(path: Path, *, phase: str = "hypothesis_selected") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "run_id": "diagnostic_smoke_001",
                "phase": phase,
                "config": {},
                "artifacts": {
                    "rollout_metrics_path": "/repo/run/metrics/rollout.json",
                    "spectrum_metrics_path": "/repo/run/metrics/spectrum.json",
                    "physics_check_path": "/repo/run/metrics/physics_check.json",
                    "error_interpretation_path": "/repo/run/diagnostics/error_interpretation.json",
                    "selected_hypothesis_path": "/repo/run/diagnostics/selected_hypothesis.json",
                },
                "summary": {
                    "dominant_spectral_issue": "excessive_smoothing",
                    "physics_issue": "overdissipation",
                    "dominant_failure_mode": "excessive_smoothing",
                    "selected_hypothesis": "add_spectral_consistency_loss",
                },
                "completeness": {},
                "errors": [],
            }
        ),
        encoding="utf-8",
    )


def test_diagnostic_audit_passes_for_expected_session(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_130000_diag01.json"
    manifest_path = tmp_path / "runs" / "diagnostic_smoke_001" / "artifacts" / "manifest.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "passed"
    assert report["diagnosis"] == {
        "dominant_failure_mode": "excessive_smoothing",
        "selected_hypothesis": "add_spectral_consistency_loss",
    }
    assert report["errors"] == []


def test_diagnostic_audit_fails_on_missing_diagnostic_command(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_130000_diag01.json"
    manifest_path = tmp_path / "runs" / "diagnostic_smoke_001" / "artifacts" / "manifest.json"
    commands = [command for command in ALLOWED_COMMANDS if "analyze-spectrum" not in command]
    session_path.write_text(json.dumps(_session(commands)), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "allowed_command_sequence" in report["errors"]


def test_diagnostic_audit_fails_on_forbidden_training_command(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_130000_diag01.json"
    manifest_path = tmp_path / "runs" / "diagnostic_smoke_001" / "artifacts" / "manifest.json"
    commands = [
        *ALLOWED_COMMANDS,
        "/repo/.venv/bin/pde-agent train-baseline --run-id diagnostic_smoke_001 --run-root /repo/run",
    ]
    session_path.write_text(json.dumps(_session(commands)), encoding="utf-8")
    _write_manifest(manifest_path)

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "forbidden_actions" in report["errors"]


def test_diagnostic_audit_fails_on_manifest_phase_mismatch(tmp_path: Path) -> None:
    session_path = tmp_path / "session_20260424_130000_diag01.json"
    manifest_path = tmp_path / "runs" / "diagnostic_smoke_001" / "artifacts" / "manifest.json"
    session_path.write_text(json.dumps(_session()), encoding="utf-8")
    _write_manifest(manifest_path, phase="results_collected")

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=REQUIRED_READS,
        allowed_commands=ALLOWED_COMMANDS,
    )

    assert report["status"] == "failed"
    assert "manifest" in report["errors"]
```

- [ ] **Step 2: Run audit tests to verify they fail**

Run:

```bash
uv run pytest tests/test_hermes_diagnostic_audit.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'pde_agent.core.hermes_diagnostic_audit'
```

- [ ] **Step 3: Implement diagnostic audit core**

Create `src/pde_agent/core/hermes_diagnostic_audit.py` with this top-level function:

```python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def audit_hermes_diagnostic_session(
    *,
    session_path: Path,
    manifest_path: Path,
    required_reads: list[str],
    allowed_commands: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    if not session_path.exists():
        return _error_report(
            session_path, manifest_path, f"session file not found: {session_path}"
        )

    session = json.loads(session_path.read_text(encoding="utf-8"))
    messages = session.get("messages", [])
    if not any(message.get("role") == "user" for message in messages):
        errors.append("session has no user message")

    tool_calls = _extract_tool_calls(messages)
    read_paths = [
        call["arguments"].get("path")
        for call in tool_calls
        if call["name"] == "read_file"
    ]
    terminal_commands = [
        call["arguments"].get("command")
        for call in tool_calls
        if call["name"] == "terminal"
    ]
    tool_summary = dict(Counter(call["name"] for call in tool_calls))

    _add_check(checks, errors, "required_reads", read_paths[: len(required_reads)] == required_reads, {"expected": required_reads, "actual_prefix": read_paths[: len(required_reads)]})
    _add_check(checks, errors, "allowed_command_sequence", terminal_commands == allowed_commands, {"expected": allowed_commands, "actual": terminal_commands})
    _add_check(checks, errors, "forbidden_actions", not _forbidden_calls(tool_calls), {"offenders": _forbidden_calls(tool_calls)})
    _add_check(checks, errors, "terminal_exit_codes", _terminal_exit_codes_ok(messages), {"terminal_result_count": len(_terminal_results(messages))})

    final_payload = _final_terminal_payload(messages)
    _add_check(
        checks,
        errors,
        "final_state",
        final_payload.get("status") == "ok"
        and final_payload.get("state_hint") == "HYPOTHESIS_SELECTED",
        {"final_payload": final_payload},
    )
    manifest_ok, diagnosis = _manifest_ok(manifest_path)
    _add_check(checks, errors, "manifest", manifest_ok, {"manifest_path": str(manifest_path)})

    return {
        "status": "passed" if not errors else "failed",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "manifest_path": str(manifest_path),
        "checks": checks,
        "tool_call_summary": tool_summary,
        "terminal_commands": terminal_commands,
        "diagnosis": diagnosis,
        "errors": errors,
    }
```

Add these helper functions in the same file:

```python
def _manifest_ok(manifest_path: Path) -> tuple[bool, dict[str, object]]:
    if not manifest_path.exists():
        return False, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    summary = manifest.get("summary", {})
    required_artifacts = [
        "rollout_metrics_path",
        "spectrum_metrics_path",
        "physics_check_path",
        "error_interpretation_path",
        "selected_hypothesis_path",
    ]
    diagnosis = {
        "dominant_failure_mode": summary.get("dominant_failure_mode"),
        "selected_hypothesis": summary.get("selected_hypothesis"),
    }
    ok = (
        manifest.get("status") == "ok"
        and manifest.get("phase") == "hypothesis_selected"
        and manifest.get("errors") == []
        and all(artifacts.get(key) for key in required_artifacts)
        and bool(summary.get("dominant_spectral_issue"))
        and bool(summary.get("physics_issue"))
        and bool(summary.get("dominant_failure_mode"))
        and bool(summary.get("selected_hypothesis"))
    )
    return ok, diagnosis


def _forbidden_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    forbidden_terms = [
        " train-baseline",
        " build-submission",
        " --help",
    ]
    for call in tool_calls:
        name = call["name"]
        command = call["arguments"].get("command", "")
        if name not in {"read_file", "terminal"}:
            offenders.append(call)
        elif any(term in command for term in forbidden_terms):
            offenders.append(call)
    return offenders
```

Add these shared parsing helpers below `_forbidden_calls`:

```python
def _extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = json.loads(function.get("arguments") or "{}")
            calls.append({"name": function.get("name", ""), "arguments": arguments})
    return calls


def _terminal_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content") or "{}")
        except json.JSONDecodeError:
            result = {"parse_error": message.get("content")}
        if "exit_code" in result or "output" in result or "parse_error" in result:
            results.append(result)
    return results


def _terminal_exit_codes_ok(messages: list[dict[str, Any]]) -> bool:
    results = _terminal_results(messages)
    return bool(results) and all(result.get("exit_code") == 0 for result in results)


def _final_terminal_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    results = _terminal_results(messages)
    if not results:
        return {}
    try:
        return json.loads(results[-1].get("output") or "{}")
    except json.JSONDecodeError:
        return {"parse_error": results[-1].get("output")}


def _add_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    passed: bool,
    details: dict[str, Any],
) -> None:
    checks.append(
        {"name": name, "status": "passed" if passed else "failed", "details": details}
    )
    if not passed:
        errors.append(name)


def _error_report(session_path: Path, manifest_path: Path, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "session_id": _session_id_from_path(session_path),
        "session_path": str(session_path),
        "manifest_path": str(manifest_path),
        "checks": [],
        "tool_call_summary": {},
        "terminal_commands": [],
        "diagnosis": {},
        "errors": [message],
    }


def _session_id_from_path(path: Path) -> str:
    name = path.stem
    return name.removeprefix("session_")
```

- [ ] **Step 4: Add diagnostic audit CLI**

Create `scripts/audit_hermes_diagnostic.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pde_agent.core.hermes_diagnostic_audit import audit_hermes_diagnostic_session


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a saved Hermes PDE diagnostic smoke-test session."
    )
    parser.add_argument("--fixture-root", default="integration_tests/hermes_diagnostic_smoke")
    parser.add_argument("--session-id")
    parser.add_argument("--session-path")
    parser.add_argument("--report-path")
    args = parser.parse_args()

    repo_root = Path.cwd()
    fixture_root = (repo_root / args.fixture_root).resolve()
    session_path = _resolve_session_path(fixture_root, args.session_id, args.session_path)
    run_state = (fixture_root / "run_state.yaml").read_text(encoding="utf-8")
    required_reads = _yaml_list(run_state, "required_reads")
    allowed_commands = _yaml_command_list(run_state)
    manifest_path = Path(_yaml_list(run_state, "required_artifacts")[0])

    report = audit_hermes_diagnostic_session(
        session_path=session_path,
        manifest_path=manifest_path,
        required_reads=required_reads,
        allowed_commands=allowed_commands,
    )
    report_path = (
        Path(args.report_path)
        if args.report_path
        else fixture_root / "reports" / f"audit_{report['session_id']}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report_path": str(report_path)}, indent=2))
    return 0 if report["status"] == "passed" else 1
```

Add these helper functions to `scripts/audit_hermes_diagnostic.py`:

```python
def _resolve_session_path(
    fixture_root: Path, session_id: str | None, session_path: str | None
) -> Path:
    if session_path:
        return Path(session_path).resolve()
    sessions_dir = fixture_root / "hermes_home" / "sessions"
    if session_id:
        return sessions_dir / f"session_{session_id}.json"
    sessions = sorted(sessions_dir.glob("session_*.json"), key=lambda path: path.stat().st_mtime)
    if not sessions:
        return sessions_dir / "session_missing.json"
    return sessions[-1]


def _yaml_list(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    values: list[str] = []
    inside = False
    for line in lines:
        if line.strip() == f"{key}:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            break
        if inside and line.strip().startswith("- "):
            values.append(line.strip()[2:])
    return values


def _yaml_command_list(text: str) -> list[str]:
    commands: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))
    return commands


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run audit tests**

Run:

```bash
uv run pytest tests/test_hermes_diagnostic_audit.py -v
```

Expected:

```text
4 passed
```

- [ ] **Step 6: Commit audit implementation**

```bash
git add src/pde_agent/core/hermes_diagnostic_audit.py \
  scripts/audit_hermes_diagnostic.py \
  tests/test_hermes_diagnostic_audit.py
git commit -m "feat: add hermes diagnostic audit"
```

---

### Task 5: Add Hermes diagnostic fixture and real runner

**Files:**
- Create directory: `integration_tests/hermes_diagnostic_smoke`
- Create: `integration_tests/hermes_diagnostic_smoke/run_state.yaml`
- Create: `integration_tests/hermes_diagnostic_smoke/bootstrap_prompt.md`
- Create: `integration_tests/hermes_diagnostic_smoke/task_spec.json`
- Create: `integration_tests/hermes_diagnostic_smoke/base_config.json`
- Create: `integration_tests/hermes_diagnostic_smoke/override.json`
- Create: `integration_tests/hermes_diagnostic_smoke/prompt_template.md`
- Create: `integration_tests/hermes_diagnostic_smoke/README.md`
- Create: `integration_tests/hermes_diagnostic_smoke/hermes_home/.gitkeep`
- Create: `integration_tests/hermes_diagnostic_smoke/runs/.gitkeep`
- Create: `integration_tests/hermes_diagnostic_smoke/reports/.gitkeep`
- Modify: `.gitignore`
- Create: `scripts/run_hermes_diagnostic.py`

- [ ] **Step 1: Add fixture directories**

Run:

```bash
mkdir -p integration_tests/hermes_diagnostic_smoke/hermes_home \
  integration_tests/hermes_diagnostic_smoke/runs \
  integration_tests/hermes_diagnostic_smoke/reports
touch integration_tests/hermes_diagnostic_smoke/hermes_home/.gitkeep \
  integration_tests/hermes_diagnostic_smoke/runs/.gitkeep \
  integration_tests/hermes_diagnostic_smoke/reports/.gitkeep
```

- [ ] **Step 2: Add ignore rules for fixture runtime outputs**

Append to `.gitignore`:

```gitignore
integration_tests/hermes_diagnostic_smoke/hermes_home/*
!integration_tests/hermes_diagnostic_smoke/hermes_home/.gitkeep

integration_tests/hermes_diagnostic_smoke/runs/*
!integration_tests/hermes_diagnostic_smoke/runs/.gitkeep

integration_tests/hermes_diagnostic_smoke/reports/*
!integration_tests/hermes_diagnostic_smoke/reports/.gitkeep
```

- [ ] **Step 3: Add diagnostic fixture inputs**

Create `integration_tests/hermes_diagnostic_smoke/task_spec.json`:

```json
{
  "problem_id": "PDE_Burgers_Diagnostic_Synthetic",
  "equation": "1D Burgers",
  "mode": "synthetic_diagnostic_fixture",
  "input_window": 10,
  "prediction_window": 190,
  "spatial_resolution": 256,
  "goal": "diagnose synthetic rollout failure and select a bounded next hypothesis"
}
```

Create `integration_tests/hermes_diagnostic_smoke/base_config.json`:

```json
{
  "run_id": "diagnostic_smoke_001",
  "model_family": "synthetic_neural_operator",
  "diagnostic_mode": true,
  "training_allowed": false,
  "commands_expected": [
    "init-run",
    "resolve-config",
    "eval-rollout",
    "analyze-spectrum",
    "check-physics",
    "interpret-error",
    "select-hypothesis",
    "record-timing",
    "collect-artifacts"
  ]
}
```

Create `integration_tests/hermes_diagnostic_smoke/override.json`:

```json
{
  "config_hash": "diagnostic-smoke-001",
  "agent_constraints": {
    "patch_code": false,
    "run_training": false,
    "inspect_help": false
  }
}
```

- [ ] **Step 4: Add run state**

Create `integration_tests/hermes_diagnostic_smoke/run_state.yaml` using absolute paths for this machine:

```yaml
run_id: diagnostic_smoke_001
run_root: /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001
working_directory: /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke
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
  - call_analyze_spectrum
  - call_check_physics
  - call_interpret_error
  - call_select_hypothesis
  - call_record_timing
  - call_collect_artifacts

forbidden_actions:
  - patch_code
  - run_training
  - call_build_submission
  - guess_missing_paths
  - skip_required_reads
  - search_outside_current_directory
  - inspect_command_help
  - probe_command_surface

required_reads:
  - /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/run_state.yaml
  - /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/bootstrap_prompt.md
  - /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/task_spec.json
  - /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/base_config.json
  - /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/override.json

required_artifacts:
  - /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001/artifacts/manifest.json

allowed_commands:
  - name: init_run
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent init-run --run-id diagnostic_smoke_001 --runs-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs
  - name: resolve_config
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent resolve-config --run-id diagnostic_smoke_001 --base-config-file /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/base_config.json --override-file /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/override.json --task-spec-file /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/task_spec.json --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001
  - name: eval_rollout
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent eval-rollout --run-id diagnostic_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001
  - name: analyze_spectrum
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent analyze-spectrum --run-id diagnostic_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001
  - name: check_physics
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent check-physics --run-id diagnostic_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001
  - name: interpret_error
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent interpret-error --run-id diagnostic_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001
  - name: select_hypothesis
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent select-hypothesis --run-id diagnostic_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001
  - name: record_timing
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent record-timing --run-id diagnostic_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001
  - name: collect_artifacts
    command: /Users/daiduo2/pde-agent/.venv/bin/pde-agent collect-artifacts --run-id diagnostic_smoke_001 --run-root /Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/runs/diagnostic_smoke_001 --config-hash diagnostic-smoke-001

success_transition:
  on_complete: HYPOTHESIS_SELECTED
```

- [ ] **Step 5: Add diagnostic bootstrap and prompt template**

Create `integration_tests/hermes_diagnostic_smoke/bootstrap_prompt.md`:

```markdown
# Hermes PDE Diagnostic Smoke Protocol

You are running a protocol-compliance diagnostic smoke test for `pde-agent`.

Rules:
- Read all required files before taking action.
- Use `run_state.yaml` as the source of truth.
- Execute only exact command strings listed under `allowed_commands`.
- Continue past intermediate rollout results until `HYPOTHESIS_SELECTED`.
- After `select-hypothesis`, execute `record-timing`, then `collect-artifacts`.
- Do not manually read generated run artifacts with shell commands such as `cat`.
- Do not patch source code.
- Do not run training.
- Do not inspect help output.
- Do not probe command surfaces.
- Do not search outside the current directory.
```

Create `integration_tests/hermes_diagnostic_smoke/prompt_template.md`:

```markdown
This is a PDE diagnostic protocol-compliance smoke test for pde-agent.

Required files to read before action:
- {run_state_path}
- {bootstrap_prompt_path}
- {task_spec_path}
- {base_config_path}
- {override_path}

Hard rules:
- Treat run_state.yaml as the source of truth.
- Execute only exact command strings from allowed_commands.
- Do not infer alternate paths.
- Do not search for files.
- Do not inspect command help.
- Do not patch code.
- Do not run training.
- Continue until the allowed success transition HYPOTHESIS_SELECTED is reached.
- After select-hypothesis, do not inspect generated artifact files manually.
- Execute record-timing and collect-artifacts using their exact allowed command strings.
- Do not use cat, ls, find, grep, sed, head, tail, python, or any other shell command unless it is exactly listed under allowed_commands.

At the end, briefly summarize:
- dominant failure mode
- selected hypothesis
- why no forbidden action was needed
```

- [ ] **Step 6: Add diagnostic README**

Create `integration_tests/hermes_diagnostic_smoke/README.md`:

```markdown
# Hermes Diagnostic Smoke Test

This fixture verifies that Hermes can execute the synthetic PDE diagnostic chain:

1. init-run
2. resolve-config
3. eval-rollout
4. analyze-spectrum
5. check-physics
6. interpret-error
7. select-hypothesis
8. record-timing
9. collect-artifacts

The fixture is deterministic and does not use real PDEBench data, checkpoints, training, or code patching.

Run audit only:

```bash
uv run python scripts/audit_hermes_diagnostic.py
```

Run real Hermes diagnostic smoke:

```bash
uv run python scripts/run_hermes_diagnostic.py --timeout-seconds 300
```
```

- [ ] **Step 7: Add real diagnostic runner**

Create `scripts/run_hermes_diagnostic.py` with the same runner structure as `scripts/run_hermes_smoke.py`. Use these exact diagnostic-specific values in the new file:

```python
parser = argparse.ArgumentParser(
    description="Run real Hermes PDE diagnostic smoke test and audit it."
)
parser.add_argument("--fixture-root", default="integration_tests/hermes_diagnostic_smoke")
parser.add_argument("--profile", default="hermes-protocol-smoke")
```

Change the audit command:

```python
audit_cmd = [
    sys.executable,
    "scripts/audit_hermes_diagnostic.py",
    "--fixture-root",
    _fixture_arg(repo_root, fixture_root),
]
```

Change success detection:

```python
def _success_transition_seen(output: str) -> bool:
    normalized = output.replace("*", "").replace("`", "")
    if "HYPOTHESIS_SELECTED" not in normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "Success transition",
            "completed successfully",
            "No forbidden actions",
            "Final state",
            "Final state_hint",
        )
    )
```

- [ ] **Step 8: Run deterministic verification**

Run:

```bash
uv run pytest tests -v
uv run python scripts/audit_hermes_diagnostic.py --session-path tests/does-not-exist.json
```

Expected:

```text
pytest: all tests pass
audit command: exits non-zero with a JSON status of error
```

- [ ] **Step 9: Commit fixture and runner**

```bash
git add .gitignore \
  integration_tests/hermes_diagnostic_smoke \
  scripts/run_hermes_diagnostic.py
git commit -m "feat: add hermes diagnostic smoke fixture"
```

---

### Task 6: Run one real Hermes diagnostic smoke test

**Files:**
- Runtime outputs under: `integration_tests/hermes_diagnostic_smoke/hermes_home`
- Runtime outputs under: `integration_tests/hermes_diagnostic_smoke/runs`
- Runtime outputs under: `integration_tests/hermes_diagnostic_smoke/reports`

- [ ] **Step 1: Run the real diagnostic smoke runner**

Run:

```bash
uv run python scripts/run_hermes_diagnostic.py --timeout-seconds 300
```

Expected: stdout contains `"status": "passed"` and `report_path` points under `/Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/reports/`.

- [ ] **Step 2: Audit the latest diagnostic session again**

Run audit-only mode on the latest diagnostic session:

```bash
uv run python scripts/run_hermes_diagnostic.py --audit-only
```

Expected: stdout contains `"status": "passed"` and `report_path` points under `/Users/daiduo2/pde-agent/integration_tests/hermes_diagnostic_smoke/reports/`.

- [ ] **Step 3: Run final deterministic tests**

Run:

```bash
uv run pytest tests -v
git status --short
```

Expected:

```text
all collected tests pass, including the new Phase 3 tests.
git status shows no tracked file modifications.
```

- [ ] **Step 4: Record verified session in final response**

Do not commit runtime Hermes sessions or generated reports. They are ignored by `.gitignore`.

Final response should include:

- latest real diagnostic session id,
- audit status,
- test command result,
- latest commit hash.
