# FNO Rollout Diagnosis V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one command that runs official/fixture FNO rollout prediction, scoring, error diagnosis, Burgers physics check, and writes a compact agent-readable summary.

**Architecture:** Create a thin core orchestration module that reuses existing `fno_backend`, `local_scoring`, `rollout_error_diagnosis`, and `burgers_physics_local` functions. Add a command wrapper and CLI registration that expose the artifact bundle without changing model, scoring, or physics formulas.

**Tech Stack:** Python 3.12+, NumPy, optional PyTorch through existing FNO backend, Typer, pytest.

---

## File Structure

- Create `src/pde_agent/core/fno_rollout_diagnosis.py`: orchestration, artifact path management, overwrite protection, summary JSON.
- Create `src/pde_agent/commands/diagnose_fno_rollout.py`: command wrapper returning `success_payload`.
- Modify `src/pde_agent/cli.py`: register `diagnose-fno-rollout`.
- Modify `src/pde_agent/core/state_hints.py`: add `FNO_ROLLOUT_DIAGNOSED`.
- Create `tests/test_fno_rollout_diagnosis.py`: core and CLI tests using synthetic fixture checkpoints.
- Modify `README.md`: document the diagnosis command and artifact bundle.

Do not modify FNO model behavior, training code, scoring formulas, physics formulas, or dependency declarations.

---

### Task 1: Core FNO Rollout Diagnosis Orchestration

**Files:**
- Create: `src/pde_agent/core/fno_rollout_diagnosis.py`
- Create: `tests/test_fno_rollout_diagnosis.py`

- [ ] **Step 1: Add failing core orchestration tests**

Create `tests/test_fno_rollout_diagnosis.py`:

```python
import json
from pathlib import Path

import pytest

from pde_agent.core.fno_rollout_diagnosis import (
    FNORolloutDiagnosisError,
    diagnose_fno_rollout,
)

torch = pytest.importorskip("torch", reason="FNO rollout diagnosis tests require torch")
np = pytest.importorskip("numpy")


def _write_fixture_checkpoint(path: Path) -> None:
    torch.save(
        {
            "model_type": "fno_v0_fixture",
            "prediction_rule": "delta_extrapolation",
            "config": {"modes": 2, "width": 4},
            "model_state_dict": {"weight": torch.ones(1)},
        },
        path,
    )


def _write_windows(path: Path) -> tuple[object, object]:
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    context = np.zeros((2, 4, 8), dtype=np.float32)
    target = np.zeros((2, 3, 8), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.1)
        delta = 0.04 * np.cos(x) + 0.01 * sample
        for time_index in range(4):
            context[sample, time_index, :] = base + time_index * delta
        for time_index in range(3):
            target[sample, time_index, :] = base + (4 + time_index) * delta
    np.savez(path, context=context, target=target)
    return context, target


def test_diagnose_fno_rollout_writes_artifact_bundle(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_dir = tmp_path / "diagnosis"
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    summary = diagnose_fno_rollout(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    expected_files = {
        "fno_prediction.npz",
        "fno_score.json",
        "fno_rollout_error.json",
        "fno_burgers_physics.json",
        "fno_rollout_summary.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert summary["state_hint"] == "FNO_ROLLOUT_DIAGNOSED"
    assert summary["model"]["adapter_type"] == "fixture_delta_extrapolation"
    assert summary["score"]["competition_like_score_short"] == pytest.approx(100.0)
    assert "dominant_error_pattern" in summary["error_diagnosis"]
    assert "physics_issue" in summary["physics"]
    assert summary["recommendation"]["next_command"] == "select-pde-hypothesis"
    written = json.loads((output_dir / "fno_rollout_summary.json").read_text())
    assert written == summary


def test_diagnose_fno_rollout_rejects_existing_artifact_without_force(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_dir = tmp_path / "diagnosis"
    output_dir.mkdir()
    (output_dir / "fno_score.json").write_text("previous", encoding="utf-8")
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    with pytest.raises(FNORolloutDiagnosisError, match="already exists"):
        diagnose_fno_rollout(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            nu=0.001,
            dt=1.0,
            dx=1.0,
            force=False,
        )

    assert (output_dir / "fno_score.json").read_text(encoding="utf-8") == "previous"
```

- [ ] **Step 2: Run core tests and verify RED**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_rollout_diagnosis.py -q
```

Expected before implementation: fail because `pde_agent.core.fno_rollout_diagnosis` does not exist.

- [ ] **Step 3: Implement core orchestration**

Create `src/pde_agent/core/fno_rollout_diagnosis.py`:

```python
"""Orchestrate local FNO rollout diagnostics into one artifact bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.burgers_physics_local import check_burgers_physics_npz
from pde_agent.core.fno_backend import predict_fno_local
from pde_agent.core.local_scoring import score_prediction_npz
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz


class FNORolloutDiagnosisError(RuntimeError):
    pass


def diagnose_fno_rollout(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    windows_file = _require_file(Path(windows_path), "windows")
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    output_root = Path(output_dir)
    paths = _artifact_paths(output_root)
    _prepare_output_paths(paths=paths, force=force)

    prediction_summary = predict_fno_local(
        windows_path=windows_file,
        checkpoint_path=checkpoint_file,
        output_path=paths["prediction_npz_path"],
        metadata_path=None,
        force=force,
    )
    score = score_prediction_npz(
        prediction_path=paths["prediction_npz_path"],
        target_path=windows_file,
        output_path=paths["score_json_path"],
    )
    error = diagnose_rollout_error_npz(
        prediction_path=paths["prediction_npz_path"],
        target_path=windows_file,
        score_path=paths["score_json_path"],
        output_path=paths["rollout_error_json_path"],
    )
    physics = check_burgers_physics_npz(
        prediction_path=paths["prediction_npz_path"],
        target_path=windows_file,
        output_path=paths["burgers_physics_json_path"],
        nu=nu,
        dt=dt,
        dx=dx,
    )
    summary = _build_summary(
        prediction_summary=prediction_summary,
        score=score,
        error=error,
        physics=physics,
        paths=paths,
    )
    summary["summary_path"] = str(paths["summary_json_path"])
    paths["summary_json_path"].write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "prediction_npz_path": output_dir / "fno_prediction.npz",
        "score_json_path": output_dir / "fno_score.json",
        "rollout_error_json_path": output_dir / "fno_rollout_error.json",
        "burgers_physics_json_path": output_dir / "fno_burgers_physics.json",
        "summary_json_path": output_dir / "fno_rollout_summary.json",
    }


def _prepare_output_paths(*, paths: dict[str, Path], force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FNORolloutDiagnosisError(f"diagnosis artifact already exists: {names}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _build_summary(
    *,
    prediction_summary: dict[str, Any],
    score: dict[str, Any],
    error: dict[str, Any],
    physics: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    model = {
        "model_type": prediction_summary["model_type"],
        "adapter_type": prediction_summary.get("adapter_type", "unknown"),
        "checkpoint_path": prediction_summary["checkpoint_path"],
        "checkpoint_format": prediction_summary["checkpoint_format"],
    }
    for key in ("archive_member", "history_steps", "modes", "width", "fc_hidden"):
        if key in prediction_summary:
            model[key] = prediction_summary[key]
    return {
        "schema_version": 1,
        "state_hint": "FNO_ROLLOUT_DIAGNOSED",
        "model": model,
        "artifacts": {
            "prediction_npz_path": str(paths["prediction_npz_path"]),
            "score_json_path": str(paths["score_json_path"]),
            "rollout_error_json_path": str(paths["rollout_error_json_path"]),
            "burgers_physics_json_path": str(paths["burgers_physics_json_path"]),
        },
        "score": {
            "competition_like_score_short": score["competition_like_score_short"],
            "rel_mse": score["rel_mse"],
            "rmse": score["rmse"],
        },
        "error_diagnosis": {
            "dominant_error_pattern": error["dominant_error_pattern"],
            "late_rel_mse": error["late_rel_mse"],
            "time_error_slope": error["time_error_slope"],
            "shock_region_error_ratio": error["shock_region_error_ratio"],
        },
        "physics": {
            "physics_issue": physics["physics_issue"],
            "residual_ratio": physics["residual_ratio"],
            "total_variation_ratio": physics["total_variation_ratio"],
            "gradient_error_ratio": physics["gradient_error_ratio"],
        },
        "recommendation": {
            "next_command": "select-pde-hypothesis",
            "rationale": (
                "Use existing hypothesis selection with this score, error diagnosis, "
                "and physics check."
            ),
        },
    }


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FNORolloutDiagnosisError(f"missing {label} path: {path}")
    if not path.is_file():
        raise FNORolloutDiagnosisError(f"{label} path is not a file: {path}")
    return path
```

- [ ] **Step 4: Run core tests and full tests**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_rollout_diagnosis.py -q
uv run --frozen pytest tests -q
```

Expected result: new core tests pass and full tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/pde_agent/core/fno_rollout_diagnosis.py tests/test_fno_rollout_diagnosis.py
git commit -m "feat: add fno rollout diagnosis core"
```

---

### Task 2: Command Wrapper, CLI, And State Hint

**Files:**
- Create: `src/pde_agent/commands/diagnose_fno_rollout.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Modify: `tests/test_fno_rollout_diagnosis.py`

- [ ] **Step 1: Add failing CLI test**

Append to `tests/test_fno_rollout_diagnosis.py`:

```python
def test_diagnose_fno_rollout_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_dir = tmp_path / "diagnosis"
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "diagnose-fno-rollout",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--nu",
            "0.001",
            "--dt",
            "1.0",
            "--dx",
            "1.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "diagnose-fno-rollout"
    assert payload["state_hint"] == "FNO_ROLLOUT_DIAGNOSED"
    assert payload["artifacts"]["fno_rollout_summary_json_path"] == str(
        output_dir / "fno_rollout_summary.json"
    )
    assert payload["metrics"]["adapter_type"] == "fixture_delta_extrapolation"
    assert payload["metrics"]["competition_like_score_short"] == pytest.approx(100.0)
    assert payload["metrics"]["physics_issue"]
    assert payload["next_hint"] == "select-pde-hypothesis"
```

- [ ] **Step 2: Run CLI test and verify RED**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_rollout_diagnosis.py::test_diagnose_fno_rollout_cli_writes_payload -q
```

Expected before implementation: fail because command is not registered.

- [ ] **Step 3: Add command wrapper**

Create `src/pde_agent/commands/diagnose_fno_rollout.py`:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.fno_rollout_diagnosis import diagnose_fno_rollout as diagnose_core
from pde_agent.core.json_output import success_payload


def diagnose_fno_rollout(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_dir: str,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    summary = diagnose_core(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="diagnose-fno-rollout",
        state_hint="FNO_ROLLOUT_DIAGNOSED",
        artifacts={
            "fno_prediction_npz_path": summary["artifacts"]["prediction_npz_path"],
            "fno_score_json_path": summary["artifacts"]["score_json_path"],
            "fno_rollout_error_json_path": summary["artifacts"]["rollout_error_json_path"],
            "fno_burgers_physics_json_path": summary["artifacts"]["burgers_physics_json_path"],
            "fno_rollout_summary_json_path": summary["summary_path"],
        },
        metrics={
            "adapter_type": summary["model"]["adapter_type"],
            "competition_like_score_short": summary["score"]["competition_like_score_short"],
            "rel_mse": summary["score"]["rel_mse"],
            "rmse": summary["score"]["rmse"],
            "dominant_error_pattern": summary["error_diagnosis"]["dominant_error_pattern"],
            "physics_issue": summary["physics"]["physics_issue"],
            "residual_ratio": summary["physics"]["residual_ratio"],
            "total_variation_ratio": summary["physics"]["total_variation_ratio"],
        },
        next_hint="select-pde-hypothesis",
    )
```

- [ ] **Step 4: Register state hint and CLI command**

Modify `src/pde_agent/core/state_hints.py`, adding after `"FNO_PREDICTION_READY",`:

```python
    "FNO_ROLLOUT_DIAGNOSED",
```

Modify `src/pde_agent/cli.py`.

Add import:

```python
from pde_agent.commands.diagnose_fno_rollout import diagnose_fno_rollout
```

Add command after `predict-fno-local`:

```python
@app.command("diagnose-fno-rollout")
def diagnose_fno_rollout_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            diagnose_fno_rollout(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )
```

- [ ] **Step 5: Run CLI tests and full tests**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_rollout_diagnosis.py -q
uv run --frozen pytest tests -q
```

Expected result: new CLI test passes and full tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/pde_agent/commands/diagnose_fno_rollout.py src/pde_agent/cli.py src/pde_agent/core/state_hints.py tests/test_fno_rollout_diagnosis.py
git commit -m "feat: expose fno rollout diagnosis cli"
```

---

### Task 3: README And Real Smoke

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add this section after `FNO Backend V0`:

```markdown
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
```

Ensure Markdown fences remain balanced.

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_rollout_diagnosis.py -q
```

Expected result: pass.

- [ ] **Step 3: Run optional real smoke**

Run:

```bash
if test -f data/pdebench/checkpoints/burgers_FNO.tar && test -f data/derived/local_windows_task1_smoke.npz; then
  rm -rf /tmp/pde-agent-fno-rollout-diagnosis-smoke
  uv run --frozen --extra torch pde-agent diagnose-fno-rollout \
    --windows-path data/derived/local_windows_task1_smoke.npz \
    --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
    --output-dir /tmp/pde-agent-fno-rollout-diagnosis-smoke \
    --nu 0.001 \
    --dt 1.0 \
    --dx 1.0 \
    --force
fi
```

Expected result if files exist: command exits 0 and writes all five artifacts.
If files do not exist, record that smoke was skipped.

- [ ] **Step 4: Commit Task 3**

Run:

```bash
git add README.md
git commit -m "docs: document fno rollout diagnosis"
```

---

### Task 4: Final Verification

**Files:**
- No expected source edits. If this task finds a bug, fix the smallest relevant file and add a regression test before committing.

- [ ] **Step 1: Verify no-extra suite**

Run:

```bash
uv run --frozen pytest tests -q
```

Expected result: all tests pass in the current environment.

- [ ] **Step 2: Verify data extra suite**

Run:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected result: all tests pass.

- [ ] **Step 3: Verify torch-specific diagnosis tests**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_rollout_diagnosis.py tests/test_fno_backend_v0.py -q
```

Expected result: all tests pass.

- [ ] **Step 4: Verify CLI help**

Run:

```bash
uv run --frozen pde-agent diagnose-fno-rollout --help
```

Expected output includes:

```text
--windows-path
--checkpoint-path
--output-dir
--nu
--dt
--dx
--force
```

- [ ] **Step 5: Verify real smoke and summary contract**

If real local files exist, run:

```bash
rm -rf /tmp/pde-agent-fno-rollout-diagnosis-final
uv run --frozen --extra torch pde-agent diagnose-fno-rollout \
  --windows-path data/derived/local_windows_task1_smoke.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir /tmp/pde-agent-fno-rollout-diagnosis-final \
  --nu 0.001 \
  --dt 1.0 \
  --dx 1.0 \
  --force
```

Then inspect:

```bash
uv run --frozen python - <<'PY'
import json
from pathlib import Path

root = Path("/tmp/pde-agent-fno-rollout-diagnosis-final")
expected = {
    "fno_prediction.npz",
    "fno_score.json",
    "fno_rollout_error.json",
    "fno_burgers_physics.json",
    "fno_rollout_summary.json",
}
assert {path.name for path in root.iterdir()} == expected
summary = json.loads((root / "fno_rollout_summary.json").read_text())
print(summary["model"]["adapter_type"])
print(summary["score"]["competition_like_score_short"])
print(summary["error_diagnosis"]["dominant_error_pattern"])
print(summary["physics"]["physics_issue"])
assert summary["state_hint"] == "FNO_ROLLOUT_DIAGNOSED"
assert summary["recommendation"]["next_command"] == "select-pde-hypothesis"
PY
```

Expected result: prints model adapter, score, error pattern, and physics issue;
assertions pass.

- [ ] **Step 6: Diff/status checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected result: `git diff --check` prints nothing. `git status --short --branch` shows a clean branch.

- [ ] **Step 7: Final checklist**

Verify directly:

- No FNO model behavior changed.
- No training command was added.
- No scoring or physics formula changed.
- `FNO_ROLLOUT_DIAGNOSED` is in allowed state hints.
- `diagnose-fno-rollout` writes all five artifacts.
- Summary JSON is compact and agent-readable.

No commit is needed for this task unless a fix is required.
