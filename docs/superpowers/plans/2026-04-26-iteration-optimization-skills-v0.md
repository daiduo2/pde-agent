# Iteration Optimization Skills V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic `plan-next-experiment` skill that converts local PDE diagnostics into a machine-readable next-experiment plan.

**Architecture:** Implement one focused core planner that reads score, rollout-error, physics, and hypothesis JSON artifacts and writes `next_experiment_plan.json`. Add a thin command wrapper, Typer CLI registration, state-hint registration, tests, and README usage docs. Keep phase 2 `generate-experiment-config` out of implementation scope; it appears only as `next_hint` / `next_command`.

**Tech Stack:** Python standard library JSON/pathlib, Typer CLI, pytest, existing `success_payload` and `ALLOWED_STATE_HINTS` patterns.

---

## File Structure

- Create `src/pde_agent/core/next_experiment_plan.py`: deterministic planner, JSON loading, validation, intervention policy table, score-band classification, artifact writing.
- Create `src/pde_agent/commands/plan_next_experiment.py`: thin wrapper around the core planner returning the standard JSON success payload.
- Modify `src/pde_agent/cli.py`: import and register `plan-next-experiment`.
- Modify `src/pde_agent/core/state_hints.py`: add `NEXT_EXPERIMENT_PLANNED`.
- Modify `tests/test_pde_diagnostic_skills_v1.py`: add planner core and CLI tests.
- Modify `README.md`: document the command after `select-pde-hypothesis`.

---

### Task 1: Core Next-Experiment Planner

**Files:**
- Create: `src/pde_agent/core/next_experiment_plan.py`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Add failing tests for low-pass and drift planning**

Append these helpers and tests to `tests/test_pde_diagnostic_skills_v1.py` near the existing PDE hypothesis tests:

```python
from pde_agent.core.next_experiment_plan import (
    NextExperimentPlanError,
    plan_next_experiment_from_files,
)


def _write_error_diagnosis(path: Path, pattern: str) -> None:
    path.write_text(
        json.dumps(
            {
                "dominant_error_pattern": pattern,
                "global_rmse": 0.05,
                "time_error_slope": 0.01,
            }
        ),
        encoding="utf-8",
    )


def _write_physics_check(path: Path, issue: str) -> None:
    path.write_text(
        json.dumps(
            {
                "physics_issue": issue,
                "residual_ratio": 0.9,
                "total_variation_ratio": 2.7,
            }
        ),
        encoding="utf-8",
    )


def _write_hypothesis(path: Path, hypothesis: str) -> None:
    path.write_text(
        json.dumps(
            {
                "selected_hypothesis": hypothesis,
                "next_experiment_scope": "local ablation",
            }
        ),
        encoding="utf-8",
    )
```

Then add:

```python
def test_plan_next_experiment_low_pass_for_nonphysical_oscillation(
    tmp_path: Path,
) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    output_path = tmp_path / "next_experiment_plan.json"
    _write_score(score_path, score=77.18, rmse=0.0506)
    _write_error_diagnosis(error_path, "shock_region_error")
    _write_physics_check(physics_path, "nonphysical_oscillation")
    _write_hypothesis(hypothesis_path, "apply_low_pass_stabilization")

    plan = plan_next_experiment_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        hypothesis_path=hypothesis_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert plan["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert plan["selected_intervention"] == "apply_low_pass_stabilization"
    assert plan["source_hypothesis"] == "apply_low_pass_stabilization"
    assert plan["failure_mode"]["score_band"] == "medium"
    assert plan["change_scope"] == "postprocess_or_inference_only"
    assert plan["budget"]["requires_training"] is False
    assert plan["budget"]["max_experiment_count"] == 3
    assert plan["intervention_plan"]["candidate_parameters"][
        "low_pass_cutoff_ratio"
    ] == [0.5, 0.625, 0.75]
    assert plan["next_command"] == "generate-experiment-config"


def test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune(
    tmp_path: Path,
) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    _write_score(score_path, score=45.0, rmse=0.5)
    _write_error_diagnosis(error_path, "long_horizon_drift")
    _write_physics_check(physics_path, "inconclusive")
    _write_hypothesis(hypothesis_path, "use_scheduled_sampling_rollout_finetune")

    plan = plan_next_experiment_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        hypothesis_path=hypothesis_path,
        output_path=None,
    )

    assert plan["selected_intervention"] == "use_scheduled_sampling_rollout_finetune"
    assert plan["change_scope"] == "training_schedule_only"
    assert plan["budget"]["requires_training"] is True
    assert plan["budget"]["max_experiment_count"] == 2
    assert plan["failure_mode"]["score_band"] == "low"
    assert plan["intervention_plan"]["candidate_parameters"]["rollout_steps"] == [
        10,
        20,
        31,
    ]
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_low_pass_for_nonphysical_oscillation tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune -q
```

Expected result: fail with `ModuleNotFoundError: No module named 'pde_agent.core.next_experiment_plan'`.

- [ ] **Step 3: Create the core planner**

Create `src/pde_agent/core/next_experiment_plan.py`:

```python
"""Deterministic next-experiment planning from PDE diagnostic artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class NextExperimentPlanError(RuntimeError):
    pass


def plan_next_experiment_from_files(
    *,
    score_path: str | Path,
    error_diagnosis_path: str | Path,
    physics_check_path: str | Path,
    hypothesis_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    score_file = Path(score_path)
    error_file = Path(error_diagnosis_path)
    physics_file = Path(physics_check_path)
    hypothesis_file = Path(hypothesis_path)
    score = _load_json_object(score_file, "score")
    error = _load_json_object(error_file, "error diagnosis")
    physics = _load_json_object(physics_file, "physics check")
    hypothesis = _load_json_object(hypothesis_file, "hypothesis")

    plan = plan_next_experiment(
        score=score,
        error=error,
        physics=physics,
        hypothesis=hypothesis,
    )
    plan.update(
        {
            "score_path": str(score_file),
            "error_diagnosis_path": str(error_file),
            "physics_check_path": str(physics_file),
            "hypothesis_path": str(hypothesis_file),
        }
    )

    if output_path is not None:
        plan_file = Path(output_path)
        plan["next_experiment_plan_path"] = str(plan_file)
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(
            json.dumps(plan, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return plan


def plan_next_experiment(
    *,
    score: dict[str, Any],
    error: dict[str, Any],
    physics: dict[str, Any],
    hypothesis: dict[str, Any],
) -> dict[str, Any]:
    selected_hypothesis = _require_string(
        hypothesis, "selected_hypothesis", "hypothesis"
    )
    dominant_pattern = _require_string(
        error, "dominant_error_pattern", "error diagnosis"
    )
    physics_issue = _require_string(physics, "physics_issue", "physics check")
    selected_intervention = _select_intervention(
        selected_hypothesis=selected_hypothesis,
        dominant_pattern=dominant_pattern,
        physics_issue=physics_issue,
    )
    policy = deepcopy(_INTERVENTION_POLICIES[selected_intervention])
    score_value = _number_or_none(score.get("competition_like_score_short"))
    evidence = {
        "competition_like_score_short": score_value,
        "rel_mse": _number_or_none(score.get("rel_mse")),
        "rmse": _number_or_none(score.get("rmse")),
        "residual_ratio": _number_or_none(physics.get("residual_ratio")),
        "total_variation_ratio": _number_or_none(
            physics.get("total_variation_ratio")
        ),
    }

    return {
        "schema_version": 1,
        "state_hint": "NEXT_EXPERIMENT_PLANNED",
        "selected_intervention": selected_intervention,
        "source_hypothesis": selected_hypothesis,
        "failure_mode": {
            "dominant_error_pattern": dominant_pattern,
            "physics_issue": physics_issue,
            "score_band": _score_band(score_value),
            "evidence": evidence,
        },
        "change_scope": policy["change_scope"],
        "intervention_plan": policy["intervention_plan"],
        "budget": policy["budget"],
        "controls": {
            "baseline_required": True,
            "baseline_artifacts": [
                "previous prediction NPZ",
                "previous score JSON",
                "previous physics check JSON",
            ],
        },
        "acceptance_criteria": policy["acceptance_criteria"],
        "rejection_criteria": policy["rejection_criteria"],
        "required_next_artifacts": [
            "candidate prediction NPZ",
            "candidate score JSON",
            "candidate rollout error diagnosis JSON",
            "candidate Burgers physics check JSON",
            "candidate selected hypothesis JSON",
        ],
        "next_command": "generate-experiment-config",
        "rationale": policy["rationale"].format(
            selected_hypothesis=selected_hypothesis,
            dominant_pattern=dominant_pattern,
            physics_issue=physics_issue,
        ),
    }


def _select_intervention(
    *,
    selected_hypothesis: str,
    dominant_pattern: str,
    physics_issue: str,
) -> str:
    if physics_issue == "nonphysical_oscillation":
        return "apply_low_pass_stabilization"
    if dominant_pattern == "long_horizon_drift":
        return "use_scheduled_sampling_rollout_finetune"
    if selected_hypothesis in _INTERVENTION_POLICIES:
        return selected_hypothesis
    if dominant_pattern == "shock_region_error":
        return "add_gradient_weighted_loss"
    if physics_issue == "residual_mismatch":
        return "add_residual_correction_head"
    if dominant_pattern in {"overdamped_rollout", "overdamped"}:
        return "add_spectral_consistency_loss"
    return "keep_baseline_and_collect_stronger_evidence"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise NextExperimentPlanError(f"missing {label} path: {path}")
    if not path.is_file():
        raise NextExperimentPlanError(f"{label} path is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NextExperimentPlanError(
            f"failed to parse {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise NextExperimentPlanError(f"{label} JSON must be an object")
    return data


def _require_string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise NextExperimentPlanError(f"{label} must include string field '{field}'")
    return value


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _score_band(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 95.0:
        return "excellent"
    if score >= 85.0:
        return "high"
    if score >= 60.0:
        return "medium"
    return "low"


_INTERVENTION_POLICIES: dict[str, dict[str, Any]] = {
    "apply_low_pass_stabilization": {
        "change_scope": "postprocess_or_inference_only",
        "intervention_plan": {
            "description": (
                "Sweep low-pass thresholds to suppress nonphysical oscillations "
                "while auditing shock-region error."
            ),
            "allowed_actions": [
                "add deterministic low-pass postprocessing",
                "sweep spectral cutoff",
                "compare against unchanged prediction",
            ],
            "forbidden_actions": [
                "change training data split",
                "submit without baseline comparison",
                "overwrite previous artifacts",
            ],
            "candidate_parameters": {
                "low_pass_cutoff_ratio": [0.5, 0.625, 0.75],
                "preserve_mean": [True],
            },
        },
        "budget": {
            "max_experiment_count": 3,
            "max_train_time_seconds": 0,
            "max_inference_time_seconds": 120,
            "requires_training": False,
        },
        "acceptance_criteria": [
            "competition_like_score_short improves by at least 1.0",
            "total_variation_ratio moves closer to 1.0",
            "shock-region error does not worsen by more than 5%",
        ],
        "rejection_criteria": [
            "score decreases by more than 0.5",
            "physics issue changes to overdissipation",
            "prediction contains non-finite values",
        ],
        "rationale": (
            "Selected because physics_issue={physics_issue} and "
            "source_hypothesis={selected_hypothesis} indicate oscillation control."
        ),
    },
    "add_gradient_weighted_loss": {
        "change_scope": "training_loss_only",
        "intervention_plan": {
            "description": (
                "Ablate gradient-weighted losses on shock-heavy windows while "
                "tracking smooth-region regression."
            ),
            "allowed_actions": [
                "add gradient-weighted loss term",
                "sweep shock quantile",
                "compare shock-region error against unchanged baseline",
            ],
            "forbidden_actions": [
                "drop global score checks",
                "train without baseline artifact comparison",
            ],
            "candidate_parameters": {
                "gradient_weight": [0.5, 1.0, 2.0],
                "shock_quantile": [0.85, 0.9],
            },
        },
        "budget": {
            "max_experiment_count": 2,
            "max_train_time_seconds": 7200,
            "max_inference_time_seconds": 120,
            "requires_training": True,
        },
        "acceptance_criteria": [
            "shock-region error evidence improves",
            "competition_like_score_short does not decrease by more than 0.5",
        ],
        "rejection_criteria": [
            "smooth-region error dominates after weighting",
            "training exceeds budget",
        ],
        "rationale": (
            "Selected because dominant_error_pattern={dominant_pattern} points "
            "to localized sharp-gradient failure."
        ),
    },
    "use_scheduled_sampling_rollout_finetune": {
        "change_scope": "training_schedule_only",
        "intervention_plan": {
            "description": (
                "Run a short rollout-finetune schedule to reduce exposure-bias "
                "drift over later prediction steps."
            ),
            "allowed_actions": [
                "sweep teacher forcing decay",
                "sweep rollout training length",
                "audit early/middle/late error",
            ],
            "forbidden_actions": [
                "optimize late horizon while ignoring early-step collapse",
                "change dataset split during schedule comparison",
            ],
            "candidate_parameters": {
                "teacher_forcing_decay": [0.95, 0.9, 0.85],
                "rollout_steps": [10, 20, 31],
            },
        },
        "budget": {
            "max_experiment_count": 2,
            "max_train_time_seconds": 7200,
            "max_inference_time_seconds": 120,
            "requires_training": True,
        },
        "acceptance_criteria": [
            "late error improves without early error collapse",
            "time_error_slope moves closer to zero",
        ],
        "rejection_criteria": [
            "early-step score regresses materially",
            "rollout becomes numerically unstable",
        ],
        "rationale": (
            "Selected because dominant_error_pattern={dominant_pattern} points "
            "to exposure-bias drift."
        ),
    },
    "add_spectral_consistency_loss": {
        "change_scope": "training_loss_only",
        "intervention_plan": {
            "description": (
                "Ablate spectral consistency loss to reduce overdamping while "
                "guarding against oscillation growth."
            ),
            "allowed_actions": [
                "add spectral consistency loss",
                "sweep spectral loss weight",
                "audit spectral energy alignment",
            ],
            "forbidden_actions": [
                "increase high-frequency energy without physics checks",
                "skip total variation audit",
            ],
            "candidate_parameters": {
                "spectral_loss_weight": [0.01, 0.05, 0.1],
                "spectral_band": ["mid", "high"],
            },
        },
        "budget": {
            "max_experiment_count": 2,
            "max_train_time_seconds": 7200,
            "max_inference_time_seconds": 120,
            "requires_training": True,
        },
        "acceptance_criteria": [
            "spectral energy alignment improves",
            "physics issue does not become nonphysical_oscillation",
        ],
        "rejection_criteria": [
            "total variation ratio grows away from 1.0",
            "score decreases by more than 0.5",
        ],
        "rationale": (
            "Selected because source_hypothesis={selected_hypothesis} indicates "
            "spectral damping correction."
        ),
    },
    "add_residual_correction_head": {
        "change_scope": "model_head_only",
        "intervention_plan": {
            "description": (
                "Add a residual correction head only after simpler diagnostics "
                "indicate persistent PDE residual mismatch."
            ),
            "allowed_actions": [
                "add residual correction head",
                "sweep residual loss weight",
                "compare residual ratio against unchanged model",
            ],
            "forbidden_actions": [
                "replace the full backbone in this experiment",
                "skip local score comparison",
            ],
            "candidate_parameters": {
                "residual_head_width": [16, 32],
                "residual_loss_weight": [0.01, 0.05],
            },
        },
        "budget": {
            "max_experiment_count": 2,
            "max_train_time_seconds": 7200,
            "max_inference_time_seconds": 120,
            "requires_training": True,
        },
        "acceptance_criteria": [
            "residual_ratio improves",
            "competition_like_score_short does not regress",
        ],
        "rejection_criteria": [
            "residual proxy improves but score collapses",
            "training exceeds budget",
        ],
        "rationale": (
            "Selected because physics_issue={physics_issue} indicates residual "
            "mismatch."
        ),
    },
    "keep_baseline_and_collect_stronger_evidence": {
        "change_scope": "data_diagnostic_only",
        "intervention_plan": {
            "description": (
                "Collect more local diagnostic windows before changing model or "
                "training code."
            ),
            "allowed_actions": [
                "build additional local windows",
                "rerun scoring",
                "rerun rollout and physics diagnostics",
            ],
            "forbidden_actions": [
                "modify model code before evidence stabilizes",
                "submit speculative changes",
            ],
            "candidate_parameters": {
                "sample_limit": [16, 32, 64],
                "start_index": [0, 128, 512],
            },
        },
        "budget": {
            "max_experiment_count": 3,
            "max_train_time_seconds": 0,
            "max_inference_time_seconds": 120,
            "requires_training": False,
        },
        "acceptance_criteria": [
            "repeat diagnostics identify a stable dominant failure mode",
            "score band is reproducible across windows",
        ],
        "rejection_criteria": [
            "diagnostics remain contradictory after three reruns",
        ],
        "rationale": (
            "Selected because current evidence is insufficient for a targeted "
            "model intervention."
        ),
    },
}
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_low_pass_for_nonphysical_oscillation tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune -q
```

Expected result: both tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/pde_agent/core/next_experiment_plan.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: add next experiment planner core"
```

---

### Task 2: CLI Command Wrapper And State Hint

**Files:**
- Create: `src/pde_agent/commands/plan_next_experiment.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Add failing CLI payload test**

Append this test to `tests/test_pde_diagnostic_skills_v1.py`:

```python
def test_plan_next_experiment_cli_writes_payload(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    output_path = tmp_path / "plans" / "next_experiment.json"
    _write_score(score_path, score=77.18, rmse=0.0506)
    _write_error_diagnosis(error_path, "shock_region_error")
    _write_physics_check(physics_path, "nonphysical_oscillation")
    _write_hypothesis(hypothesis_path, "apply_low_pass_stabilization")

    result = CliRunner().invoke(
        app,
        [
            "plan-next-experiment",
            "--score-path",
            str(score_path),
            "--error-diagnosis-path",
            str(error_path),
            "--physics-check-path",
            str(physics_path),
            "--hypothesis-path",
            str(hypothesis_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "plan-next-experiment"
    assert payload["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert payload["artifacts"]["next_experiment_plan_path"] == str(output_path)
    assert payload["metrics"] == {
        "selected_intervention": "apply_low_pass_stabilization",
        "change_scope": "postprocess_or_inference_only",
        "max_experiment_count": 3,
        "stop_condition": "generate-experiment-config",
    }
    assert payload["next_hint"] == "generate-experiment-config"
    assert output_path.exists()
```

- [ ] **Step 2: Run the CLI test and verify failure**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_cli_writes_payload -q
```

Expected result: fail because `plan-next-experiment` is not registered.

- [ ] **Step 3: Add command wrapper**

Create `src/pde_agent/commands/plan_next_experiment.py`:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.next_experiment_plan import plan_next_experiment_from_files


def plan_next_experiment(
    *,
    score_path: str,
    error_diagnosis_path: str,
    physics_check_path: str,
    hypothesis_path: str,
    output_path: str,
) -> dict[str, object]:
    result = plan_next_experiment_from_files(
        score_path=Path(score_path),
        error_diagnosis_path=Path(error_diagnosis_path),
        physics_check_path=Path(physics_check_path),
        hypothesis_path=Path(hypothesis_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="plan-next-experiment",
        state_hint="NEXT_EXPERIMENT_PLANNED",
        artifacts={"next_experiment_plan_path": result["next_experiment_plan_path"]},
        metrics={
            "selected_intervention": result["selected_intervention"],
            "change_scope": result["change_scope"],
            "max_experiment_count": result["budget"]["max_experiment_count"],
            "stop_condition": result["next_command"],
        },
        next_hint="generate-experiment-config",
    )
```

- [ ] **Step 4: Register state hint**

Modify `src/pde_agent/core/state_hints.py` by adding:

```python
    "NEXT_EXPERIMENT_PLANNED",
```

Place it after `"PDE_HYPOTHESIS_SELECTED",`.

- [ ] **Step 5: Register CLI command**

Modify `src/pde_agent/cli.py`.

Add the import near the other command imports:

```python
from pde_agent.commands.plan_next_experiment import plan_next_experiment
```

Add this command after `select_pde_hypothesis_cmd`:

```python
@app.command("plan-next-experiment")
def plan_next_experiment_cmd(
    score_path: str = typer.Option(...),
    error_diagnosis_path: str = typer.Option(...),
    physics_check_path: str = typer.Option(...),
    hypothesis_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            plan_next_experiment(
                score_path=score_path,
                error_diagnosis_path=error_diagnosis_path,
                physics_check_path=physics_check_path,
                hypothesis_path=hypothesis_path,
                output_path=output_path,
            )
        )
    )
```

- [ ] **Step 6: Run focused CLI test**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_cli_writes_payload -q
```

Expected result: pass.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add src/pde_agent/commands/plan_next_experiment.py src/pde_agent/cli.py src/pde_agent/core/state_hints.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: expose next experiment planner cli"
```

---

### Task 3: Validation Tests And README

**Files:**
- Modify: `tests/test_pde_diagnostic_skills_v1.py`
- Modify: `README.md`

- [ ] **Step 1: Add missing-field validation test**

Append this test to `tests/test_pde_diagnostic_skills_v1.py`:

```python
def test_plan_next_experiment_rejects_missing_required_fields(
    tmp_path: Path,
) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    _write_score(score_path, score=80.0, rmse=0.1)
    _write_error_diagnosis(error_path, "shock_region_error")
    physics_path.write_text(json.dumps({"residual_ratio": 1.0}), encoding="utf-8")
    _write_hypothesis(hypothesis_path, "add_gradient_weighted_loss")

    with pytest.raises(NextExperimentPlanError, match="physics_issue"):
        plan_next_experiment_from_files(
            score_path=score_path,
            error_diagnosis_path=error_path,
            physics_check_path=physics_path,
            hypothesis_path=hypothesis_path,
            output_path=None,
        )
```

- [ ] **Step 2: Run validation test**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_plan_next_experiment_rejects_missing_required_fields -q
```

Expected result: pass.

- [ ] **Step 3: Add README command sequence**

Modify `README.md` after the `select-pde-hypothesis` command in the `Lightweight Spectral Baseline` section:

```markdown
Use the selected hypothesis plus diagnostics to plan the next bounded experiment:

```bash
uv run --frozen pde-agent plan-next-experiment \
  --score-path runs/local_spectral_score.json \
  --error-diagnosis-path runs/local_spectral_rollout_error.json \
  --physics-check-path runs/local_spectral_burgers_physics.json \
  --hypothesis-path runs/local_spectral_pde_hypothesis.json \
  --output-path runs/local_spectral_next_experiment_plan.json
```

The planner emits machine-readable JSON only. It does not modify model code,
run experiments, or generate human-readable research notes. The next intended
step is a future `generate-experiment-config` command.
```

When inserting into Markdown, close and reopen code fences correctly so the README renders.

- [ ] **Step 4: Run focused PDE diagnostic tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected result: all tests in the file pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add tests/test_pde_diagnostic_skills_v1.py README.md
git commit -m "docs: document next experiment planning"
```

---

### Task 4: End-To-End Verification

**Files:**
- No expected source edits. If this task finds a bug, fix in the smallest relevant file and add a regression test before committing.

- [ ] **Step 1: Run CLI help**

Run:

```bash
uv run --frozen pde-agent plan-next-experiment --help
```

Expected output includes required options:

```text
--score-path
--error-diagnosis-path
--physics-check-path
--hypothesis-path
--output-path
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected result: pass.

- [ ] **Step 3: Run full test suite**

Run:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected result: pass.

- [ ] **Step 4: Run diff and status checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected result: `git diff --check` prints nothing. `git status --short --branch` shows only the current branch unless a deliberate committed-state check is being done.

- [ ] **Step 5: Final review checklist**

Verify these requirements directly:

- `src/pde_agent/core/next_experiment_plan.py` contains no HDF5, NumPy, PyTorch, or model-training dependency.
- `plan-next-experiment` writes a JSON artifact with `state_hint = NEXT_EXPERIMENT_PLANNED`.
- README says the planner emits machine-readable JSON only.
- No `generate-experiment-config` implementation exists yet.
- Existing model and diagnostic commands still pass tests.

If no source edits are required, no commit is needed for Task 4.
