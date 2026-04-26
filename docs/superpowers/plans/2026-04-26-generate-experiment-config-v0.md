# Generate Experiment Config V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `generate-experiment-config`, a deterministic config generator that converts `next_experiment_plan.json` into machine-readable candidate experiment configurations without executing them.

**Architecture:** Implement one core module for plan validation and candidate Cartesian expansion, one thin command wrapper returning the existing JSON payload shape, one Typer CLI registration, state-hint registration, tests, and README docs. The generated config names a future `execute-candidate-experiment` step but does not implement or run it.

**Tech Stack:** Python standard library `json`, `itertools`, `math`, `pathlib`; Typer CLI; pytest; existing `success_payload` and state-hint conventions.

---

## File Structure

- Create `src/pde_agent/core/experiment_config.py`: load/validate plan JSON, expand candidate parameters, write config JSON.
- Create `src/pde_agent/commands/generate_experiment_config.py`: thin success-payload wrapper.
- Modify `src/pde_agent/cli.py`: register `generate-experiment-config`.
- Modify `src/pde_agent/core/state_hints.py`: add `EXPERIMENT_CONFIG_GENERATED`.
- Modify `tests/test_pde_diagnostic_skills_v1.py`: add core and CLI tests.
- Modify `README.md`: document the config generation step after `plan-next-experiment`.

---

### Task 1: Core Experiment Config Generator

**Files:**
- Create: `src/pde_agent/core/experiment_config.py`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Add failing core tests**

Append imports near the existing next-experiment imports in `tests/test_pde_diagnostic_skills_v1.py`:

```python
from pde_agent.core.experiment_config import (
    ExperimentConfigError,
    generate_experiment_config_from_file,
)
```

Append this helper near the next-experiment helpers:

```python
def _write_next_experiment_plan(
    path: Path,
    *,
    max_experiment_count: int = 3,
    candidate_parameters: dict[str, list[object]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "NEXT_EXPERIMENT_PLANNED",
                "selected_intervention": "apply_low_pass_stabilization",
                "source_hypothesis": "apply_low_pass_stabilization",
                "failure_mode": {
                    "dominant_error_pattern": "shock_region_error",
                    "physics_issue": "nonphysical_oscillation",
                    "score_band": "medium",
                },
                "change_scope": "postprocess_or_inference_only",
                "intervention_plan": {
                    "description": "Sweep low-pass thresholds.",
                    "allowed_actions": ["sweep spectral cutoff"],
                    "forbidden_actions": ["overwrite previous artifacts"],
                    "candidate_parameters": (
                        {
                            "low_pass_cutoff_ratio": [0.5, 0.625, 0.75],
                            "preserve_mean": [True],
                        }
                        if candidate_parameters is None
                        else candidate_parameters
                    ),
                },
                "budget": {
                    "max_experiment_count": max_experiment_count,
                    "max_train_time_seconds": 0,
                    "max_inference_time_seconds": 120,
                    "requires_training": False,
                },
                "controls": {
                    "baseline_required": True,
                    "baseline_artifacts": [
                        "previous prediction NPZ",
                        "previous score JSON",
                    ],
                },
                "acceptance_criteria": ["score improves"],
                "rejection_criteria": ["score decreases"],
                "required_next_artifacts": [
                    "candidate prediction NPZ",
                    "candidate score JSON",
                ],
                "next_command": "generate-experiment-config",
                "rationale": "Fixture plan.",
            }
        ),
        encoding="utf-8",
    )
```

Append these tests:

```python
def test_generate_experiment_config_expands_low_pass_candidates(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "next_experiment_plan.json"
    output_path = tmp_path / "experiment_config.json"
    _write_next_experiment_plan(plan_path)

    config = generate_experiment_config_from_file(
        plan_path=plan_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert config["state_hint"] == "EXPERIMENT_CONFIG_GENERATED"
    assert config["source_plan_path"] == str(plan_path)
    assert config["selected_intervention"] == "apply_low_pass_stabilization"
    assert config["change_scope"] == "postprocess_or_inference_only"
    assert config["requires_training"] is False
    assert [candidate["experiment_id"] for candidate in config["candidate_experiments"]] == [
        "apply_low_pass_stabilization_001",
        "apply_low_pass_stabilization_002",
        "apply_low_pass_stabilization_003",
    ]
    assert config["candidate_experiments"][0]["parameters"] == {
        "low_pass_cutoff_ratio": 0.5,
        "preserve_mean": True,
    }
    assert config["candidate_experiments"][2]["parameters"] == {
        "low_pass_cutoff_ratio": 0.75,
        "preserve_mean": True,
    }
    assert config["next_command"] == "execute-candidate-experiment"


def test_generate_experiment_config_respects_budget_limit(tmp_path: Path) -> None:
    plan_path = tmp_path / "next_experiment_plan.json"
    _write_next_experiment_plan(
        plan_path,
        max_experiment_count=2,
        candidate_parameters={
            "alpha": [1, 2],
            "beta": ["a", "b"],
        },
    )

    config = generate_experiment_config_from_file(
        plan_path=plan_path,
        output_path=None,
    )

    assert len(config["candidate_experiments"]) == 2
    assert config["candidate_experiments"][0]["parameters"] == {
        "alpha": 1,
        "beta": "a",
    }
    assert config["candidate_experiments"][1]["parameters"] == {
        "alpha": 1,
        "beta": "b",
    }
```

- [ ] **Step 2: Run focused tests and verify import failure**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_expands_low_pass_candidates tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_respects_budget_limit -q
```

Expected result: fail with `ModuleNotFoundError: No module named 'pde_agent.core.experiment_config'`.

- [ ] **Step 3: Create core implementation**

Create `src/pde_agent/core/experiment_config.py`:

```python
"""Generate deterministic experiment configs from next-experiment plans."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any


class ExperimentConfigError(RuntimeError):
    pass


def generate_experiment_config_from_file(
    *,
    plan_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    plan_file = Path(plan_path)
    plan = _load_json_object(plan_file, "next experiment plan")
    config = generate_experiment_config(plan=plan, source_plan_path=str(plan_file))

    if output_path is not None:
        config_file = Path(output_path)
        config["experiment_config_path"] = str(config_file)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            json.dumps(config, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return config


def generate_experiment_config(
    *,
    plan: dict[str, Any],
    source_plan_path: str,
) -> dict[str, Any]:
    selected_intervention = _require_string(
        plan, "selected_intervention", "next experiment plan"
    )
    change_scope = _require_string(plan, "change_scope", "next experiment plan")
    intervention_plan = _require_object(
        plan, "intervention_plan", "next experiment plan"
    )
    candidate_parameters = _require_object(
        intervention_plan,
        "candidate_parameters",
        "intervention_plan",
    )
    budget = _require_object(plan, "budget", "next experiment plan")
    max_experiment_count = _require_positive_int(
        budget, "max_experiment_count", "budget"
    )
    requires_training = _require_bool(budget, "requires_training", "budget")
    controls = _require_object(plan, "controls", "next experiment plan")
    acceptance_criteria = _require_list(
        plan, "acceptance_criteria", "next experiment plan"
    )
    rejection_criteria = _require_list(
        plan, "rejection_criteria", "next experiment plan"
    )
    required_next_artifacts = _require_list(
        plan, "required_next_artifacts", "next experiment plan"
    )
    forbidden_actions = _require_list(
        intervention_plan, "forbidden_actions", "intervention_plan"
    )
    candidates = _expand_candidates(
        selected_intervention=selected_intervention,
        change_scope=change_scope,
        requires_training=requires_training,
        candidate_parameters=candidate_parameters,
        max_experiment_count=max_experiment_count,
        required_next_artifacts=required_next_artifacts,
        forbidden_actions=forbidden_actions,
    )

    config: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": "EXPERIMENT_CONFIG_GENERATED",
        "source_plan_path": source_plan_path,
        "selected_intervention": selected_intervention,
        "change_scope": change_scope,
        "requires_training": requires_training,
        "budget": budget,
        "candidate_experiments": candidates,
        "controls": controls,
        "acceptance_criteria": acceptance_criteria,
        "rejection_criteria": rejection_criteria,
        "required_artifacts": required_next_artifacts,
        "next_command": "execute-candidate-experiment",
    }
    for optional_field in ("source_hypothesis", "failure_mode", "rationale"):
        if optional_field in plan:
            config[optional_field] = plan[optional_field]
    return config


def _expand_candidates(
    *,
    selected_intervention: str,
    change_scope: str,
    requires_training: bool,
    candidate_parameters: dict[str, Any],
    max_experiment_count: int,
    required_next_artifacts: list[Any],
    forbidden_actions: list[Any],
) -> list[dict[str, Any]]:
    parameter_grid = _validate_parameter_grid(candidate_parameters)
    keys = sorted(parameter_grid)
    combinations = itertools.product(*(parameter_grid[key] for key in keys))
    candidates: list[dict[str, Any]] = []
    for index, values in enumerate(combinations, start=1):
        if len(candidates) >= max_experiment_count:
            break
        parameters = dict(zip(keys, values, strict=True))
        candidates.append(
            {
                "experiment_id": f"{selected_intervention}_{index:03d}",
                "intervention": selected_intervention,
                "change_scope": change_scope,
                "requires_training": requires_training,
                "parameters": parameters,
                "expected_artifacts": required_next_artifacts,
                "forbidden_actions": forbidden_actions,
            }
        )
    return candidates


def _validate_parameter_grid(data: dict[str, Any]) -> dict[str, list[Any]]:
    if not data:
        raise ExperimentConfigError("candidate_parameters must not be empty")
    grid: dict[str, list[Any]] = {}
    for key in sorted(data):
        if not isinstance(key, str):
            raise ExperimentConfigError("candidate parameter keys must be strings")
        values = data[key]
        if not isinstance(values, list):
            raise ExperimentConfigError(
                f"candidate parameter '{key}' must be a list"
            )
        if not values:
            raise ExperimentConfigError(
                f"candidate parameter '{key}' must not be empty"
            )
        for value in values:
            if not _is_json_primitive(value):
                raise ExperimentConfigError(
                    f"candidate parameter '{key}' contains non-primitive value"
                )
        grid[key] = values
    return grid


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentConfigError(f"missing {label} path: {path}")
    if not path.is_file():
        raise ExperimentConfigError(f"{label} path is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentConfigError(
            f"failed to parse {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ExperimentConfigError(f"{label} JSON must be an object")
    return data


def _require_string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ExperimentConfigError(f"{label} must include string field '{field}'")
    return value


def _require_object(data: dict[str, Any], field: str, label: str) -> dict[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{label} must include object field '{field}'")
    return value


def _require_list(data: dict[str, Any], field: str, label: str) -> list[Any]:
    value = data.get(field)
    if not isinstance(value, list):
        raise ExperimentConfigError(f"{label} must include list field '{field}'")
    return value


def _require_bool(data: dict[str, Any], field: str, label: str) -> bool:
    value = data.get(field)
    if not isinstance(value, bool):
        raise ExperimentConfigError(f"{label} must include boolean field '{field}'")
    return value


def _require_positive_int(data: dict[str, Any], field: str, label: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExperimentConfigError(
            f"{label} must include positive integer field '{field}'"
        )
    return value


def _is_json_primitive(value: Any) -> bool:
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False
```

- [ ] **Step 4: Run focused core tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_expands_low_pass_candidates tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_respects_budget_limit -q
```

Expected result: both tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/pde_agent/core/experiment_config.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: add experiment config generator core"
```

---

### Task 2: CLI Wrapper And State Hint

**Files:**
- Create: `src/pde_agent/commands/generate_experiment_config.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/state_hints.py`
- Test: `tests/test_pde_diagnostic_skills_v1.py`

- [ ] **Step 1: Add failing CLI test**

Append this test to `tests/test_pde_diagnostic_skills_v1.py`:

```python
def test_generate_experiment_config_cli_writes_payload(tmp_path: Path) -> None:
    plan_path = tmp_path / "next_experiment_plan.json"
    output_path = tmp_path / "configs" / "experiment_config.json"
    _write_next_experiment_plan(plan_path)

    result = CliRunner().invoke(
        app,
        [
            "generate-experiment-config",
            "--plan-path",
            str(plan_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "generate-experiment-config"
    assert payload["state_hint"] == "EXPERIMENT_CONFIG_GENERATED"
    assert payload["artifacts"]["experiment_config_path"] == str(output_path)
    assert payload["metrics"] == {
        "selected_intervention": "apply_low_pass_stabilization",
        "candidate_count": 3,
        "requires_training": False,
        "change_scope": "postprocess_or_inference_only",
    }
    assert payload["next_hint"] == "execute-candidate-experiment"
    assert output_path.exists()
```

- [ ] **Step 2: Run CLI test and verify failure**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_cli_writes_payload -q
```

Expected result: fail because the CLI command is not registered.

- [ ] **Step 3: Add command wrapper**

Create `src/pde_agent/commands/generate_experiment_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.experiment_config import generate_experiment_config_from_file
from pde_agent.core.json_output import success_payload


def generate_experiment_config(
    *,
    plan_path: str,
    output_path: str,
) -> dict[str, object]:
    config = generate_experiment_config_from_file(
        plan_path=Path(plan_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="generate-experiment-config",
        state_hint="EXPERIMENT_CONFIG_GENERATED",
        artifacts={"experiment_config_path": config["experiment_config_path"]},
        metrics={
            "selected_intervention": config["selected_intervention"],
            "candidate_count": len(config["candidate_experiments"]),
            "requires_training": config["requires_training"],
            "change_scope": config["change_scope"],
        },
        next_hint="execute-candidate-experiment",
    )
```

- [ ] **Step 4: Register state hint**

Modify `src/pde_agent/core/state_hints.py` by adding:

```python
    "EXPERIMENT_CONFIG_GENERATED",
```

Place it after `"NEXT_EXPERIMENT_PLANNED",`.

- [ ] **Step 5: Register CLI command**

Modify `src/pde_agent/cli.py`.

Add import:

```python
from pde_agent.commands.generate_experiment_config import generate_experiment_config
```

Add command after `plan_next_experiment_cmd`:

```python
@app.command("generate-experiment-config")
def generate_experiment_config_cmd(
    plan_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            generate_experiment_config(
                plan_path=plan_path,
                output_path=output_path,
            )
        )
    )
```

- [ ] **Step 6: Run CLI test**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_cli_writes_payload -q
```

Expected result: pass.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add src/pde_agent/commands/generate_experiment_config.py src/pde_agent/cli.py src/pde_agent/core/state_hints.py tests/test_pde_diagnostic_skills_v1.py
git commit -m "feat: expose experiment config generator cli"
```

---

### Task 3: Validation Test And README

**Files:**
- Modify: `tests/test_pde_diagnostic_skills_v1.py`
- Modify: `README.md`

- [ ] **Step 1: Add empty-parameter validation test**

Append this test to `tests/test_pde_diagnostic_skills_v1.py`:

```python
def test_generate_experiment_config_rejects_empty_candidate_parameters(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "next_experiment_plan.json"
    _write_next_experiment_plan(plan_path, candidate_parameters={})

    with pytest.raises(ExperimentConfigError, match="candidate_parameters"):
        generate_experiment_config_from_file(
            plan_path=plan_path,
            output_path=None,
        )


def test_generate_experiment_config_rejects_empty_parameter_values(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "next_experiment_plan.json"
    _write_next_experiment_plan(
        plan_path,
        candidate_parameters={"low_pass_cutoff_ratio": []},
    )

    with pytest.raises(ExperimentConfigError, match="must not be empty"):
        generate_experiment_config_from_file(
            plan_path=plan_path,
            output_path=None,
        )
```

- [ ] **Step 2: Run validation tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_rejects_empty_candidate_parameters tests/test_pde_diagnostic_skills_v1.py::test_generate_experiment_config_rejects_empty_parameter_values -q
```

Expected result: both tests pass.

- [ ] **Step 3: Update README**

Modify `README.md` after the `plan-next-experiment` command in the Lightweight Spectral Baseline section:

```markdown
Generate a concrete candidate experiment configuration from the plan:

```bash
uv run --frozen pde-agent generate-experiment-config \
  --plan-path runs/local_spectral_next_experiment_plan.json \
  --output-path runs/local_spectral_experiment_config.json
```

`generate-experiment-config` expands bounded candidate parameters into explicit
candidate experiment records. It does not run those candidates. The
`execute-candidate-experiment` step is a future execution hook.
```

Ensure Markdown code fences remain balanced.

- [ ] **Step 4: Run focused diagnostic tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected result: pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add tests/test_pde_diagnostic_skills_v1.py README.md
git commit -m "docs: document experiment config generation"
```

---

### Task 4: Final Verification

**Files:**
- No expected source edits. If this task finds a bug, fix the smallest relevant file and add a regression test before committing.

- [ ] **Step 1: Run CLI help**

Run:

```bash
uv run --frozen pde-agent generate-experiment-config --help
```

Expected output includes:

```text
--plan-path
--output-path
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run --frozen pytest tests/test_pde_diagnostic_skills_v1.py -q
```

Expected result: pass.

- [ ] **Step 3: Run full tests**

Run:

```bash
uv run --frozen --extra data pytest tests -q
```

Expected result: pass.

- [ ] **Step 4: Run diff/status checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected result: `git diff --check` prints nothing. `git status --short --branch` shows only the current branch unless a deliberate committed-state check is being done.

- [ ] **Step 5: Final checklist**

Verify directly:

- `src/pde_agent/core/experiment_config.py` has no NumPy, HDF5, PyTorch, model-training, inference, or experiment-execution dependency.
- CLI payload state hint is `EXPERIMENT_CONFIG_GENERATED`.
- Generated config has deterministic candidate experiment IDs.
- `execute-candidate-experiment` is only a future `next_hint` / `next_command`; no command implementation exists.
- README says candidates are not executed.

If no source edits are required, no commit is needed for Task 4.
