"""Audit generated experiment configs before candidate execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentConfigAuditError(RuntimeError):
    pass


def audit_experiment_config_from_file(
    *,
    config_path: str | Path,
    output_path: str | Path | None,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = _load_json_object(config_file, "experiment config")
    source_plan_path = _resolve_plan_path(config=config, explicit_plan_path=plan_path)
    plan_file = Path(source_plan_path)
    plan = _load_json_object(plan_file, "next experiment plan")

    audit = audit_experiment_config(
        config=config,
        plan=plan,
        config_path=str(config_file),
        plan_path=str(plan_file),
    )
    if output_path is not None:
        audit_file = Path(output_path)
        audit["audit_path"] = str(audit_file)
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_file.write_text(
            json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return audit


def audit_experiment_config(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    config_path: str,
    plan_path: str,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    _check_equal(
        issues,
        code="invalid_config_state_hint",
        path="config.state_hint",
        actual=config.get("state_hint"),
        expected="EXPERIMENT_CONFIG_GENERATED",
    )
    _check_equal(
        issues,
        code="invalid_plan_state_hint",
        path="plan.state_hint",
        actual=plan.get("state_hint"),
        expected="NEXT_EXPERIMENT_PLANNED",
    )
    _check_equal(
        issues,
        code="source_plan_path_mismatch",
        path="config.source_plan_path",
        actual=config.get("source_plan_path"),
        expected=plan_path,
    )

    selected_intervention = plan.get("selected_intervention")
    change_scope = plan.get("change_scope")
    budget = plan.get("budget")
    intervention_plan = plan.get("intervention_plan")
    if not isinstance(budget, dict):
        _issue(issues, "missing_plan_budget", "plan.budget")
        budget = {}
    if not isinstance(intervention_plan, dict):
        _issue(issues, "missing_intervention_plan", "plan.intervention_plan")
        intervention_plan = {}
    requires_training = budget.get("requires_training")
    max_experiment_count = budget.get("max_experiment_count")
    candidate_grid = intervention_plan.get("candidate_parameters")
    fixed_parameters = intervention_plan.get("fixed_parameters")
    forbidden_actions = intervention_plan.get("forbidden_actions")
    required_next_artifacts = plan.get("required_next_artifacts")

    _check_equal(
        issues,
        code="selected_intervention_mismatch",
        path="config.selected_intervention",
        actual=config.get("selected_intervention"),
        expected=selected_intervention,
    )
    _check_equal(
        issues,
        code="change_scope_mismatch",
        path="config.change_scope",
        actual=config.get("change_scope"),
        expected=change_scope,
    )
    _check_equal(
        issues,
        code="requires_training_mismatch",
        path="config.requires_training",
        actual=config.get("requires_training"),
        expected=requires_training,
    )
    _check_equal(
        issues,
        code="budget_mismatch",
        path="config.budget",
        actual=config.get("budget"),
        expected=budget,
    )
    _check_equal(
        issues,
        code="next_command_mismatch",
        path="config.next_command",
        actual=config.get("next_command"),
        expected="execute-candidate-experiment",
    )

    candidates = config.get("candidate_experiments")
    if not isinstance(candidates, list):
        _issue(issues, "candidate_experiments_not_list", "config.candidate_experiments")
        candidates = []
    if len(candidates) == 0:
        _issue(issues, "candidate_experiments_empty", "config.candidate_experiments")
    if isinstance(max_experiment_count, int) and len(candidates) > max_experiment_count:
        _issue(
            issues,
            "candidate_count_exceeds_budget",
            "config.candidate_experiments",
            actual=len(candidates),
            expected=max_experiment_count,
        )

    if not isinstance(candidate_grid, dict):
        _issue(issues, "candidate_parameter_grid_missing", "plan.intervention_plan")
        candidate_grid = {}
    if not isinstance(forbidden_actions, list):
        _issue(issues, "forbidden_actions_missing", "plan.intervention_plan")
        forbidden_actions = []
    if not isinstance(required_next_artifacts, list):
        _issue(issues, "required_next_artifacts_missing", "plan")
        required_next_artifacts = []

    for index, candidate in enumerate(candidates):
        _audit_candidate(
            issues=issues,
            index=index,
            candidate=candidate,
            selected_intervention=selected_intervention,
            change_scope=change_scope,
            requires_training=requires_training,
            candidate_grid=candidate_grid,
            fixed_parameters=fixed_parameters,
            forbidden_actions=forbidden_actions,
            required_next_artifacts=required_next_artifacts,
        )

    decision = "approved_for_execution" if not issues else "rejected"
    return {
        "schema_version": 1,
        "state_hint": "EXPERIMENT_CONFIG_AUDITED",
        "decision": decision,
        "issue_count": len(issues),
        "issues": issues,
        "config_path": config_path,
        "plan_path": plan_path,
        "summary": {
            "selected_intervention": config.get("selected_intervention"),
            "candidate_count": len(candidates),
            "requires_training": config.get("requires_training"),
            "next_command": config.get("next_command"),
        },
    }


def _audit_candidate(
    *,
    issues: list[dict[str, Any]],
    index: int,
    candidate: Any,
    selected_intervention: Any,
    change_scope: Any,
    requires_training: Any,
    candidate_grid: dict[Any, Any],
    fixed_parameters: Any,
    forbidden_actions: list[Any],
    required_next_artifacts: list[Any],
) -> None:
    path = f"config.candidate_experiments[{index}]"
    if not isinstance(candidate, dict):
        _issue(issues, "candidate_not_object", path)
        return
    _check_equal(
        issues,
        code="candidate_intervention_mismatch",
        path=f"{path}.intervention",
        actual=candidate.get("intervention"),
        expected=selected_intervention,
    )
    _check_equal(
        issues,
        code="candidate_change_scope_mismatch",
        path=f"{path}.change_scope",
        actual=candidate.get("change_scope"),
        expected=change_scope,
    )
    _check_equal(
        issues,
        code="candidate_requires_training_mismatch",
        path=f"{path}.requires_training",
        actual=candidate.get("requires_training"),
        expected=requires_training,
    )
    _check_equal(
        issues,
        code="candidate_expected_artifacts_mismatch",
        path=f"{path}.expected_artifacts",
        actual=candidate.get("expected_artifacts"),
        expected=required_next_artifacts,
    )
    _check_equal(
        issues,
        code="candidate_forbidden_actions_mismatch",
        path=f"{path}.forbidden_actions",
        actual=candidate.get("forbidden_actions"),
        expected=forbidden_actions,
    )
    if fixed_parameters is not None:
        _check_equal(
            issues,
            code="candidate_fixed_parameters_mismatch",
            path=f"{path}.fixed_parameters",
            actual=candidate.get("fixed_parameters"),
            expected=fixed_parameters,
        )

    parameters = candidate.get("parameters")
    if not isinstance(parameters, dict):
        _issue(issues, "candidate_parameters_not_object", f"{path}.parameters")
        return
    for parameter_name, parameter_value in parameters.items():
        allowed_values = candidate_grid.get(parameter_name)
        if not isinstance(allowed_values, list) or parameter_value not in allowed_values:
            _issue(
                issues,
                "candidate_parameter_out_of_grid",
                f"{path}.parameters.{parameter_name}",
                actual=parameter_value,
                expected=allowed_values,
            )
    for parameter_name in candidate_grid:
        if parameter_name not in parameters:
            _issue(
                issues,
                "candidate_parameter_missing",
                f"{path}.parameters.{parameter_name}",
            )


def _resolve_plan_path(
    *,
    config: dict[str, Any],
    explicit_plan_path: str | Path | None,
) -> str:
    if explicit_plan_path is not None:
        return str(Path(explicit_plan_path))
    source_plan_path = config.get("source_plan_path")
    if not isinstance(source_plan_path, str):
        raise ExperimentConfigAuditError(
            "experiment config must include string source_plan_path when "
            "plan_path is omitted"
        )
    return source_plan_path


def _check_equal(
    issues: list[dict[str, Any]],
    *,
    code: str,
    path: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        _issue(issues, code, path, actual=actual, expected=expected)


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    path: str,
    *,
    actual: Any = None,
    expected: Any = None,
) -> None:
    issue = {"severity": "error", "code": code, "path": path}
    if actual is not None:
        issue["actual"] = actual
    if expected is not None:
        issue["expected"] = expected
    issues.append(issue)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentConfigAuditError(f"missing {label} path: {path}")
    if not path.is_file():
        raise ExperimentConfigAuditError(f"{label} path is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentConfigAuditError(
            f"failed to parse {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ExperimentConfigAuditError(f"{label} JSON must be an object")
    return data
