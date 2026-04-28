"""Generate deterministic experiment configs from next-experiment plans."""

from __future__ import annotations

import itertools
import json
import math
from copy import deepcopy
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
        _validate_json_value(config, "experiment config")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            serialized_config = (
                json.dumps(config, allow_nan=False, indent=2, sort_keys=True) + "\n"
            )
        except ValueError as exc:
            raise ExperimentConfigError(
                f"experiment config contains invalid JSON value: {exc}"
            ) from exc
        config_file.write_text(serialized_config, encoding="utf-8")

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
    fixed_parameters = _optional_object(
        intervention_plan,
        "fixed_parameters",
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
        fixed_parameters=fixed_parameters,
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
        "budget": deepcopy(budget),
        "candidate_experiments": candidates,
        "controls": deepcopy(controls),
        "acceptance_criteria": deepcopy(acceptance_criteria),
        "rejection_criteria": deepcopy(rejection_criteria),
        "required_artifacts": deepcopy(required_next_artifacts),
        "next_command": "execute-candidate-experiment",
    }
    for optional_field in ("source_hypothesis", "failure_mode", "rationale"):
        if optional_field in plan:
            config[optional_field] = deepcopy(plan[optional_field])
    _validate_json_value(config, "experiment config")
    return config


def _expand_candidates(
    *,
    selected_intervention: str,
    change_scope: str,
    requires_training: bool,
    candidate_parameters: dict[str, Any],
    fixed_parameters: dict[str, Any] | None,
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
        candidate = {
            "experiment_id": f"{selected_intervention}_{index:03d}",
            "intervention": selected_intervention,
            "change_scope": change_scope,
            "requires_training": requires_training,
            "parameters": parameters,
            "expected_artifacts": deepcopy(required_next_artifacts),
            "forbidden_actions": deepcopy(forbidden_actions),
        }
        if fixed_parameters is not None:
            candidate["fixed_parameters"] = deepcopy(fixed_parameters)
        candidates.append(candidate)
    return candidates


def _validate_parameter_grid(data: dict[str, Any]) -> dict[str, list[Any]]:
    if not data:
        raise ExperimentConfigError("candidate_parameters must not be empty")
    for key in data:
        if not isinstance(key, str):
            raise ExperimentConfigError("candidate parameter keys must be strings")
    grid: dict[str, list[Any]] = {}
    for key in sorted(data):
        values = data[key]
        if not isinstance(values, list):
            raise ExperimentConfigError(f"candidate parameter '{key}' must be a list")
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


def _optional_object(
    data: dict[str, Any], field: str, label: str
) -> dict[str, Any] | None:
    if field not in data:
        return None
    value = data[field]
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{label} field '{field}' must be an object")
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


def _validate_json_value(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ExperimentConfigError(
                    f"{path} contains non-string object key {key!r}"
                )
            _validate_json_value(child, f"{path}.{key}")
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return

    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentConfigError(f"{path} contains non-finite numeric value")

    if value is None or isinstance(value, str | bool | int | float):
        return

    raise ExperimentConfigError(
        f"{path} contains non-JSON primitive value of type {type(value).__name__}"
    )
