"""Executable blueprint DSL normalization and validation."""

import json
import re
from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError

EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT = "EXECUTABLE_BLUEPRINTS_VALIDATED"

EXECUTABLE_BLUEPRINT_BASELINES = {
    "persistence",
    "fno_checkpoint",
    "retrieval_prior",
}

EXECUTABLE_BLUEPRINT_TRANSFORMS = {
    "linear_trend",
    "spectral_smooth",
    "residual_fit",
    "retrieval_prior",
    "retrieval_refinement",
    "checkpoint_calibration",
}

EXECUTABLE_BLUEPRINT_TRAINING_MODES = {
    "none",
    "finetune",
    "supervised",
    "physics_loss",
}

EXECUTABLE_BLUEPRINT_EXECUTORS = {
    "controlled_numpy_operator",
    "controlled_numpy_residual_operator",
    "controlled_numpy_retrieval_prior",
    "controlled_numpy_retrieval_refinement",
    "controlled_numpy_task1_early_refinement",
    "controlled_numpy_sem_burgers_rollout",
    "controlled_numpy_burgers_residual_refinement",
    "controlled_numpy_pi_deeponet_correction_head",
    "official_checkpoint_candidate_pipeline",
}

def validate_executable_blueprints(
    *,
    blueprints_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    blueprint_file = _require_file(Path(blueprints_path), "blueprints")
    payload = _read_json(blueprint_file)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    report = audit_executable_blueprint_payload(
        payload,
        blueprints_path=str(blueprint_file),
    )
    report["executable_blueprint_audit_path"] = str(output_file)
    _write_json(output_file, report)
    return report

def audit_executable_blueprint_payload(
    payload: dict[str, Any],
    *,
    blueprints_path: str | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    blueprints_value = payload.get("blueprints")
    if not isinstance(blueprints_value, list):
        _append_blueprint_issue(
            issues,
            code="invalid_blueprints",
            path="blueprints",
            failure_code="blueprint_schema_invalid",
            message="blueprints must be a list",
        )
        blueprints: list[Any] = []
    else:
        blueprints = blueprints_value
    normalized = []
    for index, blueprint in enumerate(blueprints):
        item = _normalize_executable_blueprint(blueprint, index=index, issues=issues)
        if item is not None:
            normalized.append(item)
            _validate_executable_blueprint(item, index=index, issues=issues)
    decision = "approved" if not issues else "rejected"
    report: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT,
        "decision": decision,
        "blueprint_count": len(blueprints),
        "normalized_blueprint_count": len(normalized),
        "issue_count": len(issues),
        "issues": issues,
        "normalized_blueprints": normalized,
        "next_command": "run-blueprint-sandbox"
        if decision == "approved"
        else "fix-executable-blueprint-dsl",
    }
    if blueprints_path is not None:
        report["blueprints_path"] = blueprints_path
    return report



def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_executable_blueprint(
    blueprint: Any,
    *,
    index: int,
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    base = f"blueprints[{index}]"
    if not isinstance(blueprint, dict):
        _append_blueprint_issue(
            issues,
            code="invalid_blueprint",
            path=base,
            failure_code="blueprint_schema_invalid",
            message="blueprint must be an object",
        )
        return None
    if _has_native_executable_blueprint_shape(blueprint):
        return dict(blueprint)
    blueprint_id = blueprint.get("blueprint_id")
    candidate_family = blueprint.get("candidate_family")
    sandbox = blueprint.get("sandbox") if isinstance(blueprint.get("sandbox"), dict) else {}
    parameter_grid = (
        blueprint.get("parameter_grid")
        if isinstance(blueprint.get("parameter_grid"), dict)
        else {}
    )
    executor_id = sandbox.get("allowed_executor")
    source_candidate_id = blueprint.get("source_candidate_id") or blueprint.get(
        "hypothesis_id"
    )
    baseline = _blueprint_baseline_for_family(str(candidate_family or ""))
    transforms = _blueprint_transforms_for_family(
        family=str(candidate_family or ""),
        parameter_grid=parameter_grid,
    )
    return {
        "blueprint_id": blueprint_id,
        "source_candidate_id": source_candidate_id,
        "candidate_family": candidate_family,
        "executor_id": executor_id,
        "baseline": baseline,
        "transforms": transforms,
        "training_plan": {
            "mode": "finetune"
            if candidate_family == "official_checkpoint_calibration"
            else "none"
        },
        "parameter_grid": parameter_grid,
        "metrics": _blueprint_metrics_from_sandbox(sandbox),
        "safety_policy": {
            "preserve_context_steps": 10,
            "forbid_target_leakage": True,
            "forbid_external_code": True,
        },
        "expected_artifacts": _blueprint_expected_artifacts(str(candidate_family or "")),
        "source_blueprint_shape": "hypothesis_lab_legacy",
    }


def _materialize_executable_blueprints_from_candidates(
    *,
    candidates: dict[str, Any],
    output_path: Path,
    force: bool,
) -> dict[str, Any]:
    _guard_output(output_path, force=force)
    blueprints = [
        _executable_blueprint_from_candidate(candidate)
        for candidate in _object_list(
            candidates.get("algorithm_candidates"),
            "algorithm_candidates",
        )
    ]
    report = {
        "schema_version": 1,
        "state_hint": "EXECUTABLE_BLUEPRINTS_DESIGNED",
        "executable_blueprints_path": str(output_path),
        "blueprint_count": len(blueprints),
        "blueprints": blueprints,
        "next_command": "validate-executable-blueprints",
    }
    _write_json(output_path, report)
    return report


def _executable_blueprint_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = _nonempty_text(
        candidate.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    family = str(candidate.get("candidate_family") or "unknown_family")
    executor_id = _executor_id_for_candidate(candidate)
    parameter_grid = (
        candidate.get("parameter_grid")
        if isinstance(candidate.get("parameter_grid"), dict)
        and candidate.get("parameter_grid")
        else {"enabled": [True]}
    )
    return {
        "blueprint_id": f"bp_{_slug(candidate_id)}_001",
        "source_candidate_id": candidate_id,
        "candidate_family": family,
        "executor_id": executor_id,
        "baseline": _blueprint_baseline_for_family(family),
        "transforms": _blueprint_transforms_for_family(
            family=_blueprint_transform_family_alias(family),
            parameter_grid=parameter_grid,
        ),
        "training_plan": {
            "mode": "finetune" if family == "official_checkpoint_family" else "none"
        },
        "parameter_grid": parameter_grid,
        "metrics": ["score", "runtime", "burgers_residual"],
        "safety_policy": {
            "preserve_context_steps": 10,
            "forbid_target_leakage": True,
            "forbid_external_code": True,
        },
        "expected_artifacts": _blueprint_expected_artifacts(family),
    }


def _executor_id_for_candidate(candidate: dict[str, Any]) -> str:
    family = str(candidate.get("candidate_family") or "")
    if family == "official_checkpoint_family":
        return "official_checkpoint_candidate_pipeline"
    if family == "lightweight_residual_corrector":
        return "controlled_numpy_residual_operator"
    if family in {"task1_retrieval_prior"}:
        return "controlled_numpy_retrieval_prior"
    if family in {"task1_retrieval_refinement"}:
        return "controlled_numpy_retrieval_refinement"
    if family in {"task1_early_refinement"}:
        return "controlled_numpy_task1_early_refinement"
    return "controlled_numpy_operator"


def _blueprint_transform_family_alias(family: str) -> str:
    if family == "lightweight_residual_corrector":
        return "learned_residual_correction"
    if family == "official_checkpoint_family":
        return "official_checkpoint_calibration"
    if family in {
        "spectral_rollout_stabilization",
        "operator_correction_postprocess",
        "physics_residual_regularization",
        "scheduled_rollout_training",
    }:
        return "linear_trend_spectral"
    return family


def _first_candidate_id(candidates: dict[str, Any]) -> str:
    items = _object_list(candidates.get("algorithm_candidates"), "algorithm_candidates")
    if not items:
        raise LiteratureAlgorithmLabError("algorithm candidates must be non-empty")
    return _nonempty_text(
        items[0].get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )


def _has_native_executable_blueprint_shape(blueprint: dict[str, Any]) -> bool:
    return all(
        key in blueprint
        for key in [
            "source_candidate_id",
            "executor_id",
            "baseline",
            "transforms",
            "training_plan",
            "metrics",
            "safety_policy",
            "expected_artifacts",
        ]
    )


def _blueprint_baseline_for_family(family: str) -> str:
    if family in {
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
        "task1_early_refinement",
    }:
        return "retrieval_prior"
    if family == "official_checkpoint_calibration":
        return "fno_checkpoint"
    return "persistence"


def _blueprint_transforms_for_family(
    *, family: str, parameter_grid: dict[str, Any]
) -> list[dict[str, Any]]:
    if family == "learned_residual_correction":
        return [{"name": "residual_fit", "params": parameter_grid}]
    if family == "task1_retrieval_prior":
        return [{"name": "retrieval_prior", "params": parameter_grid}]
    if family in {"task1_retrieval_refinement", "task1_early_refinement"}:
        return [{"name": "retrieval_refinement", "params": parameter_grid}]
    if family == "official_checkpoint_calibration":
        return [{"name": "checkpoint_calibration", "params": parameter_grid}]
    transforms = [{"name": "linear_trend", "params": parameter_grid}]
    if "spectral_strength" in parameter_grid:
        transforms.append({"name": "spectral_smooth", "params": parameter_grid})
    return transforms


def _blueprint_metrics_from_sandbox(sandbox: dict[str, Any]) -> list[str]:
    metrics = ["score", "runtime"]
    primary = sandbox.get("primary_metric")
    if isinstance(primary, str) and primary.strip() and primary not in metrics:
        metrics.append(primary)
    return metrics


def _blueprint_expected_artifacts(family: str) -> list[str]:
    artifacts = ["sandbox_evidence"]
    if family == "official_checkpoint_calibration":
        artifacts.append("checkpoint_sandbox_report")
    else:
        artifacts.append("prediction_hdf5")
    return artifacts


def _validate_executable_blueprint(
    blueprint: dict[str, Any],
    *,
    index: int,
    issues: list[dict[str, Any]],
) -> None:
    base = f"blueprints[{index}]"
    for field in [
        "blueprint_id",
        "source_candidate_id",
        "candidate_family",
        "executor_id",
        "baseline",
    ]:
        if not _is_nonempty_string(blueprint.get(field)):
            _append_blueprint_issue(
                issues,
                code=f"missing_{field}",
                path=f"{base}.{field}",
                failure_code="blueprint_schema_invalid",
                message=f"{field} must be a non-empty string",
            )
    baseline = blueprint.get("baseline")
    if isinstance(baseline, str) and baseline not in EXECUTABLE_BLUEPRINT_BASELINES:
        _append_blueprint_issue(
            issues,
            code="invalid_baseline",
            path=f"{base}.baseline",
            failure_code="blueprint_schema_invalid",
            message=f"baseline must be one of {sorted(EXECUTABLE_BLUEPRINT_BASELINES)}",
        )
    executor_id = blueprint.get("executor_id")
    if isinstance(executor_id, str) and executor_id not in EXECUTABLE_BLUEPRINT_EXECUTORS:
        _append_blueprint_issue(
            issues,
            code="invalid_executor",
            path=f"{base}.executor_id",
            failure_code="unsupported_executor",
            message=f"executor_id is not approved: {executor_id}",
        )
    _validate_blueprint_transforms(blueprint.get("transforms"), base=base, issues=issues)
    _validate_blueprint_training_plan(
        blueprint.get("training_plan"), base=base, issues=issues
    )
    if not isinstance(blueprint.get("parameter_grid"), dict) or not blueprint.get(
        "parameter_grid"
    ):
        _append_blueprint_issue(
            issues,
            code="empty_parameter_grid",
            path=f"{base}.parameter_grid",
            failure_code="blueprint_schema_invalid",
            message="parameter_grid must be a non-empty object",
        )
    for field in ["metrics", "expected_artifacts"]:
        value = blueprint.get(field)
        if not isinstance(value, list) or not value:
            _append_blueprint_issue(
                issues,
                code=f"empty_{field}",
                path=f"{base}.{field}",
                failure_code="blueprint_schema_invalid",
                message=f"{field} must be a non-empty list",
            )
        elif not all(_is_nonempty_string(item) for item in value):
            _append_blueprint_issue(
                issues,
                code=f"invalid_{field}",
                path=f"{base}.{field}",
                failure_code="blueprint_schema_invalid",
                message=f"{field} items must be non-empty strings",
            )
    _validate_blueprint_safety_policy(
        blueprint.get("safety_policy"), base=base, issues=issues
    )


def _validate_blueprint_transforms(
    transforms: Any,
    *,
    base: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(transforms, list) or not transforms:
        _append_blueprint_issue(
            issues,
            code="empty_transforms",
            path=f"{base}.transforms",
            failure_code="blueprint_schema_invalid",
            message="transforms must be a non-empty list",
        )
        return
    for item_index, transform in enumerate(transforms):
        path = f"{base}.transforms[{item_index}]"
        if not isinstance(transform, dict):
            _append_blueprint_issue(
                issues,
                code="invalid_transform",
                path=path,
                failure_code="blueprint_schema_invalid",
                message="transform must be an object",
            )
            continue
        name = transform.get("name")
        if not isinstance(name, str) or name not in EXECUTABLE_BLUEPRINT_TRANSFORMS:
            _append_blueprint_issue(
                issues,
                code="invalid_transform_name",
                path=f"{path}.name",
                failure_code="blueprint_schema_invalid",
                message=f"transform name is not approved: {name}",
            )
        if not isinstance(transform.get("params"), dict):
            _append_blueprint_issue(
                issues,
                code="invalid_transform_params",
                path=f"{path}.params",
                failure_code="blueprint_schema_invalid",
                message="transform params must be an object",
            )


def _validate_blueprint_training_plan(
    training_plan: Any,
    *,
    base: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(training_plan, dict):
        _append_blueprint_issue(
            issues,
            code="invalid_training_plan",
            path=f"{base}.training_plan",
            failure_code="blueprint_schema_invalid",
            message="training_plan must be an object",
        )
        return
    mode = training_plan.get("mode")
    if not isinstance(mode, str) or mode not in EXECUTABLE_BLUEPRINT_TRAINING_MODES:
        _append_blueprint_issue(
            issues,
            code="invalid_training_mode",
            path=f"{base}.training_plan.mode",
            failure_code="blueprint_schema_invalid",
            message=f"training mode must be one of {sorted(EXECUTABLE_BLUEPRINT_TRAINING_MODES)}",
        )


def _validate_blueprint_safety_policy(
    safety_policy: Any,
    *,
    base: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(safety_policy, dict):
        _append_blueprint_issue(
            issues,
            code="invalid_safety_policy",
            path=f"{base}.safety_policy",
            failure_code="blueprint_schema_invalid",
            message="safety_policy must be an object",
        )
        return
    preserve_steps = safety_policy.get("preserve_context_steps")
    if not isinstance(preserve_steps, int) or preserve_steps < 10:
        _append_blueprint_issue(
            issues,
            code="unsafe_context_preservation",
            path=f"{base}.safety_policy.preserve_context_steps",
            failure_code="context_mismatch",
            message="preserve_context_steps must be at least 10",
        )
    if safety_policy.get("forbid_target_leakage") is not True:
        _append_blueprint_issue(
            issues,
            code="unsafe_target_leakage",
            path=f"{base}.safety_policy.forbid_target_leakage",
            failure_code="target_leakage_risk",
            message="forbid_target_leakage must be true",
        )
    if safety_policy.get("forbid_external_code") is not True:
        _append_blueprint_issue(
            issues,
            code="unsafe_external_code",
            path=f"{base}.safety_policy.forbid_external_code",
            failure_code="requires_external_code",
            message="forbid_external_code must be true",
        )


def _append_blueprint_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    path: str,
    failure_code: str,
    message: str,
) -> None:
    issues.append(
        {
            "code": code,
            "path": path,
            "failure_code": failure_code,
            "message": message,
        }
    )

def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError(f"{label} items must be objects")
        result.append(item)
    return result


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureAlgorithmLabError(f"{label} must be a non-empty string")
    return value.strip()


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "source"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiteratureAlgorithmLabError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise LiteratureAlgorithmLabError(f"output file already exists: {path}")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise LiteratureAlgorithmLabError(f"{label} not found: {path}")
    return path
