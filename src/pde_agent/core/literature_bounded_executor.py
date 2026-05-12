"""Bounded executor authoring, audit, sandbox, and evidence recording."""

import ast
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError
from pde_agent.core.literature_candidate_promotion import (
    evaluate_algorithm_candidate_promotion,
)
from pde_agent.core.literature_registry import (
    record_literature_registry_event,
    update_literature_registry,
)

LITERATURE_REGISTRY_EVENT_STATE_HINT = "LITERATURE_REGISTRY_EVENT_RECORDED"
EXECUTOR_DESIGN_REQUEST_STATE_HINT = "EXECUTOR_DESIGN_REQUEST_CREATED"
BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT = "BOUNDED_EXECUTOR_DESIGN_SPEC_CREATED"
BOUNDED_EXECUTOR_TEMPLATE_STATE_HINT = "BOUNDED_EXECUTOR_TEMPLATE_MATERIALIZED"
BOUNDED_EXECUTOR_AUDIT_STATE_HINT = "BOUNDED_EXECUTOR_IMPLEMENTATION_VALIDATED"
BOUNDED_EXECUTOR_SANDBOX_STATE_HINT = "BOUNDED_EXECUTOR_SANDBOX_RAN"
BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT = "BOUNDED_EXECUTOR_AUTHORING_TASK_PREPARED"
BOUNDED_EXECUTOR_AUTHORING_AUDIT_STATE_HINT = "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED"
BOUNDED_EXECUTOR_AUTHORING_COMPLETION_STATE_HINT = "BOUNDED_EXECUTOR_AUTHORING_COMPLETED"

def create_bounded_executor_design_spec(
    *,
    executor_design_request_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    request = _read_json(
        _require_file(Path(executor_design_request_path), "executor design request")
    )
    if request.get("state_hint") != EXECUTOR_DESIGN_REQUEST_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "executor design request has invalid state_hint"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    candidate_id = _nonempty_text(
        request.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    requested_executor_id = _nonempty_text(
        request.get("requested_executor_id"),
        "requested_executor_id",
    )
    report = {
        "schema_version": 1,
        "state_hint": BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT,
        "bounded_executor_design_spec_path": str(output_file),
        "executor_design_request_path": str(Path(executor_design_request_path)),
        "algorithm_candidate_id": candidate_id,
        "candidate_family": request.get("candidate_family"),
        "requested_executor_id": requested_executor_id,
        "hypothesis": request.get("hypothesis"),
        "mechanism": request.get("mechanism"),
        "source_grounding": request.get("source_grounding")
        if isinstance(request.get("source_grounding"), dict)
        else {},
        "executor_api": {
            "function_name": "predict",
            "required_args": ["context", "candidate", "params"],
            "context_shape": ["N", 10, "X"],
            "prediction_shape": ["N", 200, "X"],
            "returns": "numpy.ndarray with first 10 steps exactly preserved",
        },
        "security_policy": {
            "allowed_imports": ["numpy", "math", "typing"],
            "forbidden_calls": [
                "open",
                "eval",
                "exec",
                "compile",
                "__import__",
                "input",
                "getattr",
            ],
            "forbidden_numpy_calls": [
                "load",
                "save",
                "savez",
                "savez_compressed",
                "fromfile",
                "memmap",
            ],
            "forbidden_actions": [
                "read or write arbitrary files",
                "perform network access",
                "execute shell commands",
                "mutate input arrays in place",
                "use target labels inside predict",
            ],
        },
        "sandbox_contract": {
            "required_inputs": [
                "design_spec_path",
                "manifest_path",
                "executor_path",
                "context_path",
                "target_path",
                "output_dir",
            ],
            "metrics": [
                "segment_score",
                "task1_prediction_score",
                "score_delta_vs_persistence",
                "inference_seconds",
                "context_preserved",
            ],
            "promotion_event_command": "record-bounded-executor-sandbox-event",
        },
        "minimal_experiment": request.get("minimal_experiment", {}),
        "risk_controls": request.get("risk_controls", []),
        "next_command": "materialize-bounded-executor-template",
    }
    _write_json(output_file, report)
    return report


def materialize_bounded_executor_template(
    *,
    design_spec_path: str | Path,
    output_dir: str | Path,
    force: bool,
) -> dict[str, Any]:
    design_file = _require_file(Path(design_spec_path), "bounded executor design spec")
    design = _read_json(design_file)
    if design.get("state_hint") != BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "bounded executor design spec has invalid state_hint"
        )
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    executor_path = root / "executor.py"
    report_path = root / "bounded_executor_template.json"
    for path in [manifest_path, executor_path, report_path]:
        _guard_output(path, force=force)
    root.mkdir(parents=True, exist_ok=True)
    candidate_id = _nonempty_text(
        design.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    manifest = {
        "schema_version": 1,
        "algorithm_candidate_id": candidate_id,
        "requested_executor_id": design.get("requested_executor_id"),
        "design_spec_path": str(design_file),
        "implementation_status": "scaffold_baseline",
        "entrypoint": "executor.py:predict",
    }
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    executor_path.write_text(_bounded_executor_template_source(), encoding="utf-8")
    report = {
        "schema_version": 1,
        "state_hint": BOUNDED_EXECUTOR_TEMPLATE_STATE_HINT,
        "bounded_executor_template_path": str(report_path),
        "design_spec_path": str(design_file),
        "manifest_path": str(manifest_path),
        "executor_path": str(executor_path),
        "algorithm_candidate_id": candidate_id,
        "implementation_status": "scaffold_baseline",
        "next_command": "validate-bounded-executor-implementation",
    }
    _write_json(report_path, report)
    return report


def validate_bounded_executor_implementation(
    *,
    design_spec_path: str | Path,
    manifest_path: str | Path,
    executor_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    design = _read_json(
        _require_file(Path(design_spec_path), "bounded executor design spec")
    )
    if design.get("state_hint") != BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "bounded executor design spec has invalid state_hint"
        )
    manifest_file = _require_file(Path(manifest_path), "bounded executor manifest")
    manifest = _read_json(manifest_file)
    source_file = _require_file(Path(executor_path), "bounded executor source")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    issues = _audit_bounded_executor_source(
        source_file.read_text(encoding="utf-8"),
        design=design,
    )
    decision = "approved" if not issues else "rejected"
    report = {
        "schema_version": 1,
        "state_hint": BOUNDED_EXECUTOR_AUDIT_STATE_HINT,
        "bounded_executor_audit_path": str(output_file),
        "design_spec_path": str(Path(design_spec_path)),
        "manifest_path": str(manifest_file),
        "executor_path": str(source_file),
        "algorithm_candidate_id": manifest.get("algorithm_candidate_id")
        or design.get("algorithm_candidate_id"),
        "decision": decision,
        "issue_count": len(issues),
        "issues": issues,
        "next_command": "run-bounded-executor-sandbox"
        if decision == "approved"
        else "revise-bounded-executor",
    }
    _write_json(output_file, report)
    return report


def run_bounded_executor_sandbox(
    *,
    design_spec_path: str | Path,
    manifest_path: str | Path,
    executor_path: str | Path,
    context_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    submission_id: str,
    force: bool,
) -> dict[str, Any]:
    from pde_agent.core.burgers_hdf5_io import load_hdf5_tensor
    from pde_agent.core.manual_operator_tuning import (
        make_persistence_prediction,
        score_burgers_segments,
    )

    design = _read_json(
        _require_file(Path(design_spec_path), "bounded executor design spec")
    )
    if design.get("state_hint") != BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "bounded executor design spec has invalid state_hint"
        )
    manifest = _read_json(_require_file(Path(manifest_path), "bounded executor manifest"))
    source_file = _require_file(Path(executor_path), "bounded executor source")
    audit_issues = _audit_bounded_executor_source(
        source_file.read_text(encoding="utf-8"),
        design=design,
    )
    if audit_issues:
        raise LiteratureAlgorithmLabError("bounded executor failed static audit")
    root = Path(output_dir)
    report_path = root / "bounded_executor_sandbox_report.json"
    prediction_path = root / "prediction.npz"
    _guard_output(report_path, force=force)
    _guard_output(prediction_path, force=force)
    root.mkdir(parents=True, exist_ok=True)
    context = load_hdf5_tensor(context_path)
    if context.shape[1] != 10:
        context = context[:, :10, :]
    target = load_hdf5_tensor(target_path)
    candidate = {
        "algorithm_candidate_id": design.get("algorithm_candidate_id"),
        "requested_executor_id": design.get("requested_executor_id"),
        "hypothesis": design.get("hypothesis"),
        "mechanism": design.get("mechanism"),
        "source_grounding": design.get("source_grounding", {}),
    }
    params = {"horizon": int(target.shape[1])}
    module = _load_bounded_executor_module(source_file)
    start = time.perf_counter()
    prediction = module.predict(context.copy(), candidate, params)
    inference_seconds = time.perf_counter() - start
    implementation_status = getattr(
        module,
        "IMPLEMENTATION_STATUS",
        manifest.get("implementation_status"),
    )
    if not isinstance(implementation_status, str) or not implementation_status.strip():
        implementation_status = manifest.get("implementation_status")
    np = __import__("numpy")
    prediction = np.asarray(prediction)
    if prediction.shape != target.shape:
        raise LiteratureAlgorithmLabError(
            f"bounded executor prediction shape {prediction.shape} != target {target.shape}"
        )
    context_preserved = bool(
        np.allclose(prediction[:, : context.shape[1], :], context, atol=1e-6)
    )
    score = score_burgers_segments(prediction=prediction, target=target)
    persistence = make_persistence_prediction(context)
    baseline_score = score_burgers_segments(prediction=persistence, target=target)
    score_delta = float(score["segment_score"]) - float(baseline_score["segment_score"])
    np.savez_compressed(prediction_path, prediction=prediction)
    rules_passed = context_preserved and prediction.shape == target.shape
    report = {
        "schema_version": 1,
        "state_hint": BOUNDED_EXECUTOR_SANDBOX_STATE_HINT,
        "bounded_executor_sandbox_path": str(report_path),
        "submission_id": _nonempty_text(submission_id, "submission_id"),
        "design_spec_path": str(Path(design_spec_path)),
        "manifest_path": str(Path(manifest_path)),
        "executor_path": str(source_file),
        "algorithm_candidate_id": candidate["algorithm_candidate_id"],
        "requested_executor_id": candidate["requested_executor_id"],
        "source_grounding": candidate["source_grounding"],
        "implementation_status": implementation_status,
        "rules_passed": rules_passed,
        "context_preserved": context_preserved,
        "metric_results": {
            "segment_score": score["segment_score"],
            "task1_prediction_score": score["task1_prediction_score"],
            "baseline_segment_score": baseline_score["segment_score"],
            "baseline_task1_prediction_score": baseline_score[
                "task1_prediction_score"
            ],
            "score_delta_vs_persistence": score_delta,
            "inference_seconds": inference_seconds,
        },
        "segments": score["segments"],
        "artifacts": {
            "prediction_path": str(prediction_path),
            "sandbox_report_path": str(report_path),
        },
        "next_command": "record-bounded-executor-sandbox-event"
        if rules_passed
        else "revise-bounded-executor",
    }
    _write_json(report_path, report)
    return report


def record_bounded_executor_sandbox_event(
    *,
    sandbox_report_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    sandbox_file = _require_file(Path(sandbox_report_path), "bounded executor sandbox")
    sandbox = _read_json(sandbox_file)
    if sandbox.get("state_hint") != BOUNDED_EXECUTOR_SANDBOX_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "bounded executor sandbox has invalid state_hint"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    rules_passed = bool(sandbox.get("rules_passed"))
    grounding = (
        sandbox.get("source_grounding")
        if isinstance(sandbox.get("source_grounding"), dict)
        else {}
    )
    source_ids = [item for item in grounding.get("source_ids", []) if isinstance(item, str)]
    report = {
        "schema_version": 1,
        "state_hint": LITERATURE_REGISTRY_EVENT_STATE_HINT,
        "literature_registry_event_path": str(output_file),
        "event_source_path": str(sandbox_file),
        "event_type": "candidate_sandbox_verified"
        if rules_passed
        else "candidate_sandbox_failed",
        "algorithm_candidate_id": sandbox.get("algorithm_candidate_id"),
        "candidate_family": "bounded_executor",
        "evidence_id": sandbox.get("bounded_executor_sandbox_path"),
        "decision": "sandbox_verified" if rules_passed else "sandbox_failed",
        "failure_codes": [] if rules_passed else ["bounded_executor_sandbox_failed"],
        "source_ids": source_ids,
        "algorithm_card_id": grounding.get("algorithm_card_id"),
        "recommended_next": "evaluate-algorithm-candidate-promotion"
        if rules_passed
        else "revise-bounded-executor",
        "retry_allowed": not rules_passed,
        "metric_results": sandbox.get("metric_results", {}),
        "next_command": "update-literature-registry",
    }
    _write_json(output_file, report)
    return report


def record_bounded_executor_sandbox_evidence(
    *,
    sandbox_report_path: str | Path,
    output_path: str | Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    sandbox_file = _require_file(Path(sandbox_report_path), "bounded executor sandbox")
    sandbox = _read_json(sandbox_file)
    if sandbox.get("state_hint") != BOUNDED_EXECUTOR_SANDBOX_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "bounded executor sandbox has invalid state_hint"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    metrics = sandbox.get("metric_results")
    if not isinstance(metrics, dict):
        metrics = {}
    score = _float_or_default(metrics.get("segment_score"), 0.0)
    grounding = (
        sandbox.get("source_grounding")
        if isinstance(sandbox.get("source_grounding"), dict)
        else {}
    )
    source_ids = [item for item in grounding.get("source_ids", []) if isinstance(item, str)]
    evidence = {
        "schema_version": 1,
        "state_hint": "SANDBOX_EVIDENCE_RECORDED",
        "evidence_path": str(output_file),
        "evidence_id": f"evidence_{_slug(str(sandbox.get('algorithm_candidate_id') or 'bounded_executor'))}",
        "candidate_id": sandbox.get("algorithm_candidate_id"),
        "candidate_family": "bounded_executor",
        "source_report_path": str(sandbox_file),
        "source_ids": source_ids,
        "algorithm_card_id": grounding.get("algorithm_card_id"),
        "rules_passed": bool(sandbox.get("rules_passed")),
        "score": score,
        "segment_scores": _bounded_executor_segment_scores(sandbox.get("segments")),
        "runtime": {
            "inference_seconds": _float_or_default(
                metrics.get("inference_seconds"),
                0.0,
            )
        },
        "physics_metrics": {"residual_delta": 0.0},
        "ablation_metrics": {
            "score_delta": _float_or_default(
                metrics.get("score_delta_vs_persistence"),
                0.0,
            )
        },
        "score_delta_vs_current_best": score - float(current_best_score),
        "artifact_paths": [
            path
            for path in [
                sandbox.get("bounded_executor_sandbox_path"),
                _object_path(sandbox.get("artifacts"), "prediction_path"),
            ]
            if path
        ],
        "reproducibility": {"deterministic": True, "repeat_count": 1},
        "next_command": "evaluate-algorithm-candidate-promotion",
    }
    _write_json(output_file, evidence)
    return evidence


def complete_bounded_executor_authoring_task(
    *,
    authoring_task_path: str | Path,
    algorithm_candidates_path: str | Path,
    registry_path: str | Path,
    current_best_score: float,
    skill_root: str | Path,
    force: bool,
    require_authoring_audit: bool = True,
) -> dict[str, Any]:
    authoring_file = _require_file(
        Path(authoring_task_path),
        "bounded executor authoring task",
    )
    authoring = _read_json(authoring_file)
    if authoring.get("state_hint") != BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "bounded executor authoring task has invalid state_hint"
        )
    root = authoring_file.parent
    completion_path = root / "bounded_executor_authoring_completion.json"
    _guard_output(completion_path, force=force)
    authoring_audit = audit_bounded_executor_authoring_task(
        authoring_task_path=authoring_file,
        skill_root=Path(skill_root),
        output_path=root / "bounded_executor_authoring_audit.json",
        force=force,
    )
    if require_authoring_audit and authoring_audit.get("decision") != "approved":
        report = {
            "schema_version": 1,
            "state_hint": BOUNDED_EXECUTOR_AUTHORING_COMPLETION_STATE_HINT,
            "completion_path": str(completion_path),
            "authoring_task_path": str(authoring_file),
            "algorithm_candidate_id": authoring.get("algorithm_candidate_id"),
            "decision": "blocked",
            "audit": authoring_audit,
            "artifacts": {
                "completion_path": str(completion_path),
                "authoring_audit_path": authoring_audit.get("authoring_audit_path"),
            },
            "next_command": "audit-bounded-executor-authoring-task",
        }
        _write_json(completion_path, report)
        return report

    context_path = _authoring_context_path(authoring)
    target_path = _authoring_target_path(authoring)
    expected = (
        authoring.get("expected_artifacts")
        if isinstance(authoring.get("expected_artifacts"), dict)
        else {}
    )
    audit_path = Path(
        expected.get("audit_output_path") or root / "bounded_executor_audit.json"
    )
    sandbox_dir = Path(
        expected.get("sandbox_report_path", root / "sandbox" / "x.json")
    ).parent
    sandbox_event_path = Path(
        expected.get("event_output_path")
        or root / "bounded_executor_registry_event.json"
    )
    evidence_path = root / "bounded_executor_sandbox_evidence.json"
    promotion_path = root / "bounded_executor_candidate_promotion.json"
    promotion_event_path = root / "bounded_executor_promotion_registry_event.json"

    implementation_audit = validate_bounded_executor_implementation(
        design_spec_path=authoring["design_spec_path"],
        manifest_path=authoring["manifest_path"],
        executor_path=authoring["executor_path"],
        output_path=audit_path,
        force=force,
    )
    sandbox = run_bounded_executor_sandbox(
        design_spec_path=authoring["design_spec_path"],
        manifest_path=authoring["manifest_path"],
        executor_path=authoring["executor_path"],
        context_path=context_path,
        target_path=target_path,
        output_dir=sandbox_dir,
        submission_id="team_default_bounded_executor_completion",
        force=force,
    )
    sandbox_event = record_bounded_executor_sandbox_event(
        sandbox_report_path=sandbox["bounded_executor_sandbox_path"],
        output_path=sandbox_event_path,
        force=force,
    )
    evidence = record_bounded_executor_sandbox_evidence(
        sandbox_report_path=sandbox["bounded_executor_sandbox_path"],
        output_path=evidence_path,
        current_best_score=current_best_score,
        force=force,
    )
    promotion = evaluate_algorithm_candidate_promotion(
        algorithm_candidates_path=algorithm_candidates_path,
        algorithm_candidate_id=_nonempty_text(
            authoring.get("algorithm_candidate_id"),
            "algorithm_candidate_id",
        ),
        sandbox_evidence_paths=[evidence["evidence_path"]],
        output_path=promotion_path,
        current_best_score=current_best_score,
        force=force,
    )
    promotion_event = record_literature_registry_event(
        event_source_path=promotion["algorithm_candidate_promotion_path"],
        output_path=promotion_event_path,
        force=force,
    )
    registry = update_literature_registry(
        registry_path=registry_path,
        event_paths=[
            sandbox_event["literature_registry_event_path"],
            promotion_event["literature_registry_event_path"],
        ],
        force=not Path(registry_path).exists(),
    )
    report = {
        "schema_version": 1,
        "state_hint": BOUNDED_EXECUTOR_AUTHORING_COMPLETION_STATE_HINT,
        "completion_path": str(completion_path),
        "authoring_task_path": str(authoring_file),
        "decision": promotion["decision"],
        "audit": authoring_audit,
        "implementation_audit": implementation_audit,
        "sandbox": sandbox,
        "sandbox_event": sandbox_event,
        "evidence": evidence,
        "promotion": promotion,
        "promotion_event": promotion_event,
        "registry": registry,
        "artifacts": {
            "completion_path": str(completion_path),
            "authoring_audit_path": authoring_audit.get("authoring_audit_path"),
            "bounded_executor_audit_path": implementation_audit[
                "bounded_executor_audit_path"
            ],
            "sandbox_report_path": sandbox["bounded_executor_sandbox_path"],
            "sandbox_registry_event_path": sandbox_event[
                "literature_registry_event_path"
            ],
            "sandbox_evidence_path": evidence["evidence_path"],
            "candidate_promotion_path": promotion[
                "algorithm_candidate_promotion_path"
            ],
            "promotion_registry_event_path": promotion_event[
                "literature_registry_event_path"
            ],
            "literature_registry_path": registry["registry_path"],
        },
        "next_command": promotion_event["recommended_next"],
    }
    _write_json(completion_path, report)
    return report


def _authoring_context_path(authoring: dict[str, Any]) -> Path:
    design = str(authoring.get("design_spec_path"))
    manifest = str(authoring.get("manifest_path"))
    executor = str(authoring.get("executor_path"))
    for item in _text_list(authoring.get("allowed_read_paths"), "allowed_read_paths"):
        if item not in {design, manifest, executor}:
            return Path(item)
    raise LiteratureAlgorithmLabError("authoring task missing context read path")


def _authoring_target_path(authoring: dict[str, Any]) -> Path:
    forbidden = _text_list(
        authoring.get("forbidden_read_paths"),
        "forbidden_read_paths",
    )
    if not forbidden:
        raise LiteratureAlgorithmLabError("authoring task missing target path")
    return Path(forbidden[0])


def _bounded_executor_segment_scores(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if isinstance(item, dict) and isinstance(item.get("score"), int | float):
            result[str(key)] = float(item["score"])
    return result


def _object_path(value: Any, key: str) -> str | None:
    if not isinstance(value, dict):
        return None
    item = value.get(key)
    if isinstance(item, str) and item.strip():
        return item
    return None


def prepare_bounded_executor_authoring_task(
    *,
    design_spec_path: str | Path,
    manifest_path: str | Path,
    executor_path: str | Path,
    context_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    sandbox_output_dir: str | Path,
    audit_output_path: str | Path,
    event_output_path: str | Path,
    submission_id: str,
    force: bool,
) -> dict[str, Any]:
    design_file = _require_file(Path(design_spec_path), "bounded executor design spec")
    design = _read_json(design_file)
    if design.get("state_hint") != BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "bounded executor design spec has invalid state_hint"
        )
    manifest_file = _require_file(Path(manifest_path), "bounded executor manifest")
    executor_file = _require_file(Path(executor_path), "bounded executor source")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    candidate_id = _nonempty_text(
        design.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    audit_path = Path(audit_output_path)
    sandbox_dir = Path(sandbox_output_dir)
    event_path = Path(event_output_path)
    validation_command = (
        "uv run --frozen pde-agent validate-bounded-executor-implementation "
        f"--design-spec-path {design_file} "
        f"--manifest-path {manifest_file} "
        f"--executor-path {executor_file} "
        f"--output-path {audit_path} --force"
    )
    sandbox_command = (
        "uv run --frozen pde-agent run-bounded-executor-sandbox "
        f"--design-spec-path {design_file} "
        f"--manifest-path {manifest_file} "
        f"--executor-path {executor_file} "
        f"--context-path {Path(context_path)} "
        f"--target-path {Path(target_path)} "
        f"--output-dir {sandbox_dir} "
        f"--submission-id {_shell_safe_token(submission_id)} --force"
    )
    event_command = (
        "uv run --frozen pde-agent record-bounded-executor-sandbox-event "
        f"--sandbox-report-path {sandbox_dir / 'bounded_executor_sandbox_report.json'} "
        f"--output-path {event_path} --force"
    )
    report = {
        "schema_version": 1,
        "state_hint": BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT,
        "bounded_executor_authoring_task_path": str(output_file),
        "algorithm_candidate_id": candidate_id,
        "requested_executor_id": design.get("requested_executor_id"),
        "design_spec_path": str(design_file),
        "manifest_path": str(manifest_file),
        "executor_path": str(executor_file),
        "allowed_edit_paths": [str(executor_file)],
        "allowed_read_paths": [
            str(design_file),
            str(manifest_file),
            str(executor_file),
            str(Path(context_path)),
        ],
        "forbidden_read_paths": [str(Path(target_path))],
        "agent_contract": {
            "skill_name": "pde-bounded-executor-author",
            "task": "Edit only executor_path and implement predict(context, candidate, params).",
            "success_criteria": [
                "static audit decision approved",
                "sandbox rules_passed true",
                "first 10 context steps preserved exactly",
                "metric_results and registry event written",
            ],
            "forbidden_actions": [
                "read target_path manually",
                "edit any file outside allowed_edit_paths",
                "import modules outside security_policy.allowed_imports",
                "use file, network, shell, eval, exec, or dynamic import",
            ],
            "implementation_guidance": [
                "Use design_spec hypothesis and mechanism to guide the rollout rule.",
                "Keep the implementation small and reversible.",
                "If the mechanism is unclear, implement a conservative baseline and record the limitation through the sandbox result.",
            ],
        },
        "validation_commands": [
            validation_command,
            sandbox_command,
            event_command,
        ],
        "expected_artifacts": {
            "audit_output_path": str(audit_path),
            "sandbox_report_path": str(sandbox_dir / "bounded_executor_sandbox_report.json"),
            "event_output_path": str(event_path),
        },
        "next_command": "audit-bounded-executor-authoring-task",
    }
    _write_json(output_file, report)
    return report


def audit_bounded_executor_authoring_task(
    *,
    authoring_task_path: str | Path,
    skill_root: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    authoring_file = _require_file(Path(authoring_task_path), "bounded executor authoring task")
    authoring = _read_json(authoring_file)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    issues: list[dict[str, str]] = []

    if authoring.get("state_hint") != BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT:
        issues.append(
            {
                "code": "invalid_state_hint",
                "message": "authoring task must be BOUNDED_EXECUTOR_AUTHORING_TASK_PREPARED",
            }
        )

    agent_contract = authoring.get("agent_contract")
    if not isinstance(agent_contract, dict):
        agent_contract = {}
        issues.append(
            {
                "code": "agent_contract_missing",
                "message": "authoring task must include agent_contract",
            }
        )
    skill_name = agent_contract.get("skill_name")
    if not isinstance(skill_name, str) or not skill_name.strip():
        skill_name = ""
        issues.append(
            {
                "code": "skill_name_missing",
                "message": "agent_contract.skill_name must be a non-empty string",
            }
        )
    skill_path = _bounded_executor_authoring_skill_path(
        skill_root=Path(skill_root),
        skill_name=skill_name,
    )
    if skill_path is None:
        issues.append(
            {
                "code": "skill_file_missing",
                "message": f"skill file not found for {skill_name or '<missing>'}",
            }
        )
    else:
        _audit_bounded_executor_skill_text(
            skill_path=skill_path,
            skill_name=skill_name,
            issues=issues,
        )

    executor_path = _artifact_path_or_issue(authoring, "executor_path", issues)
    design_path = _artifact_path_or_issue(authoring, "design_spec_path", issues)
    manifest_path = _artifact_path_or_issue(authoring, "manifest_path", issues)
    allowed_edit_paths = _path_set_or_issue(
        authoring.get("allowed_edit_paths"),
        "allowed_edit_paths",
        issues,
    )
    allowed_read_paths = _path_set_or_issue(
        authoring.get("allowed_read_paths"),
        "allowed_read_paths",
        issues,
    )
    forbidden_read_paths = _path_set_or_issue(
        authoring.get("forbidden_read_paths"),
        "forbidden_read_paths",
        issues,
    )
    expected_edit_paths = {str(executor_path)} if executor_path is not None else set()
    if allowed_edit_paths != expected_edit_paths:
        issues.append(
            {
                "code": "allowed_edit_paths_not_executor_only",
                "message": "allowed_edit_paths must contain only executor_path",
            }
        )
    required_read_paths = {
        str(path)
        for path in [design_path, manifest_path, executor_path]
        if path is not None
    }
    missing_required_reads = sorted(required_read_paths - allowed_read_paths)
    if missing_required_reads:
        issues.append(
            {
                "code": "allowed_read_paths_missing_required_artifacts",
                "message": ",".join(missing_required_reads),
            }
        )
    overlapping_reads = sorted(allowed_read_paths & forbidden_read_paths)
    if overlapping_reads:
        issues.append(
            {
                "code": "forbidden_read_path_allowed",
                "message": ",".join(overlapping_reads),
            }
        )
    if not forbidden_read_paths:
        issues.append(
            {
                "code": "forbidden_read_paths_empty",
                "message": "target or labels must be listed as forbidden read paths",
            }
        )
    for path in _unique_path_objects(
        [design_path, manifest_path, executor_path]
        + [Path(path) for path in allowed_read_paths]
    ):
        if not path.exists():
            issues.append(
                {
                    "code": "required_authoring_path_missing",
                    "message": str(path),
                }
            )

    commands = authoring.get("validation_commands")
    if not isinstance(commands, list) or len(commands) != 3:
        issues.append(
            {
                "code": "validation_commands_invalid",
                "message": "validation_commands must contain exactly three commands",
            }
        )
    else:
        expected_tokens = [
            "validate-bounded-executor-implementation",
            "run-bounded-executor-sandbox",
            "record-bounded-executor-sandbox-event",
        ]
        for index, token in enumerate(expected_tokens):
            command = commands[index]
            if not isinstance(command, str) or token not in command:
                issues.append(
                    {
                        "code": "validation_command_order_invalid",
                        "message": f"validation command {index} must include {token}",
                    }
                )
    if authoring.get("next_command") != "audit-bounded-executor-authoring-task":
        issues.append(
            {
                "code": "next_command_invalid",
                "message": "next_command must be audit-bounded-executor-authoring-task",
            }
        )

    issue_codes = sorted({issue["code"] for issue in issues})
    decision = "approved" if not issues else "rejected"
    report = {
        "schema_version": 1,
        "state_hint": BOUNDED_EXECUTOR_AUTHORING_AUDIT_STATE_HINT,
        "authoring_audit_path": str(output_file),
        "authoring_task_path": str(authoring_file),
        "algorithm_candidate_id": authoring.get("algorithm_candidate_id"),
        "decision": decision,
        "issue_count": len(issues),
        "issue_codes": issue_codes,
        "issues": issues,
        "skill_name": skill_name,
        "skill_path": str(skill_path) if skill_path is not None else None,
        "allowed_edit_paths": sorted(allowed_edit_paths),
        "forbidden_read_paths": sorted(forbidden_read_paths),
        "next_command": "agent-author-bounded-executor"
        if decision == "approved"
        else "prepare-expansion-blocked-executor-authoring-tasks",
    }
    _write_json(output_file, report)
    return report


def _bounded_executor_authoring_skill_path(
    *,
    skill_root: Path,
    skill_name: str,
) -> Path | None:
    if not skill_name:
        return None
    candidates = [
        skill_root / skill_name / "SKILL.md",
        skill_root / skill_name.replace("-", "_") / "SKILL.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _audit_bounded_executor_skill_text(
    *,
    skill_path: Path,
    skill_name: str,
    issues: list[dict[str, str]],
) -> None:
    text = skill_path.read_text(encoding="utf-8")
    required_terms = [
        f"name: {skill_name}",
        BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT,
        "allowed_edit_paths",
        "forbidden_read_paths",
        "validation_commands",
        "def predict(context, candidate, params)",
    ]
    missing_terms = [term for term in required_terms if term not in text]
    if missing_terms:
        issues.append(
            {
                "code": "skill_contract_terms_missing",
                "message": ",".join(missing_terms),
            }
        )


def _artifact_path_or_issue(
    artifact: dict[str, Any],
    key: str,
    issues: list[dict[str, str]],
) -> Path | None:
    value = artifact.get(key)
    if not isinstance(value, str) or not value.strip():
        issues.append(
            {
                "code": f"{key}_missing",
                "message": f"{key} must be a non-empty path",
            }
        )
        return None
    return Path(value)


def _path_set_or_issue(
    value: Any,
    key: str,
    issues: list[dict[str, str]],
) -> set[str]:
    if not isinstance(value, list):
        issues.append(
            {
                "code": f"{key}_invalid",
                "message": f"{key} must be a list of paths",
            }
        )
        return set()
    result = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            issues.append(
                {
                    "code": f"{key}_invalid_item",
                    "message": f"{key} contains a non-string path",
                }
            )
            continue
        result.add(str(Path(item)))
    return result


def _unique_path_objects(paths: list[Path | None]) -> list[Path]:
    seen: set[str] = set()
    result = []
    for path in paths:
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result

def _bounded_executor_template_source() -> str:
    return '''"""Bounded PDE executor scaffold.

Agent edits should stay inside predict(context, candidate, params).
The verifier allows only a small import surface and rejects file/network/shell access.
"""

from __future__ import annotations

import numpy as np


IMPLEMENTATION_STATUS = "scaffold_baseline"


def predict(context, candidate, params):
    """Return a (N, horizon, X) rollout while preserving the context window."""
    array = np.asarray(context)
    if array.ndim != 3:
        raise ValueError("context must have shape (N, T, X)")
    horizon = int(params.get("horizon", 200))
    if horizon < array.shape[1]:
        raise ValueError("horizon must be at least the context length")
    prediction = np.empty((array.shape[0], horizon, array.shape[2]), dtype=array.dtype)
    context_steps = array.shape[1]
    prediction[:, :context_steps, :] = array
    prediction[:, context_steps:, :] = array[:, context_steps - 1 : context_steps, :]
    return prediction
'''


def _audit_bounded_executor_source(
    source: str,
    *,
    design: dict[str, Any],
) -> list[dict[str, Any]]:
    policy = design.get("security_policy") if isinstance(design.get("security_policy"), dict) else {}
    allowed_imports = set(_text_list(policy.get("allowed_imports", []), "allowed_imports"))
    forbidden_calls = set(_text_list(policy.get("forbidden_calls", []), "forbidden_calls"))
    forbidden_numpy_calls = set(
        _text_list(policy.get("forbidden_numpy_calls", []), "forbidden_numpy_calls")
        if isinstance(policy.get("forbidden_numpy_calls"), list)
        else []
    )
    api = design.get("executor_api") if isinstance(design.get("executor_api"), dict) else {}
    function_name = str(api.get("function_name") or "predict")
    required_args = [
        str(item)
        for item in api.get("required_args", ["context", "candidate", "params"])
        if isinstance(item, str)
    ]
    issues: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            {
                "code": "syntax_error",
                "message": str(exc),
                "line": exc.lineno,
            }
        ]
    found_function = False
    import_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".", 1)[0]
                if root_name not in allowed_imports:
                    issues.append(
                        {
                            "code": "disallowed_import",
                            "module": alias.name,
                            "line": node.lineno,
                        }
                    )
                import_aliases[alias.asname or root_name] = root_name
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module == "__future__":
                continue
            if module not in allowed_imports:
                issues.append(
                    {
                        "code": "disallowed_import",
                        "module": node.module,
                        "line": node.lineno,
                    }
                )
            for alias in node.names:
                if module == "numpy" and alias.name in forbidden_numpy_calls:
                    issues.append(
                        {
                            "code": "forbidden_numpy_import",
                            "module": node.module,
                            "name": alias.name,
                            "line": node.lineno,
                        }
                    )
                import_aliases[alias.asname or alias.name] = module
        elif isinstance(node, ast.FunctionDef) and node.name == function_name:
            found_function = True
            args = [arg.arg for arg in node.args.args]
            if args[: len(required_args)] != required_args:
                issues.append(
                    {
                        "code": "invalid_predict_signature",
                        "expected_args": required_args,
                        "actual_args": args,
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.Call):
            call_name = _ast_call_name(node.func)
            if call_name in forbidden_calls:
                issues.append(
                    {
                        "code": "forbidden_call",
                        "call": call_name,
                        "line": node.lineno,
                    }
                )
            if "." in call_name:
                receiver, attr = call_name.split(".", 1)
                root_module = import_aliases.get(receiver, receiver)
                if root_module == "numpy" and attr in forbidden_numpy_calls:
                    issues.append(
                        {
                            "code": "forbidden_numpy_call",
                            "call": call_name,
                            "line": node.lineno,
                        }
                    )
            else:
                root_module = import_aliases.get(call_name)
                if root_module == "numpy" and call_name in forbidden_numpy_calls:
                    issues.append(
                        {
                            "code": "forbidden_numpy_call",
                            "call": call_name,
                            "line": node.lineno,
                        }
                    )
    if not found_function:
        issues.append(
            {
                "code": "missing_predict_function",
                "function_name": function_name,
            }
        )
    return issues


def _ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _load_bounded_executor_module(source_file: Path) -> Any:
    module_name = f"_pde_agent_bounded_executor_{_slug(source_file.stem)}_{_sha256_file(source_file)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise LiteratureAlgorithmLabError(f"cannot load executor module: {source_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise LiteratureAlgorithmLabError("bounded executor missing predict function")
    return module


def _shell_safe_token(value: str) -> str:
    token = _nonempty_text(value, "token")
    if not re.fullmatch(r"[A-Za-z0-9_.=-]+", token):
        raise LiteratureAlgorithmLabError(f"unsafe shell token: {value}")
    return token

def _float_or_default(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError(f"{label} items must be objects")
        result.append(item)
    return result


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        result.append(_nonempty_text(item, label))
    return result


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureAlgorithmLabError(f"{label} must be a non-empty string")
    return value.strip()


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


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "source"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
