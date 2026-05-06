"""Audit Hypothesis Lab as Hermes runtime infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDITED"
ACCEPTANCE_STATE_HINT = "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN"
CHAIN_STATE_HINT = "HERMES_RUNTIME_AUDIT_CHAIN_RAN"


class HermesHypothesisLabInfrastructureAuditError(RuntimeError):
    pass


def audit_hermes_hypothesis_lab_infrastructure(
    *,
    acceptance_smoke_path: str | Path,
    preflight_chain_report_path: str | Path,
    runbook_path: str | Path,
    output_path: str | Path,
    min_completed_iterations: int = 1,
    stability_chain_report_path: str | Path | None = None,
    candidate_pressure_chain_report_path: str | Path | None = None,
    force: bool,
) -> dict[str, Any]:
    if min_completed_iterations < 1:
        raise HermesHypothesisLabInfrastructureAuditError(
            "min_completed_iterations must be positive"
        )
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise HermesHypothesisLabInfrastructureAuditError(
            f"infrastructure audit exists: {output_file}"
        )

    acceptance_path = Path(acceptance_smoke_path)
    chain_path = Path(preflight_chain_report_path)
    runbook_file = Path(runbook_path)
    stability_chain_path = (
        Path(stability_chain_report_path)
        if stability_chain_report_path is not None
        else None
    )
    candidate_pressure_chain_path = (
        Path(candidate_pressure_chain_report_path)
        if candidate_pressure_chain_report_path is not None
        else None
    )
    acceptance = _read_json_if_exists(acceptance_path)
    chain = _read_json_if_exists(chain_path)
    stability_chain = (
        _read_json_if_exists(stability_chain_path)
        if stability_chain_path is not None
        else {}
    )
    candidate_pressure_chain = (
        _read_json_if_exists(candidate_pressure_chain_path)
        if candidate_pressure_chain_path is not None
        else {}
    )
    runbook_text = runbook_file.read_text(encoding="utf-8") if runbook_file.exists() else ""

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    _add_check(
        checks,
        errors,
        "acceptance_file_exists",
        acceptance_path.exists(),
        {"path": str(acceptance_path)},
    )
    _add_check(
        checks,
        errors,
        "acceptance_state_hint",
        acceptance.get("state_hint") == ACCEPTANCE_STATE_HINT,
        {"actual": acceptance.get("state_hint"), "expected": ACCEPTANCE_STATE_HINT},
    )
    _add_check(
        checks,
        errors,
        "acceptance_decision_passed",
        acceptance.get("decision") == "passed",
        {"actual": acceptance.get("decision")},
    )
    acceptance_checks = acceptance.get("checks")
    _add_check(
        checks,
        errors,
        "acceptance_all_checks_passed",
        isinstance(acceptance_checks, dict)
        and bool(acceptance_checks)
        and all(value is True for value in acceptance_checks.values()),
        {"checks": acceptance_checks},
    )
    acceptance_metrics = acceptance.get("metrics")
    _add_check(
        checks,
        errors,
        "acceptance_metric_consistency",
        isinstance(acceptance_metrics, dict)
        and acceptance_metrics.get("passed_check_count")
        == acceptance_metrics.get("check_count")
        and isinstance(acceptance_metrics.get("check_count"), int)
        and acceptance_metrics.get("check_count", 0) > 0,
        {"metrics": acceptance_metrics},
    )

    _add_check(
        checks,
        errors,
        "preflight_chain_file_exists",
        chain_path.exists(),
        {"path": str(chain_path)},
    )
    _add_check(
        checks,
        errors,
        "preflight_chain_state_hint",
        chain.get("state_hint") == CHAIN_STATE_HINT,
        {"actual": chain.get("state_hint"), "expected": CHAIN_STATE_HINT},
    )
    _add_check(
        checks,
        errors,
        "preflight_chain_decision_valid",
        chain.get("decision") in {"passed", "repaired"},
        {"actual": chain.get("decision")},
    )
    chain_metrics = chain.get("metrics")
    completed_iterations = (
        chain_metrics.get("completed_iteration_count")
        if isinstance(chain_metrics, dict)
        else None
    )
    _add_check(
        checks,
        errors,
        "preflight_chain_completed_iterations",
        isinstance(completed_iterations, int)
        and completed_iterations >= min_completed_iterations,
        {
            "actual": completed_iterations,
            "minimum": min_completed_iterations,
        },
    )
    _add_check(
        checks,
        errors,
        "preflight_chain_failed_audit_zero",
        isinstance(chain_metrics, dict) and chain_metrics.get("failed_audit_count") == 0,
        {"metrics": chain_metrics},
    )
    _add_check(
        checks,
        errors,
        "preflight_metric_passed",
        isinstance(chain_metrics, dict)
        and chain_metrics.get("preflight_acceptance_smoke_passed") is True,
        {"metrics": chain_metrics},
    )
    _add_check(
        checks,
        errors,
        "preflight_infrastructure_metric_passed",
        not isinstance(chain_metrics, dict)
        or "preflight_infrastructure_audit_passed" not in chain_metrics
        or chain_metrics.get("preflight_infrastructure_audit_passed") is True,
        {"metrics": chain_metrics},
    )
    chain_artifacts = chain.get("artifacts")
    _add_check(
        checks,
        errors,
        "preflight_artifact_matches_acceptance",
        isinstance(chain_artifacts, dict)
        and Path(str(chain_artifacts.get("preflight_acceptance_smoke_path")))
        == acceptance_path,
        {
            "actual": chain_artifacts.get("preflight_acceptance_smoke_path")
            if isinstance(chain_artifacts, dict)
            else None,
            "expected": str(acceptance_path),
        },
    )
    _add_check(
        checks,
        errors,
        "runbook_mentions_preflight_gate",
        runbook_file.exists()
        and "run-hermes-auto-repair-memory-acceptance-smoke" in runbook_text
        and "run-hermes-runtime-audit-chain" in runbook_text
        and "--preflight-acceptance-smoke-path" in runbook_text,
        {"path": str(runbook_file)},
    )
    _add_check(
        checks,
        errors,
        "runbook_mentions_infrastructure_preflight_gate",
        runbook_file.exists()
        and "--preflight-infrastructure-audit-path" in runbook_text
        and "run_hermes_hypothesis_lab_infrastructure.py" in runbook_text,
        {"path": str(runbook_file)},
    )

    if stability_chain_path is not None:
        _add_check(
            checks,
            errors,
            "stability_chain_file_exists",
            stability_chain_path.exists(),
            {"path": str(stability_chain_path)},
        )
        _add_check(
            checks,
            errors,
            "stability_chain_dual_preflight_passed",
            _chain_dual_preflight_passed(
                stability_chain,
                min_completed_iterations=min_completed_iterations,
            ),
            {"metrics": stability_chain.get("metrics")},
        )

    if candidate_pressure_chain_path is not None:
        _add_check(
            checks,
            errors,
            "candidate_pressure_chain_file_exists",
            candidate_pressure_chain_path.exists(),
            {"path": str(candidate_pressure_chain_path)},
        )
        _add_check(
            checks,
            errors,
            "candidate_pressure_chain_dual_preflight_passed",
            _chain_dual_preflight_passed(
                candidate_pressure_chain,
                min_completed_iterations=min_completed_iterations,
            ),
            {"metrics": candidate_pressure_chain.get("metrics")},
        )
        _add_check(
            checks,
            errors,
            "candidate_pressure_attempted_nontrivial_family",
            _candidate_pressure_attempted_nontrivial_family(
                candidate_pressure_chain=candidate_pressure_chain,
                chain_path=candidate_pressure_chain_path,
            ),
            {"path": str(candidate_pressure_chain_path)},
        )

    artifacts = {
        "acceptance_smoke_path": str(acceptance_path),
        "preflight_chain_report_path": str(chain_path),
        "runbook_path": str(runbook_file),
    }
    if stability_chain_path is not None:
        artifacts["stability_chain_report_path"] = str(stability_chain_path)
    if candidate_pressure_chain_path is not None:
        artifacts["candidate_pressure_chain_report_path"] = str(
            candidate_pressure_chain_path
        )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "decision": "passed" if not errors else "failed",
        "audit_report_path": str(output_file),
        "artifacts": artifacts,
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "failed_check_count": len(errors),
        },
        "errors": errors,
        "next_command": "run-hermes-runtime-audit-chain"
        if not errors
        else "repair-hermes-hypothesis-lab-infrastructure",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _chain_dual_preflight_passed(
    chain: dict[str, Any],
    *,
    min_completed_iterations: int,
) -> bool:
    metrics = chain.get("metrics")
    return (
        chain.get("state_hint") == CHAIN_STATE_HINT
        and chain.get("decision") in {"passed", "repaired"}
        and isinstance(metrics, dict)
        and metrics.get("failed_audit_count") == 0
        and isinstance(metrics.get("completed_iteration_count"), int)
        and metrics.get("completed_iteration_count", 0) >= min_completed_iterations
        and metrics.get("preflight_acceptance_smoke_passed") is True
        and metrics.get("preflight_infrastructure_audit_passed") is True
    )


def _candidate_pressure_attempted_nontrivial_family(
    *,
    candidate_pressure_chain: dict[str, Any],
    chain_path: Path,
) -> bool:
    iterations = candidate_pressure_chain.get("iterations")
    if not isinstance(iterations, list):
        return False
    for iteration in iterations:
        if not isinstance(iteration, dict):
            continue
        runtime_path_value = iteration.get("runtime_report_path")
        if not isinstance(runtime_path_value, str) or not runtime_path_value:
            continue
        runtime_path = _resolve_artifact_path(runtime_path_value, base_path=chain_path)
        runtime = _read_json_if_exists(runtime_path)
        artifacts = runtime.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        heartbeat_path_value = artifacts.get("heartbeat_report_path")
        if not isinstance(heartbeat_path_value, str) or not heartbeat_path_value:
            continue
        heartbeat_path = _resolve_artifact_path(
            heartbeat_path_value,
            base_path=runtime_path,
        )
        heartbeat = _read_json_if_exists(heartbeat_path)
        if _heartbeat_attempted_nontrivial_family(heartbeat):
            return True
    return False


def _heartbeat_attempted_nontrivial_family(heartbeat: dict[str, Any]) -> bool:
    batches = heartbeat.get("batches")
    if not isinstance(batches, list):
        return False
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        family = batch.get("candidate_generator")
        diagnostics = batch.get("candidate_diagnostics")
        attempted = (
            diagnostics.get("attempted_candidate_count")
            if isinstance(diagnostics, dict)
            else None
        )
        if (
            isinstance(family, str)
            and family
            and family != "postprocess"
            and isinstance(attempted, int)
            and attempted > 0
        ):
            return True
    return False


def _resolve_artifact_path(value: str, *, base_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_path.parent / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesHypothesisLabInfrastructureAuditError(
            f"invalid JSON artifact: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise HermesHypothesisLabInfrastructureAuditError(
            f"artifact must be a JSON object: {path}"
        )
    return payload


def _add_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    check_id: str,
    passed: bool,
    details: dict[str, Any],
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "passed": bool(passed),
            "details": details,
        }
    )
    if not passed:
        errors.append(check_id)
