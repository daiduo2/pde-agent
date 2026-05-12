from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_common import (
    LiteratureAlgorithmLabError,
    guard_output,
    nonempty_text,
    read_json,
    require_file,
    write_json,
)
from pde_agent.core.literature_candidate_compilation import (
    ALGORITHM_CANDIDATES_STATE_HINT,
    FNO_CANDIDATE_ADAPTED_STATE_HINT,
)


ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT = "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED"


def evaluate_algorithm_candidate_promotion(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    sandbox_evidence_paths: list[str | Path],
    output_path: str | Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    candidates_report = read_json(
        require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") not in {
        ALGORITHM_CANDIDATES_STATE_HINT,
        FNO_CANDIDATE_ADAPTED_STATE_HINT,
    }:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    selected_id = nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    selected = _find_algorithm_candidate(candidates_report, selected_id)
    if not sandbox_evidence_paths:
        raise LiteratureAlgorithmLabError("sandbox_evidence_paths must be non-empty")
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    evidence_records = [
        read_json(require_file(Path(path), "sandbox evidence"))
        for path in sandbox_evidence_paths
    ]
    best_evidence = _select_best_candidate_evidence(
        evidence_records=evidence_records,
        candidate_id=selected_id,
    )
    passed_checks, failed_checks = _evaluate_candidate_promotion_checks(
        candidate=selected,
        evidence=best_evidence,
        current_best_score=current_best_score,
    )
    reason_codes = [
        str(check["failure_code"]) for check in failed_checks.values()
    ] or ["all_rubric_checks_passed"]
    decision = _candidate_promotion_decision(
        candidate=selected,
        failed_checks=failed_checks,
        evidence=best_evidence,
    )
    promote = decision == "promote"
    ledger_event = {
        "event_type": f"candidate_{'promoted' if promote else decision}",
        "algorithm_candidate_id": selected_id,
        "candidate_family": selected.get("candidate_family"),
        "evidence_id": best_evidence.get("evidence_id"),
        "decision": decision,
        "reason_codes": reason_codes,
    }
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT,
        "algorithm_candidate_promotion_path": str(output_file),
        "algorithm_candidates_path": str(Path(algorithm_candidates_path)),
        "algorithm_candidate_id": selected_id,
        "candidate_family": selected.get("candidate_family"),
        "sandbox_evidence_paths": [str(Path(path)) for path in sandbox_evidence_paths],
        "evidence_count": len(evidence_records),
        "best_evidence_id": best_evidence.get("evidence_id"),
        "current_best_score": current_best_score,
        "decision": decision,
        "promote_to_strategy_growth": promote,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "reason_codes": reason_codes,
        "metric_results": _candidate_promotion_metric_results(
            evidence=best_evidence,
            current_best_score=current_best_score,
        ),
        "ledger_event": ledger_event,
        "next_command": _candidate_promotion_next_command(decision),
    }
    mutation_evidence = selected.get("mutation_evidence")
    if isinstance(mutation_evidence, dict):
        report["mutation_evidence"] = dict(mutation_evidence)
    write_json(output_file, report)
    return report


def _find_algorithm_candidate(payload: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in payload.get("algorithm_candidates", []):
        if (
            isinstance(candidate, dict)
            and candidate.get("algorithm_candidate_id") == candidate_id
        ):
            return candidate
    raise LiteratureAlgorithmLabError(f"algorithm candidate not found: {candidate_id}")


def _select_best_candidate_evidence(
    *, evidence_records: list[dict[str, Any]], candidate_id: str
) -> dict[str, Any]:
    matching = [
        item
        for item in evidence_records
        if item.get("candidate_id") in {candidate_id, None}
    ]
    if not matching:
        raise LiteratureAlgorithmLabError(
            f"no sandbox evidence matched candidate: {candidate_id}"
        )
    return max(matching, key=lambda item: _float_or_default(item.get("score"), -1e30))


def _evaluate_candidate_promotion_checks(
    *,
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    current_best_score: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    passed: dict[str, Any] = {}
    failed: dict[str, Any] = {}
    criteria = candidate.get("promotion_criteria")
    criteria = criteria if isinstance(criteria, dict) else {}
    min_score_delta = _float_or_default(criteria.get("min_score_delta"), 0.0)
    max_inference_seconds = _float_or_default(
        criteria.get("max_inference_seconds"), 120.0
    )
    requires_ablation = bool(criteria.get("requires_ablation", False))

    executor = candidate.get("executor_spec")
    if (
        isinstance(executor, dict)
        and executor.get("cli_command") == "blocked_requires_executor_design"
    ):
        failed["executor"] = _promotion_check(
            failure_code="unsupported_executor",
            message="candidate requires bounded executor design before promotion",
        )
    else:
        passed["executor"] = {"passed": True}

    has_target_reference_issue = _evidence_has_any_gate_issue(
        evidence,
        {"target_leakage_risk", "target_reference_local_validation_only"},
    )
    if evidence.get("rules_passed") is True and not has_target_reference_issue:
        passed["validity"] = {"passed": True}
    elif has_target_reference_issue:
        failed["validity"] = _promotion_check(
            failure_code="target_leakage_risk",
            message=(
                "sandbox evidence used a target-reference local validation signal "
                "and must be refined against a legal reference split"
            ),
        )
    else:
        failed["validity"] = _promotion_check(
            failure_code="sandbox_rule_failed",
            message="sandbox rules did not pass",
        )

    score_delta = _evidence_score_delta(evidence, current_best_score)
    if score_delta >= min_score_delta:
        passed["score"] = {
            "passed": True,
            "score_delta": score_delta,
            "threshold": min_score_delta,
        }
    else:
        failed["score"] = _promotion_check(
            failure_code="score_regressed",
            message="score delta did not meet promotion threshold",
            score_delta=score_delta,
            threshold=min_score_delta,
        )

    runtime_seconds = _evidence_runtime_seconds(evidence)
    if runtime_seconds <= max_inference_seconds:
        passed["runtime"] = {
            "passed": True,
            "inference_seconds": runtime_seconds,
            "threshold": max_inference_seconds,
        }
    else:
        failed["runtime"] = _promotion_check(
            failure_code="runtime_budget_exceeded",
            message="inference runtime exceeded candidate budget",
            inference_seconds=runtime_seconds,
            threshold=max_inference_seconds,
        )

    physics_delta = _evidence_physics_delta(evidence)
    if physics_delta <= 0.0:
        passed["physics"] = {"passed": True, "residual_delta": physics_delta}
    else:
        failed["physics"] = _promotion_check(
            failure_code="physics_regressed",
            message="physics diagnostic regressed",
            residual_delta=physics_delta,
        )

    ablation = evidence.get("ablation_metrics")
    ablation_delta = (
        _float_or_default(ablation.get("score_delta"), 0.0)
        if isinstance(ablation, dict)
        else 0.0
    )
    if not requires_ablation or ablation_delta > 0.0:
        passed["ablation"] = {
            "passed": True,
            "required": requires_ablation,
            "score_delta": ablation_delta,
        }
    else:
        failed["ablation"] = _promotion_check(
            failure_code="ablation_missing_or_weak",
            message="required ablation evidence did not support candidate",
            score_delta=ablation_delta,
        )

    reproducibility = evidence.get("reproducibility")
    deterministic = (
        bool(reproducibility.get("deterministic"))
        if isinstance(reproducibility, dict)
        else False
    )
    if deterministic:
        passed["reproducibility"] = {"passed": True}
    else:
        failed["reproducibility"] = _promotion_check(
            failure_code="insufficient_reproducibility",
            message="evidence is not marked deterministic",
        )
    return passed, failed


def _candidate_promotion_decision(
    *,
    candidate: dict[str, Any],
    failed_checks: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    if "executor" in failed_checks:
        return "blocked"
    if "runtime" in failed_checks:
        return "reject"
    validity_failure = failed_checks.get("validity")
    if isinstance(validity_failure, dict):
        if validity_failure.get("failure_code") == "target_leakage_risk":
            return "refine" if "score" not in failed_checks else "deprioritize"
        return "reject"
    if "score" in failed_checks:
        return "deprioritize" if evidence.get("rules_passed") is True else "reject"
    if failed_checks:
        return "refine"
    return "promote"


def _candidate_promotion_next_command(decision: str) -> str:
    if decision == "promote":
        return "record-strategy-proposal"
    if decision == "refine":
        return "refine-algorithm-candidate"
    if decision == "blocked":
        return "create-executor-design-request"
    return "record-literature-registry-event"


def _candidate_promotion_metric_results(
    *, evidence: dict[str, Any], current_best_score: float
) -> dict[str, Any]:
    return {
        "score": evidence.get("score"),
        "current_best_score": current_best_score,
        "score_delta_vs_current_best": _evidence_score_delta(
            evidence, current_best_score
        ),
        "inference_seconds": _evidence_runtime_seconds(evidence),
        "physics_residual_delta": _evidence_physics_delta(evidence),
        "rules_passed": evidence.get("rules_passed"),
        "candidate_gate_issues": _evidence_gate_issues(evidence),
        "diagnostic_only": evidence.get("diagnostic_only"),
    }


def _promotion_check(
    *, failure_code: str, message: str, **extra: Any
) -> dict[str, Any]:
    result = {"passed": False, "failure_code": failure_code, "message": message}
    result.update(extra)
    return result


def _evidence_score_delta(evidence: dict[str, Any], current_best_score: float) -> float:
    if isinstance(evidence.get("score_delta_vs_current_best"), int | float):
        return float(evidence["score_delta_vs_current_best"])
    score = _float_or_default(evidence.get("score"), current_best_score)
    return score - current_best_score


def _evidence_runtime_seconds(evidence: dict[str, Any]) -> float:
    runtime = evidence.get("runtime")
    if isinstance(runtime, dict):
        return _float_or_default(runtime.get("inference_seconds"), 1e30)
    return _float_or_default(runtime, 1e30)


def _evidence_physics_delta(evidence: dict[str, Any]) -> float:
    physics = evidence.get("physics_metrics")
    if isinstance(physics, dict):
        return _float_or_default(physics.get("residual_delta"), 0.0)
    return 0.0


def _evidence_has_any_gate_issue(evidence: dict[str, Any], issues: set[str]) -> bool:
    return any(issue in issues for issue in _evidence_gate_issues(evidence))


def _evidence_gate_issues(evidence: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in ["candidate_gate_issues", "gate_issues", "diagnostic_flags"]:
        value = evidence.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str) and item.strip() and item not in issues:
                issues.append(item.strip())
    return issues


def _float_or_default(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default
