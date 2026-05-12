from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_common import (
    LiteratureAlgorithmLabError,
    guard_output,
    read_json,
    require_file,
    write_json,
)


ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT = "ALGORITHM_CARDS_V2_VALIDATED"

ALGORITHM_CARD_V2_OPERATOR_TARGETS = {
    "ic_to_trajectory",
    "coefficient_to_solution",
    "residual_to_correction",
    "context_to_rollout",
    "postprocess_to_solution",
}

ALGORITHM_CARD_V2_MECHANISM_TYPES = {
    "architecture",
    "loss",
    "rollout",
    "postprocess",
    "conditioning",
    "checkpoint",
    "hybrid",
}

ALGORITHM_CARD_V2_EVIDENCE_STATUSES = {
    "extracted",
    "candidate_compiled",
    "sandbox_planned",
    "tested",
    "promoted",
    "rejected",
    "blocked",
}

ALGORITHM_CARD_V2_IMPLEMENTATION_REQUIREMENTS = {
    "requires_training",
    "requires_checkpoint",
    "requires_target_data",
    "requires_new_executor",
    "requires_external_code",
    "can_run_as_postprocess",
}


def validate_algorithm_cards_v2(
    *,
    algorithm_cards_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    cards_report = read_json(
        require_file(Path(algorithm_cards_path), "algorithm cards")
    )
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    issues: list[dict[str, Any]] = []
    if cards_report.get("state_hint") != ALGORITHM_CARDS_STATE_HINT:
        _append_algorithm_card_v2_issue(
            issues,
            code="invalid_state_hint",
            path="state_hint",
            failure_code="candidate_schema_invalid",
            message="algorithm cards report must have ALGORITHM_CARDS_EXTRACTED state",
        )
    cards_value = cards_report.get("algorithm_cards")
    if not isinstance(cards_value, list):
        _append_algorithm_card_v2_issue(
            issues,
            code="invalid_algorithm_cards",
            path="algorithm_cards",
            failure_code="candidate_schema_invalid",
            message="algorithm_cards must be a list",
        )
        cards: list[Any] = []
    else:
        cards = cards_value
    for index, card in enumerate(cards):
        issues.extend(_validate_algorithm_card_v2(card, index=index))
    decision = "approved" if not issues else "rejected"
    next_command = (
        "compile-algorithm-candidates"
        if decision == "approved"
        else "fix-algorithm-card-v2-schema"
    )
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT,
        "algorithm_cards_path": str(Path(algorithm_cards_path)),
        "algorithm_card_v2_audit_path": str(output_file),
        "decision": decision,
        "card_count": len(cards),
        "issue_count": len(issues),
        "issues": issues,
        "next_command": next_command,
    }
    write_json(output_file, report)
    return report


def _validate_algorithm_card_v2(card: Any, *, index: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    base = f"algorithm_cards[{index}]"
    if not isinstance(card, dict):
        _append_algorithm_card_v2_issue(
            issues,
            code="invalid_algorithm_card",
            path=base,
            failure_code="candidate_schema_invalid",
            message="algorithm card must be an object",
        )
        return issues
    for field in (
        "algorithm_card_id",
        "claim",
        "mathematical_object",
        "operation",
        "minimal_experiment",
    ):
        failure_code = (
            "no_algorithmic_claim" if field == "operation" else "candidate_schema_invalid"
        )
        if not _is_nonempty_string(card.get(field)):
            _append_algorithm_card_v2_issue(
                issues,
                code=f"missing_{field}",
                path=f"{base}.{field}",
                failure_code=failure_code,
                message=f"{field} must be a non-empty string",
            )
    for field in (
        "source_ids",
        "implementation_hooks",
        "preconditions",
        "expected_observables",
        "failure_modes",
    ):
        value = card.get(field)
        if not isinstance(value, list) or not value:
            _append_algorithm_card_v2_issue(
                issues,
                code=f"empty_{field}",
                path=f"{base}.{field}",
                failure_code="candidate_schema_invalid",
                message=f"{field} must be a non-empty list of strings",
            )
            continue
        for item_index, item in enumerate(value):
            if not _is_nonempty_string(item):
                _append_algorithm_card_v2_issue(
                    issues,
                    code=f"invalid_{field}_item",
                    path=f"{base}.{field}[{item_index}]",
                    failure_code="candidate_schema_invalid",
                    message=f"{field} items must be non-empty strings",
                )
    _validate_algorithm_card_v2_enum(
        issues,
        value=card.get("operator_target"),
        allowed=ALGORITHM_CARD_V2_OPERATOR_TARGETS,
        code="invalid_operator_target",
        path=f"{base}.operator_target",
    )
    _validate_algorithm_card_v2_enum(
        issues,
        value=card.get("mechanism_type"),
        allowed=ALGORITHM_CARD_V2_MECHANISM_TYPES,
        code="invalid_mechanism_type",
        path=f"{base}.mechanism_type",
    )
    _validate_algorithm_card_v2_enum(
        issues,
        value=card.get("evidence_status"),
        allowed=ALGORITHM_CARD_V2_EVIDENCE_STATUSES,
        code="invalid_evidence_status",
        path=f"{base}.evidence_status",
    )
    requirements = card.get("implementation_requirements")
    if not isinstance(requirements, dict):
        _append_algorithm_card_v2_issue(
            issues,
            code="missing_implementation_requirements",
            path=f"{base}.implementation_requirements",
            failure_code="candidate_schema_invalid",
            message="implementation_requirements must be an object",
        )
    else:
        for key in sorted(ALGORITHM_CARD_V2_IMPLEMENTATION_REQUIREMENTS):
            if key not in requirements:
                _append_algorithm_card_v2_issue(
                    issues,
                    code="missing_implementation_requirement",
                    path=f"{base}.implementation_requirements.{key}",
                    failure_code="candidate_schema_invalid",
                    message=f"implementation requirement missing: {key}",
                )
            elif not isinstance(requirements[key], bool):
                _append_algorithm_card_v2_issue(
                    issues,
                    code="implementation_requirement_not_bool",
                    path=f"{base}.implementation_requirements.{key}",
                    failure_code="candidate_schema_invalid",
                    message=f"implementation requirement must be boolean: {key}",
                )
    provenance = card.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        _append_algorithm_card_v2_issue(
            issues,
            code="missing_provenance",
            path=f"{base}.provenance",
            failure_code="candidate_schema_invalid",
            message="provenance must be a non-empty object",
        )
    _validate_algorithm_card_v2_mutation_evidence(
        issues,
        value=card.get("mutation_evidence"),
        base_path=f"{base}.mutation_evidence",
    )
    return issues


def _validate_algorithm_card_v2_mutation_evidence(
    issues: list[dict[str, Any]],
    *,
    value: Any,
    base_path: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        _append_algorithm_card_v2_issue(
            issues,
            code="invalid_mutation_evidence",
            path=base_path,
            failure_code="candidate_schema_invalid",
            message="mutation_evidence must be an object when present",
        )
        return
    for field in ("parent_candidate_family", "mutation_type", "reason"):
        if not _is_nonempty_string(value.get(field)):
            _append_algorithm_card_v2_issue(
                issues,
                code=f"missing_{field}",
                path=f"{base_path}.{field}",
                failure_code="candidate_schema_invalid",
                message=f"mutation_evidence.{field} must be a non-empty string",
            )


def _validate_algorithm_card_v2_enum(
    issues: list[dict[str, Any]],
    *,
    value: Any,
    allowed: set[str],
    code: str,
    path: str,
) -> None:
    if not isinstance(value, str) or value not in allowed:
        _append_algorithm_card_v2_issue(
            issues,
            code=code,
            path=path,
            failure_code="candidate_schema_invalid",
            message=f"{path} must be one of {sorted(allowed)}",
        )


def _append_algorithm_card_v2_issue(
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


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
