"""Registry summary and refinement policy for literature expansion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REGISTRY_SUMMARY_STATE_HINT = "LITERATURE_REGISTRY_SUMMARY_BUILT"
REFINEMENT_STATE_HINT = "ALGORITHM_CANDIDATE_REFINEMENT_PREPARED"


class LiteratureRegistryPolicyError(RuntimeError):
    pass


def summarize_registry_policy(
    *,
    registry: dict[str, Any],
    registry_path: str | Path,
    output_path: str | Path,
    current_best_score: float,
) -> dict[str, Any]:
    source_backlog = _registry_source_backlog(registry)
    candidate_actions = _registry_candidate_actions(
        registry,
        current_best_score=float(current_best_score),
    )
    learning_signals = _registry_learning_signals(
        candidate_actions,
        current_best_score=float(current_best_score),
    )
    family_summary = _registry_family_summary(candidate_actions)
    avoid_repetition = _registry_avoid_repetition(
        candidate_actions,
        current_best_score=float(current_best_score),
    )
    return {
        "schema_version": 1,
        "state_hint": REGISTRY_SUMMARY_STATE_HINT,
        "registry_path": str(registry_path),
        "summary_path": str(output_path),
        "current_best_score": float(current_best_score),
        "source_backlog": source_backlog,
        "candidate_action_queue": candidate_actions,
        "family_summary": family_summary,
        "learning_signals": learning_signals,
        "avoid_repetition": avoid_repetition,
        "metrics": {
            "source_backlog_count": len(source_backlog),
            "candidate_action_count": len(candidate_actions),
            "candidate_family_count": len(family_summary),
            "learning_signal_count": len(learning_signals),
            "avoid_repetition_count": len(avoid_repetition),
        },
        "next_command": _summary_next_command(candidate_actions, source_backlog),
    }


def build_refinement_policy(
    *,
    summary: dict[str, Any],
    summary_path: str | Path,
    output_path: str | Path,
    algorithm_candidate_id: str | None = None,
) -> dict[str, Any]:
    action = _select_refinement_action(
        summary,
        algorithm_candidate_id=algorithm_candidate_id,
    )
    candidate_id = _text(action.get("candidate_id"), "candidate_id")
    candidate_family = action.get("candidate_family")
    refinement_type = _refinement_type(action)
    candidate_generator = _refinement_candidate_generator(action)
    learning_objective = _refinement_learning_objective(action)
    reference_sample_limit = _refinement_reference_sample_limit(action)
    return {
        "schema_version": 1,
        "state_hint": REFINEMENT_STATE_HINT,
        "candidate_refinement_path": str(output_path),
        "registry_summary_path": str(summary_path),
        "algorithm_candidate_id": candidate_id,
        "candidate_family": candidate_family,
        "refinement_type": refinement_type,
        "reason": action.get("reason"),
        "diagnostic_score": action.get("best_score"),
        "score_delta_vs_current_best": action.get("score_delta_vs_current_best"),
        "candidate_gate_issues": _string_list(action.get("candidate_gate_issues")),
        "executor": action.get("executor"),
        "candidate_generator": candidate_generator,
        "learning_objective": learning_objective,
        "legal_reference_constraints": {
            "reference_policy": "reference_path_must_be_distinct_from_target_path",
            "forbidden_reference_modes": [
                "target_as_reference",
                "local_validation_only",
            ],
            "required_checks": [
                "rules_passed_true",
                "context_prefix_preserved",
                "no_target_reference_local_validation_only_gate_issue",
            ],
        },
        "suggested_sandbox_args": {
            "reference_path_required": refinement_type == "legal_reference_retest",
            "reference_sample_limit": reference_sample_limit,
            "candidate_generator": candidate_generator,
            "learning_objective": learning_objective,
            "algorithm_candidate_id": candidate_id,
        },
        "success_criteria": [
            "rules_passed must be true",
            *(
                [
                    "candidate_gate_issues must not include target_reference_local_validation_only",
                    "score_delta_vs_current_best should remain positive on legal reference",
                ]
                if refinement_type == "legal_reference_retest"
                else [
                    "new ablation or variant evidence must address prior failure_codes",
                    "do not promote unless ablation_missing_or_weak is cleared",
                ]
            ),
        ],
        "next_command": "run-expansion-sandbox-queue",
    }


def _select_refinement_action(
    summary: dict[str, Any],
    *,
    algorithm_candidate_id: str | None,
) -> dict[str, Any]:
    actions = [
        item
        for item in _object_items(summary.get("candidate_action_queue"))
        if item.get("recommended_action")
        in {"refine_with_legal_reference", "refine_with_ablation_or_variant"}
    ]
    if algorithm_candidate_id is not None:
        candidate_id = _text(algorithm_candidate_id, "algorithm_candidate_id")
        actions = [item for item in actions if item.get("candidate_id") == candidate_id]
    if not actions:
        raise LiteratureRegistryPolicyError("registry summary has no refinement action")
    return actions[0]


def _refinement_candidate_generator(action: dict[str, Any]) -> str:
    reason = str(action.get("reason", ""))
    candidate_id = str(action.get("candidate_id", ""))
    if "early_transition_expert" in candidate_id:
        return "task1_early_transition_expert"
    if "sem_burgers_rollout" in candidate_id:
        return "task1_sem_burgers_rollout"
    if reason == "candidate_needs_refinement":
        return "hypothesis_lab_candidate_variant_search"
    if action.get("candidate_family") == "local_strategy_skill":
        return "hypothesis_lab_early_subsegment_patch_search"
    return "auto"


def _refinement_learning_objective(action: dict[str, Any]) -> str:
    if action.get("reason") == "candidate_needs_refinement":
        return "ablation_or_variant_evidence"
    if action.get("candidate_family") == "local_strategy_skill":
        return "early_accuracy"
    return "score"


def _refinement_reference_sample_limit(action: dict[str, Any]) -> int | None:
    if action.get("candidate_family") == "local_strategy_skill":
        return 64
    return None


def _refinement_type(action: dict[str, Any]) -> str:
    if action.get("recommended_action") == "refine_with_legal_reference":
        return "legal_reference_retest"
    if action.get("recommended_action") == "refine_with_ablation_or_variant":
        return "ablation_or_variant_refinement"
    return "generic_refinement"


def _registry_avoid_repetition(
    actions: list[dict[str, Any]],
    *,
    current_best_score: float,
) -> list[dict[str, Any]]:
    avoid = [
        {
            "candidate_id": action["candidate_id"],
            "reason": "sandbox_score_below_current_best",
            "best_score": action.get("best_score"),
            "current_best_score": current_best_score,
            "recommended_next": "only_retry_with_new_variant_or_ablation",
        }
        for action in actions
        if action.get("recommended_action") == "refine_or_ablate"
        and float(action.get("score_delta_vs_current_best", 0.0)) < 0.0
    ]
    for action in actions:
        if (
            action.get("reason") == "candidate_failure_history_blocks_retry"
            and action.get("retry_allowed") is False
        ):
            avoid.append(
                {
                    "candidate_id": action["candidate_id"],
                    "reason": action["reason"],
                    "failure_codes": action.get("failure_codes", []),
                    "recommended_next": action.get("recommended_next"),
                    "retry_allowed": False,
                }
            )
        elif action.get("recommended_action") == "generate_variant_or_ablation":
            avoid.append(
                {
                    "candidate_id": action["candidate_id"],
                    "reason": action["reason"],
                    "failure_codes": action.get("failure_codes", []),
                    "recommended_next": action.get("recommended_next"),
                    "retry_allowed": False,
                }
            )
    return avoid


def _registry_source_backlog(registry: dict[str, Any]) -> list[dict[str, Any]]:
    backlog = []
    for source in registry.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_id = _text(source.get("source_id"), "source_id")
        parse_status = str(source.get("parse_status", "not_parsed"))
        card_ids = source.get("algorithm_card_ids")
        if not isinstance(card_ids, list):
            card_ids = []
        markdown_path = source.get("markdown_path")
        has_parse_artifact = isinstance(markdown_path, str) and bool(markdown_path.strip())
        if parse_status in {"not_parsed", "unknown", "failed"} and not (
            has_parse_artifact or card_ids
        ):
            backlog.append(
                {
                    "source_id": source_id,
                    "title": source.get("title"),
                    "status": "downloaded_not_parsed",
                    "next_command": "submit-mineru-parse",
                    "priority": 80,
                }
            )
        elif (parse_status == "usable" or has_parse_artifact) and not card_ids:
            backlog.append(
                {
                    "source_id": source_id,
                    "title": source.get("title"),
                    "status": "parsed_without_algorithm_cards",
                    "next_command": "extract-algorithm-cards",
                    "priority": 70,
                }
            )
    return sorted(
        backlog,
        key=lambda item: (-int(item["priority"]), str(item["source_id"])),
    )


def _registry_candidate_actions(
    registry: dict[str, Any],
    *,
    current_best_score: float,
) -> list[dict[str, Any]]:
    actions = []
    algorithm_by_candidate = _registry_algorithm_by_candidate(registry)
    for candidate in registry.get("candidate_nodes", []):
        if not isinstance(candidate, dict):
            continue
        candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
        status = str(candidate.get("status", "compiled"))
        executor = str(candidate.get("executor", ""))
        algorithm = algorithm_by_candidate.get(candidate_id, {})
        action = {
            "candidate_id": candidate_id,
            "candidate_family": candidate.get("candidate_family"),
            "status": status,
            "executor": executor,
            "algorithm_node_id": algorithm.get("node_id"),
            "source_ids": candidate.get("source_ids", []),
        }
        mutation_evidence = candidate.get("mutation_evidence")
        if isinstance(mutation_evidence, dict):
            action["mutation_evidence"] = dict(mutation_evidence)
        failure = _latest_failure_history(candidate)
        if failure and _failure_requires_local_strategy_variant(
            candidate=candidate,
            failure=failure,
        ):
            failure_codes = _string_list(failure.get("failure_codes"))
            action.update(
                {
                    "recommended_action": "generate_variant_or_ablation",
                    "next_command": "run-candidate-expansion-handoff",
                    "priority": 75,
                    "reason": "local_strategy_failed_legal_retest_requires_variant",
                    "failure_codes": failure_codes,
                    "recommended_next": "generate_variant_with_new_ablation",
                    "retry_allowed": False,
                    "failure_artifact_path": failure.get("artifact_path"),
                    "variant_constraints": {
                        "parent_candidate_id": candidate_id,
                        "required_mutation_evidence": True,
                        "forbidden_retry_without_new_ablation": True,
                        "target_failure_codes": failure_codes,
                    },
                }
            )
        elif failure and failure.get("retry_allowed") is False:
            action.update(
                {
                    "recommended_action": "deprioritize",
                    "next_command": "review-literature-registry",
                    "priority": 20,
                    "reason": "candidate_failure_history_blocks_retry",
                    "failure_codes": failure.get("failure_codes", []),
                    "recommended_next": failure.get("recommended_next"),
                    "retry_allowed": False,
                    "failure_artifact_path": failure.get("artifact_path"),
                }
            )
        elif status == "needs_refinement":
            metrics = (
                failure.get("metric_results")
                if isinstance(failure.get("metric_results"), dict)
                else {}
            )
            target_retest = _failure_has_code(failure, "target_leakage_risk")
            action.update(
                {
                    "recommended_action": (
                        "refine_with_legal_reference"
                        if target_retest
                        else "refine_with_ablation_or_variant"
                    ),
                    "next_command": "refine-algorithm-candidate",
                    "priority": 95 if target_retest else 85,
                    "reason": (
                        "target_reference_signal_requires_legal_retest"
                        if target_retest
                        else "candidate_needs_refinement"
                    ),
                    "best_score": _optional_float(metrics.get("score")),
                    "score_delta_vs_current_best": _optional_float(
                        metrics.get("score_delta_vs_current_best")
                    ),
                    "candidate_gate_issues": _string_list(
                        metrics.get("candidate_gate_issues")
                    ),
                    "diagnostic_only": bool(metrics.get("diagnostic_only")),
                    "failure_codes": failure.get("failure_codes", []),
                    "recommended_next": failure.get("recommended_next"),
                    "retry_allowed": failure.get("retry_allowed"),
                    "failure_artifact_path": failure.get("artifact_path"),
                }
            )
        elif status == "compiled":
            action.update(
                {
                    "recommended_action": "run_sandbox",
                    "next_command": executor or "run-hypothesis-lab",
                    "priority": 100,
                    "reason": "candidate_has_no_sandbox_evidence",
                }
            )
        elif status == "sandbox_verified":
            evidence = _latest_sandbox_evidence(candidate)
            score = _optional_float(evidence.get("best_score"))
            delta = None if score is None else round(score - current_best_score, 6)
            should_promote = delta is not None and delta >= 0
            action.update(
                {
                    "recommended_action": (
                        "promote_strategy" if should_promote else "refine_or_ablate"
                    ),
                    "next_command": (
                        "record-strategy-proposal"
                        if should_promote
                        else "run-literature-strategy-expansion"
                    ),
                    "priority": 90 if should_promote else 80,
                    "reason": (
                        "sandbox_score_meets_or_exceeds_current_best"
                        if should_promote
                        else "sandbox_score_below_current_best"
                    ),
                    "best_score": score,
                    "score_delta_vs_current_best": delta,
                    "rules_passed": evidence.get("rules_passed"),
                    "evidence_artifact_path": evidence.get("artifact_path"),
                }
            )
        elif status == "blocked":
            failure_codes = (
                _string_list(failure.get("failure_codes")) if failure else []
            )
            data_contract_blocked = "data_contract_incompatible" in failure_codes
            action.update(
                {
                    "recommended_action": (
                        "provide_compatible_reference_data"
                        if data_contract_blocked
                        else "design_executor"
                    ),
                    "next_command": (
                        "prepare-compatible-reference-data"
                        if data_contract_blocked
                        else "create-executor-design-request"
                    ),
                    "priority": 70 if data_contract_blocked else 60,
                    "reason": (
                        "candidate_blocked_by_data_contract"
                        if data_contract_blocked
                        else "candidate_blocked_by_executor_gap"
                    ),
                    "failure_codes": failure_codes,
                    "recommended_next": failure.get("recommended_next")
                    if failure
                    else None,
                }
            )
        elif status == "promoted":
            promotion_event = _latest_candidate_event(
                registry=registry,
                candidate_id=candidate_id,
                event_type="candidate_promoted",
            )
            candidate_promotion_path = _candidate_promotion_path_from_event(
                promotion_event
            )
            action.update(
                {
                    "recommended_action": "record_strategy_proposal",
                    "next_command": "record-promoted-algorithm-strategy-proposal",
                    "priority": 85,
                    "reason": "candidate_promotion_recorded",
                    "candidate_promotion_path": candidate_promotion_path,
                    "registry_event_path": promotion_event.get("artifact_path"),
                }
            )
        elif status == "sandbox_failed":
            action.update(
                {
                    "recommended_action": "deprioritize",
                    "next_command": "review-sandbox-failure",
                    "priority": 20,
                    "reason": "candidate_failed_sandbox_rules",
                }
            )
        else:
            action.update(
                {
                    "recommended_action": "review",
                    "next_command": "review-literature-registry",
                    "priority": 10,
                    "reason": f"unhandled_candidate_status:{status}",
                }
            )
        actions.append(action)
    return sorted(
        actions,
        key=lambda item: (
            -int(item["priority"]),
            str(item["candidate_id"]),
        ),
    )


def _registry_learning_signals(
    actions: list[dict[str, Any]],
    *,
    current_best_score: float,
) -> list[dict[str, Any]]:
    signals = []
    for action in actions:
        if action.get("recommended_action") == "refine_or_ablate":
            signals.append(
                {
                    "signal": "sandbox_underperformed_current_best",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "best_score": action.get("best_score"),
                    "current_best_score": current_best_score,
                    "score_delta_vs_current_best": action.get(
                        "score_delta_vs_current_best"
                    ),
                    "recommended_learning_objective": (
                        "search_variant_or_ablation_before_retry"
                    ),
                }
            )
        elif action.get("recommended_action") == "generate_variant_or_ablation":
            signals.append(
                {
                    "signal": "local_strategy_variant_required",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "failure_codes": action.get("failure_codes", []),
                    "variant_constraints": action.get("variant_constraints", {}),
                    "recommended_learning_objective": (
                        "generate_mutated_strategy_with_new_ablation_axis"
                    ),
                }
            )
        elif action.get("recommended_action") == "refine_with_legal_reference":
            signals.append(
                {
                    "signal": "target_reference_signal_requires_legal_retest",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "best_score": action.get("best_score"),
                    "current_best_score": current_best_score,
                    "score_delta_vs_current_best": action.get(
                        "score_delta_vs_current_best"
                    ),
                    "candidate_gate_issues": action.get("candidate_gate_issues", []),
                    "recommended_learning_objective": (
                        "legal_reference_retest_before_promotion"
                    ),
                }
            )
        elif action.get("recommended_action") == "refine_with_ablation_or_variant":
            signals.append(
                {
                    "signal": "candidate_refinement_required",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "best_score": action.get("best_score"),
                    "current_best_score": current_best_score,
                    "score_delta_vs_current_best": action.get(
                        "score_delta_vs_current_best"
                    ),
                    "failure_codes": action.get("failure_codes", []),
                    "recommended_learning_objective": (
                        "add_ablation_or_variant_before_promotion"
                    ),
                }
            )
        elif action.get("recommended_action") == "provide_compatible_reference_data":
            signals.append(
                {
                    "signal": "data_contract_blocks_candidate",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "failure_codes": action.get("failure_codes", []),
                    "recommended_learning_objective": (
                        "provide_full_horizon_reference_or_skip_until_available"
                    ),
                }
            )
        elif action.get("recommended_action") == "design_executor":
            signals.append(
                {
                    "signal": "executor_gap_blocks_algorithm_family",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "recommended_learning_objective": "build_bounded_executor",
                }
            )
        elif action.get("reason") == "candidate_failure_history_blocks_retry":
            signals.append(
                {
                    "signal": "candidate_failure_recorded",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "failure_codes": action.get("failure_codes", []),
                    "recommended_learning_objective": (
                        "avoid_repetition_until_new_evidence_or_variant"
                    ),
                }
            )
        elif action.get("recommended_action") == "run_sandbox":
            signals.append(
                {
                    "signal": "candidate_needs_first_sandbox_evidence",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "recommended_learning_objective": "measure_before_promotion",
                }
            )
    priority = {
        "target_reference_signal_requires_legal_retest": 110,
        "sandbox_underperformed_current_best": 100,
        "local_strategy_variant_required": 95,
        "candidate_refinement_required": 92,
        "candidate_failure_recorded": 90,
        "data_contract_blocks_candidate": 85,
        "executor_gap_blocks_algorithm_family": 80,
        "candidate_needs_first_sandbox_evidence": 60,
    }
    return sorted(
        signals,
        key=lambda item: (
            -priority.get(str(item.get("signal")), 0),
            str(item.get("candidate_id")),
        ),
    )


def _registry_family_summary(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        family = str(action.get("candidate_family") or "unknown_family")
        grouped.setdefault(family, []).append(action)
    summaries = []
    for family, family_actions in grouped.items():
        status_counts: dict[str, int] = {}
        best_score = None
        best_candidate_id = None
        for action in family_actions:
            status = str(action.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            score = action.get("best_score")
            if isinstance(score, int | float) and not isinstance(score, bool):
                if best_score is None or float(score) > best_score:
                    best_score = float(score)
                    best_candidate_id = str(action["candidate_id"])
        top_action = sorted(
            family_actions,
            key=lambda item: (
                -int(item.get("priority", 0)),
                str(item.get("candidate_id")),
            ),
        )[0]
        summaries.append(
            {
                "candidate_family": family,
                "candidate_count": len(family_actions),
                "status_counts": dict(sorted(status_counts.items())),
                "best_score": best_score,
                "best_candidate_id": best_candidate_id,
                "recommended_action": top_action.get("recommended_action"),
                "next_command": top_action.get("next_command"),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -int(_family_priority(str(item.get("recommended_action")))),
            str(item["candidate_family"]),
        ),
    )


def _family_priority(recommended_action: str) -> int:
    return {
        "run_sandbox": 100,
        "promote_strategy": 90,
        "refine_or_ablate": 80,
        "generate_variant_or_ablation": 75,
        "design_executor": 60,
        "deprioritize": 20,
    }.get(recommended_action, 10)


def _registry_algorithm_by_candidate(
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {}
    for algorithm in registry.get("algorithm_nodes", []):
        if not isinstance(algorithm, dict):
            continue
        for candidate_id in algorithm.get("candidate_ids", []):
            if isinstance(candidate_id, str):
                result[candidate_id] = algorithm
    return result


def _latest_sandbox_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("sandbox_evidence")
    if not isinstance(evidence, list) or not evidence:
        return {}
    latest = evidence[-1]
    return latest if isinstance(latest, dict) else {}


def _latest_failure_history(candidate: dict[str, Any]) -> dict[str, Any]:
    history = candidate.get("failure_history")
    if not isinstance(history, list) or not history:
        return {}
    latest = history[-1]
    return latest if isinstance(latest, dict) else {}


def _latest_candidate_event(
    *,
    registry: dict[str, Any],
    candidate_id: str,
    event_type: str,
) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for event in _object_items(registry.get("events")):
        if (
            event.get("algorithm_candidate_id") == candidate_id
            and event.get("event_type") == event_type
        ):
            latest = event
    return latest


def _candidate_promotion_path_from_event(event: dict[str, Any]) -> str | None:
    value = event.get("event_source_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    artifact_path = event.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None
    artifact_file = Path(artifact_path)
    if not artifact_file.is_file():
        return artifact_path.strip()
    try:
        payload = json.loads(artifact_file.read_text(encoding="utf-8"))
    except Exception:
        return artifact_path.strip()
    value = payload.get("event_source_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return artifact_path.strip()


def _failure_has_code(failure: dict[str, Any], code: str) -> bool:
    codes = failure.get("failure_codes")
    return isinstance(codes, list) and code in codes


def _failure_requires_local_strategy_variant(
    *,
    candidate: dict[str, Any],
    failure: dict[str, Any],
) -> bool:
    if candidate.get("candidate_family") != "local_strategy_skill":
        return False
    if failure.get("retry_allowed") is not False:
        return False
    codes = set(_string_list(failure.get("failure_codes")))
    return bool(codes.intersection({"score_regressed", "ablation_missing_or_weak"}))


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _summary_next_command(
    actions: list[dict[str, Any]],
    source_backlog: list[dict[str, Any]],
) -> str:
    if actions:
        command = actions[0].get("next_command")
        if isinstance(command, str) and command.strip():
            return command
    if source_backlog:
        return str(source_backlog[0]["next_command"])
    return "run-literature-strategy-expansion"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureRegistryPolicyError(f"missing {label}")
    return value.strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _object_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
