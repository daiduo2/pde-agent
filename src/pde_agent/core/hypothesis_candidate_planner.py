"""Trace-driven candidate-family planner for Hermes heartbeat runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_heartbeat import get_candidate_generator_registry


STATE_HINT = "HYPOTHESIS_CANDIDATE_PLAN_BUILT"


class HypothesisCandidatePlannerError(RuntimeError):
    pass


def build_hypothesis_candidate_plan(
    *,
    output_path: str | Path | None,
    candidate_generator: str,
    doom_loop_report_path: str | Path | None,
    hypothesis_kpi_report_path: str | Path | None,
    repair_policy_summary_path: str | Path | None = None,
    research_chronicle_path: str | Path | None = None,
    expansion_handoff_report_path: str | Path | None = None,
    audit_repair_tasks_path: str | Path | None = None,
    preferred_error_modes: list[str] | None = None,
    guard_override_candidate_generators: list[str] | None = None,
    max_schedule_length: int = 5,
    min_viable_candidate_count: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    if max_schedule_length < 1:
        raise HypothesisCandidatePlannerError("max_schedule_length must be positive")
    if min_viable_candidate_count < 1:
        raise HypothesisCandidatePlannerError(
            "min_viable_candidate_count must be positive"
        )
    output_file = Path(output_path) if output_path is not None else None
    if output_file is not None and output_file.exists() and not force:
        raise HypothesisCandidatePlannerError(
            f"candidate plan already exists: {output_file}"
        )
    guard = _load_optional_json(
        doom_loop_report_path,
        expected_state_hint="HYPOTHESIS_DOOM_LOOP_CHECKED",
        label="doom_loop_report",
    )
    kpi = _load_optional_json(
        hypothesis_kpi_report_path,
        expected_state_hint="HYPOTHESIS_KPI_REPORT_BUILT",
        label="hypothesis_kpi_report",
    )
    repair_policy = _load_optional_json(
        repair_policy_summary_path,
        expected_state_hint="HERMES_REPAIR_POLICY_SUMMARY_BUILT",
        label="repair_policy_summary",
    )
    research_chronicle = _load_optional_research_chronicle(
        research_chronicle_path
    )
    expansion_handoff = _load_optional_json(
        expansion_handoff_report_path,
        expected_state_hint="HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
        label="expansion_handoff_report",
    )
    audit_repair = _load_optional_json(
        audit_repair_tasks_path,
        expected_state_hint="HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED",
        label="audit_repair_tasks",
    )
    blocked = set(_blocked_candidate_families(guard))
    learning_constraints = _learning_constraints(kpi)
    audit_repair_tasks = _merge_repair_tasks(
        _audit_repair_tasks(audit_repair),
        _active_audit_repair_tasks(kpi),
    )
    audit_repair_policies = _audit_repair_policies(audit_repair_tasks)
    learning_constraints = _merge_audit_repair_constraints(
        learning_constraints,
        audit_repair_tasks=audit_repair_tasks,
        audit_repair_policies=audit_repair_policies,
    )
    prior_expansion_exhaustion = _prior_expansion_exhaustion(expansion_handoff)
    learning_constraints = _merge_expansion_exhaustion_constraints(
        learning_constraints,
        prior_expansion_exhaustion=prior_expansion_exhaustion,
    )
    cluster_saturation_signals = _merge_cluster_saturation_signals(
        _kpi_cluster_saturation_signals(kpi),
        _cluster_saturation_signals(prior_expansion_exhaustion),
    )
    learning_constraints = _merge_cluster_saturation_constraints(
        learning_constraints,
        cluster_saturation_signals=cluster_saturation_signals,
    )
    learning_constraints = _merge_repair_policy_summary_constraints(
        learning_constraints,
        repair_policy_summary=repair_policy,
    )
    learning_constraints = _merge_research_chronicle_repair_memory_constraints(
        learning_constraints,
        research_chronicle=research_chronicle,
    )
    learned_avoid = set(learning_constraints.get("avoid_candidate_families", []))
    saturated = set(
        item
        for item in learning_constraints.get("saturated_candidate_families", [])
        if isinstance(item, str) and item
    )
    guard_overrides = {
        item
        for item in guard_override_candidate_generators or []
        if isinstance(item, str) and item
    }
    registry = get_candidate_generator_registry()
    scored: list[dict[str, Any]] = []
    rationale: list[dict[str, Any]] = []
    for index, (name, metadata) in enumerate(registry.items()):
        directive_guard_override = (
            name == candidate_generator and name in guard_overrides
        )
        signals = _signals_for_candidate(
            name=name,
            metadata=metadata,
            kpi=kpi,
            preferred_error_modes=preferred_error_modes or [],
            audit_repair_policies=audit_repair_policies,
        )
        reason = _exclusion_reason(
            name=name,
            metadata=metadata,
            requested_candidate_generator=candidate_generator,
            blocked=blocked,
            learned_avoid=learned_avoid,
            saturated=saturated,
            directive_guard_override=directive_guard_override,
        )
        if reason is not None:
            rationale.append(
                {
                    "candidate_generator": name,
                    "decision": "excluded",
                    "reason": reason,
                    "signals": signals,
                }
            )
            continue
        score = _ranking_score(
            name=name,
            metadata=metadata,
            signals=signals,
        )
        scored.append(
            {
                "candidate_generator": name,
                "score": score,
                "registry_index": index,
                "signals": signals,
                "directive_guard_override": directive_guard_override,
            }
        )
    scored.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["registry_index"]),
            str(item["candidate_generator"]),
        )
    )
    selected = scored[:max_schedule_length]
    selected_rationale = []
    for item in selected:
        selected_rationale.append(
            {
                "candidate_generator": item["candidate_generator"],
                "decision": "selected",
                "score": item["score"],
                "signals": item["signals"],
                "directive_guard_override": bool(
                    item.get("directive_guard_override")
                ),
            }
        )
    schedule = [str(item["candidate_generator"]) for item in selected]
    expansion_required = len(scored) < min_viable_candidate_count
    decision = "candidate_space_exhausted" if expansion_required else "schedule_candidates"
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "candidate_plan_path": str(output_file) if output_file is not None else None,
        "candidate_generator": candidate_generator,
        "candidate_generator_schedule": schedule,
        "blocked_candidate_families": sorted(blocked),
        "learning_constraints": learning_constraints,
        "audit_repair_tasks": audit_repair_tasks,
        "audit_repair_policies": audit_repair_policies,
        "prior_expansion_exhaustion": prior_expansion_exhaustion,
        "cluster_saturation_signals": cluster_saturation_signals,
        "preferred_error_modes": preferred_error_modes or [],
        "directive_guard_overrides": sorted(guard_overrides),
        "eligible_candidate_count": len(scored),
        "min_viable_candidate_count": min_viable_candidate_count,
        "decision": decision,
        "expansion_required": expansion_required,
        "exhaustion_reason": "eligible_candidate_count_below_minimum"
        if expansion_required
        else "",
        "selection_rationale": selected_rationale + rationale,
        "next_command": "run-hermes-literature-strategy-expansion"
        if expansion_required
        else "run-hermes-heartbeat",
    }
    if expansion_required:
        report["expansion_handoff"] = _expansion_handoff(
            learning_constraints=learning_constraints
        )
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _load_optional_json(
    path: str | Path | None,
    *,
    expected_state_hint: str,
    label: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    file = Path(path)
    if not file.exists():
        raise HypothesisCandidatePlannerError(f"{label} does not exist: {file}")
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HypothesisCandidatePlannerError(f"{label} invalid JSON: {file}") from exc
    if not isinstance(payload, dict):
        raise HypothesisCandidatePlannerError(f"{label} must be a JSON object")
    if payload.get("state_hint") != expected_state_hint:
        raise HypothesisCandidatePlannerError(f"{label} has invalid state_hint")
    return payload


def _load_optional_research_chronicle(
    path: str | Path | None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    file = Path(path)
    if not file.exists():
        raise HypothesisCandidatePlannerError(
            f"research_chronicle does not exist: {file}"
        )
    text = file.read_text(encoding="utf-8")
    if "Hermes Research Chronicle" not in text:
        raise HypothesisCandidatePlannerError(
            "research_chronicle is missing Hermes Research Chronicle marker"
        )
    return {
        "path": str(file),
        "repair_memory": _parse_research_chronicle_repair_memory(text),
    }


def _blocked_candidate_families(guard: dict[str, Any] | None) -> list[str]:
    if not isinstance(guard, dict) or guard.get("guard_active") is not True:
        return []
    blocked = guard.get("blocked_candidate_families")
    if not isinstance(blocked, list):
        return []
    return [item for item in blocked if isinstance(item, str) and item]


def _learning_constraints(kpi: dict[str, Any] | None) -> dict[str, Any]:
    node_scoped_candidates = _node_scoped_deprioritized_candidates(kpi)
    node_scoped_families = {
        item["candidate_family"]
        for item in node_scoped_candidates
        if isinstance(item.get("candidate_family"), str)
    }
    deprioritized = [
        family
        for family in _deprioritized_candidate_families(kpi)
        if family not in node_scoped_families
    ]
    reason_codes = _deprioritization_reason_codes(
        kpi=kpi,
        deprioritized=deprioritized,
    )
    candidate_ids = [
        str(item["algorithm_candidate_id"])
        for item in node_scoped_candidates
        if isinstance(item.get("algorithm_candidate_id"), str)
        and item["algorithm_candidate_id"]
    ]
    candidate_reason_codes = _deprioritization_reason_codes_by_candidate(
        kpi,
        candidate_ids=candidate_ids,
    )
    constraints = {
        "avoid_candidate_families": deprioritized,
        "mutation_required_for_families": deprioritized,
        "reason_codes": reason_codes,
    }
    saturated = _saturated_candidate_families(kpi)
    if saturated:
        constraints["saturated_candidate_families"] = saturated
    search_guidance = _persistent_search_guidance(kpi)
    if search_guidance:
        constraints["persistent_search_guidance"] = search_guidance
        actions = _search_guidance_action_counts(kpi)
        if actions:
            constraints["search_guidance_action_counts"] = actions
            for action in actions:
                _append_unique(constraints["reason_codes"], action)
    nonconvergence_errors = _repair_nonconvergence_errors(kpi)
    if nonconvergence_errors:
        constraints["repair_nonconvergence_errors"] = nonconvergence_errors
        _append_unique(constraints["reason_codes"], "audit_repair_nonconvergence")
    convergence_errors = _repair_convergence_errors(kpi)
    if convergence_errors:
        constraints["repair_convergence_errors"] = convergence_errors
        _append_unique(constraints["reason_codes"], "audit_repair_convergence")
        converged_counts = _audit_repair_converged_error_counts(kpi)
        if converged_counts:
            constraints["audit_repair_converged_error_counts"] = converged_counts
    if candidate_ids:
        constraints["avoid_algorithm_candidate_ids"] = candidate_ids
        constraints["mutation_required_for_algorithm_candidate_ids"] = candidate_ids
        constraints["reason_codes_by_algorithm_candidate_id"] = candidate_reason_codes
    return constraints


def _prior_expansion_exhaustion(handoff: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(handoff, dict):
        return {"exhausted": False}
    exhaustion = handoff.get("expansion_exhaustion")
    if not isinstance(exhaustion, dict) or exhaustion.get("exhausted") is not True:
        return {"exhausted": False}
    families = [
        item
        for item in exhaustion.get("exhausted_candidate_families", [])
        if isinstance(item, str) and item
    ]
    attempted = [
        item
        for item in exhaustion.get("attempted_mutation_types", [])
        if isinstance(item, str) and item
    ]
    report = {
        "exhausted": True,
        "reason": str(exhaustion.get("reason") or "candidate_expansion_exhausted"),
        "exhausted_candidate_families": families,
        "attempted_mutation_types": attempted,
        "recommended_next_command": str(
            exhaustion.get("recommended_next_command") or "plan-hypothesis-candidates"
        ),
        "recommended_learning_objective": str(
            exhaustion.get("recommended_learning_objective")
            or "switch_candidate_family_after_axis_exhaustion"
        ),
    }
    candidate_ids = [
        item
        for item in exhaustion.get("exhausted_algorithm_candidate_ids", [])
        if isinstance(item, str) and item
    ]
    node_ids = [
        item
        for item in exhaustion.get("exhausted_algorithm_node_ids", [])
        if isinstance(item, str) and item
    ]
    if candidate_ids:
        report["exhausted_algorithm_candidate_ids"] = candidate_ids
    if node_ids:
        report["exhausted_algorithm_node_ids"] = node_ids
    return report


def _audit_repair_tasks(audit_repair: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(audit_repair, dict):
        return []
    values = audit_repair.get("repair_tasks")
    if not isinstance(values, list):
        return []
    tasks = []
    for item in values:
        if not isinstance(item, dict):
            continue
        audit_error = item.get("audit_error")
        task_id = item.get("task_id")
        if not isinstance(audit_error, str) or not audit_error:
            continue
        if not isinstance(task_id, str) or not task_id:
            continue
        tasks.append(dict(item))
    return tasks


def _active_audit_repair_tasks(kpi: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("active_audit_repair_errors")
    if not isinstance(values, list):
        return []
    nonconverged_errors = set(_repair_nonconvergence_errors(kpi))
    converged_errors = set(_repair_convergence_errors(kpi))
    tasks = []
    for audit_error in values:
        if not isinstance(audit_error, str) or not audit_error:
            continue
        if audit_error in converged_errors and audit_error not in nonconverged_errors:
            continue
        task = {
            "task_id": f"kpi_active_repair_{audit_error}",
            "audit_error": audit_error,
            "priority": "high",
            "repair_scope": "hypothesis_kpi_active_audit_repair",
            "recommended_action": "consume_active_audit_repair_before_next_runtime",
            "next_command": "run-hermes-runtime-closed-loop",
            "source": "hypothesis_kpi_report",
        }
        if audit_error in nonconverged_errors:
            task["repair_nonconverged"] = True
            task["recommended_action"] = (
                "switch_repair_strategy_after_nonconverged_audit_repair"
            )
        tasks.append(task)
    return tasks


def _merge_repair_tasks(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(task) for task in primary]
    seen_errors = {
        task.get("audit_error")
        for task in merged
        if isinstance(task.get("audit_error"), str)
    }
    for task in fallback:
        audit_error = task.get("audit_error")
        if not isinstance(audit_error, str) or not audit_error:
            continue
        if audit_error in seen_errors:
            continue
        merged.append(dict(task))
        seen_errors.add(audit_error)
    return merged


def _merge_audit_repair_constraints(
    constraints: dict[str, Any],
    *,
    audit_repair_tasks: list[dict[str, Any]],
    audit_repair_policies: list[dict[str, Any]],
) -> dict[str, Any]:
    if not audit_repair_tasks:
        return constraints
    merged = {
        "avoid_candidate_families": list(
            constraints.get("avoid_candidate_families", [])
        ),
        "mutation_required_for_families": list(
            constraints.get("mutation_required_for_families", [])
        ),
        "reason_codes": list(constraints.get("reason_codes", [])),
    }
    for key, value in constraints.items():
        if key not in merged:
            merged[key] = value
    merged["audit_repair_required"] = True
    merged["audit_repair_tasks"] = [dict(task) for task in audit_repair_tasks]
    if audit_repair_policies:
        merged["audit_repair_policies"] = [
            dict(policy) for policy in audit_repair_policies
        ]
    policy_errors = {
        policy.get("audit_error")
        for policy in audit_repair_policies
        if isinstance(policy.get("audit_error"), str)
    }
    generic_errors = []
    for task in audit_repair_tasks:
        audit_error = task.get("audit_error")
        if isinstance(audit_error, str) and audit_error:
            _append_unique(merged["reason_codes"], f"audit_repair:{audit_error}")
            if audit_error not in policy_errors:
                _append_unique(generic_errors, audit_error)
    if generic_errors:
        merged["generic_audit_repair_errors"] = generic_errors
    return merged


def _audit_repair_policies(
    audit_repair_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for task in audit_repair_tasks:
        audit_error = task.get("audit_error")
        repair_nonconverged = task.get("repair_nonconverged") is True
        if audit_error == "cluster_saturation_consistency" and repair_nonconverged:
            policies.append(
                {
                    "audit_error": "cluster_saturation_consistency",
                    "preferred_candidate_generators": ["physics_residual_nudge"],
                    "avoid_candidate_generators": [
                        "fast_baseline_family",
                        "sem_window_gate_blend_multi_early_smoke",
                        "sem_window_gate_blend",
                        "sem_window_blend",
                    ],
                    "repair_nonconverged": True,
                }
            )
        elif (
            audit_error == "persistent_search_guidance_consistency"
            and repair_nonconverged
        ):
            policies.append(
                {
                    "audit_error": "persistent_search_guidance_consistency",
                    "preferred_candidate_generators": ["fast_baseline_family"],
                    "avoid_candidate_generators": [
                        "physics_residual_nudge",
                        "sem_window_gate_blend_multi_early_smoke",
                        "sem_window_gate_blend",
                        "sem_window_blend",
                    ],
                    "repair_nonconverged": True,
                }
            )
        elif audit_error == "memory_bridge_inputs" and repair_nonconverged:
            policies.append(
                {
                    "audit_error": "memory_bridge_inputs",
                    "preferred_candidate_generators": ["fast_baseline_family"],
                    "avoid_candidate_generators": ["postprocess"],
                    "requires_input_bridge_reload": True,
                    "repair_nonconverged": True,
                }
            )
        elif audit_error == "repair_nonconvergence_consistency":
            policies.append(
                {
                    "audit_error": "repair_nonconvergence_consistency",
                    "preferred_candidate_generators": ["fast_baseline_family"],
                    "avoid_candidate_generators": [
                        "sem_window_gate_blend_multi_early_smoke",
                        "sem_window_gate_blend",
                        "sem_window_blend",
                    ],
                    "requires_repair_convergence_reload": True,
                }
            )
        elif audit_error == "repair_convergence_consistency":
            policies.append(
                {
                    "audit_error": "repair_convergence_consistency",
                    "preferred_candidate_generators": ["postprocess"],
                    "avoid_candidate_generators": [
                        "sem_window_gate_blend_multi_early_smoke",
                        "sem_window_gate_blend",
                        "sem_window_blend",
                    ],
                    "requires_repair_convergence_reload": True,
                }
            )
        elif audit_error == "heartbeat_saturation_policy":
            policies.append(
                {
                    "audit_error": "heartbeat_saturation_policy",
                    "preferred_candidate_generators": ["physics_residual_nudge"],
                    "avoid_candidate_generators": [
                        "sem_window_gate_blend_multi_early_smoke",
                        "sem_window_gate_blend",
                        "sem_window_blend",
                    ],
                }
            )
        elif audit_error == "persistent_search_guidance_consistency":
            policies.append(
                {
                    "audit_error": "persistent_search_guidance_consistency",
                    "preferred_candidate_generators": ["physics_residual_nudge"],
                    "avoid_candidate_generators": [
                        "sem_window_gate_blend_multi_early_smoke",
                        "sem_window_gate_blend",
                        "sem_window_blend",
                    ],
                }
            )
        elif audit_error == "cluster_saturation_consistency":
            policies.append(
                {
                    "audit_error": "cluster_saturation_consistency",
                    "preferred_candidate_generators": ["fast_baseline_family"],
                    "avoid_candidate_generators": [
                        "sem_window_gate_blend_multi_early_smoke",
                        "sem_window_gate_blend",
                        "sem_window_blend",
                    ],
                }
            )
        elif audit_error == "memory_bridge_inputs":
            policies.append(
                {
                    "audit_error": "memory_bridge_inputs",
                    "preferred_candidate_generators": ["postprocess"],
                    "requires_input_bridge_reload": True,
                }
            )
        elif audit_error == "learning_constraint_smoke_gate":
            policies.append(
                {
                    "audit_error": "learning_constraint_smoke_gate",
                    "preferred_candidate_generators": ["postprocess"],
                    "requires_learning_constraint_smoke_reload": True,
                }
            )
    return policies


def _merge_expansion_exhaustion_constraints(
    constraints: dict[str, Any],
    *,
    prior_expansion_exhaustion: dict[str, Any],
) -> dict[str, Any]:
    if prior_expansion_exhaustion.get("exhausted") is not True:
        return constraints
    merged = {
        "avoid_candidate_families": list(
            constraints.get("avoid_candidate_families", [])
        ),
        "mutation_required_for_families": list(
            constraints.get("mutation_required_for_families", [])
        ),
        "reason_codes": list(constraints.get("reason_codes", [])),
    }
    for optional_key in [
        "avoid_algorithm_candidate_ids",
        "mutation_required_for_algorithm_candidate_ids",
    ]:
        values = [
            item
            for item in constraints.get(optional_key, [])
            if isinstance(item, str) and item
        ]
        if values:
            merged[optional_key] = values
    for preserved_key in [
        "audit_repair_required",
        "audit_repair_tasks",
        "audit_repair_policies",
        "persistent_search_guidance",
        "search_guidance_action_counts",
        "saturated_candidate_families",
        "repair_convergence_errors",
        "audit_repair_converged_error_counts",
        "repair_nonconvergence_errors",
    ]:
        if preserved_key in constraints:
            merged[preserved_key] = constraints[preserved_key]
    by_candidate = constraints.get("reason_codes_by_algorithm_candidate_id")
    if isinstance(by_candidate, dict):
        cleaned = {
            str(candidate_id): [
                str(reason)
                for reason in reasons
                if isinstance(reason, str) and reason
            ]
            for candidate_id, reasons in by_candidate.items()
            if isinstance(candidate_id, str) and isinstance(reasons, list)
        }
        if cleaned:
            merged["reason_codes_by_algorithm_candidate_id"] = cleaned
    for family in prior_expansion_exhaustion.get("exhausted_candidate_families", []):
        if isinstance(family, str) and family:
            _append_unique(merged["avoid_candidate_families"], family)
            _append_unique(merged["mutation_required_for_families"], family)
    exhausted_ids = [
        item
        for item in prior_expansion_exhaustion.get(
            "exhausted_algorithm_candidate_ids",
            [],
        )
        if isinstance(item, str) and item
    ]
    if exhausted_ids:
        avoid_ids = list(merged.get("avoid_algorithm_candidate_ids", []))
        mutation_required_ids = [
            item
            for item in merged.get("mutation_required_for_algorithm_candidate_ids", [])
            if item not in exhausted_ids
        ]
        reason_by_candidate = dict(
            merged.get("reason_codes_by_algorithm_candidate_id", {})
        )
        for candidate_id in exhausted_ids:
            _append_unique(avoid_ids, candidate_id)
            reasons = list(reason_by_candidate.get(candidate_id, []))
            reason = prior_expansion_exhaustion.get("reason")
            if isinstance(reason, str) and reason:
                _append_unique(reasons, reason)
            reason_by_candidate[candidate_id] = reasons
        merged["avoid_algorithm_candidate_ids"] = avoid_ids
        if mutation_required_ids:
            merged[
                "mutation_required_for_algorithm_candidate_ids"
            ] = mutation_required_ids
        else:
            merged.pop("mutation_required_for_algorithm_candidate_ids", None)
        merged["reason_codes_by_algorithm_candidate_id"] = reason_by_candidate
    reason = prior_expansion_exhaustion.get("reason")
    if isinstance(reason, str) and reason:
        _append_unique(merged["reason_codes"], reason)
    return merged


def _cluster_saturation_signals(
    prior_expansion_exhaustion: dict[str, Any],
) -> list[dict[str, Any]]:
    if prior_expansion_exhaustion.get("exhausted") is not True:
        return []
    node_ids = [
        item
        for item in prior_expansion_exhaustion.get("exhausted_algorithm_node_ids", [])
        if isinstance(item, str) and item.startswith("local_strategy_")
    ]
    if len(node_ids) < 2:
        return []
    return [
        {
            "signal": "local_strategy_cluster_saturated",
            "candidate_family": "local_strategy_skill",
            "exhausted_node_count": len(node_ids),
            "exhausted_algorithm_node_ids": node_ids,
            "recommended_policy": "prefer_non_local_strategy_or_untried_sources",
        }
    ]


def _merge_cluster_saturation_constraints(
    constraints: dict[str, Any],
    *,
    cluster_saturation_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    if not cluster_saturation_signals:
        return constraints
    merged = dict(constraints)
    saturated = [
        item
        for item in merged.get("saturated_candidate_families", [])
        if isinstance(item, str) and item
    ]
    for signal in cluster_saturation_signals:
        family = signal.get("candidate_family")
        if isinstance(family, str) and family:
            _append_unique(saturated, family)
    if saturated:
        merged["saturated_candidate_families"] = saturated
    return merged


def _merge_repair_policy_summary_constraints(
    constraints: dict[str, Any],
    *,
    repair_policy_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(repair_policy_summary, dict):
        return constraints
    merged = dict(constraints)
    reason_codes = list(merged.get("reason_codes", []))
    converged_errors = _string_items(
        repair_policy_summary.get("converged_repair_errors")
    )
    nonconverged_errors = _string_items(
        repair_policy_summary.get("nonconverged_repair_errors")
    )
    if converged_errors:
        merged["repair_convergence_errors"] = _merge_string_lists(
            merged.get("repair_convergence_errors", []),
            converged_errors,
        )
        _append_unique(reason_codes, "repair_policy_summary")
        _append_unique(reason_codes, "audit_repair_convergence")
    if nonconverged_errors:
        merged["repair_nonconvergence_errors"] = _merge_string_lists(
            merged.get("repair_nonconvergence_errors", []),
            nonconverged_errors,
        )
        _append_unique(reason_codes, "repair_policy_summary")
        _append_unique(reason_codes, "audit_repair_nonconvergence")
    stats = repair_policy_summary.get("repair_error_stats")
    if isinstance(stats, dict) and stats:
        merged["repair_policy_error_stats"] = {
            str(key): dict(value)
            for key, value in stats.items()
            if isinstance(key, str) and isinstance(value, dict)
        }
    actions = [
        dict(item)
        for item in repair_policy_summary.get("recommended_actions", [])
        if isinstance(item, dict)
    ]
    if actions:
        merged["repair_policy_recommended_actions"] = actions
    if reason_codes:
        merged["reason_codes"] = reason_codes
    return merged


def _merge_research_chronicle_repair_memory_constraints(
    constraints: dict[str, Any],
    *,
    research_chronicle: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(research_chronicle, dict):
        return constraints
    memory = research_chronicle.get("repair_memory")
    if not isinstance(memory, dict):
        return constraints
    merged = dict(constraints)
    reason_codes = list(merged.get("reason_codes", []))
    actions = [
        dict(item)
        for item in memory.get("repair_policy_recommended_actions", [])
        if isinstance(item, dict)
    ]
    nonconverged_errors = _string_items(
        memory.get("nonconverged_repair_errors")
    )
    converged_errors = _string_items(memory.get("converged_repair_errors"))
    if actions:
        merged["repair_policy_recommended_actions"] = _merge_action_dicts(
            merged.get("repair_policy_recommended_actions", []),
            actions,
        )
    if nonconverged_errors:
        merged["repair_nonconvergence_errors"] = _merge_string_lists(
            merged.get("repair_nonconvergence_errors", []),
            nonconverged_errors,
        )
    if converged_errors:
        merged["repair_convergence_errors"] = _merge_string_lists(
            merged.get("repair_convergence_errors", []),
            converged_errors,
        )
    if actions or nonconverged_errors or converged_errors:
        merged["research_chronicle_repair_policy_memory"] = {
            "path": str(research_chronicle.get("path", "")),
            "repair_action_count": len(actions),
            "nonconverged_repair_errors": nonconverged_errors,
            "converged_repair_errors": converged_errors,
        }
        _append_unique(reason_codes, "research_chronicle_repair_policy_memory")
    if reason_codes:
        merged["reason_codes"] = reason_codes
    return merged


def _merge_cluster_saturation_signals(
    *groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for group in groups:
        for signal in group:
            if signal not in merged:
                merged.append(signal)
    return merged


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _merge_string_lists(*groups: object) -> list[str]:
    values: list[str] = []
    for group in groups:
        for item in _string_items(group):
            _append_unique(values, item)
    return sorted(values)


def _merge_action_dicts(*groups: object) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            action = dict(item)
            key = (
                str(action.get("audit_error", "")),
                str(action.get("action", "")),
                str(action.get("reason", "")),
                str(action.get("source", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(action)
    return merged


def _parse_research_chronicle_repair_memory(text: str) -> dict[str, Any]:
    section = _extract_markdown_section(text, "Repair Policy Memory:")
    actions: list[dict[str, str]] = []
    nonconverged_errors: list[str] = []
    converged_errors: list[str] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        fields = _parse_inline_fields(line[2:])
        if line.startswith("- repair_action "):
            action = {
                key: fields[key]
                for key in ("audit_error", "action", "reason")
                if key in fields
            }
            if {"audit_error", "action", "reason"}.issubset(action):
                actions.append(action)
            continue
        nonconverged_errors = _merge_string_lists(
            nonconverged_errors,
            _comma_string_items(fields.get("nonconverged_repair_errors")),
        )
        converged_errors = _merge_string_lists(
            converged_errors,
            _comma_string_items(fields.get("converged_repair_errors")),
        )
    return {
        "repair_policy_recommended_actions": actions,
        "nonconverged_repair_errors": nonconverged_errors,
        "converged_repair_errors": converged_errors,
    }


def _extract_markdown_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return ""
    collected: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("- "):
            break
        collected.append(line)
    return "\n".join(collected)


def _parse_inline_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", maxsplit=1)
        if key and value:
            fields[key] = value
    return fields


def _comma_string_items(value: Any) -> list[str]:
    if not isinstance(value, str) or value in {"", "none"}:
        return []
    return [item for item in value.split(",") if item]


def _kpi_cluster_saturation_signals(kpi: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("cluster_saturation_signals")
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def _saturated_candidate_families(kpi: dict[str, Any] | None) -> list[str]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("saturated_candidate_families")
    if not isinstance(values, list):
        return []
    return sorted({item for item in values if isinstance(item, str) and item})


def _persistent_search_guidance(kpi: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("persistent_search_guidance")
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, dict)]


def _search_guidance_action_counts(kpi: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(kpi, dict):
        return {}
    values = kpi.get("search_guidance_action_counts")
    if not isinstance(values, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in values.items():
        if isinstance(key, str) and key and isinstance(value, int) and not isinstance(
            value, bool
        ):
            result[key] = value
    return result


def _repair_nonconvergence_errors(kpi: dict[str, Any] | None) -> list[str]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("repair_nonconvergence_errors")
    if not isinstance(values, list):
        return []
    return sorted({item for item in values if isinstance(item, str) and item})


def _repair_convergence_errors(kpi: dict[str, Any] | None) -> list[str]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("repair_convergence_errors")
    if not isinstance(values, list):
        return []
    return sorted({item for item in values if isinstance(item, str) and item})


def _audit_repair_converged_error_counts(
    kpi: dict[str, Any] | None,
) -> dict[str, int]:
    if not isinstance(kpi, dict):
        return {}
    values = kpi.get("audit_repair_converged_error_counts")
    if not isinstance(values, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in values.items():
        if (
            isinstance(key, str)
            and key
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            result[key] = value
    return result


def _deprioritized_candidate_families(kpi: dict[str, Any] | None) -> list[str]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("deprioritized_candidate_families")
    if not isinstance(values, list):
        return []
    return sorted({item for item in values if isinstance(item, str) and item})


def _node_scoped_deprioritized_candidates(
    kpi: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("deprioritized_algorithm_candidates")
    if not isinstance(values, list):
        return []
    node_scoped_families = {"local_strategy_skill", "official_checkpoint_family"}
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("algorithm_candidate_id")
        family = item.get("candidate_family")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if not isinstance(family, str) or family not in node_scoped_families:
            continue
        if family == "local_strategy_skill" and not candidate_id.startswith(
            "candidate_local_strategy_"
        ):
            continue
        if family == "official_checkpoint_family" and not candidate_id.startswith(
            "candidate_local_checkpoint_"
        ):
            continue
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidates.append(
            {
                "algorithm_candidate_id": candidate_id,
                "candidate_family": family,
            }
        )
    return sorted(candidates, key=lambda item: item["algorithm_candidate_id"])


def _deprioritization_reason_codes(
    *,
    kpi: dict[str, Any] | None,
    deprioritized: list[str],
) -> list[str]:
    if not deprioritized:
        return []
    if not isinstance(kpi, dict):
        return ["deprioritized_completion_trace"]
    values = kpi.get("deprioritization_reason_codes")
    reason_codes: set[str] = set()
    if isinstance(values, dict):
        for family in deprioritized:
            family_codes = values.get(family)
            if isinstance(family_codes, list):
                reason_codes.update(
                    item for item in family_codes if isinstance(item, str) and item
                )
    if not reason_codes:
        reason_codes.add("deprioritized_completion_trace")
    return sorted(reason_codes)


def _deprioritization_reason_codes_by_candidate(
    kpi: dict[str, Any] | None,
    *,
    candidate_ids: list[str],
) -> dict[str, list[str]]:
    if not candidate_ids:
        return {}
    values = (
        kpi.get("deprioritization_reason_codes_by_candidate")
        if isinstance(kpi, dict)
        else None
    )
    result: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        codes: list[str] = []
        if isinstance(values, dict):
            raw = values.get(candidate_id)
            if isinstance(raw, list):
                codes.extend(item for item in raw if isinstance(item, str) and item)
        if not codes:
            codes.append("deprioritized_completion_trace")
        result[candidate_id] = sorted(set(codes))
    return result


def _exclusion_reason(
    *,
    name: str,
    metadata: dict[str, Any],
    requested_candidate_generator: str,
    blocked: set[str],
    learned_avoid: set[str],
    saturated: set[str],
    directive_guard_override: bool = False,
) -> str | None:
    if (
        metadata.get("factory_handoff_required") is True
        and name != requested_candidate_generator
    ):
        return "requires_candidate_factory_handoff"
    if name in blocked and not directive_guard_override:
        return "blocked_by_doom_loop_guard"
    if name in learned_avoid:
        return "blocked_by_learning_constraint"
    if _blocked_by_cluster_saturation(metadata=metadata, saturated=saturated):
        return "blocked_by_cluster_saturation"
    if metadata.get("auto_fallback_allowed") is not True:
        return "not_auto_fallback_allowed"
    return None


def _blocked_by_cluster_saturation(
    *,
    metadata: dict[str, Any],
    saturated: set[str],
) -> bool:
    if not saturated:
        return False
    aliases = metadata.get("saturation_blocked_by")
    if not isinstance(aliases, list):
        return False
    return any(isinstance(alias, str) and alias in saturated for alias in aliases)


def _signals_for_candidate(
    *,
    name: str,
    metadata: dict[str, Any],
    kpi: dict[str, Any] | None,
    preferred_error_modes: list[str],
    audit_repair_policies: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _candidate_summary(kpi, name)
    best = _best_family(kpi)
    target_modes = [
        item for item in metadata.get("targets_error_modes", []) if isinstance(item, str)
    ]
    positive_kpi = _positive_summary(summary) or (
        isinstance(best, dict)
        and best.get("candidate_family") == name
        and _positive_best_family(best)
    )
    preferred_by_repair = any(
        name in _string_items(policy.get("preferred_candidate_generators"))
        for policy in audit_repair_policies
    )
    avoided_by_repair = any(
        name in _string_items(policy.get("avoid_candidate_generators"))
        for policy in audit_repair_policies
    )
    return {
        "cost_level": metadata.get("cost_level"),
        "positive_kpi": positive_kpi,
        "history_count": int(summary.get("count", 0)),
        "failed_count": int(summary.get("failed_count", 0)),
        "mean_score_delta": summary.get("mean_score_delta"),
        "best_score_delta": summary.get("best_score_delta"),
        "matches_preferred_error_mode": bool(
            set(target_modes).intersection(preferred_error_modes)
        ),
        "matches_audit_repair_policy": preferred_by_repair,
        "avoided_by_audit_repair_policy": avoided_by_repair,
        "targets_error_modes": target_modes,
    }


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _candidate_summary(kpi: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not isinstance(kpi, dict):
        return {}
    summaries = kpi.get("candidate_family_score_summary")
    if isinstance(summaries, dict) and isinstance(summaries.get(name), dict):
        return summaries[name]
    return {}


def _best_family(kpi: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(kpi, dict):
        return None
    best = kpi.get("best_family_by_score_delta")
    return best if isinstance(best, dict) else None


def _positive_summary(summary: dict[str, Any]) -> bool:
    return _positive_number(summary.get("best_score_delta")) or _positive_number(
        summary.get("mean_score_delta")
    )


def _positive_best_family(best: dict[str, Any]) -> bool:
    return _positive_number(best.get("best_score_delta")) or _positive_number(
        best.get("mean_score_delta")
    )


def _positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0.0


def _ranking_score(
    *,
    name: str,
    metadata: dict[str, Any],
    signals: dict[str, Any],
) -> float:
    score = 0.0
    if signals["positive_kpi"]:
        score += 100.0
    if signals.get("matches_audit_repair_policy") is True:
        score += 50.0
    if signals.get("avoided_by_audit_repair_policy") is True:
        score -= 50.0
    if signals["matches_preferred_error_mode"]:
        score += 10.0
    failed_count = signals.get("failed_count")
    history_count = signals.get("history_count")
    if isinstance(failed_count, int):
        score -= min(failed_count, 5) * 5.0
    if isinstance(history_count, int) and history_count and not signals["positive_kpi"]:
        score -= min(history_count, 5)
    return score


def _expansion_handoff(*, learning_constraints: dict[str, Any]) -> dict[str, Any]:
    return {
        "reason": "candidate_space_exhausted",
        "next_command": "build-local-asset-algorithm-candidates",
        "learning_constraints": learning_constraints,
        "recommended_inputs": {
            "local_strategy_library_path": "runs/local_strategy_library.json",
            "checkpoint_paths": [
                "data/pdebench/checkpoints/burgers_FNO.tar",
                "data/pdebench/checkpoints/burgers_Unet.tar",
            ],
            "max_strategy_candidates": 5,
        },
        "follow_up_command": "run-hermes-literature-strategy-expansion",
    }
