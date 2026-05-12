from __future__ import annotations

import json
from typing import Any

import numpy as np

from pde_agent.core.hypothesis_lab_utils import (
    EARLY_REFINEMENT_ANCHORS,
    HypothesisLabError,
    _clamp_float,
    _clamp_int,
    _clamp_task1_retrieval_k,
    _string_list_from_any,
)
from pde_agent.core.hypothesis_lab_evidence import (
    _normalize_evidence_params,
)


def _base_reference_metadata(
    *,
    base_family: str,
    reference: np.ndarray,
    reference_audit: dict[str, Any],
) -> dict[str, Any]:
    if base_family not in {"task1_retrieval_prior", "task1_retrieval_refinement"}:
        return {}
    return {
        "base_reference_path": reference_audit["reference_path"],
        "base_reference_count": int(reference.shape[0]),
        "base_uses_target_as_reference": reference_audit["uses_target_as_reference"],
        "base_submission_validity": reference_audit["submission_validity"],
    }



def _candidate_pool_attribution(
    *,
    rows: list[dict[str, Any]],
    top_candidate_count: int,
    family: str,
) -> dict[str, Any]:
    if family == "task1_early_refinement":
        axis = "early_anchor"
    elif family == "task1_early_transition_expert":
        axis = "transition_mode"
    elif family == "task1_front25_dynamics":
        axis = "dynamics_mode"
    elif family == "task1_short_window_corrector":
        axis = "corrector_mode"
    else:
        axis = "mode"
    top_ids = {
        str(row.get("candidate_id", ""))
        for row in rows[: max(0, int(top_candidate_count))]
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        params = row.get("params") if isinstance(row.get("params"), dict) else {}
        value = params.get(axis)
        if value is None:
            value = row.get(axis, "unknown")
        groups.setdefault(str(value), []).append(row)
    by_value = {}
    non_top_values = []
    for value, group in sorted(groups.items()):
        top_count = sum(
            1 for row in group if str(row.get("candidate_id", "")) in top_ids
        )
        best_row = min(
            group,
            key=lambda row: (
                _candidate_early_rel_mse(row),
                -float(row.get("segment_score", 0.0)),
            ),
        )
        if top_count == 0:
            non_top_values.append(value)
        by_value[value] = {
            "candidate_count": len(group),
            "top_candidate_count": top_count,
            "best_candidate_id": best_row.get("candidate_id"),
            "best_early_rel_mse": _finite_float_or_none(
                _candidate_early_rel_mse(best_row)
            ),
            "best_segment_score": _finite_float_or_none(
                float(best_row.get("segment_score", 0.0))
            ),
            "best_task1_total_score_proxy": _finite_float_or_none(
                float(best_row.get("task1_total_score_proxy", 0.0))
            ),
            "best_params": best_row.get("params")
            if isinstance(best_row.get("params"), dict)
            else {},
        }
    return {
        "axis": axis,
        "candidate_count": len(rows),
        "top_candidate_count": min(max(0, int(top_candidate_count)), len(rows)),
        f"by_{axis}": by_value,
        f"non_top_{axis}s": non_top_values,
        "non_top_anchors": non_top_values if axis == "early_anchor" else [],
        "search_space_policy": _candidate_pool_search_space_policy(
            axis=axis,
            by_value=by_value,
            non_top_values=non_top_values,
        ),
    }



def _candidate_pool_search_space_policy(
    *,
    axis: str,
    by_value: dict[str, dict[str, Any]],
    non_top_values: list[str],
) -> dict[str, Any]:
    finite = [
        (value, payload)
        for value, payload in by_value.items()
        if isinstance(payload.get("best_early_rel_mse"), int | float)
    ]
    if not finite:
        return {
            "axis": axis,
            "focus_values": [],
            "rescue_probe_values": [],
            "deprioritize_values": non_top_values,
            "recommended_actions": [
        _candidate_pool_action(
            action="deprioritize",
            axis=axis,
            value=value,
            reason="generated_but_absent_from_top_candidates",
            payload=by_value.get(value, {}),
                )
                for value in non_top_values
            ],
        }
    best_rel_mse = min(float(payload["best_early_rel_mse"]) for _value, payload in finite)
    rescue_threshold = best_rel_mse * 1.1
    if best_rel_mse <= 0.0:
        rescue_threshold = 1e-8
    focus_values = [
        value
        for value, payload in sorted(
            finite,
            key=lambda item: float(item[1]["best_early_rel_mse"]),
        )
        if int(payload.get("top_candidate_count", 0)) > 0
        and float(payload["best_early_rel_mse"]) <= rescue_threshold
    ]
    rescue_probe_values = [
        value
        for value in non_top_values
        if isinstance(by_value.get(value, {}).get("best_early_rel_mse"), int | float)
        and float(by_value[value]["best_early_rel_mse"]) <= rescue_threshold
    ]
    deprioritize_values = [
        value for value in non_top_values if value not in rescue_probe_values
    ]
    actions = [
        _candidate_pool_action(
            action="keep_focus",
            axis=axis,
            value=value,
            reason="top_candidate_near_best_early_rel_mse",
            payload=by_value[value],
        )
        for value in focus_values
    ]
    actions.extend(
        _candidate_pool_action(
            action="budget_probe",
            axis=axis,
            value=value,
            reason="non_top_near_best_early_rel_mse",
            payload=by_value[value],
        )
        for value in rescue_probe_values
    )
    actions.extend(
        _candidate_pool_action(
            action="deprioritize",
            axis=axis,
            value=value,
            reason="generated_but_not_competitive_in_candidate_pool",
            payload=by_value[value],
        )
        for value in deprioritize_values
    )
    return {
        "axis": axis,
        "focus_values": focus_values,
        "rescue_probe_values": rescue_probe_values,
        "deprioritize_values": deprioritize_values,
        "recommended_actions": actions,
    }



def _candidate_pool_action(
    *,
    action: str,
    axis: str,
    value: str,
    reason: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "action": action,
        "axis": axis,
        "value": value,
        "reason": reason,
        "candidate_count": int(payload.get("candidate_count", 0)),
        "top_candidate_count": int(payload.get("top_candidate_count", 0)),
        "best_early_rel_mse": payload.get("best_early_rel_mse"),
        "best_segment_score": payload.get("best_segment_score"),
    }



def _candidate_early_rel_mse(row: dict[str, Any]) -> float:
    segments = row.get("segments")
    if isinstance(segments, dict):
        early = segments.get("early")
        if isinstance(early, dict):
            value = early.get("rel_mse")
            if isinstance(value, int | float):
                return float(value)
    return float("inf")



def _finite_float_or_none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)



def _early_memory_policy(
    evidence_params: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    retry_directives: list[str] = []
    avoid_directives: list[str] = []
    effective_axes: list[str] = []
    avoid_exact_params: list[dict[str, Any]] = []
    promoted_params: list[dict[str, Any]] = []
    deprioritized_anchors: list[str] = []
    rescue_probe_anchors: list[str] = []
    for item in evidence_params or []:
        if not isinstance(item, dict):
            continue
        params = item.get("params")
        if not isinstance(params, dict):
            continue
        normalized = _normalize_evidence_params(params)
        if normalized is None:
            continue
        retry_directives.extend(_string_list_from_any(item.get("retry_directives")))
        avoid_directives.extend(_string_list_from_any(item.get("avoid_directives")))
        effective_axes.extend(_string_list_from_any(item.get("effective_axes")))
        pool_policy = _candidate_pool_policy_from_evidence(item)
        rescue_probe_anchors.extend(
            _string_list_from_any(pool_policy.get("rescue_probe_values"))
        )
        policy_deprioritized = _string_list_from_any(
            pool_policy.get("deprioritize_values")
        )
        deprioritized_anchors.extend(policy_deprioritized)
        for evidence in item.get("negative_candidate_evidence", []):
            if not isinstance(evidence, dict):
                continue
            if evidence.get("axis") == "early_anchor" and isinstance(
                evidence.get("value"),
                str,
            ) and not policy_deprioritized:
                deprioritized_anchors.append(str(evidence["value"]))
        outcome = str(item.get("outcome", ""))
        if outcome == "promoted":
            promoted_params.append(normalized)
        if _should_avoid_exact_memory_params(item):
            avoid_exact_params.append(normalized)
    return {
        "retry_directives": _dedupe_strings(retry_directives),
        "avoid_directives": _dedupe_strings(avoid_directives),
        "effective_axes": _dedupe_strings(effective_axes),
        "deprioritized_anchors": _dedupe_strings(deprioritized_anchors),
        "rescue_probe_anchors": _dedupe_strings(rescue_probe_anchors),
        "avoid_exact_params": avoid_exact_params,
        "promoted_params": promoted_params,
    }



def _candidate_pool_policy_from_evidence(item: dict[str, Any]) -> dict[str, Any]:
    attribution = item.get("candidate_pool_attribution")
    if not isinstance(attribution, dict):
        return {}
    policy = attribution.get("search_space_policy")
    return policy if isinstance(policy, dict) else {}



def _should_avoid_exact_memory_params(item: dict[str, Any]) -> bool:
    avoid_directives = set(_string_list_from_any(item.get("avoid_directives")))
    return bool(
        avoid_directives
        & {
            "do_not_promote_primary_axis_without_total_score",
            "do_not_repeat_guardrail_regressing_params",
            "do_not_repeat_without_new_mechanism",
        }
    )



def _apply_early_memory_policy(
    *,
    params: list[dict[str, Any]],
    memory_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    avoid_exact_params = [
        item
        for item in memory_policy.get("avoid_exact_params", [])
        if isinstance(item, dict)
    ]
    filtered = [
        row
        for row in params
        if not any(_early_params_match(row=row, avoided=avoided) for avoided in avoid_exact_params)
    ]
    if not filtered:
        filtered = params
    promoted_params = [
        item
        for item in memory_policy.get("promoted_params", [])
        if isinstance(item, dict)
    ]
    effective_axes = set(_string_list_from_any(memory_policy.get("effective_axes")))
    deprioritized_anchors = set(
        _string_list_from_any(memory_policy.get("deprioritized_anchors"))
    )
    rescue_probe_anchors = set(
        _string_list_from_any(memory_policy.get("rescue_probe_anchors"))
    )
    filtered.sort(
        key=lambda row: _early_memory_rank_key(
            row=row,
            promoted_params=promoted_params,
            effective_axes=effective_axes,
            deprioritized_anchors=deprioritized_anchors,
            rescue_probe_anchors=rescue_probe_anchors,
        )
    )
    return filtered



def _early_params_match(*, row: dict[str, Any], avoided: dict[str, Any]) -> bool:
    for key, value in avoided.items():
        if key not in row:
            return False
        if row[key] != value:
            return False
    return True



def _early_memory_rank_key(
    *,
    row: dict[str, Any],
    promoted_params: list[dict[str, Any]],
    effective_axes: set[str],
    deprioritized_anchors: set[str],
    rescue_probe_anchors: set[str],
) -> tuple[int, int, int, int, int, int, str]:
    exact_promoted = any(_early_params_match(row=row, avoided=item) for item in promoted_params)
    early_axis_reuse = (
        "early_rel_mse" in effective_axes
        and str(row.get("early_anchor", "")) in {
            "persistence",
            "phase_aligned_retrieval",
            "retrieval_derivative_match",
            "second_order_velocity",
        }
    )
    guardrail_candidate = str(row.get("early_anchor", "")) in {
        "persistence",
        "retrieval",
    }
    deprioritized_anchor = str(row.get("early_anchor", "")) in deprioritized_anchors
    rescue_probe_anchor = str(row.get("early_anchor", "")) in rescue_probe_anchors
    return (
        0 if exact_promoted else 1,
        1 if deprioritized_anchor and not rescue_probe_anchor else 0,
        0 if early_axis_reuse else 1,
        0 if guardrail_candidate else 1,
        0 if rescue_probe_anchor else 1,
        int(row.get("k", 0)),
        _early_param_sort_key(row),
    )



def _early_param_sort_key(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True)



def _dedupe_strings(values: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped



def _early_refinement_signals(
    evidence_params: list[dict[str, Any]] | None,
) -> dict[str, bool]:
    directives: set[str] = set()
    bottlenecks: set[str] = set()
    triggered_taxonomy: set[str] = set()
    recommended_axes: set[str] = set()
    for item in evidence_params or []:
        if not isinstance(item, dict):
            continue
        diagnosis = item.get("early_segment_diagnosis")
        if not isinstance(diagnosis, dict):
            continue
        bottleneck = diagnosis.get("bottleneck_type")
        if isinstance(bottleneck, str):
            bottlenecks.add(bottleneck)
        trend = diagnosis.get("candidate_trend_attribution")
        if isinstance(trend, dict) and isinstance(trend.get("recommended_axis"), str):
            recommended_axes.add(str(trend["recommended_axis"]))
        taxonomy = diagnosis.get("error_taxonomy")
        if isinstance(taxonomy, dict):
            for name, payload in taxonomy.items():
                if isinstance(payload, dict) and payload.get("triggered"):
                    triggered_taxonomy.add(str(name))
        for directive in diagnosis.get("action_directives", []):
            if isinstance(directive, dict) and isinstance(
                directive.get("directive"),
                str,
            ):
                directives.add(str(directive["directive"]))
    return {
        "expand_linear_velocity": (
            "expand_linear_velocity_anchor_search" in directives
            or "initial_derivative_mismatch" in bottlenecks
            or "initial_derivative_mismatch" in triggered_taxonomy
        ),
        "reduce_smoothing": (
            "over_smoothing" in bottlenecks
            or "over_smoothing" in triggered_taxonomy
            or "reduce_early_smoothing_and_anchor_weight" in directives
        ),
        "prioritize_mode": (
            "mode" in recommended_axes or "prioritize_axis:mode" in directives
        ),
        "expand_phase_alignment": (
            "phase_shift" in bottlenecks
            or "phase_shift" in triggered_taxonomy
            or "add_phase_alignment_or_derivative_distance_candidates" in directives
        ),
        "expand_bias_calibration": (
            "amplitude_bias" in bottlenecks
            or "amplitude_bias" in triggered_taxonomy
            or "add_short_horizon_affine_bias_calibration" in directives
        ),
        "expand_compound_calibration": (
            (
                "phase_shift" in bottlenecks
                or "phase_shift" in triggered_taxonomy
                or "add_phase_alignment_or_derivative_distance_candidates"
                in directives
            )
            and (
                "amplitude_bias" in bottlenecks
                or "amplitude_bias" in triggered_taxonomy
                or "initial_derivative_mismatch" in bottlenecks
                or "initial_derivative_mismatch" in triggered_taxonomy
                or "add_short_horizon_affine_bias_calibration" in directives
                or "expand_linear_velocity_anchor_search" in directives
            )
        ),
        "increase_smoothing": (
            "high_frequency_noise" in bottlenecks
            or "high_frequency_noise" in triggered_taxonomy
            or "increase_early_spectral_damping" in directives
        ),
        "expand_handoff_curves": (
            "early_blend_steps" in recommended_axes
            or "persistence_weight_start" in recommended_axes
            or "phase_shift" in bottlenecks
            or "amplitude_bias" in bottlenecks
            or "phase_shift" in triggered_taxonomy
            or "amplitude_bias" in triggered_taxonomy
        ),
        "increase_k": any(
            directive.get("axis") == "k" and directive.get("direction") == "increase"
            for item in evidence_params or []
            if isinstance(item, dict)
            for directive in item.get("search_directives", [])
            if isinstance(directive, dict)
        ),
        "decrease_k": any(
            directive.get("axis") == "k" and directive.get("direction") == "decrease"
            for item in evidence_params or []
            if isinstance(item, dict)
            for directive in item.get("search_directives", [])
            if isinstance(directive, dict)
        ),
    }



def _dedupe_values(values: list[Any]) -> list[Any]:
    deduped = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped



def _early_refinement_experiment_plan(
    *,
    evidence_params: list[dict[str, Any]] | None,
    candidate_budget: int,
) -> dict[str, Any]:
    center = _retrieval_refinement_center(evidence_params)
    signals = _early_refinement_signals(evidence_params)
    memory_policy = _early_memory_policy(evidence_params)
    axis_plan_by_bottleneck: dict[str, dict[str, Any]] = {}
    if signals["expand_phase_alignment"]:
        axis_plan_by_bottleneck["phase_alignment"] = {
            "candidate_hooks": ["phase_aligned_retrieval", "phase_shift_scale"],
            "reason": "phase_shift diagnosis requires transport-aware early anchors",
        }
    if signals["expand_bias_calibration"]:
        axis_plan_by_bottleneck["bias_calibration"] = {
            "candidate_hooks": ["mean_trend_bias", "curved_handoff"],
            "reason": "amplitude_bias diagnosis requires short-horizon mean correction",
        }
    if signals["expand_linear_velocity"]:
        axis_plan_by_bottleneck["derivative_consistency"] = {
            "candidate_hooks": [
                "damped_linear_velocity",
                "scaled_damped_velocity",
                "retrieval_derivative_match",
                "second_order_velocity",
            ],
            "reason": "initial derivative mismatch requires velocity-consistent anchors",
        }
    if signals["expand_compound_calibration"]:
        axis_plan_by_bottleneck["compound_transport_calibration"] = {
            "candidate_hooks": [
                "phase_derivative_hybrid",
                "affine_context_calibration",
            ],
            "reason": (
                "combined phase, derivative, or amplitude evidence requires "
                "compound early anchors"
            ),
        }
    return {
        "plan_id": "xp_task1_early_refinement_001",
        "decision": "two_stage_budgeted_early_search",
        "objective": {
            "primary": "early_rel_mse",
            "secondary": [
                "early_score",
                "task1_total_score_proxy",
                "inference_seconds",
            ],
            "focus": "reduce_short_horizon_error_before_score_arbitration",
        },
        "budget_policy": {
            "max_candidate_budget": 160,
            "preserve_required_directive_probes": True,
        },
        "center_params": center,
        "stage_plan": [
            {
                "stage": "axis_screen",
                "objective": "early_rel_mse",
                "axes": ["k", "mode", "early_blend_steps"],
                "purpose": (
                    "screen cheap high-impact axes before trying every anchor "
                    "combination"
                ),
            },
            {
                "stage": "focused_refine",
                "objective": "guarded_early_rel_mse",
                "axes": [
                    "persistence_weight_start",
                    "spectral_strength",
                    "early_anchor",
                ],
                "purpose": (
                    "probe directive-required anchors and smoothing only around "
                    "the current center"
                ),
            },
        ],
        "axis_plan": [
            {
                "axis": "k",
                "direction": "directed",
                "reason": "expanded-family evidence favored higher retrieval counts",
                "budget": 5,
            },
            {
                "axis": "mode",
                "direction": "compare",
                "reason": "v2 diagnosis trend attribution prioritized mode",
                "budget": 3,
            },
            {
                "axis": "early_blend_steps",
                "direction": "short_horizon_grid",
                "reason": "competition early window is steeply penalized by Rel-MSE",
                "budget": 4,
            },
            {
                "axis": "persistence_weight_start",
                "direction": "early_anchor_weight_grid",
                "reason": "separate short-window accuracy from late stability smoothing",
                "budget": 3,
            },
            {
                "axis": "early_anchor",
                "direction": "probe_required_directives",
                "reason": "test derivative-aware anchors without full cross-product expansion",
                "budget": 5,
            },
        ],
        "axis_plan_by_bottleneck": axis_plan_by_bottleneck,
        "memory_policy": {
            "retry_directives": memory_policy["retry_directives"],
            "avoid_directives": memory_policy["avoid_directives"],
            "effective_axes": memory_policy["effective_axes"],
            "deprioritized_anchors": memory_policy["deprioritized_anchors"],
            "rescue_probe_anchors": memory_policy["rescue_probe_anchors"],
            "avoid_exact_count": len(memory_policy["avoid_exact_params"]),
            "promoted_seed_count": len(memory_policy["promoted_params"]),
        },
        "guardrails": [
            "middle_score_non_regression",
            "late_rmse_non_regression",
            "context_prefix_exact",
        ],
        "candidate_budget": int(candidate_budget),
        "stop_criteria": [
            "stop_if_early_rel_mse_does_not_improve",
            "stop_if_middle_or_late_guardrail_fails",
            "stop_if_context_prefix_validation_fails",
        ],
    }



def _retrieval_refinement_search_directives_from_evidence(
    evidence_params: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    directives = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in evidence_params or []:
        if not isinstance(item, dict):
            continue
        raw_directives = item.get("search_directives")
        if not isinstance(raw_directives, list):
            continue
        for directive in raw_directives:
            if not isinstance(directive, dict):
                continue
            axis = str(directive.get("axis", ""))
            direction = str(directive.get("direction", ""))
            if axis == "mode" and direction == "focus":
                values = _string_list_from_any(directive.get("values"))
                avoid_values = _string_list_from_any(directive.get("avoid_values"))
                if not values and not avoid_values:
                    continue
                key = (axis, direction, ",".join(values), ",".join(avoid_values))
                if key in seen:
                    continue
                seen.add(key)
                directives.append(
                    {
                        "axis": axis,
                        "direction": direction,
                        "values": values,
                        "avoid_values": avoid_values,
                        "reason": str(
                            directive.get("reason", "candidate_pool_search_policy")
                        ),
                    }
                )
                continue
            if axis not in {
                "k",
                "early_blend_steps",
                "persistence_weight_start",
                "spectral_strength",
            }:
                continue
            if direction not in {"increase", "decrease"}:
                continue
            key = (axis, direction, "", "")
            if key in seen:
                continue
            seen.add(key)
            directives.append({"axis": axis, "direction": direction})
    return directives



def _retrieval_refinement_experiment_plan(
    *,
    evidence_params: list[dict[str, Any]] | None,
    candidate_budget: int,
) -> dict[str, Any]:
    directives = _retrieval_refinement_search_directives_from_evidence(
        evidence_params
    )
    if directives:
        axis_plan = [
            {
                "axis": directive["axis"],
                "direction": directive["direction"],
                "reason": directive.get("reason", "top_candidate_score_trend"),
                "budget": 3,
            }
            for directive in directives
        ]
        avoid_rules = _retrieval_refinement_avoid_rules_from_directives(directives)
        decision = "axis_directed_local_search"
        focus = "score_trend"
    else:
        axis_plan = [
            {
                "axis": axis,
                "direction": "bidirectional",
                "reason": "no_clear_axis_trend",
                "budget": 2,
            }
            for axis in [
                "k",
                "early_blend_steps",
                "persistence_weight_start",
                "spectral_strength",
            ]
        ]
        avoid_rules = []
        decision = "centered_local_search"
        focus = "local_neighborhood"
    return {
        "plan_id": "xp_task1_retrieval_refinement_local_search_001",
        "decision": decision,
        "objective": {
            "primary": "task1_total_score_proxy",
            "secondary": [
                "early_rel_mse",
                "middle_rel_mse",
                "late_rmse",
                "inference_seconds",
            ],
            "focus": focus,
        },
        "axis_plan": axis_plan,
        "avoid_rules": avoid_rules,
        "candidate_budget": int(candidate_budget),
        "stop_criteria": [
            "stop_if_no_candidate_improves_seed",
            "stop_if_context_prefix_validation_fails",
            "stop_if_inference_budget_exceeded",
        ],
        "switch_criteria": [
            "switch_family_after_repeated_axis_plateau",
            "switch_family_if_score_regresses_on_heldout_split",
        ],
    }



def _retrieval_refinement_avoid_rules_from_directives(
    directives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rules = []
    for directive in directives:
        direction = directive["direction"]
        if direction == "increase":
            relation = "below_center"
        elif direction == "decrease":
            relation = "above_center"
        elif direction == "focus" and directive.get("axis") == "mode":
            focus_values = _string_list_from_any(directive.get("values"))
            avoid_values = _string_list_from_any(directive.get("avoid_values"))
            if focus_values:
                rules.append(
                    {
                        "axis": "mode",
                        "relation": "not_in_focus_values",
                        "values": focus_values,
                        "reason": directive.get(
                            "reason",
                            "candidate_pool_search_policy",
                        ),
                    }
                )
            for value in avoid_values:
                rules.append(
                    {
                        "axis": "mode",
                        "relation": "equals",
                        "value": value,
                        "reason": directive.get(
                            "reason",
                            "candidate_pool_search_policy",
                        ),
                    }
                )
            continue
        else:
            continue
        rules.append(
            {
                "axis": directive["axis"],
                "relation": relation,
                "reason": "opposite_of_score_trend",
            }
        )
    return rules



def _filter_retrieval_refinement_params_by_directives(
    *,
    params: list[dict[str, Any]],
    center: dict[str, Any],
    directives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rules = _retrieval_refinement_avoid_rules_from_directives(directives)
    if not rules:
        return params
    filtered = []
    for row in params:
        rejected = False
        for rule in rules:
            axis = rule["axis"]
            if axis == "mode":
                mode = str(row.get("mode", ""))
                if rule["relation"] == "not_in_focus_values" and mode not in set(
                    _string_list_from_any(rule.get("values"))
                ):
                    rejected = True
                if rule["relation"] == "equals" and mode == str(rule.get("value", "")):
                    rejected = True
                continue
            if axis not in row or axis not in center:
                continue
            value = float(row[axis])
            center_value = float(center[axis])
            if rule["relation"] == "below_center" and value < center_value:
                rejected = True
            if rule["relation"] == "above_center" and value > center_value:
                rejected = True
        if not rejected:
            filtered.append(row)
    return filtered



def _directed_retrieval_refinement_params(
    *,
    center: dict[str, Any],
    axis: str,
    direction: str,
) -> list[dict[str, Any]]:
    sign = 1 if direction == "increase" else -1
    rows = []
    axis_steps: dict[str, list[float]] = {
        "k": [8.0, 16.0, 24.0],
        "early_blend_steps": [10.0, 20.0, 30.0],
        "persistence_weight_start": [0.1, 0.15, 0.2],
        "spectral_strength": [0.5, 1.0, 1.5],
    }
    for step in axis_steps.get(axis, []):
        row = dict(center)
        if axis == "k":
            row["k"] = _clamp_int(int(center["k"]) + sign * int(step), 4, 128)
        elif axis == "early_blend_steps":
            row["early_blend_steps"] = _clamp_int(
                int(center["early_blend_steps"]) + sign * int(step),
                0,
                80,
            )
        elif axis == "persistence_weight_start":
            row["persistence_weight_start"] = _clamp_float(
                float(center["persistence_weight_start"]) + sign * step,
                0.0,
                1.0,
            )
        elif axis == "spectral_strength":
            row["spectral_strength"] = _clamp_float(
                float(center["spectral_strength"]) + sign * step,
                0.0,
                2.0,
            )
        rows.append(row)
    return rows



def _retrieval_refinement_center(
    evidence_params: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    for item in evidence_params or []:
        if not isinstance(item, dict):
            continue
        params = item.get("params")
        if not isinstance(params, dict):
            continue
        normalized = _normalize_evidence_params(params)
        if normalized is not None and "k" in normalized:
            return normalized
    return {
        "k": 48,
        "mode": "affine",
        "early_blend_steps": 20,
        "persistence_weight_start": 0.5,
        "spectral_strength": 0.5,
    }



