"""Task 1 FNO training policy core functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pde_agent.core.task1_short_window_common import (
    SUPPORTED_FNO_LOSS_TYPES,
    SUPPORTED_MODEL_FAMILIES,
    SUPPORTED_NORMALIZATION_TYPES,
    Task1ShortWindowTrainingError,
)
from pde_agent.core.task1_short_window_family_policy import family_record as _family_record
def _fno_candidate_record(
    *, config: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    base = _family_record(report)
    fno_selection_score = _fno_early_middle_selection_score(base)
    base["candidate_id"] = str(config["candidate_id"])
    base["score"] = fno_selection_score
    base["fno_selection_score"] = fno_selection_score
    base["selection_components"] = _fno_selection_components(base)
    base["config"] = {
        "model_family": str(config.get("model_family", "fno_lite")),
        "modes": int(config["modes"]),
        "width": int(config["width"]),
        "depth": int(config["depth"]),
        "loss_type": str(config["loss_type"]),
        "normalization": str(config.get("normalization", "none")),
        "forecast_steps": int(config["forecast_steps"]),
        "train_steps": int(config["train_steps"]),
    }
    return base

def _select_fno_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        records,
        key=lambda item: (
            -_fno_early_middle_selection_score(item),
            -float(item.get("early_score", 0.0)),
            float(item.get("final_val_loss", float("inf"))),
            float(item.get("runtime_seconds", float("inf"))),
        ),
    )
    selected = dict(ranked[0])
    selected["selection_components"] = _fno_selection_components(selected)
    selected["fno_selection_score"] = _fno_early_middle_selection_score(selected)
    selected["selection_reason_codes"] = _fno_selection_reason_codes(
        selected=selected,
        records=records,
    )
    return selected

def _fno_selection_reason_codes(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> list[str]:
    reasons = ["multi_objective_selection"]
    if float(selected.get("early_score", 0.0)) >= max(
        float(item.get("early_score", 0.0)) for item in records
    ):
        reasons.append("best_early_score")
    if _fno_early_middle_selection_score(selected) >= max(
        _fno_early_middle_selection_score(item) for item in records
    ):
        reasons.append("early_middle_objective_selected")
    if selected.get("middle_guardrail", {}).get("evaluated"):
        reasons.append("middle_guardrail_available")
    if selected.get("late_guardrail", {}).get("evaluated"):
        reasons.append("late_guardrail_available")
    if selected.get("config", {}).get("loss_type") != "mse":
        reasons.append("non_default_loss_selected")
    if len({item["config"]["loss_type"] for item in records}) > 1:
        reasons.append("loss_family_compared")
    return reasons

def _fno_selection_components(record: dict[str, Any]) -> dict[str, Any]:
    middle = record.get("middle_guardrail", {})
    late = record.get("late_guardrail", {})
    return {
        "early": {
            "evaluated": True,
            "score": float(record.get("early_score", 0.0)),
        },
        "middle": {
            "evaluated": bool(middle.get("evaluated")),
            "score": float(middle.get("score", 0.0)) if middle.get("evaluated") else 0.0,
        },
        "late": {
            "evaluated": bool(late.get("evaluated")),
            "score": float(late.get("score", 0.0)) if late.get("evaluated") else 0.0,
        },
    }

def _fno_selection_score(record: dict[str, Any]) -> float:
    components = _fno_selection_components(record)
    return (
        0.60 * components["early"]["score"]
        + 0.25 * components["middle"]["score"]
        + 0.15 * components["late"]["score"]
    )

def _fno_early_middle_selection_score(record: dict[str, Any]) -> float:
    middle = record.get("middle_guardrail", {})
    late = record.get("late_guardrail", {})
    early_rel = _float_or_default_local(record.get("early_rel_mse"), 5.0)
    early_quality = max(0.0, 1.0 - min(early_rel, 5.0) / 5.0)
    middle_quality = _guardrail_quality(middle)
    late_score = (
        _float_or_default_local(late.get("score"), 0.0) if late.get("evaluated") else 0.0
    )
    late_guardrail_bonus = min(late_score, 5.0) / 5.0
    return 100.0 * (
        0.62 * early_quality
        + 0.35 * middle_quality
        + 0.03 * late_guardrail_bonus
    )

def _guardrail_quality(guardrail: dict[str, Any]) -> float:
    if not guardrail.get("evaluated"):
        return 0.0
    if "rel_mse" in guardrail:
        rel_mse = _float_or_default_local(guardrail.get("rel_mse"), 5.0)
        return max(0.0, 1.0 - min(rel_mse, 5.0) / 5.0)
    return max(0.0, min(_float_or_default_local(guardrail.get("score"), 0.0), 100.0) / 100.0)

def _fno_optimization_constraints(
    *,
    selected: dict[str, Any],
    directive: dict[str, Any],
    records: list[dict[str, Any]],
    graft_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = selected["config"]
    constraints = {
        "training_model_families": [str(config.get("model_family", "fno_lite"))]
        if directive["action"] != "fallback_to_conv1d"
        else ["conv1d_residual"],
        "prefer_training_objectives": [],
        "refine_training_horizons": [],
        "avoid_training_model_families": [],
        "fno_training_configs": [item["config"] for item in records],
    }
    if directive["action"] == "change_loss":
        constraints["prefer_training_objectives"].extend(
            ["early_weighted_mse", "spectral_mse"]
        )
    else:
        constraints["prefer_training_objectives"].append(str(config["loss_type"]))
    if directive["action"] == "change_horizon":
        constraints["refine_training_horizons"].append("shorter_early_window")
    if directive["action"] == "fallback_to_conv1d":
        constraints["avoid_training_model_families"].append("fno_lite")
    if _any_guardrail_evaluated(records, "middle_guardrail"):
        constraints["refine_training_horizons"].append("middle_guardrail")
    if _any_guardrail_evaluated(records, "late_guardrail"):
        constraints["refine_training_horizons"].append("late_guardrail")
    if graft_evidence is not None:
        constraints["graft_objective"] = {
            "required": True,
            "promote_only_positive_delta": True,
            "late_non_regression_required": True,
            "last_best_task_prediction_score_delta": graft_evidence[
                "best_task_prediction_score_delta"
            ],
            "last_best_window": graft_evidence["best_window"],
            "last_best_blend_weight": graft_evidence["best_blend_weight"],
            "last_positive_candidate_count": graft_evidence["positive_candidate_count"],
        }
        constraints["refine_training_horizons"].extend(
            graft_evidence["recommended_refine_training_horizons"]
        )
        constraints["prefer_training_objectives"].extend(
            graft_evidence["recommended_training_objectives"]
        )
        if _graft_plateau_requires_new_dynamics_source(
            graft_evidence=graft_evidence,
            selected=selected,
        ):
            constraints["training_model_families"] = _dedupe(
                constraints["training_model_families"]
                + ["fno_ar1", "conv1d_residual"]
            )
            constraints["prefer_training_objectives"].extend(
                ["rollout_consistency", "autoregressive_residual"]
            )
            constraints["refine_training_horizons"].append(
                "new_early_dynamics_source"
            )
            constraints["graft_objective"][
                "plateau_requires_new_dynamics_source"
            ] = True
    constraints["prefer_training_objectives"] = _dedupe(
        constraints["prefer_training_objectives"]
    )
    constraints["refine_training_horizons"] = _dedupe(
        constraints["refine_training_horizons"]
    )
    constraints["avoid_training_model_families"] = _dedupe(
        constraints["avoid_training_model_families"]
    )
    return constraints

def _graft_plateau_requires_new_dynamics_source(
    *,
    graft_evidence: dict[str, Any],
    selected: dict[str, Any],
) -> bool:
    delta = _float_or_default_local(
        graft_evidence.get("best_task_prediction_score_delta"),
        0.0,
    )
    positive_count = int(graft_evidence.get("positive_candidate_count", 0))
    config = selected.get("config") if isinstance(selected, dict) else None
    model_family = (
        str(config.get("model_family", "fno_lite"))
        if isinstance(config, dict)
        else "fno_lite"
    )
    return (
        model_family == "fno_lite"
        and positive_count > 0
        and 0.0 < delta < 0.2
    )

def _has_positive_late_safe_graft_evidence(evidences: list[dict[str, Any]]) -> bool:
    for evidence in evidences:
        graft = evidence.get("graft_evidence")
        if not isinstance(graft, dict):
            continue
        delta = _float_or_default_local(
            graft.get("best_task_prediction_score_delta"),
            0.0,
        )
        if delta > 0.0 and graft.get("late_non_regression_passed") is True:
            return True
    return False

def _fno_graft_evidence_summary(path: Path) -> dict[str, Any]:
    report = _load_json(path, "Task1 FNO graft report")
    if report.get("state_hint") != "TASK1_FNO_GRAFT_EVALUATOR_RAN":
        raise Task1ShortWindowTrainingError("graft report has invalid state_hint")
    best = report.get("best_candidate")
    if not isinstance(best, dict):
        raise Task1ShortWindowTrainingError("graft report is missing best_candidate")
    candidate_records = [
        record for record in report.get("candidate_records", []) if isinstance(record, dict)
    ]
    positive = [
        record
        for record in candidate_records
        if _float_or_default_local(record.get("task_prediction_score_delta"), 0.0) > 0.0
    ]
    best_window = [
        int(best.get("window_start", 0)),
        int(best.get("window_stop", 0)),
    ]
    return {
        "schema_version": 1,
        "state_hint": "FNO_GRAFT_EVIDENCE_SUMMARIZED",
        "graft_report_path": str(path),
        "baseline_task_prediction_score": _float_or_default_local(
            report.get("baseline_score", {}).get("task_prediction_score")
            if isinstance(report.get("baseline_score"), dict)
            else None,
            0.0,
        ),
        "source_task_prediction_score": _float_or_default_local(
            report.get("source_score", {}).get("task_prediction_score")
            if isinstance(report.get("source_score"), dict)
            else None,
            0.0,
        ),
        "best_task_prediction_score": _float_or_default_local(
            best.get("task_prediction_score"),
            0.0,
        ),
        "best_task_prediction_score_delta": _float_or_default_local(
            best.get("task_prediction_score_delta"),
            0.0,
        ),
        "late_non_regression_passed": bool(best.get("late_non_regression_passed")),
        "late_score_delta": _float_or_default_local(best.get("late_score_delta"), 0.0),
        "best_window": best_window,
        "best_blend_weight": _float_or_default_local(best.get("blend_weight"), 0.0),
        "candidate_count": len(candidate_records),
        "positive_candidate_count": len(positive),
        "positive_candidate_ids": [
            str(record.get("candidate_id"))
            for record in positive[:10]
            if record.get("candidate_id") is not None
        ],
        "recommended_refine_training_horizons": _graft_refine_horizons(best_window),
        "recommended_training_objectives": _graft_training_objectives(best),
    }

def _graft_refine_horizons(best_window: list[int]) -> list[str]:
    start, stop = best_window
    if start <= 10 and stop <= 57:
        return ["early_window", "early_middle_graft_source"]
    if start < 57 and stop <= 105:
        return ["early_to_middle_transition", "early_middle_graft_source"]
    if start >= 57:
        return ["middle_window", "early_middle_graft_source"]
    return ["early_middle_graft_source"]

def _graft_training_objectives(best: dict[str, Any]) -> list[str]:
    delta = _float_or_default_local(best.get("task_prediction_score_delta"), 0.0)
    if delta <= 0.0:
        return ["early_weighted_mse", "horizon_weighted_mse", "spectral_mse"]
    window_stop = int(best.get("window_stop", 0))
    if window_stop <= 57:
        return ["early_weighted_mse", "horizon_weighted_mse"]
    return ["horizon_weighted_mse", "spectral_mse", "early_weighted_mse"]

def _any_guardrail_evaluated(records: list[dict[str, Any]], key: str) -> bool:
    return any(record.get(key, {}).get("evaluated") for record in records)

def _fno_planning_preferences(evidences: list[dict[str, Any]]) -> dict[str, Any]:
    preferences = {
        "model_families": [],
        "loss_types": [],
        "forecast_steps": [],
        "modes": [],
        "widths": [],
        "depths": [],
        "train_steps": [],
        "normalizations": [],
        "anchor_configs": [],
    }
    best_evidence = None
    best_selected = None
    best_score = float("-inf")
    latest_evidence = evidences[-1] if evidences else None
    requires_new_dynamics_source = False
    for evidence in evidences:
        health_guidance = evidence.get("search_guidance")
        if (
            evidence.get("state_hint") == "FNO_TRAINING_HEALTH_FEEDBACK_BUILT"
            and isinstance(health_guidance, list)
        ):
            for guidance in health_guidance:
                if not isinstance(guidance, dict):
                    continue
                if guidance.get("guidance_type") != "operator_training_policy":
                    continue
                preferences["model_families"].extend(
                    _fno_model_family_list_optional(
                        guidance.get("preferred_model_families")
                    )
                )
                preferences["loss_types"].extend(
                    [
                        value for value in _string_list(guidance.get("preferred_loss_types"))
                        if value in SUPPORTED_FNO_LOSS_TYPES
                    ]
                )
                if "model_family" in _string_list(guidance.get("required_axis_changes")):
                    preferences["requires_model_family_change"] = True
                if guidance.get("next_action") in {
                    "expand_search_radius",
                    "rebalance_candidate_families",
                    "challenge_anchor",
                }:
                    preferences["requires_model_family_change"] = True
                if guidance.get("next_action") == "repair_regression":
                    preferences["health_requested_regression_repair"] = True
                    preferences.setdefault("health_feedback_actions", []).append(
                        "repair_regression"
                    )
        for anchor in _training_memory_anchor_list_optional(
            evidence.get("training_memory_anchors")
        ):
            config = anchor.get("config")
            if not isinstance(config, dict):
                continue
            preferences["anchor_configs"].append(dict(config))
            preferences["model_families"].append(
                str(config.get("model_family", "fno_lite"))
            )
            preferences["loss_types"].append(str(config.get("loss_type", "mse")))
            preferences["modes"].extend(
                _positive_int_list_optional([config.get("modes")])
            )
            preferences["widths"].extend(
                _positive_int_list_optional([config.get("width")])
            )
            preferences["depths"].extend(
                _positive_int_list_optional([config.get("depth")])
            )
            preferences["train_steps"].extend(
                _positive_int_list_optional([config.get("train_steps")])
            )
            preferences["forecast_steps"].extend(
                _positive_int_list_optional([config.get("forecast_steps")])
            )
            preferences["normalizations"].extend(
                _normalization_list_optional([config.get("normalization", "none")])
            )
            score = _float_or_default_local(anchor.get("score"), float("-inf"))
            if score > best_score:
                best_score = score
                best_evidence = evidence
                best_selected = {
                    "config": config,
                    "fno_selection_score": score,
                }
        selected = evidence.get("selected_candidate")
        if not isinstance(selected, dict) and evidence.get("state_hint") == "FNO_TRAINING_MEMORY_BUILT":
            best_so_far = evidence.get("best_so_far")
            if isinstance(best_so_far, dict):
                selected = {
                    "config": best_so_far.get("config"),
                    "fno_selection_score": best_so_far.get("score"),
                }
        if isinstance(selected, dict):
            score = _float_or_default_local(
                selected.get("fno_selection_score"),
                float("-inf"),
            )
            if score > best_score:
                best_score = score
                best_evidence = evidence
                best_selected = selected
        constraints_sources = []
        constraints = evidence.get("next_generation_constraints")
        if isinstance(constraints, dict):
            constraints_sources.append(constraints)
        if isinstance(selected, dict):
            selected_constraints = selected.get("next_generation_constraints")
            if isinstance(selected_constraints, dict):
                constraints_sources.append(selected_constraints)
        for constraints in constraints_sources:
            objectives = _string_list(constraints.get("prefer_training_objectives"))
            preferences["model_families"].extend(
                _fno_model_family_list_optional(constraints.get("training_model_families"))
            )
            for objective in objectives:
                if objective in SUPPORTED_FNO_LOSS_TYPES:
                    preferences["loss_types"].append(objective)
                if objective == "short_window_mse":
                    preferences["loss_types"].append("mse")
            horizons = _string_list(constraints.get("refine_training_horizons"))
            if any(
                "shorter" in horizon or horizon == "early_window"
                for horizon in horizons
            ):
                preferences["forecast_steps"].extend([12, 20])
            if any(
                "early_middle" in horizon
                or "early_to_middle" in horizon
                or "middle" in horizon
                for horizon in horizons
            ):
                preferences["forecast_steps"].extend([47, 95])
            if any(
                "new" in horizon and "dynamics" in horizon for horizon in horizons
            ):
                requires_new_dynamics_source = True
            graft_objective = constraints.get("graft_objective")
            if (
                isinstance(graft_objective, dict)
                and graft_objective.get("plateau_requires_new_dynamics_source") is True
            ):
                requires_new_dynamics_source = True
            preferences["modes"].extend(_positive_int_list_optional(constraints.get("modes")))
            preferences["widths"].extend(_positive_int_list_optional(constraints.get("widths")))
            preferences["depths"].extend(_positive_int_list_optional(constraints.get("depths")))
            preferences["train_steps"].extend(
                _positive_int_list_optional(constraints.get("train_steps"))
            )
            preferences["forecast_steps"].extend(
                _positive_int_list_optional(constraints.get("forecast_steps"))
            )
            preferences["normalizations"].extend(
                _normalization_list_optional(constraints.get("normalizations"))
            )
        if isinstance(selected, dict) and isinstance(selected.get("config"), dict):
            config = selected["config"]
            modes = int(config.get("modes", 0))
            width = int(config.get("width", 0))
            depth = int(config.get("depth", 0))
            train_steps = int(config.get("train_steps", 0))
            forecast_steps = int(config.get("forecast_steps", 0))
            preferences["modes"].extend([modes, modes + 4])
            preferences["widths"].extend([width, width + 16])
            preferences["depths"].extend([depth, depth + 1])
            preferences["train_steps"].extend([train_steps, train_steps + 50])
            preferences["forecast_steps"].append(forecast_steps)
            preferences["normalizations"].append(str(config.get("normalization", "none")))
            preferences["model_families"].append(str(config.get("model_family", "fno_lite")))
    graft_anchor_evidence = _best_late_safe_graft_evidence(evidences)
    if isinstance(graft_anchor_evidence, dict):
        best_evidence = graft_anchor_evidence
    anchor_selected = None
    if isinstance(best_selected, dict) and isinstance(best_selected.get("config"), dict):
        anchor_selected = best_selected["config"]
    elif isinstance(best_evidence, dict):
        selected = best_evidence.get("selected_candidate")
        if isinstance(selected, dict) and isinstance(selected.get("config"), dict):
            anchor_selected = selected["config"]
    regression_mode = False
    if isinstance(latest_evidence, dict):
        latest_selected = latest_evidence.get("selected_candidate")
        if not isinstance(latest_selected, dict) and latest_evidence.get("state_hint") == "FNO_TRAINING_MEMORY_BUILT":
            latest_best = latest_evidence.get("best_so_far")
            if isinstance(latest_best, dict):
                latest_selected = {
                    "config": latest_best.get("config"),
                    "fno_selection_score": latest_best.get("score"),
                }
        latest_score = _float_or_default_local(
            latest_selected.get("fno_selection_score") if isinstance(latest_selected, dict) else None,
            float("-inf"),
        )
        regression_mode = latest_score < best_score
    if preferences.get("health_requested_regression_repair"):
        regression_mode = True
    expansion_regression_memory = (
        _has_capacity_expansion_regression(evidences=evidences, anchor_config=anchor_selected)
        if isinstance(anchor_selected, dict)
        else False
    )
    cleaned = {
        key: _dedupe([str(item) for item in value])
        for key, value in preferences.items()
        if key != "anchor_configs" and isinstance(value, list)
    }
    for key, value in preferences.items():
        if key == "anchor_configs" or isinstance(value, list):
            continue
        cleaned[key] = value
    cleaned["anchor_configs"] = [
        dict(item)
        for item in preferences.get("anchor_configs", [])
        if isinstance(item, dict)
    ]
    if isinstance(best_evidence, dict):
        cleaned = _prioritize_graft_anchor_preferences(
            preferences=cleaned,
            anchor_evidence=best_evidence,
        )
    requires_middle_guardrail_source = any(
        int(value) >= 47 for value in preferences.get("forecast_steps", [])
    )
    if (regression_mode or expansion_regression_memory) and isinstance(anchor_selected, dict):
        cleaned = _fno_regression_aware_preferences(
            preferences=cleaned,
            anchor_config=anchor_selected,
            preserve_middle_forecasts=requires_middle_guardrail_source,
        )
    cleaned["regression_mode"] = regression_mode
    cleaned["health_requested_regression_repair"] = bool(
        preferences.get("health_requested_regression_repair")
    )
    cleaned["capacity_expansion_regression_memory"] = expansion_regression_memory
    cleaned["requires_new_dynamics_source"] = bool(requires_new_dynamics_source)
    cleaned["requires_middle_guardrail_source"] = bool(
        requires_middle_guardrail_source
    )
    if regression_mode and requires_new_dynamics_source:
        cleaned["model_families"] = _dedupe(
            cleaned.get("model_families", []) + ["fno_ar1", "conv1d_residual"]
        )
    cleaned["best_selected_config"] = anchor_selected or {}
    cleaned["best_selected_score"] = best_score if best_score != float("-inf") else None
    return cleaned

def _operator_policy_from_search_guidance(
    search_guidance: list[dict[str, Any]],
) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    for guidance in search_guidance:
        if not isinstance(guidance, dict):
            continue
        if guidance.get("guidance_type") != "operator_training_policy":
            continue
        for key in [
            "next_action",
            "required_axis_changes",
            "preferred_loss_types",
            "preferred_model_families",
            "preferred_forecast_steps",
            "preferred_normalizations",
            "budget_adjustments",
            "hypothesis_lab_requests",
            "reason_codes",
        ]:
            value = guidance.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                policy[key] = _dedupe([*(policy.get(key, []) or []), *value])
            elif isinstance(value, dict):
                merged = dict(policy.get(key, {}) or {})
                merged.update(value)
                policy[key] = merged
            else:
                policy[key] = value
    return policy

def _best_late_safe_graft_evidence(
    evidences: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best = None
    best_delta = float("-inf")
    for evidence in evidences:
        graft = evidence.get("graft_evidence")
        if not isinstance(graft, dict):
            continue
        if graft.get("late_non_regression_passed") is not True:
            continue
        delta = _float_or_default_local(
            graft.get("best_task_prediction_score_delta"),
            float("-inf"),
        )
        if delta > best_delta:
            best_delta = delta
            best = evidence
    return best

def _prioritize_graft_anchor_preferences(
    *,
    preferences: dict[str, Any],
    anchor_evidence: dict[str, Any],
) -> dict[str, Any]:
    cleaned = {
        key: list(value) if isinstance(value, list) else value
        for key, value in preferences.items()
    }
    selected = anchor_evidence.get("selected_candidate")
    config = selected.get("config") if isinstance(selected, dict) else None
    if isinstance(config, dict):
        _move_preference_to_front(cleaned, "loss_types", str(config.get("loss_type", "")))
        _move_preference_to_front(
            cleaned,
            "normalizations",
            str(config.get("normalization", "")),
        )
        for pref_key, config_key in [
            ("modes", "modes"),
            ("widths", "width"),
            ("depths", "depth"),
            ("forecast_steps", "forecast_steps"),
            ("train_steps", "train_steps"),
            ("model_families", "model_family"),
        ]:
            value = config.get(config_key)
            if value is not None:
                _move_preference_to_front(cleaned, pref_key, str(value))
    graft = anchor_evidence.get("graft_evidence")
    if isinstance(graft, dict):
        for objective in reversed(_string_list(graft.get("recommended_training_objectives"))):
            if objective in SUPPORTED_FNO_LOSS_TYPES:
                _move_preference_to_front(cleaned, "loss_types", objective)
    return cleaned

def _move_preference_to_front(
    preferences: dict[str, list[str]],
    key: str,
    value: str,
) -> None:
    if not value:
        return
    rows = [item for item in preferences.get(key, []) if item != value]
    preferences[key] = [value, *rows]

def _has_capacity_expansion_regression(
    *,
    evidences: list[dict[str, Any]],
    anchor_config: dict[str, Any] | None,
) -> bool:
    if not isinstance(anchor_config, dict):
        return False
    anchor_modes = int(anchor_config.get("modes", 0))
    anchor_width = int(anchor_config.get("width", 0))
    anchor_depth = int(anchor_config.get("depth", 0))
    anchor_train_steps = int(anchor_config.get("train_steps", 0))
    anchor_score = float("-inf")
    for evidence in evidences:
        selected = evidence.get("selected_candidate")
        if not isinstance(selected, dict):
            continue
        config = selected.get("config")
        if not isinstance(config, dict):
            continue
        if _same_fno_capacity(config, anchor_config):
            anchor_score = max(
                anchor_score,
                _float_or_default_local(selected.get("fno_selection_score"), float("-inf")),
            )
    if anchor_score == float("-inf"):
        return False
    for evidence in evidences:
        selected = evidence.get("selected_candidate")
        if not isinstance(selected, dict):
            continue
        config = selected.get("config")
        if not isinstance(config, dict):
            continue
        expanded = (
            int(config.get("modes", 0)) > anchor_modes
            or int(config.get("width", 0)) > anchor_width
            or int(config.get("depth", 0)) > anchor_depth
            or int(config.get("train_steps", 0)) > anchor_train_steps
        )
        score = _float_or_default_local(selected.get("fno_selection_score"), float("-inf"))
        if expanded and score < anchor_score:
            return True
    return False

def _same_fno_capacity(config: dict[str, Any], anchor_config: dict[str, Any]) -> bool:
    return (
        int(config.get("modes", 0)) == int(anchor_config.get("modes", 0))
        and int(config.get("width", 0)) == int(anchor_config.get("width", 0))
        and int(config.get("depth", 0)) == int(anchor_config.get("depth", 0))
        and int(config.get("train_steps", 0)) == int(anchor_config.get("train_steps", 0))
        and int(config.get("forecast_steps", 0)) == int(anchor_config.get("forecast_steps", 0))
    )

def _fno_regression_aware_preferences(
    *,
    preferences: dict[str, Any],
    anchor_config: dict[str, Any],
    preserve_middle_forecasts: bool = False,
) -> dict[str, Any]:
    cleaned = {
        key: list(value) if isinstance(value, list) else value
        for key, value in preferences.items()
    }
    anchor_modes = str(int(anchor_config.get("modes", 0)))
    anchor_widths = str(int(anchor_config.get("width", 0)))
    anchor_depths = str(int(anchor_config.get("depth", 0)))
    anchor_train_steps = str(int(anchor_config.get("train_steps", 0)))
    anchor_forecast_steps = str(int(anchor_config.get("forecast_steps", 0)))
    cleaned["modes"] = _dedupe(
        [value for value in cleaned.get("modes", []) if int(value) <= int(anchor_modes)]
        + [anchor_modes]
    )
    cleaned["widths"] = _dedupe(
        [value for value in cleaned.get("widths", []) if int(value) <= int(anchor_widths)]
        + [anchor_widths]
    )
    cleaned["depths"] = _dedupe(
        [value for value in cleaned.get("depths", []) if int(value) <= int(anchor_depths)]
        + [anchor_depths]
    )
    cleaned["train_steps"] = _dedupe(
        [value for value in cleaned.get("train_steps", []) if int(value) <= int(anchor_train_steps)]
        + [anchor_train_steps]
    )
    if preserve_middle_forecasts:
        cleaned["forecast_steps"] = _dedupe(
            cleaned.get("forecast_steps", []) + [anchor_forecast_steps]
        )
    else:
        cleaned["forecast_steps"] = _dedupe(
            [value for value in cleaned.get("forecast_steps", []) if int(value) <= int(anchor_forecast_steps)]
            + [anchor_forecast_steps]
        )
    return cleaned

def _anchor_train_step_values(
    preferences: dict[str, Any],
    fallback: list[int],
) -> list[int]:
    anchor = preferences.get("best_selected_config")
    if not isinstance(anchor, dict):
        return fallback
    train_steps = int(anchor.get("train_steps", 0))
    if train_steps <= 0:
        return fallback
    values = [train_steps]
    if train_steps > 50:
        values.append(train_steps - 50)
    return sorted(_dedupe(values))

def _medium_training_budget_values(values: list[int]) -> list[int]:
    return sorted(_dedupe([value for value in values if int(value) >= 20] + [50]))

def _cap_positive_int_values(
    values: list[int],
    *,
    maximum: int,
    fallback: list[int],
) -> list[int]:
    capped = [int(value) for value in values if 0 < int(value) <= maximum]
    if not capped:
        capped = [int(value) for value in fallback if 0 < int(value) <= maximum]
    return sorted(_dedupe(capped))

def _float_or_default_local(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default

def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Task1ShortWindowTrainingError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Task1ShortWindowTrainingError(f"{label} must be a JSON object")
    return payload

def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Task1ShortWindowTrainingError(f"{label} must be a positive integer")
    return value

def _positive_int_list_optional(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    rows = []
    for value in values:
        try:
            rows.append(_positive_int(int(value), "candidate_space"))
        except (TypeError, ValueError, Task1ShortWindowTrainingError):
            continue
    return sorted(set(rows))

def _fno_model_family_list_optional(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    rows = []
    for value in values:
        text = str(value)
        if text in {"conv1d_residual", "fno_ar1", "fno_lite"}:
            rows.append(text)
    return _dedupe(rows)

def _training_memory_anchor_list_optional(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    rows = []
    for value in values:
        if not isinstance(value, dict):
            continue
        if value.get("anchor_type") != "operator_training_skill_candidate":
            continue
        if not isinstance(value.get("config"), dict):
            continue
        rows.append(value)
    return rows

def _fno_model_family_values(preferences: dict[str, Any]) -> list[str]:
    values = _fno_model_family_list_optional(preferences.get("model_families"))
    if values:
        return values
    return ["fno_lite"]

def _loss_type_list(values: list[str]) -> list[str]:
    if not values:
        raise Task1ShortWindowTrainingError("loss_types must not be empty")
    rows = []
    for value in values:
        text = str(value)
        if text not in SUPPORTED_FNO_LOSS_TYPES:
            raise Task1ShortWindowTrainingError(f"unsupported loss_type: {text}")
        rows.append(text)
    return _dedupe(rows)

def _normalization_list_optional(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    rows = []
    for value in values:
        text = str(value)
        if text in SUPPORTED_NORMALIZATION_TYPES:
            rows.append(text)
    return sorted(set(rows))

def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]

def _dedupe(values: list[Any]) -> list[Any]:
    rows = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        rows.append(value)
    return rows
