"""Task 1 FNO training policy utils functions."""

from __future__ import annotations

from typing import Any
from pde_agent.core.task1_short_window_common import (
    SUPPORTED_FNO_LOSS_TYPES,
    SUPPORTED_MODEL_FAMILIES,
    SUPPORTED_NORMALIZATION_TYPES,
    Task1ShortWindowTrainingError,
)
def _attempted_fno_config_signatures(evidences: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    signatures = set()
    for evidence in evidences:
        for record in evidence.get("family_records", []) or []:
            if isinstance(record, dict) and isinstance(record.get("config"), dict):
                signatures.add(_fno_config_signature(record["config"]))
        for record in evidence.get("candidate_records", []) or []:
            if isinstance(record, dict) and isinstance(record.get("config"), dict):
                signatures.add(_fno_config_signature(record["config"]))
        constraints = evidence.get("next_generation_constraints")
        if isinstance(constraints, dict):
            for config in constraints.get("fno_training_configs", []) or []:
                if isinstance(config, dict):
                    signatures.add(_fno_config_signature(config))
    return signatures

def _fno_config_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(config.get("model_family", "fno_lite")),
        int(config.get("modes", 0)),
        int(config.get("width", 0)),
        int(config.get("depth", 0)),
        str(config.get("loss_type", "mse")),
        str(config.get("normalization", "none")),
        int(config.get("forecast_steps", 0)),
        int(config.get("train_steps", 0)),
    )

def _preferred_probe_forecast(
    ordered_forecasts: list[int],
    preferences: dict[str, Any],
) -> int:
    if preferences.get("requires_middle_guardrail_source"):
        for value in ordered_forecasts:
            if int(value) >= 47:
                return int(value)
    return int(min(ordered_forecasts))

def _fno_regression_repair_candidates(
    *,
    preferences: dict[str, Any],
    loss_values: list[str],
) -> list[dict[str, Any]]:
    anchor = preferences.get("best_selected_config")
    if not isinstance(anchor, dict):
        return []
    modes = int(anchor.get("modes", 0))
    width = int(anchor.get("width", 0))
    depth = int(anchor.get("depth", 0))
    forecast_steps = int(anchor.get("forecast_steps", 0))
    train_steps = int(anchor.get("train_steps", 0))
    normalization = str(anchor.get("normalization", "none"))
    candidates = []
    for loss_type in loss_values:
        if loss_type == str(anchor.get("loss_type", "")):
            continue
        candidates.append(
            {
                "model_family": "fno_lite",
                "modes": modes,
                "width": width,
                "depth": depth,
                "loss_type": loss_type,
                "normalization": normalization,
                "forecast_steps": forecast_steps,
                "train_steps": train_steps,
                "rationale": [
                    "regression_repair_anchor_capacity",
                    "vary_loss_without_capacity_growth",
                ],
            }
        )
    if depth > 1:
        candidates.append(
            {
                "model_family": "fno_lite",
                "modes": modes,
                "width": width,
                "depth": depth - 1,
                "loss_type": str(anchor.get("loss_type", "early_weighted_mse")),
                "normalization": normalization,
                "forecast_steps": forecast_steps,
                "train_steps": train_steps,
                "rationale": [
                    "regression_repair_reduce_depth_only",
                    "preserve_width_modes_train_budget",
                ],
            }
        )
    if train_steps > 50:
        candidates.append(
            {
                "model_family": "fno_lite",
                "modes": modes,
                "width": width,
                "depth": depth,
                "loss_type": str(anchor.get("loss_type", "early_weighted_mse")),
                "normalization": normalization,
                "forecast_steps": forecast_steps,
                "train_steps": max(train_steps - 50, 1),
                "rationale": [
                    "regression_repair_reduce_train_steps_only",
                    "preserve_capacity_and_loss",
                ],
            }
        )
    return candidates

def _anchor_neighborhood_candidates(anchor: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    try:
        modes = int(anchor.get("modes", 0))
        width = int(anchor.get("width", 0))
        depth = int(anchor.get("depth", 0))
        forecast_steps = int(anchor.get("forecast_steps", 0))
        train_steps = int(anchor.get("train_steps", 0))
    except (TypeError, ValueError):
        return rows
    if min(modes, width, depth, forecast_steps, train_steps) <= 0:
        return rows
    base = {
        "model_family": str(anchor.get("model_family", "fno_lite")),
        "loss_type": str(anchor.get("loss_type", "mse")),
        "normalization": str(anchor.get("normalization", "none")),
        "forecast_steps": forecast_steps,
        "modes": modes,
        "width": width,
        "depth": depth,
        "train_steps": train_steps,
    }
    variants = [
        {
            **base,
            "width": width + 8,
            "rationale": [
                "training_memory_anchor_neighborhood",
                "increase_width_near_anchor",
            ],
        },
        {
            **base,
            "train_steps": train_steps + 2,
            "rationale": [
                "training_memory_anchor_neighborhood",
                "increase_train_steps_near_anchor",
            ],
        },
    ]
    if modes >= 8:
        variants.append(
            {
                **base,
                "modes": max(4, modes - 4),
                "rationale": [
                    "training_memory_anchor_neighborhood",
                    "test_lower_spectral_capacity_near_anchor",
                ],
            }
        )
    if base["loss_type"] != "horizon_weighted_mse":
        variants.append(
            {
                **base,
                "loss_type": "horizon_weighted_mse",
                "rationale": [
                    "training_memory_anchor_neighborhood",
                    "test_horizon_weighted_loss_near_anchor",
                ],
            }
        )
    return variants

def _anchor_family_challenge_candidates(anchor: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        modes = int(anchor.get("modes", 0))
        width = int(anchor.get("width", 0))
        depth = int(anchor.get("depth", 0))
        forecast_steps = int(anchor.get("forecast_steps", 0))
        train_steps = int(anchor.get("train_steps", 0))
    except (TypeError, ValueError):
        return []
    if min(modes, width, depth, forecast_steps, train_steps) <= 0:
        return []
    current_family = str(anchor.get("model_family", "fno_lite"))
    alternatives = [
        family for family in ["fno_ar1", "conv1d_residual", "fno_lite"]
        if family != current_family
    ]
    rows = []
    for family in alternatives[:1]:
        rows.append(
            {
                "model_family": family,
                "loss_type": str(anchor.get("loss_type", "mse")),
                "normalization": str(anchor.get("normalization", "none")),
                "forecast_steps": forecast_steps,
                "modes": modes,
                "width": width,
                "depth": depth,
                "train_steps": train_steps,
                "rationale": [
                    "training_memory_anchor_family_challenge",
                    "compare_anchor_against_alternative_dynamics_family",
                ],
            }
        )
    return rows

def _anchor_only_int_values(
    preferences: dict[str, Any],
    key: str,
    fallback: list[int],
) -> list[int]:
    anchor = preferences.get("best_selected_config")
    if not isinstance(anchor, dict):
        return fallback
    config_key = {"modes": "modes", "widths": "width", "depths": "depth"}.get(key)
    if config_key is None:
        return fallback
    value = int(anchor.get(config_key, 0))
    return [value] if value > 0 else fallback

def _prefers_medium_training_budget(preferences: dict[str, Any]) -> bool:
    train_steps = []
    for item in preferences.get("train_steps", []):
        try:
            train_steps.append(int(item))
        except (TypeError, ValueError):
            continue
    anchor = preferences.get("best_selected_config")
    if isinstance(anchor, dict):
        try:
            train_steps.append(int(anchor.get("train_steps", 0)))
        except (TypeError, ValueError):
            pass
    return any(value >= 100 for value in train_steps)

def _has_expanded_capacity_axis(
    *,
    mode_values: list[int],
    width_values: list[int],
    depth_values: list[int],
    train_values: list[int],
) -> bool:
    return (
        max(mode_values or [0]) >= 12
        or max(width_values or [0]) >= 48
        or max(depth_values or [0]) >= 3
        or max(train_values or [0]) >= 100
    )

def _first_available_loss(values: list[str], *, preferred: tuple[str, ...]) -> str:
    for value in preferred:
        if value in values:
            return value
    return values[0]

def _first_available_string(values: list[str], *, preferred: tuple[str, ...]) -> str:
    for value in preferred:
        if value in values:
            return value
    return values[0]

def _candidate_rationale(
    *, loss_type: str, forecast_steps: int, preferences: dict[str, Any]
) -> list[str]:
    reasons = []
    if str(loss_type) in preferences.get("loss_types", []):
        reasons.append("preferred_by_previous_evidence")
    if str(forecast_steps) in preferences.get("forecast_steps", []):
        reasons.append("matches_refined_horizon_constraint")
    if not reasons:
        reasons.append("coverage_candidate")
    return reasons

def _ordered_values(values: list[Any], preferred_values: list[Any]) -> list[Any]:
    preferred = [str(item) for item in preferred_values]
    return sorted(
        values,
        key=lambda item: (
            0 if str(item) in preferred else 1,
            preferred.index(str(item)) if str(item) in preferred else 999,
            item,
        ),
    )

def _merge_preferred_ints(values: list[int], preferred_values: list[Any]) -> list[int]:
    merged = list(values)
    for item in preferred_values:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value > 0 and value not in merged:
            merged.append(value)
    return sorted(merged)

def _merge_preferred_strings(values: list[str], preferred_values: list[Any]) -> list[str]:
    merged = list(values)
    for item in preferred_values:
        value = str(item)
        if value in SUPPORTED_NORMALIZATION_TYPES and value not in merged:
            merged.append(value)
    return merged

def _training_decision(
    *, loss_curve: dict[str, list[float]], metrics: dict[str, Any]
) -> str:
    initial = float(loss_curve["val"][0])
    final = float(loss_curve["val"][-1])
    early_rel = float(metrics["early"]["rel_mse"])
    if final < initial or early_rel < 1.0:
        return "continue_training"
    if early_rel < 5.0:
        return "change_loss_or_horizon"
    return "abandon_model_family"

def _training_constraints(
    *, model_family: str, decision: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    constraints = {
        "training_model_families": [model_family],
        "prefer_training_objectives": [],
        "refine_training_horizons": [],
        "avoid_training_model_families": [],
    }
    if decision == "continue_training":
        constraints["prefer_training_objectives"].append("short_window_mse")
        constraints["refine_training_horizons"].append("early_window")
    elif decision == "change_loss_or_horizon":
        constraints["prefer_training_objectives"].extend(
            ["early_weighted_mse", "rollout_consistency"]
        )
        constraints["refine_training_horizons"].append("shorter_early_window")
    else:
        constraints["avoid_training_model_families"].append(model_family)
    if metrics["middle_guardrail"]["evaluated"]:
        constraints["refine_training_horizons"].append("middle_guardrail")
    if metrics["late_guardrail"]["evaluated"]:
        constraints["refine_training_horizons"].append("late_guardrail")
    return constraints

def _decision_reasons(*, decision: str, metrics: dict[str, Any]) -> list[str]:
    reasons = [decision]
    if float(metrics["early"]["rel_mse"]) >= 1.0:
        reasons.append("early_rel_mse_high")
    else:
        reasons.append("early_signal_learned")
    if metrics["middle_guardrail"]["evaluated"]:
        reasons.append("middle_guardrail_available")
    if metrics["late_guardrail"]["evaluated"]:
        reasons.append("late_guardrail_available")
    return reasons

def _diversified_model_family_values(values: list[str]) -> list[str]:
    ordered = ["fno_ar1", "conv1d_residual"]
    for family in values:
        if family in SUPPORTED_MODEL_FAMILIES and family not in ordered:
            ordered.append(family)
    return ordered

def _prioritize_new_dynamics_families(values: list[str]) -> list[str]:
    ordered = []
    for family in ["fno_ar1", "conv1d_residual", "fno_lite"]:
        if family in values:
            ordered.append(family)
    return ordered or ["fno_ar1", "conv1d_residual", "fno_lite"]

def _normalize_fno_candidate_config(config: dict[str, Any]) -> dict[str, Any]:
    row = dict(config)
    for key in ["modes", "width", "depth", "forecast_steps", "train_steps"]:
        if key in row:
            row[key] = int(row[key])
    if "model_family" in row:
        row["model_family"] = str(row["model_family"])
    if "loss_type" in row:
        row["loss_type"] = str(row["loss_type"])
    if "normalization" in row:
        row["normalization"] = str(row["normalization"])
    return row

def _normalization_list(values: list[str]) -> list[str]:
    if not values:
        raise Task1ShortWindowTrainingError("normalizations must not be empty")
    rows = []
    for value in values:
        text = str(value)
        if text not in SUPPORTED_NORMALIZATION_TYPES:
            raise Task1ShortWindowTrainingError(f"unsupported normalization: {text}")
        rows.append(text)
    return sorted(set(rows))
