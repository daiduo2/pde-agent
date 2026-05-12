"""Generic diagnosis and policy contract for operator training loops."""

from __future__ import annotations

from typing import Any


STATE_HINT = "OPERATOR_TRAINING_DECISION_BUILT"


def build_operator_training_decision(
    *,
    trajectory: list[dict[str, Any]],
    previous_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best = _best_record(trajectory)
    observations = _observations(best=best, previous_memory=previous_memory)
    diagnosis = _diagnosis(
        best=best,
        observations=observations,
        previous_memory=previous_memory,
    )
    policy = _policy(best=best, diagnosis=diagnosis)
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "observations": observations,
        "diagnosis": diagnosis,
        "policy": policy,
    }


def _best_record(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    records = [item for item in trajectory if isinstance(item, dict)]
    if not records:
        return {}
    return max(records, key=lambda item: _float_or_default(item.get("score"), float("-inf")))


def _observations(
    *,
    best: dict[str, Any],
    previous_memory: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score = _float_or_none(best.get("score"))
    if score is not None:
        rows.append({"code": "best_score_observed", "value": score})
    early_rel = _float_or_none(best.get("early_rel_mse"))
    if early_rel is not None:
        rows.append({"code": "early_rel_mse_observed", "value": early_rel})
    train_curve, val_curve = _loss_curves(best)
    if len(train_curve) >= 2:
        rows.append(
            {
                "code": "train_loss_delta_observed",
                "value": train_curve[-1] - train_curve[0],
            }
        )
    if len(val_curve) >= 2:
        rows.append(
            {
                "code": "val_loss_delta_observed",
                "value": val_curve[-1] - val_curve[0],
            }
        )
    previous_score = _previous_best_score(previous_memory)
    if score is not None and previous_score is not None:
        rows.append(
            {
                "code": "score_delta_vs_previous_observed",
                "value": score - previous_score,
            }
        )
    no_improvement = _no_improvement_count(previous_memory)
    if no_improvement:
        rows.append({"code": "no_improvement_count_observed", "value": no_improvement})
    return rows


def _diagnosis(
    *,
    best: dict[str, Any],
    observations: list[dict[str, Any]],
    previous_memory: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    early_rel = _float_or_default(best.get("early_rel_mse"), 5.0)
    train_curve, val_curve = _loss_curves(best)
    train_drop = train_curve[0] - train_curve[-1] if len(train_curve) >= 2 else 0.0
    val_drop = val_curve[0] - val_curve[-1] if len(val_curve) >= 2 else 0.0
    score_delta = _observation_value(observations, "score_delta_vs_previous_observed")
    no_improvement = _no_improvement_count(previous_memory)
    config = best.get("config") if isinstance(best.get("config"), dict) else {}
    train_steps = _int_or_default(config.get("train_steps"), 0)
    width = _int_or_default(config.get("width"), 0)
    modes = _int_or_default(config.get("modes"), 0)
    depth = _int_or_default(config.get("depth"), 0)

    if early_rel >= 1.0 and (train_drop > 0.0 or val_drop > 0.0):
        rows.append(
            {
                "code": "undertrained",
                "severity": "medium",
                "evidence": {
                    "early_rel_mse": early_rel,
                    "train_loss_drop": train_drop,
                    "val_loss_drop": val_drop,
                },
            }
        )
    if early_rel >= 1.0 and train_drop > 0.0 and val_drop <= 0.0:
        rows.append(
            {
                "code": "loss_score_mismatch",
                "severity": "high",
                "evidence": {
                    "early_rel_mse": early_rel,
                    "train_loss_drop": train_drop,
                    "val_loss_drop": val_drop,
                },
            }
        )
    if score_delta is not None and score_delta < -0.05:
        rows.append(
            {
                "code": "score_regression",
                "severity": "high",
                "evidence": {"score_delta_vs_previous": score_delta},
            }
        )
    if no_improvement >= 2:
        rows.append(
            {
                "code": "stagnation",
                "severity": "high",
                "evidence": {"no_improvement_count": no_improvement},
            }
        )
    if early_rel >= 2.0:
        rows.append(
            {
                "code": "short_horizon_dynamics_gap",
                "severity": "high",
                "evidence": {"early_rel_mse": early_rel},
            }
        )
    if (
        0.5 <= early_rel < 1.0
        and train_steps >= 50
        and train_drop <= 0.05
        and val_drop <= 0.05
    ):
        rows.append(
            {
                "code": "capacity_limited",
                "severity": "medium",
                "evidence": {
                    "early_rel_mse": early_rel,
                    "train_steps": train_steps,
                    "modes": modes,
                    "width": width,
                    "depth": depth,
                    "train_loss_drop": train_drop,
                    "val_loss_drop": val_drop,
                },
            }
        )
    if not rows:
        rows.append(
            {
                "code": "continue_promising_direction",
                "severity": "low",
                "evidence": {"best_candidate_id": best.get("candidate_id")},
            }
        )
    return rows


def _policy(
    *,
    best: dict[str, Any],
    diagnosis: list[dict[str, Any]],
) -> dict[str, Any]:
    codes = {str(item.get("code")) for item in diagnosis if isinstance(item, dict)}
    config = best.get("config") if isinstance(best.get("config"), dict) else {}
    policy = {
        "next_action": "continue_current_direction",
        "preferred_model_families": _compact_strings([config.get("model_family")]),
        "preferred_loss_types": _compact_strings([config.get("loss_type")]),
        "preferred_forecast_steps": _compact_ints([config.get("forecast_steps")]),
        "preferred_normalizations": _compact_strings([config.get("normalization")]),
        "required_axis_changes": [],
        "avoid_exact_configs": [config] if config else [],
        "budget_adjustments": {},
        "hypothesis_lab_requests": [],
        "reason_codes": sorted(codes),
    }
    if "undertrained" in codes:
        policy["next_action"] = "increase_training_budget"
        policy["required_axis_changes"].append("train_steps")
        policy["budget_adjustments"]["train_steps_delta"] = 2
    if "loss_score_mismatch" in codes:
        policy["next_action"] = "change_loss_or_horizon"
        policy["required_axis_changes"].extend(["loss_type", "forecast_steps"])
        policy["preferred_loss_types"] = _dedupe(
            ["early_weighted_mse", "horizon_weighted_mse", *policy["preferred_loss_types"]]
        )
    if "score_regression" in codes:
        policy["next_action"] = "repair_regression"
        policy["required_axis_changes"].append("capacity")
    if "short_horizon_dynamics_gap" in codes:
        policy["next_action"] = "request_new_dynamics_source"
        policy["required_axis_changes"].append("model_family")
        policy["preferred_model_families"] = _dedupe(
            ["fno_ar1", "conv1d_residual", *policy["preferred_model_families"]]
        )
        policy["hypothesis_lab_requests"].append(
            {
                "request_type": "new_dynamics_source",
                "reason": "early_rel_mse_remains_high",
            }
        )
    if "capacity_limited" in codes:
        policy["next_action"] = "increase_capacity"
        policy["required_axis_changes"].append("capacity")
        policy["budget_adjustments"]["width_delta"] = max(
            int(policy["budget_adjustments"].get("width_delta", 0)),
            8,
        )
    if "stagnation" in codes:
        policy["next_action"] = "expand_search_radius"
        policy["required_axis_changes"].extend(["model_family", "loss_type", "capacity"])
        policy["preferred_model_families"] = _dedupe(
            ["fno_ar1", "conv1d_residual", *policy["preferred_model_families"]]
        )
        policy["preferred_loss_types"] = _dedupe(
            ["early_weighted_mse", "horizon_weighted_mse", *policy["preferred_loss_types"]]
        )
        policy["budget_adjustments"]["width_delta"] = 8
        policy["hypothesis_lab_requests"].append(
            {
                "request_type": "expand_candidate_family",
                "reason": "training_memory_stagnation",
            }
        )
    policy["required_axis_changes"] = _dedupe(policy["required_axis_changes"])
    return policy


def _loss_curves(record: dict[str, Any]) -> tuple[list[float], list[float]]:
    curve = record.get("loss_curve")
    if not isinstance(curve, dict):
        return [], []
    train = [_float_or_default(item, 0.0) for item in curve.get("train", [])]
    val = [_float_or_default(item, 0.0) for item in curve.get("val", [])]
    return train, val


def _previous_best_score(memory: dict[str, Any] | None) -> float | None:
    if not isinstance(memory, dict):
        return None
    best = memory.get("best_so_far")
    if isinstance(best, dict):
        return _float_or_none(best.get("score"))
    return None


def _no_improvement_count(memory: dict[str, Any] | None) -> int:
    if not isinstance(memory, dict):
        return 0
    history = memory.get("convergence_history")
    if not isinstance(history, dict):
        return 0
    value = history.get("no_improvement_count")
    return int(value) if isinstance(value, int) and value > 0 else 0


def _observation_value(observations: list[dict[str, Any]], code: str) -> float | None:
    for item in observations:
        if item.get("code") == code:
            return _float_or_none(item.get("value"))
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _float_or_default(value: Any, default: float) -> float:
    numeric = _float_or_none(value)
    return default if numeric is None else numeric


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _compact_strings(values: list[Any]) -> list[str]:
    return _dedupe([str(value) for value in values if isinstance(value, str) and value])


def _compact_ints(values: list[Any]) -> list[int]:
    rows = []
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            rows.append(value)
    return _dedupe(rows)


def _dedupe(values: list[Any]) -> list[Any]:
    rows = []
    for value in values:
        if value not in rows:
            rows.append(value)
    return rows
