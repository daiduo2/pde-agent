"""Build reusable iteration memory from FNO training optimization trajectories."""

import json
from pathlib import Path
from typing import Any

from pde_agent.core.operator_training_policy import build_operator_training_decision


STATE_HINT = "FNO_TRAINING_MEMORY_BUILT"


class FnoTrainingMemoryError(RuntimeError):
    pass


def build_training_iteration_memory(
    *,
    optimization_report: dict[str, Any],
    iteration_index: int,
    previous_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if optimization_report.get("state_hint") != "FNO_TRAINING_OPTIMIZATION_LOOP_RAN":
        raise FnoTrainingMemoryError("optimization report has invalid state_hint")
    trajectory = _trajectory(optimization_report)
    current_best = _best(optimization_report)
    best = _global_best(current_best=current_best, previous_memory=previous_memory)
    avoid = _avoid_repeating(trajectory=trajectory, best=current_best)
    family_outcomes = _family_outcomes(trajectory=trajectory, best=current_best)
    positive_signals = _positive_signals(trajectory=trajectory)
    search_guidance = _search_guidance(trajectory=trajectory)
    training_memory_anchors = _training_memory_anchors(
        current_best=current_best,
        best=best,
        previous_memory=previous_memory,
    )
    operator_decision = build_operator_training_decision(
        trajectory=trajectory,
        previous_memory=previous_memory,
    )
    decision_policy = operator_decision.get("policy", {})
    search_guidance.extend(_policy_search_guidance(decision_policy))
    avoid.extend(_policy_avoid_repeating(decision_policy))
    convergence_history = _merge_convergence_history(
        previous_memory=previous_memory,
        best=current_best,
    )
    memory = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "iteration_index": iteration_index,
        "source_optimization_report_path": None,
        "current_best": current_best,
        "best_so_far": best,
        "trajectory_summary": [_trajectory_item(item) for item in trajectory],
        "avoid_repeating": avoid,
        "family_outcomes": family_outcomes,
        "positive_signals": positive_signals,
        "training_memory_anchors": training_memory_anchors,
        "search_guidance": search_guidance,
        "operator_training_decision": operator_decision,
        "decision_policy": decision_policy,
        "convergence_history": convergence_history,
        "next_hints": _next_hints(
            best=best,
            avoid=avoid,
            family_outcomes=family_outcomes,
            positive_signals=positive_signals,
            training_memory_anchors=training_memory_anchors,
            search_guidance=search_guidance,
            convergence_history=convergence_history,
        ),
    }
    return memory


def merge_training_memory_with_hypothesis_feedback(
    memory: dict[str, Any],
    hypothesis_report: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(memory)
    feedback_hints: list[str] = []
    if hypothesis_report.get("state_hint") == "HYPOTHESIS_LAB_STRATEGY_FEEDBACK_INTEGRATED":
        feedback = hypothesis_report.get("feedback", {})
        if isinstance(feedback, dict):
            suggested_families = feedback.get("suggested_families")
            if isinstance(suggested_families, list):
                feedback_hints.append(
                    f"Hypothesis lab suggests families: {suggested_families}"
                )
            blocked_families = feedback.get("blocked_families")
            if isinstance(blocked_families, list):
                for family in blocked_families:
                    merged.setdefault("family_outcomes", {})
                    if family not in merged["family_outcomes"]:
                        merged["family_outcomes"][family] = {
                            "candidate_family": family,
                            "status": "blocked",
                            "attempt_count": 0,
                            "failure_reasons": ["blocked_by_hypothesis_lab_feedback"],
                        }
                    else:
                        merged["family_outcomes"][family]["status"] = "blocked"
                        _append_unique(
                            merged["family_outcomes"][family].setdefault(
                                "failure_reasons", []
                            ),
                            "blocked_by_hypothesis_lab_feedback",
                        )
            new_search_guidance = feedback.get("search_guidance")
            if isinstance(new_search_guidance, list):
                existing = set(
                    json.dumps(item, sort_keys=True)
                    for item in merged.get("search_guidance", [])
                    if isinstance(item, dict)
                )
                for item in new_search_guidance:
                    if isinstance(item, dict):
                        key = json.dumps(item, sort_keys=True)
                        if key not in existing:
                            merged.setdefault("search_guidance", []).append(item)
                            existing.add(key)
    existing_hints = merged.get("next_hints", [])
    if feedback_hints:
        merged["next_hints"] = feedback_hints + list(existing_hints)
    return merged


def load_training_iteration_memory(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if value.get("state_hint") != STATE_HINT:
        return None
    return value


def _trajectory(report: dict[str, Any]) -> list[dict[str, Any]]:
    value = report.get("candidate_records")
    if not isinstance(value, list):
        raise FnoTrainingMemoryError("optimization report missing candidate_records")
    result = [item for item in value if isinstance(item, dict)]
    if not result:
        raise FnoTrainingMemoryError("optimization report candidate_records is empty")
    return result


def _best(report: dict[str, Any]) -> dict[str, Any]:
    selected = report.get("selected_candidate")
    if isinstance(selected, dict):
        return {
            "candidate_id": selected.get("candidate_id"),
            "score": selected.get("score"),
            "config": selected.get("config"),
            "model_family": _record_value(selected, "model_family"),
            "loss_type": _record_value(selected, "loss_type"),
            "forecast_steps": _record_value(selected, "forecast_steps"),
            "train_steps": _record_value(selected, "train_steps"),
            "iteration": selected.get("iteration"),
        }
    records = report.get("candidate_records", [])
    if not records:
        raise FnoTrainingMemoryError("optimization report has no candidate_records")
    best_record = max(
        records,
        key=lambda item: float(item.get("score", float("-inf")))
        if isinstance(item.get("score"), int | float)
        else float("-inf"),
    )
    return {
        "candidate_id": best_record.get("candidate_id"),
        "score": best_record.get("score"),
        "config": best_record.get("config"),
        "model_family": _record_value(best_record, "model_family"),
        "loss_type": _record_value(best_record, "loss_type"),
        "forecast_steps": _record_value(best_record, "forecast_steps"),
        "train_steps": _record_value(best_record, "train_steps"),
        "iteration": None,
    }


def _global_best(
    *,
    current_best: dict[str, Any],
    previous_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    previous_best = None
    if isinstance(previous_memory, dict):
        candidate = previous_memory.get("best_so_far")
        if isinstance(candidate, dict):
            previous_best = candidate
    if previous_best is None:
        return current_best
    current_score = _score(current_best)
    previous_score = _score(previous_best)
    if current_score is None:
        return previous_best
    if previous_score is None or current_score > previous_score:
        return current_best
    return previous_best


def _trajectory_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id"),
        "score": item.get("score"),
        "model_family": _record_value(item, "model_family"),
        "loss_type": _record_value(item, "loss_type"),
        "forecast_steps": _record_value(item, "forecast_steps"),
        "train_steps": _record_value(item, "train_steps"),
        "config": item.get("config"),
        "effective_improvement": item.get("effective_improvement"),
        "score_delta_vs_best": item.get("score_delta_vs_best"),
    }


def _avoid_repeating(
    *,
    trajectory: list[dict[str, Any]],
    best: dict[str, Any],
) -> list[dict[str, Any]]:
    avoid: list[dict[str, Any]] = []
    best_id = best.get("candidate_id")
    for item in trajectory:
        if item.get("candidate_id") == best_id:
            continue
        reason = None
        if item.get("rules_passed") is False:
            reason = "rule_gate_failed"
        elif item.get("effective_improvement") is False:
            reason = "below_min_effective_improvement"
        if reason is None:
            continue
        avoid.append(
            {
                "candidate_id": item.get("candidate_id"),
                "model_family": item.get("model_family"),
                "loss_type": item.get("loss_type"),
                "config": item.get("config"),
                "reason": reason,
            }
        )
    return avoid


def _family_outcomes(
    *,
    trajectory: list[dict[str, Any]],
    best: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    best_id = best.get("candidate_id")
    rows: dict[str, dict[str, Any]] = {}
    for item in trajectory:
        family = item.get("model_family")
        if not isinstance(family, str) or not family:
            continue
        score = _score(item)
        row = rows.setdefault(
            family,
            {
                "candidate_family": family,
                "status": "neutral",
                "attempt_count": 0,
                "best_candidate_id": None,
                "best_score": None,
                "worst_candidate_id": None,
                "worst_score": None,
                "failure_reasons": [],
            },
        )
        row["attempt_count"] += 1
        if score is not None and (
            row["best_score"] is None or score > float(row["best_score"])
        ):
            row["best_candidate_id"] = item.get("candidate_id")
            row["best_score"] = score
        if score is not None and (
            row["worst_score"] is None or score < float(row["worst_score"])
        ):
            row["worst_candidate_id"] = item.get("candidate_id")
            row["worst_score"] = score
        if item.get("rules_passed") is False:
            _append_unique(row["failure_reasons"], "rule_gate_failed")
        elif item.get("effective_improvement") is False:
            _append_unique(row["failure_reasons"], "below_min_effective_improvement")
        if item.get("candidate_id") == best_id and score is not None and score > 0.0:
            row["status"] = "positive_signal"
        elif row["status"] != "positive_signal":
            worst_score = row.get("worst_score")
            if isinstance(worst_score, int | float) and worst_score <= 0.0:
                row["status"] = "blocked"
            elif row["failure_reasons"]:
                row["status"] = "weak_or_failed"
    return rows


def _positive_signals(*, trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals = []
    for item in trajectory:
        score = _score(item)
        if score is None or score <= 0.0:
            continue
        signals.append(
            {
                "candidate_id": item.get("candidate_id"),
                "model_family": _record_value(item, "model_family"),
                "loss_type": _record_value(item, "loss_type"),
                "score": score,
                "config": item.get("config"),
            }
        )
    signals.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return signals


def _training_memory_anchors(
    *,
    current_best: dict[str, Any],
    best: dict[str, Any],
    previous_memory: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    anchors = _updated_previous_training_memory_anchors(
        previous_memory=previous_memory,
        current_best=current_best,
    )
    current_score = _score(current_best)
    best_score = _score(best)
    if current_score is None or best_score is None:
        return anchors
    if current_score < best_score:
        return anchors
    config = current_best.get("config")
    if not isinstance(config, dict):
        return anchors
    model_family = str(_record_value(current_best, "model_family") or "")
    loss_type = str(_record_value(current_best, "loss_type") or "")
    if not model_family or not loss_type:
        return anchors
    anchor = {
        "anchor_id": _training_anchor_id(config=config),
        "anchor_type": "operator_training_skill_candidate",
        "status": "active",
        "model_family": model_family,
        "loss_type": loss_type,
        "score": current_score,
        "best_score": current_score,
        "reuse_count": 0,
        "confirmation_count": 1,
        "regression_count": 0,
        "config": config,
        "evidence": {
            "candidate_id": current_best.get("candidate_id"),
            "promotion_reason": "new_global_best_training_candidate",
        },
        "planner_preferences": {
            "model_families": [model_family],
            "loss_types": [loss_type],
            "forecast_steps": _positive_values([config.get("forecast_steps")]),
            "modes": _positive_values([config.get("modes")]),
            "widths": _positive_values([config.get("width")]),
            "depths": _positive_values([config.get("depth")]),
            "train_steps": _positive_values([config.get("train_steps")]),
            "normalizations": [str(config.get("normalization", "none"))],
        },
        "next_actions": [
            "reuse_as_training_anchor",
            "search_local_neighborhood",
            "compare_against_alternative_dynamics_family",
        ],
    }
    return _merge_training_anchors(anchors=anchors, anchor=anchor)


def _previous_training_memory_anchors(
    previous_memory: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(previous_memory, dict):
        return []
    anchors = previous_memory.get("training_memory_anchors")
    if not isinstance(anchors, list):
        return []
    return [item for item in anchors if isinstance(item, dict)]


def _updated_previous_training_memory_anchors(
    *,
    previous_memory: dict[str, Any] | None,
    current_best: dict[str, Any],
) -> list[dict[str, Any]]:
    anchors = _previous_training_memory_anchors(previous_memory)
    current_config = current_best.get("config")
    current_score = _score(current_best)
    if not isinstance(current_config, dict) or current_score is None:
        return anchors
    updated = []
    for anchor in anchors:
        row = dict(anchor)
        anchor_config = row.get("config")
        if not isinstance(anchor_config, dict) or not _same_training_config(
            anchor_config,
            current_config,
        ):
            updated.append(row)
            continue
        row["reuse_count"] = int(row.get("reuse_count", 0)) + 1
        anchor_score = _score(row) or float(row.get("best_score", float("-inf")))
        if current_score >= anchor_score - 0.25:
            row["confirmation_count"] = int(row.get("confirmation_count", 0)) + 1
            row["best_score"] = max(float(row.get("best_score", anchor_score)), current_score)
            row["score"] = row["best_score"]
            row["status"] = "active"
        else:
            row["regression_count"] = int(row.get("regression_count", 0)) + 1
            if int(row["regression_count"]) >= 3:
                row["status"] = "deprecated"
            elif int(row["regression_count"]) >= 2:
                row["status"] = "challenged"
        updated.append(row)
    return updated


def _same_training_config(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = [
        "model_family",
        "loss_type",
        "normalization",
        "forecast_steps",
        "modes",
        "width",
        "depth",
        "train_steps",
    ]
    for key in keys:
        if str(left.get(key)) != str(right.get(key)):
            return False
    return True


def _training_anchor_id(*, config: dict[str, Any]) -> str:
    parts = [
        "training_anchor",
        str(config.get("model_family", "unknown")),
        str(config.get("loss_type", "unknown")),
        f"h{int(config.get('forecast_steps', 0))}",
        f"m{int(config.get('modes', 0))}",
        f"w{int(config.get('width', 0))}",
        f"d{int(config.get('depth', 0))}",
        f"s{int(config.get('train_steps', 0))}",
    ]
    return "_".join(parts)


def _merge_training_anchors(
    *,
    anchors: list[dict[str, Any]],
    anchor: dict[str, Any],
) -> list[dict[str, Any]]:
    anchor_id = anchor.get("anchor_id")
    merged = [
        item for item in anchors
        if item.get("anchor_id") != anchor_id
    ]
    merged.append(anchor)
    merged.sort(
        key=lambda item: float(item.get("score") or float("-inf")),
        reverse=True,
    )
    return merged[:8]


def _positive_values(values: list[Any]) -> list[int]:
    rows = []
    for value in values:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer > 0:
            rows.append(integer)
    return rows


def _search_guidance(*, trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guidance: list[dict[str, Any]] = []
    for item in trajectory:
        if item.get("effective_improvement") is False:
            family = item.get("model_family")
            loss_type = item.get("loss_type")
            if family == "fno_lite" and loss_type == "mse":
                guidance.append(
                    {
                        "guidance_type": "try_alternative_loss",
                        "reason": "fno_lite with mse failed to improve",
                        "candidate_id": item.get("candidate_id"),
                        "recommended_action": "try_early_weighted_or_spectral_loss",
                    }
                )
            elif family == "conv1d_residual":
                guidance.append(
                    {
                        "guidance_type": "consider_capacity_increase",
                        "reason": "conv1d_residual failed to improve",
                        "candidate_id": item.get("candidate_id"),
                        "recommended_action": "try_fno_family_with_more_modes",
                    }
                )
    return guidance


def _policy_search_guidance(policy: Any) -> list[dict[str, Any]]:
    if not isinstance(policy, dict):
        return []
    guidance = [
        {
            "guidance_type": "operator_training_policy",
            "next_action": policy.get("next_action"),
            "required_axis_changes": policy.get("required_axis_changes", []),
            "preferred_loss_types": policy.get("preferred_loss_types", []),
            "preferred_model_families": policy.get("preferred_model_families", []),
            "preferred_forecast_steps": policy.get("preferred_forecast_steps", []),
            "preferred_normalizations": policy.get("preferred_normalizations", []),
            "budget_adjustments": policy.get("budget_adjustments", {}),
            "hypothesis_lab_requests": policy.get("hypothesis_lab_requests", []),
            "reason_codes": policy.get("reason_codes", []),
            "recommended_action": policy.get("next_action"),
        }
    ]
    return guidance


def _policy_avoid_repeating(policy: Any) -> list[dict[str, Any]]:
    if not isinstance(policy, dict):
        return []
    rows = []
    for config in policy.get("avoid_exact_configs", []):
        if not isinstance(config, dict):
            continue
        if not config.get("model_family") or not config.get("loss_type"):
            continue
        rows.append(
            {
                "candidate_id": None,
                "model_family": config.get("model_family"),
                "loss_type": config.get("loss_type"),
                "config": config,
                "reason": "operator_training_policy_avoid_exact_config",
            }
        )
    return rows


def _merge_convergence_history(
    *,
    previous_memory: dict[str, Any] | None,
    best: dict[str, Any],
) -> dict[str, Any]:
    history = {
        "no_improvement_count": 0,
        "best_score_history": [],
    }
    if previous_memory is not None:
        prev_history = previous_memory.get("convergence_history")
        if isinstance(prev_history, dict):
            prev_count = prev_history.get("no_improvement_count")
            if isinstance(prev_count, int):
                history["no_improvement_count"] = prev_count
            prev_scores = prev_history.get("best_score_history")
            if isinstance(prev_scores, list):
                history["best_score_history"] = list(prev_scores)
    current_score = best.get("score")
    if isinstance(current_score, int | float):
        history["best_score_history"].append(float(current_score))
        if len(history["best_score_history"]) >= 2:
            prev_best = max(history["best_score_history"][:-1])
            if float(current_score) <= prev_best:
                history["no_improvement_count"] += 1
            else:
                history["no_improvement_count"] = 0
    return history


def _next_hints(
    *,
    best: dict[str, Any],
    avoid: list[dict[str, Any]],
    family_outcomes: dict[str, dict[str, Any]],
    positive_signals: list[dict[str, Any]],
    training_memory_anchors: list[dict[str, Any]],
    search_guidance: list[dict[str, Any]],
    convergence_history: dict[str, Any],
) -> list[str]:
    hints = []
    model_family = best.get("model_family")
    if model_family:
        hints.append(f"Best family so far: {model_family}")
    if search_guidance:
        hints.append("Review search_guidance before repeating failed families.")
    blocked = [
        family
        for family, row in family_outcomes.items()
        if row.get("status") == "blocked"
    ]
    if blocked:
        hints.append(f"Blocked families: {blocked}")
    if positive_signals:
        hints.append("Promote positive signals into next candidate plan.")
    if training_memory_anchors:
        hints.append("Reuse training_memory_anchors before broad exploration.")
    if avoid:
        hints.append("Exclude avoid_repeating configs unless new evidence changes.")
    no_improvement = convergence_history.get("no_improvement_count", 0)
    if no_improvement >= 2:
        hints.append(
            f"No improvement for {no_improvement} iterations; consider exploration expansion."
        )
    return hints


def _score(item: dict[str, Any]) -> float | None:
    value = item.get("score")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _record_value(item: dict[str, Any], key: str) -> Any:
    config = item.get("config")
    if isinstance(config, dict) and config.get(key) is not None:
        return config.get(key)
    value = item.get(key)
    if value is not None:
        return value
    return None


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)
