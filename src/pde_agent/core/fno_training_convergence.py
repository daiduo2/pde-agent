"""Convergence, stagnation, and doom-loop detection for FNO training iterations."""

from typing import Any


STATE_HINT = "FNO_TRAINING_CONVERGENCE_CHECKED"


class FnoTrainingConvergenceError(RuntimeError):
    pass


def check_training_convergence(
    *,
    memory: dict[str, Any],
    target_score: float | None = None,
    patience: int = 3,
    min_score_delta: float = 0.1,
) -> dict[str, Any]:
    if not isinstance(memory, dict):
        raise FnoTrainingConvergenceError("memory must be a dict")
    if memory.get("state_hint") != "FNO_TRAINING_MEMORY_BUILT":
        raise FnoTrainingConvergenceError("memory has invalid state_hint")
    trajectory = memory.get("trajectory_summary", [])
    if not isinstance(trajectory, list):
        raise FnoTrainingConvergenceError("memory missing trajectory_summary")
    best = memory.get("best_so_far", {})
    convergence_history = memory.get("convergence_history", {})
    no_improvement_count = _int_or_default(
        convergence_history.get("no_improvement_count"), 0
    )
    best_score = _float_or_none(best.get("score"))
    family_outcomes = memory.get("family_outcomes", {})

    triggers: list[dict[str, Any]] = []
    blocked_families: list[str] = []
    recommended_actions: list[dict[str, Any]] = []
    decision = "continue"
    reason = "iteration_completed_with_room_for_improvement"
    confidence = 0.5

    if target_score is not None and best_score is not None and best_score >= target_score:
        decision = "converged"
        reason = f"target_score_met ({best_score:.4f} >= {target_score:.4f})"
        confidence = 0.95
        recommended_actions.append(
            {"action": "stop", "reason": "target_score_met", "params": {}}
        )
        return _build_report(
            decision=decision,
            reason=reason,
            confidence=confidence,
            recommended_actions=recommended_actions,
            blocked_families=blocked_families,
            triggers=triggers,
        )

    doom_loop = check_training_doom_loop(
        trajectory=trajectory,
        best_score_history=convergence_history.get("best_score_history", []),
        no_improvement_count=no_improvement_count,
        min_score_delta=min_score_delta,
    )
    if doom_loop.get("guard_active"):
        triggers.extend(doom_loop.get("triggers", []))
        blocked_families.extend(doom_loop.get("blocked_candidate_families", []))
        decision = "doom_loop"
        reason = doom_loop.get("reason", "training_doom_loop_detected")
        confidence = 0.85
        recommended_actions.append(
            {
                "action": "escalate_to_hypothesis_lab",
                "reason": doom_loop.get("reason"),
                "params": {"blocked_families": blocked_families},
            }
        )
        return _build_report(
            decision=decision,
            reason=reason,
            confidence=confidence,
            recommended_actions=recommended_actions,
            blocked_families=blocked_families,
            triggers=triggers,
        )

    all_stagnant = _all_families_stagnant(family_outcomes)
    if all_stagnant and len(family_outcomes) >= 2:
        decision = "stagnant"
        reason = "all_families_stagnant"
        confidence = 0.8
        recommended_actions.append(
            {
                "action": "escalate_to_hypothesis_lab",
                "reason": "all_families_stagnant",
                "params": {},
            }
        )
        return _build_report(
            decision=decision,
            reason=reason,
            confidence=confidence,
            recommended_actions=recommended_actions,
            blocked_families=blocked_families,
            triggers=triggers,
        )

    if no_improvement_count >= patience:
        decision = "stagnant"
        reason = f"no_improvement_for_{no_improvement_count}_iterations"
        confidence = 0.75
        recommended_actions.append(
            {
                "action": "expand_search_radius",
                "reason": "patience_exhausted",
                "params": {
                    "train_steps_delta": 4,
                    "patience_delta": 1,
                },
            }
        )
        recommended_actions.append(
            {
                "action": "escalate_to_hypothesis_lab",
                "reason": "patience_exhausted",
                "params": {},
            }
        )
        return _build_report(
            decision=decision,
            reason=reason,
            confidence=confidence,
            recommended_actions=recommended_actions,
            blocked_families=blocked_families,
            triggers=triggers,
        )

    recommended_actions.append(
        {
            "action": "continue",
            "reason": reason,
            "params": {},
        }
    )
    return _build_report(
        decision=decision,
        reason=reason,
        confidence=confidence,
        recommended_actions=recommended_actions,
        blocked_families=blocked_families,
        triggers=triggers,
    )


def check_training_doom_loop(
    *,
    trajectory: list[dict[str, Any]],
    best_score_history: list[Any] | None = None,
    no_improvement_count: int = 0,
    min_score_delta: float = 0.1,
) -> dict[str, Any]:
    triggers: list[dict[str, Any]] = []
    blocked: list[str] = []
    reason = ""

    score_history = [
        float(item)
        for item in (best_score_history or [])
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]
    if len(score_history) >= 4 and no_improvement_count >= 2:
        repeated_family = _repeated_non_improving_family(
            trajectory,
            threshold=3,
            min_score_delta=min_score_delta,
        )
        if repeated_family is not None:
            triggers.append(repeated_family)
            blocked.append(repeated_family["candidate_family"])
            reason = f"repeated_non_improving_family: {repeated_family['candidate_family']}"

    if (
        len(score_history) >= 4
        and no_improvement_count >= 2
        and not _score_history_improved(score_history, min_score_delta)
    ):
        capacity_regression = _capacity_expansion_regression(trajectory)
        if capacity_regression is not None:
            triggers.append(capacity_regression)
            reason = capacity_regression.get("reason", "capacity_expansion_regression")

    loss_exhaustion = _loss_type_exhaustion(trajectory)
    if loss_exhaustion is not None:
        triggers.append(loss_exhaustion)
        if loss_exhaustion.get("candidate_family") and loss_exhaustion["candidate_family"] not in blocked:
            blocked.append(loss_exhaustion["candidate_family"])
        reason = loss_exhaustion.get("reason", "loss_type_exhaustion")

    return {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_DOOM_LOOP_CHECKED",
        "guard_active": bool(triggers),
        "triggers": triggers,
        "blocked_candidate_families": sorted(set(blocked)),
        "reason": reason,
    }


def detect_overfitting(*, loss_curves: dict[str, list[float]]) -> bool:
    train = loss_curves.get("train")
    val = loss_curves.get("val")
    if not isinstance(train, list) or not isinstance(val, list):
        return False
    if len(train) < 3 or len(val) < 3:
        return False
    window = min(3, len(train))
    train_trend = train[-window] - train[-1]
    val_trend = val[-1] - val[-window]
    return train_trend > 0 and val_trend > 0


def _repeated_non_improving_family(
    trajectory: list[dict[str, Any]],
    threshold: int,
    min_score_delta: float,
) -> dict[str, Any] | None:
    run: list[dict[str, Any]] = []
    for item in trajectory:
        family = item.get("model_family")
        if not isinstance(family, str):
            continue
        if run and run[-1].get("model_family") != family:
            run = []
        run.append(item)
        if len(run) >= threshold and _all_non_improving(run[-threshold:], min_score_delta):
            return {
                "trigger": "repeated_non_improving_family",
                "candidate_family": family,
                "count": len(run),
                "min_score_delta": min_score_delta,
            }
    return None


def _all_non_improving(
    items: list[dict[str, Any]],
    min_score_delta: float,
) -> bool:
    for item in items:
        delta = item.get("score_delta_vs_best")
        if isinstance(delta, int | float) and float(delta) >= min_score_delta:
            return False
    return True


def _score_history_improved(scores: list[float], min_score_delta: float) -> bool:
    if len(scores) < 2:
        return False
    return scores[-1] - max(scores[:-1]) >= min_score_delta


def _capacity_expansion_regression(
    trajectory: list[dict[str, Any]],
) -> dict[str, Any] | None:
    fno_records = [
        item for item in trajectory
        if isinstance(item.get("model_family"), str)
        and item.get("model_family") in {"fno_ar1", "fno_lite"}
    ]
    if len(fno_records) < 2:
        return None
    scores: list[tuple[int, float]] = []
    for item in fno_records:
        config = item.get("config")
        if not isinstance(config, dict):
            continue
        modes = config.get("modes")
        width = config.get("width")
        score = _float_or_none(item.get("score"))
        if not isinstance(modes, int) or not isinstance(width, int) or score is None:
            continue
        capacity = modes * width
        scores.append((capacity, score))
    if len(scores) < 2:
        return None
    scores.sort(key=lambda x: x[0])
    for i in range(1, len(scores)):
        if scores[i][0] > scores[i - 1][0] and scores[i][1] < scores[i - 1][1]:
            return {
                "trigger": "capacity_expansion_regression",
                "candidate_family": "fno_lite",
                "reason": "larger_model_worse_than_smaller",
                "smaller_capacity": scores[i - 1][0],
                "larger_capacity": scores[i][0],
                "smaller_score": scores[i - 1][1],
                "larger_score": scores[i][1],
            }
    return None


def _loss_type_exhaustion(
    trajectory: list[dict[str, Any]],
) -> dict[str, Any] | None:
    from pde_agent.core.task1_short_window_training import SUPPORTED_FNO_LOSS_TYPES

    family_losses: dict[str, set[str]] = {}
    family_improved: dict[str, bool] = {}
    for item in trajectory:
        family = item.get("model_family")
        loss_type = item.get("loss_type")
        if not isinstance(family, str) or not isinstance(loss_type, str):
            continue
        family_losses.setdefault(family, set()).add(loss_type)
        improved = item.get("effective_improvement")
        if improved is True:
            family_improved[family] = True
    for family, losses in family_losses.items():
        if losses.issuperset(SUPPORTED_FNO_LOSS_TYPES) and not family_improved.get(family):
            return {
                "trigger": "loss_type_exhaustion",
                "candidate_family": family,
                "reason": f"all_loss_types_tried_without_improvement_for_{family}",
                "tried_losses": sorted(losses),
            }
    return None


def _all_families_stagnant(family_outcomes: dict[str, Any]) -> bool:
    if not family_outcomes:
        return False
    for row in family_outcomes.values():
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status in {"positive_signal", "near_miss"}:
            return False
    return True


def _build_report(
    *,
    decision: str,
    reason: str,
    confidence: float,
    recommended_actions: list[dict[str, Any]],
    blocked_families: list[str],
    triggers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "recommended_actions": recommended_actions,
        "blocked_families": blocked_families,
        "suggested_budget_adjustment": {
            "train_steps_delta": 4 if decision == "stagnant" else 0,
            "patience_delta": 1 if decision == "stagnant" else 0,
        },
        "triggers": triggers,
        "next_command": "run-fno-training-iteration",
    }


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default
