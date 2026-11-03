"""Task 2 FNO training convergence checker.

Standalone module extracted from task2_fno_training_closed_loop.
"""
from __future__ import annotations

from typing import Any

STATE_HINT = "TASK2_FNO_TRAINING_CONVERGENCE_CHECKED"
MEMORY_STATE_HINT = "TASK2_FNO_TRAINING_MEMORY_BUILT"


class Task2FnoTrainingConvergenceError(RuntimeError):
    pass


def check_task2_fno_training_convergence(
    *,
    memory: dict[str, Any],
    target_score: float | None = None,
    patience: int = 3,
    min_score_delta: float = 0.5,
) -> dict[str, Any]:
    """Check if Task 2 training has converged."""
    if not isinstance(memory, dict):
        raise Task2FnoTrainingConvergenceError("memory must be a dict")
    if memory.get("state_hint") != MEMORY_STATE_HINT:
        raise Task2FnoTrainingConvergenceError("memory has invalid state_hint")

    trajectory = memory.get("trajectory_summary", [])
    convergence_history = memory.get("convergence_history", {})
    no_improvement_count = _int_or_default(
        convergence_history.get("no_improvement_count"), 0
    )
    best_score = _float_or_none(memory.get("best_so_far", {}).get("score"))

    triggers: list[dict[str, Any]] = []
    recommended_actions: list[dict[str, Any]] = []
    decision = "continue"
    reason = "iteration_completed_with_room_for_improvement"
    confidence = 0.5

    if target_score is not None and best_score is not None and best_score >= target_score:
        decision = "converged"
        reason = f"target_score_met ({best_score:.4f} >= {target_score:.4f})"
        confidence = 0.95
        recommended_actions.append({"action": "stop", "reason": "target_score_met", "params": {}})
        return _build_convergence_report(
            decision=decision,
            reason=reason,
            confidence=confidence,
            recommended_actions=recommended_actions,
            triggers=triggers,
        )

    # Doom loop: repeated same config without improvement
    if no_improvement_count >= 2 and len(trajectory) >= 3:
        repeated = _repeated_non_improving_config(trajectory, min_score_delta=min_score_delta)
        if repeated is not None:
            triggers.append(repeated)
            decision = "doom_loop"
            reason = repeated.get("reason", "repeated_non_improving_config")
            confidence = 0.85
            recommended_actions.append({
                "action": "escalate_to_hypothesis_lab",
                "reason": reason,
                "params": {"blocked_config_signature": repeated.get("config_signature")},
            })
            return _build_convergence_report(
                decision=decision,
                reason=reason,
                confidence=confidence,
                recommended_actions=recommended_actions,
                triggers=triggers,
            )

    if no_improvement_count >= patience:
        decision = "stagnant"
        reason = f"no_improvement_for_{no_improvement_count}_iterations"
        confidence = 0.75
        recommended_actions.append({
            "action": "expand_search_radius",
            "reason": "patience_exhausted",
            "params": {"train_steps_delta": 2, "patience_delta": 1},
        })
        recommended_actions.append({
            "action": "escalate_to_hypothesis_lab",
            "reason": "patience_exhausted",
            "params": {},
        })
        return _build_convergence_report(
            decision=decision,
            reason=reason,
            confidence=confidence,
            recommended_actions=recommended_actions,
            triggers=triggers,
        )

    recommended_actions.append({
        "action": "continue",
        "reason": reason,
        "params": {},
    })
    return _build_convergence_report(
        decision=decision,
        reason=reason,
        confidence=confidence,
        recommended_actions=recommended_actions,
        triggers=triggers,
    )


def _build_convergence_report(
    *,
    decision: str,
    reason: str,
    confidence: float,
    recommended_actions: list[dict[str, Any]],
    triggers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "recommended_actions": recommended_actions,
        "triggers": triggers,
        "next_command": "run-task2-fno-training-iteration",
    }


def _repeated_non_improving_config(
    trajectory: list[dict[str, Any]],
    min_score_delta: float,
) -> dict[str, Any] | None:
    if len(trajectory) < 3:
        return None
    sigs: list[str] = []
    for item in trajectory:
        config = item.get("config")
        if isinstance(config, dict):
            sigs.append(_config_signature(config))
        else:
            sigs.append("")
    # Check if last 3 have same signature and no improvement
    if len(sigs) >= 3 and sigs[-1] and sigs[-1] == sigs[-2] == sigs[-3]:
        scores = [_float_or_none(item.get("score")) for item in trajectory[-3:]]
        if all(s is not None for s in scores):
            if max(scores) - min(scores) < min_score_delta:
                return {
                    "trigger": "repeated_non_improving_config",
                    "config_signature": sigs[-1],
                    "reason": "same_config_repeated_without_improvement",
                    "scores": scores,
                }
    return None


def _config_signature(config: dict[str, Any]) -> str:
    return (
        f"h{config.get('hidden_channels')}"
        f"_m{config.get('modes')}"
        f"_d{config.get('depth')}"
        f"_pw{config.get('physics_weight')}"
        f"_lr{config.get('learning_rate')}"
        f"_s{config.get('train_steps')}"
        f"_b{config.get('batch_size')}"
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default
