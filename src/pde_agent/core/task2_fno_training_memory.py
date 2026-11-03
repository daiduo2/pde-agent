"""Task 2 FNO training memory builder and loader.

Extracted from task2_fno_training_closed_loop so that memory logic can be
imported and used standalone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_HINT = "TASK2_FNO_TRAINING_MEMORY_BUILT"


class Task2FnoTrainingMemoryError(RuntimeError):
    pass


def build_task2_fno_training_memory(
    *,
    optimization_report: dict[str, Any],
    iteration_index: int,
    previous_memory: dict[str, Any] | None = None,
    source_optimization_report_path: str | None = None,
) -> dict[str, Any]:
    """Build iteration memory from optimization report."""
    if optimization_report.get("state_hint") != "TASK2_FNO_OPTIMIZATION_LOOP_RAN":
        raise Task2FnoTrainingMemoryError("optimization report has invalid state_hint")
    records = optimization_report.get("candidate_records", [])
    if not isinstance(records, list):
        raise Task2FnoTrainingMemoryError("missing candidate_records")

    current_best = _best_from_records(records)
    best_so_far = _global_best(current_best=current_best, previous_memory=previous_memory)
    trajectory = [_trajectory_item(r) for r in records if isinstance(r, dict)]
    blocked_configs = _blocked_configs(trajectory=trajectory, best=current_best)
    search_guidance = _search_guidance(trajectory=trajectory)
    convergence_history = _merge_convergence_history(
        previous_memory=previous_memory,
        best=current_best,
    )

    memory = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "iteration_index": iteration_index,
        "source_optimization_report_path": source_optimization_report_path,
        "current_best": current_best,
        "best_so_far": best_so_far,
        "trajectory_summary": trajectory,
        "blocked_configs": blocked_configs,
        "search_guidance": search_guidance,
        "convergence_history": convergence_history,
        "next_hints": _memory_next_hints(
            best=best_so_far,
            blocked_configs=blocked_configs,
            search_guidance=search_guidance,
            convergence_history=convergence_history,
        ),
    }
    return memory


def load_task2_fno_training_memory(path: str | Path) -> dict[str, Any] | None:
    """Load a memory file if it exists, is valid JSON, and has the correct state_hint."""
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


# -- Internal helpers --


def _best_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(
        records,
        key=lambda item: float(item.get("score", float("-inf")))
        if isinstance(item.get("score"), int | float)
        else float("-inf"),
    )
    return {
        "candidate_id": best.get("candidate_id"),
        "score": best.get("score"),
        "config": best.get("config"),
        "candidate_family": best.get("candidate_family"),
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
    current_score = _float_or_none(current_best.get("score"))
    previous_score = _float_or_none(previous_best.get("score"))
    if current_score is None:
        return previous_best
    if previous_score is None or current_score > previous_score:
        return current_best
    return previous_best


def _trajectory_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id"),
        "score": item.get("score"),
        "segment_score": item.get("segment_score"),
        "candidate_family": item.get("candidate_family"),
        "config": item.get("config"),
    }


def _blocked_configs(
    *,
    trajectory: list[dict[str, Any]],
    best: dict[str, Any],
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    best_id = best.get("candidate_id")
    for item in trajectory:
        if item.get("candidate_id") == best_id:
            continue
        score = _float_or_none(item.get("score"))
        if score is not None and score <= 0.0:
            blocked.append({
                "candidate_id": item.get("candidate_id"),
                "config": item.get("config"),
                "reason": "zero_or_negative_score",
            })
    return blocked


def _search_guidance(
    *,
    trajectory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    guidance: list[dict[str, Any]] = []
    for item in trajectory:
        score = _float_or_none(item.get("score"))
        if score is None or score <= 0.0:
            config = item.get("config", {})
            guidance.append({
                "guidance_type": "avoid_config",
                "reason": "poor_performance",
                "config_signature": _config_signature(config),
            })
    return guidance


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


def _memory_next_hints(
    *,
    best: dict[str, Any],
    blocked_configs: list[dict[str, Any]],
    search_guidance: list[dict[str, Any]],
    convergence_history: dict[str, Any],
) -> list[str]:
    hints = []
    score = best.get("score")
    if score is not None:
        hints.append(f"Best score so far: {score}")
    if search_guidance:
        hints.append("Review search_guidance before planning next iteration.")
    if blocked_configs:
        hints.append(f"Blocked configs: {len(blocked_configs)}")
    no_improvement = convergence_history.get("no_improvement_count", 0)
    if no_improvement >= 2:
        hints.append(
            f"No improvement for {no_improvement} iterations; consider exploration expansion."
        )
    return hints


def merge_task2_training_memory_with_hypothesis_feedback(
    memory: dict[str, Any],
    hypothesis_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge hypothesis lab feedback into Task 2 training memory."""
    merged = dict(memory)
    if not isinstance(hypothesis_report, dict):
        return merged
    feedback_hints: list[str] = []
    strategy_recommendations = hypothesis_report.get("strategy_recommendations")
    if isinstance(strategy_recommendations, list):
        for rec in strategy_recommendations:
            if isinstance(rec, dict):
                actions = rec.get("actions", [])
                if actions:
                    feedback_hints.append(f"Strategy '{rec.get('strategy')}': {actions[0]}")
    blocked_configs = hypothesis_report.get("blocked_configs")
    if isinstance(blocked_configs, list):
        existing_blocked = merged.setdefault("blocked_configs", [])
        existing_ids = {b.get("candidate_id") for b in existing_blocked if isinstance(b, dict)}
        for bc in blocked_configs:
            if isinstance(bc, dict) and bc.get("candidate_id") not in existing_ids:
                existing_blocked.append(bc)
                existing_ids.add(bc.get("candidate_id"))
    search_guidance = hypothesis_report.get("search_guidance")
    if isinstance(search_guidance, list):
        existing = set(
            json.dumps(item, sort_keys=True)
            for item in merged.get("search_guidance", [])
            if isinstance(item, dict)
        )
        for item in search_guidance:
            if isinstance(item, dict):
                key = json.dumps(item, sort_keys=True)
                if key not in existing:
                    merged.setdefault("search_guidance", []).append(item)
                    existing.add(key)
    if feedback_hints:
        merged["next_hints"] = feedback_hints + list(merged.get("next_hints", []))
    return merged


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default
