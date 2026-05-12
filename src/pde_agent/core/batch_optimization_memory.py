from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.batch_optimization_common import BatchOptimizationLoopError


def record_skip(
    *,
    pair: tuple[str, float],
    skipped_candidates: list[dict[str, Any]],
) -> None:
    strategy, weight = pair
    if any(
        item.get("strategy") == strategy and item.get("blend_weight") == weight
        for item in skipped_candidates
    ):
        return
    skipped_candidates.append(
        {
            "strategy": strategy,
            "blend_weight": weight,
            "reason": "memory_avoid_repeating",
        }
    )


def best_postprocess_metadata(candidate: object) -> tuple[str, float] | None:
    if not isinstance(candidate, dict):
        return None
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("source") != "postprocess_generator":
        return None
    strategy = metadata.get("strategy")
    weight = metadata.get("blend_weight")
    if not isinstance(strategy, str) or not isinstance(weight, int | float):
        return None
    return strategy, normalize_weight(float(weight))


def adaptive_weight_neighbors(*, weight: float, step: float) -> list[float]:
    lower = normalize_weight(weight - step)
    upper = normalize_weight(weight + step)
    if lower == weight:
        return [upper] if upper != weight else []
    if upper == weight:
        return [lower] if lower != weight else []
    return [lower, upper]


def normalize_weight(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 6)


def load_avoid_set(path: str | Path | None) -> set[tuple[str, float]]:
    if path is None:
        return set()
    payload = load_memory_json(path)
    return avoid_set_from_payload(payload)


def avoid_set_from_payload(payload: dict[str, Any]) -> set[tuple[str, float]]:
    avoid = payload.get("avoid_repeating")
    if not isinstance(avoid, list):
        return set()
    result: set[tuple[str, float]] = set()
    for item in avoid:
        if not isinstance(item, dict):
            continue
        strategy = item.get("strategy")
        weight = item.get("blend_weight")
        if isinstance(strategy, str) and isinstance(weight, int | float):
            result.add((strategy, normalize_weight(float(weight))))
    return result


def load_search_guidance(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = load_memory_json(path)
    return search_guidance_from_payload(payload)


def search_guidance_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    guidance = payload.get("search_guidance")
    if not isinstance(guidance, list):
        return []
    return [dict(item) for item in guidance if isinstance(item, dict)]


def memory_constraints_summary(
    *,
    memory_json_path: str | Path | None,
    avoid_repeating: bool,
    memory_payload: dict[str, Any],
    avoid_count: int,
    search_guidance_count: int,
) -> dict[str, Any]:
    repair_actions = repair_policy_actions_from_payload(memory_payload)
    return {
        "memory_json_path": str(memory_json_path) if memory_json_path is not None else None,
        "avoid_repeating": avoid_repeating,
        "avoid_count": avoid_count,
        "search_guidance_count": search_guidance_count,
        "repair_policy_recommended_action_count": len(repair_actions),
        "repair_policy_recommended_actions": repair_actions,
        "input_sources": memory_input_sources(memory_payload),
    }


def repair_policy_actions_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("repair_policy_recommended_actions")
    if not isinstance(actions, list):
        return []
    return [dict(item) for item in actions if isinstance(item, dict)]


def memory_input_sources(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("input_sources")
    if not isinstance(value, dict):
        return {}
    return dict(value)


def load_memory_json(path: str | Path) -> dict[str, Any]:
    memory_file = Path(path)
    if not memory_file.exists():
        raise BatchOptimizationLoopError(f"memory_json_path not found: {path}")
    try:
        payload = json.loads(memory_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchOptimizationLoopError(f"invalid memory JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BatchOptimizationLoopError("memory JSON must be an object")
    return payload
