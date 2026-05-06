"""Build bounded Hermes exploration schedules from iteration memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HERMES_EXPLORATION_SCHEDULE_BUILT"
MEMORY_STATE_HINT = "HERMES_ITERATION_MEMORY_BUILT"


class HermesExplorationSchedulerError(RuntimeError):
    pass


def build_hermes_exploration_schedule(
    *,
    memory_json_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise HermesExplorationSchedulerError(
            f"exploration schedule already exists: {output_file}"
        )
    memory_file = Path(memory_json_path)
    memory = _read_memory(memory_file)
    blocked = _blocked_families(memory)
    decision, schedule = _schedule(memory=memory, blocked=blocked)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "schedule_path": str(output_file),
        "inputs": {"memory_json_path": str(memory_file)},
        "decision": decision,
        "schedule": schedule,
        "next_command": "run-hermes-batch-optimization-loop",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _schedule(
    *,
    memory: dict[str, Any],
    blocked: list[str],
) -> tuple[str, dict[str, Any]]:
    positive = _positive_signals(memory)
    if positive:
        signal = positive[0]
        strategy = _text(signal.get("strategy"))
        if strategy:
            return (
                "refine_positive_signal",
                {
                    "candidate_generator": "postprocess",
                    "candidate_generator_fallback": "auto",
                    "learning_objective": "score",
                    "postprocess_strategy": strategy,
                    "postprocess_search_mode": "adaptive",
                    "postprocess_adaptive_step": 0.001,
                    "seed_candidate_id": signal.get("candidate_id"),
                    "seed_blend_weight": signal.get("blend_weight"),
                    "blocked_candidate_families": blocked,
                    "promotion_gate": {
                        "min_score_delta": 0.0001,
                        "require_audit_passed": True,
                        "require_rules_passed": True,
                    },
                    "experiment_design": {
                        "search_budget": "local_refinement",
                        "primary_metric": "task1_total_score",
                        "compare_against_current_best": True,
                    },
                },
            )
        family = _text(signal.get("candidate_family"))
        if family:
            return (
                "refine_positive_signal",
                {
                    "candidate_generator": _candidate_generator_for_family(family),
                    "source_candidate_family": family,
                    "candidate_generator_fallback": "auto",
                    "learning_objective": "score",
                    "search_budget": "local_refinement",
                    "seed_candidate_id": signal.get("candidate_id"),
                    "blocked_candidate_families": blocked,
                    "promotion_gate": {
                        "min_score_delta": 0.001,
                        "require_audit_passed": True,
                        "require_rules_passed": True,
                    },
                    "experiment_design": {
                        "search_budget": "local_refinement",
                        "primary_metric": "task1_total_score",
                        "secondary_metrics": ["early_segment_score", "inference_time"],
                        "compare_against_current_best": True,
                    },
                },
            )
    near_miss = _best_near_miss(memory)
    if near_miss is not None:
        family = str(near_miss["candidate_family"])
        candidate_generator = _candidate_generator_for_family(family)
        return (
            "refine_near_miss_family",
            _near_miss_schedule(
                {
                    "candidate_generator": candidate_generator,
                    "source_candidate_family": family,
                    "candidate_generator_fallback": "auto",
                    "learning_objective": "score",
                    "search_budget": "micro",
                    "blocked_candidate_families": blocked,
                    "promotion_gate": {
                        "min_score_delta": 0.001,
                        "require_audit_passed": True,
                        "require_rules_passed": True,
                    },
                    "experiment_design": {
                        "search_budget": "micro",
                        "primary_metric": "task1_total_score",
                        "secondary_metrics": ["early_segment_score", "inference_time"],
                        "compare_against_current_best": True,
                    },
                },
                near_miss=near_miss,
            ),
        )
    return (
        "expand_candidate_space",
        {
            "candidate_generator": _fallback_generator(blocked),
            "candidate_generator_fallback": "auto",
            "learning_objective": "score",
            "search_budget": "smoke",
            "blocked_candidate_families": blocked,
            "promotion_gate": {
                "min_score_delta": 0.01,
                "require_audit_passed": True,
                "require_rules_passed": True,
            },
            "experiment_design": {
                "search_budget": "smoke",
                "primary_metric": "task1_total_score",
                "compare_against_current_best": True,
            },
        },
    )


def _read_memory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HermesExplorationSchedulerError(f"memory JSON missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesExplorationSchedulerError(f"invalid memory JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HermesExplorationSchedulerError("memory JSON must be an object")
    if payload.get("state_hint") != MEMORY_STATE_HINT:
        raise HermesExplorationSchedulerError("memory JSON has invalid state_hint")
    return payload


def _blocked_families(memory: dict[str, Any]) -> list[str]:
    blocked = set()
    for item in memory.get("avoid_candidate_families", []):
        if not isinstance(item, dict):
            continue
        family = item.get("candidate_family")
        if isinstance(family, str) and family:
            blocked.add(family)
    family_outcomes = memory.get("family_outcomes")
    if isinstance(family_outcomes, dict):
        for family, outcome in family_outcomes.items():
            if isinstance(family, str) and isinstance(outcome, dict):
                if outcome.get("status") == "blocked":
                    blocked.add(family)
    return sorted(blocked)


def _positive_signals(memory: dict[str, Any]) -> list[dict[str, Any]]:
    signals = [
        dict(item)
        for item in memory.get("positive_signals", [])
        if isinstance(item, dict)
    ]
    signals.sort(
        key=lambda item: _float(item.get("objective_delta_vs_initial")) or 0.0,
        reverse=True,
    )
    return signals


def _best_near_miss(memory: dict[str, Any]) -> dict[str, Any] | None:
    outcomes = memory.get("family_outcomes")
    if not isinstance(outcomes, dict):
        return None
    candidates = []
    for family, outcome in outcomes.items():
        if not isinstance(family, str) or not isinstance(outcome, dict):
            continue
        if outcome.get("status") != "near_miss":
            continue
        row = dict(outcome)
        row["candidate_family"] = family
        candidates.append(row)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: _float(item.get("best_delta_vs_initial")) or -999999.0,
        reverse=True,
    )
    return candidates[0]


def _fallback_generator(blocked: list[str]) -> str:
    for candidate in (
        "physics_residual_nudge",
        "hypothesis_lab_front25_dynamics_smoke",
        "hypothesis_lab_early_subsegment_patch_search",
    ):
        if candidate not in blocked:
            return candidate
    return "postprocess"


def _candidate_generator_for_family(family: str) -> str:
    if family == "postprocess_generator":
        return "postprocess"
    if family == "sem_window_gate_blend_multi_search":
        return "sem_window_gate_blend_multi_early_smoke"
    return family


def _near_miss_schedule(
    base: dict[str, Any],
    *,
    near_miss: dict[str, Any],
) -> dict[str, Any]:
    schedule = dict(base)
    if schedule.get("candidate_generator") == "postprocess":
        strategy = _text(near_miss.get("best_strategy"))
        if strategy:
            schedule["postprocess_strategy"] = strategy
            schedule["postprocess_search_mode"] = "adaptive"
            schedule["postprocess_adaptive_step"] = 0.001
        if "best_candidate_id" in near_miss:
            schedule["seed_candidate_id"] = near_miss.get("best_candidate_id")
        if "best_blend_weight" in near_miss:
            schedule["seed_blend_weight"] = near_miss.get("best_blend_weight")
    return schedule


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
