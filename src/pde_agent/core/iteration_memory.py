"""Build reusable Hermes iteration memory from batch optimization trajectories."""


import json
from pathlib import Path
from typing import Any


STATE_HINT = "ITERATION_MEMORY_BUILT"
MARKER = "Hermes Iteration Memory"


class IterationMemoryError(RuntimeError):
    pass


def build_iteration_memory(
    *,
    batch_loop_report_path: str | Path,
    output_path: str | Path,
    memory_json_path: str | Path,
    submission_dir: str | Path | None = None,
    task_ids: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    loop_file = Path(batch_loop_report_path)
    loop = _read_json_object(loop_file)
    if loop.get("state_hint") != "BATCH_OPTIMIZATION_LOOP_RAN":
        raise IterationMemoryError("batch loop report has invalid state_hint")
    markdown_path = Path(output_path)
    json_path = Path(memory_json_path)
    if (markdown_path.exists() or json_path.exists()) and not force:
        raise IterationMemoryError("iteration memory output already exists")
    memory = _build_memory(loop=loop, source_path=loop_file)
    markdown = _render_markdown(memory)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(memory, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    updated_logs: list[str] = []
    if submission_dir is not None:
        for task_id in task_ids or [1, 2]:
            log_path = Path(submission_dir) / f"task{task_id}_logs.log"
            if log_path.exists():
                _append_memory(log_path=log_path, markdown=markdown)
                updated_logs.append(str(log_path))
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "artifacts": {
            "memory_markdown_path": str(markdown_path),
            "memory_json_path": str(json_path),
            "updated_log_paths": updated_logs,
        },
        "metrics": {
            "trajectory_count": len(memory["trajectory_summary"]),
            "avoid_count": len(memory["avoid_repeating"]),
            "search_guidance_count": len(memory["search_guidance"]),
            "updated_log_count": len(updated_logs),
        },
        "next_command": "run-batch-optimization-loop",
    }


def _build_memory(*, loop: dict[str, Any], source_path: Path) -> dict[str, Any]:
    trajectory = _trajectory(loop)
    best = _best(loop)
    avoid = _avoid_repeating(trajectory=trajectory, best=best)
    family_outcomes = _family_outcomes(trajectory=trajectory, best=best)
    avoid_candidate_families = _avoid_candidate_families(family_outcomes)
    positive_signals = _positive_signals(trajectory=trajectory)
    search_guidance = _search_guidance(trajectory=trajectory)
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "source_batch_loop_report_path": str(source_path),
        "stop_reason": loop.get("stop_reason"),
        "decision": loop.get("decision"),
        "best_candidate": best,
        "trajectory_summary": [_trajectory_item(item) for item in trajectory],
        "avoid_repeating": avoid,
        "avoid_candidate_families": avoid_candidate_families,
        "family_outcomes": family_outcomes,
        "positive_signals": positive_signals,
        "search_guidance": search_guidance,
        "next_hints": _next_hints(
            best=best,
            avoid=avoid,
            avoid_candidate_families=avoid_candidate_families,
            positive_signals=positive_signals,
            search_guidance=search_guidance,
        ),
    }


def _trajectory(loop: dict[str, Any]) -> list[dict[str, Any]]:
    value = loop.get("trajectory")
    if not isinstance(value, list):
        raise IterationMemoryError("batch loop report missing trajectory")
    result = [item for item in value if isinstance(item, dict)]
    if not result:
        raise IterationMemoryError("batch loop trajectory is empty")
    return result


def _best(loop: dict[str, Any]) -> dict[str, Any]:
    value = loop.get("best_candidate")
    if not isinstance(value, dict):
        raise IterationMemoryError("batch loop report missing best_candidate")
    return {
        "candidate_id": value.get("candidate_id"),
        "score": value.get("score"),
        "objective": _objective(value),
        "strategy": _metadata(value).get("strategy"),
        "blend_weight": _metadata(value).get("blend_weight"),
        "source": _metadata(value).get("source"),
        "submission_dir": value.get("submission_dir"),
    }


def _trajectory_item(item: dict[str, Any]) -> dict[str, Any]:
    meta = _metadata(item)
    return {
        "round_index": item.get("round_index"),
        "candidate_id": item.get("candidate_id"),
        "score": item.get("score"),
        "objective": _objective(item),
        "rules_passed": item.get("rules_passed"),
        "effective_improvement": item.get("improved"),
        "score_delta_vs_best": item.get("score_delta_vs_best"),
        "strategy": meta.get("strategy"),
        "blend_weight": meta.get("blend_weight"),
        "source": meta.get("source"),
        "candidate_family": _candidate_family(item),
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
        meta = _metadata(item)
        if not meta.get("strategy"):
            continue
        reason = None
        if item.get("rules_passed") is False:
            reason = "rule_gate_failed"
        elif item.get("improved") is False:
            reason = "below_min_effective_improvement"
        if reason is None:
            continue
        avoid.append(
            {
                "strategy": meta.get("strategy"),
                "blend_weight": meta.get("blend_weight"),
                "reason": reason,
            }
        )
    return avoid


def _family_outcomes(
    *,
    trajectory: list[dict[str, Any]],
    best: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    initial_score = _objective_score(trajectory[0])
    best_id = best.get("candidate_id")
    rows: dict[str, dict[str, Any]] = {}
    for item in trajectory[1:]:
        family = _candidate_family(item)
        if not family:
            continue
        score = _objective_score(item)
        delta = None
        if score is not None and initial_score is not None:
            delta = score - initial_score
        row = rows.setdefault(
            family,
            {
                "candidate_family": family,
                "status": "neutral",
                "attempt_count": 0,
                "best_candidate_id": None,
                "best_score": None,
                "best_delta_vs_initial": None,
                "worst_candidate_id": None,
                "worst_score": None,
                "worst_delta_vs_initial": None,
                "failure_reasons": [],
            },
        )
        row["attempt_count"] += 1
        if score is not None and (
            row["best_score"] is None or score > float(row["best_score"])
        ):
            meta = _metadata(item)
            row["best_candidate_id"] = item.get("candidate_id")
            row["best_score"] = score
            row["best_delta_vs_initial"] = delta
            row["best_strategy"] = meta.get("strategy")
            row["best_blend_weight"] = meta.get("blend_weight")
        if score is not None and (
            row["worst_score"] is None or score < float(row["worst_score"])
        ):
            row["worst_candidate_id"] = item.get("candidate_id")
            row["worst_score"] = score
            row["worst_delta_vs_initial"] = delta
        if item.get("rules_passed") is False:
            _append_unique(row["failure_reasons"], "rule_gate_failed")
        elif delta is not None and delta <= -5.0:
            _append_unique(row["failure_reasons"], "severe_score_regression")
        elif item.get("improved") is False:
            _append_unique(row["failure_reasons"], "below_min_effective_improvement")
        if item.get("candidate_id") == best_id and delta is not None and delta > 0.0:
            row["status"] = "positive_signal"
        elif row["status"] != "positive_signal":
            best_delta = row.get("best_delta_vs_initial")
            if isinstance(best_delta, int | float) and best_delta <= -5.0:
                row["status"] = "blocked"
            elif isinstance(best_delta, int | float) and best_delta >= -0.05:
                row["status"] = "near_miss"
            elif row["failure_reasons"]:
                row["status"] = "weak_or_failed"
    for row in rows.values():
        if (
            row["status"] == "weak_or_failed"
            and "severe_score_regression" in row["failure_reasons"]
        ):
            row["status"] = "blocked"
    return rows


def _avoid_candidate_families(
    family_outcomes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    avoid = []
    for family, row in sorted(family_outcomes.items()):
        if row.get("status") != "blocked":
            continue
        reason = "blocked_by_prior_evidence"
        reasons = row.get("failure_reasons")
        if isinstance(reasons, list) and "severe_score_regression" in reasons:
            reason = "severe_score_regression"
        elif isinstance(reasons, list) and reasons:
            reason = str(reasons[0])
        avoid.append({"candidate_family": family, "reason": reason})
    return avoid


def _positive_signals(*, trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initial_score = _objective_score(trajectory[0])
    signals = []
    for item in trajectory[1:]:
        score = _objective_score(item)
        if score is None or initial_score is None or score <= initial_score:
            continue
        meta = _metadata(item)
        signals.append(
            {
                "candidate_id": item.get("candidate_id"),
                "candidate_family": _candidate_family(item),
                "strategy": meta.get("strategy"),
                "blend_weight": meta.get("blend_weight"),
                "objective_delta_vs_initial": round(score - initial_score, 10),
                "score": score,
            }
        )
    signals.sort(
        key=lambda item: float(item.get("objective_delta_vs_initial") or 0.0),
        reverse=True,
    )
    return signals


def _search_guidance(*, trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trajectory:
        return []
    initial_score = _objective_score(trajectory[0])
    guidance: list[dict[str, Any]] = []
    for item in trajectory[1:]:
        meta = _metadata(item)
        if meta.get("source") != "sem_window_gate_blend_multi_search":
            continue
        memory_guidance_spec_count = _int_value(
            meta.get("memory_guidance_spec_count")
        )
        if (
            memory_guidance_spec_count is not None
            and memory_guidance_spec_count > 0
            and item.get("improved") is False
        ):
            guidance.append(
                {
                    "guidance_type": "escalate_after_guided_search_exhaustion",
                    "reason": "memory_guided_specs_failed_to_improve",
                    "candidate_id": item.get("candidate_id"),
                    "source": meta.get("source"),
                    "memory_guidance_spec_count": memory_guidance_spec_count,
                    "search_evaluated_count": meta.get("search_evaluated_count"),
                    "feature_name": meta.get("feature_name"),
                    "validation_selection_metric": meta.get(
                        "validation_selection_metric"
                    ),
                    "validation_selection_score": meta.get(
                        "validation_selection_score"
                    ),
                    "recommended_action": (
                        "switch_gate_feature_family_or_try_new_dynamics_source"
                    ),
                }
            )
            continue
        if _int_value(meta.get("selected_sample_count")) != 0:
            continue
        objective_score = _objective_score(item)
        if objective_score is None or initial_score is None:
            continue
        delta = round(objective_score - initial_score, 6)
        if delta < -0.25:
            continue
        guidance.append(
            {
                "guidance_type": "refine_gating_after_near_miss",
                "reason": "near_miss_with_empty_gate_selection",
                "candidate_id": item.get("candidate_id"),
                "source": meta.get("source"),
                "objective_delta_vs_initial": delta,
                "feature_name": meta.get("feature_name"),
                "threshold": meta.get("threshold"),
                "selected_sample_count": meta.get("selected_sample_count"),
                "search_evaluated_count": meta.get("search_evaluated_count"),
                "validation_selection_metric": meta.get(
                    "validation_selection_metric"
                ),
                "validation_selection_score": meta.get("validation_selection_score"),
                "window_start": meta.get("window_start"),
                "window_stop": meta.get("window_stop"),
                "recommended_action": (
                    "relax_or_recenter_gate_threshold_and_window_before_repeating_family"
                ),
            }
        )
    return guidance


def _next_hints(
    *,
    best: dict[str, Any],
    avoid: list[dict[str, Any]],
    avoid_candidate_families: list[dict[str, Any]],
    positive_signals: list[dict[str, Any]],
    search_guidance: list[dict[str, Any]],
) -> list[str]:
    hints = []
    strategy = best.get("strategy")
    weight = best.get("blend_weight")
    if strategy and isinstance(weight, int | float):
        hints.append(
            f"Try local refinement around blend_weight={weight} for strategy={strategy}."
        )
    guidance_types = {item.get("guidance_type") for item in search_guidance}
    if "escalate_after_guided_search_exhaustion" in guidance_types:
        hints.append(
            "Switch gate feature family or dynamics source before repeating the "
            "guided sem-window search."
        )
    elif search_guidance:
        hints.append(
            "Use search_guidance before repeating a near-miss generator; refine the "
            "gate/window instead of treating the whole candidate family as exhausted."
        )
    if avoid:
        hints.append("Exclude avoid_repeating entries unless new evidence changes.")
    if avoid_candidate_families:
        hints.append("Block failed candidate families unless a smoke test changes evidence.")
    if positive_signals:
        hints.append("Promote positive_signals into the exploration scheduler before adding new families.")
    hints.append("Carry this memory into the next batch to avoid repeated low-value trials.")
    return hints


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata")
    return value if isinstance(value, dict) else {}


def _candidate_family(item: dict[str, Any]) -> str | None:
    meta = _metadata(item)
    source = meta.get("source")
    if isinstance(source, str) and source and source != "initial_submission":
        return source
    candidate_id = item.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        return None
    for prefix in (
        "linear_trend_spectral",
        "physics_residual_nudge",
        "checkpoint_window_gate",
        "sem_window_gate_blend",
        "postprocess",
        "hypothesis_lab",
    ):
        if candidate_id.startswith(prefix):
            return prefix
    return candidate_id.split("_w", maxsplit=1)[0]


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _objective(item: dict[str, Any]) -> dict[str, Any] | None:
    value = item.get("objective")
    return value if isinstance(value, dict) else None


def _objective_score(item: dict[str, Any]) -> float | None:
    objective = _objective(item)
    if isinstance(objective, dict):
        value = objective.get("objective_score")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    value = item.get("score")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _render_markdown(memory: dict[str, Any]) -> str:
    best = memory["best_candidate"]
    best_objective = best.get("objective") or {}
    lines = [
        MARKER,
        "",
        "Best Candidate:",
        f"- candidate_id: {best.get('candidate_id')}",
        f"- score: {best.get('score')}",
        f"- selected_objective: {best_objective.get('selected_objective')}",
        f"- objective_score: {best_objective.get('objective_score')}",
        f"- strategy: {best.get('strategy')}",
        f"- blend_weight: {best.get('blend_weight')}",
        "",
        "Trajectory Summary:",
    ]
    for item in memory["trajectory_summary"]:
        objective = item.get("objective") or {}
        lines.append(
            "- round="
            + str(item.get("round_index"))
            + " candidate="
            + str(item.get("candidate_id"))
            + " score="
            + str(item.get("score"))
            + " selected_objective="
            + str(objective.get("selected_objective"))
            + " objective_score="
            + str(objective.get("objective_score"))
            + " inference_time="
            + str(objective.get("inference_time"))
            + " strategy="
            + str(item.get("strategy"))
            + " blend_weight="
            + str(item.get("blend_weight"))
            + " effective_improvement="
            + str(item.get("effective_improvement"))
        )
    lines.extend(["", "Do not repeat:"])
    avoid = memory["avoid_repeating"]
    if avoid:
        for item in avoid:
            lines.append(
                "- strategy="
                + str(item.get("strategy"))
                + " blend_weight="
                + str(item.get("blend_weight"))
                + " reason="
                + str(item.get("reason"))
            )
    else:
        lines.append("- No avoid entries were identified.")
    lines.extend(["", "Family Outcomes:"])
    family_outcomes = memory.get("family_outcomes")
    if isinstance(family_outcomes, dict) and family_outcomes:
        for family, item in sorted(family_outcomes.items()):
            if not isinstance(item, dict):
                continue
            lines.append(
                "- family="
                + str(family)
                + " status="
                + str(item.get("status"))
                + " best_score="
                + str(item.get("best_score"))
                + " best_delta="
                + str(item.get("best_delta_vs_initial"))
                + " failures="
                + str(item.get("failure_reasons"))
            )
    else:
        lines.append("- No family outcomes were identified.")
    lines.extend(["", "Positive Signals:"])
    positive_signals = memory.get("positive_signals")
    if isinstance(positive_signals, list) and positive_signals:
        for item in positive_signals:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- candidate="
                + str(item.get("candidate_id"))
                + " family="
                + str(item.get("candidate_family"))
                + " delta="
                + str(item.get("objective_delta_vs_initial"))
                + " strategy="
                + str(item.get("strategy"))
                + " blend_weight="
                + str(item.get("blend_weight"))
            )
    else:
        lines.append("- No positive signals were identified.")
    lines.extend(["", "Search Guidance:"])
    guidance = memory["search_guidance"]
    if guidance:
        for item in guidance:
            lines.append(
                "- type="
                + str(item.get("guidance_type"))
                + " reason="
                + str(item.get("reason"))
                + " candidate="
                + str(item.get("candidate_id"))
                + " delta="
                + str(item.get("objective_delta_vs_initial"))
                + " selected_sample_count="
                + str(item.get("selected_sample_count"))
                + " action="
                + str(item.get("recommended_action"))
            )
    else:
        lines.append("- No search guidance was identified.")
    lines.extend(["", "Next Hints:"])
    for hint in memory["next_hints"]:
        lines.append("- " + hint)
    lines.append("")
    return "\n".join(lines)


def _append_memory(*, log_path: Path, markdown: str) -> None:
    original = log_path.read_text(encoding="utf-8")
    marker = "\n" + MARKER + "\n"
    if marker in original:
        original = original.split(marker, maxsplit=1)[0].rstrip() + "\n"
    log_path.write_text(original.rstrip() + "\n\n" + markdown, encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IterationMemoryError(f"missing batch loop report: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IterationMemoryError(f"invalid batch loop JSON: {path}") from exc
    if not isinstance(value, dict):
        raise IterationMemoryError("batch loop report JSON must be an object")
    return value
