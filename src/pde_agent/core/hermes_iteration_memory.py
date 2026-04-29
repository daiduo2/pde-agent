"""Build reusable Hermes iteration memory from batch optimization trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HERMES_ITERATION_MEMORY_BUILT"
MARKER = "Hermes Iteration Memory"


class HermesIterationMemoryError(RuntimeError):
    pass


def build_hermes_iteration_memory(
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
    if loop.get("state_hint") != "HERMES_BATCH_OPTIMIZATION_LOOP_RAN":
        raise HermesIterationMemoryError("batch loop report has invalid state_hint")
    markdown_path = Path(output_path)
    json_path = Path(memory_json_path)
    if (markdown_path.exists() or json_path.exists()) and not force:
        raise HermesIterationMemoryError("iteration memory output already exists")
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
            "updated_log_count": len(updated_logs),
        },
        "next_command": "run-hermes-batch-optimization-loop",
    }


def _build_memory(*, loop: dict[str, Any], source_path: Path) -> dict[str, Any]:
    trajectory = _trajectory(loop)
    best = _best(loop)
    avoid = _avoid_repeating(trajectory=trajectory, best=best)
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "source_batch_loop_report_path": str(source_path),
        "stop_reason": loop.get("stop_reason"),
        "decision": loop.get("decision"),
        "best_candidate": best,
        "trajectory_summary": [_trajectory_item(item) for item in trajectory],
        "avoid_repeating": avoid,
        "next_hints": _next_hints(best=best, avoid=avoid),
    }


def _trajectory(loop: dict[str, Any]) -> list[dict[str, Any]]:
    value = loop.get("trajectory")
    if not isinstance(value, list):
        raise HermesIterationMemoryError("batch loop report missing trajectory")
    result = [item for item in value if isinstance(item, dict)]
    if not result:
        raise HermesIterationMemoryError("batch loop trajectory is empty")
    return result


def _best(loop: dict[str, Any]) -> dict[str, Any]:
    value = loop.get("best_candidate")
    if not isinstance(value, dict):
        raise HermesIterationMemoryError("batch loop report missing best_candidate")
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


def _next_hints(*, best: dict[str, Any], avoid: list[dict[str, Any]]) -> list[str]:
    hints = []
    strategy = best.get("strategy")
    weight = best.get("blend_weight")
    if strategy and isinstance(weight, int | float):
        hints.append(
            f"Try local refinement around blend_weight={weight} for strategy={strategy}."
        )
    if avoid:
        hints.append("Exclude avoid_repeating entries unless new evidence changes.")
    hints.append("Carry this memory into the next batch to avoid repeated low-value trials.")
    return hints


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata")
    return value if isinstance(value, dict) else {}


def _objective(item: dict[str, Any]) -> dict[str, Any] | None:
    value = item.get("objective")
    return value if isinstance(value, dict) else None


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
        raise HermesIterationMemoryError(f"missing batch loop report: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesIterationMemoryError(f"invalid batch loop JSON: {path}") from exc
    if not isinstance(value, dict):
        raise HermesIterationMemoryError("batch loop report JSON must be an object")
    return value
