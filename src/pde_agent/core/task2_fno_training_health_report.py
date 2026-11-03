"""Task 2 FNO training health report builder.

Extracted from task2_fno_training_closed_loop so the health report can be
built standalone without importing the full closed-loop runner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.json_output import success_payload

STATE_HINT = "TASK2_FNO_TRAINING_HEALTH_REPORT_BUILT"


class Task2FnoTrainingHealthReportError(RuntimeError):
    pass


def build_task2_fno_training_health_report(
    *,
    workspace_dir: str | Path,
    output_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build health report across all iterations."""
    workspace = Path(workspace_dir).expanduser().resolve()
    if not workspace.is_dir():
        raise Task2FnoTrainingHealthReportError(
            f"workspace does not exist: {workspace}"
        )

    memory_files = sorted(workspace.glob("task2_fno_memory_*.json"))
    plan_files = sorted(workspace.glob("task2_fno_plan_*.json"))
    convergence_files = sorted(workspace.glob("task2_fno_convergence_*.json"))

    memories = [_load_json(path) for path in memory_files]
    plans = [_load_json(path) for path in plan_files]
    convergences = [_load_json(path) for path in convergence_files]
    memories = [m for m in memories if isinstance(m, dict)]
    plans = [p for p in plans if isinstance(p, dict)]
    convergences = [c for c in convergences if isinstance(c, dict)]

    score_history = [
        _float_or_none(m.get("current_best", {}).get("score"))
        for m in memories
        if isinstance(m.get("current_best"), dict)
    ]
    score_history = [s for s in score_history if s is not None]
    best_scores = [
        _float_or_none(m.get("best_so_far", {}).get("score"))
        for m in memories
        if isinstance(m.get("best_so_far"), dict)
    ]
    best_scores = [s for s in best_scores if s is not None]

    convergence_counts = _convergence_counts(convergences)
    latest_convergence = convergences[-1] if convergences else {}
    issues = _health_issues(
        score_history=score_history,
        convergence_counts=convergence_counts,
        latest_convergence=latest_convergence,
    )

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "workspace_dir": str(workspace),
        "iteration_count": len(memories),
        "score": {
            "current_best_history": score_history,
            "global_best": max(best_scores) if best_scores else None,
            "latest_current_best": score_history[-1] if score_history else None,
            "improvement_from_first": (
                score_history[-1] - score_history[0]
                if len(score_history) >= 2
                else None
            ),
        },
        "exploration": {
            "convergence_decision_counts": convergence_counts,
        },
        "health": {
            "status": "healthy" if not issues else "needs_attention",
            "issues": issues,
        },
        "supervisor_questions": [
            "Is the agent improving task2_proxy score consistently?",
            "Are we exploring diverse hyperparameter combinations?",
            "Did convergence request hypothesis-lab escalation?",
        ],
        "next_hint": "review health.status and let the agent continue or escalate",
    }
    out_file = (
        Path(output_path)
        if output_path is not None
        else workspace / "task2_fno_training_health_report.json"
    )
    _write_json(out_file, report, force=force)
    return report


def _health_issues(
    *,
    score_history: list[float],
    convergence_counts: dict[str, int],
    latest_convergence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if convergence_counts.get("doom_loop", 0) or convergence_counts.get(
        "stagnant", 0
    ):
        latest = latest_convergence if isinstance(latest_convergence, dict) else {}
        issues.append({
            "code": "convergence_warning",
            "severity": "high",
            "evidence": {
                "convergence_decision_counts": convergence_counts,
                "latest_decision": latest.get("decision"),
                "latest_reason": latest.get("reason"),
            },
        })
    if convergence_counts.get("error", 0):
        issues.append({
            "code": "iteration_errors_detected",
            "severity": "high",
            "evidence": {"convergence_decision_counts": convergence_counts},
        })
    if convergence_counts and not score_history:
        issues.append({
            "code": "no_successful_score_observed",
            "severity": "high",
            "evidence": {"convergence_decision_counts": convergence_counts},
        })
    if len(score_history) >= 4 and score_history[-1] <= max(score_history[:-1]):
        issues.append({
            "code": "latest_iteration_not_improving",
            "severity": "low",
            "evidence": {
                "latest_current_best": score_history[-1],
                "previous_best_current_score": max(score_history[:-1]),
            },
        })
    return issues


def _convergence_counts(convergences: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in convergences:
        decision = c.get("decision")
        if isinstance(decision, str) and decision:
            counts[decision] = counts.get(decision, 0) + 1
    return counts


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise Task2FnoTrainingHealthReportError(
            f"refusing to overwrite: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
