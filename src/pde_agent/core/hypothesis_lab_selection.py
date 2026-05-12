"""Candidate selection helpers for Hypothesis Lab sandbox reports."""

from typing import Any

import numpy as np


def _context_preserved(*, prediction: np.ndarray, context: np.ndarray) -> bool:
    return bool(np.allclose(prediction[:, : context.shape[1], :], context, atol=1e-6))


def _blueprint_selection_objective_name(
    *,
    blueprint: dict[str, Any],
    family: str,
) -> str | None:
    sandbox = blueprint.get("sandbox")
    requested = None
    if isinstance(sandbox, dict):
        requested = sandbox.get("selection_objective")
        if requested is None:
            requested = sandbox.get("primary_metric")
    if str(requested) in {
        "middle_late_score",
        "task1_middle_late_score",
        "task1_middle_late_base_score",
    }:
        return "middle_late_score"
    if str(requested) in {
        "early_rel_mse",
        "task1_early_rel_mse_guarded_score",
    } and family in {
        "task1_early_refinement",
        "task1_early_transition_expert",
        "task1_front25_dynamics",
        "task1_short_window_corrector",
    }:
        return "early_rel_mse"
    return None


def _candidate_selection_key(
    row: dict[str, Any],
    *,
    family: str,
    selection_objective: str | None = None,
) -> tuple[float, ...]:
    if selection_objective == "middle_late_score":
        segments = row.get("segments") if isinstance(row.get("segments"), dict) else {}
        middle = segments.get("middle") if isinstance(segments.get("middle"), dict) else {}
        late = segments.get("late") if isinstance(segments.get("late"), dict) else {}
        middle_score = middle.get("score", 0.0)
        late_score = late.get("score", 0.0)
        middle_late_score = 0.25 * float(middle_score) + 0.5 * float(late_score)
        return (
            middle_late_score,
            float(row.get("segment_score", 0.0)),
            float(row.get("task1_total_score_proxy", 0.0)),
            -float(row.get("elapsed_seconds", 0.0)),
        )
    if family in {
        "task1_early_refinement",
        "task1_early_transition_expert",
        "task1_front25_dynamics",
        "task1_short_window_corrector",
    }:
        early = row.get("segments", {}).get("early", {})
        rel_mse = early.get("rel_mse")
        if not isinstance(rel_mse, int | float):
            rel_mse = float("inf")
        return (
            -float(rel_mse),
            float(row.get("segment_score", 0.0)),
            float(row.get("task1_total_score_proxy", 0.0)),
            -float(row.get("elapsed_seconds", 0.0)),
        )
    return (
        float(row.get("task1_total_score_proxy", 0.0)),
        float(row.get("segment_score", 0.0)),
        -float(row.get("elapsed_seconds", 0.0)),
    )


def _selection_objective(
    family: str,
    *,
    selection_objective: str | None = None,
) -> dict[str, Any]:
    if selection_objective == "middle_late_score":
        return {
            "primary": "middle_late_score",
            "direction": "maximize",
            "secondary": [
                "segment_score",
                "task1_total_score_proxy",
                "elapsed_seconds",
            ],
        }
    if selection_objective == "early_rel_mse" or family in {
        "task1_early_refinement",
        "task1_early_transition_expert",
        "task1_front25_dynamics",
        "task1_short_window_corrector",
    }:
        return {
            "primary": "early_rel_mse",
            "direction": "minimize",
            "secondary": [
                "segment_score",
                "task1_total_score_proxy",
                "elapsed_seconds",
            ],
        }
    return {
        "primary": "task1_total_score_proxy",
        "direction": "maximize",
        "secondary": ["segment_score", "elapsed_seconds"],
    }
