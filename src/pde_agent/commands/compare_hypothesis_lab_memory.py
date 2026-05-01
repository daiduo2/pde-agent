from __future__ import annotations

from pathlib import Path

from pde_agent.core.hypothesis_lab_memory_ab import (
    STATE_HINT,
    compare_hypothesis_lab_memory,
)
from pde_agent.core.json_output import success_payload


def compare_hypothesis_lab_memory_command(
    *,
    manual_report_paths: list[str],
    context_path: str,
    target_path: str,
    output_dir: str,
    submission_id: str,
    force: bool,
    adaptive_report_paths: list[str] | None = None,
    log_paths: list[str] | None = None,
    checkpoint_paths: list[str] | None = None,
    literature_paths: list[str] | None = None,
    algorithm_candidate_paths: list[str] | None = None,
    failure_memory_paths: list[str] | None = None,
) -> dict[str, object]:
    report = compare_hypothesis_lab_memory(
        manual_report_paths=[Path(path) for path in manual_report_paths],
        adaptive_report_paths=[
            Path(path) for path in (adaptive_report_paths or [])
        ],
        log_paths=[Path(path) for path in (log_paths or [])],
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])],
        literature_paths=[Path(path) for path in (literature_paths or [])],
        algorithm_candidate_paths=[
            Path(path) for path in (algorithm_candidate_paths or [])
        ],
        failure_memory_paths=[Path(path) for path in (failure_memory_paths or [])],
        context_path=Path(context_path),
        target_path=Path(target_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="compare-hypothesis-lab-memory",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "decision": report["decision"],
            "score_delta_diff": report["score_delta_diff"],
            "best_segment_score_diff": report["best_segment_score_diff"],
            "candidate_budget_diff": report["candidate_budget_diff"],
            "best_candidate_changed": report["best_candidate_changed"],
        },
        next_hint=report["next_command"],
    )
