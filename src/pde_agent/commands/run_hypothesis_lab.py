from __future__ import annotations

from pathlib import Path

from pde_agent.core.hypothesis_lab import RUN_STATE_HINT, run_hypothesis_lab
from pde_agent.core.json_output import success_payload


def run_hypothesis_lab_command(
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
    failure_memory_output_path: str | None = None,
) -> dict[str, object]:
    report = run_hypothesis_lab(
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
        failure_memory_output_path=Path(failure_memory_output_path)
        if failure_memory_output_path is not None
        else None,
        context_path=Path(context_path),
        target_path=Path(target_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        force=force,
    )
    sandbox = report["sandbox"]
    return success_payload(
        run_id="local",
        command="run-hypothesis-lab",
        state_hint=RUN_STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "card_count": report["cards"]["count"],
            "hypothesis_count": report["hypotheses"]["count"],
            "blueprint_count": report["blueprints"]["count"],
            "best_candidate_id": sandbox["best_candidate"]["candidate_id"],
            "best_score": sandbox["verifier"]["best_segment_score"],
            "score_delta_vs_baseline": sandbox["verifier"][
                "score_delta_vs_baseline"
            ],
            "decision": sandbox["verifier"]["decision"],
        },
        next_hint=report["next_command"],
    )
