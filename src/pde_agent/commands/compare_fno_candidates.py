from __future__ import annotations

from pathlib import Path

from pde_agent.core.fno_candidate_comparison import compare_fno_candidates
from pde_agent.core.json_output import success_payload


def compare_fno_candidates_command(
    *,
    evaluation_report_paths: list[str],
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = compare_fno_candidates(
        evaluation_report_paths=[Path(path) for path in evaluation_report_paths],
        output_path=Path(output_path),
        force=force,
    )
    winner = report["winner"]
    return success_payload(
        run_id="local",
        command="compare-fno-candidates",
        state_hint="FNO_CANDIDATES_COMPARED",
        artifacts={
            "comparison_report_path": str(output_path),
        },
        metrics={
            "candidate_count": report["candidate_count"],
            "winner_experiment_id": winner["experiment_id"],
            "winner_intervention": winner["intervention"],
            "winner_score": winner["score"],
            "winner_score_delta": winner["score_delta"],
            "rejected_count": len(report["rejected_candidates"]),
        },
        next_hint="prepare-next-fno-iteration",
    )
