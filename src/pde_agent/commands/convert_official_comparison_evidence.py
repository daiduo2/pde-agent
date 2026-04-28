from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.official_comparison_evidence import (
    STATE_HINT,
    convert_official_comparison_to_evidence,
)


def convert_official_comparison_evidence_command(
    *,
    comparison_report_path: str,
    output_path: str,
    long_window_threshold: int,
    force: bool,
) -> dict[str, object]:
    evidence = convert_official_comparison_to_evidence(
        comparison_report_path=Path(comparison_report_path),
        output_path=Path(output_path),
        long_window_threshold=long_window_threshold,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="convert-official-comparison-evidence",
        state_hint=STATE_HINT,
        artifacts={
            "strategy_evidence_summary_path": evidence["evidence_summary_path"],
            "source_official_comparison_report_path": evidence[
                "source_official_comparison_report_path"
            ],
        },
        metrics={
            "model_family": evidence["model_family"],
            "evaluation_window_count": evidence["evaluation_window_count"],
            "mean_score_delta": evidence["mean_score_delta"],
            "long_window_score_delta": evidence["long_window_score_delta"],
            "failure_rate": evidence["failure_rate"],
            "candidate_record_count": len(evidence["candidate_records"]),
            "strategy_ranking_count": len(evidence["strategy_ranking"]),
            "best_strategy_id": evidence["best_strategy_id"],
        },
        next_hint=str(evidence["next_command"]),
    )
