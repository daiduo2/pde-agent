from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.operator_strategy_evidence import (
    STATE_HINT,
    convert_operator_evidence_summary,
)


def convert_operator_evidence_summary_command(
    *,
    sweep_summary_path: str,
    output_path: str,
    model_family: str,
    source_state_hint: str,
    long_window_threshold: int,
    force: bool,
) -> dict[str, object]:
    evidence = convert_operator_evidence_summary(
        sweep_summary_path=Path(sweep_summary_path),
        output_path=Path(output_path),
        model_family=model_family,
        source_state_hint=source_state_hint,
        long_window_threshold=long_window_threshold,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="convert-operator-evidence-summary",
        state_hint=STATE_HINT,
        artifacts={
            "strategy_evidence_summary_path": evidence["evidence_summary_path"],
            "source_evidence_summary_path": evidence["source_evidence_summary_path"],
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
