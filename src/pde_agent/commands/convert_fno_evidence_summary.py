from __future__ import annotations

from pathlib import Path

from pde_agent.core.fno_strategy_evidence import (
    STATE_HINT,
    convert_fno_evidence_summary,
)
from pde_agent.core.json_output import success_payload


def convert_fno_evidence_summary_command(
    *,
    sweep_summary_path: str,
    output_path: str,
    long_window_threshold: int,
    force: bool,
) -> dict[str, object]:
    evidence = convert_fno_evidence_summary(
        sweep_summary_path=Path(sweep_summary_path),
        output_path=Path(output_path),
        long_window_threshold=long_window_threshold,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="convert-fno-evidence-summary",
        state_hint=STATE_HINT,
        artifacts={
            "strategy_evidence_summary_path": evidence["evidence_summary_path"],
            "source_fno_evidence_summary_path": evidence[
                "source_fno_evidence_summary_path"
            ],
        },
        metrics={
            "evaluation_window_count": evidence["evaluation_window_count"],
            "mean_score_delta": evidence["mean_score_delta"],
            "long_window_score_delta": evidence["long_window_score_delta"],
            "failure_rate": evidence["failure_rate"],
        },
        next_hint=str(evidence["next_command"]),
    )
