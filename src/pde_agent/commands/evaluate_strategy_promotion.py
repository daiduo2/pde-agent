
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import (
    PROMOTION_STATE_HINT,
    evaluate_strategy_promotion,
)


def evaluate_strategy_promotion_command(
    *,
    proposal_path: str,
    rubric_path: str,
    evidence_summary_path: str,
    output_path: str,
) -> dict[str, object]:
    report = evaluate_strategy_promotion(
        proposal_path=Path(proposal_path),
        rubric_path=Path(rubric_path),
        evidence_summary_path=Path(evidence_summary_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="evaluate-strategy-promotion",
        state_hint=PROMOTION_STATE_HINT,
        artifacts={
            "strategy_proposal_path": report["strategy_proposal_path"],
            "experiment_rubric_path": report["experiment_rubric_path"],
            "evidence_summary_path": report["evidence_summary_path"],
            "promotion_report_path": report["promotion_report_path"],
        },
        metrics={
            "strategy_id": report["strategy_id"],
            "decision": report["decision"],
            "promote_to_skill_candidate": report["promote_to_skill_candidate"],
            "reason_count": len(report["reason_codes"]),
        },
        next_hint=str(report["next_command"]),
    )
