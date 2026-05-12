
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_decision import (
    STATE_HINT,
    recommend_strategy_next_action,
)


def recommend_strategy_next_action_command(
    *,
    evidence_summary_path: str,
    output_path: str,
    min_evaluation_windows: int,
) -> dict[str, object]:
    report = recommend_strategy_next_action(
        evidence_summary_path=Path(evidence_summary_path),
        output_path=Path(output_path),
        min_evaluation_windows=min_evaluation_windows,
    )
    return success_payload(
        run_id="local",
        command="recommend-strategy-next-action",
        state_hint=STATE_HINT,
        artifacts={
            "strategy_decision_path": report["strategy_decision_path"],
            "evidence_summary_path": report["evidence_summary_path"],
        },
        metrics={
            "model_family": report["model_family"],
            "best_strategy_id": report["best_strategy_id"],
            "decision": report["decision"],
            "reason_count": len(report["reason_codes"]),
            "recommended_source_type": report["recommended_source_type"],
        },
        next_hint=str(report["next_command"]),
    )
