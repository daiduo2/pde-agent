
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import LEDGER_STATE_HINT, append_strategy_ledger


def append_strategy_ledger_command(
    *,
    proposal_path: str,
    rubric_path: str,
    evidence_summary_path: str,
    promotion_report_path: str,
    ledger_path: str,
) -> dict[str, object]:
    row = append_strategy_ledger(
        proposal_path=Path(proposal_path),
        rubric_path=Path(rubric_path),
        evidence_summary_path=Path(evidence_summary_path),
        promotion_report_path=Path(promotion_report_path),
        ledger_path=Path(ledger_path),
    )
    return success_payload(
        run_id="local",
        command="append-strategy-ledger",
        state_hint=LEDGER_STATE_HINT,
        artifacts={"strategy_ledger_path": row["strategy_ledger_path"]},
        metrics={
            "strategy_id": row["strategy_id"],
            "decision": row["decision"],
            "promote_to_skill_candidate": row["promote_to_skill_candidate"],
        },
        next_hint=str(row["next_command"]),
    )
