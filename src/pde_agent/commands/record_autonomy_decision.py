from __future__ import annotations

from pathlib import Path

from pde_agent.core.autonomy_decision import STATE_HINT, record_autonomy_decision
from pde_agent.core.json_output import success_payload


def record_autonomy_decision_command(
    *,
    decision: str,
    reason_code: str,
    evidence_source_path: str,
    output_path: str,
) -> dict[str, object]:
    report = record_autonomy_decision(
        decision=decision,
        reason_code=reason_code,
        evidence_source_path=Path(evidence_source_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="record-autonomy-decision",
        state_hint=STATE_HINT,
        artifacts={
            "autonomy_decision_path": report["autonomy_decision_path"],
            "evidence_source_path": report["evidence_source_path"],
        },
        metrics={
            "decision": report["decision"],
            "reason_code": report["reason_code"],
            "next_command": report["next_command"],
        },
        next_hint=str(report["next_command"]),
    )
