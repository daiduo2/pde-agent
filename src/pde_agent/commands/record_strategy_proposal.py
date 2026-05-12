
from pathlib import Path
from typing import Any

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import (
    PROPOSAL_STATE_HINT,
    record_strategy_proposal,
)


def record_strategy_proposal_command(
    *,
    strategy_id: str,
    title: str,
    source_type: str,
    problem_context: str,
    hypothesis: str,
    mechanism: str,
    implementation_plan: dict[str, Any],
    expected_observables: list[str],
    known_risks: list[str],
    output_path: str,
) -> dict[str, object]:
    report = record_strategy_proposal(
        strategy_id=strategy_id,
        title=title,
        source_type=source_type,
        problem_context=problem_context,
        hypothesis=hypothesis,
        mechanism=mechanism,
        implementation_plan=implementation_plan,
        expected_observables=expected_observables,
        known_risks=known_risks,
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="record-strategy-proposal",
        state_hint=PROPOSAL_STATE_HINT,
        artifacts={"strategy_proposal_path": report["strategy_proposal_path"]},
        metrics={
            "strategy_id": report["strategy_id"],
            "source_type": report["source_type"],
            "observable_count": len(expected_observables),
            "risk_count": len(known_risks),
        },
        next_hint=str(report["next_command"]),
    )
