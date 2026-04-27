from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.external_strategy_candidate import (
    STATE_HINT,
    formalize_external_strategy_candidate,
)
from pde_agent.core.json_output import success_payload


def formalize_external_strategy_candidate_command(
    *,
    strategy_decision_path: str,
    output_path: str,
    strategy_id: str,
    title: str,
    source_title: str,
    source_url: str,
    scientific_claim: str,
    mechanism: str,
    applicable_pde: str,
    targeted_failure_modes: list[str],
    implementation_plan: dict[str, Any],
    expected_observables: list[str],
    known_risks: list[str],
) -> dict[str, object]:
    report = formalize_external_strategy_candidate(
        strategy_decision_path=Path(strategy_decision_path),
        output_path=Path(output_path),
        strategy_id=strategy_id,
        title=title,
        source_title=source_title,
        source_url=source_url,
        scientific_claim=scientific_claim,
        mechanism=mechanism,
        applicable_pde=applicable_pde,
        targeted_failure_modes=targeted_failure_modes,
        implementation_plan=implementation_plan,
        expected_observables=expected_observables,
        known_risks=known_risks,
    )
    return success_payload(
        run_id="local",
        command="formalize-external-strategy-candidate",
        state_hint=STATE_HINT,
        artifacts={
            "retrieved_strategy_candidate_path": report[
                "retrieved_strategy_candidate_path"
            ],
            "strategy_decision_path": report["strategy_decision_path"],
        },
        metrics={
            "strategy_id": report["record_strategy_proposal_fields"]["strategy_id"],
            "source_type": report["source_type"],
            "targeted_failure_mode_count": len(report["targeted_failure_modes"]),
            "expected_observable_count": len(
                report["record_strategy_proposal_fields"]["expected_observables"]
            ),
        },
        next_hint=str(report["next_command"]),
    )
