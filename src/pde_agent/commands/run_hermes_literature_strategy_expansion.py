from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_literature_expansion import (
    FAILURE_MEMORY_STATE_HINT,
    STATE_HINT,
    record_hermes_literature_candidate_outcome,
    run_hermes_literature_strategy_expansion,
)
from pde_agent.core.json_output import success_payload


def run_hermes_literature_strategy_expansion_command(
    *,
    algorithm_candidates_path: str,
    workspace_dir: str,
    checkpoint_paths: list[str] | None,
    failure_memory_path: str | None,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        workspace_dir=Path(workspace_dir),
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])],
        failure_memory_path=Path(failure_memory_path)
        if failure_memory_path is not None
        else None,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-literature-strategy-expansion",
        state_hint=STATE_HINT,
        artifacts={"expansion_report_path": report["expansion_report_path"]},
        metrics={
            "action_count": report["action_count"],
            "executable_count": report["executable_count"],
            "proposal_count": report["proposal_count"],
            "skipped_count": report["skipped_count"],
        },
        next_hint=report["next_command"],
    )


def record_hermes_literature_candidate_outcome_command(
    *,
    algorithm_candidate_id: str,
    candidate_key: str,
    outcome: str,
    reason_codes: list[str],
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = record_hermes_literature_candidate_outcome(
        algorithm_candidate_id=algorithm_candidate_id,
        candidate_key=candidate_key,
        outcome=outcome,
        reason_codes=reason_codes,
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="record-hermes-literature-candidate-outcome",
        state_hint=FAILURE_MEMORY_STATE_HINT,
        artifacts={"failure_memory_path": str(output_path)},
        metrics={
            "failed_candidate_count": len(
                report.get("failed_algorithm_candidate_ids", [])
            ),
            "outcome_count": len(report.get("outcomes", [])),
        },
        next_hint="run-hermes-literature-strategy-expansion",
    )
