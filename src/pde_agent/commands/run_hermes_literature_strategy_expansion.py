from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_literature_expansion import (
    FAILURE_MEMORY_STATE_HINT,
    LOCAL_ASSET_CANDIDATES_STATE_HINT,
    REGISTRY_SUMMARY_STATE_HINT,
    STATE_HINT,
    build_local_asset_algorithm_candidates,
    record_hermes_literature_candidate_outcome,
    run_hermes_literature_strategy_expansion,
    summarize_literature_registry_for_hermes,
)
from pde_agent.core.json_output import success_payload


def run_hermes_literature_strategy_expansion_command(
    *,
    algorithm_candidates_path: str,
    workspace_dir: str,
    checkpoint_paths: list[str] | None,
    failure_memory_path: str | None,
    registry_summary_path: str | None,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        workspace_dir=Path(workspace_dir),
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])],
        failure_memory_path=Path(failure_memory_path)
        if failure_memory_path is not None
        else None,
        registry_summary_path=Path(registry_summary_path)
        if registry_summary_path is not None
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


def build_local_asset_algorithm_candidates_command(
    *,
    local_strategy_library_path: str,
    checkpoint_paths: list[str] | None,
    output_path: str,
    max_strategy_candidates: int,
    force: bool,
) -> dict[str, object]:
    report = build_local_asset_algorithm_candidates(
        local_strategy_library_path=Path(local_strategy_library_path),
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])],
        output_path=Path(output_path),
        max_strategy_candidates=max_strategy_candidates,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-local-asset-algorithm-candidates",
        state_hint="ALGORITHM_CANDIDATES_COMPILED",
        artifacts={"algorithm_candidates_path": report["algorithm_candidates_path"]},
        metrics={
            "source_state_hint": LOCAL_ASSET_CANDIDATES_STATE_HINT,
            "algorithm_candidate_count": report["algorithm_candidate_count"],
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


def summarize_literature_registry_for_hermes_command(
    *,
    registry_path: str,
    output_path: str,
    current_best_score: float,
    force: bool,
) -> dict[str, object]:
    report = summarize_literature_registry_for_hermes(
        registry_path=Path(registry_path),
        output_path=Path(output_path),
        current_best_score=current_best_score,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="summarize-literature-registry-for-hermes",
        state_hint=REGISTRY_SUMMARY_STATE_HINT,
        artifacts={"hermes_registry_summary_path": report["summary_path"]},
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )
