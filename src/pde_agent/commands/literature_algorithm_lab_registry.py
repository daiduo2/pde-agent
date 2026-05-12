"""Literature algorithm lab registry commands."""

from __future__ import annotations

from pathlib import Path
from pde_agent.core.json_output import success_payload
from pde_agent.core.literature_algorithm_lab import (
    ALGORITHM_CANDIDATES_STATE_HINT,
    ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT,
    ALGORITHM_CARDS_STATE_HINT,
    ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT,
    ARXIV_MINERU_PARSE_PIPELINE_STATE_HINT,
    BOUNDED_EXECUTOR_AUDIT_STATE_HINT,
    BOUNDED_EXECUTOR_AUTHORING_AUDIT_STATE_HINT,
    BOUNDED_EXECUTOR_AUTHORING_COMPLETION_STATE_HINT,
    BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT,
    BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT,
    BOUNDED_EXECUTOR_SANDBOX_STATE_HINT,
    BOUNDED_EXECUTOR_TEMPLATE_STATE_HINT,
    BLOCKED_CANDIDATE_HANDOFF_STATE_HINT,
    DEEPONET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
    EXECUTOR_DESIGN_REQUEST_STATE_HINT,
    EXECUTOR_TAXONOMY_STATE_HINT,
    EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT,
    FNO_CANDIDATE_ADAPTED_STATE_HINT,
    FNO_LITERATURE_ADAPTATION_TASK_STATE_HINT,
    LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT,
    LITERATURE_EXTRACTION_CONTEXT_STATE_HINT,
    LITERATURE_REGISTRY_UPDATED_STATE_HINT,
    LITERATURE_REGISTRY_EVENT_STATE_HINT,
    MINERU_RESULT_DOWNLOADED_STATE_HINT,
    MINERU_TASK_POLLED_STATE_HINT,
    MINERU_TASK_SUBMITTED_STATE_HINT,
    PAPER_PARSE_STATE_HINT,
    SANDBOX_PLAN_STATE_HINT,
    SOURCE_DOCUMENT_STATE_HINT,
    SOURCE_SEARCH_STATE_HINT,
    SOURCE_RECORDS_STATE_HINT,
    UNET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
    adapt_fno_architecture_candidate,
    audit_bounded_executor_authoring_task,
    build_blocked_candidate_handoff,
    build_executor_taxonomy,
    build_literature_source_records,
    complete_bounded_executor_authoring_task,
    compile_algorithm_candidates,
    create_bounded_executor_design_spec,
    create_executor_design_request,
    download_mineru_parse_result,
    download_source_document,
    evaluate_algorithm_candidate_promotion,
    extract_algorithm_cards,
    ingest_mineru_parse,
    normalize_arxiv_search_results,
    normalize_github_search_results,
    normalize_openalex_search_results,
    plan_candidate_sandbox,
    poll_mineru_parse_task,
    prepare_bounded_executor_authoring_task,
    prepare_deeponet_literature_adaptation_task,
    prepare_fno_literature_adaptation_task,
    prepare_literature_extraction_context,
    prepare_unet_literature_adaptation_task,
    record_bounded_executor_sandbox_evidence,
    materialize_bounded_executor_template,
    record_literature_registry_event,
    record_agent_algorithm_cards,
    record_bounded_executor_sandbox_event,
    run_arxiv_mineru_parse_pipeline,
    run_bounded_executor_sandbox,
    run_literature_algorithm_v2_pipeline,
    submit_mineru_parse_task,
    update_literature_registry,
    validate_algorithm_cards_v2,
    validate_bounded_executor_implementation,
    validate_executable_blueprints,
)

def update_literature_registry_command(
    *,
    registry_path: str,
    downloaded_source_paths: list[str],
    paper_parse_paths: list[str],
    algorithm_cards_paths: list[str],
    algorithm_candidates_paths: list[str],
    event_paths: list[str],
    force: bool,
) -> dict[str, object]:
    report = update_literature_registry(
        registry_path=Path(registry_path),
        downloaded_source_paths=[Path(path) for path in downloaded_source_paths],
        paper_parse_paths=[Path(path) for path in paper_parse_paths],
        algorithm_cards_paths=[Path(path) for path in algorithm_cards_paths],
        algorithm_candidates_paths=[
            Path(path) for path in algorithm_candidates_paths
        ],
        event_paths=[Path(path) for path in event_paths],
        force=force,
    )
    return success_payload(
        run_id="local",
        command="update-literature-registry",
        state_hint=LITERATURE_REGISTRY_UPDATED_STATE_HINT,
        artifacts={"literature_registry_path": report["registry_path"]},
        metrics={
            "source_count": report["source_count"],
            "algorithm_node_count": report["algorithm_node_count"],
            "candidate_node_count": report["candidate_node_count"],
            "event_count": report["event_count"],
        },
        next_hint=report["next_command"],
    )

def record_literature_registry_event_command(
    *,
    event_source_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = record_literature_registry_event(
        event_source_path=Path(event_source_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="record-literature-registry-event",
        state_hint=LITERATURE_REGISTRY_EVENT_STATE_HINT,
        artifacts={
            "literature_registry_event_path": report[
                "literature_registry_event_path"
            ],
            "event_source_path": report["event_source_path"],
        },
        metrics={
            "event_type": report["event_type"],
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "failure_code_count": len(report["failure_codes"]),
            "retry_allowed": report["retry_allowed"],
        },
        next_hint=report["next_command"],
    )
