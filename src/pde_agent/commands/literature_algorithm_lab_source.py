"""Literature algorithm lab source commands."""

from __future__ import annotations

import os
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

def normalize_openalex_sources_command(
    *,
    search_results_path: str,
    query: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = normalize_openalex_search_results(
        search_results_path=Path(search_results_path),
        query=query,
        output_path=Path(output_path),
        force=force,
    )
    return _source_search_payload("normalize-openalex-sources", report)

def normalize_arxiv_sources_command(
    *,
    search_results_path: str,
    query: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = normalize_arxiv_search_results(
        search_results_path=Path(search_results_path),
        query=query,
        output_path=Path(output_path),
        force=force,
    )
    return _source_search_payload("normalize-arxiv-sources", report)

def normalize_github_sources_command(
    *,
    search_results_path: str,
    query: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = normalize_github_search_results(
        search_results_path=Path(search_results_path),
        query=query,
        output_path=Path(output_path),
        force=force,
    )
    return _source_search_payload("normalize-github-sources", report)

def run_arxiv_mineru_parse_pipeline_command(
    *,
    query: str,
    output_dir: str,
    mode: str,
    model_version: str,
    max_results: int,
    selected_index: int,
    max_polls: int,
    poll_interval_seconds: float,
    force: bool,
) -> dict[str, object]:
    token = None if mode == "agent_url" else os.environ.get("MINERU_API_TOKEN")
    report = run_arxiv_mineru_parse_pipeline(
        query=query,
        output_dir=Path(output_dir),
        token=token,
        mode=mode,
        model_version=model_version,
        max_results=max_results,
        selected_index=selected_index,
        max_polls=max_polls,
        poll_interval_seconds=poll_interval_seconds,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-arxiv-mineru-parse-pipeline",
        state_hint=ARXIV_MINERU_PARSE_PIPELINE_STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "selected_source_id": report["selected_source_id"],
            "mineru_state": report["mineru_state"],
            "mineru_mode": report["mineru_mode"],
        },
        next_hint=str(report["next_command"]),
    )

def download_source_document_command(
    *,
    source_candidates_path: str,
    source_id: str,
    output_path: str,
    cache_dir: str,
    force: bool,
) -> dict[str, object]:
    report = download_source_document(
        source_candidates_path=Path(source_candidates_path),
        source_id=source_id,
        output_path=Path(output_path),
        cache_dir=Path(cache_dir),
        force=force,
    )
    source_record = report["source_record"]
    return success_payload(
        run_id="local",
        command="download-source-document",
        state_hint=SOURCE_DOCUMENT_STATE_HINT,
        artifacts={
            "downloaded_source_path": report["downloaded_source_path"],
            "local_raw_path": source_record["local_raw_path"],
        },
        metrics={
            "source_id": source_record["source_id"],
            "source_type": source_record["source_type"],
        },
        next_hint=str(report["next_command"]),
    )

def build_literature_source_records_command(
    *,
    source_fixture_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = build_literature_source_records(
        source_fixture_path=Path(source_fixture_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-literature-source-records",
        state_hint=SOURCE_RECORDS_STATE_HINT,
        artifacts={"source_records_path": report["source_records_path"]},
        metrics={"source_count": report["source_count"]},
        next_hint=report["next_command"],
    )

def _source_search_payload(command: str, report: dict[str, object]) -> dict[str, object]:
    return success_payload(
        run_id="local",
        command=command,
        state_hint=SOURCE_SEARCH_STATE_HINT,
        artifacts={"source_candidates_path": report["source_candidates_path"]},
        metrics={
            "provider": report["provider"],
            "source_candidate_count": report["source_candidate_count"],
        },
        next_hint=str(report["next_command"]),
    )
