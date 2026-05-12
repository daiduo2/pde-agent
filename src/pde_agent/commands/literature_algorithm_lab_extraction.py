"""Literature algorithm lab extraction commands."""

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

def prepare_literature_extraction_context_command(
    *,
    paper_parse_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = prepare_literature_extraction_context(
        paper_parse_path=Path(paper_parse_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="prepare-literature-extraction-context",
        state_hint=LITERATURE_EXTRACTION_CONTEXT_STATE_HINT,
        artifacts={"literature_extraction_context_path": str(Path(output_path))},
        metrics={
            "source_id": report["source_id"],
            "context_unit_count": report["context_unit_count"],
            "output_state_hint": report["agent_contract"]["output_state_hint"],
        },
        next_hint=str(report["next_command"]),
    )

def prepare_fno_literature_adaptation_task_command(
    *,
    literature_extraction_context_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = prepare_fno_literature_adaptation_task(
        literature_extraction_context_path=Path(literature_extraction_context_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="prepare-fno-literature-adaptation-task",
        state_hint=FNO_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        artifacts={
            "fno_literature_adaptation_task_path": report[
                "fno_literature_adaptation_task_path"
            ]
        },
        metrics={
            "source_id": report["source_id"],
            "selected_context_unit_count": report["selected_context_unit_count"],
            "operator_family": report["operator_family"],
        },
        next_hint=str(report["next_command"]),
    )

def prepare_deeponet_literature_adaptation_task_command(
    *,
    literature_extraction_context_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = prepare_deeponet_literature_adaptation_task(
        literature_extraction_context_path=Path(literature_extraction_context_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="prepare-deeponet-literature-adaptation-task",
        state_hint=DEEPONET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        artifacts={
            "deeponet_literature_adaptation_task_path": report[
                "deeponet_literature_adaptation_task_path"
            ]
        },
        metrics={
            "source_id": report["source_id"],
            "selected_context_unit_count": report["selected_context_unit_count"],
            "operator_family": report["operator_family"],
        },
        next_hint=str(report["next_command"]),
    )

def prepare_unet_literature_adaptation_task_command(
    *,
    literature_extraction_context_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = prepare_unet_literature_adaptation_task(
        literature_extraction_context_path=Path(literature_extraction_context_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="prepare-unet-literature-adaptation-task",
        state_hint=UNET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        artifacts={
            "unet_literature_adaptation_task_path": report[
                "unet_literature_adaptation_task_path"
            ]
        },
        metrics={
            "source_id": report["source_id"],
            "selected_context_unit_count": report["selected_context_unit_count"],
            "operator_family": report["operator_family"],
        },
        next_hint=str(report["next_command"]),
    )

def ingest_mineru_parse_command(
    *,
    source_records_path: str,
    source_id: str,
    markdown_path: str,
    asset_dir: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = ingest_mineru_parse(
        source_records_path=Path(source_records_path),
        source_id=source_id,
        markdown_path=Path(markdown_path),
        asset_dir=Path(asset_dir),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="ingest-mineru-parse",
        state_hint=PAPER_PARSE_STATE_HINT,
        artifacts={"paper_parse_path": report["paper_parse_path"]},
        metrics={
            "formula_count": report["formula_count"],
            "figure_count": report["figure_count"],
            "quality": report["parse_quality"]["quality"],
        },
        next_hint=report["next_command"],
    )

def extract_algorithm_cards_command(
    *,
    paper_parse_paths: list[str],
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = extract_algorithm_cards(
        paper_parse_paths=[Path(path) for path in paper_parse_paths],
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="extract-algorithm-cards",
        state_hint=ALGORITHM_CARDS_STATE_HINT,
        artifacts={"algorithm_cards_path": report["algorithm_cards_path"]},
        metrics={
            "algorithm_card_count": report["algorithm_card_count"],
            "extraction_status": report.get("extraction_status", "extracted"),
            "failure_code": report.get("failure_code"),
        },
        next_hint=report["next_command"],
    )

def record_agent_algorithm_cards_command(
    *,
    cards_json: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = record_agent_algorithm_cards(
        cards_json=cards_json,
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="record-agent-algorithm-cards",
        state_hint=ALGORITHM_CARDS_STATE_HINT,
        artifacts={"algorithm_cards_path": report["algorithm_cards_path"]},
        metrics={
            "algorithm_card_count": report["algorithm_card_count"],
            "extraction_status": report.get("extraction_status", "extracted"),
        },
        next_hint=report["next_command"],
    )
