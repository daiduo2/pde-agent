"""Literature algorithm lab mineru commands."""

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

def submit_mineru_parse_command(
    *,
    downloaded_source_path: str,
    output_path: str,
    mode: str,
    model_version: str,
    force: bool,
) -> dict[str, object]:
    token = None if mode == "agent_url" else os.environ.get("MINERU_API_TOKEN")
    report = submit_mineru_parse_task(
        downloaded_source_path=Path(downloaded_source_path),
        output_path=Path(output_path),
        mode=mode,
        token=token,
        model_version=model_version,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="submit-mineru-parse",
        state_hint=MINERU_TASK_SUBMITTED_STATE_HINT,
        artifacts={"mineru_task_path": report["mineru_task_path"]},
        metrics={
            "task_id": report["task_id"],
            "mineru_mode": report["mineru_mode"],
            "model_version": report.get("model_version"),
        },
        next_hint=str(report["next_command"]),
    )

def poll_mineru_parse_command(
    *,
    mineru_task_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    task = Path(mineru_task_path).read_text(encoding="utf-8")
    token = os.environ.get("MINERU_API_TOKEN") if "precise_url" in task else None
    report = poll_mineru_parse_task(
        mineru_task_path=Path(mineru_task_path),
        output_path=Path(output_path),
        token=token,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="poll-mineru-parse",
        state_hint=MINERU_TASK_POLLED_STATE_HINT,
        artifacts={"mineru_poll_path": report["mineru_poll_path"]},
        metrics={
            "task_id": report["task_id"],
            "state": report["state"],
            "mineru_mode": report["mineru_mode"],
        },
        next_hint=str(report["next_command"]),
    )

def download_mineru_result_command(
    *,
    mineru_poll_path: str,
    output_path: str,
    output_dir: str,
    force: bool,
) -> dict[str, object]:
    report = download_mineru_parse_result(
        mineru_poll_path=Path(mineru_poll_path),
        output_path=Path(output_path),
        output_dir=Path(output_dir),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="download-mineru-result",
        state_hint=MINERU_RESULT_DOWNLOADED_STATE_HINT,
        artifacts={
            "mineru_result_path": report["mineru_result_path"],
            "local_markdown_path": report["local_markdown_path"],
            "local_zip_path": report["local_zip_path"],
        },
        metrics={"task_id": report["task_id"], "source_id": report["source_id"]},
        next_hint=str(report["next_command"]),
    )
