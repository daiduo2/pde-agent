"""Literature algorithm lab validation commands."""

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

def validate_algorithm_cards_v2_command(
    *,
    algorithm_cards_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = validate_algorithm_cards_v2(
        algorithm_cards_path=Path(algorithm_cards_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="validate-algorithm-cards-v2",
        state_hint=ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT,
        artifacts={
            "algorithm_card_v2_audit_path": report[
                "algorithm_card_v2_audit_path"
            ],
            "algorithm_cards_path": report["algorithm_cards_path"],
        },
        metrics={
            "decision": report["decision"],
            "card_count": report["card_count"],
            "issue_count": report["issue_count"],
        },
        next_hint=report["next_command"],
    )

def validate_executable_blueprints_command(
    *,
    blueprints_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = validate_executable_blueprints(
        blueprints_path=Path(blueprints_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="validate-executable-blueprints",
        state_hint=EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT,
        artifacts={
            "executable_blueprint_audit_path": report[
                "executable_blueprint_audit_path"
            ],
            "blueprints_path": report["blueprints_path"],
        },
        metrics={
            "decision": report["decision"],
            "blueprint_count": report["blueprint_count"],
            "normalized_blueprint_count": report["normalized_blueprint_count"],
            "issue_count": report["issue_count"],
        },
        next_hint=report["next_command"],
    )

def run_literature_algorithm_v2_pipeline_command(
    *,
    algorithm_cards_path: str,
    workspace_dir: str,
    registry_path: str,
    current_best_score: float,
    algorithm_candidate_id: str | None,
    sandbox_evidence_paths: list[str],
    paper_parse_paths: list[str],
    force: bool,
) -> dict[str, object]:
    report = run_literature_algorithm_v2_pipeline(
        algorithm_cards_path=Path(algorithm_cards_path),
        workspace_dir=Path(workspace_dir),
        registry_path=Path(registry_path),
        current_best_score=current_best_score,
        algorithm_candidate_id=algorithm_candidate_id,
        sandbox_evidence_paths=[Path(path) for path in sandbox_evidence_paths],
        paper_parse_paths=[Path(path) for path in paper_parse_paths],
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-literature-algorithm-v2-pipeline",
        state_hint=LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT,
        artifacts=report["artifacts"],
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )
