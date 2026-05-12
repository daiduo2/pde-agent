"""Literature algorithm lab candidate commands."""

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

def compile_algorithm_candidates_command(
    *,
    algorithm_cards_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = compile_algorithm_candidates(
        algorithm_cards_path=Path(algorithm_cards_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="compile-algorithm-candidates",
        state_hint=ALGORITHM_CANDIDATES_STATE_HINT,
        artifacts={"algorithm_candidates_path": report["algorithm_candidates_path"]},
        metrics={"algorithm_candidate_count": report["algorithm_candidate_count"]},
        next_hint=report["next_command"],
    )

def plan_candidate_sandbox_command(
    *,
    algorithm_candidates_path: str,
    algorithm_candidate_id: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = plan_candidate_sandbox(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        algorithm_candidate_id=algorithm_candidate_id,
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="plan-algorithm-candidate-sandbox",
        state_hint=SANDBOX_PLAN_STATE_HINT,
        artifacts={"sandbox_plan_path": report["sandbox_plan_path"]},
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "sandbox_level": report["sandbox_level"],
            "blocked": report["blocked"],
        },
        next_hint=report["next_command"],
    )

def evaluate_algorithm_candidate_promotion_command(
    *,
    algorithm_candidates_path: str,
    algorithm_candidate_id: str,
    sandbox_evidence_paths: list[str],
    output_path: str,
    current_best_score: float,
    force: bool,
) -> dict[str, object]:
    report = evaluate_algorithm_candidate_promotion(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        algorithm_candidate_id=algorithm_candidate_id,
        sandbox_evidence_paths=[Path(path) for path in sandbox_evidence_paths],
        output_path=Path(output_path),
        current_best_score=current_best_score,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="evaluate-algorithm-candidate-promotion",
        state_hint=ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT,
        artifacts={
            "algorithm_candidate_promotion_path": report[
                "algorithm_candidate_promotion_path"
            ],
            "algorithm_candidates_path": report["algorithm_candidates_path"],
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "decision": report["decision"],
            "evidence_count": report["evidence_count"],
            "reason_code_count": len(report["reason_codes"]),
        },
        next_hint=report["next_command"],
    )
