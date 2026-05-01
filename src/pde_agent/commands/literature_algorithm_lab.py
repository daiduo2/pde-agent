from __future__ import annotations

import os
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.literature_algorithm_lab import (
    ALGORITHM_CANDIDATES_STATE_HINT,
    ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT,
    ALGORITHM_CARDS_STATE_HINT,
    ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT,
    BLOCKED_CANDIDATE_HANDOFF_STATE_HINT,
    EXECUTOR_DESIGN_REQUEST_STATE_HINT,
    EXECUTOR_TAXONOMY_STATE_HINT,
    FNO_CANDIDATE_ADAPTED_STATE_HINT,
    LITERATURE_REGISTRY_UPDATED_STATE_HINT,
    MINERU_RESULT_DOWNLOADED_STATE_HINT,
    MINERU_TASK_POLLED_STATE_HINT,
    MINERU_TASK_SUBMITTED_STATE_HINT,
    PAPER_PARSE_STATE_HINT,
    SANDBOX_PLAN_STATE_HINT,
    SOURCE_DOCUMENT_STATE_HINT,
    SOURCE_SEARCH_STATE_HINT,
    SOURCE_RECORDS_STATE_HINT,
    adapt_fno_architecture_candidate,
    build_blocked_candidate_handoff,
    build_executor_taxonomy,
    build_literature_source_records,
    compile_algorithm_candidates,
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
    submit_mineru_parse_task,
    update_literature_registry,
    validate_algorithm_cards_v2,
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
        metrics={"algorithm_card_count": report["algorithm_card_count"]},
        next_hint=report["next_command"],
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


def build_executor_taxonomy_command(
    *,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = build_executor_taxonomy(
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-executor-taxonomy",
        state_hint=EXECUTOR_TAXONOMY_STATE_HINT,
        artifacts={"executor_taxonomy_path": report["executor_taxonomy_path"]},
        metrics=report["route_counts"],
        next_hint=report["next_command"],
    )


def adapt_fno_architecture_candidate_command(
    *,
    algorithm_candidates_path: str,
    algorithm_candidate_id: str,
    checkpoint_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = adapt_fno_architecture_candidate(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        algorithm_candidate_id=algorithm_candidate_id,
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="adapt-fno-architecture-candidate",
        state_hint=FNO_CANDIDATE_ADAPTED_STATE_HINT,
        artifacts={"adapted_candidate_path": report["adapted_candidate_path"]},
        metrics={
            "source_algorithm_candidate_id": report[
                "source_algorithm_candidate_id"
            ],
            "adapted_candidate_id": report["adapted_candidate"][
                "algorithm_candidate_id"
            ],
        },
        next_hint=report["next_command"],
    )


def create_executor_design_request_command(
    *,
    algorithm_candidates_path: str,
    algorithm_candidate_id: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = create_executor_design_request(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        algorithm_candidate_id=algorithm_candidate_id,
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="create-executor-design-request",
        state_hint=EXECUTOR_DESIGN_REQUEST_STATE_HINT,
        artifacts={
            "executor_design_request_path": report["executor_design_request_path"]
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "requested_executor_id": report["requested_executor_id"],
        },
        next_hint=report["next_command"],
    )


def build_blocked_candidate_handoff_command(
    *,
    executor_design_request_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = build_blocked_candidate_handoff(
        executor_design_request_path=Path(executor_design_request_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-blocked-candidate-handoff",
        state_hint=BLOCKED_CANDIDATE_HANDOFF_STATE_HINT,
        artifacts={
            "blocked_candidate_handoff_path": report[
                "blocked_candidate_handoff_path"
            ]
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "requires_executor": report["blocked_evidence_summary"][
                "requires_executor"
            ],
        },
        next_hint=report["next_command"],
    )
