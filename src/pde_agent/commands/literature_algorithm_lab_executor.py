"""Literature algorithm lab executor commands."""

from __future__ import annotations

from pathlib import Path
from pde_agent.core.json_output import success_payload
from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
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

def create_bounded_executor_design_spec_command(
    *,
    executor_design_request_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = create_bounded_executor_design_spec(
        executor_design_request_path=Path(executor_design_request_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="create-bounded-executor-design-spec",
        state_hint=BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT,
        artifacts={
            "bounded_executor_design_spec_path": report[
                "bounded_executor_design_spec_path"
            ]
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "requested_executor_id": report["requested_executor_id"],
        },
        next_hint=report["next_command"],
    )

def materialize_bounded_executor_template_command(
    *,
    design_spec_path: str,
    output_dir: str,
    force: bool,
) -> dict[str, object]:
    report = materialize_bounded_executor_template(
        design_spec_path=Path(design_spec_path),
        output_dir=Path(output_dir),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="materialize-bounded-executor-template",
        state_hint=BOUNDED_EXECUTOR_TEMPLATE_STATE_HINT,
        artifacts={
            "bounded_executor_template_path": report[
                "bounded_executor_template_path"
            ],
            "manifest_path": report["manifest_path"],
            "executor_path": report["executor_path"],
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "implementation_status": report["implementation_status"],
        },
        next_hint=report["next_command"],
    )

def validate_bounded_executor_implementation_command(
    *,
    design_spec_path: str,
    manifest_path: str,
    executor_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = validate_bounded_executor_implementation(
        design_spec_path=Path(design_spec_path),
        manifest_path=Path(manifest_path),
        executor_path=Path(executor_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="validate-bounded-executor-implementation",
        state_hint=BOUNDED_EXECUTOR_AUDIT_STATE_HINT,
        artifacts={"bounded_executor_audit_path": report["bounded_executor_audit_path"]},
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "decision": report["decision"],
            "issue_count": report["issue_count"],
        },
        next_hint=report["next_command"],
    )

def run_bounded_executor_sandbox_command(
    *,
    design_spec_path: str,
    manifest_path: str,
    executor_path: str,
    context_path: str,
    target_path: str,
    output_dir: str,
    submission_id: str,
    force: bool,
    trace_path: str | None = None,
) -> dict[str, object]:
    report = run_bounded_executor_sandbox(
        design_spec_path=Path(design_spec_path),
        manifest_path=Path(manifest_path),
        executor_path=Path(executor_path),
        context_path=Path(context_path),
        target_path=Path(target_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        force=force,
    )
    trace_event = None
    if trace_path is not None:
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="bounded_executor_sandbox",
            payload={
                "algorithm_candidate_id": report["algorithm_candidate_id"],
                "candidate_family": report.get("candidate_family"),
                "outcome": "passed" if report["rules_passed"] else "failed",
                "score_delta": report["metric_results"][
                    "score_delta_vs_persistence"
                ],
                "rules_passed": report["rules_passed"],
                "inference_seconds": report["metric_results"]["inference_seconds"],
            },
            run_id="local",
            source_artifact_path=report["artifacts"]["sandbox_report_path"],
            tags=["literature", "bounded_executor", "sandbox"],
        )
    return success_payload(
        run_id="local",
        command="run-bounded-executor-sandbox",
        state_hint=BOUNDED_EXECUTOR_SANDBOX_STATE_HINT,
        artifacts={
            **report["artifacts"],
            **({"trace_path": trace_path, "trace_event_id": trace_event["event_id"]}
               if trace_event is not None
               else {}),
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "rules_passed": report["rules_passed"],
            "score_delta_vs_persistence": report["metric_results"][
                "score_delta_vs_persistence"
            ],
            "inference_seconds": report["metric_results"]["inference_seconds"],
            "trace_event_recorded": trace_event is not None,
        },
        next_hint=report["next_command"],
    )

def record_bounded_executor_sandbox_event_command(
    *,
    sandbox_report_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = record_bounded_executor_sandbox_event(
        sandbox_report_path=Path(sandbox_report_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="record-bounded-executor-sandbox-event",
        state_hint=LITERATURE_REGISTRY_EVENT_STATE_HINT,
        artifacts={
            "literature_registry_event_path": report[
                "literature_registry_event_path"
            ],
            "event_source_path": report["event_source_path"],
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "event_type": report["event_type"],
            "retry_allowed": report["retry_allowed"],
        },
        next_hint=report["next_command"],
    )

def record_bounded_executor_sandbox_evidence_command(
    *,
    sandbox_report_path: str,
    output_path: str,
    current_best_score: float,
    force: bool,
) -> dict[str, object]:
    report = record_bounded_executor_sandbox_evidence(
        sandbox_report_path=Path(sandbox_report_path),
        output_path=Path(output_path),
        current_best_score=current_best_score,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="record-bounded-executor-sandbox-evidence",
        state_hint="SANDBOX_EVIDENCE_RECORDED",
        artifacts={
            "sandbox_evidence_path": report["evidence_path"],
            "source_report_path": report["source_report_path"],
        },
        metrics={
            "candidate_id": report["candidate_id"],
            "rules_passed": report["rules_passed"],
            "score": report["score"],
            "score_delta_vs_current_best": report[
                "score_delta_vs_current_best"
            ],
        },
        next_hint=report["next_command"],
    )

def prepare_bounded_executor_authoring_task_command(
    *,
    design_spec_path: str,
    manifest_path: str,
    executor_path: str,
    context_path: str,
    target_path: str,
    output_path: str,
    sandbox_output_dir: str,
    audit_output_path: str,
    event_output_path: str,
    submission_id: str,
    force: bool,
) -> dict[str, object]:
    report = prepare_bounded_executor_authoring_task(
        design_spec_path=Path(design_spec_path),
        manifest_path=Path(manifest_path),
        executor_path=Path(executor_path),
        context_path=Path(context_path),
        target_path=Path(target_path),
        output_path=Path(output_path),
        sandbox_output_dir=Path(sandbox_output_dir),
        audit_output_path=Path(audit_output_path),
        event_output_path=Path(event_output_path),
        submission_id=submission_id,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="prepare-bounded-executor-authoring-task",
        state_hint=BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT,
        artifacts={
            "bounded_executor_authoring_task_path": report[
                "bounded_executor_authoring_task_path"
            ],
            "executor_path": report["executor_path"],
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "validation_command_count": len(report["validation_commands"]),
        },
        next_hint=report["next_command"],
    )

def audit_bounded_executor_authoring_task_command(
    *,
    authoring_task_path: str,
    skill_root: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = audit_bounded_executor_authoring_task(
        authoring_task_path=Path(authoring_task_path),
        skill_root=Path(skill_root),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="audit-bounded-executor-authoring-task",
        state_hint=BOUNDED_EXECUTOR_AUTHORING_AUDIT_STATE_HINT,
        artifacts={
            "bounded_executor_authoring_audit_path": report[
                "authoring_audit_path"
            ],
            "authoring_task_path": report["authoring_task_path"],
            "skill_path": report["skill_path"],
        },
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "decision": report["decision"],
            "issue_count": report["issue_count"],
        },
        next_hint=report["next_command"],
    )

def complete_bounded_executor_authoring_task_command(
    *,
    authoring_task_path: str,
    algorithm_candidates_path: str,
    registry_path: str,
    current_best_score: float,
    skill_root: str,
    force: bool,
    trace_path: str | None = None,
) -> dict[str, object]:
    report = complete_bounded_executor_authoring_task(
        authoring_task_path=Path(authoring_task_path),
        algorithm_candidates_path=Path(algorithm_candidates_path),
        registry_path=Path(registry_path),
        current_best_score=current_best_score,
        skill_root=Path(skill_root),
        force=force,
    )
    trace_event = None
    if trace_path is not None:
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="bounded_executor_completion",
            payload=_completion_trace_payload(report),
            run_id="local",
            source_artifact_path=_completion_source_artifact_path(report),
            tags=["literature", "bounded_executor", "completion"],
        )
    return success_payload(
        run_id="local",
        command="complete-bounded-executor-authoring-task",
        state_hint=BOUNDED_EXECUTOR_AUTHORING_COMPLETION_STATE_HINT,
        artifacts={
            **report["artifacts"],
            **(
                {"trace_path": trace_path, "trace_event_id": trace_event["event_id"]}
                if trace_event is not None
                else {}
            ),
        },
        metrics={
            "algorithm_candidate_id": _completion_algorithm_candidate_id(report),
            "decision": report["decision"],
            "issue_count": report.get("audit", {}).get("issue_count")
            if isinstance(report.get("audit"), dict)
            else None,
            "rules_passed": report.get("sandbox", {}).get("rules_passed")
            if isinstance(report.get("sandbox"), dict)
            else None,
            "registry_event_count": report.get("registry", {}).get("event_count")
            if isinstance(report.get("registry"), dict)
            else None,
            "trace_event_recorded": trace_event is not None,
        },
        next_hint=report["next_command"],
    )

def _completion_algorithm_candidate_id(report: dict[str, object]) -> object:
    promotion = report.get("promotion")
    if isinstance(promotion, dict):
        return promotion.get("algorithm_candidate_id")
    return report.get("algorithm_candidate_id")

def _completion_trace_payload(report: dict[str, object]) -> dict[str, object]:
    promotion = report.get("promotion") if isinstance(report.get("promotion"), dict) else {}
    sandbox = report.get("sandbox") if isinstance(report.get("sandbox"), dict) else {}
    audit = report.get("audit") if isinstance(report.get("audit"), dict) else {}
    registry = report.get("registry") if isinstance(report.get("registry"), dict) else {}
    promotion_metrics = (
        promotion.get("metric_results")
        if isinstance(promotion.get("metric_results"), dict)
        else {}
    )
    sandbox_metrics = (
        sandbox.get("metric_results")
        if isinstance(sandbox.get("metric_results"), dict)
        else {}
    )
    score_delta = _completion_number(
        promotion_metrics.get("score_delta_vs_current_best")
    )
    if score_delta is None:
        score_delta = _completion_number(sandbox_metrics.get("score_delta_vs_persistence"))
    return {
        "algorithm_candidate_id": _completion_algorithm_candidate_id(report),
        "candidate_family": promotion.get("candidate_family")
        or report.get("candidate_family"),
        "outcome": promotion.get("decision") or report.get("decision"),
        "rules_passed": sandbox.get("rules_passed"),
        "score_delta": score_delta,
        "score_delta_vs_current_best": _completion_number(
            promotion_metrics.get("score_delta_vs_current_best")
        ),
        "score_delta_vs_persistence": _completion_number(
            sandbox_metrics.get("score_delta_vs_persistence")
        ),
        "segment_score": _completion_number(sandbox_metrics.get("segment_score")),
        "inference_seconds": _completion_number(sandbox_metrics.get("inference_seconds")),
        "audit_decision": audit.get("decision"),
        "issue_count": audit.get("issue_count"),
        "registry_event_count": registry.get("event_count"),
        "next_command": report.get("next_command"),
    }

def _completion_source_artifact_path(report: dict[str, object]) -> object:
    artifacts = report.get("artifacts")
    if isinstance(artifacts, dict):
        return artifacts.get("completion_path")
    return None

def _completion_number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None
