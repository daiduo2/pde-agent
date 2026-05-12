
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab import RUN_STATE_HINT, run_hypothesis_lab
from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
from pde_agent.core.json_output import success_payload


def run_hypothesis_lab_command(
    *,
    manual_report_paths: list[str],
    context_path: str,
    target_path: str,
    output_dir: str,
    submission_id: str,
    force: bool,
    adaptive_report_paths: list[str] | None = None,
    log_paths: list[str] | None = None,
    checkpoint_paths: list[str] | None = None,
    literature_paths: list[str] | None = None,
    algorithm_candidate_paths: list[str] | None = None,
    failure_memory_paths: list[str] | None = None,
    failure_memory_output_path: str | None = None,
    reference_path: str | None = None,
    reference_sample_limit: int | None = None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    trace_path: str | None = None,
) -> dict[str, object]:
    report = run_hypothesis_lab(
        manual_report_paths=[Path(path) for path in manual_report_paths],
        adaptive_report_paths=[
            Path(path) for path in (adaptive_report_paths or [])
        ],
        log_paths=[Path(path) for path in (log_paths or [])],
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])],
        literature_paths=[Path(path) for path in (literature_paths or [])],
        algorithm_candidate_paths=[
            Path(path) for path in (algorithm_candidate_paths or [])
        ],
        failure_memory_paths=[Path(path) for path in (failure_memory_paths or [])],
        failure_memory_output_path=Path(failure_memory_output_path)
        if failure_memory_output_path is not None
        else None,
        context_path=Path(context_path),
        target_path=Path(target_path),
        reference_path=Path(reference_path) if reference_path is not None else None,
        reference_sample_limit=reference_sample_limit,
        output_dir=Path(output_dir),
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    sandbox = report["sandbox"]
    trace_event = None
    if trace_path is not None:
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="hypothesis_lab_run",
            payload=_trace_payload_from_report(report),
            run_id="local",
            source_artifact_path=report["artifacts"]["hypothesis_lab_report_path"],
            tags=["hypothesis_lab", "sandbox"],
        )
        report["artifacts"]["trace_path"] = str(trace_path)
        report["artifacts"]["trace_event_id"] = trace_event["event_id"]
    return success_payload(
        run_id="local",
        command="run-hypothesis-lab",
        state_hint=RUN_STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "card_count": report["cards"]["count"],
            "hypothesis_count": report["hypotheses"]["count"],
            "blueprint_count": report["blueprints"]["count"],
            "best_candidate_id": sandbox["best_candidate"]["candidate_id"],
            "best_score": sandbox["verifier"]["best_segment_score"],
            "score_delta_vs_baseline": sandbox["verifier"][
                "score_delta_vs_baseline"
            ],
            "decision": sandbox["verifier"]["decision"],
            "trace_event_recorded": trace_event is not None,
        },
        next_hint=report["next_command"],
    )


def _trace_payload_from_report(report: dict[str, Any]) -> dict[str, Any]:
    sandbox = report.get("sandbox") if isinstance(report.get("sandbox"), dict) else {}
    best = (
        sandbox.get("best_candidate")
        if isinstance(sandbox.get("best_candidate"), dict)
        else {}
    )
    verifier = (
        sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
    )
    decision = str(verifier.get("decision", ""))
    return {
        "algorithm_candidate_id": best.get("candidate_id"),
        "candidate_family": sandbox.get("candidate_family") or best.get("family"),
        "outcome": _trace_outcome(decision),
        "decision": decision,
        "best_score": verifier.get("best_segment_score") or best.get("segment_score"),
        "score_delta": verifier.get("score_delta_vs_baseline"),
        "hypothesis_count": (report.get("hypotheses") or {}).get("count"),
        "blueprint_count": (report.get("blueprints") or {}).get("count"),
    }


def _trace_outcome(decision: str) -> str:
    if decision in {
        "promote_to_strategy_proposal",
        "promote",
        "accepted",
    }:
        return "promoted"
    if decision in {"reject", "rejected", "failed", "reject_or_revise"}:
        return "failed"
    if decision in {"blocked", "needs_executor"}:
        return "blocked"
    return "observed"
