"""Command functions for trace domain."""

from __future__ import annotations

from pathlib import Path
from pde_agent.core.hypothesis_lab_trace import (
    HypothesisLabTraceStore,
    SUMMARY_STATE_HINT,
    summarize_hypothesis_lab_trace,
)
from pde_agent.core.json_output import success_payload

TRACE_EVENT_STATE_HINT = "HYPOTHESIS_LAB_TRACE_EVENT_RECORDED"

def record_hypothesis_lab_trace_event_command(
    *,
    trace_path: str,
    event_type: str,
    payload_json: str,
    run_id: str | None,
    source_artifact_path: str | None,
    tags: list[str] | None,
) -> dict[str, object]:
    payload = _parse_json_object(payload_json, "payload_json")
    event = HypothesisLabTraceStore(trace_path).append_event(
        event_type=event_type,
        payload=payload,
        run_id=run_id,
        source_artifact_path=source_artifact_path,
        tags=tags or [],
    )
    return success_payload(
        run_id=run_id or "local",
        command="record-hypothesis-lab-trace-event",
        state_hint=TRACE_EVENT_STATE_HINT,
        artifacts={
            "trace_path": str(trace_path),
            "event_id": event["event_id"],
        },
        metrics={
            "event_type": event["event_type"],
            "candidate_family": payload.get("candidate_family"),
            "outcome": payload.get("outcome"),
        },
        next_hint="summarize-hypothesis-lab-trace",
    )

def summarize_hypothesis_lab_trace_command(
    *,
    trace_path: str,
    output_path: str,
    force: bool = False,
) -> dict[str, object]:
    summary = summarize_hypothesis_lab_trace(
        trace_path=Path(trace_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="summarize-hypothesis-lab-trace",
        state_hint=SUMMARY_STATE_HINT,
        artifacts={
            "trace_path": trace_path,
            "trace_summary_json": output_path,
            "summary_path": output_path,
        },
        metrics={
            "event_count": summary["event_count"],
            "promoted_candidate_count": len(summary["promoted_candidate_ids"]),
            "failed_candidate_family_count": len(summary["failed_candidate_families"]),
        },
        next_hint=summary["next_command"],
    )
