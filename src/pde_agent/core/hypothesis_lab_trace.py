"""Append-only trace events for Hypothesis Lab and Hermes research loops."""


import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TRACE_EVENT_SCHEMA_VERSION = 1
SUMMARY_STATE_HINT = "HYPOTHESIS_LAB_TRACE_SUMMARIZED"


class HypothesisLabTraceError(RuntimeError):
    pass


class HypothesisLabTraceStore:
    """Append-only JSONL trace store for agent scientific events."""

    def __init__(self, trace_path: str | Path):
        self.trace_path = Path(trace_path)

    def append_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        run_id: str | None = None,
        source_artifact_path: str | Path | None = None,
        tags: list[str] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        event_type = _required_text(event_type, "event_type")
        if not isinstance(payload, dict):
            raise HypothesisLabTraceError("payload must be a JSON object")
        emitted_at = timestamp or datetime.now(UTC).isoformat()
        event = {
            "schema_version": TRACE_EVENT_SCHEMA_VERSION,
            "event_id": _event_id(
                trace_path=self.trace_path,
                event_type=event_type,
                payload=payload,
                timestamp=emitted_at,
            ),
            "timestamp": emitted_at,
            "event_type": event_type,
            "run_id": run_id,
            "source_artifact_path": str(source_artifact_path)
            if source_artifact_path is not None
            else None,
            "tags": [tag for tag in tags or [] if isinstance(tag, str)],
            "payload": payload,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")
        return event

    def read_events(self) -> list[dict[str, Any]]:
        return read_hypothesis_lab_trace(self.trace_path)


def read_hypothesis_lab_trace(trace_path: str | Path) -> list[dict[str, Any]]:
    trace_file = Path(trace_path)
    if not trace_file.exists():
        raise HypothesisLabTraceError(f"trace file does not exist: {trace_file}")
    events = []
    for line_number, line in enumerate(
        trace_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HypothesisLabTraceError(
                f"invalid trace JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise HypothesisLabTraceError(
                f"trace line {line_number} must be a JSON object"
            )
        if event.get("schema_version") != TRACE_EVENT_SCHEMA_VERSION:
            raise HypothesisLabTraceError(
                f"trace line {line_number} has unsupported schema_version"
            )
        events.append(event)
    return events


def summarize_hypothesis_lab_trace(
    *,
    trace_path: str | Path,
    output_path: str | Path | None,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    events = read_hypothesis_lab_trace(trace_path)
    event_counts_by_type: dict[str, int] = {}
    candidate_family_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    promoted_candidate_ids: list[str] = []
    failed_candidate_families: list[str] = []
    score_deltas: list[float] = []

    for event in events:
        event_type = str(event.get("event_type", ""))
        _increment(event_counts_by_type, event_type)
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        family = _optional_text(payload.get("candidate_family"))
        if family:
            _increment(candidate_family_counts, family)
        outcome = _optional_text(payload.get("outcome"))
        if outcome:
            _increment(outcome_counts, outcome)
        candidate_id = _optional_text(payload.get("algorithm_candidate_id"))
        if outcome == "promoted" and candidate_id:
            _append_unique(promoted_candidate_ids, candidate_id)
        if outcome in {"failed", "rejected", "blocked"} and family:
            _append_unique(failed_candidate_families, family)
        score_delta = payload.get("score_delta")
        if isinstance(score_delta, int | float) and not isinstance(score_delta, bool):
            score_deltas.append(float(score_delta))

    summary = {
        "schema_version": 1,
        "state_hint": SUMMARY_STATE_HINT,
        "trace_path": str(trace_path),
        "event_count": len(events),
        "event_counts_by_type": event_counts_by_type,
        "candidate_family_counts": candidate_family_counts,
        "outcome_counts": outcome_counts,
        "promoted_candidate_ids": promoted_candidate_ids,
        "failed_candidate_families": failed_candidate_families,
        "score_delta_summary": _score_delta_summary(score_deltas),
        "next_command": "detect-hypothesis-doom-loop",
    }
    _write_optional_json(output_file, summary)
    return summary


def _event_id(
    *,
    trace_path: Path,
    event_type: str,
    payload: dict[str, Any],
    timestamp: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(trace_path).encode())
    digest.update(event_type.encode())
    digest.update(timestamp.encode())
    digest.update(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HypothesisLabTraceError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _increment(counts: dict[str, int], key: str) -> None:
    if key:
        counts[key] = counts.get(key, 0) + 1


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _score_delta_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "best": None, "worst": None, "mean": None}
    return {
        "count": len(values),
        "best": max(values),
        "worst": min(values),
        "mean": sum(values) / len(values),
    }


def _guard_output(path: Path | None, *, force: bool) -> None:
    if path is not None and path.exists() and not force:
        raise HypothesisLabTraceError(f"output already exists: {path}")


def _write_optional_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
