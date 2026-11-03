"""Event streaming infrastructure for Hypothesis Lab agent infrastructure.

Provides typed event emission and consumption for agent-observable
scientific experimentation workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EVENT_SCHEMA_VERSION = 1


class HypothesisLabEventError(RuntimeError):
    pass


@dataclass(frozen=True)
class HypothesisLabEvent:
    """A single event in the hypothesis lab event stream."""

    event_type: str
    payload: dict[str, Any]
    timestamp: str
    event_id: str
    run_id: str | None = None
    source_artifact_path: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "source_artifact_path": self.source_artifact_path,
            "tags": self.tags,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HypothesisLabEvent:
        return cls(
            event_type=str(data["event_type"]),
            payload=dict(data.get("payload", {})),
            timestamp=str(data["timestamp"]),
            event_id=str(data["event_id"]),
            run_id=data.get("run_id"),
            source_artifact_path=data.get("source_artifact_path"),
            tags=list(data.get("tags", [])),
        )


class HypothesisLabEventStream:
    """Append-only event stream for agent-observable experiment workflows.

    Events are written to a JSONL file and can be filtered by type,
    allowing agents to consume the stream incrementally.
    """

    def __init__(self, stream_path: str | Path):
        self.stream_path = Path(stream_path)

    def emit(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        run_id: str | None = None,
        source_artifact_path: str | Path | None = None,
        tags: list[str] | None = None,
    ) -> HypothesisLabEvent:
        if not isinstance(event_type, str) or not event_type.strip():
            raise HypothesisLabEventError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            raise HypothesisLabEventError("payload must be a JSON object")

        timestamp = datetime.now(UTC).isoformat()
        event = HypothesisLabEvent(
            event_type=event_type,
            payload=payload,
            timestamp=timestamp,
            event_id=_event_id(self.stream_path, event_type, payload, timestamp),
            run_id=run_id,
            source_artifact_path=str(source_artifact_path) if source_artifact_path is not None else None,
            tags=[tag for tag in (tags or []) if isinstance(tag, str)],
        )
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stream_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True))
            handle.write("\n")
        return event

    def read_events(
        self,
        *,
        event_type: str | None = None,
        tag: str | None = None,
        after_event_id: str | None = None,
        limit: int | None = None,
    ) -> list[HypothesisLabEvent]:
        if not self.stream_path.exists():
            return []

        events: list[HypothesisLabEvent] = []
        skip_until_after = after_event_id is not None

        for line in self.stream_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            evt = HypothesisLabEvent.from_dict(data)

            if skip_until_after:
                if evt.event_id == after_event_id:
                    skip_until_after = False
                continue

            if event_type is not None and evt.event_type != event_type:
                continue
            if tag is not None and tag not in evt.tags:
                continue

            events.append(evt)
            if limit is not None and len(events) >= limit:
                break

        return events

    def read_latest(self, event_type: str) -> HypothesisLabEvent | None:
        events = self.read_events(event_type=event_type)
        return events[-1] if events else None

    def count(self, *, event_type: str | None = None, tag: str | None = None) -> int:
        return len(self.read_events(event_type=event_type, tag=tag))


# Standard event types for agent infrastructure
EVENT_STAGE_STARTED = "stage_started"
EVENT_STAGE_COMPLETED = "stage_completed"
EVENT_STAGE_FAILED = "stage_failed"
EVENT_ARTIFACT_CREATED = "artifact_created"
EVENT_ARTIFACT_UPDATED = "artifact_updated"
EVENT_SANDBOX_PARAM_STARTED = "sandbox_param_started"
EVENT_SANDBOX_PARAM_COMPLETED = "sandbox_param_completed"
EVENT_SANDBOX_PARAM_FAILED = "sandbox_param_failed"
EVENT_CHECKPOINT_SAVED = "checkpoint_saved"
EVENT_CHECKPOINT_RESUMED = "checkpoint_resumed"
EVENT_CANDIDATE_PROMOTED = "candidate_promoted"
EVENT_CANDIDATE_REJECTED = "candidate_rejected"
EVENT_CANDIDATE_REFINED = "candidate_refined"
EVENT_QUERY_EXECUTED = "query_executed"


def _event_id(
    stream_path: Path,
    event_type: str,
    payload: dict[str, Any],
    timestamp: str,
) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(stream_path).encode())
    digest.update(event_type.encode())
    digest.update(timestamp.encode())
    digest.update(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:16]
