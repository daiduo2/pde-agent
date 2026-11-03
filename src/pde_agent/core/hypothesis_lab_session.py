"""HypothesisLabSession - unified execution context for agent infrastructure.

Provides a queryable, incrementally persistent session that replaces
the scattered JSON file passing pattern with a unified state container.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_artifact_store import (
    ArtifactStoreError,
    HypothesisLabArtifactStore,
)
from pde_agent.core.hypothesis_lab_events import (
    EVENT_CHECKPOINT_RESUMED,
    EVENT_CHECKPOINT_SAVED,
    HypothesisLabEventError,
    HypothesisLabEventStream,
)


SESSION_SCHEMA_VERSION = 1


class HypothesisLabSessionError(RuntimeError):
    pass


@dataclass
class SessionCheckpoint:
    """Checkpoint for resumable execution."""

    stage: str
    stage_data: dict[str, Any]
    completed_steps: list[str]
    timestamp: str
    checkpoint_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage,
            "stage_data": self.stage_data,
            "completed_steps": self.completed_steps,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionCheckpoint:
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            stage=str(data["stage"]),
            stage_data=dict(data.get("stage_data", {})),
            completed_steps=list(data.get("completed_steps", [])),
            timestamp=str(data["timestamp"]),
        )


class HypothesisLabSession:
    """Unified execution context for Hypothesis Lab agent workflows.

    A Session manages:
    - Artifact storage (via ArtifactStore)
    - Event streaming (via EventStream)
    - Checkpoint save/resume for incremental execution
    - Query interface for agent introspection

    Usage:
        session = HypothesisLabSession.create(workspace="runs/exp_001")
        # ... run primitives, each operating on the session ...
        session.save_checkpoint("sandbox", {"param_idx": 5, "best_score": 0.023})
    """

    def __init__(
        self,
        session_id: str,
        workspace: str | Path,
        artifact_store: HypothesisLabArtifactStore,
        event_stream: HypothesisLabEventStream,
    ):
        self.session_id = session_id
        self.workspace = Path(workspace)
        self.artifact_store = artifact_store
        self.event_stream = event_stream
        self._checkpoint_path = self.workspace / "checkpoints" / "latest.json"

    @classmethod
    def create(
        cls,
        workspace: str | Path,
        session_id: str | None = None,
    ) -> HypothesisLabSession:
        """Create a new session with fresh workspace."""
        ws = Path(workspace)
        ws.mkdir(parents=True, exist_ok=True)
        sid = session_id or _new_session_id()

        event_stream = HypothesisLabEventStream(ws / "events.jsonl")
        artifact_store = HypothesisLabArtifactStore(ws, event_stream=event_stream)

        session = cls(
            session_id=sid,
            workspace=ws,
            artifact_store=artifact_store,
            event_stream=event_stream,
        )
        session._write_session_meta()
        return session

    @classmethod
    def load(cls, workspace: str | Path) -> HypothesisLabSession:
        """Load an existing session from workspace."""
        ws = Path(workspace)
        meta_path = ws / "session.json"
        if not meta_path.exists():
            raise HypothesisLabSessionError(f"no session found at {ws}")

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise HypothesisLabSessionError(f"invalid session meta: {exc}") from exc

        sid = str(meta.get("session_id", ""))
        if not sid:
            raise HypothesisLabSessionError("session_id missing from session meta")

        event_stream = HypothesisLabEventStream(ws / "events.jsonl")
        artifact_store = HypothesisLabArtifactStore(ws, event_stream=event_stream)

        return cls(
            session_id=sid,
            workspace=ws,
            artifact_store=artifact_store,
            event_stream=event_stream,
        )

    def _write_session_meta(self) -> None:
        meta = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "workspace": str(self.workspace),
        }
        meta_path = self.workspace / "session.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # -- Artifact convenience methods --

    def put_artifact(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.artifact_store.put(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            payload=payload,
        )

    def get_artifact(
        self,
        artifact_type: str,
        artifact_id: str,
    ) -> dict[str, Any] | None:
        return self.artifact_store.get(artifact_type, artifact_id)

    def list_artifacts(self, artifact_type: str) -> list[dict[str, Any]]:
        refs = self.artifact_store.list_by_type(artifact_type)
        result: list[dict[str, Any]] = []
        for ref in refs:
            payload = self.artifact_store.get(ref.artifact_type, ref.artifact_id)
            if payload is not None:
                result.append({
                    "artifact_type": ref.artifact_type,
                    "artifact_id": ref.artifact_id,
                    "version": ref.version,
                    "created_at": ref.created_at,
                    "updated_at": ref.updated_at,
                    "payload": payload,
                })
        return result

    # -- Event convenience methods --

    def emit(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        tags: list[str] | None = None,
    ) -> None:
        self.event_stream.emit(
            event_type=event_type,
            payload=payload,
            run_id=self.session_id,
            tags=tags,
        )

    def get_events(
        self,
        *,
        event_type: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        events = self.event_stream.read_events(
            event_type=event_type,
            tag=tag,
            limit=limit,
        )
        return [evt.to_dict() for evt in events]

    # -- Checkpointing --

    def save_checkpoint(
        self,
        stage: str,
        stage_data: dict[str, Any],
        completed_steps: list[str] | None = None,
    ) -> SessionCheckpoint:
        checkpoint = SessionCheckpoint(
            checkpoint_id=_new_checkpoint_id(),
            stage=stage,
            stage_data=dict(stage_data),
            completed_steps=list(completed_steps or []),
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.write_text(
            json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.emit(
            event_type=EVENT_CHECKPOINT_SAVED,
            payload={
                "checkpoint_id": checkpoint.checkpoint_id,
                "stage": stage,
                "completed_steps": checkpoint.completed_steps,
            },
            tags=["checkpoint", stage],
        )
        return checkpoint

    def load_checkpoint(self) -> SessionCheckpoint | None:
        if not self._checkpoint_path.exists():
            return None
        try:
            data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            checkpoint = SessionCheckpoint.from_dict(data)
            self.emit(
                event_type=EVENT_CHECKPOINT_RESUMED,
                payload={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "stage": checkpoint.stage,
                    "completed_steps": checkpoint.completed_steps,
                },
                tags=["checkpoint", checkpoint.stage],
            )
            return checkpoint
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    def clear_checkpoint(self) -> None:
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

    # -- Query interface (lightweight) --

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of current session state for agent introspection."""
        manifest = self.artifact_store.get_manifest()
        artifact_counts: dict[str, int] = {}
        for entry in manifest.get("artifacts", {}).values():
            if isinstance(entry, dict):
                atype = str(entry.get("artifact_type", "unknown"))
                artifact_counts[atype] = artifact_counts.get(atype, 0) + 1

        checkpoint = self.load_checkpoint()

        return {
            "session_id": self.session_id,
            "workspace": str(self.workspace),
            "artifact_counts": artifact_counts,
            "total_artifacts": sum(artifact_counts.values()),
            "event_count": self.event_stream.count(),
            "current_checkpoint": checkpoint.to_dict() if checkpoint else None,
        }

    def get_latest_artifact(self, artifact_type: str) -> dict[str, Any] | None:
        refs = self.artifact_store.list_by_type(artifact_type)
        if not refs:
            return None
        latest = refs[0]
        payload = self.artifact_store.get(latest.artifact_type, latest.artifact_id)
        if payload is None:
            return None
        return {
            "artifact_type": latest.artifact_type,
            "artifact_id": latest.artifact_id,
            "version": latest.version,
            "payload": payload,
        }


def _new_session_id() -> str:
    return f"hls_{uuid.uuid4().hex[:12]}"


def _new_checkpoint_id() -> str:
    return f"ck_{uuid.uuid4().hex[:12]}"
