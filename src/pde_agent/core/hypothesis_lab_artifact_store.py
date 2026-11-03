"""Artifact store for unified Hypothesis Lab state management.

Replaces scattered JSON file passing with a queryable, indexed store
that agents can introspect without knowing file paths or internal schemas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_events import (
    EVENT_ARTIFACT_CREATED,
    EVENT_ARTIFACT_UPDATED,
    HypothesisLabEventStream,
)


ARTIFACT_STORE_SCHEMA_VERSION = 1


class ArtifactStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a stored artifact."""

    artifact_type: str
    artifact_id: str
    path: Path
    created_at: str
    updated_at: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "path": str(self.path),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }


class HypothesisLabArtifactStore:
    """Unified artifact store for Hypothesis Lab experiments.

    Artifacts are stored as JSON files in a structured directory layout:
        workspace/artifacts/{type}/{id}.json

    A manifest tracks all artifacts with metadata for querying.
    """

    def __init__(
        self,
        workspace: str | Path,
        event_stream: HypothesisLabEventStream | None = None,
    ):
        self.workspace = Path(workspace)
        self.artifacts_dir = self.workspace / "artifacts"
        self.manifest_path = self.artifacts_dir / "manifest.json"
        self.event_stream = event_stream
        self._manifest: dict[str, Any] | None = None

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is not None:
            return self._manifest
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._manifest = data
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        self._manifest = {
            "schema_version": ARTIFACT_STORE_SCHEMA_VERSION,
            "artifacts": {},
        }
        return self._manifest

    def _save_manifest(self) -> None:
        manifest = self._load_manifest()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _artifact_path(self, artifact_type: str, artifact_id: str) -> Path:
        safe_type = _safe_name(artifact_type)
        safe_id = _safe_name(artifact_id)
        return self.artifacts_dir / safe_type / f"{safe_id}.json"

    def put(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> ArtifactRef:
        if not isinstance(artifact_type, str) or not artifact_type.strip():
            raise ArtifactStoreError("artifact_type must be a non-empty string")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ArtifactStoreError("artifact_id must be a non-empty string")
        if not isinstance(payload, dict):
            raise ArtifactStoreError("payload must be a JSON object")

        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        path = self._artifact_path(artifact_type, artifact_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest = self._load_manifest()
        artifacts = manifest.setdefault("artifacts", {})
        key = f"{artifact_type}:{artifact_id}"

        existing = artifacts.get(key)
        version = 1
        created_at = now
        if existing is not None:
            version = int(existing.get("version", 1)) + 1
            created_at = existing.get("created_at", now)

        ref = ArtifactRef(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            path=path,
            created_at=created_at,
            updated_at=now,
            version=version,
        )
        artifacts[key] = ref.to_dict()
        self._save_manifest()

        if self.event_stream is not None:
            event_type = EVENT_ARTIFACT_UPDATED if version > 1 else EVENT_ARTIFACT_CREATED
            self.event_stream.emit(
                event_type=event_type,
                payload={
                    "artifact_type": artifact_type,
                    "artifact_id": artifact_id,
                    "version": version,
                    "path": str(path),
                },
                tags=[artifact_type],
            )

        return ref

    def get(
        self,
        artifact_type: str,
        artifact_id: str,
    ) -> dict[str, Any] | None:
        path = self._artifact_path(artifact_type, artifact_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def get_ref(self, artifact_type: str, artifact_id: str) -> ArtifactRef | None:
        manifest = self._load_manifest()
        key = f"{artifact_type}:{artifact_id}"
        entry = manifest.get("artifacts", {}).get(key)
        if entry is None:
            return None
        return ArtifactRef(
            artifact_type=str(entry["artifact_type"]),
            artifact_id=str(entry["artifact_id"]),
            path=Path(entry["path"]),
            created_at=str(entry["created_at"]),
            updated_at=str(entry["updated_at"]),
            version=int(entry.get("version", 1)),
        )

    def list_by_type(self, artifact_type: str) -> list[ArtifactRef]:
        manifest = self._load_manifest()
        refs: list[ArtifactRef] = []
        for key, entry in manifest.get("artifacts", {}).items():
            if isinstance(entry, dict) and entry.get("artifact_type") == artifact_type:
                refs.append(
                    ArtifactRef(
                        artifact_type=str(entry["artifact_type"]),
                        artifact_id=str(entry["artifact_id"]),
                        path=Path(entry["path"]),
                        created_at=str(entry["created_at"]),
                        updated_at=str(entry["updated_at"]),
                        version=int(entry.get("version", 1)),
                    )
                )
        return sorted(refs, key=lambda r: r.updated_at, reverse=True)

    def list_all(self) -> list[ArtifactRef]:
        manifest = self._load_manifest()
        refs: list[ArtifactRef] = []
        for entry in manifest.get("artifacts", {}).values():
            if isinstance(entry, dict):
                refs.append(
                    ArtifactRef(
                        artifact_type=str(entry["artifact_type"]),
                        artifact_id=str(entry["artifact_id"]),
                        path=Path(entry["path"]),
                        created_at=str(entry["created_at"]),
                        updated_at=str(entry["updated_at"]),
                        version=int(entry.get("version", 1)),
                    )
                )
        return sorted(refs, key=lambda r: r.updated_at, reverse=True)

    def exists(self, artifact_type: str, artifact_id: str) -> bool:
        return self.get(artifact_type, artifact_id) is not None

    def delete(self, artifact_type: str, artifact_id: str) -> bool:
        manifest = self._load_manifest()
        key = f"{artifact_type}:{artifact_id}"
        if key not in manifest.get("artifacts", {}):
            return False
        path = self._artifact_path(artifact_type, artifact_id)
        if path.exists():
            path.unlink()
        del manifest["artifacts"][key]
        self._save_manifest()
        return True

    def get_manifest(self) -> dict[str, Any]:
        return self._load_manifest()


def _safe_name(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_\-]", "_", value)[:128]
