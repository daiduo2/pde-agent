from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class LiteratureAlgorithmLabError(ValueError):
    """Raised when literature algorithm lab artifacts are invalid."""


def source_candidate(
    *,
    source_id: str,
    source_type: str,
    title: str,
    url: str,
    document_url: str,
    retrieval_query: str,
    discovery_provider: str,
    license_note: str,
    external_ids: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_type": source_type,
        "title": title,
        "url": url,
        "document_url": document_url,
        "retrieval_query": retrieval_query,
        "discovery_provider": discovery_provider,
        "license_note": license_note,
        "external_ids": external_ids,
    }


def object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError(f"{label} item must be an object")
        result.append(item)
    return result


def first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def xml_text(element: ET.Element, path: str, namespace: dict[str, str]) -> str | None:
    found = element.find(path, namespace)
    if found is None or found.text is None:
        return None
    return found.text.strip()


def slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return text or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    from pde_agent.core.json_io import read_json_dict

    return read_json_dict(
        path,
        error_cls=LiteratureAlgorithmLabError,
        error_message="JSON artifact must be an object",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    from pde_agent.core.json_io import write_json as _write_json

    _write_json(path, payload)


def guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise LiteratureAlgorithmLabError(f"artifact already exists: {path}")


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise LiteratureAlgorithmLabError(f"{label} not found: {path}")
    return path


def nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureAlgorithmLabError(f"{label} must be a non-empty string")
    return value.strip()
