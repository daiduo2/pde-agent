"""Offline artifact backbone for PDE literature-to-algorithm candidates."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SOURCE_RECORDS_STATE_HINT = "LITERATURE_SOURCE_RECORDS_BUILT"
PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"
ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
ALGORITHM_CANDIDATES_STATE_HINT = "ALGORITHM_CANDIDATES_COMPILED"
SANDBOX_PLAN_STATE_HINT = "ALGORITHM_CANDIDATE_SANDBOX_PLANNED"

VALID_SOURCE_TYPES = {
    "openalex_work",
    "arxiv_paper",
    "github_repo",
    "competition_resource",
    "manual_source",
}


class LiteratureAlgorithmLabError(ValueError):
    """Raised when literature algorithm lab artifacts are invalid."""


def build_literature_source_records(
    *,
    source_fixture_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    source_file = _require_file(Path(source_fixture_path), "source fixture")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    fixture = _read_json(source_file)
    query = _nonempty_text(fixture.get("query"), "query")
    raw_sources = fixture.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise LiteratureAlgorithmLabError("sources must be a non-empty list")

    records = []
    seen_ids: set[str] = set()
    for item in raw_sources:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError("each source must be an object")
        source_type = _source_type(item.get("source_type"))
        title = _nonempty_text(item.get("title"), "title")
        raw_path = _require_file(
            Path(_nonempty_text(item.get("local_raw_path"), "local_raw_path")),
            "local raw source",
        )
        source_id = _dedupe_id(f"{source_type}_{_slug(title)}", seen_ids)
        seen_ids.add(source_id)
        records.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "title": title,
                "url": _nonempty_text(item.get("url"), "url"),
                "retrieved_at": _utc_now(),
                "license_note": _nonempty_text(
                    item.get("license_note"),
                    "license_note",
                ),
                "local_raw_path": str(raw_path),
                "content_hash": _sha256_file(raw_path),
                "related_source_ids": _text_list(
                    item.get("related_source_ids", []),
                    "related_source_ids",
                ),
                "retrieval_query": query,
            }
        )

    report = {
        "schema_version": 1,
        "state_hint": SOURCE_RECORDS_STATE_HINT,
        "source_records_path": str(output_file),
        "source_count": len(records),
        "source_records": records,
        "next_command": "ingest-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def _source_type(value: Any) -> str:
    text = _nonempty_text(value, "source_type")
    if text not in VALID_SOURCE_TYPES:
        raise LiteratureAlgorithmLabError(f"unsupported source_type: {text}")
    return text


def _dedupe_id(base: str, seen: set[str]) -> str:
    if base not in seen:
        return base
    index = 2
    while f"{base}_{index}" in seen:
        index += 1
    return f"{base}_{index}"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "source"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiteratureAlgorithmLabError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise LiteratureAlgorithmLabError(f"output file already exists: {path}")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise LiteratureAlgorithmLabError(f"{label} not found: {path}")
    return path


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureAlgorithmLabError(f"{label} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        result.append(_nonempty_text(item, label))
    return result
