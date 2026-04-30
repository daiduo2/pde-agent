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


def ingest_mineru_parse(
    *,
    source_records_path: str | Path,
    source_id: str,
    markdown_path: str | Path,
    asset_dir: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    source_records = _read_json(
        _require_file(Path(source_records_path), "source records")
    )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    markdown_file = _require_file(Path(markdown_path), "MinerU markdown")
    selected_source_id = _nonempty_text(source_id, "source_id")
    source_ids = {
        record.get("source_id")
        for record in source_records.get("source_records", [])
        if isinstance(record, dict)
    }
    if selected_source_id not in source_ids:
        raise LiteratureAlgorithmLabError(
            f"source_id not found in source records: {selected_source_id}"
        )

    markdown = markdown_file.read_text(encoding="utf-8")
    section_index = _section_index(markdown)
    formula_count = markdown.count("$$") // 2
    figure_count = markdown.count("![")
    table_count = markdown.count("\n|")
    quality_flags = _parse_quality_flags(
        markdown=markdown,
        section_index=section_index,
        formula_count=formula_count,
    )
    quality = "usable" if not quality_flags else "low_quality"
    report = {
        "schema_version": 1,
        "state_hint": PAPER_PARSE_STATE_HINT,
        "parse_id": f"parse_{selected_source_id}",
        "source_id": selected_source_id,
        "parser": "mineru",
        "parser_version": "external",
        "markdown_path": str(markdown_file),
        "asset_dir": str(Path(asset_dir)),
        "parse_quality": {
            "quality": quality,
            "quality_flags": quality_flags,
        },
        "section_index": section_index,
        "formula_count": formula_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "paper_parse_path": str(output_file),
        "next_command": "extract-algorithm-cards",
    }
    _write_json(output_file, report)
    return report


def extract_algorithm_cards(
    *,
    paper_parse_paths: list[str | Path],
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    if not paper_parse_paths:
        raise LiteratureAlgorithmLabError("paper_parse_paths must be non-empty")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    cards = []
    for parse_path in paper_parse_paths:
        parse = _read_json(_require_file(Path(parse_path), "paper parse"))
        if parse.get("state_hint") != PAPER_PARSE_STATE_HINT:
            raise LiteratureAlgorithmLabError("paper parse has invalid state_hint")
        if parse.get("parse_quality", {}).get("quality") != "usable":
            continue
        markdown_file = _require_file(
            Path(_nonempty_text(parse.get("markdown_path"), "markdown_path")),
            "parsed markdown",
        )
        fields = _extract_colon_fields(markdown_file.read_text(encoding="utf-8"))
        source_id = _nonempty_text(parse.get("source_id"), "source_id")
        hook = _nonempty_text(fields.get("hook"), "Hook")
        card = {
            "algorithm_card_id": f"card_{source_id}_{_slug(hook)}",
            "source_ids": [source_id],
            "claim": _nonempty_text(fields.get("claim"), "Claim"),
            "core_mechanism": _nonempty_text(
                fields.get("mechanism"),
                "Mechanism",
            ),
            "operation": _nonempty_text(fields.get("operation"), "Operation"),
            "preconditions": ["PDEBench-style neural operator rollout"],
            "implementation_hooks": [hook],
            "expected_observables": [
                _nonempty_text(fields.get("observable"), "Observable")
            ],
            "failure_modes": [_nonempty_text(fields.get("risk"), "Risk")],
            "minimal_experiment": _nonempty_text(
                fields.get("minimal experiment"),
                "Minimal Experiment",
            ),
            "evidence_status": "extracted",
        }
        cards.append(card)
    if not cards:
        raise LiteratureAlgorithmLabError("no usable algorithm cards extracted")
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CARDS_STATE_HINT,
        "algorithm_cards_path": str(output_file),
        "algorithm_card_count": len(cards),
        "algorithm_cards": cards,
        "next_command": "compile-algorithm-candidates",
    }
    _write_json(output_file, report)
    return report


def _section_index(markdown: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            name = stripped.lstrip("#").strip().lower()
            if name:
                sections[name] = line_number
    return sections


def _parse_quality_flags(
    *,
    markdown: str,
    section_index: dict[str, int],
    formula_count: int,
) -> list[str]:
    flags = []
    lowered_sections = set(section_index)
    if "abstract" not in lowered_sections:
        flags.append("missing_abstract")
    if "method" not in lowered_sections and "methods" not in lowered_sections:
        flags.append("missing_method")
    if len(markdown.strip()) < 200:
        flags.append("markdown_too_short")
    if formula_count == 0:
        flags.append("no_formula_blocks")
    return flags


def _extract_colon_fields(markdown: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in markdown.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = label.strip().lower()
        text = value.strip()
        if key and text:
            fields[key] = text
    return fields


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
