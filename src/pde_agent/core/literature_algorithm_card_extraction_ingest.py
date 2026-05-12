"""Literature algorithm card extraction ingest functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError
from pde_agent.core.literature_algorithm_card_extraction_helpers import (
    _guard_output,
    _nonempty_text,
    _read_json,
    _require_file,
    _write_json,
)

SOURCE_DOCUMENT_STATE_HINT = "SOURCE_DOCUMENT_DOWNLOADED"
PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"

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
    source_ids = _source_ids_from_source_artifact(source_records)
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
    has_method_section = any(
        keyword in section
        for section in lowered_sections
        for keyword in ["method", "approach", "algorithm", "neural operator"]
    )
    if not has_method_section:
        flags.append("missing_method")
    if len(markdown.strip()) < 200:
        flags.append("markdown_too_short")
    if formula_count == 0:
        flags.append("no_formula_blocks")
    return flags

def _section_index(markdown: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            name = stripped.lstrip("#").strip().lower()
            if name:
                sections[name] = line_number
    return sections

def _source_ids_from_source_artifact(payload: dict[str, Any]) -> set[Any]:
    if payload.get("state_hint") == SOURCE_DOCUMENT_STATE_HINT:
        record = payload.get("source_record")
        if isinstance(record, dict):
            return {record.get("source_id")}
        return set()
    return {
        record.get("source_id")
        for record in payload.get("source_records", [])
        if isinstance(record, dict)
    }
