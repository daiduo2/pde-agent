# Literature Algorithm Lab Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline artifact backbone for a PDE-focused literature-to-algorithm lab that turns local source fixtures and MinerU markdown into benchmark-ready algorithm candidates and sandbox plans.

**Architecture:** Add a pure core module that validates and writes `source_record`, `paper_parse`, `algorithm_card`, `algorithm_candidate`, and `sandbox_plan` artifacts. Add thin command wrappers and CLI registration following the existing `pde_agent.commands.*` pattern. Keep V0 offline: no live network calls, no MinerU runtime invocation, and no third-party code execution.

**Tech Stack:** Python 3.11+, `pathlib`, `json`, `hashlib`, existing Typer CLI, existing `success_payload`, existing pytest fixture style.

---

## Scope

This plan implements only Phase 1 from the design spec:

- normalize local search/source fixtures into source records,
- ingest existing MinerU markdown output and calculate parse quality,
- extract deterministic algorithm cards from structured markdown notes,
- compile supported cards into algorithm candidates,
- plan the cheapest valid Hermes sandbox route,
- expose these actions through JSON-only CLI commands.

This plan intentionally does not implement live OpenAlex/arXiv/GitHub search, PDF download, or MinerU execution.

## File Structure

- Create `src/pde_agent/core/literature_algorithm_lab.py`
  - Owns all schema validation, deterministic extraction, candidate compilation, sandbox planning, and JSON writing.
- Create `src/pde_agent/commands/literature_algorithm_lab.py`
  - Thin command wrappers returning `success_payload`.
- Modify `src/pde_agent/cli.py`
  - Register commands:
    - `build-literature-source-records`
    - `ingest-mineru-parse`
    - `extract-algorithm-cards`
    - `compile-algorithm-candidates`
    - `plan-algorithm-candidate-sandbox`
- Create `tests/test_literature_algorithm_lab.py`
  - Unit tests for core functions and CLI wrappers.

---

### Task 1: Source Record Normalization

**Files:**
- Create: `src/pde_agent/core/literature_algorithm_lab.py`
- Create: `tests/test_literature_algorithm_lab.py`

- [ ] **Step 1: Write the failing test**

Add this test file with the source-record test first:

```python
from __future__ import annotations

import json
from pathlib import Path


def test_build_source_records_normalizes_fixture_sources(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        build_literature_source_records,
    )

    raw_pdf = tmp_path / "fno.pdf"
    raw_pdf.write_bytes(b"fno paper bytes")
    search_fixture = tmp_path / "search_fixture.json"
    search_fixture.write_text(
        json.dumps(
            {
                "query": "Fourier neural operator Burgers",
                "sources": [
                    {
                        "source_type": "arxiv_paper",
                        "title": "Fourier Neural Operator for Parametric PDEs",
                        "url": "https://arxiv.org/abs/2010.08895",
                        "local_raw_path": str(raw_pdf),
                        "license_note": "arXiv non-commercial research use",
                        "related_source_ids": ["openalex:W123"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_literature_source_records(
        source_fixture_path=search_fixture,
        output_path=tmp_path / "source_records.json",
        force=False,
    )

    assert report["state_hint"] == "LITERATURE_SOURCE_RECORDS_BUILT"
    assert report["source_count"] == 1
    record = report["source_records"][0]
    assert record["source_id"].startswith("arxiv_paper_fourier_neural_operator")
    assert record["source_type"] == "arxiv_paper"
    assert record["retrieval_query"] == "Fourier neural operator Burgers"
    assert record["local_raw_path"] == str(raw_pdf)
    assert record["content_hash"]
    assert record["related_source_ids"] == ["openalex:W123"]
    assert report["next_command"] == "ingest-mineru-parse"


def test_build_source_records_rejects_missing_raw_file(tmp_path: Path) -> None:
    import pytest

    from pde_agent.core.literature_algorithm_lab import (
        LiteratureAlgorithmLabError,
        build_literature_source_records,
    )

    search_fixture = tmp_path / "search_fixture.json"
    search_fixture.write_text(
        json.dumps(
            {
                "query": "missing",
                "sources": [
                    {
                        "source_type": "arxiv_paper",
                        "title": "Missing Paper",
                        "url": "https://arxiv.org/abs/0000.00000",
                        "local_raw_path": str(tmp_path / "missing.pdf"),
                        "license_note": "unknown",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LiteratureAlgorithmLabError, match="local raw source"):
        build_literature_source_records(
            source_fixture_path=search_fixture,
            output_path=tmp_path / "source_records.json",
            force=False,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py::test_build_source_records_normalizes_fixture_sources tests/test_literature_algorithm_lab.py::test_build_source_records_rejects_missing_raw_file
```

Expected: fails with `ModuleNotFoundError` for `pde_agent.core.literature_algorithm_lab`.

- [ ] **Step 3: Implement source-record core**

Create `src/pde_agent/core/literature_algorithm_lab.py`:

```python
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
        raw_path = _require_file(Path(_nonempty_text(item.get("local_raw_path"), "local_raw_path")), "local raw source")
        source_id = _dedupe_id(f"{source_type}_{_slug(title)}", seen_ids)
        seen_ids.add(source_id)
        records.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "title": title,
                "url": _nonempty_text(item.get("url"), "url"),
                "retrieved_at": _utc_now(),
                "license_note": _nonempty_text(item.get("license_note"), "license_note"),
                "local_raw_path": str(raw_path),
                "content_hash": _sha256_file(raw_path),
                "related_source_ids": _text_list(item.get("related_source_ids", []), "related_source_ids"),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py::test_build_source_records_normalizes_fixture_sources tests/test_literature_algorithm_lab.py::test_build_source_records_rejects_missing_raw_file
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add -- src/pde_agent/core/literature_algorithm_lab.py tests/test_literature_algorithm_lab.py
git commit -m "feat: add literature source records"
```

---

### Task 2: MinerU Parse Ingestion And Algorithm Cards

**Files:**
- Modify: `src/pde_agent/core/literature_algorithm_lab.py`
- Modify: `tests/test_literature_algorithm_lab.py`

- [ ] **Step 1: Write the failing tests**

Append these tests:

```python
def test_ingest_mineru_parse_scores_markdown_quality(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        build_literature_source_records,
        ingest_mineru_parse,
    )

    raw_pdf = tmp_path / "paper.pdf"
    raw_pdf.write_bytes(b"paper")
    fixture = tmp_path / "sources.json"
    fixture.write_text(
        json.dumps(
            {
                "query": "Burgers residual correction",
                "sources": [
                    {
                        "source_type": "arxiv_paper",
                        "title": "Residual Correction for Neural Operators",
                        "url": "https://arxiv.org/abs/2601.00001",
                        "local_raw_path": str(raw_pdf),
                        "license_note": "arXiv research use",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sources = build_literature_source_records(
        source_fixture_path=fixture,
        output_path=tmp_path / "source_records.json",
        force=False,
    )
    markdown = tmp_path / "paper.md"
    markdown.write_text(
        "\n".join(
            [
                "# Residual Correction for Neural Operators",
                "## Abstract",
                "We study neural operator rollout errors.",
                "## Method",
                "Claim: Horizon-conditioned residual correction reduces systematic rollout bias.",
                "Mechanism: Fit a lightweight residual map over checkpoint rollouts.",
                "Hook: learned_residual_correction",
                "Observable: lower early and middle Rel-MSE",
                "Risk: residual correction can overfit validation windows",
                "$$ u_t + u u_x = nu u_xx $$",
                "![Figure 1](figures/f1.png)",
            ]
        ),
        encoding="utf-8",
    )

    report = ingest_mineru_parse(
        source_records_path=tmp_path / "source_records.json",
        source_id=sources["source_records"][0]["source_id"],
        markdown_path=markdown,
        asset_dir=tmp_path / "figures",
        output_path=tmp_path / "paper_parse.json",
        force=False,
    )

    assert report["state_hint"] == "MINERU_PARSE_INGESTED"
    assert report["parse_quality"]["quality"] == "usable"
    assert report["formula_count"] == 1
    assert report["figure_count"] == 1
    assert "abstract" in report["section_index"]
    assert "method" in report["section_index"]
    assert report["next_command"] == "extract-algorithm-cards"


def test_extract_algorithm_cards_reads_operational_markdown(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import extract_algorithm_cards

    markdown = tmp_path / "paper.md"
    markdown.write_text(
        "\n".join(
            [
                "Title: Residual Correction for Neural Operators",
                "Claim: Horizon-conditioned residual correction reduces systematic rollout bias.",
                "Mechanism: Fit a lightweight residual map over checkpoint rollouts.",
                "Operation: train a small residual corrector and shrink it by horizon.",
                "Hook: learned_residual_correction",
                "Observable: lower early and middle Rel-MSE",
                "Risk: residual correction can overfit validation windows",
                "Minimal Experiment: compare persistence, checkpoint, and residual-corrected checkpoint on validation windows.",
            ]
        ),
        encoding="utf-8",
    )
    parse_report = {
        "schema_version": 1,
        "state_hint": "MINERU_PARSE_INGESTED",
        "source_id": "arxiv_paper_residual_correction",
        "markdown_path": str(markdown),
        "parse_quality": {"quality": "usable"},
    }
    parse_path = tmp_path / "paper_parse.json"
    parse_path.write_text(json.dumps(parse_report), encoding="utf-8")

    report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )

    assert report["state_hint"] == "ALGORITHM_CARDS_EXTRACTED"
    assert report["algorithm_card_count"] == 1
    card = report["algorithm_cards"][0]
    assert card["algorithm_card_id"].startswith("card_arxiv_paper_residual")
    assert card["claim"].startswith("Horizon-conditioned residual correction")
    assert card["core_mechanism"].startswith("Fit a lightweight residual map")
    assert card["operation"].startswith("train a small residual corrector")
    assert card["implementation_hooks"] == ["learned_residual_correction"]
    assert card["evidence_status"] == "extracted"
    assert report["next_command"] == "compile-algorithm-candidates"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py::test_ingest_mineru_parse_scores_markdown_quality tests/test_literature_algorithm_lab.py::test_extract_algorithm_cards_reads_operational_markdown
```

Expected: fails because `ingest_mineru_parse` and `extract_algorithm_cards` are undefined.

- [ ] **Step 3: Implement parse ingestion and card extraction**

Append these functions to `src/pde_agent/core/literature_algorithm_lab.py`:

```python
def ingest_mineru_parse(
    *,
    source_records_path: str | Path,
    source_id: str,
    markdown_path: str | Path,
    asset_dir: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    source_records = _read_json(_require_file(Path(source_records_path), "source records"))
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
        raise LiteratureAlgorithmLabError(f"source_id not found in source records: {selected_source_id}")

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
        markdown_file = _require_file(Path(_nonempty_text(parse.get("markdown_path"), "markdown_path")), "parsed markdown")
        fields = _extract_colon_fields(markdown_file.read_text(encoding="utf-8"))
        source_id = _nonempty_text(parse.get("source_id"), "source_id")
        hook = _nonempty_text(fields.get("hook"), "Hook")
        card = {
            "algorithm_card_id": f"card_{source_id}_{_slug(hook)}",
            "source_ids": [source_id],
            "claim": _nonempty_text(fields.get("claim"), "Claim"),
            "core_mechanism": _nonempty_text(fields.get("mechanism"), "Mechanism"),
            "operation": _nonempty_text(fields.get("operation"), "Operation"),
            "preconditions": ["PDEBench-style neural operator rollout"],
            "implementation_hooks": [hook],
            "expected_observables": [_nonempty_text(fields.get("observable"), "Observable")],
            "failure_modes": [_nonempty_text(fields.get("risk"), "Risk")],
            "minimal_experiment": _nonempty_text(fields.get("minimal experiment"), "Minimal Experiment"),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py::test_ingest_mineru_parse_scores_markdown_quality tests/test_literature_algorithm_lab.py::test_extract_algorithm_cards_reads_operational_markdown
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add -- src/pde_agent/core/literature_algorithm_lab.py tests/test_literature_algorithm_lab.py
git commit -m "feat: ingest literature algorithm cards"
```

---

### Task 3: Algorithm Candidate Compilation And Sandbox Planning

**Files:**
- Modify: `src/pde_agent/core/literature_algorithm_lab.py`
- Modify: `tests/test_literature_algorithm_lab.py`

- [ ] **Step 1: Write the failing tests**

Append these tests:

```python
def test_compile_algorithm_candidates_maps_supported_hooks(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import compile_algorithm_candidates

    cards_path = tmp_path / "algorithm_cards.json"
    cards_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": "card_checkpoint",
                        "source_ids": ["source_checkpoint"],
                        "claim": "Official checkpoints may beat persistence.",
                        "core_mechanism": "Use official checkpoint rollout and calibrate it.",
                        "operation": "evaluate checkpoint rollout with calibration weights",
                        "preconditions": ["checkpoint file available"],
                        "implementation_hooks": ["official_checkpoint_calibration"],
                        "expected_observables": ["higher local judge score"],
                        "failure_modes": ["checkpoint adapter mismatch"],
                        "minimal_experiment": "run checkpoint family sandbox",
                        "evidence_status": "extracted",
                    },
                    {
                        "algorithm_card_id": "card_residual",
                        "source_ids": ["source_residual"],
                        "claim": "Residual correction reduces rollout bias.",
                        "core_mechanism": "Fit horizon residual correction.",
                        "operation": "train a small residual corrector",
                        "preconditions": ["validation target available"],
                        "implementation_hooks": ["learned_residual_correction"],
                        "expected_observables": ["lower Rel-MSE"],
                        "failure_modes": ["overfit"],
                        "minimal_experiment": "run residual proxy sandbox",
                        "evidence_status": "extracted",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = compile_algorithm_candidates(
        algorithm_cards_path=cards_path,
        output_path=tmp_path / "algorithm_candidates.json",
        force=False,
    )

    assert report["state_hint"] == "ALGORITHM_CANDIDATES_COMPILED"
    assert report["algorithm_candidate_count"] == 2
    by_family = {
        item["candidate_family"]: item for item in report["algorithm_candidates"]
    }
    checkpoint = by_family["official_checkpoint_family"]
    assert checkpoint["executor_spec"]["cli_command"] == "run-checkpoint-family-sandbox"
    assert checkpoint["target_problem"] == "PDE_Burgers"
    residual = by_family["lightweight_residual_corrector"]
    assert residual["executor_spec"]["cli_command"] == "run-blueprint-sandbox"
    assert residual["promotion_criteria"]["requires_ablation"] is True
    assert report["next_command"] == "plan-algorithm-candidate-sandbox"


def test_plan_candidate_sandbox_selects_level_one_for_checkpoint(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import plan_candidate_sandbox

    candidate_path = tmp_path / "algorithm_candidates.json"
    candidate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_checkpoint",
                        "candidate_family": "official_checkpoint_family",
                        "executor_spec": {
                            "cli_command": "run-checkpoint-family-sandbox",
                            "required_options": [
                                "--checkpoint-path",
                                "--task-id",
                                "--test-path",
                                "--target-path",
                                "--output-dir",
                            ],
                        },
                        "risk_controls": ["do not execute external repo code"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = plan_candidate_sandbox(
        algorithm_candidates_path=candidate_path,
        algorithm_candidate_id="candidate_checkpoint",
        output_path=tmp_path / "sandbox_plan.json",
        force=False,
    )

    assert report["state_hint"] == "ALGORITHM_CANDIDATE_SANDBOX_PLANNED"
    assert report["algorithm_candidate_id"] == "candidate_checkpoint"
    assert report["sandbox_level"] == 1
    assert report["executor"]["cli_command"] == "run-checkpoint-family-sandbox"
    assert report["next_command"] == "run-checkpoint-family-sandbox"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py::test_compile_algorithm_candidates_maps_supported_hooks tests/test_literature_algorithm_lab.py::test_plan_candidate_sandbox_selects_level_one_for_checkpoint
```

Expected: fails because candidate compilation and sandbox planning functions are undefined.

- [ ] **Step 3: Implement candidate compilation and sandbox planning**

Append these functions and constants to `src/pde_agent/core/literature_algorithm_lab.py`:

```python
HOOK_TO_FAMILY = {
    "official_checkpoint_calibration": {
        "candidate_family": "official_checkpoint_family",
        "model_family": "checkpoint",
        "cli_command": "run-checkpoint-family-sandbox",
        "sandbox_level": 1,
    },
    "learned_residual_correction": {
        "candidate_family": "lightweight_residual_corrector",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "spectral_rollout_stabilization": {
        "candidate_family": "spectral_rollout_stabilization",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "physics_residual_regularization": {
        "candidate_family": "physics_residual_regularization",
        "model_family": "FNO",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 1,
    },
    "scheduled_rollout_training": {
        "candidate_family": "scheduled_rollout_training",
        "model_family": "FNO",
        "cli_command": "execute-fno-finetune-candidate",
        "sandbox_level": 1,
    },
}


def compile_algorithm_candidates(
    *,
    algorithm_cards_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    cards_report = _read_json(_require_file(Path(algorithm_cards_path), "algorithm cards"))
    if cards_report.get("state_hint") != ALGORITHM_CARDS_STATE_HINT:
        raise LiteratureAlgorithmLabError("algorithm cards report has invalid state_hint")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    candidates = []
    for card in cards_report.get("algorithm_cards", []):
        if not isinstance(card, dict):
            raise LiteratureAlgorithmLabError("algorithm card must be an object")
        hooks = _text_list(card.get("implementation_hooks"), "implementation_hooks")
        hook = hooks[0]
        family_spec = HOOK_TO_FAMILY.get(hook)
        if family_spec is None:
            candidates.append(_blocked_candidate(card=card, hook=hook))
            continue
        candidate_id = f"candidate_{_slug(card['algorithm_card_id'])}_{family_spec['candidate_family']}"
        candidates.append(
            {
                "algorithm_candidate_id": candidate_id,
                "parent_algorithm_card_id": card["algorithm_card_id"],
                "candidate_family": family_spec["candidate_family"],
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "model_family": family_spec["model_family"],
                "hypothesis": card["claim"],
                "mechanism": card["core_mechanism"],
                "executor_spec": _executor_spec(str(family_spec["cli_command"])),
                "parameter_grid": _default_parameter_grid(str(family_spec["candidate_family"])),
                "benchmark_plan": {
                    "judge": "hermes_local_judge",
                    "primary_metric": "task1_total_score_proxy",
                    "sandbox_level": family_spec["sandbox_level"],
                },
                "ablation_plan": {
                    "baseline": "persistence_or_current_best",
                    "contrast": "candidate_disabled_or_weight_zero",
                },
                "promotion_criteria": {
                    "min_score_delta": 0.01,
                    "max_inference_seconds": 120.0,
                    "requires_ablation": True,
                },
                "risk_controls": ["preserve first 10 context steps", "do not execute external repo code"],
                "source_grounding": {
                    "source_ids": card["source_ids"],
                    "algorithm_card_id": card["algorithm_card_id"],
                },
            }
        )
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CANDIDATES_STATE_HINT,
        "algorithm_candidates_path": str(output_file),
        "algorithm_candidate_count": len(candidates),
        "algorithm_candidates": candidates,
        "next_command": "plan-algorithm-candidate-sandbox",
    }
    _write_json(output_file, report)
    return report


def plan_candidate_sandbox(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(_require_file(Path(algorithm_candidates_path), "algorithm candidates"))
    if candidates_report.get("state_hint") != ALGORITHM_CANDIDATES_STATE_HINT:
        raise LiteratureAlgorithmLabError("algorithm candidates report has invalid state_hint")
    selected_id = _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    selected = None
    for candidate in candidates_report.get("algorithm_candidates", []):
        if isinstance(candidate, dict) and candidate.get("algorithm_candidate_id") == selected_id:
            selected = candidate
            break
    if selected is None:
        raise LiteratureAlgorithmLabError(f"algorithm candidate not found: {selected_id}")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    executor = selected.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    cli_command = _nonempty_text(executor.get("cli_command"), "cli_command")
    blocked = cli_command == "blocked_requires_executor_design"
    report = {
        "schema_version": 1,
        "state_hint": SANDBOX_PLAN_STATE_HINT,
        "sandbox_plan_path": str(output_file),
        "algorithm_candidate_id": selected_id,
        "candidate_family": selected.get("candidate_family"),
        "sandbox_level": _sandbox_level(str(selected.get("candidate_family"))),
        "executor": executor,
        "blocked": blocked,
        "reason_codes": ["requires_new_bounded_executor"] if blocked else [],
        "risk_controls": selected.get("risk_controls", []),
        "next_command": "design-bounded-executor" if blocked else cli_command,
    }
    _write_json(output_file, report)
    return report


def _blocked_candidate(*, card: dict[str, Any], hook: str) -> dict[str, Any]:
    return {
        "algorithm_candidate_id": f"candidate_{_slug(card['algorithm_card_id'])}_blocked",
        "parent_algorithm_card_id": card["algorithm_card_id"],
        "candidate_family": "unsupported",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": "hybrid",
        "hypothesis": card["claim"],
        "mechanism": card["core_mechanism"],
        "executor_spec": {
            "cli_command": "blocked_requires_executor_design",
            "unsupported_hook": hook,
            "required_options": [],
        },
        "parameter_grid": {},
        "benchmark_plan": {
            "judge": "hermes_local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 2,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "candidate_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": ["blocked until a bounded executor is designed"],
        "source_grounding": {
            "source_ids": card["source_ids"],
            "algorithm_card_id": card["algorithm_card_id"],
        },
    }


def _executor_spec(cli_command: str) -> dict[str, Any]:
    if cli_command == "run-checkpoint-family-sandbox":
        return {
            "cli_command": cli_command,
            "required_options": [
                "--checkpoint-path",
                "--task-id",
                "--test-path",
                "--target-path",
                "--output-dir",
            ],
        }
    if cli_command == "execute-fno-finetune-candidate":
        return {
            "cli_command": cli_command,
            "required_options": [
                "--candidate-plan-path",
                "--windows-path",
                "--checkpoint-path",
                "--max-train-steps",
            ],
        }
    return {
        "cli_command": cli_command,
        "required_options": [
            "--blueprint-path",
            "--test-path",
            "--target-path",
            "--output-dir",
        ],
    }


def _default_parameter_grid(candidate_family: str) -> dict[str, Any]:
    if candidate_family == "official_checkpoint_family":
        return {"calibration_weights": [0.0, 0.25, 0.5, 0.75, 1.0], "fno_finetune_steps": [0, 1]}
    if candidate_family == "lightweight_residual_corrector":
        return {"residual_shrinkage": [0.25, 0.5, 0.75], "horizon_decay": [40.0, 80.0]}
    if candidate_family == "spectral_rollout_stabilization":
        return {"spectral_strength": [0.25, 0.5, 1.0], "cutoff_ratio": [0.5, 0.75]}
    return {"candidate_enabled": [True]}


def _sandbox_level(candidate_family: str) -> int:
    if candidate_family in {"lightweight_residual_corrector", "spectral_rollout_stabilization"}:
        return 0
    if candidate_family in {"official_checkpoint_family", "physics_residual_regularization", "scheduled_rollout_training"}:
        return 1
    return 2
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py::test_compile_algorithm_candidates_maps_supported_hooks tests/test_literature_algorithm_lab.py::test_plan_candidate_sandbox_selects_level_one_for_checkpoint
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add -- src/pde_agent/core/literature_algorithm_lab.py tests/test_literature_algorithm_lab.py
git commit -m "feat: compile literature algorithm candidates"
```

---

### Task 4: Command Wrappers And CLI

**Files:**
- Create: `src/pde_agent/commands/literature_algorithm_lab.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `tests/test_literature_algorithm_lab.py`

- [ ] **Step 1: Write the failing CLI test**

Append this test:

```python
def test_literature_algorithm_lab_cli_source_records(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    raw_pdf = tmp_path / "paper.pdf"
    raw_pdf.write_bytes(b"paper")
    fixture = tmp_path / "sources.json"
    output_path = tmp_path / "source_records.json"
    fixture.write_text(
        json.dumps(
            {
                "query": "operator learning",
                "sources": [
                    {
                        "source_type": "arxiv_paper",
                        "title": "Operator Learning Paper",
                        "url": "https://arxiv.org/abs/2601.00002",
                        "local_raw_path": str(raw_pdf),
                        "license_note": "arXiv research use",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "build-literature-source-records",
            "--source-fixture-path",
            str(fixture),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "LITERATURE_SOURCE_RECORDS_BUILT"
    assert payload["artifacts"]["source_records_path"] == str(output_path)
    assert payload["metrics"]["source_count"] == 1
    assert payload["next_hint"] == "ingest-mineru-parse"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py::test_literature_algorithm_lab_cli_source_records
```

Expected: fails because `build-literature-source-records` is not registered.

- [ ] **Step 3: Add command wrappers**

Create `src/pde_agent/commands/literature_algorithm_lab.py`:

```python
from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.literature_algorithm_lab import (
    ALGORITHM_CANDIDATES_STATE_HINT,
    ALGORITHM_CARDS_STATE_HINT,
    PAPER_PARSE_STATE_HINT,
    SANDBOX_PLAN_STATE_HINT,
    SOURCE_RECORDS_STATE_HINT,
    build_literature_source_records,
    compile_algorithm_candidates,
    extract_algorithm_cards,
    ingest_mineru_parse,
    plan_candidate_sandbox,
)


def build_literature_source_records_command(
    *,
    source_fixture_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = build_literature_source_records(
        source_fixture_path=Path(source_fixture_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-literature-source-records",
        state_hint=SOURCE_RECORDS_STATE_HINT,
        artifacts={"source_records_path": report["source_records_path"]},
        metrics={"source_count": report["source_count"]},
        next_hint=report["next_command"],
    )


def ingest_mineru_parse_command(
    *,
    source_records_path: str,
    source_id: str,
    markdown_path: str,
    asset_dir: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = ingest_mineru_parse(
        source_records_path=Path(source_records_path),
        source_id=source_id,
        markdown_path=Path(markdown_path),
        asset_dir=Path(asset_dir),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="ingest-mineru-parse",
        state_hint=PAPER_PARSE_STATE_HINT,
        artifacts={"paper_parse_path": report["paper_parse_path"]},
        metrics={
            "formula_count": report["formula_count"],
            "figure_count": report["figure_count"],
            "quality": report["parse_quality"]["quality"],
        },
        next_hint=report["next_command"],
    )


def extract_algorithm_cards_command(
    *,
    paper_parse_paths: list[str],
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = extract_algorithm_cards(
        paper_parse_paths=[Path(path) for path in paper_parse_paths],
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="extract-algorithm-cards",
        state_hint=ALGORITHM_CARDS_STATE_HINT,
        artifacts={"algorithm_cards_path": report["algorithm_cards_path"]},
        metrics={"algorithm_card_count": report["algorithm_card_count"]},
        next_hint=report["next_command"],
    )


def compile_algorithm_candidates_command(
    *,
    algorithm_cards_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = compile_algorithm_candidates(
        algorithm_cards_path=Path(algorithm_cards_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="compile-algorithm-candidates",
        state_hint=ALGORITHM_CANDIDATES_STATE_HINT,
        artifacts={"algorithm_candidates_path": report["algorithm_candidates_path"]},
        metrics={"algorithm_candidate_count": report["algorithm_candidate_count"]},
        next_hint=report["next_command"],
    )


def plan_candidate_sandbox_command(
    *,
    algorithm_candidates_path: str,
    algorithm_candidate_id: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = plan_candidate_sandbox(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        algorithm_candidate_id=algorithm_candidate_id,
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="plan-algorithm-candidate-sandbox",
        state_hint=SANDBOX_PLAN_STATE_HINT,
        artifacts={"sandbox_plan_path": report["sandbox_plan_path"]},
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "sandbox_level": report["sandbox_level"],
            "blocked": report["blocked"],
        },
        next_hint=report["next_command"],
    )
```

- [ ] **Step 4: Register CLI commands**

Modify `src/pde_agent/cli.py`.

Add imports near other command imports:

```python
from pde_agent.commands.literature_algorithm_lab import (
    build_literature_source_records_command,
    compile_algorithm_candidates_command,
    extract_algorithm_cards_command,
    ingest_mineru_parse_command,
    plan_candidate_sandbox_command,
)
```

Add command functions near `run-hypothesis-lab`:

```python
@app.command("build-literature-source-records")
def build_literature_source_records_cmd(
    source_fixture_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_literature_source_records_command(
                source_fixture_path=source_fixture_path,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("ingest-mineru-parse")
def ingest_mineru_parse_cmd(
    source_records_path: str = typer.Option(...),
    source_id: str = typer.Option(...),
    markdown_path: str = typer.Option(...),
    asset_dir: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            ingest_mineru_parse_command(
                source_records_path=source_records_path,
                source_id=source_id,
                markdown_path=markdown_path,
                asset_dir=asset_dir,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("extract-algorithm-cards")
def extract_algorithm_cards_cmd(
    paper_parse_paths: list[str] = typer.Option(..., "--paper-parse-path"),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            extract_algorithm_cards_command(
                paper_parse_paths=paper_parse_paths,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("compile-algorithm-candidates")
def compile_algorithm_candidates_cmd(
    algorithm_cards_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            compile_algorithm_candidates_command(
                algorithm_cards_path=algorithm_cards_path,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("plan-algorithm-candidate-sandbox")
def plan_algorithm_candidate_sandbox_cmd(
    algorithm_candidates_path: str = typer.Option(...),
    algorithm_candidate_id: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            plan_candidate_sandbox_command(
                algorithm_candidates_path=algorithm_candidates_path,
                algorithm_candidate_id=algorithm_candidate_id,
                output_path=output_path,
                force=force,
            )
        )
    )
```

- [ ] **Step 5: Run CLI test to verify it passes**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py::test_literature_algorithm_lab_cli_source_records
```

Expected: `1 passed`.

- [ ] **Step 6: Run all literature lab tests**

Run:

```bash
uv run --frozen pytest -q tests/test_literature_algorithm_lab.py
```

Expected: all tests in `tests/test_literature_algorithm_lab.py` pass.

- [ ] **Step 7: Commit**

```bash
git add -- src/pde_agent/commands/literature_algorithm_lab.py src/pde_agent/cli.py tests/test_literature_algorithm_lab.py
git commit -m "feat: expose literature algorithm lab cli"
```

---

### Task 5: Verification And Documentation Sync

**Files:**
- Modify: `docs/superpowers/specs/2026-04-30-literature-to-algorithm-lab-design.md`
- Modify: `docs/superpowers/plans/2026-04-30-literature-algorithm-lab-phase1.md`

- [ ] **Step 1: Run full test suite**

Run:

```bash
uv run --frozen pytest -q
```

Expected: full suite passes.

- [ ] **Step 2: Add an implementation note to the spec**

Append this section to `docs/superpowers/specs/2026-04-30-literature-to-algorithm-lab-design.md`:

```markdown
## Phase 1 Implementation Note

Phase 1 is intentionally offline. It validates artifact contracts and candidate
handoff semantics with local fixture data before adding live source retrieval or
MinerU runtime execution. Network retrieval and parser invocation remain smoke
test concerns for later phases.
```

- [ ] **Step 3: Mark this plan complete**

In `docs/superpowers/plans/2026-04-30-literature-algorithm-lab-phase1.md`, change the unchecked boxes for completed tasks to checked boxes as the final worker action. Keep command outputs in the commit message or final response rather than pasting large logs into the plan.

- [ ] **Step 4: Commit verification docs**

```bash
git add -- docs/superpowers/specs/2026-04-30-literature-to-algorithm-lab-design.md docs/superpowers/plans/2026-04-30-literature-algorithm-lab-phase1.md
git commit -m "docs: record literature algorithm lab phase1 verification"
```

---

## Self-Review

Spec coverage:

- Source provenance is covered by Task 1.
- MinerU markdown ingestion and parse quality are covered by Task 2.
- Operational algorithm cards are covered by Task 2.
- Benchmark-ready algorithm candidates are covered by Task 3.
- Sandbox planning and blocked executor routing are covered by Task 3.
- CLI integration is covered by Task 4.
- Live OpenAlex/arXiv/GitHub retrieval, PDF download, and MinerU execution are outside Phase 1 by design and remain for later plans.

Completeness scan:

- The plan contains no unresolved sections.
- Each task includes concrete file paths, test commands, expected outcomes, and commit commands.

Type consistency:

- State hints match the constants defined in Task 1.
- Command names match the CLI functions in Task 4.
- Artifact field names are consistent across tests, core functions, and command wrappers.
