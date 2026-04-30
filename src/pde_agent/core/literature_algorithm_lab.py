"""Offline artifact backbone for PDE literature-to-algorithm candidates."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


SOURCE_RECORDS_STATE_HINT = "LITERATURE_SOURCE_RECORDS_BUILT"
SOURCE_SEARCH_STATE_HINT = "EXTERNAL_SOURCE_SEARCH_NORMALIZED"
SOURCE_DOCUMENT_STATE_HINT = "SOURCE_DOCUMENT_DOWNLOADED"
MINERU_TASK_SUBMITTED_STATE_HINT = "MINERU_PARSE_TASK_SUBMITTED"
MINERU_TASK_POLLED_STATE_HINT = "MINERU_PARSE_TASK_POLLED"
MINERU_RESULT_DOWNLOADED_STATE_HINT = "MINERU_PARSE_RESULT_DOWNLOADED"
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

MINERU_AGENT_URL_ENDPOINT = "https://mineru.net/api/v1/agent/parse/url"
MINERU_AGENT_POLL_ENDPOINT = "https://mineru.net/api/v1/agent/parse"
MINERU_PRECISE_URL_ENDPOINT = "https://mineru.net/api/v4/extract/task"

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


class LiteratureAlgorithmLabError(ValueError):
    """Raised when literature algorithm lab artifacts are invalid."""


def normalize_openalex_search_results(
    *,
    search_results_path: str | Path,
    query: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    payload = _read_json(_require_file(Path(search_results_path), "OpenAlex results"))
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    retrieval_query = _nonempty_text(query, "query")
    candidates = []
    for item in _object_list(payload.get("results"), "results"):
        title = _nonempty_text(item.get("display_name"), "display_name")
        open_access = item.get("open_access") if isinstance(item.get("open_access"), dict) else {}
        primary = (
            item.get("primary_location")
            if isinstance(item.get("primary_location"), dict)
            else {}
        )
        url = _first_text(
            primary.get("landing_page_url"),
            item.get("doi"),
            item.get("id"),
        )
        document_url = _first_text(open_access.get("oa_url"), url)
        source_id = f"openalex_work_{_slug(title)}"
        candidates.append(
            _source_candidate(
                source_id=source_id,
                source_type="openalex_work",
                title=title,
                url=url,
                document_url=document_url,
                retrieval_query=retrieval_query,
                discovery_provider="openalex",
                license_note="OpenAlex metadata; document license must be checked at source",
                external_ids={
                    "openalex": item.get("id"),
                    "doi": item.get("doi"),
                    "publication_year": item.get("publication_year"),
                    "cited_by_count": item.get("cited_by_count"),
                },
            )
        )
    return _write_source_search_report(
        provider="openalex",
        retrieval_query=retrieval_query,
        candidates=candidates,
        output_file=output_file,
    )


def normalize_arxiv_search_results(
    *,
    search_results_path: str | Path,
    query: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    xml_file = _require_file(Path(search_results_path), "arXiv results")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    retrieval_query = _nonempty_text(query, "query")
    root = ET.fromstring(xml_file.read_text(encoding="utf-8"))
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    candidates = []
    for entry in root.findall("atom:entry", namespace):
        title = _nonempty_text(_xml_text(entry, "atom:title", namespace), "title")
        entry_id = _nonempty_text(_xml_text(entry, "atom:id", namespace), "id")
        pdf_url = None
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        document_url = _nonempty_text(pdf_url or entry_id.replace("/abs/", "/pdf/"), "document_url")
        arxiv_id = entry_id.rstrip("/").split("/")[-1]
        candidates.append(
            _source_candidate(
                source_id=f"arxiv_paper_{_slug(title)}",
                source_type="arxiv_paper",
                title=" ".join(title.split()),
                url=entry_id,
                document_url=document_url,
                retrieval_query=retrieval_query,
                discovery_provider="arxiv",
                license_note="arXiv paper; check entry license before reuse",
                external_ids={
                    "arxiv_id": arxiv_id,
                    "published": _xml_text(entry, "atom:published", namespace),
                    "updated": _xml_text(entry, "atom:updated", namespace),
                },
            )
        )
    return _write_source_search_report(
        provider="arxiv",
        retrieval_query=retrieval_query,
        candidates=candidates,
        output_file=output_file,
    )


def normalize_github_search_results(
    *,
    search_results_path: str | Path,
    query: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    payload = _read_json(_require_file(Path(search_results_path), "GitHub results"))
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    retrieval_query = _nonempty_text(query, "query")
    candidates = []
    for item in _object_list(payload.get("items"), "items"):
        full_name = _nonempty_text(item.get("full_name"), "full_name")
        url = _nonempty_text(item.get("html_url"), "html_url")
        candidates.append(
            _source_candidate(
                source_id=f"github_repo_{_slug(full_name)}",
                source_type="github_repo",
                title=full_name,
                url=url,
                document_url=url,
                retrieval_query=retrieval_query,
                discovery_provider="github",
                license_note="GitHub repository metadata; inspect repository license before reuse",
                external_ids={
                    "full_name": full_name,
                    "default_branch": item.get("default_branch"),
                    "stargazers_count": item.get("stargazers_count"),
                },
            )
        )
    return _write_source_search_report(
        provider="github",
        retrieval_query=retrieval_query,
        candidates=candidates,
        output_file=output_file,
    )


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


def download_source_document(
    *,
    source_candidates_path: str | Path,
    source_id: str,
    output_path: str | Path,
    cache_dir: str | Path,
    downloader: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(source_candidates_path), "source candidates")
    )
    if candidates_report.get("state_hint") != SOURCE_SEARCH_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "source candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(source_id, "source_id")
    candidate = _find_by_id(
        candidates_report.get("source_candidates"),
        selected_id,
        "source_id",
    )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    download_url = _nonempty_text(candidate.get("document_url"), "document_url")
    extension = _extension_for_url(download_url)
    local_raw_path = Path(cache_dir) / f"{_slug(selected_id)}{extension}"
    if local_raw_path.exists() and not force:
        content = local_raw_path.read_bytes()
    else:
        fetch = downloader or _download_bytes
        content = fetch(download_url)
        if not isinstance(content, bytes) or not content:
            raise LiteratureAlgorithmLabError("downloaded content must be non-empty bytes")
        local_raw_path.parent.mkdir(parents=True, exist_ok=True)
        local_raw_path.write_bytes(content)

    source_record = {
        "source_id": candidate["source_id"],
        "source_type": candidate["source_type"],
        "title": candidate["title"],
        "url": candidate["url"],
        "retrieved_at": _utc_now(),
        "license_note": candidate["license_note"],
        "local_raw_path": str(local_raw_path),
        "content_hash": hashlib.sha256(content).hexdigest(),
        "related_source_ids": [],
        "retrieval_query": candidate["retrieval_query"],
        "download_url": download_url,
        "discovery_provider": candidate["discovery_provider"],
        "external_ids": candidate.get("external_ids", {}),
    }
    report = {
        "schema_version": 1,
        "state_hint": SOURCE_DOCUMENT_STATE_HINT,
        "downloaded_source_path": str(output_file),
        "source_record": source_record,
        "next_command": "submit-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def submit_mineru_parse_task(
    *,
    downloaded_source_path: str | Path,
    output_path: str | Path,
    mode: str,
    token: str | None,
    model_version: str,
    client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    downloaded = _read_json(
        _require_file(Path(downloaded_source_path), "downloaded source")
    )
    if downloaded.get("state_hint") != SOURCE_DOCUMENT_STATE_HINT:
        raise LiteratureAlgorithmLabError("downloaded source has invalid state_hint")
    source_record = downloaded.get("source_record")
    if not isinstance(source_record, dict):
        raise LiteratureAlgorithmLabError("downloaded source missing source_record")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    mineru_mode = _mineru_mode(mode)
    http = client or MinerUHttpClient()
    download_url = _nonempty_text(source_record.get("download_url"), "download_url")
    if mineru_mode == "agent_url":
        endpoint = MINERU_AGENT_URL_ENDPOINT
        payload = {
            "url": download_url,
            "language": "en",
            "enable_table": True,
            "is_ocr": False,
            "enable_formula": True,
        }
        auth_token = None
    else:
        endpoint = MINERU_PRECISE_URL_ENDPOINT
        payload = {
            "url": download_url,
            "model_version": _nonempty_text(model_version, "model_version"),
            "enable_formula": True,
            "enable_table": True,
            "is_ocr": False,
            "data_id": source_record["source_id"],
        }
        auth_token = _nonempty_text(token, "MINERU_API_TOKEN")
    response = http.post_json(endpoint, payload, auth_token)
    data = _mineru_data(response)
    task_id = _nonempty_text(data.get("task_id"), "task_id")
    report = {
        "schema_version": 1,
        "state_hint": MINERU_TASK_SUBMITTED_STATE_HINT,
        "mineru_task_path": str(output_file),
        "source_id": source_record["source_id"],
        "source_title": source_record.get("title"),
        "source_content_hash": source_record.get("content_hash"),
        "mineru_mode": mineru_mode,
        "submit_endpoint": endpoint,
        "model_version": model_version,
        "task_id": task_id,
        "request_payload": payload,
        "token_source": "MINERU_API_TOKEN" if auth_token else None,
        "next_command": "poll-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def poll_mineru_parse_task(
    *,
    mineru_task_path: str | Path,
    output_path: str | Path,
    token: str | None,
    client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    task = _read_json(_require_file(Path(mineru_task_path), "MinerU task"))
    if task.get("state_hint") != MINERU_TASK_SUBMITTED_STATE_HINT:
        raise LiteratureAlgorithmLabError("MinerU task has invalid state_hint")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    http = client or MinerUHttpClient()
    task_id = _nonempty_text(task.get("task_id"), "task_id")
    mode = _mineru_mode(task.get("mineru_mode"))
    endpoint = (
        f"{MINERU_AGENT_POLL_ENDPOINT}/{task_id}"
        if mode == "agent_url"
        else f"{MINERU_PRECISE_URL_ENDPOINT}/{task_id}"
    )
    response = http.get_json(endpoint, None if mode == "agent_url" else token)
    data = _mineru_data(response)
    state = _nonempty_text(data.get("state"), "state")
    report = {
        "schema_version": 1,
        "state_hint": MINERU_TASK_POLLED_STATE_HINT,
        "mineru_poll_path": str(output_file),
        "mineru_task_path": str(Path(mineru_task_path)),
        "source_id": task.get("source_id"),
        "mineru_mode": mode,
        "task_id": task_id,
        "poll_endpoint": endpoint,
        "state": state,
        "markdown_url": data.get("markdown_url"),
        "full_zip_url": data.get("full_zip_url"),
        "err_msg": data.get("err_msg"),
        "err_code": data.get("err_code"),
        "next_command": "download-mineru-result"
        if state == "done"
        else "poll-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def download_mineru_parse_result(
    *,
    mineru_poll_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    poll = _read_json(_require_file(Path(mineru_poll_path), "MinerU poll"))
    if poll.get("state_hint") != MINERU_TASK_POLLED_STATE_HINT:
        raise LiteratureAlgorithmLabError("MinerU poll has invalid state_hint")
    if poll.get("state") != "done":
        raise LiteratureAlgorithmLabError("MinerU task is not done")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    http = client or MinerUHttpClient()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    markdown_url = poll.get("markdown_url")
    full_zip_url = poll.get("full_zip_url")
    local_markdown_path = None
    local_zip_path = None
    if isinstance(markdown_url, str) and markdown_url.strip():
        local_markdown_path = root / "full.md"
        local_markdown_path.write_bytes(http.download_bytes(markdown_url))
    if isinstance(full_zip_url, str) and full_zip_url.strip():
        local_zip_path = root / "mineru_result.zip"
        local_zip_path.write_bytes(http.download_bytes(full_zip_url))
    if local_markdown_path is None and local_zip_path is None:
        raise LiteratureAlgorithmLabError(
            "MinerU result has neither markdown_url nor full_zip_url"
        )
    report = {
        "schema_version": 1,
        "state_hint": MINERU_RESULT_DOWNLOADED_STATE_HINT,
        "mineru_result_path": str(output_file),
        "mineru_poll_path": str(Path(mineru_poll_path)),
        "source_id": poll.get("source_id"),
        "task_id": poll.get("task_id"),
        "local_markdown_path": str(local_markdown_path)
        if local_markdown_path is not None
        else None,
        "local_zip_path": str(local_zip_path) if local_zip_path is not None else None,
        "markdown_url": markdown_url,
        "full_zip_url": full_zip_url,
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


def compile_algorithm_candidates(
    *,
    algorithm_cards_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    cards_report = _read_json(
        _require_file(Path(algorithm_cards_path), "algorithm cards")
    )
    if cards_report.get("state_hint") != ALGORITHM_CARDS_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm cards report has invalid state_hint"
        )
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
        candidate_id = (
            f"candidate_{_slug(card['algorithm_card_id'])}_"
            f"{family_spec['candidate_family']}"
        )
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
                "parameter_grid": _default_parameter_grid(
                    str(family_spec["candidate_family"])
                ),
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
                "risk_controls": [
                    "preserve first 10 context steps",
                    "do not execute external repo code",
                ],
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
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") != ALGORITHM_CANDIDATES_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    selected = None
    for candidate in candidates_report.get("algorithm_candidates", []):
        if (
            isinstance(candidate, dict)
            and candidate.get("algorithm_candidate_id") == selected_id
        ):
            selected = candidate
            break
    if selected is None:
        raise LiteratureAlgorithmLabError(
            f"algorithm candidate not found: {selected_id}"
        )
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
        return {
            "calibration_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
            "fno_finetune_steps": [0, 1],
        }
    if candidate_family == "lightweight_residual_corrector":
        return {"residual_shrinkage": [0.25, 0.5, 0.75], "horizon_decay": [40.0, 80.0]}
    if candidate_family == "spectral_rollout_stabilization":
        return {"spectral_strength": [0.25, 0.5, 1.0], "cutoff_ratio": [0.5, 0.75]}
    return {"candidate_enabled": [True]}


def _sandbox_level(candidate_family: str) -> int:
    if candidate_family in {
        "lightweight_residual_corrector",
        "spectral_rollout_stabilization",
    }:
        return 0
    if candidate_family in {
        "official_checkpoint_family",
        "physics_residual_regularization",
        "scheduled_rollout_training",
    }:
        return 1
    return 2


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


def _write_source_search_report(
    *,
    provider: str,
    retrieval_query: str,
    candidates: list[dict[str, Any]],
    output_file: Path,
) -> dict[str, Any]:
    if not candidates:
        raise LiteratureAlgorithmLabError("source search produced no candidates")
    report = {
        "schema_version": 1,
        "state_hint": SOURCE_SEARCH_STATE_HINT,
        "provider": provider,
        "retrieval_query": retrieval_query,
        "source_candidate_count": len(candidates),
        "source_candidates": candidates,
        "source_candidates_path": str(output_file),
        "next_command": "download-source-document",
    }
    _write_json(output_file, report)
    return report


class MinerUHttpClient:
    def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        token: str | None,
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_json(self, endpoint: str, token: str | None) -> dict[str, Any]:
        headers = {"Accept": "*/*"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(endpoint, headers=headers, method="GET")
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def download_bytes(self, url: str) -> bytes:
        return _download_bytes(url)


def _mineru_mode(value: Any) -> str:
    mode = _nonempty_text(value, "mineru_mode")
    if mode not in {"agent_url", "precise_url"}:
        raise LiteratureAlgorithmLabError(f"unsupported MinerU mode: {mode}")
    return mode


def _mineru_data(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("code") != 0:
        raise LiteratureAlgorithmLabError(f"MinerU API error: {response.get('msg')}")
    data = response.get("data")
    if not isinstance(data, dict):
        raise LiteratureAlgorithmLabError("MinerU response missing data object")
    return data


def _find_by_id(value: Any, selected_id: str, id_key: str) -> dict[str, Any]:
    for item in _object_list(value, "items"):
        if item.get(id_key) == selected_id:
            return item
    raise LiteratureAlgorithmLabError(f"{id_key} not found: {selected_id}")


def _extension_for_url(url: str) -> str:
    lowered = url.lower().split("?", 1)[0].rstrip("/")
    for extension in [".pdf", ".zip", ".md", ".html", ".docx", ".doc", ".pptx", ".ppt"]:
        if lowered.endswith(extension):
            return extension
    if "/pdf/" in lowered or "arxiv.org/pdf" in lowered:
        return ".pdf"
    return ".bin"


def _download_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "pde-agent-literature-algorithm-lab/0.1"},
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def _source_candidate(
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
        "external_ids": {
            key: value for key, value in external_ids.items() if value is not None
        },
    }


def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError(f"{label} items must be objects")
        result.append(item)
    return result


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise LiteratureAlgorithmLabError("expected at least one non-empty text value")


def _xml_text(element: ET.Element, path: str, namespace: dict[str, str]) -> str | None:
    child = element.find(path, namespace)
    if child is None or child.text is None:
        return None
    return child.text.strip()


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
