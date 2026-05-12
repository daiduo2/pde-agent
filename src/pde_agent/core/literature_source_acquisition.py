from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pde_agent.core.literature_algorithm_common import (
    LiteratureAlgorithmLabError,
    first_text,
    guard_output,
    nonempty_text,
    object_list,
    read_json,
    require_file,
    slug,
    source_candidate,
    utc_now,
    write_json,
    xml_text,
)


SOURCE_SEARCH_STATE_HINT = "EXTERNAL_SOURCE_SEARCH_NORMALIZED"
SOURCE_DOCUMENT_STATE_HINT = "SOURCE_DOCUMENT_DOWNLOADED"
MINERU_TASK_SUBMITTED_STATE_HINT = "MINERU_PARSE_TASK_SUBMITTED"
MINERU_TASK_POLLED_STATE_HINT = "MINERU_PARSE_TASK_POLLED"
MINERU_RESULT_DOWNLOADED_STATE_HINT = "MINERU_PARSE_RESULT_DOWNLOADED"
ARXIV_MINERU_PARSE_PIPELINE_STATE_HINT = "ARXIV_MINERU_PARSE_PIPELINE_RAN"

MINERU_AGENT_URL_ENDPOINT = "https://mineru.net/api/v1/agent/parse/url"
MINERU_AGENT_POLL_ENDPOINT = "https://mineru.net/api/v1/agent/parse"
MINERU_PRECISE_URL_ENDPOINT = "https://mineru.net/api/v4/extract/task"


def normalize_openalex_search_results(
    *,
    search_results_path: str | Path,
    query: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    payload = read_json(require_file(Path(search_results_path), "OpenAlex results"))
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    retrieval_query = nonempty_text(query, "query")
    candidates = []
    for item in object_list(payload.get("results"), "results"):
        title = nonempty_text(item.get("display_name"), "display_name")
        open_access = item.get("open_access") if isinstance(item.get("open_access"), dict) else {}
        primary = (
            item.get("primary_location")
            if isinstance(item.get("primary_location"), dict)
            else {}
        )
        url = first_text(
            primary.get("landing_page_url"),
            item.get("doi"),
            item.get("id"),
        )
        document_url = first_text(open_access.get("oa_url"), url)
        source_id = f"openalex_work_{slug(title)}"
        candidates.append(
            source_candidate(
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
    xml_file = require_file(Path(search_results_path), "arXiv results")
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    retrieval_query = nonempty_text(query, "query")
    root = ET.fromstring(xml_file.read_text(encoding="utf-8"))
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    candidates = []
    for entry in root.findall("atom:entry", namespace):
        title = nonempty_text(xml_text(entry, "atom:title", namespace), "title")
        entry_id = nonempty_text(xml_text(entry, "atom:id", namespace), "id")
        pdf_url = None
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        document_url = nonempty_text(pdf_url or entry_id.replace("/abs/", "/pdf/"), "document_url")
        arxiv_id = entry_id.rstrip("/").split("/")[-1]
        candidates.append(
            source_candidate(
                source_id=f"arxiv_paper_{slug(title)}",
                source_type="arxiv_paper",
                title=" ".join(title.split()),
                url=entry_id,
                document_url=document_url,
                retrieval_query=retrieval_query,
                discovery_provider="arxiv",
                license_note="arXiv paper; check entry license before reuse",
                external_ids={
                    "arxiv_id": arxiv_id,
                    "published": xml_text(entry, "atom:published", namespace),
                    "updated": xml_text(entry, "atom:updated", namespace),
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
    payload = read_json(require_file(Path(search_results_path), "GitHub results"))
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    retrieval_query = nonempty_text(query, "query")
    candidates = []
    for item in object_list(payload.get("items"), "items"):
        full_name = nonempty_text(item.get("full_name"), "full_name")
        url = nonempty_text(item.get("html_url"), "html_url")
        candidates.append(
            source_candidate(
                source_id=f"github_repo_{slug(full_name)}",
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


def run_arxiv_mineru_parse_pipeline(
    *,
    query: str,
    output_dir: str | Path,
    token: str | None,
    mode: str,
    model_version: str,
    max_results: int,
    selected_index: int,
    max_polls: int,
    poll_interval_seconds: float,
    downloader: Any | None = None,
    mineru_client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    retrieval_query = nonempty_text(query, "query")
    if max_results <= 0:
        raise LiteratureAlgorithmLabError("max_results must be positive")
    if selected_index < -1:
        raise LiteratureAlgorithmLabError(
            "selected_index must be -1 for relevance selection or non-negative"
        )
    if max_polls <= 0:
        raise LiteratureAlgorithmLabError("max_polls must be positive")
    if poll_interval_seconds < 0:
        raise LiteratureAlgorithmLabError("poll_interval_seconds must be non-negative")

    root = Path(output_dir)
    raw_dir = root / "raw"
    cache_dir = root / "cache"
    mineru_dir = root / "mineru"
    mineru_result_dir = mineru_dir / "result"
    report_path = root / "arxiv_mineru_parse_pipeline.json"
    guard_output(report_path, force=force)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mineru_dir.mkdir(parents=True, exist_ok=True)

    fetch = downloader or _download_bytes
    arxiv_url = _arxiv_api_query_url(retrieval_query, max_results=max_results)
    arxiv_xml_path = raw_dir / "arxiv_search.xml"
    guard_output(arxiv_xml_path, force=force)
    arxiv_xml = fetch(arxiv_url)
    if not isinstance(arxiv_xml, bytes) or not arxiv_xml:
        raise LiteratureAlgorithmLabError("arXiv response must be non-empty bytes")
    arxiv_xml_path.write_bytes(arxiv_xml)

    candidates = normalize_arxiv_search_results(
        search_results_path=arxiv_xml_path,
        query=retrieval_query,
        output_path=root / "source_candidates.json",
        force=force,
    )
    source_candidates = object_list(candidates.get("source_candidates"), "source_candidates")
    effective_selected_index = (
        _select_relevant_source_index(source_candidates, retrieval_query)
        if selected_index == -1
        else selected_index
    )
    if effective_selected_index >= len(source_candidates):
        raise LiteratureAlgorithmLabError(
            f"selected_index out of range for {len(source_candidates)} candidates"
        )
    selected = source_candidates[effective_selected_index]
    selected_source_id = nonempty_text(selected.get("source_id"), "source_id")

    downloaded = download_source_document(
        source_candidates_path=candidates["source_candidates_path"],
        source_id=selected_source_id,
        output_path=root / "downloaded_source.json",
        cache_dir=cache_dir,
        downloader=downloader,
        force=force,
    )
    mineru_task = submit_mineru_parse_task(
        downloaded_source_path=downloaded["downloaded_source_path"],
        output_path=mineru_dir / "task.json",
        mode=mode,
        token=token,
        model_version=model_version,
        client=mineru_client,
        force=force,
    )

    latest_poll: dict[str, Any] | None = None
    for poll_index in range(1, max_polls + 1):
        latest_poll = poll_mineru_parse_task(
            mineru_task_path=mineru_task["mineru_task_path"],
            output_path=mineru_dir / f"poll_{poll_index:03d}.json",
            token=token,
            client=mineru_client,
            force=force,
        )
        if latest_poll.get("state") == "done":
            break
        if poll_index < max_polls and poll_interval_seconds:
            time.sleep(poll_interval_seconds)

    if latest_poll is None:
        raise LiteratureAlgorithmLabError("MinerU polling did not run")

    result: dict[str, Any] | None = None
    local_markdown_path = None
    if latest_poll.get("state") == "done":
        result = download_mineru_parse_result(
            mineru_poll_path=latest_poll["mineru_poll_path"],
            output_path=mineru_dir / "result.json",
            output_dir=mineru_result_dir,
            client=mineru_client,
            force=force,
        )
        local_markdown_path = _materialize_mineru_markdown_from_result(
            result_report=result,
            output_dir=mineru_result_dir,
        )

    artifacts = {
        "arxiv_search_url": arxiv_url,
        "arxiv_search_xml_path": str(arxiv_xml_path),
        "source_candidates_path": candidates["source_candidates_path"],
        "downloaded_source_path": downloaded["downloaded_source_path"],
        "source_records_path": downloaded["downloaded_source_path"],
        "mineru_task_path": mineru_task["mineru_task_path"],
        "latest_mineru_poll_path": latest_poll["mineru_poll_path"],
        "mineru_result_path": result["mineru_result_path"] if result else None,
        "local_markdown_path": local_markdown_path,
        "local_zip_path": result.get("local_zip_path") if result else None,
    }
    next_command = "ingest-mineru-parse" if local_markdown_path else "poll-mineru-parse"
    report = {
        "schema_version": 1,
        "state_hint": ARXIV_MINERU_PARSE_PIPELINE_STATE_HINT,
        "query": retrieval_query,
        "selected_index": selected_index,
        "effective_selected_index": effective_selected_index,
        "selection_policy": "query_relevance" if selected_index == -1 else "explicit_index",
        "selected_source_id": selected_source_id,
        "selected_title": selected.get("title"),
        "mineru_state": latest_poll.get("state"),
        "mineru_mode": mineru_task.get("mineru_mode"),
        "artifacts": artifacts,
        "suggested_ingest_args": {
            "source_records_path": downloaded["downloaded_source_path"],
            "source_id": selected_source_id,
            "markdown_path": local_markdown_path,
            "asset_dir": str(mineru_result_dir),
        },
        "next_command": next_command,
    }
    write_json(report_path, report)
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
    candidates_report = read_json(
        require_file(Path(source_candidates_path), "source candidates")
    )
    if candidates_report.get("state_hint") != SOURCE_SEARCH_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "source candidates report has invalid state_hint"
        )
    selected_id = nonempty_text(source_id, "source_id")
    candidate = _find_by_id(
        candidates_report.get("source_candidates"),
        selected_id,
        "source_id",
    )
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    download_url = nonempty_text(candidate.get("document_url"), "document_url")
    extension = _extension_for_url(download_url)
    local_raw_path = Path(cache_dir) / f"{slug(selected_id)}{extension}"
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
        "retrieved_at": utc_now(),
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
    write_json(output_file, report)
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
    downloaded = read_json(
        require_file(Path(downloaded_source_path), "downloaded source")
    )
    if downloaded.get("state_hint") != SOURCE_DOCUMENT_STATE_HINT:
        raise LiteratureAlgorithmLabError("downloaded source has invalid state_hint")
    source_record = downloaded.get("source_record")
    if not isinstance(source_record, dict):
        raise LiteratureAlgorithmLabError("downloaded source missing source_record")
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    mineru_mode = _mineru_mode(mode)
    http = client or MinerUHttpClient()
    download_url = nonempty_text(source_record.get("download_url"), "download_url")
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
            "model_version": nonempty_text(model_version, "model_version"),
            "enable_formula": True,
            "enable_table": True,
            "is_ocr": False,
            "data_id": source_record["source_id"],
        }
        auth_token = nonempty_text(token, "MINERU_API_TOKEN")
    response = http.post_json(endpoint, payload, auth_token)
    data = _mineru_data(response)
    task_id = nonempty_text(data.get("task_id"), "task_id")
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
    write_json(output_file, report)
    return report


def poll_mineru_parse_task(
    *,
    mineru_task_path: str | Path,
    output_path: str | Path,
    token: str | None,
    client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    task = read_json(require_file(Path(mineru_task_path), "MinerU task"))
    if task.get("state_hint") != MINERU_TASK_SUBMITTED_STATE_HINT:
        raise LiteratureAlgorithmLabError("MinerU task has invalid state_hint")
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    http = client or MinerUHttpClient()
    task_id = nonempty_text(task.get("task_id"), "task_id")
    mode = _mineru_mode(task.get("mineru_mode"))
    endpoint = (
        f"{MINERU_AGENT_POLL_ENDPOINT}/{task_id}"
        if mode == "agent_url"
        else f"{MINERU_PRECISE_URL_ENDPOINT}/{task_id}"
    )
    response = http.get_json(endpoint, None if mode == "agent_url" else token)
    data = _mineru_data(response)
    state = nonempty_text(data.get("state"), "state")
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
    write_json(output_file, report)
    return report


def download_mineru_parse_result(
    *,
    mineru_poll_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    poll = read_json(require_file(Path(mineru_poll_path), "MinerU poll"))
    if poll.get("state_hint") != MINERU_TASK_POLLED_STATE_HINT:
        raise LiteratureAlgorithmLabError("MinerU poll has invalid state_hint")
    if poll.get("state") != "done":
        raise LiteratureAlgorithmLabError("MinerU task is not done")
    output_file = Path(output_path)
    guard_output(output_file, force=force)
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
    write_json(output_file, report)
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


def _select_relevant_source_index(
    source_candidates: list[dict[str, Any]],
    query: str,
) -> int:
    if not source_candidates:
        raise LiteratureAlgorithmLabError("source_candidates must be non-empty")
    query_terms = _relevance_terms(query)
    scored = [
        (
            _source_candidate_relevance_score(candidate, query_terms=query_terms),
            index,
        )
        for index, candidate in enumerate(source_candidates)
    ]
    return max(scored, key=lambda item: (item[0], -item[1]))[1]


def _relevance_terms(text: str) -> list[str]:
    terms = []
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
        if len(term) < 3 or term in {"for", "the", "and", "with", "all", "partial"}:
            continue
        terms.append(term)
    aliases = {
        "fno": ["fourier", "neural", "operator"],
        "burgers": ["burgers"],
    }
    expanded = list(terms)
    for term in terms:
        expanded.extend(aliases.get(term, []))
    return sorted(set(expanded))


def _source_candidate_relevance_score(
    candidate: dict[str, Any],
    *,
    query_terms: list[str],
) -> float:
    title = str(candidate.get("title", "")).lower()
    source_id = str(candidate.get("source_id", "")).lower()
    external_ids = candidate.get("external_ids")
    arxiv_id = ""
    if isinstance(external_ids, dict):
        arxiv_id = str(external_ids.get("arxiv_id", "")).lower()
    haystack = f"{title} {source_id} {arxiv_id}"
    score = 0.0
    for term in query_terms:
        if term in title:
            score += 3.0
        elif term in haystack:
            score += 1.0
    phrase_bonus = {
        "fourier neural operator": 25.0,
        "neural operator": 10.0,
        "parametric partial differential equations": 8.0,
        "burgers": 4.0,
    }
    for phrase, bonus in phrase_bonus.items():
        if phrase in title:
            score += bonus
    if "stability of solutions" in title and "neural operator" not in title:
        score -= 8.0
    return score


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
    write_json(output_file, report)
    return report


def _mineru_mode(value: Any) -> str:
    mode = nonempty_text(value, "mineru_mode")
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
    for item in object_list(value, "items"):
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


def _arxiv_api_query_url(query: str, *, max_results: int) -> str:
    return "https://export.arxiv.org/api/query?" + urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        }
    )


def _download_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "pde-agent-literature-algorithm-lab/0.1"},
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def _materialize_mineru_markdown_from_result(
    *,
    result_report: dict[str, Any],
    output_dir: str | Path,
) -> str | None:
    markdown_path = result_report.get("local_markdown_path")
    if isinstance(markdown_path, str) and markdown_path.strip():
        return str(require_file(Path(markdown_path), "MinerU markdown"))
    zip_path = result_report.get("local_zip_path")
    if not isinstance(zip_path, str) or not zip_path.strip():
        return None
    archive_path = require_file(Path(zip_path), "MinerU result zip")
    root = Path(output_dir) / "extracted"
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        selected_name = "full.md" if "full.md" in names else None
        if selected_name is None:
            markdown_names = [name for name in names if name.lower().endswith(".md")]
            if not markdown_names:
                return None
            selected_name = sorted(markdown_names, key=lambda name: (len(name), name))[0]
        local_markdown_path = root / Path(selected_name).name
        local_markdown_path.write_bytes(archive.read(selected_name))
        return str(local_markdown_path)
