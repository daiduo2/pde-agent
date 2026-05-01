from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest


def _write_hdf5(path: Path, data: object, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def test_normalize_openalex_search_results_builds_source_candidates(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        normalize_openalex_search_results,
    )

    fixture = tmp_path / "openalex.json"
    fixture.write_text(
        json.dumps(
            {
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "doi": "https://doi.org/10.48550/arXiv.2010.08895",
                        "display_name": "Fourier Neural Operator for Parametric PDEs",
                        "publication_year": 2021,
                        "cited_by_count": 1000,
                        "open_access": {
                            "is_oa": True,
                            "oa_url": "https://arxiv.org/pdf/2010.08895",
                        },
                        "primary_location": {
                            "landing_page_url": "https://arxiv.org/abs/2010.08895"
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = normalize_openalex_search_results(
        search_results_path=fixture,
        query="Fourier neural operator Burgers",
        output_path=tmp_path / "source_candidates.json",
        force=False,
    )

    assert report["state_hint"] == "EXTERNAL_SOURCE_SEARCH_NORMALIZED"
    assert report["provider"] == "openalex"
    candidate = report["source_candidates"][0]
    assert candidate["source_type"] == "openalex_work"
    assert candidate["source_id"].startswith("openalex_work_fourier")
    assert candidate["document_url"] == "https://arxiv.org/pdf/2010.08895"
    assert candidate["external_ids"]["openalex"] == "https://openalex.org/W123"
    assert report["next_command"] == "download-source-document"


def test_normalize_arxiv_search_results_extracts_pdf_url(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import normalize_arxiv_search_results

    fixture = tmp_path / "arxiv.xml"
    fixture.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2010.08895v3</id>
    <updated>2021-01-01T00:00:00Z</updated>
    <published>2020-10-17T00:00:00Z</published>
    <title>Fourier Neural Operator for Parametric Partial Differential Equations</title>
    <summary>We propose Fourier neural operators.</summary>
    <link href="http://arxiv.org/abs/2010.08895v3" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2010.08895v3" rel="related" type="application/pdf"/>
  </entry>
</feed>
""",
        encoding="utf-8",
    )

    report = normalize_arxiv_search_results(
        search_results_path=fixture,
        query="Fourier neural operator",
        output_path=tmp_path / "source_candidates.json",
        force=False,
    )

    candidate = report["source_candidates"][0]
    assert report["provider"] == "arxiv"
    assert candidate["source_type"] == "arxiv_paper"
    assert candidate["document_url"] == "http://arxiv.org/pdf/2010.08895v3"
    assert candidate["external_ids"]["arxiv_id"] == "2010.08895v3"


def test_normalize_github_search_results_builds_repo_candidates(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import normalize_github_search_results

    fixture = tmp_path / "github.json"
    fixture.write_text(
        json.dumps(
            {
                "total_count": 1,
                "items": [
                    {
                        "full_name": "neuraloperator/neuraloperator",
                        "html_url": "https://github.com/neuraloperator/neuraloperator",
                        "description": "Learning neural operators in PyTorch.",
                        "stargazers_count": 1234,
                        "default_branch": "main",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = normalize_github_search_results(
        search_results_path=fixture,
        query="FNO PDEBench checkpoint",
        output_path=tmp_path / "source_candidates.json",
        force=False,
    )

    candidate = report["source_candidates"][0]
    assert report["provider"] == "github"
    assert candidate["source_type"] == "github_repo"
    assert candidate["url"] == "https://github.com/neuraloperator/neuraloperator"
    assert candidate["document_url"] == "https://github.com/neuraloperator/neuraloperator"
    assert candidate["external_ids"]["full_name"] == "neuraloperator/neuraloperator"


def test_download_source_document_uses_cache_and_hashes_content(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import download_source_document

    candidates_path = tmp_path / "source_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXTERNAL_SOURCE_SEARCH_NORMALIZED",
                "source_candidates": [
                    {
                        "source_id": "arxiv_paper_fno",
                        "source_type": "arxiv_paper",
                        "title": "Fourier Neural Operator",
                        "url": "https://arxiv.org/abs/2010.08895",
                        "document_url": "https://arxiv.org/pdf/2010.08895",
                        "retrieval_query": "FNO",
                        "discovery_provider": "arxiv",
                        "license_note": "arXiv paper",
                        "external_ids": {"arxiv_id": "2010.08895"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_downloader(url: str) -> bytes:
        calls.append(url)
        return b"%PDF fixture"

    report = download_source_document(
        source_candidates_path=candidates_path,
        source_id="arxiv_paper_fno",
        output_path=tmp_path / "downloaded_source.json",
        cache_dir=tmp_path / "cache",
        downloader=fake_downloader,
        force=False,
    )

    assert calls == ["https://arxiv.org/pdf/2010.08895"]
    assert report["state_hint"] == "SOURCE_DOCUMENT_DOWNLOADED"
    assert Path(report["source_record"]["local_raw_path"]).exists()
    assert report["source_record"]["content_hash"]
    assert report["source_record"]["download_url"] == "https://arxiv.org/pdf/2010.08895"
    assert report["next_command"] == "submit-mineru-parse"


def test_submit_mineru_precise_url_redacts_token(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import submit_mineru_parse_task

    source_path = tmp_path / "downloaded_source.json"
    source_path.write_text(
        json.dumps(
            {
                "state_hint": "SOURCE_DOCUMENT_DOWNLOADED",
                "source_record": {
                    "source_id": "arxiv_paper_fno",
                    "title": "FNO",
                    "download_url": "https://arxiv.org/pdf/2010.08895",
                    "local_raw_path": str(tmp_path / "fno.pdf"),
                    "content_hash": "abc123",
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    class FakeMinerUClient:
        def post_json(self, endpoint: str, payload: dict[str, object], token: str | None) -> dict[str, object]:
            calls.append({"endpoint": endpoint, "payload": payload, "token": token})
            return {"code": 0, "msg": "ok", "data": {"task_id": "task-123"}}

    report = submit_mineru_parse_task(
        downloaded_source_path=source_path,
        output_path=tmp_path / "mineru_task.json",
        mode="precise_url",
        token="secret-token",
        model_version="vlm",
        client=FakeMinerUClient(),
        force=False,
    )

    assert calls[0]["token"] == "secret-token"
    assert calls[0]["endpoint"] == "https://mineru.net/api/v4/extract/task"
    assert report["state_hint"] == "MINERU_PARSE_TASK_SUBMITTED"
    assert report["task_id"] == "task-123"
    assert report["mineru_mode"] == "precise_url"
    assert "secret-token" not in json.dumps(report)
    assert report["next_command"] == "poll-mineru-parse"


def test_submit_mineru_agent_url_does_not_require_token(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import submit_mineru_parse_task

    source_path = tmp_path / "downloaded_source.json"
    source_path.write_text(
        json.dumps(
            {
                "state_hint": "SOURCE_DOCUMENT_DOWNLOADED",
                "source_record": {
                    "source_id": "arxiv_paper_fno",
                    "title": "FNO",
                    "download_url": "https://example.test/fno.pdf",
                    "local_raw_path": str(tmp_path / "fno.pdf"),
                    "content_hash": "abc123",
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeMinerUClient:
        def post_json(self, endpoint: str, payload: dict[str, object], token: str | None) -> dict[str, object]:
            assert token is None
            return {"code": 0, "msg": "ok", "data": {"task_id": "agent-task"}}

    report = submit_mineru_parse_task(
        downloaded_source_path=source_path,
        output_path=tmp_path / "mineru_task.json",
        mode="agent_url",
        token=None,
        model_version="pipeline",
        client=FakeMinerUClient(),
        force=False,
    )

    assert report["task_id"] == "agent-task"
    assert report["submit_endpoint"] == "https://mineru.net/api/v1/agent/parse/url"


def test_poll_and_download_mineru_result(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        download_mineru_parse_result,
        poll_mineru_parse_task,
    )

    task_path = tmp_path / "mineru_task.json"
    task_path.write_text(
        json.dumps(
            {
                "state_hint": "MINERU_PARSE_TASK_SUBMITTED",
                "task_id": "task-123",
                "mineru_mode": "agent_url",
                "source_id": "arxiv_paper_fno",
            }
        ),
        encoding="utf-8",
    )

    class FakeMinerUClient:
        def get_json(self, endpoint: str, token: str | None) -> dict[str, object]:
            assert endpoint == "https://mineru.net/api/v1/agent/parse/task-123"
            assert token is None
            return {
                "code": 0,
                "msg": "ok",
                "data": {
                    "task_id": "task-123",
                    "state": "done",
                    "markdown_url": "https://cdn.example/full.md",
                },
            }

        def download_bytes(self, url: str) -> bytes:
            assert url == "https://cdn.example/full.md"
            return b"# Parsed paper\n\n## Abstract\nText\n\n## Method\nText\n$$x$$"

    poll_report = poll_mineru_parse_task(
        mineru_task_path=task_path,
        output_path=tmp_path / "mineru_poll.json",
        token=None,
        client=FakeMinerUClient(),
        force=False,
    )

    assert poll_report["state_hint"] == "MINERU_PARSE_TASK_POLLED"
    assert poll_report["state"] == "done"
    assert poll_report["markdown_url"] == "https://cdn.example/full.md"
    assert poll_report["next_command"] == "download-mineru-result"

    download_report = download_mineru_parse_result(
        mineru_poll_path=tmp_path / "mineru_poll.json",
        output_path=tmp_path / "mineru_result.json",
        output_dir=tmp_path / "mineru_result",
        client=FakeMinerUClient(),
        force=False,
    )

    assert download_report["state_hint"] == "MINERU_PARSE_RESULT_DOWNLOADED"
    assert Path(download_report["local_markdown_path"]).exists()
    assert download_report["next_command"] == "ingest-mineru-parse"


def test_run_arxiv_mineru_parse_pipeline_materializes_markdown(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        run_arxiv_mineru_parse_pipeline,
    )

    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2010.08895v3</id>
    <title>Fourier Neural Operator for Parametric Partial Differential Equations</title>
    <updated>2021-05-01T00:00:00Z</updated>
    <published>2020-10-17T00:00:00Z</published>
    <link href="https://arxiv.org/pdf/2010.08895v3" type="application/pdf" title="pdf"/>
  </entry>
</feed>
""".encode()

    class FakeMinerUClient:
        def post_json(
            self, endpoint: str, payload: dict[str, object], token: str | None
        ) -> dict[str, object]:
            assert token == "token-for-test"
            assert payload["url"] == "https://arxiv.org/pdf/2010.08895v3"
            return {"code": 0, "data": {"task_id": "task-fno"}}

        def get_json(self, endpoint: str, token: str | None) -> dict[str, object]:
            assert token == "token-for-test"
            return {
                "code": 0,
                "data": {
                    "state": "done",
                    "full_zip_url": "https://mineru.example/result.zip",
                },
            }

        def download_bytes(self, url: str) -> bytes:
            assert url == "https://mineru.example/result.zip"
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr(
                    "full.md",
                    "# FNO\n\n## Abstract\nText\n\n## Method\nFourier Neural Operator text\n$$x$$",
                )
            return buffer.getvalue()

    def fake_downloader(url: str) -> bytes:
        if "export.arxiv.org" in url:
            return arxiv_xml
        if url == "https://arxiv.org/pdf/2010.08895v3":
            return b"%PDF fno"
        raise AssertionError(f"unexpected download url: {url}")

    report = run_arxiv_mineru_parse_pipeline(
        query="Fourier Neural Operator Burgers",
        output_dir=tmp_path / "run",
        token="token-for-test",
        mode="precise_url",
        model_version="vlm",
        max_results=1,
        selected_index=0,
        max_polls=1,
        poll_interval_seconds=0.0,
        downloader=fake_downloader,
        mineru_client=FakeMinerUClient(),
        force=False,
    )

    assert report["state_hint"] == "ARXIV_MINERU_PARSE_PIPELINE_RAN"
    assert report["selected_source_id"].startswith("arxiv_paper_fourier")
    assert report["mineru_state"] == "done"
    assert Path(report["artifacts"]["local_markdown_path"]).exists()
    assert report["next_command"] == "ingest-mineru-parse"
    assert "token-for-test" not in json.dumps(report)


def test_prepare_literature_extraction_context_packs_sections(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        prepare_literature_extraction_context,
    )

    markdown = tmp_path / "paper.md"
    markdown.write_text(
        "\n".join(
            [
                "# Fourier Neural Operator",
                "## Abstract",
                "We learn operators for PDEs.",
                "## Fourier Neural Operator",
                "The kernel integral operator is parameterized in Fourier space.",
                "The Fourier layer multiplies low-frequency modes by learned weights.",
                "## Experiments",
                "Burgers equation is evaluated.",
                "$$ u_t + u u_x = \\nu u_{xx} $$",
            ]
        ),
        encoding="utf-8",
    )
    parse = tmp_path / "parse.json"
    parse.write_text(
        json.dumps(
            {
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_fno",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
                "section_index": {"abstract": 2, "fourier neural operator": 4},
            }
        ),
        encoding="utf-8",
    )

    report = prepare_literature_extraction_context(
        paper_parse_path=parse,
        output_path=tmp_path / "context.json",
        force=False,
    )

    assert report["state_hint"] == "LITERATURE_EXTRACTION_CONTEXT_PREPARED"
    assert report["source_id"] == "source_fno"
    assert report["context_unit_count"] == 4
    method_unit = next(
        unit
        for unit in report["context_units"]
        if unit["section_title"] == "Fourier Neural Operator"
    )
    assert method_unit["line_start"] == 4
    assert "Fourier layer" in method_unit["text"]
    assert report["agent_contract"]["output_state_hint"] == "ALGORITHM_CARDS_EXTRACTED"
    assert report["next_command"] == "agent-extract-algorithm-cards"


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


def test_update_literature_registry_keeps_downloaded_sources_and_attempts(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import update_literature_registry

    raw = tmp_path / "fno.pdf"
    raw.write_bytes(b"%PDF fno")
    downloaded = tmp_path / "downloaded_fno.json"
    downloaded.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "SOURCE_DOCUMENT_DOWNLOADED",
                "source_record": {
                    "source_id": "arxiv_paper_fno",
                    "source_type": "arxiv_paper",
                    "title": "Fourier Neural Operator",
                    "url": "https://arxiv.org/abs/2010.08895",
                    "local_raw_path": str(raw),
                    "content_hash": "hash-fno",
                    "retrieval_query": "Burgers FNO",
                    "license_note": "arXiv paper",
                },
            }
        ),
        encoding="utf-8",
    )
    markdown = tmp_path / "full.md"
    markdown.write_text("# Fourier Neural Operator\n", encoding="utf-8")
    parse = tmp_path / "parse.json"
    parse.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "parse_id": "parse_arxiv_paper_fno",
                "source_id": "arxiv_paper_fno",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
            }
        ),
        encoding="utf-8",
    )
    cards = tmp_path / "cards.json"
    cards.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": "card_arxiv_paper_fno_fourier",
                        "source_ids": ["arxiv_paper_fno"],
                        "claim": "FNO learns PDE solution operators.",
                        "core_mechanism": "Fourier kernels parameterize operators.",
                        "implementation_hooks": [
                            "fourier_neural_operator_architecture"
                        ],
                        "evidence_status": "extracted",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "parent_algorithm_card_id": "card_arxiv_paper_fno_fourier",
                        "candidate_family": "unsupported",
                        "hypothesis": "FNO improves Burgers rollout.",
                        "executor_spec": {
                            "cli_command": "blocked_requires_executor_design",
                            "unsupported_hook": "fourier_neural_operator_architecture",
                        },
                        "source_grounding": {
                            "source_ids": ["arxiv_paper_fno"],
                            "algorithm_card_id": "card_arxiv_paper_fno_fourier",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "BLOCKED_CANDIDATE_HANDOFF_BUILT",
                "algorithm_candidate_id": "candidate_fno_blocked",
                "blocked_evidence_summary": {
                    "blocked": True,
                    "reason_codes": ["requires_executor_design"],
                    "source_grounding": {
                        "source_ids": ["arxiv_paper_fno"],
                        "algorithm_card_id": "card_arxiv_paper_fno_fourier",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = update_literature_registry(
        registry_path=tmp_path / "literature_registry.json",
        downloaded_source_paths=[downloaded],
        paper_parse_paths=[parse],
        algorithm_cards_paths=[cards],
        algorithm_candidates_paths=[candidates],
        event_paths=[handoff],
        force=False,
    )

    assert report["state_hint"] == "LITERATURE_REGISTRY_UPDATED"
    registry = json.loads(Path(report["registry_path"]).read_text(encoding="utf-8"))
    assert registry["source_count"] == 1
    source = registry["sources"][0]
    assert source["source_id"] == "arxiv_paper_fno"
    assert source["source_type"] == "arxiv_paper"
    assert source["title"] == "Fourier Neural Operator"
    assert source["download_status"] == "downloaded"
    assert source["parse_status"] == "usable"
    assert source["algorithm_card_ids"] == ["card_arxiv_paper_fno_fourier"]
    algorithm = registry["algorithm_nodes"][0]
    assert algorithm["node_id"] == "algorithm_card_arxiv_paper_fno_fourier"
    assert algorithm["strategy_status"] == "blocked"
    assert algorithm["candidate_ids"] == ["candidate_fno_blocked"]
    candidate = registry["candidate_nodes"][0]
    assert candidate["candidate_id"] == "candidate_fno_blocked"
    assert candidate["status"] == "blocked"
    assert registry["event_count"] == 5


def test_update_literature_registry_records_empty_algorithm_extraction(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        extract_algorithm_cards,
        update_literature_registry,
    )

    markdown = tmp_path / "survey.md"
    markdown.write_text(
        "\n".join(
            [
                "# Survey of PDE Benchmarks",
                "## Abstract",
                "This paper discusses datasets and evaluation challenges.",
                "## Discussion",
                "The text does not describe a reproducible algorithmic operation.",
            ]
        ),
        encoding="utf-8",
    )
    parse = tmp_path / "parse.json"
    parse.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "parse_id": "parse_source_survey_no_algorithm",
                "source_id": "source_survey_no_algorithm",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
            }
        ),
        encoding="utf-8",
    )
    cards = tmp_path / "algorithm_cards.json"
    extract_algorithm_cards(
        paper_parse_paths=[parse],
        output_path=cards,
        force=False,
    )

    report = update_literature_registry(
        registry_path=tmp_path / "literature_registry.json",
        paper_parse_paths=[parse],
        algorithm_cards_paths=[cards],
        force=False,
    )

    registry = json.loads(Path(report["registry_path"]).read_text(encoding="utf-8"))
    source = registry["sources"][0]
    assert source["source_id"] == "source_survey_no_algorithm"
    assert source["parse_status"] == "usable"
    assert source["algorithm_card_ids"] == []
    assert source["algorithm_extraction_status"] == "empty"
    assert source["algorithm_extraction_failure_code"] == "no_algorithmic_claim"
    assert source["algorithm_extraction_audit"]["extracted_block_count"] == 0
    event_types = [event["event_type"] for event in registry["events"]]
    assert event_types == ["paper_parse_ingested", "algorithm_extraction_empty"]


def test_update_literature_registry_links_promoted_strategy_to_algorithm_node(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import update_literature_registry

    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_residual",
                        "parent_algorithm_card_id": "card_residual",
                        "candidate_family": "lightweight_residual_corrector",
                        "hypothesis": "Residual correction improves rollout.",
                        "executor_spec": {"cli_command": "run-blueprint-sandbox"},
                        "source_grounding": {
                            "source_ids": ["source_residual_paper"],
                            "algorithm_card_id": "card_residual",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    proposal = tmp_path / "strategy_proposal.json"
    proposal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_PROPOSAL_RECORDED",
                "strategy_id": "candidate_residual",
                "source_type": "external_retrieval",
                "implementation_plan": {
                    "algorithm_candidate_id": "candidate_residual",
                    "source_grounding": {
                        "source_ids": ["source_residual_paper"],
                        "algorithm_card_id": "card_residual",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = update_literature_registry(
        registry_path=tmp_path / "literature_registry.json",
        algorithm_candidates_paths=[candidates],
        event_paths=[proposal],
        force=False,
    )

    registry = json.loads(Path(report["registry_path"]).read_text(encoding="utf-8"))
    algorithm = registry["algorithm_nodes"][0]
    candidate = registry["candidate_nodes"][0]
    assert algorithm["node_id"] == "algorithm_card_residual"
    assert algorithm["strategy_status"] == "strategy_proposed"
    assert algorithm["strategy_ids"] == ["candidate_residual"]
    assert candidate["status"] == "promoted"
    assert registry["events"][-1]["event_type"] == "strategy_proposal_recorded"


def test_update_literature_registry_links_checkpoint_sandbox_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import update_literature_registry

    candidates = tmp_path / "adapted_candidate.json"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "FNO_ARCHITECTURE_CANDIDATE_ADAPTED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_fno_checkpoint",
                        "parent_algorithm_card_id": "card_fno",
                        "candidate_family": "official_checkpoint_family",
                        "executor_spec": {
                            "cli_command": "run-checkpoint-family-sandbox"
                        },
                        "resource_bindings": {"checkpoint_paths": ["fno.pt"]},
                        "source_grounding": {
                            "source_ids": ["source_fno"],
                            "algorithm_card_id": "card_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sandbox = tmp_path / "checkpoint_family_sandbox_report.json"
    sandbox.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CHECKPOINT_FAMILY_SANDBOX_RAN",
                "best_candidate": {
                    "checkpoint_path": "fno.pt",
                    "score": 83.6,
                    "rules_passed": True,
                    "inference_time": 5.9,
                    "calibration_weight": 0.0,
                },
                "top_candidates": [],
            }
        ),
        encoding="utf-8",
    )

    report = update_literature_registry(
        registry_path=tmp_path / "literature_registry.json",
        algorithm_candidates_paths=[candidates],
        event_paths=[sandbox],
        force=False,
    )

    registry = json.loads(Path(report["registry_path"]).read_text(encoding="utf-8"))
    algorithm = registry["algorithm_nodes"][0]
    candidate = registry["candidate_nodes"][0]
    assert algorithm["strategy_status"] == "sandbox_verified"
    assert candidate["status"] == "sandbox_verified"
    assert candidate["sandbox_evidence"] == [
        {
            "artifact_path": str(sandbox),
            "best_score": 83.6,
            "rules_passed": True,
            "inference_time": 5.9,
            "calibration_weight": 0.0,
        }
    ]
    assert registry["events"][-1]["event_type"] == "candidate_sandbox_verified"


def test_record_literature_registry_event_normalizes_candidate_promotion_failure(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import record_literature_registry_event

    promotion = tmp_path / "candidate_promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED",
                "algorithm_candidate_id": "candidate_residual",
                "candidate_family": "lightweight_residual_corrector",
                "decision": "reject",
                "reason_codes": [
                    "sandbox_rule_failed",
                    "score_regressed",
                    "score_regressed",
                ],
                "ledger_event": {
                    "event_type": "candidate_reject",
                    "algorithm_candidate_id": "candidate_residual",
                    "candidate_family": "lightweight_residual_corrector",
                    "evidence_id": "evidence_residual_001",
                },
                "next_command": "record-literature-registry-event",
            }
        ),
        encoding="utf-8",
    )

    report = record_literature_registry_event(
        event_source_path=promotion,
        output_path=tmp_path / "registry_event.json",
        force=False,
    )

    assert report["state_hint"] == "LITERATURE_REGISTRY_EVENT_RECORDED"
    assert report["event_type"] == "candidate_rejected"
    assert report["algorithm_candidate_id"] == "candidate_residual"
    assert report["failure_codes"] == ["sandbox_rule_failed", "score_regressed"]
    assert report["recommended_next"] == "avoid_repeating_without_new_evidence"
    assert report["retry_allowed"] is False
    assert report["next_command"] == "update-literature-registry"


def test_record_literature_registry_event_rejects_unknown_failure_code(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        LiteratureAlgorithmLabError,
        record_literature_registry_event,
    )

    promotion = tmp_path / "candidate_promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "state_hint": "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED",
                "algorithm_candidate_id": "candidate_bad",
                "decision": "reject",
                "reason_codes": ["made_up_failure_code"],
                "ledger_event": {"event_type": "candidate_reject"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LiteratureAlgorithmLabError, match="unknown failure_code"):
        record_literature_registry_event(
            event_source_path=promotion,
            output_path=tmp_path / "registry_event.json",
            force=False,
        )


def test_update_literature_registry_writes_failure_event_back_to_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        record_literature_registry_event,
        update_literature_registry,
    )

    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_residual",
                        "parent_algorithm_card_id": "card_residual",
                        "candidate_family": "lightweight_residual_corrector",
                        "hypothesis": "Residual correction improves rollout.",
                        "executor_spec": {"cli_command": "run-blueprint-sandbox"},
                        "source_grounding": {
                            "source_ids": ["source_residual_paper"],
                            "algorithm_card_id": "card_residual",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    promotion = tmp_path / "candidate_promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED",
                "algorithm_candidate_id": "candidate_residual",
                "candidate_family": "lightweight_residual_corrector",
                "decision": "reject",
                "reason_codes": ["sandbox_rule_failed", "score_regressed"],
                "ledger_event": {
                    "event_type": "candidate_reject",
                    "algorithm_candidate_id": "candidate_residual",
                    "candidate_family": "lightweight_residual_corrector",
                    "evidence_id": "evidence_residual_001",
                },
                "metric_results": {"score_delta_vs_current_best": -1.0},
            }
        ),
        encoding="utf-8",
    )
    event = record_literature_registry_event(
        event_source_path=promotion,
        output_path=tmp_path / "registry_event.json",
        force=False,
    )

    report = update_literature_registry(
        registry_path=tmp_path / "literature_registry.json",
        algorithm_candidates_paths=[candidates],
        event_paths=[event["literature_registry_event_path"]],
        force=False,
    )

    registry = json.loads(Path(report["registry_path"]).read_text(encoding="utf-8"))
    candidate = registry["candidate_nodes"][0]
    algorithm = registry["algorithm_nodes"][0]
    assert candidate["status"] == "rejected"
    assert candidate["failure_history"] == [
        {
            "artifact_path": str(tmp_path / "registry_event.json"),
            "failure_codes": ["sandbox_rule_failed", "score_regressed"],
            "recommended_next": "avoid_repeating_without_new_evidence",
            "retry_allowed": False,
        }
    ]
    assert algorithm["strategy_status"] == "rejected"
    assert registry["events"][-1]["event_type"] == "candidate_rejected"
    assert registry["events"][-1]["reason_codes"] == [
        "sandbox_rule_failed",
        "score_regressed",
    ]


def test_record_literature_registry_event_cli_returns_payload(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    promotion = tmp_path / "candidate_promotion.json"
    output_path = tmp_path / "registry_event.json"
    promotion.write_text(
        json.dumps(
            {
                "state_hint": "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED",
                "algorithm_candidate_id": "candidate_refine",
                "decision": "refine",
                "reason_codes": ["ablation_missing_or_weak"],
                "ledger_event": {
                    "event_type": "candidate_refine",
                    "algorithm_candidate_id": "candidate_refine",
                },
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "record-literature-registry-event",
            "--event-source-path",
            str(promotion),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "LITERATURE_REGISTRY_EVENT_RECORDED"
    assert payload["metrics"]["event_type"] == "candidate_refined"
    assert payload["metrics"]["failure_code_count"] == 1
    assert payload["next_hint"] == "update-literature-registry"
    assert output_path.exists()


def test_validate_executable_blueprints_approves_native_dsl(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import validate_executable_blueprints

    blueprints = tmp_path / "blueprints.json"
    blueprints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXECUTABLE_BLUEPRINTS_DESIGNED",
                "blueprints": [
                    {
                        "blueprint_id": "bp_residual_001",
                        "source_candidate_id": "candidate_residual",
                        "candidate_family": "lightweight_residual_corrector",
                        "executor_id": "controlled_numpy_residual_operator",
                        "baseline": "persistence",
                        "transforms": [
                            {
                                "name": "residual_fit",
                                "params": {"shrinkage": [0.25, 0.5]},
                            }
                        ],
                        "training_plan": {"mode": "none"},
                        "parameter_grid": {"residual_shrinkage": [0.25, 0.5]},
                        "metrics": ["score", "runtime", "burgers_residual"],
                        "safety_policy": {
                            "preserve_context_steps": 10,
                            "forbid_target_leakage": True,
                            "forbid_external_code": True,
                        },
                        "expected_artifacts": [
                            "sandbox_evidence",
                            "prediction_hdf5",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_executable_blueprints(
        blueprints_path=blueprints,
        output_path=tmp_path / "blueprint_audit.json",
        force=False,
    )

    assert report["state_hint"] == "EXECUTABLE_BLUEPRINTS_VALIDATED"
    assert report["decision"] == "approved"
    assert report["issue_count"] == 0
    normalized = report["normalized_blueprints"][0]
    assert normalized["executor_id"] == "controlled_numpy_residual_operator"
    assert normalized["safety_policy"]["forbid_external_code"] is True
    assert report["next_command"] == "run-blueprint-sandbox"


def test_validate_executable_blueprints_normalizes_hypothesis_lab_blueprint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import validate_executable_blueprints

    blueprints = tmp_path / "hypothesis_blueprints.json"
    blueprints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED",
                "blueprints": [
                    {
                        "blueprint_id": "bp_task1_retrieval_prior_001",
                        "hypothesis_id": "hyp_retrieval_001",
                        "candidate_family": "task1_retrieval_prior",
                        "parameter_grid": {"k": [8, 12], "mode": ["affine"]},
                        "sandbox": {
                            "allowed_executor": "controlled_numpy_retrieval_prior",
                            "primary_metric": "task1_total_score_proxy",
                            "promotion_min_score_delta": 0.01,
                            "max_inference_seconds": 120.0,
                        },
                        "risks": ["nearest-neighbor leakage must be controlled"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_executable_blueprints(
        blueprints_path=blueprints,
        output_path=tmp_path / "blueprint_audit.json",
        force=False,
    )

    assert report["decision"] == "approved"
    normalized = report["normalized_blueprints"][0]
    assert normalized["source_candidate_id"] == "hyp_retrieval_001"
    assert normalized["baseline"] == "retrieval_prior"
    assert normalized["transforms"][0]["name"] == "retrieval_prior"
    assert normalized["safety_policy"] == {
        "preserve_context_steps": 10,
        "forbid_target_leakage": True,
        "forbid_external_code": True,
    }


def test_validate_executable_blueprints_rejects_unsafe_blueprint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import validate_executable_blueprints

    blueprints = tmp_path / "unsafe_blueprints.json"
    blueprints.write_text(
        json.dumps(
            {
                "state_hint": "EXECUTABLE_BLUEPRINTS_DESIGNED",
                "blueprints": [
                    {
                        "blueprint_id": "bp_unsafe_001",
                        "source_candidate_id": "candidate_unsafe",
                        "candidate_family": "custom_external_solver",
                        "executor_id": "arbitrary_python",
                        "baseline": "target_oracle",
                        "transforms": [{"name": "shell_exec", "params": {}}],
                        "training_plan": {"mode": "unbounded"},
                        "parameter_grid": {},
                        "metrics": [],
                        "safety_policy": {
                            "preserve_context_steps": 5,
                            "forbid_target_leakage": False,
                            "forbid_external_code": False,
                        },
                        "expected_artifacts": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_executable_blueprints(
        blueprints_path=blueprints,
        output_path=tmp_path / "blueprint_audit.json",
        force=False,
    )

    assert report["decision"] == "rejected"
    issues = {issue["code"]: issue for issue in report["issues"]}
    assert issues["invalid_baseline"]["failure_code"] == "blueprint_schema_invalid"
    assert issues["unsafe_target_leakage"]["failure_code"] == "target_leakage_risk"
    assert issues["unsafe_external_code"]["failure_code"] == "requires_external_code"
    assert report["next_command"] == "fix-executable-blueprint-dsl"


def test_validate_executable_blueprints_cli_returns_payload(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    blueprints = tmp_path / "blueprints.json"
    output_path = tmp_path / "blueprint_audit.json"
    blueprints.write_text(
        json.dumps(
            {
                "state_hint": "EXECUTABLE_BLUEPRINTS_DESIGNED",
                "blueprints": [
                    {
                        "blueprint_id": "bp_minimal_bad",
                        "source_candidate_id": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-executable-blueprints",
            "--blueprints-path",
            str(blueprints),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "EXECUTABLE_BLUEPRINTS_VALIDATED"
    assert payload["metrics"]["issue_count"] > 0
    assert payload["next_hint"] == "fix-executable-blueprint-dsl"
    assert output_path.exists()


def test_run_literature_algorithm_v2_pipeline_chains_promotion_and_registry(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        run_literature_algorithm_v2_pipeline,
    )

    cards = tmp_path / "algorithm_cards.json"
    candidate_id = "candidate_card_residual_lightweight_residual_corrector"
    cards.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": "card_residual",
                        "source_ids": ["source_residual"],
                        "claim": "Residual correction can improve rollout.",
                        "core_mechanism": "Fit a bounded residual map.",
                        "operator_target": "residual_to_correction",
                        "mechanism_type": "postprocess",
                        "mathematical_object": "horizon_residual",
                        "operation": "Run a residual correction sandbox.",
                        "implementation_hooks": ["learned_residual_correction"],
                        "implementation_requirements": {
                            "requires_training": False,
                            "requires_checkpoint": False,
                            "requires_target_data": True,
                            "requires_new_executor": False,
                            "requires_external_code": False,
                            "can_run_as_postprocess": True,
                        },
                        "preconditions": ["Burgers validation windows exist"],
                        "expected_observables": ["score improves"],
                        "failure_modes": ["overfits validation windows"],
                        "minimal_experiment": "Run residual correction sandbox.",
                        "evidence_status": "extracted",
                        "provenance": {
                            "parse_id": "parse_residual",
                            "source_content_hash": "hash_residual",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "sandbox_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "state_hint": "SANDBOX_EVIDENCE_RECORDED",
                "evidence_id": "evidence_residual_001",
                "candidate_id": candidate_id,
                "rules_passed": True,
                "score": 86.3,
                "runtime": {"inference_seconds": 12.0},
                "physics_metrics": {"residual_delta": -0.01},
                "ablation_metrics": {"score_delta": 0.2},
                "score_delta_vs_current_best": 0.2,
                "artifact_paths": [str(tmp_path / "pred.hdf5")],
                "reproducibility": {"deterministic": True},
            }
        ),
        encoding="utf-8",
    )

    report = run_literature_algorithm_v2_pipeline(
        algorithm_cards_path=cards,
        workspace_dir=tmp_path / "workspace",
        registry_path=tmp_path / "literature_registry.json",
        current_best_score=86.1,
        algorithm_candidate_id=candidate_id,
        sandbox_evidence_paths=[evidence],
        force=False,
    )

    assert report["state_hint"] == "LITERATURE_ALGORITHM_V2_PIPELINE_RAN"
    assert report["pipeline_decision"] == "promotion_recorded"
    assert report["artifacts"]["algorithm_cards_v2_audit_path"]
    assert report["artifacts"]["executable_blueprint_audit_path"]
    assert report["artifacts"]["candidate_promotion_path"]
    assert report["artifacts"]["literature_registry_event_path"]
    assert report["artifacts"]["hermes_registry_summary_path"]
    assert report["metrics"]["candidate_count"] == 1
    assert report["metrics"]["blueprint_issue_count"] == 0
    assert report["next_command"] == "record-strategy-proposal"

    registry = json.loads((tmp_path / "literature_registry.json").read_text())
    assert registry["candidate_nodes"][0]["status"] == "promoted"


def test_run_literature_algorithm_v2_pipeline_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    cards = tmp_path / "algorithm_cards.json"
    cards.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": "card_spectral",
                        "source_ids": ["source_spectral"],
                        "claim": "Spectral smoothing stabilizes rollout.",
                        "core_mechanism": "Damp high frequency modes.",
                        "operator_target": "postprocess_to_solution",
                        "mechanism_type": "postprocess",
                        "mathematical_object": "spectral_filter",
                        "operation": "Run spectral rollout sandbox.",
                        "implementation_hooks": ["spectral_rollout_stabilization"],
                        "implementation_requirements": {
                            "requires_training": False,
                            "requires_checkpoint": False,
                            "requires_target_data": False,
                            "requires_new_executor": False,
                            "requires_external_code": False,
                            "can_run_as_postprocess": True,
                        },
                        "preconditions": ["Burgers context exists"],
                        "expected_observables": ["late RMSE improves"],
                        "failure_modes": ["shock oversmoothing"],
                        "minimal_experiment": "Run spectral sandbox.",
                        "evidence_status": "extracted",
                        "provenance": {"parse_id": "parse_spectral"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run-literature-algorithm-v2-pipeline",
            "--algorithm-cards-path",
            str(cards),
            "--workspace-dir",
            str(tmp_path / "workspace"),
            "--registry-path",
            str(tmp_path / "literature_registry.json"),
            "--current-best-score",
            "86.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "LITERATURE_ALGORITHM_V2_PIPELINE_RAN"
    assert payload["metrics"]["candidate_count"] == 1
    assert payload["next_hint"] == "run-blueprint-sandbox"


def test_compile_algorithm_candidates_expands_all_literature_strategy_families(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import compile_algorithm_candidates

    cards = tmp_path / "cards.json"
    cards.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": "card_multi_strategy",
                        "source_ids": ["paper_001"],
                        "claim": "Several PDE operator interventions are plausible.",
                        "core_mechanism": "Combine residual, spectral, rollout, and correction ideas.",
                        "implementation_hooks": [
                            "learned_residual_correction",
                            "spectral_rollout_stabilization",
                            "physics_informed_loss",
                            "scheduled_sampling",
                            "operator_correction_postprocess",
                        ],
                        "evidence_status": "extracted",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = compile_algorithm_candidates(
        algorithm_cards_path=cards,
        output_path=tmp_path / "candidates.json",
        force=False,
    )

    families = {
        candidate["candidate_family"]: candidate
        for candidate in report["algorithm_candidates"]
    }
    assert report["algorithm_candidate_count"] == 5
    assert set(families) == {
        "lightweight_residual_corrector",
        "spectral_rollout_stabilization",
        "physics_residual_regularization",
        "scheduled_rollout_training",
        "operator_correction_postprocess",
    }
    assert families["physics_residual_regularization"]["normalized_hook"] == (
        "physics_residual_regularization"
    )
    assert families["physics_residual_regularization"]["literature_hook"] == (
        "physics_informed_loss"
    )
    assert families["scheduled_rollout_training"]["normalized_hook"] == (
        "scheduled_rollout_training"
    )
    assert families["operator_correction_postprocess"]["parameter_grid"] == {
        "correction_strength": [0.25, 0.5, 0.75],
        "spectral_strength": [0.0, 0.5],
    }


def test_validate_algorithm_cards_v2_approves_semiformal_cards(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import validate_algorithm_cards_v2

    cards = tmp_path / "cards_v2.json"
    cards.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": "card_v2_fno",
                        "source_ids": ["paper_fno"],
                        "claim": "Fourier kernels learn solution operators.",
                        "operator_target": "ic_to_trajectory",
                        "mechanism_type": "architecture",
                        "mathematical_object": "fourier_modes",
                        "operation": "Evaluate Fourier-mode operator rollout on Burgers.",
                        "implementation_hooks": [
                            "fourier_neural_operator_architecture"
                        ],
                        "implementation_requirements": {
                            "requires_training": False,
                            "requires_checkpoint": True,
                            "requires_target_data": False,
                            "requires_new_executor": True,
                            "requires_external_code": False,
                            "can_run_as_postprocess": False,
                        },
                        "preconditions": ["PDEBench Burgers context is available"],
                        "expected_observables": ["lower rollout RMSE"],
                        "failure_modes": ["checkpoint adapter underperforms"],
                        "minimal_experiment": "Run checkpoint-family sandbox.",
                        "evidence_status": "extracted",
                        "provenance": {
                            "parse_id": "parse_fno",
                            "source_content_hash": "hash_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_algorithm_cards_v2(
        algorithm_cards_path=cards,
        output_path=tmp_path / "cards_v2_audit.json",
        force=False,
    )

    assert report["state_hint"] == "ALGORITHM_CARDS_V2_VALIDATED"
    assert report["decision"] == "approved"
    assert report["issue_count"] == 0
    assert report["card_count"] == 1
    assert report["next_command"] == "compile-algorithm-candidates"


def test_validate_algorithm_cards_v2_rejects_missing_operation_and_bad_enum(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import validate_algorithm_cards_v2

    cards = tmp_path / "cards_v2.json"
    cards.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": "card_bad",
                        "source_ids": ["paper_bad"],
                        "claim": "This is too vague.",
                        "operator_target": "generic_ai_solver",
                        "mechanism_type": "architecture",
                        "mathematical_object": "fourier_modes",
                        "operation": "",
                        "implementation_hooks": [],
                        "implementation_requirements": {
                            "requires_training": "maybe",
                        },
                        "preconditions": [],
                        "expected_observables": [],
                        "failure_modes": [],
                        "minimal_experiment": "",
                        "evidence_status": "extracted",
                        "provenance": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_algorithm_cards_v2(
        algorithm_cards_path=cards,
        output_path=tmp_path / "cards_v2_audit.json",
        force=False,
    )

    assert report["decision"] == "rejected"
    assert report["issue_count"] >= 4
    issues = {issue["code"]: issue for issue in report["issues"]}
    assert issues["invalid_operator_target"]["failure_code"] == (
        "candidate_schema_invalid"
    )
    assert issues["missing_operation"]["failure_code"] == "no_algorithmic_claim"
    assert issues["empty_implementation_hooks"]["failure_code"] == (
        "candidate_schema_invalid"
    )
    assert issues["implementation_requirement_not_bool"]["path"] == (
        "algorithm_cards[0].implementation_requirements.requires_training"
    )


def test_validate_algorithm_cards_v2_cli_returns_payload(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    cards = tmp_path / "cards_v2.json"
    output_path = tmp_path / "cards_v2_audit.json"
    cards.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": "card_minimal_bad",
                        "source_ids": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-algorithm-cards-v2",
            "--algorithm-cards-path",
            str(cards),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "ALGORITHM_CARDS_V2_VALIDATED"
    assert payload["metrics"]["issue_count"] > 0
    assert payload["next_hint"] == "fix-algorithm-card-v2-schema"
    assert output_path.exists()


def test_evaluate_algorithm_candidate_promotion_promotes_valid_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        evaluate_algorithm_candidate_promotion,
    )

    candidate_id = "candidate_card_spectral_spectral_rollout_stabilization"
    candidates = tmp_path / "algorithm_candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": "spectral_rollout_stabilization",
                        "executor_spec": {"cli_command": "run-blueprint-sandbox"},
                        "promotion_criteria": {
                            "min_score_delta": 0.05,
                            "max_inference_seconds": 120.0,
                            "requires_ablation": True,
                        },
                        "source_grounding": {
                            "source_ids": ["paper_spectral"],
                            "algorithm_card_id": "card_spectral",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "sandbox_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "SANDBOX_EVIDENCE_RECORDED",
                "evidence_id": "evidence_spectral_001",
                "candidate_id": candidate_id,
                "blueprint_id": "blueprint_spectral_001",
                "sandbox_id": "sandbox_task1_local",
                "dataset_id": "pdebench_burgers_task1_validation",
                "rules_passed": True,
                "score": 86.25,
                "segment_scores": {"early": 93.0, "middle": 88.0, "late": 82.0},
                "runtime": {"inference_seconds": 34.0},
                "physics_metrics": {"residual_delta": -0.02},
                "ablation_metrics": {"score_delta": 0.31},
                "score_delta_vs_current_best": 0.31,
                "artifact_paths": [str(tmp_path / "prediction.hdf5")],
                "reproducibility": {"deterministic": True, "repeat_count": 2},
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_algorithm_candidate_promotion(
        algorithm_candidates_path=candidates,
        algorithm_candidate_id=candidate_id,
        sandbox_evidence_paths=[evidence],
        output_path=tmp_path / "candidate_promotion.json",
        current_best_score=85.94,
        force=False,
    )

    assert report["state_hint"] == "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED"
    assert report["decision"] == "promote"
    assert report["promote_to_strategy_growth"] is True
    assert report["reason_codes"] == ["all_rubric_checks_passed"]
    assert report["ledger_event"]["event_type"] == "candidate_promoted"
    assert report["next_command"] == "record-strategy-proposal"


def test_evaluate_algorithm_candidate_promotion_rejects_rule_failures(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        evaluate_algorithm_candidate_promotion,
    )

    candidate_id = "candidate_card_residual_lightweight_residual_corrector"
    candidates = tmp_path / "algorithm_candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": "lightweight_residual_corrector",
                        "executor_spec": {"cli_command": "run-blueprint-sandbox"},
                        "promotion_criteria": {
                            "min_score_delta": 0.1,
                            "max_inference_seconds": 60.0,
                            "requires_ablation": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "sandbox_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "state_hint": "SANDBOX_EVIDENCE_RECORDED",
                "evidence_id": "evidence_residual_001",
                "candidate_id": candidate_id,
                "rules_passed": False,
                "score": 84.0,
                "runtime": {"inference_seconds": 88.0},
                "physics_metrics": {"residual_delta": 0.03},
                "ablation_metrics": {},
                "score_delta_vs_current_best": -1.0,
                "artifact_paths": [],
                "reproducibility": {"deterministic": False},
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_algorithm_candidate_promotion(
        algorithm_candidates_path=candidates,
        algorithm_candidate_id=candidate_id,
        sandbox_evidence_paths=[evidence],
        output_path=tmp_path / "candidate_promotion.json",
        current_best_score=85.0,
        force=False,
    )

    assert report["decision"] == "reject"
    assert report["promote_to_strategy_growth"] is False
    assert "sandbox_rule_failed" in report["reason_codes"]
    assert "score_regressed" in report["reason_codes"]
    assert report["failed_checks"]["validity"]["failure_code"] == "sandbox_rule_failed"
    assert report["next_command"] == "record-literature-registry-event"


def test_evaluate_algorithm_candidate_promotion_blocks_unsupported_executor(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        evaluate_algorithm_candidate_promotion,
    )

    candidate_id = "candidate_card_fno_unsupported"
    candidates = tmp_path / "algorithm_candidates.json"
    evidence = tmp_path / "sandbox_evidence.json"
    candidates.write_text(
        json.dumps(
            {
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": "unsupported",
                        "executor_spec": {
                            "cli_command": "blocked_requires_executor_design"
                        },
                        "promotion_criteria": {
                            "min_score_delta": 0.01,
                            "max_inference_seconds": 120.0,
                            "requires_ablation": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence.write_text(
        json.dumps(
            {
                "state_hint": "SANDBOX_EVIDENCE_RECORDED",
                "evidence_id": "evidence_blocked_001",
                "candidate_id": candidate_id,
                "rules_passed": True,
                "score": 90.0,
                "runtime": {"inference_seconds": 10.0},
                "physics_metrics": {"residual_delta": -0.01},
                "ablation_metrics": {},
                "score_delta_vs_current_best": 4.0,
                "artifact_paths": [str(tmp_path / "pred.hdf5")],
                "reproducibility": {"deterministic": True},
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_algorithm_candidate_promotion(
        algorithm_candidates_path=candidates,
        algorithm_candidate_id=candidate_id,
        sandbox_evidence_paths=[evidence],
        output_path=tmp_path / "candidate_promotion.json",
        current_best_score=86.0,
        force=False,
    )

    assert report["decision"] == "blocked"
    assert report["promote_to_strategy_growth"] is False
    assert report["failed_checks"]["executor"]["failure_code"] == (
        "unsupported_executor"
    )
    assert report["next_command"] == "create-executor-design-request"


def test_evaluate_algorithm_candidate_promotion_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    candidate_id = "candidate_card_checkpoint_official_checkpoint_family"
    candidates = tmp_path / "algorithm_candidates.json"
    evidence = tmp_path / "sandbox_evidence.json"
    output_path = tmp_path / "candidate_promotion.json"
    candidates.write_text(
        json.dumps(
            {
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": candidate_id,
                        "candidate_family": "official_checkpoint_family",
                        "executor_spec": {"cli_command": "run-checkpoint-family-sandbox"},
                        "promotion_criteria": {
                            "min_score_delta": 0.01,
                            "max_inference_seconds": 120.0,
                            "requires_ablation": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence.write_text(
        json.dumps(
            {
                "state_hint": "SANDBOX_EVIDENCE_RECORDED",
                "evidence_id": "evidence_checkpoint_001",
                "candidate_id": candidate_id,
                "rules_passed": True,
                "score": 85.8,
                "runtime": {"inference_seconds": 21.0},
                "physics_metrics": {"residual_delta": -0.01},
                "ablation_metrics": {},
                "score_delta_vs_current_best": 0.2,
                "artifact_paths": [str(tmp_path / "pred.hdf5")],
                "reproducibility": {"deterministic": True},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-algorithm-candidate-promotion",
            "--algorithm-candidates-path",
            str(candidates),
            "--algorithm-candidate-id",
            candidate_id,
            "--sandbox-evidence-path",
            str(evidence),
            "--output-path",
            str(output_path),
            "--current-best-score",
            "85.6",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED"
    assert payload["metrics"]["decision"] == "promote"
    assert payload["metrics"]["evidence_count"] == 1
    assert payload["next_hint"] == "record-strategy-proposal"
    assert output_path.exists()


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


def test_ingest_mineru_parse_accepts_downloaded_source_record(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import ingest_mineru_parse

    source_path = tmp_path / "downloaded_source.json"
    source_path.write_text(
        json.dumps(
            {
                "state_hint": "SOURCE_DOCUMENT_DOWNLOADED",
                "source_record": {
                    "source_id": "source_fno",
                    "title": "FNO",
                    "local_raw_path": str(tmp_path / "paper.pdf"),
                    "content_hash": "abc",
                },
            }
        ),
        encoding="utf-8",
    )
    markdown = tmp_path / "full.md"
    markdown.write_text(
        "\n".join(
            [
                "# Paper",
                "## Abstract",
                "Neural operator abstract.",
                "## Method",
                "The method learns operators.",
                "$$x$$",
            ]
        ),
        encoding="utf-8",
    )

    report = ingest_mineru_parse(
        source_records_path=source_path,
        source_id="source_fno",
        markdown_path=markdown,
        asset_dir=tmp_path / "images",
        output_path=tmp_path / "parse.json",
        force=False,
    )

    assert report["state_hint"] == "MINERU_PARSE_INGESTED"
    assert report["source_id"] == "source_fno"


def test_ingest_mineru_parse_accepts_neural_operator_method_sections(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import ingest_mineru_parse

    source_path = tmp_path / "downloaded_source.json"
    source_path.write_text(
        json.dumps(
            {
                "state_hint": "SOURCE_DOCUMENT_DOWNLOADED",
                "source_record": {"source_id": "source_fno"},
            }
        ),
        encoding="utf-8",
    )
    markdown = tmp_path / "full.md"
    markdown.write_text(
        "\n".join(
            [
                "# Fourier Neural Operator for PDEs",
                "## Abstract",
                "We introduce Fourier neural operators.",
                "## Neural Operator",
                "The model learns a map between function spaces.",
                "## Fourier Neural Operator",
                "The kernel integral is parameterized in Fourier space.",
                "$$x$$",
            ]
        ),
        encoding="utf-8",
    )

    report = ingest_mineru_parse(
        source_records_path=source_path,
        source_id="source_fno",
        markdown_path=markdown,
        asset_dir=tmp_path / "images",
        output_path=tmp_path / "parse.json",
        force=False,
    )

    assert report["parse_quality"]["quality"] == "usable"
    assert report["parse_quality"]["quality_flags"] == []


def test_extract_algorithm_cards_reads_operational_markdown(
    tmp_path: Path,
) -> None:
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


def test_extract_algorithm_cards_infers_fno_from_paper_markdown(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        extract_algorithm_cards,
        validate_algorithm_cards_v2,
    )

    markdown = tmp_path / "fno.md"
    markdown.write_text(
        "\n".join(
            [
                "# Fourier Neural Operator for Parametric Partial Differential Equations",
                "## Abstract",
                "We propose Fourier neural operators for parametric PDEs.",
                "## Fourier Neural Operator",
                "The kernel integral operator is parameterized directly in Fourier space.",
                "The model learns mappings between infinite-dimensional function spaces.",
                "## Numerical Experiments",
                "Burgers equation, Darcy flow, and Navier-Stokes equation are evaluated.",
                "$$ u_t + u u_x = nu u_xx $$",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_fno",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
            }
        ),
        encoding="utf-8",
    )

    report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "cards.json",
        force=False,
    )
    audit = validate_algorithm_cards_v2(
        algorithm_cards_path=tmp_path / "cards.json",
        output_path=tmp_path / "cards_v2_audit.json",
        force=False,
    )

    card = report["algorithm_cards"][0]
    assert card["implementation_hooks"] == ["fourier_neural_operator_architecture"]
    assert "Fourier" in card["claim"]
    assert "Fourier space" in card["core_mechanism"]
    assert card["operator_target"] == "ic_to_trajectory"
    assert card["mechanism_type"] == "architecture"
    assert card["mathematical_object"] == "fourier_modes"
    assert card["implementation_requirements"]["requires_new_executor"] is True
    assert card["minimal_experiment"] == (
        "Design a bounded FNO architecture executor or map the source to an existing checkpoint-family adapter before sandbox execution."
    )
    assert audit["decision"] == "approved"


def test_extract_algorithm_cards_emits_algorithm_card_v2_fields(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        extract_algorithm_cards,
        validate_algorithm_cards_v2,
    )

    markdown = tmp_path / "residual.md"
    markdown.write_text(
        "\n".join(
            [
                "# Residual Neural Operator Correction",
                "Claim: A learned residual correction can reduce rollout bias in Burgers forecasts.",
                "Mechanism: Estimate the checkpoint residual as a small postprocess operator.",
                "Operation: Fit a residual map on local validation windows and apply a horizon-decayed correction.",
                "Hook: learned_residual_correction",
                "Observable: lower early and middle Rel-MSE without increasing inference time materially.",
                "Risk: residual correction can overfit the local validation windows.",
                "Minimal Experiment: compare checkpoint rollout against residual-corrected checkpoint rollout on a held-out window.",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_residual_operator",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
                "section_index": ["abstract", "method", "experiment"],
            }
        ),
        encoding="utf-8",
    )

    report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )
    audit = validate_algorithm_cards_v2(
        algorithm_cards_path=tmp_path / "algorithm_cards.json",
        output_path=tmp_path / "algorithm_cards_v2_audit.json",
        force=False,
    )

    card = report["algorithm_cards"][0]
    assert card["operator_target"] == "residual_to_correction"
    assert card["mechanism_type"] == "postprocess"
    assert card["mathematical_object"] == "residual_operator"
    assert card["implementation_requirements"] == {
        "requires_training": True,
        "requires_checkpoint": True,
        "requires_target_data": True,
        "requires_new_executor": False,
        "requires_external_code": False,
        "can_run_as_postprocess": True,
    }
    assert card["provenance"]["source_id"] == "source_residual_operator"
    assert card["provenance"]["extractor"] == "heuristic_algorithm_card_v2"
    assert audit["decision"] == "approved"
    assert audit["issue_count"] == 0


def test_extract_algorithm_cards_reads_multiple_structured_sections(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        extract_algorithm_cards,
        validate_algorithm_cards_v2,
    )

    markdown = tmp_path / "multi.md"
    markdown.write_text(
        "\n".join(
            [
                "# Burgers Operator Stabilization Notes",
                "## Abstract",
                "We study several bounded neural-operator improvements for Burgers rollouts.",
                "## Method",
                "### Residual Corrector",
                "Claim: Residual correction reduces short-horizon checkpoint bias.",
                "Mechanism: Learn a residual operator from local validation windows.",
                "Operation: Fit and apply a horizon-decayed residual correction.",
                "Hook: learned_residual_correction",
                "Observable: lower early Rel-MSE.",
                "Risk: can overfit validation windows.",
                "Minimal Experiment: compare checkpoint rollout against residual-corrected rollout.",
                "### Spectral Stabilizer",
                "Claim: Spectral damping suppresses late-horizon high-frequency drift.",
                "Mechanism: Damp high Fourier modes only after the context window.",
                "Operation: Apply a cutoff spectral filter during autoregressive rollout.",
                "Hook: spectral_rollout_stabilization",
                "Observable: lower late RMSE.",
                "Risk: can oversmooth shocks.",
                "Minimal Experiment: sweep spectral cutoff and damping on held-out Burgers windows.",
                "## Experiments",
                "$$ u_t + u u_x = \\nu u_{xx} $$",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_multi_method",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
                "section_index": {
                    "abstract": 2,
                    "method": 4,
                    "residual corrector": 5,
                    "spectral stabilizer": 13,
                },
            }
        ),
        encoding="utf-8",
    )

    report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )
    audit = validate_algorithm_cards_v2(
        algorithm_cards_path=tmp_path / "algorithm_cards.json",
        output_path=tmp_path / "algorithm_cards_v2_audit.json",
        force=False,
    )

    assert report["algorithm_card_count"] == 2
    hooks = [card["implementation_hooks"][0] for card in report["algorithm_cards"]]
    assert hooks == [
        "learned_residual_correction",
        "spectral_rollout_stabilization",
    ]
    assert {
        card["provenance"]["section_title"] for card in report["algorithm_cards"]
    } == {"Residual Corrector", "Spectral Stabilizer"}
    assert len({card["algorithm_card_id"] for card in report["algorithm_cards"]}) == 2
    assert audit["decision"] == "approved"
    assert audit["issue_count"] == 0


def test_extract_algorithm_cards_records_audit_and_evidence_spans(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import extract_algorithm_cards

    markdown = tmp_path / "grounded.md"
    markdown.write_text(
        "\n".join(
            [
                "# Burgers Operator Stabilization Notes",
                "## Abstract",
                "This note contains prose that should be scanned but not extracted.",
                "## Method",
                "### Residual Corrector",
                "Claim: Residual correction reduces short-horizon checkpoint bias.",
                "Mechanism: Learn a residual operator from local validation windows.",
                "Operation: Fit and apply a horizon-decayed residual correction.",
                "Hook: learned_residual_correction",
                "Observable: lower early Rel-MSE.",
                "Risk: can overfit validation windows.",
                "Minimal Experiment: compare checkpoint rollout against residual-corrected rollout.",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_grounded_method",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
                "section_index": {
                    "abstract": 2,
                    "method": 4,
                    "residual corrector": 5,
                },
            }
        ),
        encoding="utf-8",
    )

    report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )

    audit = report["extraction_audit"]
    assert audit["parse_count"] == 1
    assert audit["usable_parse_count"] == 1
    assert audit["scanned_block_count"] == 4
    assert audit["extracted_block_count"] == 1
    skipped_titles = {
        block["section_title"]
        for block in audit["skipped_blocks"]
        if block["reason_code"] == "missing_algorithm_card_fields"
    }
    assert "Abstract" in skipped_titles
    assert "Method" in skipped_titles

    card = report["algorithm_cards"][0]
    spans = card["evidence_spans"]
    assert spans["claim"]["line_start"] == 6
    assert spans["claim"]["line_end"] == 6
    assert spans["claim"]["text"].startswith("Claim: Residual correction")
    assert spans["mechanism"]["line_start"] == 7
    assert spans["operation"]["line_start"] == 8
    assert spans["minimal_experiment"]["line_start"] == 12


def test_extract_algorithm_cards_infers_nonstructured_method_section(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        extract_algorithm_cards,
        validate_algorithm_cards_v2,
    )

    markdown = tmp_path / "unstructured_method.md"
    markdown.write_text(
        "\n".join(
            [
                "# Spectral Viscosity for Burgers Rollouts",
                "## Abstract",
                "We study stable long-horizon neural operator rollouts.",
                "## Method",
                "### Spectral Viscosity Rollout",
                "We stabilize Burgers autoregressive rollouts by damping high Fourier modes after the context window.",
                "At each predicted step, a cutoff spectral filter is applied to the Fourier coefficients.",
                "This reduces late-horizon high-frequency drift but can oversmooth shocks near discontinuities.",
                "## Experiments",
                "$$ u_t + u u_x = \\nu u_{xx} $$",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_spectral_unstructured",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
                "section_index": {
                    "abstract": 2,
                    "method": 4,
                    "spectral viscosity rollout": 5,
                    "experiments": 9,
                },
            }
        ),
        encoding="utf-8",
    )

    report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )
    audit = validate_algorithm_cards_v2(
        algorithm_cards_path=tmp_path / "algorithm_cards.json",
        output_path=tmp_path / "algorithm_cards_v2_audit.json",
        force=False,
    )

    assert report["algorithm_card_count"] == 1
    assert report["extraction_audit"]["inferred_block_count"] == 1
    card = report["algorithm_cards"][0]
    assert card["implementation_hooks"] == ["spectral_rollout_stabilization"]
    assert card["claim"].startswith("Spectral damping can stabilize")
    assert card["provenance"]["source_shape"] == "inferred_section"
    assert card["provenance"]["section_title"] == "Spectral Viscosity Rollout"
    assert card["evidence_spans"]["claim"]["line_start"] == 6
    assert card["evidence_spans"]["mechanism"]["line_start"] == 7
    assert card["evidence_spans"]["risk"]["line_start"] == 8
    assert audit["decision"] == "approved"


def test_extract_algorithm_cards_prefers_fno_architecture_section_over_intro(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import extract_algorithm_cards

    markdown = tmp_path / "fno_realistic.md"
    markdown.write_text(
        "\n".join(
            [
                "# Fourier Neural Operator for Parametric Partial Differential Equations",
                "## 1 Introduction",
                "Figure 1 shows the architecture of the Fourier layer for fluid flow prediction.",
                "## 4 Fourier Neural Operator",
                "We formulate the Fourier neural operator by parameterizing the kernel integral operator in Fourier space.",
                "The Fourier layer applies a Fourier transform, truncates to low-frequency modes, multiplies learned weights, and transforms back to physical space.",
                "This architecture learns mappings between infinite-dimensional function spaces across discretizations.",
                "## 5 Numerical Experiments",
                "We evaluate on Burgers equation, Darcy flow, and Navier-Stokes equation.",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_fno_realistic",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
                "section_index": {
                    "1 introduction": 2,
                    "4 fourier neural operator": 4,
                    "5 numerical experiments": 8,
                },
            }
        ),
        encoding="utf-8",
    )

    report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )

    assert report["algorithm_card_count"] == 1
    card = report["algorithm_cards"][0]
    assert card["implementation_hooks"] == ["fourier_neural_operator_architecture"]
    assert card["operator_target"] == "ic_to_trajectory"
    assert card["mechanism_type"] == "architecture"
    assert card["mathematical_object"] == "fourier_modes"
    assert card["provenance"]["section_title"] == "4 Fourier Neural Operator"
    assert card["evidence_spans"]["mechanism"]["line_start"] == 5


def test_extract_algorithm_cards_records_empty_extraction_artifact(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import extract_algorithm_cards

    markdown = tmp_path / "survey.md"
    markdown.write_text(
        "\n".join(
            [
                "# Survey of PDE Benchmarks",
                "## Abstract",
                "This paper discusses datasets and evaluation challenges.",
                "## Discussion",
                "The text mentions Burgers equations but does not describe a reproducible algorithmic operation.",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_survey_no_algorithm",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
                "section_index": {"abstract": 2, "discussion": 4},
            }
        ),
        encoding="utf-8",
    )

    report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )

    assert report["state_hint"] == "ALGORITHM_CARDS_EXTRACTED"
    assert report["algorithm_card_count"] == 0
    assert report["algorithm_cards"] == []
    assert report["extraction_status"] == "empty"
    assert report["failure_code"] == "no_algorithmic_claim"
    assert report["next_command"] == "update-literature-registry"
    assert report["extraction_audit"]["scanned_block_count"] == 3
    assert report["extraction_audit"]["extracted_block_count"] == 0
    assert report["extraction_audit"]["skipped_blocks"]


def test_extract_algorithm_cards_command_reports_empty_status(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.literature_algorithm_lab import (
        extract_algorithm_cards_command,
    )

    markdown = tmp_path / "survey.md"
    markdown.write_text(
        "# Survey\n## Discussion\nThis paper only discusses benchmark metadata.",
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_cli_empty",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
            }
        ),
        encoding="utf-8",
    )

    payload = extract_algorithm_cards_command(
        paper_parse_paths=[str(parse_path)],
        output_path=str(tmp_path / "algorithm_cards.json"),
        force=False,
    )

    assert payload["metrics"]["algorithm_card_count"] == 0
    assert payload["metrics"]["extraction_status"] == "empty"
    assert payload["metrics"]["failure_code"] == "no_algorithmic_claim"
    assert payload["next_hint"] == "update-literature-registry"


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
    assert checkpoint["executor_spec"]["cli_command"] == (
        "run-checkpoint-family-sandbox"
    )
    assert checkpoint["target_problem"] == "PDE_Burgers"
    residual = by_family["lightweight_residual_corrector"]
    assert residual["executor_spec"]["cli_command"] == "run-blueprint-sandbox"
    assert residual["promotion_criteria"]["requires_ablation"] is True
    assert report["next_command"] == "plan-algorithm-candidate-sandbox"


def test_compile_algorithm_candidates_keeps_long_card_ids_unique(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import compile_algorithm_candidates

    long_prefix = (
        "card_arxiv_paper_fourier_neural_operator_for_parametric_partial_"
        "differential_equations"
    )
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CARDS_EXTRACTED",
                "algorithm_cards": [
                    {
                        "algorithm_card_id": f"{long_prefix}_section_fourier",
                        "source_ids": ["source_fno"],
                        "claim": "FNO section claim.",
                        "core_mechanism": "FNO section mechanism.",
                        "operation": "Design bounded FNO executor.",
                        "preconditions": ["PDEBench Burgers"],
                        "implementation_hooks": ["fourier_neural_operator_architecture"],
                        "expected_observables": ["score improves"],
                        "failure_modes": ["executor mismatch"],
                        "minimal_experiment": "compile blocked candidate",
                        "evidence_status": "extracted",
                    },
                    {
                        "algorithm_card_id": f"{long_prefix}_appendix_spectral",
                        "source_ids": ["source_fno"],
                        "claim": "FNO appendix claim.",
                        "core_mechanism": "FNO appendix mechanism.",
                        "operation": "Design bounded FNO executor.",
                        "preconditions": ["PDEBench Burgers"],
                        "implementation_hooks": ["fourier_neural_operator_architecture"],
                        "expected_observables": ["score improves"],
                        "failure_modes": ["executor mismatch"],
                        "minimal_experiment": "compile blocked candidate",
                        "evidence_status": "extracted",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = compile_algorithm_candidates(
        algorithm_cards_path=cards_path,
        output_path=tmp_path / "candidates.json",
        force=False,
    )

    ids = [
        candidate["algorithm_candidate_id"]
        for candidate in report["algorithm_candidates"]
    ]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_literature_residual_candidate_runs_in_hypothesis_lab_sandbox(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
        run_blueprint_sandbox,
    )
    from pde_agent.core.literature_algorithm_lab import (
        compile_algorithm_candidates,
        extract_algorithm_cards,
    )

    markdown = tmp_path / "residual_operator.md"
    markdown.write_text(
        "\n".join(
            [
                "# Residual-Corrected Neural Operators for Burgers Rollout",
                "## Abstract",
                "We use residual corrections to reduce autoregressive rollout bias.",
                "## Algorithm",
                "Claim: Horizon residual correction can reduce Burgers rollout bias.",
                "Mechanism: Estimate a horizon-conditioned residual from local validation windows.",
                "Operation: Apply a bounded residual correction to a persistence rollout.",
                "Hook: learned_residual_correction",
                "Observable: local sandbox score improves over persistence",
                "Risk: residual correction can overfit a small validation split",
                "Minimal Experiment: run a lightweight residual correction sandbox",
                "$$ u_t + u u_x = \\nu u_{xx} $$",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_residual_operator",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
            }
        ),
        encoding="utf-8",
    )
    cards_report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )
    candidates_report = compile_algorithm_candidates(
        algorithm_cards_path=cards_report["algorithm_cards_path"],
        output_path=tmp_path / "algorithm_candidates.json",
        force=False,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[candidates_report["algorithm_candidates_path"]],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["source_type"] == "algorithm_candidate"
    assert cards["cards"][0]["implementation_hooks"] == [
        "learned_residual_correction"
    ]
    assert blueprints["blueprints"][0]["candidate_family"] == (
        "learned_residual_correction"
    )

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    base = np.sin(x)[None, None, :]
    context = np.repeat(base, 10, axis=1).astype(np.float32)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    target[:, 10:, :] = context[:, -1:, :] + np.float32(0.1)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    sandbox = run_blueprint_sandbox(
        blueprint=blueprints["blueprints"][0],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes_literature_sandbox",
        force=False,
    )

    assert sandbox["state_hint"] == "HYPOTHESIS_LAB_SANDBOX_VERIFIED"
    assert sandbox["verifier"]["decision"] == "promote_to_strategy_proposal"
    assert sandbox["best_candidate"]["candidate_id"].startswith("learned_residual")


def test_literature_spectral_candidate_maps_to_bounded_hypothesis_sandbox(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
        run_blueprint_sandbox,
    )
    from pde_agent.core.literature_algorithm_lab import (
        compile_algorithm_candidates,
        extract_algorithm_cards,
    )

    markdown = tmp_path / "spectral_operator.md"
    markdown.write_text(
        "\n".join(
            [
                "# Spectral Stabilization for Neural Operator Rollouts",
                "## Abstract",
                "We damp high frequency modes to stabilize Burgers rollouts.",
                "## Algorithm",
                "Claim: Spectral rollout stabilization can reduce late-horizon Burgers drift.",
                "Mechanism: Damp high-frequency Fourier modes during rollout postprocessing.",
                "Operation: Apply bounded spectral smoothing to predicted futures.",
                "Hook: spectral_rollout_stabilization",
                "Observable: late RMSE improves without changing the context prefix",
                "Risk: spectral damping can oversmooth shocks",
                "Minimal Experiment: run a bounded spectral rollout sandbox",
                "$$ u_t + u u_x = \\nu u_{xx} $$",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_spectral_operator",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
            }
        ),
        encoding="utf-8",
    )
    cards_report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )
    candidates_report = compile_algorithm_candidates(
        algorithm_cards_path=cards_report["algorithm_cards_path"],
        output_path=tmp_path / "algorithm_candidates.json",
        force=False,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[candidates_report["algorithm_candidates_path"]],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["implementation_hooks"] == ["linear_trend_spectral"]
    assert blueprints["blueprints"][0]["candidate_family"] == "linear_trend_spectral"

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    context = np.repeat(np.sin(x)[None, None, :], 10, axis=1).astype(np.float32)
    target = np.concatenate(
        [context, np.repeat(context[:, -1:, :], 190, axis=1)],
        axis=1,
    )
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    sandbox = run_blueprint_sandbox(
        blueprint=blueprints["blueprints"][0],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes_literature_spectral_sandbox",
        force=False,
    )

    assert sandbox["state_hint"] == "HYPOTHESIS_LAB_SANDBOX_VERIFIED"
    assert sandbox["best_candidate"]["candidate_id"].startswith(
        "linear_trend_spectral"
    )
    assert sandbox["verifier"]["metric_results"]["context_preserved"]["passed"]


def test_literature_candidate_promotion_preserves_external_grounding(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pde_agent.core.hypothesis_lab import run_hypothesis_lab
    from pde_agent.core.literature_algorithm_lab import (
        compile_algorithm_candidates,
        extract_algorithm_cards,
    )

    markdown = tmp_path / "residual_operator.md"
    markdown.write_text(
        "\n".join(
            [
                "# Residual-Corrected Neural Operators for Burgers Rollout",
                "## Abstract",
                "We use residual corrections to reduce autoregressive rollout bias.",
                "## Algorithm",
                "Claim: Horizon residual correction can reduce Burgers rollout bias.",
                "Mechanism: Estimate a horizon-conditioned residual from local validation windows.",
                "Operation: Apply a bounded residual correction to a persistence rollout.",
                "Hook: learned_residual_correction",
                "Observable: local sandbox score improves over persistence",
                "Risk: residual correction can overfit a small validation split",
                "Minimal Experiment: run a lightweight residual correction sandbox",
                "$$ u_t + u u_x = \\nu u_{xx} $$",
            ]
        ),
        encoding="utf-8",
    )
    parse_path = tmp_path / "parse.json"
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "MINERU_PARSE_INGESTED",
                "source_id": "source_residual_operator",
                "markdown_path": str(markdown),
                "parse_quality": {"quality": "usable"},
            }
        ),
        encoding="utf-8",
    )
    cards_report = extract_algorithm_cards(
        paper_parse_paths=[parse_path],
        output_path=tmp_path / "algorithm_cards.json",
        force=False,
    )
    candidates_report = compile_algorithm_candidates(
        algorithm_cards_path=cards_report["algorithm_cards_path"],
        output_path=tmp_path / "algorithm_candidates.json",
        force=False,
    )
    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    context = np.repeat(np.sin(x)[None, None, :], 10, axis=1).astype(np.float32)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    target[:, 10:, :] = context[:, -1:, :] + np.float32(0.1)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    report = run_hypothesis_lab(
        manual_report_paths=[],
        algorithm_candidate_paths=[candidates_report["algorithm_candidates_path"]],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "lab",
        submission_id="team_hermes_literature_grounding",
        force=False,
    )

    proposal_path = Path(
        report["strategy_growth"]["artifacts"]["strategy_proposal_path"]
    )
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    assert proposal["source_type"] == "external_retrieval"
    assert proposal["implementation_plan"]["source_grounding"]["source_ids"] == [
        "source_residual_operator"
    ]
    assert proposal["implementation_plan"]["algorithm_candidate_id"].startswith(
        "candidate_card_source_residual_operator"
    )


def test_plan_candidate_sandbox_selects_level_one_for_checkpoint(
    tmp_path: Path,
) -> None:
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


def test_build_executor_taxonomy_marks_fno_architecture_as_design_required(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import build_executor_taxonomy

    report = build_executor_taxonomy(
        output_path=tmp_path / "executor_taxonomy.json",
        force=False,
    )

    assert report["state_hint"] == "EXECUTOR_TAXONOMY_BUILT"
    mapping = report["hook_mappings"]["fourier_neural_operator_architecture"]
    assert mapping["route"] == "requires_executor_design"
    assert mapping["requested_executor_id"] == "bounded_fno_architecture_executor"
    assert mapping["fallback_route"] == "official_checkpoint_family"
    assert report["next_command"] == "create-executor-design-request"


def test_create_executor_design_request_from_blocked_fno_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        create_executor_design_request,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "parent_algorithm_card_id": "card_fno",
                        "candidate_family": "unsupported",
                        "hypothesis": "FNO improves Burgers rollout.",
                        "mechanism": "Fourier kernels learn global modes.",
                        "executor_spec": {
                            "cli_command": "blocked_requires_executor_design",
                            "unsupported_hook": "fourier_neural_operator_architecture",
                            "required_options": [],
                        },
                        "benchmark_plan": {"sandbox_level": 2},
                        "risk_controls": [
                            "blocked until a bounded executor is designed"
                        ],
                        "source_grounding": {
                            "source_ids": ["source_fno"],
                            "algorithm_card_id": "card_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = create_executor_design_request(
        algorithm_candidates_path=candidates_path,
        algorithm_candidate_id="candidate_fno_blocked",
        output_path=tmp_path / "executor_design_request.json",
        force=False,
    )

    assert report["state_hint"] == "EXECUTOR_DESIGN_REQUEST_CREATED"
    assert report["algorithm_candidate_id"] == "candidate_fno_blocked"
    assert report["requested_executor_id"] == "bounded_fno_architecture_executor"
    assert "unsupported_algorithm_hook" in report["reason_codes"]
    assert report["executor_contract"]["inputs"] == [
        "algorithm_candidate_path",
        "validation_windows_path",
        "optional_checkpoint_path",
        "output_dir",
    ]
    assert "sandbox_result_path" in report["executor_contract"]["outputs"]
    assert report["strategy_proposal_fields"]["strategy_id"].startswith(
        "external_fno_executor_design_"
    )
    assert report["next_command"] == "build-blocked-candidate-handoff"


def test_create_bounded_executor_design_spec_from_request(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        create_bounded_executor_design_spec,
    )

    request_path = tmp_path / "executor_design_request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXECUTOR_DESIGN_REQUEST_CREATED",
                "algorithm_candidate_id": "candidate_fno_blocked",
                "candidate_family": "unsupported",
                "requested_executor_id": "bounded_fno_architecture_executor",
                "hypothesis": "FNO improves Burgers rollout.",
                "mechanism": "Fourier kernels learn global modes.",
                "source_grounding": {
                    "source_ids": ["source_fno"],
                    "algorithm_card_id": "card_fno",
                },
                "minimal_experiment": {"level": 2},
                "risk_controls": ["require local sandbox score before promotion"],
            }
        ),
        encoding="utf-8",
    )

    report = create_bounded_executor_design_spec(
        executor_design_request_path=request_path,
        output_path=tmp_path / "bounded_executor_design_spec.json",
        force=False,
    )

    assert report["state_hint"] == "BOUNDED_EXECUTOR_DESIGN_SPEC_CREATED"
    assert report["algorithm_candidate_id"] == "candidate_fno_blocked"
    assert report["requested_executor_id"] == "bounded_fno_architecture_executor"
    assert report["executor_api"]["function_name"] == "predict"
    assert report["executor_api"]["required_args"] == ["context", "candidate", "params"]
    assert "numpy" in report["security_policy"]["allowed_imports"]
    assert "open" in report["security_policy"]["forbidden_calls"]
    assert report["next_command"] == "materialize-bounded-executor-template"


def test_materialize_and_validate_bounded_executor_template(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        materialize_bounded_executor_template,
        validate_bounded_executor_implementation,
    )

    design_path = tmp_path / "bounded_executor_design_spec.json"
    design_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "BOUNDED_EXECUTOR_DESIGN_SPEC_CREATED",
                "algorithm_candidate_id": "candidate_fno_blocked",
                "requested_executor_id": "bounded_fno_architecture_executor",
                "source_grounding": {
                    "source_ids": ["source_fno"],
                    "algorithm_card_id": "card_fno",
                },
                "executor_api": {
                    "function_name": "predict",
                    "required_args": ["context", "candidate", "params"],
                },
                "security_policy": {
                    "allowed_imports": ["numpy", "math", "typing"],
                    "forbidden_calls": ["open", "eval", "exec", "__import__"],
                },
            }
        ),
        encoding="utf-8",
    )

    materialized = materialize_bounded_executor_template(
        design_spec_path=design_path,
        output_dir=tmp_path / "executor",
        force=False,
    )
    audit = validate_bounded_executor_implementation(
        design_spec_path=design_path,
        manifest_path=materialized["manifest_path"],
        executor_path=materialized["executor_path"],
        output_path=tmp_path / "executor_audit.json",
        force=False,
    )

    assert materialized["state_hint"] == "BOUNDED_EXECUTOR_TEMPLATE_MATERIALIZED"
    assert Path(materialized["manifest_path"]).exists()
    assert Path(materialized["executor_path"]).exists()
    assert audit["state_hint"] == "BOUNDED_EXECUTOR_IMPLEMENTATION_VALIDATED"
    assert audit["decision"] == "approved"
    assert audit["issue_count"] == 0
    assert audit["next_command"] == "run-bounded-executor-sandbox"


def test_validate_bounded_executor_rejects_unsafe_import(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        validate_bounded_executor_implementation,
    )

    design_path = tmp_path / "bounded_executor_design_spec.json"
    design_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "BOUNDED_EXECUTOR_DESIGN_SPEC_CREATED",
                "algorithm_candidate_id": "candidate_unsafe",
                "requested_executor_id": "bounded_unsafe_executor",
                "security_policy": {
                    "allowed_imports": ["numpy"],
                    "forbidden_calls": ["open", "eval", "exec", "__import__"],
                },
                "executor_api": {
                    "function_name": "predict",
                    "required_args": ["context", "candidate", "params"],
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"algorithm_candidate_id": "candidate_unsafe"}),
        encoding="utf-8",
    )
    executor_path = tmp_path / "executor.py"
    executor_path.write_text(
        "import os\n\ndef predict(context, candidate, params):\n    return context\n",
        encoding="utf-8",
    )

    audit = validate_bounded_executor_implementation(
        design_spec_path=design_path,
        manifest_path=manifest_path,
        executor_path=executor_path,
        output_path=tmp_path / "audit.json",
        force=False,
    )

    assert audit["decision"] == "rejected"
    assert audit["issue_count"] == 1
    assert audit["issues"][0]["code"] == "disallowed_import"
    assert audit["next_command"] == "revise-bounded-executor"


def test_run_bounded_executor_sandbox_and_record_event(tmp_path: Path) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        materialize_bounded_executor_template,
        record_bounded_executor_sandbox_event,
        run_bounded_executor_sandbox,
    )

    np = __import__("numpy")
    context = np.zeros((2, 10, 256), dtype=np.float32)
    context[:, :, :] = np.linspace(0.0, 1.0, 10, dtype=np.float32)[None, :, None]
    target = np.empty((2, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    target[:, 10:, :] = context[:, -1:, :]
    context_path = _write_hdf5(tmp_path / "context.hdf5", context)
    target_path = _write_hdf5(tmp_path / "target.hdf5", target)
    design_path = tmp_path / "bounded_executor_design_spec.json"
    design_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "BOUNDED_EXECUTOR_DESIGN_SPEC_CREATED",
                "algorithm_candidate_id": "candidate_fno_blocked",
                "requested_executor_id": "bounded_fno_architecture_executor",
                "hypothesis": "Use a bounded executor.",
                "mechanism": "Template persistence baseline.",
                "source_grounding": {
                    "source_ids": ["source_fno"],
                    "algorithm_card_id": "card_fno",
                },
                "executor_api": {
                    "function_name": "predict",
                    "required_args": ["context", "candidate", "params"],
                },
                "security_policy": {
                    "allowed_imports": ["numpy", "math", "typing"],
                    "forbidden_calls": ["open", "eval", "exec", "__import__"],
                },
            }
        ),
        encoding="utf-8",
    )
    materialized = materialize_bounded_executor_template(
        design_spec_path=design_path,
        output_dir=tmp_path / "executor",
        force=False,
    )

    sandbox = run_bounded_executor_sandbox(
        design_spec_path=design_path,
        manifest_path=materialized["manifest_path"],
        executor_path=materialized["executor_path"],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="bounded-smoke",
        force=False,
    )
    event = record_bounded_executor_sandbox_event(
        sandbox_report_path=sandbox["bounded_executor_sandbox_path"],
        output_path=tmp_path / "event.json",
        force=False,
    )

    assert sandbox["state_hint"] == "BOUNDED_EXECUTOR_SANDBOX_RAN"
    assert sandbox["rules_passed"] is True
    assert sandbox["context_preserved"] is True
    assert sandbox["metric_results"]["score_delta_vs_persistence"] == pytest.approx(0.0)
    assert Path(sandbox["artifacts"]["prediction_path"]).exists()
    assert sandbox["next_command"] == "record-bounded-executor-sandbox-event"
    assert event["state_hint"] == "LITERATURE_REGISTRY_EVENT_RECORDED"
    assert event["event_type"] == "candidate_sandbox_verified"
    assert event["algorithm_candidate_id"] == "candidate_fno_blocked"
    assert event["next_command"] == "update-literature-registry"


def test_adapt_fno_architecture_candidate_to_checkpoint_resource(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        adapt_fno_architecture_candidate,
    )

    checkpoint = tmp_path / "1D_Burgers_Sols_Nu0.001_FNO.pt"
    checkpoint.write_bytes(b"checkpoint")
    candidates_path = tmp_path / "algorithm_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "parent_algorithm_card_id": "card_fno",
                        "candidate_family": "unsupported",
                        "hypothesis": "FNO improves Burgers rollout.",
                        "mechanism": "Fourier kernels learn global modes.",
                        "executor_spec": {
                            "cli_command": "blocked_requires_executor_design",
                            "unsupported_hook": "fourier_neural_operator_architecture",
                            "required_options": [],
                        },
                        "source_grounding": {
                            "source_ids": ["source_fno"],
                            "algorithm_card_id": "card_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = adapt_fno_architecture_candidate(
        algorithm_candidates_path=candidates_path,
        algorithm_candidate_id="candidate_fno_blocked",
        checkpoint_path=checkpoint,
        output_path=tmp_path / "adapted_candidate.json",
        force=False,
    )

    assert report["state_hint"] == "FNO_ARCHITECTURE_CANDIDATE_ADAPTED"
    adapted = report["adapted_candidate"]
    assert adapted["candidate_family"] == "official_checkpoint_family"
    assert adapted["executor_spec"]["cli_command"] == "run-checkpoint-family-sandbox"
    assert adapted["resource_bindings"]["checkpoint_paths"] == [str(checkpoint)]
    assert adapted["source_grounding"]["adapter_source_candidate_id"] == (
        "candidate_fno_blocked"
    )
    assert report["next_command"] == "plan-algorithm-candidate-sandbox"


def test_build_blocked_candidate_handoff_exposes_strategy_proposal_fields(
    tmp_path: Path,
) -> None:
    from pde_agent.core.literature_algorithm_lab import (
        build_blocked_candidate_handoff,
    )

    request_path = tmp_path / "executor_design_request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "EXECUTOR_DESIGN_REQUEST_CREATED",
                "algorithm_candidate_id": "candidate_fno_blocked",
                "requested_executor_id": "bounded_fno_architecture_executor",
                "hypothesis": "FNO improves Burgers rollout.",
                "mechanism": "Fourier kernels learn global modes.",
                "source_grounding": {
                    "source_ids": ["source_fno"],
                    "algorithm_card_id": "card_fno",
                },
                "strategy_proposal_fields": {
                    "strategy_id": "external_fno_executor_design_candidate_fno_blocked",
                    "title": "Design bounded FNO executor",
                    "rationale": "FNO improves Burgers rollout.",
                    "implementation_plan": [
                        "Implement a bounded executor without external repo code.",
                        "Evaluate on local Burgers validation windows.",
                    ],
                    "expected_observables": ["higher local judge score"],
                    "known_risks": ["tensor convention mismatch"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_blocked_candidate_handoff(
        executor_design_request_path=request_path,
        output_path=tmp_path / "handoff.json",
        force=False,
    )

    assert report["state_hint"] == "BLOCKED_CANDIDATE_HANDOFF_BUILT"
    assert report["record_strategy_proposal_fields"]["strategy_id"] == (
        "external_fno_executor_design_candidate_fno_blocked"
    )
    assert report["blocked_evidence_summary"]["blocked"] is True
    assert report["blocked_evidence_summary"]["requires_executor"] == (
        "bounded_fno_architecture_executor"
    )
    assert report["next_command"] == "record-strategy-proposal"


def test_literature_algorithm_lab_cli_create_executor_design_request(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    candidates_path = tmp_path / "algorithm_candidates.json"
    output_path = tmp_path / "executor_design_request.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "candidate_family": "unsupported",
                        "hypothesis": "FNO improves Burgers rollout.",
                        "mechanism": "Fourier kernels learn global modes.",
                        "executor_spec": {
                            "cli_command": "blocked_requires_executor_design",
                            "unsupported_hook": "fourier_neural_operator_architecture",
                            "required_options": [],
                        },
                        "risk_controls": [
                            "blocked until a bounded executor is designed"
                        ],
                        "source_grounding": {
                            "source_ids": ["source_fno"],
                            "algorithm_card_id": "card_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "create-executor-design-request",
            "--algorithm-candidates-path",
            str(candidates_path),
            "--algorithm-candidate-id",
            "candidate_fno_blocked",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "EXECUTOR_DESIGN_REQUEST_CREATED"
    assert payload["metrics"]["requested_executor_id"] == (
        "bounded_fno_architecture_executor"
    )
    assert payload["next_hint"] == "build-blocked-candidate-handoff"


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


def test_literature_algorithm_lab_cli_normalize_openalex_sources(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    fixture = tmp_path / "openalex.json"
    output_path = tmp_path / "source_candidates.json"
    fixture.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Neural Operator Paper",
                        "open_access": {"oa_url": "https://example.test/paper.pdf"},
                        "primary_location": {
                            "landing_page_url": "https://example.test/paper"
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "normalize-openalex-sources",
            "--search-results-path",
            str(fixture),
            "--query",
            "neural operator",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "EXTERNAL_SOURCE_SEARCH_NORMALIZED"
    assert payload["metrics"]["source_candidate_count"] == 1
    assert payload["next_hint"] == "download-source-document"


def test_literature_algorithm_lab_cli_submit_mineru_agent_parse(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    import pde_agent.commands.literature_algorithm_lab as command_module

    source_path = tmp_path / "downloaded_source.json"
    output_path = tmp_path / "mineru_task.json"
    source_path.write_text(
        json.dumps(
            {
                "state_hint": "SOURCE_DOCUMENT_DOWNLOADED",
                "source_record": {
                    "source_id": "source",
                    "title": "Paper",
                    "download_url": "https://example.test/paper.pdf",
                    "local_raw_path": str(tmp_path / "paper.pdf"),
                    "content_hash": "abc",
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_submit(**_: object) -> dict[str, object]:
        return {
            "state_hint": "MINERU_PARSE_TASK_SUBMITTED",
            "mineru_task_path": str(output_path),
            "task_id": "task-1",
            "mineru_mode": "agent_url",
            "next_command": "poll-mineru-parse",
        }

    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(command_module, "submit_mineru_parse_task", fake_submit)
    try:
        result = CliRunner().invoke(
            app,
            [
                "submit-mineru-parse",
                "--downloaded-source-path",
                str(source_path),
                "--output-path",
                str(output_path),
                "--mode",
                "agent_url",
            ],
        )
    finally:
        monkeypatch.undo()

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "MINERU_PARSE_TASK_SUBMITTED"
    assert payload["metrics"]["task_id"] == "task-1"
