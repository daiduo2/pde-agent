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
