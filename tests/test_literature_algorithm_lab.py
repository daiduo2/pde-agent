from __future__ import annotations

import json
from pathlib import Path


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
    from pde_agent.core.literature_algorithm_lab import extract_algorithm_cards

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

    card = report["algorithm_cards"][0]
    assert card["implementation_hooks"] == ["fourier_neural_operator_architecture"]
    assert "Fourier" in card["claim"]
    assert "Fourier space" in card["core_mechanism"]
    assert card["minimal_experiment"] == (
        "Design a bounded FNO architecture executor or map the source to an existing checkpoint-family adapter before sandbox execution."
    )


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
