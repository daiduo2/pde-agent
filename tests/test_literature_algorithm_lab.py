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
