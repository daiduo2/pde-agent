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
