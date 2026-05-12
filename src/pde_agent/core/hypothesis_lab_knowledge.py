"""Hypothesis Lab knowledge submodule."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from pde_agent.core.hypothesis_lab_utils import (
    EARLY_HANDOFF_CURVES,
    EARLY_REFINEMENT_ANCHORS,
    HypothesisLabError,
    _clamp_float,
    _clamp_int,
    _clamp_task1_retrieval_k,
    _guard_output,
    _normalize_retrieval_refinement_params,
    _read_json,
    _require_state,
    _string_list_from_any,
    _write_json,
    _write_optional_json,
)
from pde_agent.core.hypothesis_lab_knowledge_cards import (
    _checkpoint_card,
    _checkpoint_family,
    _external_blend_card,
    _linear_trend_spectral_card,
    _literature_expansion_report_card,
    _literature_note_card,
    _parse_literature_note,
)
from pde_agent.core.hypothesis_lab_search_policy import (
    _adaptive_hypothesis_lab_search_policy_card,
    _adaptive_plateau_card,
    _adaptive_retrieval_refinement_card,
    _first_log_best_candidate,
    _scientific_log_card,
)
from pde_agent.core.hypothesis_lab_pareto import (
    _algorithm_candidate_cards,
    _executed_algorithm_candidate_evidence,
    _pareto_memory_cards,
    _record_algorithm_candidate_outcome,
    _record_hypothesis_strategy_growth_artifacts,
    _record_pareto_failure_memory,
)

CARDS_STATE_HINT = "HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT"

def build_knowledge_cards(
    *,
    manual_report_paths: list[str | Path],
    adaptive_report_paths: list[str | Path] | None = None,
    failure_memory_paths: list[str | Path] | None = None,
    log_paths: list[str | Path] | None = None,
    checkpoint_paths: list[str | Path] | None = None,
    literature_paths: list[str | Path] | None = None,
    algorithm_candidate_paths: list[str | Path] | None = None,
    output_path: str | Path | None,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    cards: list[dict[str, Any]] = []
    for report_path in manual_report_paths:
        report_file = Path(report_path)
        report = _read_json(report_file)
        seen_candidate_ids: set[str] = set()
        for candidate in _manual_candidates(report):
            candidate_id = str(candidate.get("candidate_id", ""))
            if candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_id)
            family = str(candidate.get("family", ""))
            if family == "linear_trend_spectral":
                cards.append(_linear_trend_spectral_card(report_file, candidate))
            elif family == "external_blend":
                cards.append(_external_blend_card(report_file, candidate))
    for report_path in adaptive_report_paths or []:
        report_file = Path(report_path)
        report = _read_json(report_file)
        card = _adaptive_retrieval_refinement_card(report_file, report)
        if card is None:
            card = _adaptive_hypothesis_lab_search_policy_card(report_file, report)
        if card is None:
            card = _adaptive_plateau_card(report_file, report)
        if card is not None:
            cards.append(card)
    for memory_path in failure_memory_paths or []:
        cards.extend(_pareto_memory_cards(Path(memory_path)))
    for log_path in log_paths or []:
        log_file = Path(log_path)
        card = _scientific_log_card(log_file)
        if card is not None:
            cards.append(card)
    for checkpoint_path in checkpoint_paths or []:
        cards.append(_checkpoint_card(Path(checkpoint_path)))
    for literature_path in literature_paths or []:
        card = _literature_note_card(Path(literature_path))
        if card is not None:
            cards.append(card)
    for candidate_path in algorithm_candidate_paths or []:
        cards.extend(_algorithm_candidate_cards(Path(candidate_path)))
    result = {
        "schema_version": 1,
        "state_hint": CARDS_STATE_HINT,
        "card_count": len(cards),
        "cards": cards,
        "next_command": "generate-hypotheses",
    }
    _write_optional_json(output_file, result)
    return result

def _manual_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    best = report.get("best_candidate")
    if isinstance(best, dict):
        candidates.append(best)
    top = report.get("top_candidates")
    if isinstance(top, list):
        candidates.extend([item for item in top if isinstance(item, dict)])
    return candidates
