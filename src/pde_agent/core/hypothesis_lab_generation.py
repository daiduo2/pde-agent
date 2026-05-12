"""Hypothesis Lab generation submodule."""

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
from pde_agent.core.hypothesis_lab_evidence import (
    _merge_card_list,
    _merge_ranked_evidence_row,
    _normalize_evidence_params,
    _rank_hypotheses,
    _ranked_evidence_params,
)

CARDS_STATE_HINT = "HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT"
HYPOTHESES_STATE_HINT = "HYPOTHESIS_LAB_HYPOTHESES_GENERATED"

def generate_hypotheses(
    *,
    cards: dict[str, Any],
    output_path: str | Path | None,
    force: bool,
) -> dict[str, Any]:
    _require_state(cards, CARDS_STATE_HINT)
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    grouped_cards: dict[str, list[dict[str, Any]]] = {}
    for card in cards.get("cards", []):
        if not isinstance(card, dict):
            continue
        hooks = set(card.get("implementation_hooks", []))
        if "linear_trend_spectral" in hooks:
            grouped_cards.setdefault("linear_trend_spectral", []).append(card)
        if "linear_trend_spectral_escape" in hooks:
            grouped_cards.setdefault("linear_trend_spectral_escape", []).append(card)
        if "linear_trend_late_spectral_ramp" in hooks:
            grouped_cards.setdefault("linear_trend_late_spectral_ramp", []).append(card)
        if "external_residual_blend" in hooks:
            grouped_cards.setdefault("external_residual_blend", []).append(card)
        if "official_checkpoint_calibration" in hooks:
            grouped_cards.setdefault("official_checkpoint_calibration", []).append(card)
        if "learned_residual_correction" in hooks:
            grouped_cards.setdefault("learned_residual_correction", []).append(card)
        if "task1_retrieval_prior" in hooks:
            grouped_cards.setdefault("task1_retrieval_prior", []).append(card)
        if "task1_retrieval_refinement_local_search" in hooks:
            grouped_cards.setdefault(
                "task1_retrieval_refinement_local_search", []
            ).append(card)
        if "task1_retrieval_refinement" in hooks:
            grouped_cards.setdefault("task1_retrieval_refinement", []).append(card)
        if "task1_early_refinement" in hooks:
            grouped_cards.setdefault("task1_early_refinement", []).append(card)
        if "task1_early_transition_expert" in hooks:
            grouped_cards.setdefault("task1_early_transition_expert", []).append(card)
        if "task1_physics_residual_refinement" in hooks:
            grouped_cards.setdefault("task1_physics_residual_refinement", []).append(
                card
            )
        if "task1_pi_deeponet_correction_head" in hooks:
            grouped_cards.setdefault("task1_pi_deeponet_correction_head", []).append(
                card
            )
        if "task1_sem_burgers_rollout" in hooks:
            grouped_cards.setdefault("task1_sem_burgers_rollout", []).append(card)
        if "task1_front25_dynamics" in hooks:
            grouped_cards.setdefault("task1_front25_dynamics", []).append(card)
    hypotheses = []
    if "task1_front25_dynamics" in grouped_cards:
        source_cards = grouped_cards["task1_front25_dynamics"]
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_front25_dynamics_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "task1_front25_dynamics",
                "mechanism": (
                    "Fit a bounded front-window dynamics source from legal "
                    "reference trajectories and splice only the first predicted "
                    "steps into the rollout to target early Rel-MSE."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "evidence_params": _ranked_evidence_params(source_cards),
                "priority": -10,
            }
        )
    if "task1_pi_deeponet_correction_head" in grouped_cards:
        source_cards = grouped_cards["task1_pi_deeponet_correction_head"]
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_pi_deeponet_correction_head_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "task1_pi_deeponet_correction_head",
                "mechanism": (
                    "Fit a small branch-trunk correction head over a strong Task 1 "
                    "prior and optionally blend a Burgers residual nudge, so the "
                    "agent can test PI-DeepONet-style residual correction without "
                    "committing to a heavy model-training path."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "priority": -9,
            }
        )
    if "task1_physics_residual_refinement" in grouped_cards:
        source_cards = grouped_cards["task1_physics_residual_refinement"]
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_physics_residual_refinement_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "task1_physics_residual_refinement",
                "mechanism": (
                    "Use Burgers residual as a target-free diagnostic and nudge "
                    "a strong data-prior trajectory toward one-step PDE consistency "
                    "without replacing the retrieval prior."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "priority": -8,
            }
        )
    if "task1_sem_burgers_rollout" in grouped_cards:
        source_cards = grouped_cards["task1_sem_burgers_rollout"]
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_sem_burgers_rollout_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "task1_sem_burgers_rollout",
                "mechanism": (
                    "Use a fixed-nu physics rollout for Task 1 Burgers, advancing "
                    "the last context frame with element-local spectral derivatives "
                    "instead of relying only on retrieval or persistence."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "priority": -7,
            }
        )
    if "task1_early_transition_expert" in grouped_cards:
        source_cards = grouped_cards["task1_early_transition_expert"]
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_early_transition_expert_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "task1_early_transition_expert",
                "mechanism": (
                    "Fit a lightweight local transition expert from legal "
                    "reference trajectories, then splice only the first "
                    "predicted segment into a guarded base trajectory."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "evidence_params": _ranked_evidence_params(source_cards),
                "priority": -6,
            }
        )
    if "task1_early_refinement" in grouped_cards:
        source_cards = grouped_cards["task1_early_refinement"]
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_early_refinement_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "task1_early_refinement",
                "mechanism": (
                    "Treat the early Burgers rollout window as an explicit "
                    "optimization objective, using retrieval-refinement evidence "
                    "as the base executor while sweeping early anchors and short "
                    "horizon blend weights under middle/late guardrails."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "evidence_params": _ranked_evidence_params(source_cards),
                "priority": -6,
            }
        )
    retrieval_refinement_cards = [
        *grouped_cards.get("task1_retrieval_refinement", []),
        *grouped_cards.get("task1_retrieval_prior", []),
    ]
    if (
        "task1_retrieval_refinement_local_search" in grouped_cards
        or "task1_retrieval_refinement" in grouped_cards
    ):
        source_cards = grouped_cards.get(
            "task1_retrieval_refinement_local_search",
            grouped_cards.get("task1_retrieval_refinement", []),
        )
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_retrieval_refinement_local_search_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "task1_retrieval_refinement_local_search",
                "mechanism": (
                    "Use the promoted retrieval-refinement evidence as the center "
                    "of a narrow local search over neighbor count, early blend "
                    "length, persistence anchor weight, and smoothing strength."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "evidence_params": _ranked_evidence_params(source_cards),
                "priority": -5,
            }
        )
    if retrieval_refinement_cards:
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_retrieval_refinement_001",
                "source_card_ids": [card["card_id"] for card in retrieval_refinement_cards],
                "candidate_family": "task1_retrieval_refinement",
                "mechanism": (
                    "Refine the Task 1 retrieval prior by sweeping retrieval modes, "
                    "larger neighbor sets, early-horizon persistence blending, and "
                    "optional spectral smoothing to target weak early/middle windows "
                    "without losing late-window stability."
                ),
                "preconditions": _merge_card_list(
                    retrieval_refinement_cards,
                    "preconditions",
                ),
                "expected_observables": _merge_card_list(
                    retrieval_refinement_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(
                    retrieval_refinement_cards,
                    "known_risks",
                ),
                "priority": -4,
            }
        )
    if "task1_retrieval_prior" in grouped_cards:
        source_cards = grouped_cards["task1_retrieval_prior"]
        hypotheses.append(
            {
                "hypothesis_id": "h_task1_retrieval_prior_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "task1_retrieval_prior",
                "mechanism": (
                    "Use affine-normalized nearest-neighbor trajectory retrieval "
                    "from a Task 1 reference library to escape the persistence "
                    "and postprocessing plateau without model training."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "priority": -3,
            }
        )
    if "official_checkpoint_calibration" in grouped_cards:
        source_cards = grouped_cards["official_checkpoint_calibration"]
        hypotheses.append(
            {
                "hypothesis_id": "h_burgers_official_checkpoint_calibration_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "official_checkpoint_calibration",
                "mechanism": (
                    "Use official PDEBench FNO/UNet checkpoints as a new model "
                    "family, then let the agent calibrate rollout horizon, blending, "
                    "or short fine-tuning instead of staying in persistence-only "
                    "postprocessing."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "checkpoint_candidates": [
                    card["evidence"]
                    for card in source_cards
                    if isinstance(card.get("evidence"), dict)
                ],
                "priority": -2,
            }
        )
    if "learned_residual_correction" in grouped_cards:
        source_cards = grouped_cards["learned_residual_correction"]
        hypotheses.append(
            {
                "hypothesis_id": "h_burgers_learned_residual_correction_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "learned_residual_correction",
                "mechanism": (
                    "Fit a lightweight horizon-conditioned residual correction "
                    "over persistence or checkpoint rollouts, then shrink the "
                    "correction to reduce systematic early/middle Rel-MSE without "
                    "modifying the first 10 context steps."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "priority": -1,
            }
        )
    if "linear_trend_late_spectral_ramp" in grouped_cards:
        source_cards = grouped_cards["linear_trend_late_spectral_ramp"]
        hypotheses.append(
            {
                "hypothesis_id": "h_burgers_late_spectral_ramp_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "linear_trend_late_spectral_ramp",
                "mechanism": (
                    "After plateau_escape has already selected strong global "
                    "spectral smoothing, preserve the early rollout behavior and "
                    "apply an additional late-window spectral ramp only after the "
                    "middle horizon where the score is dominated by long-window RMSE."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "evidence_params": _ranked_evidence_params(source_cards),
                "priority": -1,
            }
        )
    if "linear_trend_spectral_escape" in grouped_cards:
        source_cards = grouped_cards["linear_trend_spectral_escape"]
        hypotheses.append(
            {
                "hypothesis_id": "h_burgers_trend_spectral_plateau_escape_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "linear_trend_spectral_escape",
                "mechanism": (
                    "When the verified damped-trend spectral blueprint plateaus, "
                    "escape its local basin by widening slope windows, allowing "
                    "near-zero damping, and testing both unsmoothed and stronger "
                    "spectral variants under the same controlled NumPy executor."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "evidence_params": _ranked_evidence_params(source_cards),
                "priority": 0,
            }
        )
    if "linear_trend_spectral" in grouped_cards:
        source_cards = grouped_cards["linear_trend_spectral"]
        hypotheses.append(
            {
                "hypothesis_id": "h_burgers_damped_trend_spectral_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "linear_trend_spectral",
                "mechanism": (
                    "Use a small negative trend extrapolation from the context "
                    "window, then damp high-frequency modes to improve late "
                    "Burgers stability without copying future data."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "evidence_params": _ranked_evidence_params(source_cards),
                "priority": 1,
            }
        )
    if "external_residual_blend" in grouped_cards:
        source_cards = grouped_cards["external_residual_blend"]
        hypotheses.append(
            {
                "hypothesis_id": "h_burgers_weak_model_residual_blend_001",
                "source_card_ids": [card["card_id"] for card in source_cards],
                "candidate_family": "external_residual_blend",
                "mechanism": (
                    "Treat a weak neural operator as a low-weight residual "
                    "corrector over persistence rather than as the full rollout."
                ),
                "preconditions": _merge_card_list(source_cards, "preconditions"),
                "expected_observables": _merge_card_list(
                    source_cards,
                    "expected_observables",
                ),
                "known_risks": _merge_card_list(source_cards, "known_risks"),
                "priority": 2,
            }
        )
    result = {
        "schema_version": 1,
        "state_hint": HYPOTHESES_STATE_HINT,
        "hypothesis_count": len(hypotheses),
        "hypotheses": _rank_hypotheses(hypotheses),
        "next_command": "design-algorithm-blueprints",
    }
    _write_optional_json(output_file, result)
    return result
