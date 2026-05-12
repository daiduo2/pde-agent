"""Audited local hypothesis lab for PDE operator strategy generation."""

from __future__ import annotations

import json
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
from pde_agent.core.hypothesis_lab_anchors import (
    _affine_context_calibration_anchor_prediction,
    _damped_linear_velocity_anchor_prediction,
    _early_handoff_weights,
    _linear_velocity_anchor_prediction,
    _mean_trend_bias_anchor_prediction,
    _phase_aligned_retrieval_anchor_prediction,
    _phase_derivative_hybrid_anchor_prediction,
    _retrieval_derivative_match_anchor_prediction,
    _second_order_velocity_anchor_prediction,
)
from pde_agent.core.hypothesis_lab_scoring import _task1_total_score_proxy
from pde_agent.core.hypothesis_lab_scoring import _sandbox_pareto_promotion_gate
from pde_agent.core.hypothesis_lab_predictions import (
    _base_prediction_for_physics_residual_refinement,
)
from pde_agent.core.hypothesis_lab_pareto import (
    _algorithm_candidate_cards,
    _executed_algorithm_candidate_evidence,
    _pareto_memory_cards,
    _record_algorithm_candidate_outcome,
    _record_hypothesis_strategy_growth_artifacts,
    _record_pareto_failure_memory,
)
from pde_agent.core.hypothesis_lab_candidate_pool import (
    _apply_early_memory_policy,
    _candidate_pool_action,
    _candidate_pool_attribution,
    _candidate_pool_policy_from_evidence,
    _candidate_pool_search_space_policy,
    _candidate_early_rel_mse,
    _dedupe_strings,
    _dedupe_values,
    _directed_retrieval_refinement_params,
    _early_memory_policy,
    _early_memory_rank_key,
    _early_param_sort_key,
    _early_params_match,
    _early_refinement_experiment_plan,
    _early_refinement_signals,
    _filter_retrieval_refinement_params_by_directives,
    _finite_float_or_none,
    _retrieval_refinement_avoid_rules_from_directives,
    _retrieval_refinement_center,
    _retrieval_refinement_experiment_plan,
    _retrieval_refinement_search_directives_from_evidence,
    _should_avoid_exact_memory_params,
)
from pde_agent.core.hypothesis_lab_candidate_params import (
    _default_retrieval_refinement_params,
    _early_refinement_params,
    _early_transition_expert_parameter_grid,
    _early_transition_expert_params,
    _front25_dynamics_candidate_params,
    _local_retrieval_refinement_params,
)
from pde_agent.core.hypothesis_lab_sandbox_runner import (
    _iter_parameter_grid,
    run_blueprint_sandbox,
)
from pde_agent.core.hypothesis_lab_selection import (
    _candidate_selection_key,
    _selection_objective,
)


CARDS_STATE_HINT = "HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT"
HYPOTHESES_STATE_HINT = "HYPOTHESIS_LAB_HYPOTHESES_GENERATED"
BLUEPRINTS_STATE_HINT = "HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED"
VERIFIED_BLUEPRINTS_STATE_HINT = "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED"
SANDBOX_STATE_HINT = "HYPOTHESIS_LAB_SANDBOX_VERIFIED"
RUN_STATE_HINT = "HYPOTHESIS_LAB_RAN"
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


def design_algorithm_blueprints(
    *,
    hypotheses: dict[str, Any],
    output_path: str | Path | None,
    force: bool,
) -> dict[str, Any]:
    from pde_agent.core.literature_algorithm_lab import (
        audit_executable_blueprint_payload,
    )

    _require_state(hypotheses, HYPOTHESES_STATE_HINT)
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    blueprints = []
    for hypothesis in hypotheses.get("hypotheses", []):
        if not isinstance(hypothesis, dict):
            continue
        if hypothesis.get("candidate_family") == "official_checkpoint_calibration":
            blueprints.append(
                {
                    "blueprint_id": "bp_official_checkpoint_calibration_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "official_checkpoint_calibration",
                    "search_mode": "checkpoint_calibration",
                    "checkpoint_candidates": hypothesis.get("checkpoint_candidates", []),
                    "algorithm": {
                        "summary": (
                            "Evaluate official PDEBench checkpoints as a distinct "
                            "model family, then calibrate horizon blending or run "
                            "short fine-tuning only if local judge evidence beats "
                            "postprocessing baselines."
                        ),
                        "pseudocode": [
                            "load official checkpoint through FNO/UNet backend",
                            "roll out first 10 context steps to 200-step prediction",
                            "sweep horizon calibration and blend only under budget",
                            "promote only with multi-window score and runtime evidence",
                        ],
                    },
                    "sandbox": {
                        "allowed_executor": "official_checkpoint_candidate_pipeline",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                        "executable_in_hypothesis_lab": False,
                    },
                    "execution_plan": {
                        "cli_command": "run-checkpoint-family-sandbox",
                        "required_options": [
                            "--checkpoint-path",
                            "--task-id",
                            "--test-path",
                            "--target-path",
                            "--output-dir",
                        ],
                        "optional_options": [
                            "--calibration-weight",
                            "--fno-finetune-steps",
                            "--submission-id",
                            "--dataset-key",
                            "--prediction-dataset-key",
                            "--force",
                        ],
                        "handoff_reason": (
                            "Checkpoint candidates require the audited Torch "
                            "checkpoint adapters and may run short FNO fine-tuning, "
                            "so Hypothesis Lab emits an executable handoff instead "
                            "of running them inside the lightweight numpy sandbox."
                        ),
                    },
                    "next_command": "run-checkpoint-family-sandbox",
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_sem_burgers_rollout":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_sem_burgers_rollout_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_sem_burgers_rollout",
                    "search_mode": "physics_numeric_rollout",
                    "algorithm": {
                        "summary": (
                            "Advance fixed-nu 1D Burgers from the last context "
                            "frame with a lightweight spectral-element style "
                            "collocation executor."
                        ),
                        "pseudocode": [
                            "state = context[:, -1, :]",
                            "for each requested future step:",
                            "  substep RK4 on u_t = -u u_x + nu u_xx",
                            "  compute derivatives with element-local spectral grids",
                            "  blend element interfaces and optionally filter modes",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": [
                            {
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "element_count": 16,
                                "polynomial_order": 8,
                                "substeps": 8,
                                "filter_strength": 0.02,
                                "interface_penalty": 0.02,
                            },
                            {
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "element_count": 32,
                                "polynomial_order": 8,
                                "substeps": 8,
                                "filter_strength": 0.05,
                                "interface_penalty": 0.04,
                            },
                        ]
                    },
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_sem_burgers_rollout",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_physics_residual_refinement":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_physics_residual_refinement_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_physics_residual_refinement",
                    "search_mode": "physics_residual_refinement",
                    "algorithm": {
                        "summary": (
                            "Score a strong Task 1 data-prior trajectory with the "
                            "Burgers residual and apply a bounded one-step physics "
                            "nudge over early and middle horizons."
                        ),
                        "pseudocode": [
                            "base = retrieval_refinement(context, reference_library)",
                            "for t in selected future window:",
                            "  physics_next = u[t-1] + dt*(-u*u_x + nu*u_xx)",
                            "  u[t] = blend(base[t], physics_next, correction_weight)",
                            "record base and candidate Burgers residual summaries",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": [
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                                "correction_weight": 0.0,
                                "stop_step": 57,
                                "residual_spectral_strength": 0.0,
                            },
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                                "correction_weight": 0.02,
                                "stop_step": 57,
                                "residual_spectral_strength": 0.0,
                            },
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                                "correction_weight": 0.05,
                                "stop_step": 105,
                                "residual_spectral_strength": 0.0,
                            },
                        ]
                    },
                    "sandbox": {
                        "allowed_executor": (
                            "controlled_numpy_burgers_residual_refinement"
                        ),
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_pi_deeponet_correction_head":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_pi_deeponet_correction_head_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_pi_deeponet_correction_head",
                    "search_mode": "pi_deeponet_correction_head",
                    "algorithm": {
                        "summary": (
                            "Fit a low-rank branch-trunk correction field over a "
                            "strong Task 1 base trajectory and keep the context "
                            "prefix exact."
                        ),
                        "pseudocode": [
                            "base = retrieval_refinement(context, reference_library)",
                            "branch = statistics(context)",
                            "trunk = polynomial/sinusoidal features(t, x)",
                            "solve ridge regression for correction target - base",
                            "prediction = base + correction_weight * correction",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": [
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "ridge_alpha": 1e-4,
                                "correction_weight": 0.1,
                                "physics_weight": 0.0,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                            },
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "ridge_alpha": 1e-3,
                                "correction_weight": 0.25,
                                "physics_weight": 0.02,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                            },
                        ]
                    },
                    "sandbox": {
                        "allowed_executor": (
                            "controlled_numpy_pi_deeponet_correction_head"
                        ),
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_early_refinement":
            candidate_params = _early_refinement_params(
                hypothesis.get("evidence_params", [])
            )
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_early_refinement_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_early_refinement",
                    "search_mode": "early_segment_objective_search",
                    "algorithm": {
                        "summary": (
                            "Run the retrieval-refinement executor with the early "
                            "segment as a first-class objective, testing short "
                            "horizon anchors such as persistence, retrieval-only, "
                            "and linear-velocity extrapolation while guarding "
                            "middle and late windows."
                        ),
                        "pseudocode": [
                            "center = best retrieval-refinement evidence params",
                            "retrieval = knn_retrieval(context, reference, k, mode)",
                            "anchor = early_anchor(context, retrieval)",
                            "blend first early_blend_steps toward anchor",
                            "score early rel-mse plus middle/late guardrails",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": candidate_params,
                    },
                    "experiment_plan": _early_refinement_experiment_plan(
                        evidence_params=hypothesis.get("evidence_params", []),
                        candidate_budget=len(candidate_params),
                    ),
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_task1_early_refinement",
                        "primary_metric": "task1_early_rel_mse_guarded_score",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_early_transition_expert":
            candidate_params = _early_transition_expert_params(
                hypothesis.get("evidence_params", [])
            )
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_early_transition_expert_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_early_transition_expert",
                    "search_mode": "early_segment_transition_expert_search",
                    "algorithm": {
                        "summary": (
                            "Fit a local reference-conditioned transition map for "
                            "the first 47 predicted frames, using the base "
                            "retrieval-refinement candidate only as the middle and "
                            "late guardrail trajectory."
                        ),
                        "pseudocode": [
                            "base = retrieval_refinement(context, reference, base params)",
                            "neighbors = retrieve reference contexts by last frame and velocity",
                            "expert = local ridge map from context velocity to future delta",
                            "splice expert into frames 10:57 with correction_weight",
                            "score early rel-mse with middle/late guardrails",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": candidate_params,
                    },
                    "experiment_plan": _early_refinement_experiment_plan(
                        evidence_params=hypothesis.get("evidence_params", []),
                        candidate_budget=len(candidate_params),
                    ),
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": (
                            "controlled_numpy_task1_early_transition_expert"
                        ),
                        "primary_metric": "task1_early_rel_mse_guarded_score",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_retrieval_prior":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_retrieval_prior_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_retrieval_prior",
                    "search_mode": "affine_knn_trajectory_prior",
                    "algorithm": {
                        "summary": (
                            "Retrieve full Burgers trajectories whose first 10 "
                            "frames match the query context, then affine-normalize "
                            "their future windows to the query statistics."
                        ),
                        "pseudocode": [
                            "reference = validation trajectories",
                            "neighbors = top-k z-normalized context matches",
                            "future = affine_denormalize(weighted_neighbor_futures)",
                            "prediction = concat(query_context, future)",
                        ],
                    },
                    "parameter_grid": {
                        "k": [8, 12, 20],
                        "mode": ["affine"],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_retrieval_prior",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_retrieval_refinement":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_retrieval_refinement_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_retrieval_refinement",
                    "search_mode": "evidence_focused_coarse_to_fine",
                    "algorithm": {
                        "summary": (
                            "Blend a nearest-neighbor retrieval future with a "
                            "persistence fallback over the early horizon, then "
                            "optionally smooth predicted future frames while "
                            "restoring the context prefix exactly."
                        ),
                        "pseudocode": [
                            "retrieval = knn_retrieval(context, reference, k, mode)",
                            "fallback = persistence(context)",
                            "w = linear_decay(persistence_weight_start, early_blend_steps)",
                            "future = w * fallback_future + (1 - w) * retrieval_future",
                            "future = optional_spectral_smooth(future)",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": _default_retrieval_refinement_params(),
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_retrieval_refinement",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_retrieval_refinement_local_search":
            candidate_params = _local_retrieval_refinement_params(
                hypothesis.get("evidence_params", [])
            )
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_retrieval_refinement_local_search_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_retrieval_refinement_local_search",
                    "search_mode": "evidence_centered_local_search",
                    "algorithm": {
                        "summary": (
                            "Search tightly around the promoted Task 1 retrieval "
                            "refinement, preserving the same bounded NumPy executor "
                            "while perturbing the strongest observed parameters."
                        ),
                        "pseudocode": [
                            "center = best verified sandbox retrieval-refinement params",
                            "sweep nearby k, early blend, persistence weight, smoothing",
                            "retain raw/delta sentinels at the center",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": candidate_params,
                    },
                    "experiment_plan": _retrieval_refinement_experiment_plan(
                        evidence_params=hypothesis.get("evidence_params", []),
                        candidate_budget=len(candidate_params),
                    ),
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_retrieval_refinement",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "learned_residual_correction":
            blueprints.append(
                {
                    "blueprint_id": "bp_learned_residual_correction_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "learned_residual_correction",
                    "search_mode": "horizon_mean_residual",
                    "algorithm": {
                        "summary": (
                            "Fit a horizon-wise mean residual over local windows and "
                            "apply a shrinkage factor to persistence or checkpoint "
                            "rollouts while preserving context steps."
                        ),
                        "pseudocode": [
                            "baseline = persistence(context)",
                            "residual[t, x] = mean(target - baseline, axis=samples)",
                            "future = baseline + shrinkage * residual",
                            "future = optional spectral_smooth(future, strength)",
                        ],
                    },
                    "parameter_grid": {
                        "residual_shrinkage": [0.25, 0.5, 0.75],
                        "residual_spectral_strength": [0.0, 0.5],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_residual_operator",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_front25_dynamics":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_front25_dynamics_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_front25_dynamics",
                    "search_mode": "front25_short_dynamics_search",
                    "algorithm": {
                        "summary": (
                            "Fit a bounded front-window dynamics source from "
                            "reference trajectories and apply it only to the "
                            "early predicted segment."
                        ),
                        "pseudocode": [
                            "match legal reference trajectories to context",
                            "fit short-horizon velocity or gradient features",
                            "replace only the front horizon with guarded dynamics",
                            "preserve the first 10 context frames exactly",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": _front25_dynamics_candidate_params(
                            hypothesis.get("evidence_params", [])
                        ),
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_task1_front25_dynamics",
                        "primary_metric": "task1_early_rel_mse_guarded_score",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "linear_trend_spectral":
            blueprints.append(
                {
                    "blueprint_id": "bp_linear_trend_spectral_grid_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "linear_trend_spectral",
                    "search_mode": "evidence_refine",
                    "algorithm": {
                        "summary": (
                            "Estimate a damped negative linear trend from the "
                            "context window and optionally apply spectral smoothing "
                            "to the predicted future."
                        ),
                        "pseudocode": [
                            "slope = (u[-1] - u[-1-window]) / window",
                            "future[t] = u[-1] + gain * t * exp(-t/decay) * slope",
                            "future = spectral_smooth(future, strength)",
                        ],
                    },
                    "parameter_grid": {
                        "slope_window": [4, 5, 6],
                        "gain": [-0.15, -0.125, -0.1, -0.075],
                        "decay": [40.0, 50.0, 60.0, 70.0],
                        "spectral_strength": [0.5, 0.75, 1.0],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_operator",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "linear_trend_spectral_escape":
            blueprints.append(
                {
                    "blueprint_id": "bp_linear_trend_spectral_escape_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "linear_trend_spectral",
                    "search_mode": "plateau_escape",
                    "algorithm": {
                        "summary": (
                            "Use the same controlled damped-trend spectral "
                            "operator, but search outside the verified blueprint's "
                            "local basin after repeated plateau evidence."
                        ),
                        "pseudocode": [
                            "slope = (u[-1] - u[-1-window]) / window",
                            "future[t] = u[-1] + gain * t * exp(-t/decay) * slope",
                            "future = optional spectral_smooth(future, strength)",
                        ],
                    },
                    "parameter_grid": {
                        "slope_window": [2, 3, 4, 5, 8],
                        "gain": [-0.2, -0.15, -0.1, -0.075, -0.05, -0.025],
                        "decay": [20.0, 40.0, 70.0, 100.0],
                        "spectral_strength": [0.0, 0.25, 0.5, 1.0, 1.5],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_operator",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "linear_trend_late_spectral_ramp":
            blueprints.append(
                {
                    "blueprint_id": "bp_linear_trend_late_spectral_ramp_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "linear_trend_spectral",
                    "search_mode": "late_spectral_ramp",
                    "algorithm": {
                        "summary": (
                            "Reuse the plateau_escape damped trend candidate, then "
                            "apply extra spectral smoothing only to later future "
                            "steps to target long-horizon RMSE without changing "
                            "the context prefix."
                        ),
                        "pseudocode": [
                            "future = damped_linear_trend(context, base_params)",
                            "future = spectral_smooth(future, base_strength)",
                            "late_future = spectral_smooth(future, late_strength)",
                            "future[t >= late_start] = late_future[t >= late_start]",
                        ],
                    },
                    "parameter_grid": {
                        "slope_window": [5],
                        "gain": [-0.075, -0.05],
                        "decay": [70.0, 100.0],
                        "spectral_strength": [1.5],
                        "late_spectral_strength": [1.75, 2.0, 2.25],
                        "late_start": [60, 80, 100],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_operator",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
    audit = audit_executable_blueprint_payload(
        {
            "schema_version": 1,
            "state_hint": BLUEPRINTS_STATE_HINT,
            "blueprints": blueprints,
        }
    )
    result = {
        "schema_version": 1,
        "state_hint": BLUEPRINTS_STATE_HINT,
        "blueprint_count": len(blueprints),
        "blueprints": blueprints,
        "executable_blueprint_audit": {
            "state_hint": audit["state_hint"],
            "decision": audit["decision"],
            "issue_count": audit["issue_count"],
            "issues": audit["issues"],
            "normalized_blueprint_count": audit["normalized_blueprint_count"],
            "next_command": audit["next_command"],
        },
        "next_command": "run-blueprint-sandbox",
    }
    _write_optional_json(output_file, result)
    return result


def run_hypothesis_lab(
    *,
    manual_report_paths: list[str | Path],
    adaptive_report_paths: list[str | Path] | None = None,
    log_paths: list[str | Path] | None = None,
    checkpoint_paths: list[str | Path] | None = None,
    literature_paths: list[str | Path] | None = None,
    algorithm_candidate_paths: list[str | Path] | None = None,
    failure_memory_paths: list[str | Path] | None = None,
    failure_memory_output_path: str | Path | None = None,
    context_path: str | Path,
    target_path: str | Path,
    reference_path: str | Path | None = None,
    reference_sample_limit: int | None = None,
    output_dir: str | Path,
    submission_id: str,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_path = root / "hypothesis_lab_report.json"
    _guard_output(report_path, force=force)
    root.mkdir(parents=True, exist_ok=True)
    cards = build_knowledge_cards(
        manual_report_paths=manual_report_paths,
        adaptive_report_paths=adaptive_report_paths,
        log_paths=log_paths,
        checkpoint_paths=checkpoint_paths,
        literature_paths=literature_paths,
        algorithm_candidate_paths=algorithm_candidate_paths,
        failure_memory_paths=failure_memory_paths,
        output_path=root / "knowledge_cards.json",
        force=force,
    )
    hypotheses = generate_hypotheses(
        cards=cards,
        output_path=root / "hypotheses.json",
        force=force,
    )
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=root / "blueprints.json",
        force=force,
    )
    if not blueprints["blueprints"]:
        raise HypothesisLabError("no executable blueprints generated")
    executable_blueprint = _first_executable_blueprint(blueprints["blueprints"])
    if executable_blueprint is None:
        raise HypothesisLabError("no sandbox-executable blueprints generated")
    sandbox = run_blueprint_sandbox(
        blueprint=executable_blueprint,
        context_path=context_path,
        target_path=target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        output_dir=root / "sandbox",
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    verified_blueprints_path = root / "verified_blueprints.json"
    verified_blueprints = _verified_blueprints(
        blueprints=blueprints,
        sandbox=sandbox,
    )
    _write_json(verified_blueprints_path, verified_blueprints)
    candidate_outcome = _record_algorithm_candidate_outcome(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
        sandbox=sandbox,
        failure_memory_output_path=failure_memory_output_path,
        force=force,
    )
    pareto_memory = _record_pareto_failure_memory(
        sandbox=sandbox,
        failure_memory_output_path=failure_memory_output_path,
        algorithm_candidate_evidence=_executed_algorithm_candidate_evidence(
            cards=cards,
            hypotheses=hypotheses,
            executable_blueprint=executable_blueprint,
        ),
        force=force,
    )
    strategy_growth = _record_hypothesis_strategy_growth_artifacts(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
        sandbox=sandbox,
        output_dir=root,
        force=force,
    )
    report = {
        "schema_version": 1,
        "state_hint": RUN_STATE_HINT,
        "cards": {
            "count": cards["card_count"],
            "path": str(root / "knowledge_cards.json"),
        },
        "hypotheses": {
            "count": hypotheses["hypothesis_count"],
            "path": str(root / "hypotheses.json"),
        },
        "blueprints": {
            "count": blueprints["blueprint_count"],
            "path": str(root / "blueprints.json"),
        },
        "sandbox": sandbox,
        "artifacts": {
            "hypothesis_lab_report_path": str(report_path),
            "sandbox_report_path": sandbox["artifacts"]["sandbox_report_path"],
            "best_submission_dir": sandbox["artifacts"]["best_submission_dir"],
            "verified_blueprint_path": str(verified_blueprints_path),
        },
        "next_command": strategy_growth["next_command"]
        if strategy_growth is not None
        else sandbox["next_command"],
    }
    if strategy_growth is not None:
        report["strategy_growth"] = strategy_growth
        report["artifacts"].update(strategy_growth["artifacts"])
    if candidate_outcome is not None:
        report["candidate_outcome"] = candidate_outcome
    if pareto_memory is not None:
        report["pareto_memory"] = pareto_memory
    _write_json(report_path, report)
    return report


def _first_executable_blueprint(blueprints: list[Any]) -> dict[str, Any] | None:
    for blueprint in blueprints:
        if not isinstance(blueprint, dict):
            continue
        sandbox = blueprint.get("sandbox")
        if not isinstance(sandbox, dict):
            continue
        if sandbox.get("executable_in_hypothesis_lab") is False:
            continue
        if blueprint.get("candidate_family") in {
            "linear_trend_spectral",
            "learned_residual_correction",
            "task1_retrieval_prior",
            "task1_retrieval_refinement",
            "task1_retrieval_refinement_local_search",
            "task1_early_refinement",
            "task1_early_transition_expert",
            "task1_sem_burgers_rollout",
            "task1_physics_residual_refinement",
            "task1_pi_deeponet_correction_head",
            "task1_front25_dynamics",
        }:
            return blueprint
    return None


def _verified_blueprints(
    *,
    blueprints: dict[str, Any],
    sandbox: dict[str, Any],
) -> dict[str, Any]:
    verified = json.loads(json.dumps(blueprints))
    verified["state_hint"] = VERIFIED_BLUEPRINTS_STATE_HINT
    verified["source_sandbox_report_path"] = sandbox["artifacts"]["sandbox_report_path"]
    best = sandbox["best_candidate"]
    for blueprint in verified.get("blueprints", []):
        if not isinstance(blueprint, dict):
            continue
        if blueprint.get("blueprint_id") != sandbox.get("blueprint_id"):
            continue
        seed = {
            "params": best["params"],
            "source": "sandbox_best",
            "candidate_id": best["candidate_id"],
            "task1_total_score_proxy": best["task1_total_score_proxy"],
            "segment_score": best["segment_score"],
        }
        existing = blueprint.get("seed_params")
        if not isinstance(existing, list):
            existing = []
        blueprint["seed_params"] = [seed, *_dedupe_seed_params(existing, seed["params"])]
    return verified


def _dedupe_seed_params(
    rows: list[Any],
    first_params: dict[str, Any],
) -> list[dict[str, Any]]:
    deduped = []
    first_key = json.dumps(first_params, sort_keys=True)
    seen = {first_key}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("params"), dict):
            continue
        key = json.dumps(row["params"], sort_keys=True)
        if key in seen:
            continue
        deduped.append(row)
        seen.add(key)
    return deduped
