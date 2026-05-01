"""Audited local hypothesis lab for PDE operator strategy generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.hermes_literature_expansion import (
    record_hermes_literature_candidate_outcome,
)
from pde_agent.core.hermes_local_judge import judge_hermes_submission
from pde_agent.core.manual_operator_tuning import (
    linear_trend_prediction,
    load_hdf5_tensor,
    make_persistence_prediction,
    score_prediction_arrays,
    spectral_smooth_prediction,
)
from pde_agent.core.strategy_growth import record_strategy_proposal
from pde_agent.core.task1_retrieval_prior import predict_retrieval_prior


CARDS_STATE_HINT = "HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT"
HYPOTHESES_STATE_HINT = "HYPOTHESIS_LAB_HYPOTHESES_GENERATED"
BLUEPRINTS_STATE_HINT = "HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED"
VERIFIED_BLUEPRINTS_STATE_HINT = "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED"
SANDBOX_STATE_HINT = "HYPOTHESIS_LAB_SANDBOX_VERIFIED"
RUN_STATE_HINT = "HYPOTHESIS_LAB_RAN"


class HypothesisLabError(RuntimeError):
    pass


def build_knowledge_cards(
    *,
    manual_report_paths: list[str | Path],
    adaptive_report_paths: list[str | Path] | None = None,
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
            card = _adaptive_plateau_card(report_file, report)
        if card is not None:
            cards.append(card)
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
    hypotheses = []
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
                    "family, then let Hermes calibrate rollout horizon, blending, "
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


def _merge_card_list(cards: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for card in cards:
        raw = card.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, str) or item in seen:
                continue
            values.append(item)
            seen.add(item)
    return values


def _ranked_evidence_params(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for card in cards:
        evidence = card.get("evidence")
        if not isinstance(evidence, dict):
            continue
        params = evidence.get("params")
        if not isinstance(params, dict):
            continue
        normalized = _normalize_evidence_params(params)
        if normalized is None:
            continue
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "params": normalized,
                "source_card_id": card["card_id"],
                "task1_total_fast_proxy": evidence.get("task1_total_fast_proxy"),
                "task1_total_score_proxy": evidence.get("task1_total_score_proxy"),
                "segment_score": evidence.get("segment_score"),
                "search_directives": evidence.get("search_directives", []),
            }
        )
    rows.sort(
        key=lambda item: (
            -float(
                item.get("task1_total_fast_proxy")
                or item.get("task1_total_score_proxy")
            )
            if isinstance(
                item.get("task1_total_fast_proxy")
                or item.get("task1_total_score_proxy"),
                int | float,
            )
            else 0.0,
            str(item["source_card_id"]),
        )
    )
    return rows


def _normalize_evidence_params(params: dict[str, Any]) -> dict[str, Any] | None:
    if {"k", "mode", "early_blend_steps", "persistence_weight_start"}.issubset(
        params
    ):
        return _normalize_retrieval_refinement_params(params)
    if {"slope_window", "gain", "decay", "spectral_strength"}.issubset(params):
        return {
            "slope_window": int(params["slope_window"]),
            "gain": float(params["gain"]),
            "decay": float(params["decay"]),
            "spectral_strength": float(params["spectral_strength"]),
        }
    return None


def _rank_hypotheses(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for hypothesis in hypotheses:
        item = dict(hypothesis)
        family = str(item.get("candidate_family", ""))
        priority = int(item.get("priority", 10))
        novelty_bonus = {
            "official_checkpoint_calibration": 45.0,
            "task1_retrieval_refinement_local_search": 46.0,
            "task1_retrieval_refinement": 44.0,
            "task1_retrieval_prior": 43.0,
            "learned_residual_correction": 40.0,
            "linear_trend_late_spectral_ramp": 25.0,
            "linear_trend_spectral_escape": 20.0,
            "external_residual_blend": 4.0,
            "linear_trend_spectral": 5.0,
        }.get(family, 0.0)
        executable_bonus = 10.0 if family != "official_checkpoint_calibration" else 2.0
        source_bonus = min(10.0, 2.0 * len(item.get("source_card_ids", [])))
        ranking_score = novelty_bonus + executable_bonus + source_bonus - float(priority)
        item["ranking_score"] = ranking_score
        item["ranking_reasons"] = [
            f"novelty_bonus={novelty_bonus:g}",
            f"executable_bonus={executable_bonus:g}",
            f"source_bonus={source_bonus:g}",
            f"priority={priority}",
        ]
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item["ranking_score"]),
            int(item.get("priority", 10)),
            str(item.get("candidate_family", "")),
        )
    )
    return ranked


def design_algorithm_blueprints(
    *,
    hypotheses: dict[str, Any],
    output_path: str | Path | None,
    force: bool,
) -> dict[str, Any]:
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
    result = {
        "schema_version": 1,
        "state_hint": BLUEPRINTS_STATE_HINT,
        "blueprint_count": len(blueprints),
        "blueprints": blueprints,
        "next_command": "run-blueprint-sandbox",
    }
    _write_optional_json(output_file, result)
    return result


def run_blueprint_sandbox(
    *,
    blueprint: dict[str, Any],
    context_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    submission_id: str,
    force: bool,
) -> dict[str, Any]:
    family = str(blueprint.get("candidate_family"))
    if family not in {
        "linear_trend_spectral",
        "learned_residual_correction",
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
        "task1_retrieval_refinement_local_search",
    }:
        raise HypothesisLabError(
            f"unsupported blueprint candidate_family: {blueprint.get('candidate_family')}"
        )
    root = Path(output_dir)
    report_path = root / "hypothesis_lab_sandbox_report.json"
    _guard_output(report_path, force=force)
    root.mkdir(parents=True, exist_ok=True)
    context = load_hdf5_tensor(context_path)
    if context.shape[1] != 10:
        context = context[:, :10, :]
    target = load_hdf5_tensor(target_path)
    baseline_prediction = make_persistence_prediction(context)
    baseline_score = score_prediction_arrays(
        prediction=baseline_prediction,
        target=target,
    )
    rows = []
    for params in _iter_parameter_grid(blueprint["parameter_grid"], family=family):
        start = time.perf_counter()
        prediction = _prediction_from_blueprint_params(
            family=family,
            context=context,
            target=target,
            params=params,
        )
        elapsed = time.perf_counter() - start
        score = score_prediction_arrays(prediction=prediction, target=target)
        rows.append(
            {
                "candidate_id": _candidate_id(params),
                "params": params,
                "metadata": _candidate_metadata(
                    family=family,
                    context=context,
                    target=target,
                    params=params,
                ),
                "elapsed_seconds": elapsed,
                "segment_score": score["segment_score"],
                "task1_prediction_score": score["task1_prediction_score"],
                "task1_total_score_proxy": _task1_total_score_proxy(
                    prediction_score=score["task1_prediction_score"],
                    inference_seconds=elapsed,
                ),
                "segments": score["segments"],
                "_prediction": prediction,
            }
        )
    rows.sort(key=lambda item: float(item["task1_total_score_proxy"]), reverse=True)
    best = rows[0]
    prediction_path = root / "best_prediction.npz"
    np.savez_compressed(prediction_path, prediction=best["_prediction"])
    submission = build_competition_submission(
        task_id=1,
        test_path=context_path,
        output_dir=root / "submission_package",
        submission_id=submission_id,
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=float(best["elapsed_seconds"]),
        force=force,
    )
    judge = judge_hermes_submission(
        submission_dir=submission["artifacts"]["submission_dir"],
        output_path=root / "judge_report.json",
        task1_test_path=context_path,
        task1_target_path=target_path,
    )
    context_preserved = _context_preserved(
        prediction=best["_prediction"],
        context=context,
    )
    baseline_total = _task1_total_score_proxy(
        prediction_score=float(baseline_score["task1_prediction_score"]),
        inference_seconds=0.0,
    )
    delta = (
        float(judge["total_score_proxy"])
        - baseline_total
    )
    metric_results = _sandbox_metric_results(
        judge=judge,
        baseline_score=baseline_score,
        best_segments=best["segments"],
        score_delta=delta,
        context_preserved=context_preserved,
        inference_seconds=float(best["elapsed_seconds"]),
        max_inference_seconds=float(blueprint["sandbox"]["max_inference_seconds"]),
        promotion_min_score_delta=float(blueprint["sandbox"]["promotion_min_score_delta"]),
    )
    multiview_judge = _task1_multiview_local_judge(
        prediction=best["_prediction"],
        baseline_prediction=baseline_prediction,
        target=target,
    )
    verifier = {
        "decision": "promote_to_strategy_proposal"
        if _sandbox_metrics_pass(metric_results)
        else "reject_or_revise",
        "rules_passed": bool(judge["rules_passed"]),
        "context_preserved": context_preserved,
        "score_delta_vs_baseline": delta,
        "baseline_segment_score": baseline_score["segment_score"],
        "best_segment_score": judge["tasks"]["1"]["score"]["segment_score"],
        "inference_time": float(best["elapsed_seconds"]),
        "judge_decision": judge["decision"],
        "metric_results": metric_results,
        "multiview_judge": multiview_judge,
        "promotion_score": _sandbox_promotion_score(metric_results),
    }
    serializable_rows = [{k: v for k, v in row.items() if k != "_prediction"} for row in rows]
    report = {
        "schema_version": 1,
        "state_hint": SANDBOX_STATE_HINT,
        "blueprint_id": blueprint["blueprint_id"],
        "baseline": {
            "segment_score": baseline_score["segment_score"],
            "task1_prediction_score": baseline_score["task1_prediction_score"],
            "task1_total_score_proxy": _task1_total_score_proxy(
                prediction_score=baseline_score["task1_prediction_score"],
                inference_seconds=0.0,
            ),
        },
        "best_candidate": {k: v for k, v in best.items() if k != "_prediction"},
        "verifier": verifier,
        "top_candidates": serializable_rows[:20],
        "artifacts": {
            "sandbox_report_path": str(report_path),
            "best_prediction_path": str(prediction_path),
            "best_submission_dir": submission["artifacts"]["submission_dir"],
            "judge_report_path": str(root / "judge_report.json"),
        },
        "next_command": "record-strategy-proposal"
        if verifier["decision"] == "promote_to_strategy_proposal"
        else "revise-hypothesis",
    }
    _write_json(report_path, report)
    return report


def run_hypothesis_lab(
    *,
    manual_report_paths: list[str | Path],
    adaptive_report_paths: list[str | Path] | None = None,
    log_paths: list[str | Path] | None = None,
    checkpoint_paths: list[str | Path] | None = None,
    literature_paths: list[str | Path] | None = None,
    algorithm_candidate_paths: list[str | Path] | None = None,
    failure_memory_output_path: str | Path | None = None,
    context_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    submission_id: str,
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
        output_dir=root / "sandbox",
        submission_id=submission_id,
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


def _linear_trend_spectral_card(
    report_file: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    return {
        "card_id": f"manual_{candidate.get('candidate_id', 'linear_trend_spectral')}",
        "source_type": "manual_tuning_report",
        "source_path": str(report_file),
        "claim": (
            "Damped negative linear trend extrapolation followed by spectral "
            "smoothing can improve long-horizon 1D Burgers score over persistence."
        ),
        "procedural_heuristic": (
            "Use the last context slope with a small negative gain, exponential "
            "decay, and spectral smoothing on predicted future steps only."
        ),
        "preconditions": [
            "1D Burgers",
            "persistence plateau",
            "late-window RMSE dominates local objective",
        ],
        "implementation_hooks": ["linear_trend_spectral"],
        "expected_observables": [
            "higher segment score",
            "lower late RMSE",
            "context copy remains exact",
        ],
        "known_risks": [
            "can overfit validation viscosity",
            "early Rel-MSE may remain poor",
            "strong smoothing can suppress shocks",
        ],
        "evidence_status": "tested",
        "evidence": {
            "candidate_id": candidate.get("candidate_id"),
            "params": params,
            "segment_score": candidate.get("segment_score"),
            "task1_total_fast_proxy": candidate.get("task1_total_fast_proxy"),
            "late_rmse": candidate.get("segments", {}).get("late", {}).get("rmse")
            if isinstance(candidate.get("segments"), dict)
            else None,
        },
    }


def _external_blend_card(report_file: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": f"manual_{candidate.get('candidate_id', 'external_blend')}",
        "source_type": "manual_tuning_report",
        "source_path": str(report_file),
        "claim": (
            "A weak neural operator can still provide useful low-weight residual "
            "signal when blended with persistence."
        ),
        "procedural_heuristic": (
            "Blend only a small fraction of the weak model rollout into the fast "
            "baseline and account for full upstream inference time."
        ),
        "preconditions": ["weak model exists", "runtime budget remains feasible"],
        "implementation_hooks": ["external_residual_blend"],
        "expected_observables": ["lower late RMSE", "runtime tradeoff visible"],
        "known_risks": ["upstream model inference can erase score gains"],
        "evidence_status": "seeded",
        "evidence": {
            "candidate_id": candidate.get("candidate_id"),
            "params": candidate.get("params"),
            "segment_score": candidate.get("segment_score"),
            "task1_total_fast_proxy": candidate.get("task1_total_fast_proxy"),
        },
    }


def _checkpoint_card(checkpoint_file: Path) -> dict[str, Any]:
    if not checkpoint_file.exists():
        raise HypothesisLabError(f"checkpoint artifact not found: {checkpoint_file}")
    model_family = _checkpoint_family(checkpoint_file)
    return {
        "card_id": f"checkpoint_{checkpoint_file.stem}",
        "source_type": "official_checkpoint",
        "source_path": str(checkpoint_file),
        "claim": (
            f"Official PDEBench {model_family} checkpoint should be evaluated as "
            "a separate model family when persistence-style postprocessing plateaus."
        ),
        "procedural_heuristic": (
            "Load the checkpoint through the audited backend, run local validation "
            "rollout, then calibrate horizon blending or short fine-tuning only "
            "if runtime remains under the 2 minute inference budget."
        ),
        "preconditions": [
            "official checkpoint file is available",
            "Hermes postprocessing family has plateaued",
            "torch backend is available when executing the checkpoint pipeline",
        ],
        "implementation_hooks": ["official_checkpoint_calibration"],
        "expected_observables": [
            "candidate beats persistence-like postprocessing on early or middle Rel-MSE",
            "inference time remains under 120 seconds",
            "context prefix remains exact",
        ],
        "known_risks": [
            "checkpoint schema may not match local adapter",
            "UNet checkpoint can exceed runtime budget",
            "short fine-tuning can overfit validation windows",
        ],
        "evidence_status": "available",
        "evidence": {
            "checkpoint_path": str(checkpoint_file),
            "model_family": model_family,
        },
    }


def _checkpoint_family(checkpoint_file: Path) -> str:
    name = checkpoint_file.name.lower()
    if "unet" in name:
        return "UNet"
    if "fno" in name:
        return "FNO"
    return "NeuralOperator"


def _literature_note_card(note_file: Path) -> dict[str, Any] | None:
    if not note_file.exists():
        raise HypothesisLabError(f"literature artifact not found: {note_file}")
    text = note_file.read_text(encoding="utf-8")
    expansion_card = _literature_expansion_report_card(note_file, text)
    if expansion_card is not None:
        return expansion_card
    fields = _parse_literature_note(text)
    hook = fields.get("hook")
    claim = fields.get("claim")
    if not hook or not claim:
        return None
    return {
        "card_id": f"literature_{note_file.stem}",
        "source_type": "literature_note",
        "source_path": str(note_file),
        "claim": claim,
        "procedural_heuristic": fields.get("mechanism", claim),
        "preconditions": ["local literature note has an implementation hook"],
        "implementation_hooks": [hook],
        "expected_observables": [fields.get("observable", "sandbox score improves")],
        "known_risks": [fields.get("risk", "literature claim may not transfer")],
        "evidence_status": "seeded",
        "evidence": {
            "title": fields.get("title"),
            "hook": hook,
        },
    }


def _parse_literature_note(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower()
        if normalized in {"title", "claim", "mechanism", "hook", "observable", "risk"}:
            fields[normalized] = value.strip()
    return fields


def _literature_expansion_report_card(
    note_file: Path,
    text: str,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("state_hint") != "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN":
        return None
    skipped = [
        item
        for item in payload.get("skipped_candidates", [])
        if isinstance(item, dict)
    ]
    blocked = [
        item
        for item in payload.get("actions", [])
        if isinstance(item, dict) and item.get("executable") is False
    ]
    if not skipped and not blocked:
        return None
    skipped_ids = [
        str(item.get("algorithm_candidate_id"))
        for item in skipped
        if item.get("algorithm_candidate_id")
    ]
    blocked_ids = [
        str(item.get("algorithm_candidate_id"))
        for item in blocked
        if item.get("algorithm_candidate_id")
    ]
    return {
        "card_id": f"literature_expansion_{note_file.stem}",
        "source_type": "literature_expansion_report",
        "source_path": str(note_file),
        "claim": (
            "Hermes literature expansion should avoid failed literature "
            "candidates and prioritize candidates that can enter a bounded sandbox."
        ),
        "procedural_heuristic": (
            "Convert failed or blocked literature-candidate outcomes into "
            "negative memory before generating another hypothesis batch."
        ),
        "preconditions": [
            "Hermes literature expansion has produced skipped or blocked candidates"
        ],
        "implementation_hooks": ["avoid_failed_literature_candidate"],
        "expected_observables": [
            "future candidate rankings skip remembered failures",
            "Hermes does not repeat the same literature-candidate failure",
        ],
        "known_risks": [
            "skipped="
            + ",".join(skipped_ids or ["none"])
            + "; blocked="
            + ",".join(blocked_ids or ["none"])
        ],
        "evidence_status": "negative_memory",
        "evidence": {
            "skipped_candidate_ids": skipped_ids,
            "blocked_candidate_ids": blocked_ids,
        },
    }


def _algorithm_candidate_cards(candidate_file: Path) -> list[dict[str, Any]]:
    report = _read_json(candidate_file)
    if report.get("state_hint") != "ALGORITHM_CANDIDATES_COMPILED":
        raise HypothesisLabError("algorithm candidates artifact has invalid state_hint")
    candidates = report.get("algorithm_candidates")
    if not isinstance(candidates, list):
        raise HypothesisLabError("algorithm_candidates must be a list")
    cards = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        card = _algorithm_candidate_card(candidate_file, candidate)
        if card is not None:
            cards.append(card)
    return cards


def _algorithm_candidate_card(
    candidate_file: Path,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        return None
    hooks = _algorithm_candidate_hypothesis_hooks(candidate)
    if not hooks:
        return None
    candidate_id = str(candidate.get("algorithm_candidate_id", "")).strip()
    if not candidate_id:
        return None
    grounding = candidate.get("source_grounding")
    if not isinstance(grounding, dict):
        grounding = {}
    algorithm_card_id = str(
        grounding.get("algorithm_card_id") or candidate.get("parent_algorithm_card_id")
    ).strip()
    candidate_family = str(candidate.get("candidate_family", "")).strip()
    candidate_key = "|".join(
        [
            algorithm_card_id or "unknown_card",
            candidate_family or "unknown_family",
        ]
    )
    return {
        "card_id": f"algorithm_candidate_{candidate_id}",
        "source_type": "algorithm_candidate",
        "source_path": str(candidate_file),
        "claim": str(candidate.get("hypothesis", "")).strip()
        or "Local algorithm candidate should be tested in Hypothesis Lab.",
        "procedural_heuristic": str(candidate.get("mechanism", "")).strip()
        or "Run the bounded local sandbox before promotion.",
        "preconditions": [
            "Hermes local asset candidate queue selected this strategy",
            str(candidate.get("target_problem", "PDE_Burgers")),
        ],
        "implementation_hooks": hooks,
        "expected_observables": _candidate_string_list(
            candidate.get("expected_observables")
        )
        or ["sandbox score improves"],
        "known_risks": _candidate_string_list(candidate.get("risk_controls"))
        or ["candidate may not transfer"],
        "evidence_status": "candidate_ready",
        "evidence": {
            "algorithm_candidate_id": candidate_id,
            "candidate_key": candidate_key,
            "candidate_family": candidate_family,
            "model_family": candidate.get("model_family"),
            "source_algorithm_card_id": algorithm_card_id,
            "source_grounding": grounding,
        },
    }


def _algorithm_candidate_hypothesis_hooks(candidate: dict[str, Any]) -> list[str]:
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        return []
    hooks = _candidate_hypothesis_hooks(executor.get("implementation_hooks"))
    if hooks:
        return hooks
    family_to_hook = {
        "local_strategy_skill": None,
        "lightweight_residual_corrector": "learned_residual_correction",
        "spectral_rollout_stabilization": "linear_trend_spectral",
    }
    candidate_family = str(candidate.get("candidate_family", "")).strip()
    hook = family_to_hook.get(candidate_family)
    if hook is None:
        return []
    cli_command = str(executor.get("cli_command", "")).strip()
    if cli_command != "run-blueprint-sandbox":
        return []
    return _candidate_hypothesis_hooks([hook])


def _candidate_hypothesis_hooks(value: Any) -> list[str]:
    aliases = {
        "predict-task1-retrieval-prior": "task1_retrieval_prior",
    }
    supported = {
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
        "task1_retrieval_refinement_local_search",
        "linear_trend_spectral",
        "linear_trend_spectral_escape",
        "linear_trend_late_spectral_ramp",
        "learned_residual_correction",
        "external_residual_blend",
    }
    hooks = []
    seen: set[str] = set()
    for item in _candidate_string_list(value):
        item = aliases.get(item, item)
        if item not in supported or item in seen:
            continue
        hooks.append(item)
        seen.add(item)
    return hooks


def _candidate_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _record_algorithm_candidate_outcome(
    *,
    cards: dict[str, Any],
    hypotheses: dict[str, Any],
    executable_blueprint: dict[str, Any],
    sandbox: dict[str, Any],
    failure_memory_output_path: str | Path | None,
    force: bool,
) -> dict[str, Any] | None:
    if failure_memory_output_path is None:
        return None
    evidence = _executed_algorithm_candidate_evidence(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
    )
    if evidence is None:
        return None
    verifier = sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
    decision = str(verifier.get("decision", "unknown"))
    outcome = "passed" if decision == "promote_to_strategy_proposal" else "failed"
    reason_codes = [
        f"hypothesis_lab_decision_{decision}",
        f"score_delta_{float(verifier.get('score_delta_vs_baseline', 0.0)):.6g}",
        f"best_segment_score_{float(verifier.get('best_segment_score', 0.0)):.6g}",
    ]
    output_file = Path(failure_memory_output_path)
    record = record_hermes_literature_candidate_outcome(
        algorithm_candidate_id=str(evidence["algorithm_candidate_id"]),
        candidate_key=str(evidence["candidate_key"]),
        outcome=outcome,
        reason_codes=reason_codes,
        output_path=output_file,
        force=force and not output_file.exists(),
    )
    return {
        "algorithm_candidate_id": evidence["algorithm_candidate_id"],
        "candidate_key": evidence["candidate_key"],
        "outcome": outcome,
        "reason_codes": reason_codes,
        "failure_memory_path": str(output_file),
        "outcome_count": len(record.get("outcomes", [])),
    }


def _record_hypothesis_strategy_growth_artifacts(
    *,
    cards: dict[str, Any],
    hypotheses: dict[str, Any],
    executable_blueprint: dict[str, Any],
    sandbox: dict[str, Any],
    output_dir: Path,
    force: bool,
) -> dict[str, Any] | None:
    verifier = sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
    if verifier.get("decision") != "promote_to_strategy_proposal":
        return None
    card = _executed_algorithm_candidate_card(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
    )
    if card is None:
        return None
    evidence = card["evidence"]
    source_type = (
        "external_retrieval"
        if evidence.get("candidate_family") != "local_strategy_skill"
        else "agent_hypothesis"
    )
    proposal_path = _fresh_output(output_dir / "strategy_proposal.json", force=force)
    evidence_path = _fresh_output(
        output_dir / "strategy_evidence_summary.json",
        force=force,
    )
    strategy_id = str(evidence["algorithm_candidate_id"])
    proposal = record_strategy_proposal(
        strategy_id=strategy_id,
        title=f"Hypothesis Lab verified {strategy_id}",
        source_type=source_type,
        problem_context=(
            "Hypothesis Lab promoted a Hermes algorithm candidate after "
            "bounded 1D Burgers sandbox verification."
        ),
        hypothesis=str(card["claim"]),
        mechanism=str(card["procedural_heuristic"]),
        implementation_plan={
            "allowed_commands": ["run-hypothesis-lab"],
            "expected_artifacts": [
                "hypothesis_lab_report.json",
                "hypothesis_lab_sandbox_report.json",
                "verified_blueprints.json",
            ],
            "algorithm_candidate_id": strategy_id,
            "candidate_key": evidence["candidate_key"],
            "source_grounding": evidence.get("source_grounding", {}),
            "blueprint_id": executable_blueprint.get("blueprint_id"),
            "sandbox_report_path": sandbox["artifacts"]["sandbox_report_path"],
            "verified_blueprint_path": str(output_dir / "verified_blueprints.json"),
        },
        expected_observables=_candidate_string_list(card.get("expected_observables"))
        or ["sandbox score improves"],
        known_risks=_candidate_string_list(card.get("known_risks"))
        or ["single-window sandbox evidence may not generalize"],
        output_path=proposal_path,
    )
    summary = _hypothesis_lab_strategy_evidence_summary(
        strategy_id=strategy_id,
        evidence_path=evidence_path,
        sandbox=sandbox,
    )
    _write_json(evidence_path, summary)
    return {
        "strategy_id": strategy_id,
        "state_hint": proposal["state_hint"],
        "artifacts": {
            "strategy_proposal_path": str(proposal_path),
            "strategy_evidence_summary_path": str(evidence_path),
        },
        "next_command": proposal["next_command"],
    }


def _hypothesis_lab_strategy_evidence_summary(
    *,
    strategy_id: str,
    evidence_path: Path,
    sandbox: dict[str, Any],
) -> dict[str, Any]:
    verifier = sandbox["verifier"]
    score_delta = float(verifier["score_delta_vs_baseline"])
    best = sandbox["best_candidate"]
    return {
        "schema_version": 1,
        "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "source_state_hint": SANDBOX_STATE_HINT,
        "source_sandbox_report_path": sandbox["artifacts"]["sandbox_report_path"],
        "evidence_summary_path": str(evidence_path),
        "best_strategy_id": strategy_id,
        "evaluation_window_count": 1,
        "mean_score_delta": score_delta,
        "score_delta_std": 0.0,
        "long_window_score_delta": score_delta,
        "physics_residual_delta": 0.0,
        "inference_time_regression": float(verifier["inference_time"]),
        "failure_rate": 0.0
        if verifier["decision"] == "promote_to_strategy_proposal"
        else 1.0,
        "candidate_records": [
            {
                "strategy_id": strategy_id,
                "candidate_id": best["candidate_id"],
                "score": best["task1_total_score_proxy"],
                "score_delta": score_delta,
                "segment_score": best["segment_score"],
                "params": best["params"],
            }
        ],
        "provenance_notes": [
            "single_window_hypothesis_lab_sandbox",
            "requires_multi_window_promotion_before_skill_candidate",
        ],
        "next_command": "evaluate-strategy-promotion",
    }


def _executed_algorithm_candidate_evidence(
    *,
    cards: dict[str, Any],
    hypotheses: dict[str, Any],
    executable_blueprint: dict[str, Any],
) -> dict[str, Any] | None:
    card = _executed_algorithm_candidate_card(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
    )
    if card is None:
        return None
    evidence = card.get("evidence")
    if not isinstance(evidence, dict):
        return None
    return evidence


def _executed_algorithm_candidate_card(
    *,
    cards: dict[str, Any],
    hypotheses: dict[str, Any],
    executable_blueprint: dict[str, Any],
) -> dict[str, Any] | None:
    hypothesis_id = executable_blueprint.get("hypothesis_id")
    source_card_ids: list[str] = []
    for hypothesis in hypotheses.get("hypotheses", []):
        if not isinstance(hypothesis, dict):
            continue
        if hypothesis.get("hypothesis_id") != hypothesis_id:
            continue
        source_card_ids = [
            str(item)
            for item in hypothesis.get("source_card_ids", [])
            if isinstance(item, str)
        ]
        break
    if not source_card_ids:
        return None
    source_ids = set(source_card_ids)
    for card in cards.get("cards", []):
        if not isinstance(card, dict) or card.get("card_id") not in source_ids:
            continue
        evidence = card.get("evidence")
        if not isinstance(evidence, dict):
            continue
        if evidence.get("algorithm_candidate_id") and evidence.get("candidate_key"):
            return card
    return None


def _fresh_output(path: Path, *, force: bool) -> Path:
    if path.exists() and force:
        path.unlink()
    return path


def _adaptive_plateau_card(
    report_file: Path,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    if report.get("state_hint") != "HERMES_ADAPTIVE_HEARTBEAT_RAN":
        return None
    events = [item for item in report.get("policy_events", []) if isinstance(item, dict)]
    plateau_events = [
        item
        for item in events
        if item.get("trajectory_state") == "plateau"
        or item.get("action") == "verified_blueprint_plateau"
    ]
    if not plateau_events:
        return None
    last = plateau_events[-1]
    best_candidate_id = str(last.get("best_candidate_id", "unknown"))
    reason_codes = [
        str(item)
        for event in plateau_events
        for item in event.get("reason_codes", [])
        if isinstance(item, str)
    ]
    if last.get("action") == "verified_blueprint_plateau":
        reason_codes.append("verified_blueprint_plateau")
    escaped_once = "s1.5" in best_candidate_id or "spectral_strength': 1.5" in str(
        last.get("metadata", "")
    )
    return {
        "card_id": f"adaptive_plateau_{report_file.stem}",
        "source_type": "adaptive_heartbeat_report",
        "source_path": str(report_file),
        "claim": (
            "The verified linear trend spectral blueprint reached a local "
            "plateau and should trigger a wider controlled escape search."
        ),
        "procedural_heuristic": (
            "Keep the controlled NumPy operator, but expand slope windows, "
            "weaken the gain, and include unsmoothed plus stronger spectral "
            "settings instead of repeating the verified seed."
        ),
        "preconditions": [
            "adaptive heartbeat plateau",
            "verified blueprint reuse already attempted",
            "candidate generator is linear_trend_spectral",
        ],
        "implementation_hooks": [
            "linear_trend_late_spectral_ramp"
            if escaped_once
            else "linear_trend_spectral_escape"
        ],
        "expected_observables": [
            "new top candidate outside previous seed parameters",
            "non-negative score delta versus persistence",
            "runtime remains below inference budget",
        ],
        "known_risks": sorted(set(reason_codes)),
        "evidence_status": "plateaued",
        "evidence": {
            "candidate_id": best_candidate_id,
            "task1_total_fast_proxy": last.get("best_score"),
            "action": last.get("action"),
            "stage_index": last.get("stage_index"),
            "plateau_escape_already_used": escaped_once,
        },
    }


def _adaptive_retrieval_refinement_card(
    report_file: Path,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    if report.get("state_hint") != SANDBOX_STATE_HINT:
        return None
    family = str(report.get("candidate_family", ""))
    blueprint_id = str(report.get("blueprint_id", ""))
    if (
        family
        not in {
            "task1_retrieval_refinement",
            "task1_retrieval_refinement_local_search",
        }
        and "task1_retrieval_refinement" not in blueprint_id
    ):
        return None
    best = report.get("best_candidate")
    if not isinstance(best, dict):
        return None
    best_params = best.get("params")
    if not isinstance(best_params, dict):
        return None
    normalized = _normalize_evidence_params(best_params)
    if normalized is None or "k" not in normalized:
        return None
    top_params = []
    top_evidence = []
    seen = {json.dumps(normalized, sort_keys=True)}
    best_score = _candidate_score(best)
    scored_candidates = [
        {
            "candidate_id": best.get("candidate_id"),
            "params": normalized,
            "score": best_score,
            "segment_score": best.get("segment_score"),
            "task1_total_score_proxy": best.get("task1_total_score_proxy"),
        }
    ]
    top_candidates = report.get("top_candidates")
    if isinstance(top_candidates, list):
        for candidate in top_candidates:
            if not isinstance(candidate, dict):
                continue
            params = candidate.get("params")
            if not isinstance(params, dict):
                continue
            normalized_top = _normalize_evidence_params(params)
            if normalized_top is None or "k" not in normalized_top:
                continue
            key = json.dumps(normalized_top, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            top_params.append(normalized_top)
            score = _candidate_score(candidate)
            top_evidence.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "params": normalized_top,
                    "score": score,
                    "segment_score": candidate.get("segment_score"),
                    "task1_total_score_proxy": candidate.get(
                        "task1_total_score_proxy"
                    ),
                }
            )
            scored_candidates.append(top_evidence[-1])
    search_directives = _retrieval_refinement_search_directives(scored_candidates)
    return {
        "card_id": f"adaptive_retrieval_refinement_{report_file.stem}",
        "source_type": "adaptive_retrieval_refinement_sandbox",
        "source_path": str(report_file),
        "claim": (
            "A verified retrieval-refinement sandbox found a stronger local "
            "center, so Hermes should move the next local search around the "
            "observed best parameters instead of repeating a fixed hand-written grid."
        ),
        "procedural_heuristic": (
            "Use the sandbox best candidate as the local-search center, then "
            "perturb neighbor count, early blend length, persistence weight, "
            "and spectral strength under the same controlled NumPy executor."
        ),
        "preconditions": [
            "verified retrieval-refinement sandbox report exists",
            "best candidate preserves the first 10 context steps",
            "next iteration can reuse the same validation target as a sandbox",
        ],
        "implementation_hooks": ["task1_retrieval_refinement_local_search"],
        "expected_observables": [
            "candidate grid is centered on the latest sandbox best parameters",
            "new candidates include immediate score-neighborhood perturbations",
            "Hermes does not repeat only the original hand-written center",
        ],
        "known_risks": [
            "validation retrieval can overfit",
            "too narrow a local search can miss a different basin",
        ],
        "evidence_status": "local_search_center_ready",
        "evidence": {
            "candidate_id": best.get("candidate_id"),
            "params": normalized,
            "top_params": top_params,
            "top_candidates": top_evidence,
            "search_directives": search_directives,
            "task1_total_score_proxy": best.get("task1_total_score_proxy"),
            "task1_total_fast_proxy": best.get("task1_total_fast_proxy"),
            "segment_score": best.get("segment_score"),
            "blueprint_id": blueprint_id,
        },
    }


def _candidate_score(candidate: dict[str, Any]) -> float | None:
    for key in ["task1_total_score_proxy", "task1_total_fast_proxy", "segment_score"]:
        value = candidate.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _retrieval_refinement_search_directives(
    scored_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    directives = []
    axes = [
        "k",
        "early_blend_steps",
        "persistence_weight_start",
        "spectral_strength",
    ]
    rows = [
        item
        for item in scored_candidates
        if isinstance(item.get("params"), dict)
        and isinstance(item.get("score"), int | float)
    ]
    for axis in axes:
        axis_rows = [
            (float(item["params"][axis]), float(item["score"]))
            for item in rows
            if axis in item["params"]
        ]
        distinct_values = sorted({value for value, _score in axis_rows})
        if len(distinct_values) < 3:
            continue
        low_value = distinct_values[0]
        high_value = distinct_values[-1]
        low_score = max(score for value, score in axis_rows if value == low_value)
        high_score = max(score for value, score in axis_rows if value == high_value)
        if high_score > low_score:
            directives.append(
                {
                    "axis": axis,
                    "direction": "increase",
                    "reason": "top_candidate_score_trend",
                }
            )
        elif low_score > high_score:
            directives.append(
                {
                    "axis": axis,
                    "direction": "decrease",
                    "reason": "top_candidate_score_trend",
                }
            )
    return directives


def _scientific_log_card(log_file: Path) -> dict[str, Any] | None:
    if not log_file.exists():
        raise HypothesisLabError(f"log artifact not found: {log_file}")
    text = log_file.read_text(encoding="utf-8")
    lower = text.lower()
    if "verified_blueprint_plateau" not in lower and "plateaued" not in lower:
        return None
    if "linear_trend_spectral" not in lower and "linear trend spectral" not in lower:
        return None
    candidate_id = _first_log_best_candidate(text)
    late_ramp_seen = "_ls" in lower and "_lt" in lower
    return {
        "card_id": f"scientific_log_{log_file.stem}",
        "source_type": "scientific_log",
        "source_path": str(log_file),
        "claim": (
            "Verified blueprint reuse plateaued in the scientific log, so "
            "Hermes should explore a controlled linear trend spectral escape."
        ),
        "procedural_heuristic": (
            "Convert the log reflection into a structured memory card before "
            "choosing the next hypothesis, so the next batch does not repeat "
            "the same verified blueprint."
        ),
        "preconditions": [
            "task1 agent log contains plateau evidence",
            "linear trend spectral strategy has already been reused",
        ],
        "implementation_hooks": [
            "linear_trend_late_spectral_ramp"
            if late_ramp_seen
            else "linear_trend_spectral_escape"
        ],
        "expected_observables": [
            "hypothesis generator prioritizes plateau_escape",
            "knowledge card is visible in hypothesis sources",
        ],
        "known_risks": ["verified_blueprint_plateau", "repeating prior seed"],
        "evidence_status": "observed_plateau",
        "evidence": {
            "candidate_id": candidate_id,
            "summary": "adaptive policy log reported verified blueprint plateau",
            "late_spectral_ramp_seen": late_ramp_seen,
        },
    }


def _first_log_best_candidate(text: str) -> str | None:
    for line in text.splitlines():
        marker = "candidate="
        if marker not in line:
            continue
        return line.split(marker, 1)[1].split()[0].strip()
    for line in text.splitlines():
        marker = "best="
        if marker not in line:
            continue
        return line.split(marker, 1)[1].split()[0].strip()
    return None


def _prediction_from_blueprint_params(
    *,
    family: str,
    context: np.ndarray,
    target: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    if family == "linear_trend_spectral":
        return _prediction_from_linear_trend_spectral(
            context=context,
            params=params,
        )
    if family == "learned_residual_correction":
        return _prediction_from_learned_residual(
            context=context,
            target=target,
            params=params,
        )
    if family == "task1_retrieval_prior":
        return _prediction_from_task1_retrieval_prior(
            context=context,
            target=target,
            params=params,
        )
    if family == "task1_retrieval_refinement":
        return _prediction_from_task1_retrieval_refinement(
            context=context,
            target=target,
            params=params,
        )
    if family == "task1_retrieval_refinement_local_search":
        return _prediction_from_task1_retrieval_refinement(
            context=context,
            target=target,
            params=params,
        )
    raise HypothesisLabError(f"unsupported blueprint candidate_family: {family}")


def _prediction_from_task1_retrieval_prior(
    *,
    context: np.ndarray,
    target: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    leave_one_out = bool(
        context.shape[0] == target.shape[0]
        and np.allclose(context, target[:, : context.shape[1], :], atol=1e-6)
    )
    retrieval = predict_retrieval_prior(
        reference=target,
        context=context,
        k=int(params["k"]),
        mode=str(params.get("mode", "affine")),
        leave_one_out=leave_one_out,
    )
    prediction = make_persistence_prediction(context)
    prediction[:, context.shape[1] :, :] = retrieval.prediction
    return prediction.astype(np.float32, copy=False)


def _prediction_from_task1_retrieval_refinement(
    *,
    context: np.ndarray,
    target: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    retrieval = _prediction_from_task1_retrieval_prior(
        context=context,
        target=target,
        params=params,
    )
    persistence = make_persistence_prediction(context)
    prediction = retrieval.copy()
    forecast_steps = prediction.shape[1] - context.shape[1]
    early_steps = int(params["early_blend_steps"])
    weight_start = float(params["persistence_weight_start"])
    if early_steps > 0 and weight_start > 0.0:
        horizon = np.arange(forecast_steps, dtype=np.float32)
        weights = weight_start * np.clip(1.0 - horizon / float(early_steps), 0.0, 1.0)
        weights = weights.reshape(1, forecast_steps, 1)
        prediction[:, context.shape[1] :, :] = (
            weights * persistence[:, context.shape[1] :, :]
            + (1.0 - weights) * retrieval[:, context.shape[1] :, :]
        )
    strength = float(params.get("spectral_strength", 0.0))
    if strength > 0.0:
        prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=strength,
        )
    prediction[:, : context.shape[1], :] = context
    return prediction.astype(np.float32, copy=False)


def _candidate_metadata(
    *,
    family: str,
    context: np.ndarray,
    target: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    if family not in {
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
        "task1_retrieval_refinement_local_search",
    }:
        return {}
    metadata = {
        "leave_one_out": bool(
            context.shape[0] == target.shape[0]
            and np.allclose(context, target[:, : context.shape[1], :], atol=1e-6)
        ),
        "reference_count": int(target.shape[0]),
        "k": int(params["k"]),
        "mode": str(params.get("mode", "affine")),
    }
    if family in {
        "task1_retrieval_refinement",
        "task1_retrieval_refinement_local_search",
    }:
        metadata.update(
            {
                "early_blend_steps": int(params["early_blend_steps"]),
                "persistence_weight_start": float(
                    params["persistence_weight_start"]
                ),
                "spectral_strength": float(params.get("spectral_strength", 0.0)),
            }
        )
    return metadata


def _prediction_from_linear_trend_spectral(
    *,
    context: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    clip = (float(np.min(context)), float(np.max(context)))
    prediction = linear_trend_prediction(
        context=context,
        gain=float(params["gain"]),
        slope_window=int(params["slope_window"]),
        decay=float(params["decay"]),
        clip=clip,
    )
    strength = float(params.get("spectral_strength", 0.0))
    if strength > 0.0:
        prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=strength,
        )
    late_strength = params.get("late_spectral_strength")
    late_start = params.get("late_start")
    if isinstance(late_strength, int | float) and float(late_strength) > 0.0:
        start = int(late_start) if isinstance(late_start, int | float) else 105
        start = max(context.shape[1], min(prediction.shape[1], start))
        late_prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=float(late_strength),
        )
        prediction = prediction.copy()
        prediction[:, start:, :] = late_prediction[:, start:, :]
    return prediction


def _prediction_from_learned_residual(
    *,
    context: np.ndarray,
    target: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    baseline = make_persistence_prediction(context)
    residual = target - baseline
    residual[:, : context.shape[1], :] = 0.0
    template = np.mean(residual, axis=0, keepdims=True)
    prediction = baseline + float(params["residual_shrinkage"]) * template
    strength = float(params.get("residual_spectral_strength", 0.0))
    if strength > 0.0:
        prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=strength,
        )
        prediction[:, : context.shape[1], :] = context
    return prediction.astype(np.float32, copy=False)


def _iter_parameter_grid(
    grid: dict[str, list[Any]],
    *,
    family: str = "linear_trend_spectral",
) -> list[dict[str, Any]]:
    if family == "learned_residual_correction":
        return _residual_parameter_grid(grid)
    if family == "task1_retrieval_prior":
        return _retrieval_parameter_grid(grid)
    if family == "task1_retrieval_refinement":
        return _retrieval_refinement_parameter_grid(grid)
    if family == "task1_retrieval_refinement_local_search":
        return _retrieval_refinement_parameter_grid(grid)
    rows = []
    late_strengths = grid.get("late_spectral_strength", [None])
    late_starts = grid.get("late_start", [None])
    for slope_window in grid["slope_window"]:
        for gain in grid["gain"]:
            for decay in grid["decay"]:
                for strength in grid["spectral_strength"]:
                    for late_strength in late_strengths:
                        for late_start in late_starts:
                            row = {
                                "slope_window": int(slope_window),
                                "gain": float(gain),
                                "decay": float(decay),
                                "spectral_strength": float(strength),
                            }
                            if late_strength is not None:
                                row["late_spectral_strength"] = float(late_strength)
                            if late_start is not None:
                                row["late_start"] = int(late_start)
                            rows.append(row)
    return rows


def _residual_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    shrinkages = grid.get("residual_shrinkage")
    strengths = grid.get("residual_spectral_strength", [0.0])
    if not isinstance(shrinkages, list) or not shrinkages:
        raise HypothesisLabError("residual blueprint missing residual_shrinkage list")
    if not isinstance(strengths, list) or not strengths:
        raise HypothesisLabError(
            "residual blueprint residual_spectral_strength must be a list"
        )
    return [
        {
            "residual_shrinkage": float(shrinkage),
            "residual_spectral_strength": float(strength),
        }
        for shrinkage in shrinkages
        for strength in strengths
    ]


def _retrieval_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    ks = grid.get("k")
    modes = grid.get("mode", ["affine"])
    if not isinstance(ks, list) or not ks:
        raise HypothesisLabError("retrieval blueprint missing k list")
    if not isinstance(modes, list) or not modes:
        raise HypothesisLabError("retrieval blueprint mode must be a list")
    return [{"k": int(k), "mode": str(mode)} for k in ks for mode in modes]


def _retrieval_refinement_parameter_grid(
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    candidate_params = grid.get("candidate_params")
    if isinstance(candidate_params, list):
        rows = []
        for row in candidate_params:
            if not isinstance(row, dict):
                raise HypothesisLabError(
                    "retrieval refinement candidate_params entries must be objects"
                )
            rows.append(_normalize_retrieval_refinement_params(row))
        if not rows:
            raise HypothesisLabError(
                "retrieval refinement candidate_params must not be empty"
            )
        return rows
    ks = grid.get("k")
    modes = grid.get("mode")
    early_steps = grid.get("early_blend_steps")
    weight_starts = grid.get("persistence_weight_start")
    strengths = grid.get("spectral_strength", [0.0])
    required = {
        "k": ks,
        "mode": modes,
        "early_blend_steps": early_steps,
        "persistence_weight_start": weight_starts,
        "spectral_strength": strengths,
    }
    for key, value in required.items():
        if not isinstance(value, list) or not value:
            raise HypothesisLabError(f"retrieval refinement blueprint missing {key} list")
    rows = []
    for k in ks:
        for mode in modes:
            for steps in early_steps:
                for weight in weight_starts:
                    for strength in strengths:
                        rows.append(
                            _normalize_retrieval_refinement_params(
                                {
                                    "k": k,
                                    "mode": mode,
                                    "early_blend_steps": steps,
                                    "persistence_weight_start": weight,
                                    "spectral_strength": strength,
                                }
                            )
                        )
    return rows


def _default_retrieval_refinement_params() -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = [
        {
            "k": 20,
            "mode": "affine",
            "early_blend_steps": 0,
            "persistence_weight_start": 0.0,
            "spectral_strength": 0.0,
        },
        {
            "k": 48,
            "mode": "affine",
            "early_blend_steps": 0,
            "persistence_weight_start": 0.0,
            "spectral_strength": 0.0,
        },
    ]
    for k in [32, 48]:
        for early_steps in [20, 40]:
            for weight in [0.25, 0.5]:
                for strength in [0.5, 1.0]:
                    params.append(
                        {
                            "k": k,
                            "mode": "affine",
                            "early_blend_steps": early_steps,
                            "persistence_weight_start": weight,
                            "spectral_strength": strength,
                        }
                    )
    params.extend(
        [
            {
                "k": 48,
                "mode": "delta",
                "early_blend_steps": 20,
                "persistence_weight_start": 0.5,
                "spectral_strength": 0.5,
            },
            {
                "k": 48,
                "mode": "raw",
                "early_blend_steps": 20,
                "persistence_weight_start": 0.5,
                "spectral_strength": 0.5,
            },
        ]
    )
    return params


def _local_retrieval_refinement_params(
    evidence_params: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    center = _retrieval_refinement_center(evidence_params)
    k0 = int(center["k"])
    mode0 = str(center["mode"])
    early0 = int(center["early_blend_steps"])
    weight0 = float(center["persistence_weight_start"])
    strength0 = float(center["spectral_strength"])
    params: list[dict[str, Any]] = [center]
    for k in [k0 - 8, k0 - 4, k0, k0 + 4, k0 + 8]:
        params.append(
            {
                "k": _clamp_int(k, 4, 128),
                "mode": mode0,
                "early_blend_steps": early0,
                "persistence_weight_start": weight0,
                "spectral_strength": strength0,
            }
        )
    for early_steps in [early0 - 5, early0, early0 + 5]:
        for weight in [
            weight0 - 0.1,
            weight0 - 0.05,
            weight0,
            weight0 + 0.05,
            weight0 + 0.1,
        ]:
            params.append(
                {
                    "k": k0,
                    "mode": mode0,
                    "early_blend_steps": _clamp_int(early_steps, 0, 80),
                    "persistence_weight_start": _clamp_float(weight, 0.0, 1.0),
                    "spectral_strength": strength0,
                }
            )
    for strength in [strength0 - 0.25, strength0, strength0 + 0.25, strength0 + 0.5]:
        params.append(
            {
                "k": k0,
                "mode": mode0,
                "early_blend_steps": early0,
                "persistence_weight_start": weight0,
                "spectral_strength": _clamp_float(strength, 0.0, 2.0),
            }
        )
    for directive in _retrieval_refinement_search_directives_from_evidence(
        evidence_params
    ):
        axis = directive["axis"]
        direction = directive["direction"]
        params.extend(
            _directed_retrieval_refinement_params(
                center=center,
                axis=axis,
                direction=direction,
            )
        )
    for mode in ["delta", "raw"]:
        params.append(
            {
                "k": k0,
                "mode": mode,
                "early_blend_steps": early0,
                "persistence_weight_start": weight0,
                "spectral_strength": strength0,
            }
        )
    deduped = []
    seen: set[tuple[int, str, int, float, float]] = set()
    for row in params:
        normalized = _normalize_retrieval_refinement_params(row)
        key = (
            int(normalized["k"]),
            str(normalized["mode"]),
            int(normalized["early_blend_steps"]),
            float(normalized["persistence_weight_start"]),
            float(normalized["spectral_strength"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return _filter_retrieval_refinement_params_by_directives(
        params=deduped,
        center=center,
        directives=_retrieval_refinement_search_directives_from_evidence(
            evidence_params
        ),
    )


def _retrieval_refinement_search_directives_from_evidence(
    evidence_params: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    directives = []
    seen: set[tuple[str, str]] = set()
    for item in evidence_params or []:
        if not isinstance(item, dict):
            continue
        raw_directives = item.get("search_directives")
        if not isinstance(raw_directives, list):
            continue
        for directive in raw_directives:
            if not isinstance(directive, dict):
                continue
            axis = str(directive.get("axis", ""))
            direction = str(directive.get("direction", ""))
            if axis not in {
                "k",
                "early_blend_steps",
                "persistence_weight_start",
                "spectral_strength",
            }:
                continue
            if direction not in {"increase", "decrease"}:
                continue
            key = (axis, direction)
            if key in seen:
                continue
            seen.add(key)
            directives.append({"axis": axis, "direction": direction})
    return directives


def _retrieval_refinement_experiment_plan(
    *,
    evidence_params: list[dict[str, Any]] | None,
    candidate_budget: int,
) -> dict[str, Any]:
    directives = _retrieval_refinement_search_directives_from_evidence(
        evidence_params
    )
    if directives:
        axis_plan = [
            {
                "axis": directive["axis"],
                "direction": directive["direction"],
                "reason": directive.get("reason", "top_candidate_score_trend"),
                "budget": 3,
            }
            for directive in directives
        ]
        avoid_rules = _retrieval_refinement_avoid_rules_from_directives(directives)
        decision = "axis_directed_local_search"
        focus = "score_trend"
    else:
        axis_plan = [
            {
                "axis": axis,
                "direction": "bidirectional",
                "reason": "no_clear_axis_trend",
                "budget": 2,
            }
            for axis in [
                "k",
                "early_blend_steps",
                "persistence_weight_start",
                "spectral_strength",
            ]
        ]
        avoid_rules = []
        decision = "centered_local_search"
        focus = "local_neighborhood"
    return {
        "plan_id": "xp_task1_retrieval_refinement_local_search_001",
        "decision": decision,
        "objective": {
            "primary": "task1_total_score_proxy",
            "secondary": [
                "early_rel_mse",
                "middle_rel_mse",
                "late_rmse",
                "inference_seconds",
            ],
            "focus": focus,
        },
        "axis_plan": axis_plan,
        "avoid_rules": avoid_rules,
        "candidate_budget": int(candidate_budget),
        "stop_criteria": [
            "stop_if_no_candidate_improves_seed",
            "stop_if_context_prefix_validation_fails",
            "stop_if_inference_budget_exceeded",
        ],
        "switch_criteria": [
            "switch_family_after_repeated_axis_plateau",
            "switch_family_if_score_regresses_on_heldout_split",
        ],
    }


def _retrieval_refinement_avoid_rules_from_directives(
    directives: list[dict[str, str]],
) -> list[dict[str, str]]:
    rules = []
    for directive in directives:
        direction = directive["direction"]
        if direction == "increase":
            relation = "below_center"
        elif direction == "decrease":
            relation = "above_center"
        else:
            continue
        rules.append(
            {
                "axis": directive["axis"],
                "relation": relation,
                "reason": "opposite_of_score_trend",
            }
        )
    return rules


def _filter_retrieval_refinement_params_by_directives(
    *,
    params: list[dict[str, Any]],
    center: dict[str, Any],
    directives: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rules = _retrieval_refinement_avoid_rules_from_directives(directives)
    if not rules:
        return params
    filtered = []
    for row in params:
        rejected = False
        for rule in rules:
            axis = rule["axis"]
            if axis not in row or axis not in center:
                continue
            value = float(row[axis])
            center_value = float(center[axis])
            if rule["relation"] == "below_center" and value < center_value:
                rejected = True
            if rule["relation"] == "above_center" and value > center_value:
                rejected = True
        if not rejected:
            filtered.append(row)
    return filtered


def _directed_retrieval_refinement_params(
    *,
    center: dict[str, Any],
    axis: str,
    direction: str,
) -> list[dict[str, Any]]:
    sign = 1 if direction == "increase" else -1
    rows = []
    axis_steps: dict[str, list[float]] = {
        "k": [8.0, 16.0, 24.0],
        "early_blend_steps": [10.0, 20.0, 30.0],
        "persistence_weight_start": [0.1, 0.15, 0.2],
        "spectral_strength": [0.5, 1.0, 1.5],
    }
    for step in axis_steps.get(axis, []):
        row = dict(center)
        if axis == "k":
            row["k"] = _clamp_int(int(center["k"]) + sign * int(step), 4, 128)
        elif axis == "early_blend_steps":
            row["early_blend_steps"] = _clamp_int(
                int(center["early_blend_steps"]) + sign * int(step),
                0,
                80,
            )
        elif axis == "persistence_weight_start":
            row["persistence_weight_start"] = _clamp_float(
                float(center["persistence_weight_start"]) + sign * step,
                0.0,
                1.0,
            )
        elif axis == "spectral_strength":
            row["spectral_strength"] = _clamp_float(
                float(center["spectral_strength"]) + sign * step,
                0.0,
                2.0,
            )
        rows.append(row)
    return rows


def _retrieval_refinement_center(
    evidence_params: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    for item in evidence_params or []:
        if not isinstance(item, dict):
            continue
        params = item.get("params")
        if not isinstance(params, dict):
            continue
        normalized = _normalize_evidence_params(params)
        if normalized is not None and "k" in normalized:
            return normalized
    return {
        "k": 48,
        "mode": "affine",
        "early_blend_steps": 20,
        "persistence_weight_start": 0.5,
        "spectral_strength": 0.5,
    }


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return round(max(lower, min(upper, float(value))), 6)


def _normalize_retrieval_refinement_params(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "k": int(row["k"]),
        "mode": str(row["mode"]),
        "early_blend_steps": int(row["early_blend_steps"]),
        "persistence_weight_start": float(row["persistence_weight_start"]),
        "spectral_strength": float(row.get("spectral_strength", 0.0)),
    }


def _candidate_id(params: dict[str, Any]) -> str:
    if "early_blend_steps" in params and "persistence_weight_start" in params:
        return (
            "task1_retrieval_refinement"
            f"_k{int(params['k'])}"
            f"_{params['mode']}"
            f"_eb{int(params['early_blend_steps'])}"
            f"_pw{float(params['persistence_weight_start']):g}"
            f"_s{float(params.get('spectral_strength', 0.0)):g}"
        )
    if "k" in params and "mode" in params:
        return f"task1_retrieval_prior_k{int(params['k'])}_{params['mode']}"
    if "residual_shrinkage" in params:
        return (
            "learned_residual"
            f"_sh{params['residual_shrinkage']:g}"
            f"_rs{params.get('residual_spectral_strength', 0.0):g}"
        )
    candidate = (
        "linear_trend_spectral"
        f"_w{params['slope_window']}"
        f"_g{params['gain']:g}"
        f"_d{params['decay']:g}"
        f"_s{params['spectral_strength']:g}"
    )
    if "late_spectral_strength" in params:
        candidate += f"_ls{params['late_spectral_strength']:g}"
    if "late_start" in params:
        candidate += f"_lt{params['late_start']}"
    return candidate


def _context_preserved(*, prediction: np.ndarray, context: np.ndarray) -> bool:
    return bool(np.allclose(prediction[:, : context.shape[1], :], context, atol=1e-6))


def _sandbox_metric_results(
    *,
    judge: dict[str, Any],
    baseline_score: dict[str, Any],
    best_segments: dict[str, Any],
    score_delta: float,
    context_preserved: bool,
    inference_seconds: float,
    max_inference_seconds: float,
    promotion_min_score_delta: float,
) -> dict[str, Any]:
    best_score = judge["tasks"]["1"]["score"]
    baseline_segments = baseline_score.get("segments", {})
    late_delta = _segment_metric_delta(
        baseline_segments=baseline_segments,
        best_segments=best_segments,
        segment="late",
        metric="rmse",
        lower_is_better=True,
    )
    segment_delta = float(best_score["segment_score"]) - float(
        baseline_score["segment_score"]
    )
    return {
        "rules_passed": {
            "value": bool(judge["rules_passed"]),
            "passed": bool(judge["rules_passed"]),
        },
        "context_preserved": {
            "value": bool(context_preserved),
            "passed": bool(context_preserved),
        },
        "score_delta_vs_baseline": {
            "value": score_delta,
            "threshold": promotion_min_score_delta,
            "passed": score_delta >= promotion_min_score_delta,
        },
        "segment_score_delta_vs_baseline": {
            "value": segment_delta,
            "passed": segment_delta >= 0.0,
        },
        "late_rmse_delta_vs_baseline": {
            "value": late_delta,
            "passed": late_delta is None or late_delta >= 0.0,
        },
        "inference_time_budget": {
            "value": inference_seconds,
            "threshold": max_inference_seconds,
            "passed": inference_seconds <= max_inference_seconds,
        },
    }


def _task1_multiview_local_judge(
    *,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    full_score = score_prediction_arrays(prediction=prediction, target=target)
    full_baseline = score_prediction_arrays(
        prediction=baseline_prediction,
        target=target,
    )
    full_delta = float(full_score["segment_score"]) - float(
        full_baseline["segment_score"]
    )
    heldout_view = _task1_heldout_view(
        prediction=prediction,
        baseline_prediction=baseline_prediction,
        target=target,
    )
    late_delta = _segment_metric_delta(
        baseline_segments=full_baseline.get("segments", {}),
        best_segments=full_score.get("segments", {}),
        segment="late",
        metric="rmse",
        lower_is_better=True,
    )
    views = {
        "fast_full": {
            "sample_count": int(target.shape[0]),
            "segment_score": full_score["segment_score"],
            "baseline_segment_score": full_baseline["segment_score"],
            "segment_score_delta": full_delta,
            "passed": full_delta >= 0.0,
        },
        "heldout_split": heldout_view,
        "late_stability": {
            "sample_count": int(target.shape[0]),
            "late_rmse": full_score["segments"]["late"]["rmse"],
            "baseline_late_rmse": full_baseline["segments"]["late"]["rmse"],
            "late_rmse_delta": late_delta,
            "passed": late_delta is None or late_delta >= 0.0,
        },
    }
    required = [
        bool(views["fast_full"]["passed"]),
        bool(views["heldout_split"].get("passed")),
        bool(views["late_stability"]["passed"]),
    ]
    return {
        "schema_version": 1,
        "state_hint": "TASK1_MULTIVIEW_LOCAL_JUDGE_SCORED",
        "decision": "consistent_improvement"
        if all(required)
        else "mixed_or_insufficient_evidence",
        "views": views,
    }


def _task1_heldout_view(
    *,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    sample_count = int(target.shape[0] // 2)
    if sample_count < 1:
        return {
            "sample_count": 0,
            "status": "skipped_insufficient_samples",
            "passed": False,
        }
    heldout = np.arange(target.shape[0]) % 2 == 1
    heldout_score = score_prediction_arrays(
        prediction=prediction[heldout],
        target=target[heldout],
    )
    heldout_baseline = score_prediction_arrays(
        prediction=baseline_prediction[heldout],
        target=target[heldout],
    )
    delta = float(heldout_score["segment_score"]) - float(
        heldout_baseline["segment_score"]
    )
    return {
        "sample_count": sample_count,
        "status": "scored",
        "segment_score": heldout_score["segment_score"],
        "baseline_segment_score": heldout_baseline["segment_score"],
        "segment_score_delta": delta,
        "passed": delta >= 0.0,
    }


def _segment_metric_delta(
    *,
    baseline_segments: Any,
    best_segments: Any,
    segment: str,
    metric: str,
    lower_is_better: bool,
) -> float | None:
    if not isinstance(baseline_segments, dict) or not isinstance(best_segments, dict):
        return None
    baseline = baseline_segments.get(segment)
    best = best_segments.get(segment)
    if not isinstance(baseline, dict) or not isinstance(best, dict):
        return None
    baseline_value = baseline.get(metric)
    best_value = best.get(metric)
    if not isinstance(baseline_value, int | float) or not isinstance(
        best_value,
        int | float,
    ):
        return None
    if lower_is_better:
        return float(baseline_value) - float(best_value)
    return float(best_value) - float(baseline_value)


def _sandbox_metrics_pass(metric_results: dict[str, Any]) -> bool:
    required = [
        "rules_passed",
        "context_preserved",
        "score_delta_vs_baseline",
        "inference_time_budget",
    ]
    return all(bool(metric_results[name]["passed"]) for name in required)


def _sandbox_promotion_score(metric_results: dict[str, Any]) -> float:
    score_delta = metric_results["score_delta_vs_baseline"]["value"]
    segment_delta = metric_results["segment_score_delta_vs_baseline"]["value"]
    runtime = metric_results["inference_time_budget"]["value"]
    return float(score_delta + 0.1 * segment_delta - 0.001 * runtime)


def _task1_total_score_proxy(
    *,
    prediction_score: float,
    inference_seconds: float,
) -> float:
    inference_score = 0.0
    if inference_seconds <= 120.0:
        inference_score = 40.0 * max(0.0, 1.0 - inference_seconds / 120.0)
    return float(prediction_score + 35.0 + inference_score)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HypothesisLabError(f"JSON artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HypothesisLabError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise HypothesisLabError(f"expected JSON object: {path}")
    return value


def _require_state(payload: dict[str, Any], state_hint: str) -> None:
    if payload.get("state_hint") != state_hint:
        raise HypothesisLabError(f"expected state_hint {state_hint}")


def _guard_output(path: Path | None, *, force: bool) -> None:
    if path is not None and path.exists() and not force:
        raise HypothesisLabError(f"artifact already exists: {path}")


def _write_optional_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is not None:
        _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
