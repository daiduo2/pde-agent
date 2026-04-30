"""Audited local hypothesis lab for PDE operator strategy generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.hermes_local_judge import judge_hermes_submission
from pde_agent.core.manual_operator_tuning import (
    linear_trend_prediction,
    load_hdf5_tensor,
    make_persistence_prediction,
    score_prediction_arrays,
    spectral_smooth_prediction,
)


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
    hypotheses = []
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
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "params": {
                    "slope_window": int(params["slope_window"]),
                    "gain": float(params["gain"]),
                    "decay": float(params["decay"]),
                    "spectral_strength": float(params["spectral_strength"]),
                },
                "source_card_id": card["card_id"],
                "task1_total_fast_proxy": evidence.get("task1_total_fast_proxy"),
                "segment_score": evidence.get("segment_score"),
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["task1_total_fast_proxy"])
            if isinstance(item.get("task1_total_fast_proxy"), int | float)
            else 0.0,
            str(item["source_card_id"]),
        )
    )
    return rows


def _rank_hypotheses(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for hypothesis in hypotheses:
        item = dict(hypothesis)
        family = str(item.get("candidate_family", ""))
        priority = int(item.get("priority", 10))
        novelty_bonus = {
            "official_checkpoint_calibration": 45.0,
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
    if family not in {"linear_trend_spectral", "learned_residual_correction"}:
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
        "next_command": sandbox["next_command"],
    }
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
    fields = _parse_literature_note(note_file.read_text(encoding="utf-8"))
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
    raise HypothesisLabError(f"unsupported blueprint candidate_family: {family}")


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


def _candidate_id(params: dict[str, Any]) -> str:
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
