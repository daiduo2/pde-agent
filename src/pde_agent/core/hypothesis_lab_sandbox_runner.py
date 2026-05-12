"""Controlled sandbox runner for Hypothesis Lab blueprints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.burgers_hdf5_io import load_hdf5_tensor
from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.hypothesis_lab_candidate_pool import _candidate_pool_attribution
from pde_agent.core.hypothesis_lab_candidate_params import (
    _early_transition_expert_parameter_grid,
)
from pde_agent.core.hypothesis_lab_candidate_reporting import (
    _candidate_id,
    _candidate_metadata,
)
from pde_agent.core.hypothesis_lab_parameter_grids import (
    _front25_dynamics_parameter_grid,
    _physics_residual_refinement_parameter_grid,
    _pi_deeponet_parameter_grid,
    _residual_parameter_grid,
    _retrieval_parameter_grid,
    _retrieval_refinement_parameter_grid,
    _sem_burgers_parameter_grid,
    _short_window_corrector_parameter_grid,
)
from pde_agent.core.hypothesis_lab_predictions import _prediction_from_blueprint_params
from pde_agent.core.hypothesis_lab_sandbox_reference import (
    _load_sandbox_reference,
    _reference_audit,
)
from pde_agent.core.hypothesis_lab_scoring import (
    _sandbox_metric_results,
    _sandbox_pareto_promotion_gate,
    _sandbox_promotion_score,
    _task1_early_segment_diagnosis,
    _task1_multiview_local_judge,
    _task1_total_score_proxy,
)
from pde_agent.core.hypothesis_lab_selection import (
    _blueprint_selection_objective_name,
    _candidate_selection_key,
    _context_preserved,
    _selection_objective,
)
from pde_agent.core.hypothesis_lab_utils import HypothesisLabError, _guard_output, _write_json
from pde_agent.core.manual_operator_tuning import (
    make_persistence_prediction,
    score_burgers_segments,
)
from pde_agent.core.submission_judge import judge_submission


SANDBOX_STATE_HINT = "HYPOTHESIS_LAB_SANDBOX_VERIFIED"

SANDBOX_EXECUTABLE_FAMILIES = {
    "linear_trend_spectral",
    "learned_residual_correction",
    "task1_retrieval_prior",
    "task1_retrieval_refinement",
    "task1_retrieval_refinement_local_search",
    "task1_early_refinement",
    "task1_early_transition_expert",
    "task1_front25_dynamics",
    "task1_short_window_corrector",
    "task1_sem_burgers_rollout",
    "task1_physics_residual_refinement",
    "task1_pi_deeponet_correction_head",
}


def run_blueprint_sandbox(
    *,
    blueprint: dict[str, Any],
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
    family = str(blueprint.get("candidate_family"))
    if family not in SANDBOX_EXECUTABLE_FAMILIES:
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
    reference = _load_sandbox_reference(
        reference_path=reference_path,
        target_path=target_path,
        target=target,
        sample_limit=reference_sample_limit,
    )
    reference_audit = _reference_audit(
        reference_path=reference_path,
        target_path=target_path,
        reference=reference,
        reference_sample_limit=reference_sample_limit,
    )
    baseline_prediction = make_persistence_prediction(context)
    baseline_score = score_burgers_segments(
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
            reference=reference,
            params=params,
        )
        elapsed = time.perf_counter() - start
        score = score_burgers_segments(prediction=prediction, target=target)
        rows.append(
            {
                "candidate_id": _candidate_id(params, family=family),
                "family": family,
                "params": params,
                "metadata": _candidate_metadata(
                    family=family,
                    context=context,
                    target=target,
                    reference=reference,
                    reference_audit=reference_audit,
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
    selection_objective_name = _blueprint_selection_objective_name(
        blueprint=blueprint,
        family=family,
    )
    rows.sort(
        key=lambda item: _candidate_selection_key(
            item,
            family=family,
            selection_objective=selection_objective_name,
        ),
        reverse=True,
    )
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
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    judge = judge_submission(
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
    delta = float(judge["total_score_proxy"]) - baseline_total
    serializable_rows = [{k: v for k, v in row.items() if k != "_prediction"} for row in rows]
    candidate_pool_attribution = _candidate_pool_attribution(
        rows=serializable_rows,
        top_candidate_count=20,
        family=family,
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
    if family == "task1_physics_residual_refinement":
        metadata = best.get("metadata") if isinstance(best.get("metadata"), dict) else {}
        residual_delta = metadata.get("residual_rmse_delta")
        if isinstance(residual_delta, int | float):
            metric_results["physics_residual_rmse_delta"] = {
                "value": float(residual_delta),
                "threshold": 0.0,
                "passed": float(residual_delta) < 0.0,
            }
    if family == "task1_pi_deeponet_correction_head":
        metadata = best.get("metadata") if isinstance(best.get("metadata"), dict) else {}
        correction_rmse = metadata.get("learned_correction_rmse")
        if isinstance(correction_rmse, int | float):
            metric_results["pi_deeponet_active_correction"] = {
                "value": float(correction_rmse),
                "threshold": 1e-8,
                "passed": float(correction_rmse) > 1e-8,
            }
    multiview_judge = _task1_multiview_local_judge(
        prediction=best["_prediction"],
        baseline_prediction=baseline_prediction,
        target=target,
    )
    best_validation_score = judge["tasks"]["1"].get("validation_score")
    if not isinstance(best_validation_score, dict):
        best_validation_score = {
            "segment_score": best["segment_score"],
            "segments": best["segments"],
        }
    early_segment_diagnosis = _task1_early_segment_diagnosis(
        baseline_score=baseline_score,
        best_score=best_validation_score,
        prediction=best["_prediction"],
        baseline_prediction=baseline_prediction,
        target=target,
        candidates=serializable_rows,
    )
    selection_objective = _selection_objective(
        family,
        selection_objective=selection_objective_name,
    )
    promotion_score = _sandbox_promotion_score(metric_results)
    pareto_gate = _sandbox_pareto_promotion_gate(
        family=family,
        metric_results=metric_results,
        selection_objective=selection_objective,
        candidate_budget=len(rows),
        promotion_score=promotion_score,
    )
    verifier = {
        "decision": pareto_gate["decision"],
        "rules_passed": bool(judge["rules_passed"]),
        "context_preserved": context_preserved,
        "score_delta_vs_baseline": delta,
        "baseline_segment_score": baseline_score["segment_score"],
        "best_segment_score": judge["tasks"]["1"]["score"]["segment_score"],
        "inference_time": float(best["elapsed_seconds"]),
        "judge_decision": judge["decision"],
        "metric_results": metric_results,
        "multiview_judge": multiview_judge,
        "early_segment_diagnosis": early_segment_diagnosis,
        "selection_objective": selection_objective,
        "promotion_score": promotion_score,
        "pareto_promotion_gate": pareto_gate,
    }
    report = {
        "schema_version": 1,
        "state_hint": SANDBOX_STATE_HINT,
        "blueprint_id": blueprint["blueprint_id"],
        "candidate_family": family,
        "baseline": {
            "segment_score": baseline_score["segment_score"],
            "task1_prediction_score": baseline_score["task1_prediction_score"],
            "task1_total_score_proxy": _task1_total_score_proxy(
                prediction_score=baseline_score["task1_prediction_score"],
                inference_seconds=0.0,
            ),
        },
        "reference_audit": reference_audit,
        "best_candidate": {k: v for k, v in best.items() if k != "_prediction"},
        "verifier": verifier,
        "top_candidates": serializable_rows[:20],
        "candidate_pool_attribution": candidate_pool_attribution,
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
    if family == "task1_early_refinement":
        return _retrieval_refinement_parameter_grid(grid)
    if family == "task1_early_transition_expert":
        return _early_transition_expert_parameter_grid(grid)
    if family == "task1_front25_dynamics":
        return _front25_dynamics_parameter_grid(grid)
    if family == "task1_short_window_corrector":
        return _short_window_corrector_parameter_grid(grid)
    if family == "task1_sem_burgers_rollout":
        return _sem_burgers_parameter_grid(grid)
    if family == "task1_physics_residual_refinement":
        return _physics_residual_refinement_parameter_grid(grid)
    if family == "task1_pi_deeponet_correction_head":
        return _pi_deeponet_parameter_grid(grid)
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
