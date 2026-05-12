from __future__ import annotations

import numpy as np
from typing import Any

from pde_agent.core.hypothesis_lab_utils import _clamp_float
from pde_agent.core.manual_operator_tuning import score_burgers_segments


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
    early_score_delta = _segment_metric_delta(
        baseline_segments=baseline_segments,
        best_segments=best_segments,
        segment="early",
        metric="score",
        lower_is_better=False,
    )
    early_rel_mse_delta = _segment_metric_delta(
        baseline_segments=baseline_segments,
        best_segments=best_segments,
        segment="early",
        metric="rel_mse",
        lower_is_better=True,
    )
    middle_score_delta = _segment_metric_delta(
        baseline_segments=baseline_segments,
        best_segments=best_segments,
        segment="middle",
        metric="score",
        lower_is_better=False,
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
        "early_score_delta_vs_baseline": {
            "value": early_score_delta,
            "passed": early_score_delta is None or early_score_delta >= 0.0,
        },
        "early_rel_mse_delta_vs_baseline": {
            "value": early_rel_mse_delta,
            "passed": early_rel_mse_delta is None or early_rel_mse_delta >= 0.0,
        },
        "middle_score_guardrail": {
            "value": middle_score_delta,
            "passed": middle_score_delta is None or middle_score_delta >= 0.0,
        },
        "late_rmse_delta_vs_baseline": {
            "value": late_delta,
            "passed": late_delta is None or late_delta >= 0.0,
        },
        "late_rmse_guardrail": {
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
    full_score = score_burgers_segments(prediction=prediction, target=target)
    full_baseline = score_burgers_segments(
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
    early_score_delta = _segment_metric_delta(
        baseline_segments=full_baseline.get("segments", {}),
        best_segments=full_score.get("segments", {}),
        segment="early",
        metric="score",
        lower_is_better=False,
    )
    early_rel_mse_delta = _segment_metric_delta(
        baseline_segments=full_baseline.get("segments", {}),
        best_segments=full_score.get("segments", {}),
        segment="early",
        metric="rel_mse",
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
        "early_accuracy": {
            "sample_count": int(target.shape[0]),
            "early_score": full_score["segments"]["early"]["score"],
            "baseline_early_score": full_baseline["segments"]["early"]["score"],
            "early_rel_mse": full_score["segments"]["early"]["rel_mse"],
            "baseline_early_rel_mse": full_baseline["segments"]["early"][
                "rel_mse"
            ],
            "early_score_delta": early_score_delta,
            "early_rel_mse_delta": early_rel_mse_delta,
            "passed": early_rel_mse_delta is None or early_rel_mse_delta >= 0.0,
        },
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
        bool(views["early_accuracy"]["passed"]),
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


def _task1_early_segment_diagnosis(
    *,
    baseline_score: dict[str, Any],
    best_score: dict[str, Any],
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_early = baseline_score.get("segments", {}).get("early", {})
    candidate_early = best_score.get("segments", {}).get("early", {})
    candidate_middle = best_score.get("segments", {}).get("middle", {})
    candidate_late = best_score.get("segments", {}).get("late", {})
    early_score = _numeric(candidate_early.get("score"))
    middle_score = _numeric(candidate_middle.get("score"))
    late_score = _numeric(candidate_late.get("score"))
    early_rel_mse = _numeric(candidate_early.get("rel_mse"))
    baseline_early_rel_mse = _numeric(baseline_early.get("rel_mse"))
    early_rel_mse_delta = (
        None
        if early_rel_mse is None or baseline_early_rel_mse is None
        else baseline_early_rel_mse - early_rel_mse
    )
    reason_codes = []
    if early_score is not None and early_score < 20.0:
        reason_codes.append("early_score_low")
    if early_rel_mse is not None and early_rel_mse > 0.05:
        reason_codes.append("early_rel_mse_high")
    if (
        early_score is not None
        and middle_score is not None
        and early_score + 10.0 < middle_score
    ):
        reason_codes.append("early_weaker_than_middle")
    if (
        early_score is not None
        and late_score is not None
        and early_score + 10.0 < late_score
    ):
        reason_codes.append("early_weaker_than_late")
    decision = "early_segment_bottleneck" if reason_codes else "no_early_bottleneck"
    early_subsegments = _early_subsegment_diagnostics(
        prediction=prediction,
        baseline_prediction=baseline_prediction,
        target=target,
    )
    error_taxonomy = _early_error_taxonomy(
        prediction=prediction,
        baseline_prediction=baseline_prediction,
        target=target,
    )
    bottleneck_type = _primary_bottleneck_type(error_taxonomy)
    guardrail_diagnosis = _task1_guardrail_diagnosis(
        baseline_score=baseline_score,
        best_score=best_score,
    )
    candidate_trend_attribution = _candidate_trend_attribution(candidates)
    return {
        "schema_version": 2,
        "state_hint": "TASK1_EARLY_SEGMENT_DIAGNOSED",
        "decision": decision,
        "reason_codes": reason_codes,
        "bottleneck_type": bottleneck_type,
        "segments": {
            "baseline": {"early": baseline_early},
            "candidate": {
                "early": candidate_early,
                "middle": candidate_middle,
                "late": candidate_late,
            },
        },
        "early_subsegments": early_subsegments,
        "error_taxonomy": error_taxonomy,
        "guardrail_diagnosis": guardrail_diagnosis,
        "candidate_trend_attribution": candidate_trend_attribution,
        "deltas": {
            "early_rel_mse_delta_vs_baseline": early_rel_mse_delta,
            "early_score_delta_vs_baseline": _segment_metric_delta(
                baseline_segments=baseline_score.get("segments", {}),
                best_segments=best_score.get("segments", {}),
                segment="early",
                metric="score",
                lower_is_better=False,
            ),
        },
        "objective_recommendation": {
            "primary": "early_rel_mse",
            "secondary": "task1_total_score_proxy",
            "next_candidate_family": "task1_early_refinement",
            "guardrails": [
                "context_prefix_exact",
                "middle_score_non_regression",
                "late_rmse_non_regression",
                "inference_time_under_120s",
            ],
        },
        "action_directives": _early_action_directives(
            decision=decision,
            bottleneck_type=bottleneck_type,
            candidate_trend_attribution=candidate_trend_attribution,
        ),
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _early_subsegment_diagnostics(
    *,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    windows = [
        ("future_0_10", 0, 10),
        ("future_10_25", 10, 25),
        ("future_25_47", 25, 47),
    ]
    rows = []
    for name, future_start, future_stop in windows:
        candidate_rel_mse = _window_rel_mse(
            prediction=prediction,
            target=target,
            future_start=future_start,
            future_stop=future_stop,
        )
        baseline_rel_mse = _window_rel_mse(
            prediction=baseline_prediction,
            target=target,
            future_start=future_start,
            future_stop=future_stop,
        )
        rows.append(
            {
                "name": name,
                "future_start": future_start,
                "future_stop": future_stop,
                "time_start": 10 + future_start,
                "time_stop": 10 + future_stop,
                "candidate_rel_mse": candidate_rel_mse,
                "baseline_rel_mse": baseline_rel_mse,
                "rel_mse_delta_vs_baseline": baseline_rel_mse - candidate_rel_mse,
                "candidate_score": float(100.0 * np.exp(-20.0 * candidate_rel_mse)),
            }
        )
    return rows


def _window_rel_mse(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    future_start: int,
    future_stop: int,
) -> float:
    start = 10 + future_start
    stop = 10 + future_stop
    squared_error = (prediction[:, start:stop, :] - target[:, start:stop, :]) ** 2
    target_squared = target[:, start:stop, :] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_by_sample_time = (
            np.sum(squared_error, axis=2)
            / (np.sum(target_squared, axis=2) + 1e-12)
        )
    return float(np.mean(np.minimum(np.mean(rel_by_sample_time, axis=1), 5.0)))


def _early_error_taxonomy(
    *,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, dict[str, Any]]:
    pred = prediction[:, 10:57, :]
    base = baseline_prediction[:, 10:57, :]
    truth = target[:, 10:57, :]
    initial_derivative_error = _relative_mse(
        prediction[:, 10, :] - prediction[:, 9, :],
        target[:, 10, :] - target[:, 9, :],
    )
    phase = _phase_shift_evidence(pred=pred, truth=truth)
    bias = float(np.mean(pred - truth))
    target_scale = float(np.mean(np.abs(truth)) + 1e-12)
    amplitude_bias = abs(bias) / target_scale
    pred_hf = _high_frequency_energy(pred)
    target_hf = _high_frequency_energy(truth)
    base_hf = _high_frequency_energy(base)
    hf_ratio = pred_hf / (target_hf + 1e-12)
    base_hf_ratio = base_hf / (target_hf + 1e-12)
    return {
        "initial_derivative_mismatch": {
            "severity": _clamp_float(initial_derivative_error / 0.1, 0.0, 1.0),
            "triggered": initial_derivative_error > 0.05,
            "evidence": {
                "relative_mse": initial_derivative_error,
            },
        },
        "phase_shift": {
            "severity": _clamp_float(phase["improvement_ratio"] / 0.1, 0.0, 1.0),
            "triggered": bool(
                phase["best_shift"] != 0 and phase["improvement_ratio"] > 0.02
            ),
            "evidence": phase,
        },
        "amplitude_bias": {
            "severity": _clamp_float(amplitude_bias / 0.1, 0.0, 1.0),
            "triggered": amplitude_bias > 0.03,
            "evidence": {
                "mean_bias": bias,
                "relative_bias": amplitude_bias,
            },
        },
        "over_smoothing": {
            "severity": _clamp_float(max(0.0, 1.0 - hf_ratio) / 0.5, 0.0, 1.0),
            "triggered": hf_ratio < 0.7,
            "evidence": {
                "candidate_high_frequency_ratio": hf_ratio,
                "baseline_high_frequency_ratio": base_hf_ratio,
            },
        },
        "high_frequency_noise": {
            "severity": _clamp_float(max(0.0, hf_ratio - 1.0) / 0.5, 0.0, 1.0),
            "triggered": hf_ratio > 1.3,
            "evidence": {
                "candidate_high_frequency_ratio": hf_ratio,
                "baseline_high_frequency_ratio": base_hf_ratio,
            },
        },
    }


def _relative_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.sum((prediction - target) ** 2)
        / (np.sum(target**2) + 1e-12)
    )


def _phase_shift_evidence(*, pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    base_mse = float(np.mean((pred - truth) ** 2))
    best_shift = 0
    best_mse = base_mse
    for shift in [-3, -2, -1, 1, 2, 3]:
        shifted = np.roll(truth, shift=shift, axis=2)
        mse = float(np.mean((pred - shifted) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_shift = shift
    improvement = max(0.0, (base_mse - best_mse) / (base_mse + 1e-12))
    return {
        "best_shift": best_shift,
        "unshifted_mse": base_mse,
        "best_shifted_mse": best_mse,
        "improvement_ratio": float(improvement),
    }


def _high_frequency_energy(values: np.ndarray) -> float:
    spectrum = np.fft.rfft(values, axis=2)
    power = np.abs(spectrum) ** 2
    cutoff = max(1, int(power.shape[2] * 0.75))
    return float(np.mean(power[:, :, cutoff:]))


def _primary_bottleneck_type(taxonomy: dict[str, dict[str, Any]]) -> str:
    triggered = [
        (name, float(item.get("severity", 0.0)))
        for name, item in taxonomy.items()
        if item.get("triggered")
    ]
    if not triggered:
        return "none"
    triggered.sort(key=lambda item: item[1], reverse=True)
    return triggered[0][0]


def _task1_guardrail_diagnosis(
    *,
    baseline_score: dict[str, Any],
    best_score: dict[str, Any],
) -> dict[str, Any]:
    baseline_segments = baseline_score.get("segments", {})
    best_segments = best_score.get("segments", {})
    middle_delta = _segment_metric_delta(
        baseline_segments=baseline_segments,
        best_segments=best_segments,
        segment="middle",
        metric="score",
        lower_is_better=False,
    )
    late_delta = _segment_metric_delta(
        baseline_segments=baseline_segments,
        best_segments=best_segments,
        segment="late",
        metric="rmse",
        lower_is_better=True,
    )
    return {
        "middle": {
            "metric": "score",
            "candidate": best_segments.get("middle", {}).get("score"),
            "baseline": baseline_segments.get("middle", {}).get("score"),
            "delta_vs_baseline": middle_delta,
            "passed": middle_delta is None or middle_delta >= 0.0,
        },
        "late": {
            "metric": "rmse",
            "candidate": best_segments.get("late", {}).get("rmse"),
            "baseline": baseline_segments.get("late", {}).get("rmse"),
            "delta_vs_baseline": late_delta,
            "passed": late_delta is None or late_delta >= 0.0,
        },
    }


def _candidate_trend_attribution(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        item
        for item in candidates
        if isinstance(item.get("params"), dict)
        and isinstance(item.get("segments"), dict)
        and isinstance(item["segments"].get("early"), dict)
        and isinstance(item["segments"]["early"].get("rel_mse"), int | float)
    ]
    axes: dict[str, Any] = {}
    for axis in [
        "k",
        "mode",
        "early_blend_steps",
        "persistence_weight_start",
        "spectral_strength",
        "early_anchor",
    ]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            params = row["params"]
            if axis not in params:
                continue
            key = str(params[axis])
            grouped.setdefault(key, []).append(float(row["segments"]["early"]["rel_mse"]))
        if not grouped:
            continue
        summaries = []
        for value, metrics in grouped.items():
            summaries.append(
                {
                    "value": value,
                    "count": len(metrics),
                    "best_early_rel_mse": min(metrics),
                    "mean_early_rel_mse": float(np.mean(metrics)),
                }
            )
        summaries.sort(key=lambda item: item["best_early_rel_mse"])
        spread = (
            max(item["mean_early_rel_mse"] for item in summaries)
            - min(item["mean_early_rel_mse"] for item in summaries)
        )
        axes[axis] = {
            "best_values": [item["value"] for item in summaries[:3]],
            "value_summaries": summaries,
            "mean_rel_mse_spread": spread,
        }
    recommended_axis = None
    if axes:
        recommended_axis = max(
            axes,
            key=lambda axis: float(axes[axis]["mean_rel_mse_spread"]),
        )
    return {
        "primary_metric": "early_rel_mse",
        "candidate_count": len(rows),
        "axes": axes,
        "recommended_axis": recommended_axis,
    }


def _early_action_directives(
    *,
    decision: str,
    bottleneck_type: str,
    candidate_trend_attribution: dict[str, Any],
) -> list[dict[str, Any]]:
    target_family = "task1_early_refinement"
    if decision != "early_segment_bottleneck":
        return [
            {
                "directive": "maintain_total_score_search",
                "target_family": target_family,
                "rationale": "early segment is not currently the dominant bottleneck",
            }
        ]
    by_bottleneck = {
        "initial_derivative_mismatch": "expand_linear_velocity_anchor_search",
        "phase_shift": "add_phase_alignment_or_derivative_distance_candidates",
        "amplitude_bias": "add_short_horizon_affine_bias_calibration",
        "over_smoothing": "reduce_early_smoothing_and_anchor_weight",
        "high_frequency_noise": "increase_early_spectral_damping",
    }
    directive = by_bottleneck.get(bottleneck_type, "expand_early_anchor_search")
    recommended_axis = candidate_trend_attribution.get("recommended_axis")
    directives = [
        {
            "directive": directive,
            "target_family": target_family,
            "rationale": f"primary early bottleneck is {bottleneck_type}",
        }
    ]
    if isinstance(recommended_axis, str) and recommended_axis:
        directives.append(
            {
                "directive": f"prioritize_axis:{recommended_axis}",
                "target_family": target_family,
                "rationale": "top candidates show the largest early-rel-mse spread on this axis",
            }
        )
    return directives


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
    heldout_score = score_burgers_segments(
        prediction=prediction[heldout],
        target=target[heldout],
    )
    heldout_baseline = score_burgers_segments(
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


def _sandbox_pareto_promotion_gate(
    *,
    family: str,
    metric_results: dict[str, Any],
    selection_objective: dict[str, Any],
    candidate_budget: int,
    promotion_score: float,
) -> dict[str, Any]:
    total_axis = _pareto_metric_axis(
        metric_results=metric_results,
        metric_name="score_delta_vs_baseline",
        axis_name="task1_total_score_proxy",
        direction="maximize",
    )
    segment_axis = _pareto_metric_axis(
        metric_results=metric_results,
        metric_name="segment_score_delta_vs_baseline",
        axis_name="segment_score",
        direction="maximize",
    )
    early_axis = _pareto_metric_axis(
        metric_results=metric_results,
        metric_name="early_rel_mse_delta_vs_baseline",
        axis_name="early_rel_mse",
        direction="minimize",
    )
    middle_guardrail = _pareto_metric_axis(
        metric_results=metric_results,
        metric_name="middle_score_guardrail",
        axis_name="middle_score",
        direction="maximize",
    )
    late_guardrail = _pareto_metric_axis(
        metric_results=metric_results,
        metric_name="late_rmse_guardrail",
        axis_name="late_rmse",
        direction="minimize",
    )
    runtime_axis = _pareto_metric_axis(
        metric_results=metric_results,
        metric_name="inference_time_budget",
        axis_name="runtime_seconds",
        direction="minimize",
        positive_value_means_improved=False,
    )
    hard_constraints = {
        name: bool(metric_results.get(name, {}).get("passed"))
        for name in ["rules_passed", "context_preserved", "inference_time_budget"]
    }
    guardrail_axes = [middle_guardrail, late_guardrail]
    guardrails_passed = all(bool(axis.get("passed", True)) for axis in guardrail_axes)
    primary_axis = str(selection_objective.get("primary", "task1_total_score_proxy"))
    primary_improved = bool(
        early_axis["improved"]
        if primary_axis == "early_rel_mse"
        else total_axis["improved"]
    )
    total_passed = bool(total_axis["passed"])
    any_positive_evidence = any(
        bool(axis["improved"]) for axis in [total_axis, segment_axis, early_axis]
    )
    physics_residual_metric = metric_results.get("physics_residual_rmse_delta")
    physics_residual_required = family == "task1_physics_residual_refinement"
    physics_residual_passed = True
    physics_residual_axis = {
        "metric": "physics_residual_rmse_delta",
        "value": None,
        "direction": "minimize",
        "passed": True,
        "improved": False,
    }
    if isinstance(physics_residual_metric, dict):
        value = physics_residual_metric.get("value")
        physics_residual_passed = bool(physics_residual_metric.get("passed"))
        physics_residual_axis = {
            "metric": "physics_residual_rmse_delta",
            "value": value if isinstance(value, int | float) else None,
            "direction": "minimize",
            "passed": physics_residual_passed,
            "improved": bool(
                isinstance(value, int | float) and float(value) < 0.0
            ),
        }
    pi_correction_metric = metric_results.get("pi_deeponet_active_correction")
    pi_correction_required = family == "task1_pi_deeponet_correction_head"
    pi_correction_passed = True
    pi_correction_axis = {
        "metric": "pi_deeponet_active_correction",
        "value": None,
        "direction": "maximize",
        "passed": True,
        "improved": False,
    }
    if isinstance(pi_correction_metric, dict):
        value = pi_correction_metric.get("value")
        pi_correction_passed = bool(pi_correction_metric.get("passed"))
        pi_correction_axis = {
            "metric": "pi_deeponet_active_correction",
            "value": value if isinstance(value, int | float) else None,
            "direction": "maximize",
            "passed": pi_correction_passed,
            "improved": bool(
                isinstance(value, int | float) and float(value) > 1e-8
            ),
        }
    reason_codes: list[str] = []
    if not all(hard_constraints.values()):
        decision = "reject_or_revise"
        pareto_status = "dominated_or_invalid"
        reason_codes.extend(
            f"{name}_failed"
            for name, passed in hard_constraints.items()
            if not passed
        )
    elif physics_residual_required and not physics_residual_passed:
        decision = (
            "keep_as_evidence"
            if total_passed or any_positive_evidence
            else "reject_or_revise"
        )
        pareto_status = (
            "score_evidence_without_physics_residual_gain"
            if total_passed or any_positive_evidence
            else "dominated_or_invalid"
        )
        reason_codes.append("physics_residual_not_improved")
        if total_passed:
            reason_codes.append("total_score_passed")
    elif pi_correction_required and not pi_correction_passed:
        decision = (
            "keep_as_evidence"
            if total_passed or any_positive_evidence
            else "reject_or_revise"
        )
        pareto_status = (
            "score_evidence_without_active_pi_deeponet_correction"
            if total_passed or any_positive_evidence
            else "dominated_or_invalid"
        )
        reason_codes.append("pi_deeponet_correction_inactive")
        if total_passed:
            reason_codes.append("total_score_passed")
    elif total_passed and guardrails_passed:
        decision = "promote_to_strategy_proposal"
        pareto_status = "dominates_baseline"
        reason_codes.append("total_score_passed")
        if primary_improved:
            reason_codes.append("primary_axis_improved")
    elif primary_improved and guardrails_passed:
        decision = "keep_as_evidence"
        pareto_status = "tradeoff_evidence"
        reason_codes.append("primary_axis_improved_without_total_score")
    elif primary_improved or any_positive_evidence:
        decision = "continue_local_search"
        pareto_status = "tradeoff_needs_local_search"
        if not guardrails_passed:
            reason_codes.append("guardrail_regression")
        if primary_improved:
            reason_codes.append("primary_axis_improved")
        else:
            reason_codes.append("partial_axis_improvement")
    else:
        decision = "reject_or_revise"
        pareto_status = "dominated_or_invalid"
        reason_codes.append("no_pareto_axis_improved")
    next_command = (
        "record-strategy-proposal"
        if decision == "promote_to_strategy_proposal"
        else "revise-hypothesis"
    )
    return {
        "schema_version": 1,
        "state_hint": "PARETO_PROMOTION_GATE_EVALUATED",
        "family": family,
        "decision": decision,
        "promote_to_strategy_proposal": decision == "promote_to_strategy_proposal",
        "pareto_status": pareto_status,
        "reason_codes": reason_codes,
        "selection_objective": selection_objective,
        "promotion_score": promotion_score,
        "objective_axes": {
            "task1_total_score_proxy": total_axis,
            "segment_score": segment_axis,
            "early_rel_mse": early_axis,
            "middle_score": middle_guardrail,
            "late_rmse": late_guardrail,
            "runtime_seconds": runtime_axis,
            "physics_residual_rmse_delta": physics_residual_axis,
            "pi_deeponet_active_correction": pi_correction_axis,
            "candidate_budget": {
                "metric": "candidate_budget",
                "value": int(candidate_budget),
                "direction": "minimize",
                "passed": True,
                "improved": False,
            },
        },
        "hard_constraints": hard_constraints,
        "guardrails_passed": guardrails_passed,
        "next_command": next_command,
    }


def _pareto_metric_axis(
    *,
    metric_results: dict[str, Any],
    metric_name: str,
    axis_name: str,
    direction: str,
    positive_value_means_improved: bool = True,
) -> dict[str, Any]:
    metric = metric_results.get(metric_name)
    if not isinstance(metric, dict):
        return {
            "metric": metric_name,
            "axis": axis_name,
            "direction": direction,
            "value": None,
            "threshold": None,
            "passed": False,
            "improved": False,
        }
    value = metric.get("value")
    threshold = metric.get("threshold")
    improved = (
        positive_value_means_improved
        and isinstance(value, int | float)
        and float(value) > 0.0
    )
    return {
        "metric": metric_name,
        "axis": axis_name,
        "direction": direction,
        "value": value,
        "threshold": threshold,
        "passed": bool(metric.get("passed")),
        "improved": improved,
    }


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
