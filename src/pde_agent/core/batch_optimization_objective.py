from __future__ import annotations

from typing import Any

from pde_agent.core.batch_optimization_common import BatchOptimizationLoopError


INVALID_OBJECTIVE_SCORE = -1.0e100


def score(judge_report: dict[str, Any]) -> float:
    value = judge_report.get("total_score_proxy")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BatchOptimizationLoopError("judge report missing total_score_proxy")
    return float(value)


def with_objective(
    *,
    candidate: dict[str, Any],
    requested_objective: str,
    reference_candidate: dict[str, Any],
) -> dict[str, Any]:
    selected = select_objective(
        requested_objective=requested_objective,
        reference_candidate=reference_candidate,
    )
    signals = candidate_signals(candidate)
    candidate["objective"] = {
        "requested_objective": requested_objective,
        "selected_objective": selected,
        "objective_score": compute_objective_score(
            selected_objective=selected,
            candidate=candidate,
            signals=signals,
        ),
        "objective_reason": objective_reason(
            requested_objective=requested_objective,
            selected_objective=selected,
            reference_candidate=reference_candidate,
        ),
        **signals,
    }
    return candidate


def select_objective(
    *,
    requested_objective: str,
    reference_candidate: dict[str, Any],
) -> str:
    if requested_objective != "auto":
        return requested_objective
    signals = candidate_signals(reference_candidate)
    if not signals["rules_passed"]:
        return "feasibility"
    if signals["timeout_zeroed"] or signals["inference_time"] > 120.0:
        return "runtime"
    if signals["prediction_accuracy_score"] is not None or signals["segment_score"] is not None:
        return "accuracy"
    return "score"


def candidate_signals(candidate: dict[str, Any]) -> dict[str, Any]:
    signals = candidate.get("objective_signals")
    if isinstance(signals, dict):
        return signals
    return {
        "rules_passed": bool(candidate.get("rules_passed")),
        "timeout_zeroed": False,
        "train_time": None,
        "inference_time": None,
        "segment_score": None,
        "prediction_accuracy_score": None,
        "early_segment_score": None,
        "early_rel_mse": None,
    }


def objective_signals(judge_report: dict[str, Any]) -> dict[str, Any]:
    tasks = judge_report.get("tasks")
    task_values = [task for task in tasks.values()] if isinstance(tasks, dict) else []
    train_times: list[float] = []
    inference_times: list[float] = []
    segment_scores: list[float] = []
    prediction_scores: list[float] = []
    early_scores: list[float] = []
    early_rel_mses: list[float] = []
    timeout_zeroed = False
    for task in task_values:
        if not isinstance(task, dict):
            continue
        timing = task.get("timing")
        if isinstance(timing, dict):
            _append_number(train_times, timing.get("train_time"))
            _append_number(inference_times, timing.get("inference_time"))
        score_value = task.get("score")
        if isinstance(score_value, dict):
            timeout_zeroed = timeout_zeroed or bool(score_value.get("timeout_zeroed"))
            _append_number(segment_scores, score_value.get("segment_score"))
            _append_number(prediction_scores, score_value.get("prediction_accuracy_score"))
        validation = task.get("validation_score")
        if isinstance(validation, dict):
            if not segment_scores:
                _append_number(segment_scores, validation.get("segment_score"))
                _append_number(prediction_scores, validation.get("task_prediction_score"))
            segments = validation.get("segments")
            if isinstance(segments, dict):
                early = segments.get("early")
                if isinstance(early, dict):
                    _append_number(early_scores, early.get("score"))
                    _append_number(early_rel_mses, early.get("rel_mse"))
    inference_time = max(inference_times) if inference_times else None
    if inference_time is not None and inference_time > 120.0:
        timeout_zeroed = True
    return {
        "rules_passed": bool(judge_report.get("rules_passed")),
        "timeout_zeroed": timeout_zeroed,
        "train_time": sum(train_times) if train_times else None,
        "inference_time": inference_time,
        "segment_score": sum(segment_scores) if segment_scores else None,
        "prediction_accuracy_score": sum(prediction_scores)
        if prediction_scores
        else None,
        "early_segment_score": sum(early_scores) if early_scores else None,
        "early_rel_mse": sum(early_rel_mses) / float(len(early_rel_mses))
        if early_rel_mses
        else None,
    }


def compute_objective_score(
    *,
    selected_objective: str,
    candidate: dict[str, Any],
    signals: dict[str, Any],
) -> float:
    if selected_objective != "feasibility" and not signals.get("rules_passed"):
        return INVALID_OBJECTIVE_SCORE
    if selected_objective == "score":
        return float(candidate["score"])
    if selected_objective == "accuracy":
        for key in ("prediction_accuracy_score", "segment_score"):
            value = signals.get(key)
            if isinstance(value, int | float):
                return float(value)
        return float(candidate["score"])
    if selected_objective == "early_accuracy":
        value = signals.get("early_segment_score")
        if isinstance(value, int | float):
            return float(value)
        rel_mse = signals.get("early_rel_mse")
        if isinstance(rel_mse, int | float):
            return -float(rel_mse)
        return float(candidate["score"])
    if selected_objective == "runtime":
        value = signals.get("inference_time")
        if isinstance(value, int | float):
            return -float(value)
        return float("-inf")
    if selected_objective == "feasibility":
        value = signals.get("inference_time")
        runtime = float(value) if isinstance(value, int | float) else float("inf")
        feasible_bonus = 10000.0 if runtime <= 120.0 else 0.0
        rules_bonus = 100000.0 if signals.get("rules_passed") else 0.0
        return rules_bonus + feasible_bonus - runtime
    raise BatchOptimizationLoopError(
        f"unsupported selected objective: {selected_objective}"
    )


def objective_score(candidate: dict[str, Any]) -> float:
    objective = candidate.get("objective")
    if not isinstance(objective, dict):
        raise BatchOptimizationLoopError("candidate missing objective")
    value = objective.get("objective_score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BatchOptimizationLoopError("candidate missing objective_score")
    return float(value)


def objective_reason(
    *,
    requested_objective: str,
    selected_objective: str,
    reference_candidate: dict[str, Any],
) -> str:
    if requested_objective != "auto":
        return f"explicit_{selected_objective}_objective"
    signals = candidate_signals(reference_candidate)
    if selected_objective == "runtime":
        return "auto_runtime_because_reference_candidate_exceeds_inference_limit"
    if selected_objective == "feasibility":
        return "auto_feasibility_because_reference_candidate_fails_rules"
    if selected_objective == "accuracy":
        return "auto_accuracy_because_reference_candidate_is_within_runtime_gate"
    return "auto_score_fallback"


def meets_submit_threshold(
    candidate: dict[str, Any],
    submit_score_threshold: float,
) -> bool:
    return bool(candidate.get("rules_passed")) and candidate["score"] >= submit_score_threshold


def decision(*, best: dict[str, Any], stop_reason: str) -> dict[str, str]:
    if stop_reason == "submit_threshold_met" and best.get("rules_passed"):
        return {
            "decision": "submit_best_candidate",
            "next_command": "submit-best-candidate",
        }
    if stop_reason == "candidate_runner_exhausted":
        return {
            "decision": "provide_more_candidates",
            "next_command": "run-batch-optimization-loop",
        }
    return {
        "decision": "iterate_more",
        "next_command": "run-batch-optimization-loop",
    }


def _append_number(values: list[float], value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return
    values.append(float(value))
