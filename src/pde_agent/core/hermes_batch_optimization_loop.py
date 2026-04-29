"""Batch optimization controller that uses the Hermes local judge as objective."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_local_judge import judge_hermes_submission
from pde_agent.core.hermes_postprocess_candidate import (
    generate_hermes_postprocess_candidate,
)


STATE_HINT = "HERMES_BATCH_OPTIMIZATION_LOOP_RAN"
CandidateRunner = Callable[..., dict[str, Any] | None]
Judge = Callable[..., dict[str, Any]]


class HermesBatchOptimizationLoopError(RuntimeError):
    pass


def run_hermes_batch_optimization_loop(
    *,
    output_dir: str | Path,
    initial_submission_dir: str | Path,
    max_rounds: int,
    patience: int,
    min_score_improvement: float,
    submit_score_threshold: float,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    task2_test_path: str | Path | None = None,
    task2_target_path: str | Path | None = None,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    candidate_submission_dirs: Sequence[str | Path] | None = None,
    candidate_generator: str = "none",
    postprocess_strategies: Sequence[str] | None = None,
    postprocess_blend_weights: Sequence[float] | None = None,
    submission_id: str = "team_hermes",
    candidate_runner: CandidateRunner | None = None,
    judge: Judge | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if max_rounds < 1:
        raise HermesBatchOptimizationLoopError("max_rounds must be positive")
    if patience < 0:
        raise HermesBatchOptimizationLoopError("patience must be non-negative")
    if min_score_improvement < 0.0:
        raise HermesBatchOptimizationLoopError("min_score_improvement must be non-negative")
    root = Path(output_dir)
    report_path = root / "hermes_batch_optimization_loop.json"
    if report_path.exists() and not force:
        raise HermesBatchOptimizationLoopError(
            f"batch optimization report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)

    run_judge = judge or judge_hermes_submission
    run_candidate = candidate_runner or _build_candidate_runner(
        candidate_submission_dirs=candidate_submission_dirs,
        candidate_generator=candidate_generator,
        initial_submission_dir=Path(initial_submission_dir),
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        postprocess_strategies=postprocess_strategies,
        postprocess_blend_weights=postprocess_blend_weights,
        submission_id=submission_id,
        force=force,
    )
    trajectory: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    no_improvement_count = 0
    stop_reason = "max_rounds_exhausted"

    initial = _evaluate_candidate(
        round_index=0,
        candidate_id="initial",
        submission_dir=Path(initial_submission_dir),
        output_dir=root,
        run_judge=run_judge,
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        task2_test_path=task2_test_path,
        task2_target_path=task2_target_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        metadata={"source": "initial_submission"},
    )
    best = initial
    trajectory.append({**initial, "improved": True, "score_delta_vs_best": 0.0})
    if _meets_submit_threshold(best, submit_score_threshold):
        stop_reason = "submit_threshold_met"
    else:
        for round_index in range(1, max_rounds + 1):
            candidate = run_candidate(
                round_index=round_index,
                output_dir=root / "candidates" / f"round_{round_index:03d}",
                best_candidate=best,
                trajectory=trajectory,
            )
            if candidate is None:
                stop_reason = "candidate_runner_exhausted"
                break
            submission_dir = _candidate_submission_dir(candidate)
            candidate_id = _candidate_id(candidate, round_index)
            evaluated = _evaluate_candidate(
                round_index=round_index,
                candidate_id=candidate_id,
                submission_dir=submission_dir,
                output_dir=root,
                run_judge=run_judge,
                task1_test_path=task1_test_path,
                task1_target_path=task1_target_path,
                task2_test_path=task2_test_path,
                task2_target_path=task2_target_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                metadata=_candidate_metadata(candidate),
            )
            delta = evaluated["score"] - best["score"]
            improved = delta >= min_score_improvement
            if delta > 0.0:
                best = evaluated
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            trajectory.append(
                {
                    **evaluated,
                    "improved": improved,
                    "score_delta_vs_best": delta,
                }
            )
            if _meets_submit_threshold(best, submit_score_threshold):
                stop_reason = "submit_threshold_met"
                break
            if no_improvement_count >= patience:
                stop_reason = "patience_exhausted"
                break

    if best is None:
        raise HermesBatchOptimizationLoopError("batch optimization produced no score")
    decision = _decision(best=best, stop_reason=stop_reason)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "loop_report_path": str(report_path),
        "parameters": {
            "max_rounds": max_rounds,
            "patience": patience,
            "min_score_improvement": min_score_improvement,
            "submit_score_threshold": submit_score_threshold,
            "candidate_generator": candidate_generator,
        },
        "round_count": len(trajectory),
        "trajectory": trajectory,
        "best_candidate": best,
        "stop_reason": stop_reason,
        "decision": decision["decision"],
        "next_command": decision["next_command"],
    }
    _write_json(report_path, report)
    return report


def _evaluate_candidate(
    *,
    round_index: int,
    candidate_id: str,
    submission_dir: Path,
    output_dir: Path,
    run_judge: Judge,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    task2_test_path: str | Path | None,
    task2_target_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    judge_path = output_dir / "judges" / f"round_{round_index:03d}_judge.json"
    judge_report = run_judge(
        submission_dir=submission_dir,
        output_path=judge_path,
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        task2_test_path=task2_test_path,
        task2_target_path=task2_target_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
    )
    score = _score(judge_report)
    return {
        "round_index": round_index,
        "candidate_id": candidate_id,
        "submission_dir": str(submission_dir),
        "judge_report_path": judge_report.get("judge_report_path", str(judge_path)),
        "rules_passed": bool(judge_report.get("rules_passed")),
        "score": score,
        "judge_decision": judge_report.get("decision"),
        "metadata": metadata,
    }


def _build_candidate_runner(
    *,
    candidate_submission_dirs: Sequence[str | Path] | None,
    candidate_generator: str,
    initial_submission_dir: Path,
    task1_test_path: str | Path | None,
    task2_test_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    postprocess_strategies: Sequence[str] | None,
    postprocess_blend_weights: Sequence[float] | None,
    submission_id: str,
    force: bool,
) -> CandidateRunner:
    if candidate_generator == "none":
        return _candidate_dir_runner(candidate_submission_dirs)
    if candidate_generator == "postprocess":
        return _postprocess_candidate_runner(
            initial_submission_dir=initial_submission_dir,
            task1_test_path=task1_test_path,
            task2_test_path=task2_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            strategies=postprocess_strategies,
            blend_weights=postprocess_blend_weights,
            submission_id=submission_id,
            force=force,
        )
    raise HermesBatchOptimizationLoopError(
        f"unsupported candidate_generator: {candidate_generator}"
    )


def _candidate_dir_runner(
    candidate_submission_dirs: Sequence[str | Path] | None,
) -> CandidateRunner:
    queue = [Path(path) for path in candidate_submission_dirs or []]

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(queue):
            return None
        submission_dir = queue[round_index - 1]
        return {
            "candidate_id": f"candidate_dir_{round_index}",
            "submission_dir": str(submission_dir),
            "source": "candidate_submission_dirs",
        }

    return run_candidate


def _postprocess_candidate_runner(
    *,
    initial_submission_dir: Path,
    task1_test_path: str | Path | None,
    task2_test_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    strategies: Sequence[str] | None,
    blend_weights: Sequence[float] | None,
    submission_id: str,
    force: bool,
) -> CandidateRunner:
    task_id, test_path = _single_task_context(
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
    )
    strategy_values = list(strategies or ["context_anchor_blend", "spectral_smooth"])
    weight_values = list(blend_weights or [0.25, 0.5, 0.75])
    if not strategy_values:
        raise HermesBatchOptimizationLoopError("postprocess strategies must not be empty")
    if not weight_values:
        raise HermesBatchOptimizationLoopError("postprocess blend weights must not be empty")
    schedule = list(zip(strategy_values, weight_values, strict=False))

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(schedule):
            return None
        strategy, blend_weight = schedule[round_index - 1]
        best_candidate = kwargs.get("best_candidate")
        base_submission_dir = initial_submission_dir
        if isinstance(best_candidate, dict):
            value = best_candidate.get("submission_dir")
            if isinstance(value, str) and value:
                base_submission_dir = Path(value)
        candidate = generate_hermes_postprocess_candidate(
            base_submission_dir=base_submission_dir,
            output_dir=Path(kwargs["output_dir"]),
            task_id=task_id,
            test_path=Path(test_path),
            strategy=strategy,
            blend_weight=float(blend_weight),
            submission_id=submission_id,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            force=force,
        )
        return {
            "candidate_id": candidate["candidate_id"],
            "submission_dir": candidate["submission_dir"],
            "source": "postprocess_generator",
            "strategy": candidate["strategy"],
            "blend_weight": candidate.get("blend_weight"),
            "candidate_report_path": candidate.get("candidate_report_path"),
        }

    return run_candidate


def _single_task_context(
    *,
    task1_test_path: str | Path | None,
    task2_test_path: str | Path | None,
) -> tuple[int, Path]:
    provided = [
        (task_id, Path(path))
        for task_id, path in ((1, task1_test_path), (2, task2_test_path))
        if path is not None
    ]
    if len(provided) != 1:
        raise HermesBatchOptimizationLoopError(
            "postprocess candidate generation requires exactly one task test path"
        )
    return provided[0]


def _candidate_submission_dir(candidate: dict[str, Any]) -> Path:
    value = candidate.get("submission_dir")
    if not isinstance(value, str) or not value:
        raise HermesBatchOptimizationLoopError("candidate missing submission_dir")
    return Path(value)


def _candidate_id(candidate: dict[str, Any], round_index: int) -> str:
    value = candidate.get("candidate_id")
    if isinstance(value, str) and value:
        return value
    return f"candidate_{round_index}"


def _candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"candidate_id", "submission_dir"}
    }


def _score(judge_report: dict[str, Any]) -> float:
    value = judge_report.get("total_score_proxy")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HermesBatchOptimizationLoopError("judge report missing total_score_proxy")
    return float(value)


def _meets_submit_threshold(
    candidate: dict[str, Any],
    submit_score_threshold: float,
) -> bool:
    return bool(candidate.get("rules_passed")) and candidate["score"] >= submit_score_threshold


def _decision(*, best: dict[str, Any], stop_reason: str) -> dict[str, str]:
    if stop_reason == "submit_threshold_met" and best.get("rules_passed"):
        return {
            "decision": "submit_best_candidate",
            "next_command": "submit-best-candidate",
        }
    if stop_reason == "candidate_runner_exhausted":
        return {
            "decision": "provide_more_candidates",
            "next_command": "run-hermes-batch-optimization-loop",
        }
    return {
        "decision": "iterate_more",
        "next_command": "run-hermes-batch-optimization-loop",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
