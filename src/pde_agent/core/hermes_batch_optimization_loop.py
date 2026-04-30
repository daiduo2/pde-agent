"""Batch optimization controller that uses the Hermes local judge as objective."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory
from pde_agent.core.hermes_local_judge import judge_hermes_submission
from pde_agent.core.hermes_postprocess_candidate import (
    generate_hermes_postprocess_candidate,
)
from pde_agent.core.manual_operator_tuning import (
    linear_trend_prediction,
    load_hdf5_tensor,
    spectral_smooth_prediction,
)
from pde_agent.core.competition_candidate_materialization import (
    materialize_competition_candidate,
)
from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.unet_competition_prediction import predict_unet_competition


STATE_HINT = "HERMES_BATCH_OPTIMIZATION_LOOP_RAN"
SUPPORTED_LEARNING_OBJECTIVES = {
    "auto",
    "score",
    "accuracy",
    "runtime",
    "feasibility",
}
CandidateRunner = Callable[..., dict[str, Any] | None]
Judge = Callable[..., dict[str, Any]]
Clock = Callable[[], float]
MemoryBuilder = Callable[..., dict[str, Any]]


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
    learning_objective: str = "score",
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    task2_test_path: str | Path | None = None,
    task2_target_path: str | Path | None = None,
    unet_checkpoint_path: str | Path | None = None,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    candidate_submission_dirs: Sequence[str | Path] | None = None,
    candidate_generator: str = "none",
    postprocess_search_mode: str = "paired",
    postprocess_strategies: Sequence[str] | None = None,
    postprocess_blend_weights: Sequence[float] | None = None,
    postprocess_adaptive_step: float = 0.125,
    submission_id: str = "team_hermes",
    deadline_seconds: float | None = None,
    checkpoint_every: int = 0,
    memory_json_path: str | Path | None = None,
    avoid_repeating: bool = False,
    candidate_runner: CandidateRunner | None = None,
    judge: Judge | None = None,
    memory_builder: MemoryBuilder | None = None,
    clock: Clock | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if max_rounds < 1:
        raise HermesBatchOptimizationLoopError("max_rounds must be positive")
    if patience < 0:
        raise HermesBatchOptimizationLoopError("patience must be non-negative")
    if min_score_improvement < 0.0:
        raise HermesBatchOptimizationLoopError("min_score_improvement must be non-negative")
    if learning_objective not in SUPPORTED_LEARNING_OBJECTIVES:
        raise HermesBatchOptimizationLoopError(
            f"unsupported learning_objective: {learning_objective}"
        )
    if deadline_seconds is not None and deadline_seconds <= 0.0:
        raise HermesBatchOptimizationLoopError("deadline_seconds must be positive")
    if checkpoint_every < 0:
        raise HermesBatchOptimizationLoopError("checkpoint_every must be non-negative")
    root = Path(output_dir)
    report_path = root / "hermes_batch_optimization_loop.json"
    if report_path.exists() and not force:
        raise HermesBatchOptimizationLoopError(
            f"batch optimization report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)

    run_judge = judge or judge_hermes_submission
    run_memory_builder = memory_builder or build_hermes_iteration_memory
    now = clock or time.monotonic
    started_at = now()
    avoid_set = _load_avoid_set(memory_json_path) if avoid_repeating else set()
    skipped_candidates: list[dict[str, Any]] = []
    run_candidate = candidate_runner or _build_candidate_runner(
        candidate_submission_dirs=candidate_submission_dirs,
        candidate_generator=candidate_generator,
        initial_submission_dir=Path(initial_submission_dir),
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
        unet_checkpoint_path=unet_checkpoint_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        postprocess_strategies=postprocess_strategies,
        postprocess_blend_weights=postprocess_blend_weights,
        postprocess_search_mode=postprocess_search_mode,
        postprocess_adaptive_step=postprocess_adaptive_step,
        avoid_set=avoid_set,
        skipped_candidates=skipped_candidates,
        submission_id=submission_id,
        force=force,
    )
    trajectory: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    no_improvement_count = 0
    memory_checkpoints: list[dict[str, Any]] = []
    elapsed_seconds = 0.0
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
    initial = _with_objective(
        candidate=initial,
        requested_objective=learning_objective,
        reference_candidate=initial,
    )
    best = initial
    trajectory.append({**initial, "improved": True, "score_delta_vs_best": 0.0})
    if _meets_submit_threshold(best, submit_score_threshold):
        stop_reason = "submit_threshold_met"
    else:
        for round_index in range(1, max_rounds + 1):
            if _deadline_reached(
                started_at=started_at,
                now=now,
                deadline_seconds=deadline_seconds,
            ):
                elapsed_seconds = float(deadline_seconds)
                stop_reason = "deadline_exceeded"
                break
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
            evaluated = _with_objective(
                candidate=evaluated,
                requested_objective=learning_objective,
                reference_candidate=best,
            )
            delta = _objective_score(evaluated) - _objective_score(best)
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
            if checkpoint_every and round_index % checkpoint_every == 0:
                memory_checkpoints.append(
                    _checkpoint_memory(
                        root=root,
                        report_path=report_path,
                        run_memory_builder=run_memory_builder,
                        round_index=round_index,
                        trajectory=trajectory,
                        best=best,
                        stop_reason="checkpoint",
                    )
                )
            if _meets_submit_threshold(best, submit_score_threshold):
                stop_reason = "submit_threshold_met"
                break
            if no_improvement_count >= patience:
                stop_reason = "patience_exhausted"
                break

    if best is None:
        raise HermesBatchOptimizationLoopError("batch optimization produced no score")
    if checkpoint_every:
        final_round = len(trajectory) - 1
        if not memory_checkpoints or memory_checkpoints[-1]["round_index"] != final_round:
            memory_checkpoints.append(
                _checkpoint_memory(
                    root=root,
                    report_path=report_path,
                    run_memory_builder=run_memory_builder,
                    round_index=final_round,
                    trajectory=trajectory,
                    best=best,
                    stop_reason=stop_reason,
                )
            )
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
            "learning_objective": learning_objective,
            "candidate_generator": candidate_generator,
            "unet_checkpoint_path": str(unet_checkpoint_path)
            if unet_checkpoint_path is not None
            else None,
            "postprocess_search_mode": postprocess_search_mode,
            "postprocess_adaptive_step": postprocess_adaptive_step,
            "checkpoint_every": checkpoint_every,
            "avoid_repeating": avoid_repeating,
        },
        "budget": {
            "deadline_seconds": deadline_seconds,
            "elapsed_seconds": elapsed_seconds,
        },
        "round_count": len(trajectory),
        "trajectory": trajectory,
        "best_candidate": best,
        "skipped_candidates": skipped_candidates,
        "memory_checkpoints": memory_checkpoints,
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
        "objective_signals": _objective_signals(judge_report),
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
    unet_checkpoint_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    postprocess_strategies: Sequence[str] | None,
    postprocess_blend_weights: Sequence[float] | None,
    postprocess_search_mode: str,
    postprocess_adaptive_step: float,
    avoid_set: set[tuple[str, float]],
    skipped_candidates: list[dict[str, Any]],
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
            search_mode=postprocess_search_mode,
            adaptive_step=postprocess_adaptive_step,
            avoid_set=avoid_set,
            skipped_candidates=skipped_candidates,
            submission_id=submission_id,
            force=force,
        )
    if candidate_generator == "runtime_config_search":
        return _runtime_config_runner(
            unet_checkpoint_path=unet_checkpoint_path,
            task1_test_path=task1_test_path,
            task2_test_path=task2_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            force=force,
        )
    if candidate_generator == "fast_baseline_family":
        return _fast_baseline_runner(
            task1_test_path=task1_test_path,
            task2_test_path=task2_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            force=force,
        )
    if candidate_generator == "hybrid_gate":
        return _hybrid_gate_runner(
            unet_checkpoint_path=unet_checkpoint_path,
            task1_test_path=task1_test_path,
            task2_test_path=task2_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            force=force,
        )
    if candidate_generator == "linear_trend_spectral":
        return _linear_trend_spectral_runner(
            task1_test_path=task1_test_path,
            task2_test_path=task2_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
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
    search_mode: str,
    adaptive_step: float,
    avoid_set: set[tuple[str, float]],
    skipped_candidates: list[dict[str, Any]],
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
    if adaptive_step <= 0.0:
        raise HermesBatchOptimizationLoopError("postprocess adaptive step must be positive")
    schedule = _postprocess_schedule(
        strategies=strategy_values,
        blend_weights=weight_values,
        search_mode=search_mode,
    )
    tried: set[tuple[str, float]] = set()

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        next_pair = _next_postprocess_pair(
            round_index=round_index,
            schedule=schedule,
            search_mode=search_mode,
            tried=tried,
            avoid_set=avoid_set,
            skipped_candidates=skipped_candidates,
            best_candidate=kwargs.get("best_candidate"),
            adaptive_step=adaptive_step,
        )
        if next_pair is None:
            return None
        strategy, blend_weight = next_pair
        tried.add((strategy, blend_weight))
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


def _runtime_config_runner(
    *,
    unet_checkpoint_path: str | Path | None,
    task1_test_path: str | Path | None,
    task2_test_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    force: bool,
    configs: Sequence[tuple[int, int]] | None = None,
) -> CandidateRunner:
    if unet_checkpoint_path is None:
        raise HermesBatchOptimizationLoopError(
            "runtime_config_search requires unet_checkpoint_path"
        )
    task_id, test_path = _single_task_context(
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
    )
    schedule = list(configs or [(8, 1), (8, 2), (16, 1), (16, 2), (32, 2), (32, 4)])

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(schedule):
            return None
        inference_batch_size, torch_num_threads = schedule[round_index - 1]
        output_dir = Path(kwargs["output_dir"])
        prediction = predict_unet_competition(
            task_id=task_id,
            test_path=Path(test_path),
            checkpoint_path=Path(unet_checkpoint_path),
            output_dir=output_dir / "prediction",
            dataset_key=dataset_key,
            inference_batch_size=inference_batch_size,
            torch_num_threads=torch_num_threads,
            force=force,
        )
        materialized = materialize_competition_candidate(
            task_id=task_id,
            candidate_report_path=prediction["artifacts"][
                "candidate_execution_report_path"
            ],
            test_path=Path(test_path),
            output_dir=output_dir,
            submission_id=submission_id,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            source_prediction_key="prediction",
            force=force,
        )
        candidate_id = (
            f"runtime_config_b{inference_batch_size}_t{torch_num_threads}"
        )
        return {
            "candidate_id": candidate_id,
            "submission_dir": materialized["artifacts"]["submission_dir"],
            "source": "runtime_config_search",
            "inference_batch_size": inference_batch_size,
            "torch_num_threads": torch_num_threads,
            "candidate_report_path": prediction["artifacts"][
                "candidate_execution_report_path"
            ],
        }

    return run_candidate


def _fast_baseline_runner(
    *,
    task1_test_path: str | Path | None,
    task2_test_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    force: bool,
    methods: Sequence[str] | None = None,
) -> CandidateRunner:
    task_id, test_path = _single_task_context(
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
    )
    schedule = list(methods or ["persistence"])

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(schedule):
            return None
        method = schedule[round_index - 1]
        output_dir = Path(kwargs["output_dir"])
        submission = build_competition_submission(
            task_id=task_id,
            test_path=Path(test_path),
            output_dir=output_dir,
            submission_id=submission_id,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            method=method,
            train_time=0.0,
            force=force,
        )
        return {
            "candidate_id": f"fast_baseline_{method}",
            "submission_dir": submission["artifacts"]["submission_dir"],
            "source": "fast_baseline_family",
            "baseline_method": method,
        }

    return run_candidate


def _hybrid_gate_runner(
    *,
    unet_checkpoint_path: str | Path | None,
    task1_test_path: str | Path | None,
    task2_test_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    force: bool,
) -> CandidateRunner:
    runtime_runner = _runtime_config_runner(
        unet_checkpoint_path=unet_checkpoint_path,
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        submission_id=submission_id,
        force=force,
        configs=[(8, 1)],
    )
    fast_runner = _fast_baseline_runner(
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        submission_id=submission_id,
        force=force,
    )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index == 1:
            return runtime_runner(**kwargs)
        shifted = dict(kwargs)
        shifted["round_index"] = round_index - 1
        return fast_runner(**shifted)

    return run_candidate


def _linear_trend_spectral_runner(
    *,
    task1_test_path: str | Path | None,
    task2_test_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    force: bool,
) -> CandidateRunner:
    task_id, test_path = _single_task_context(
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
    )
    schedule = [
        {
            "slope_window": 5,
            "gain": -0.1,
            "decay": 60.0,
            "spectral_strength": 1.0,
        },
        {
            "slope_window": 5,
            "gain": -0.125,
            "decay": 50.0,
            "spectral_strength": 1.0,
        },
        {
            "slope_window": 6,
            "gain": -0.1,
            "decay": 60.0,
            "spectral_strength": 1.0,
        },
        {
            "slope_window": 5,
            "gain": -0.1,
            "decay": 60.0,
            "spectral_strength": 0.75,
        },
    ]

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(schedule):
            return None
        params = schedule[round_index - 1]
        output_dir = Path(kwargs["output_dir"])
        context = load_hdf5_tensor(test_path, dataset_key=dataset_key)
        if context.shape[1] != 10:
            context = context[:, :10, :]
        clip = (float(np.min(context)), float(np.max(context)))
        start = time.perf_counter()
        prediction = linear_trend_prediction(
            context=context,
            gain=float(params["gain"]),
            slope_window=int(params["slope_window"]),
            decay=float(params["decay"]),
            clip=clip,
        )
        strength = float(params["spectral_strength"])
        if strength > 0.0:
            prediction = spectral_smooth_prediction(
                prediction=prediction,
                strength=strength,
            )
        elapsed = time.perf_counter() - start
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = output_dir / "linear_trend_spectral_prediction.npz"
        np.savez_compressed(prediction_path, prediction=prediction)
        submission = build_competition_submission(
            task_id=task_id,
            test_path=Path(test_path),
            output_dir=output_dir / "submission_package",
            submission_id=submission_id,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            method="external_prediction",
            source_prediction_path=prediction_path,
            source_prediction_key="prediction",
            train_time=0.0,
            inference_time=elapsed,
            force=force,
        )
        candidate_id = (
            "linear_trend_spectral"
            f"_w{params['slope_window']}"
            f"_g{params['gain']:g}"
            f"_d{params['decay']:g}"
            f"_s{params['spectral_strength']:g}"
        )
        candidate_report_path = output_dir / "linear_trend_spectral_candidate.json"
        candidate_report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state_hint": "LINEAR_TREND_SPECTRAL_CANDIDATE_READY",
                    "candidate_id": candidate_id,
                    "params": params,
                    "artifacts": {
                        "candidate_prediction_path": str(prediction_path),
                        "submission_dir": submission["artifacts"]["submission_dir"],
                    },
                    "inference": {"elapsed_seconds": elapsed},
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "candidate_id": candidate_id,
            "submission_dir": submission["artifacts"]["submission_dir"],
            "source": "linear_trend_spectral_generator",
            "params": params,
            "candidate_report_path": str(candidate_report_path),
        }

    return run_candidate


def _postprocess_schedule(
    *,
    strategies: list[str],
    blend_weights: list[float],
    search_mode: str,
) -> list[tuple[str, float]]:
    weights = [_normalize_weight(value) for value in blend_weights]
    if search_mode == "paired":
        return list(zip(strategies, weights, strict=False))
    if search_mode in {"grid", "adaptive"}:
        return [(strategy, weight) for strategy in strategies for weight in weights]
    raise HermesBatchOptimizationLoopError(
        f"unsupported postprocess_search_mode: {search_mode}"
    )


def _next_postprocess_pair(
    *,
    round_index: int,
    schedule: list[tuple[str, float]],
    search_mode: str,
    tried: set[tuple[str, float]],
    avoid_set: set[tuple[str, float]],
    skipped_candidates: list[dict[str, Any]],
    best_candidate: object,
    adaptive_step: float,
) -> tuple[str, float] | None:
    if round_index <= len(schedule):
        for pair in schedule[round_index - 1 :]:
            if pair in tried:
                continue
            if pair in avoid_set:
                _record_skip(pair=pair, skipped_candidates=skipped_candidates)
                tried.add(pair)
                continue
            return pair
        return None
    if search_mode != "adaptive":
        return None
    best = _best_postprocess_metadata(best_candidate)
    if best is None:
        return None
    strategy, weight = best
    for candidate_weight in _adaptive_weight_neighbors(
        weight=weight,
        step=adaptive_step,
    ):
        pair = (strategy, candidate_weight)
        if pair in avoid_set:
            _record_skip(pair=pair, skipped_candidates=skipped_candidates)
            tried.add(pair)
            continue
        if pair not in tried:
            return pair
    return None


def _record_skip(
    *,
    pair: tuple[str, float],
    skipped_candidates: list[dict[str, Any]],
) -> None:
    strategy, weight = pair
    if any(
        item.get("strategy") == strategy and item.get("blend_weight") == weight
        for item in skipped_candidates
    ):
        return
    skipped_candidates.append(
        {
            "strategy": strategy,
            "blend_weight": weight,
            "reason": "memory_avoid_repeating",
        }
    )


def _best_postprocess_metadata(candidate: object) -> tuple[str, float] | None:
    if not isinstance(candidate, dict):
        return None
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("source") != "postprocess_generator":
        return None
    strategy = metadata.get("strategy")
    weight = metadata.get("blend_weight")
    if not isinstance(strategy, str) or not isinstance(weight, int | float):
        return None
    return strategy, _normalize_weight(float(weight))


def _adaptive_weight_neighbors(*, weight: float, step: float) -> list[float]:
    lower = _normalize_weight(weight - step)
    upper = _normalize_weight(weight + step)
    if lower == weight:
        return [upper] if upper != weight else []
    if upper == weight:
        return [lower] if lower != weight else []
    return [lower, upper]


def _normalize_weight(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 6)


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


def _with_objective(
    *,
    candidate: dict[str, Any],
    requested_objective: str,
    reference_candidate: dict[str, Any],
) -> dict[str, Any]:
    selected = _select_objective(
        requested_objective=requested_objective,
        reference_candidate=reference_candidate,
    )
    signals = _candidate_signals(candidate)
    candidate["objective"] = {
        "requested_objective": requested_objective,
        "selected_objective": selected,
        "objective_score": _compute_objective_score(
            selected_objective=selected,
            candidate=candidate,
            signals=signals,
        ),
        "objective_reason": _objective_reason(
            requested_objective=requested_objective,
            selected_objective=selected,
            reference_candidate=reference_candidate,
        ),
        **signals,
    }
    return candidate


def _select_objective(
    *,
    requested_objective: str,
    reference_candidate: dict[str, Any],
) -> str:
    if requested_objective != "auto":
        return requested_objective
    signals = _candidate_signals(reference_candidate)
    if not signals["rules_passed"]:
        return "feasibility"
    if signals["timeout_zeroed"] or signals["inference_time"] > 120.0:
        return "runtime"
    if signals["prediction_accuracy_score"] is not None or signals["segment_score"] is not None:
        return "accuracy"
    return "score"


def _candidate_signals(candidate: dict[str, Any]) -> dict[str, Any]:
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
    }


def _objective_signals(judge_report: dict[str, Any]) -> dict[str, Any]:
    tasks = judge_report.get("tasks")
    task_values = [task for task in tasks.values()] if isinstance(tasks, dict) else []
    train_times: list[float] = []
    inference_times: list[float] = []
    segment_scores: list[float] = []
    prediction_scores: list[float] = []
    timeout_zeroed = False
    for task in task_values:
        if not isinstance(task, dict):
            continue
        timing = task.get("timing")
        if isinstance(timing, dict):
            _append_number(train_times, timing.get("train_time"))
            _append_number(inference_times, timing.get("inference_time"))
        score = task.get("score")
        if isinstance(score, dict):
            timeout_zeroed = timeout_zeroed or bool(score.get("timeout_zeroed"))
            _append_number(segment_scores, score.get("segment_score"))
            _append_number(prediction_scores, score.get("prediction_accuracy_score"))
        validation = task.get("validation_score")
        if isinstance(validation, dict) and not segment_scores:
            _append_number(segment_scores, validation.get("segment_score"))
            _append_number(prediction_scores, validation.get("task_prediction_score"))
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
    }


def _append_number(values: list[float], value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return
    values.append(float(value))


def _compute_objective_score(
    *,
    selected_objective: str,
    candidate: dict[str, Any],
    signals: dict[str, Any],
) -> float:
    if selected_objective == "score":
        return float(candidate["score"])
    if selected_objective == "accuracy":
        for key in ("prediction_accuracy_score", "segment_score"):
            value = signals.get(key)
            if isinstance(value, int | float):
                return float(value)
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
    raise HermesBatchOptimizationLoopError(
        f"unsupported selected objective: {selected_objective}"
    )


def _objective_score(candidate: dict[str, Any]) -> float:
    objective = candidate.get("objective")
    if not isinstance(objective, dict):
        raise HermesBatchOptimizationLoopError("candidate missing objective")
    value = objective.get("objective_score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HermesBatchOptimizationLoopError("candidate missing objective_score")
    return float(value)


def _objective_reason(
    *,
    requested_objective: str,
    selected_objective: str,
    reference_candidate: dict[str, Any],
) -> str:
    if requested_objective != "auto":
        return f"explicit_{selected_objective}_objective"
    signals = _candidate_signals(reference_candidate)
    if selected_objective == "runtime":
        return "auto_runtime_because_reference_candidate_exceeds_inference_limit"
    if selected_objective == "feasibility":
        return "auto_feasibility_because_reference_candidate_fails_rules"
    if selected_objective == "accuracy":
        return "auto_accuracy_because_reference_candidate_is_within_runtime_gate"
    return "auto_score_fallback"


def _deadline_reached(
    *,
    started_at: float,
    now: Clock,
    deadline_seconds: float | None,
) -> bool:
    return deadline_seconds is not None and now() - started_at >= deadline_seconds


def _checkpoint_memory(
    *,
    root: Path,
    report_path: Path,
    run_memory_builder: MemoryBuilder,
    round_index: int,
    trajectory: list[dict[str, Any]],
    best: dict[str, Any],
    stop_reason: str,
) -> dict[str, Any]:
    checkpoint_report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "loop_report_path": str(report_path),
        "round_count": len(trajectory),
        "trajectory": trajectory,
        "best_candidate": best,
        "stop_reason": stop_reason,
        "decision": "checkpoint",
    }
    partial_path = root / "checkpoints" / f"round_{round_index:03d}_loop.json"
    _write_json(partial_path, checkpoint_report)
    memory = run_memory_builder(
        batch_loop_report_path=partial_path,
        output_path=root / "memory" / f"round_{round_index:03d}_memory.md",
        memory_json_path=root / "memory" / f"round_{round_index:03d}_memory.json",
        submission_dir=best.get("submission_dir") if isinstance(best, dict) else None,
        task_ids=None,
        force=True,
    )
    return {
        "round_index": round_index,
        "partial_loop_report_path": str(partial_path),
        "memory_markdown_path": memory["artifacts"]["memory_markdown_path"],
        "memory_json_path": memory["artifacts"]["memory_json_path"],
    }


def _load_avoid_set(path: str | Path | None) -> set[tuple[str, float]]:
    if path is None:
        return set()
    memory_file = Path(path)
    if not memory_file.exists():
        raise HermesBatchOptimizationLoopError(f"memory_json_path not found: {path}")
    try:
        payload = json.loads(memory_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesBatchOptimizationLoopError(f"invalid memory JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HermesBatchOptimizationLoopError("memory JSON must be an object")
    avoid = payload.get("avoid_repeating")
    if not isinstance(avoid, list):
        return set()
    result: set[tuple[str, float]] = set()
    for item in avoid:
        if not isinstance(item, dict):
            continue
        strategy = item.get("strategy")
        weight = item.get("blend_weight")
        if isinstance(strategy, str) and isinstance(weight, int | float):
            result.add((strategy, _normalize_weight(float(weight))))
    return result


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
