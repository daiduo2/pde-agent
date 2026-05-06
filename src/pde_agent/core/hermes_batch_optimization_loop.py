"""Batch optimization controller that uses the Hermes local judge as objective."""

from __future__ import annotations

import json
import copy
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory
from pde_agent.core.hermes_local_judge import judge_hermes_submission
from pde_agent.core.hermes_postprocess_candidate import (
    generate_hermes_postprocess_candidate,
)
from pde_agent.core.hermes_literature_expansion import (
    build_local_asset_algorithm_candidates,
)
from pde_agent.core.hypothesis_lab import run_blueprint_sandbox, run_hypothesis_lab
from pde_agent.core.local_strategy_library import build_local_strategy_library
from pde_agent.core.manual_operator_tuning import (
    linear_trend_prediction,
    load_hdf5_tensor,
    spectral_smooth_prediction,
)
from pde_agent.core.competition_candidate_materialization import (
    materialize_competition_candidate,
)
from pde_agent.core.competition_scoring import (
    EARLY_FUTURE_STOP,
    FUTURE_STEPS,
    MIDDLE_FUTURE_STOP,
    _late_segment,
    _rel_mse_segment,
)
from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.task1_checkpoint_front_window_adapter import (
    build_checkpoint_front_window_adapter_sources,
)
from pde_agent.core.burgers_pseudospectral import pseudospectral_burgers_rollout
from pde_agent.core.burgers_spectral_element import sem_burgers_rollout
from pde_agent.core.burgers_residual_refinement import residual_guided_refinement
from pde_agent.core.unet_competition_prediction import predict_unet_competition


STATE_HINT = "HERMES_BATCH_OPTIMIZATION_LOOP_RAN"
INVALID_OBJECTIVE_SCORE = -1.0e100
SUPPORTED_LEARNING_OBJECTIVES = {
    "auto",
    "score",
    "accuracy",
    "early_accuracy",
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
    hypothesis_blueprint_path: str | Path | None = None,
    hypothesis_reference_path: str | Path | None = None,
    hypothesis_reference_sample_limit: int | None = None,
    hypothesis_reference_sample_limits: Sequence[int] | None = None,
    hypothesis_candidate_options: Mapping[str, Any] | None = None,
    postprocess_search_mode: str = "paired",
    postprocess_strategies: Sequence[str] | None = None,
    postprocess_blend_weights: Sequence[float] | None = None,
    postprocess_adaptive_step: float = 0.125,
    submission_id: str = "team_hermes",
    timing_profile: str | None = None,
    timing_factor: float | None = None,
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
    memory_payload = _load_memory_json(memory_json_path) if avoid_repeating and memory_json_path is not None else {}
    avoid_set = _avoid_set_from_payload(memory_payload) if avoid_repeating else set()
    search_guidance = (
        _search_guidance_from_payload(memory_payload) if avoid_repeating else []
    )
    memory_constraints = _memory_constraints_summary(
        memory_json_path=memory_json_path,
        avoid_repeating=avoid_repeating,
        memory_payload=memory_payload,
        avoid_count=len(avoid_set),
        search_guidance_count=len(search_guidance),
    )
    skipped_candidates: list[dict[str, Any]] = []
    run_candidate = candidate_runner or _build_candidate_runner(
        candidate_submission_dirs=candidate_submission_dirs,
        candidate_generator=candidate_generator,
        hypothesis_blueprint_path=hypothesis_blueprint_path,
        hypothesis_reference_path=hypothesis_reference_path,
        hypothesis_reference_sample_limit=hypothesis_reference_sample_limit,
        hypothesis_reference_sample_limits=hypothesis_reference_sample_limits,
        hypothesis_candidate_options=hypothesis_candidate_options,
        initial_submission_dir=Path(initial_submission_dir),
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        task2_test_path=task2_test_path,
        unet_checkpoint_path=unet_checkpoint_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        postprocess_strategies=postprocess_strategies,
        postprocess_blend_weights=postprocess_blend_weights,
        postprocess_search_mode=postprocess_search_mode,
        postprocess_adaptive_step=postprocess_adaptive_step,
        avoid_set=avoid_set,
        search_guidance=search_guidance,
        skipped_candidates=skipped_candidates,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
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
            "hypothesis_blueprint_path": str(hypothesis_blueprint_path)
            if hypothesis_blueprint_path is not None
            else None,
            "hypothesis_reference_path": str(hypothesis_reference_path)
            if hypothesis_reference_path is not None
            else None,
            "hypothesis_reference_sample_limit": hypothesis_reference_sample_limit,
            "hypothesis_reference_sample_limits": list(hypothesis_reference_sample_limits or []),
            "hypothesis_candidate_options": dict(hypothesis_candidate_options or {}),
            "unet_checkpoint_path": str(unet_checkpoint_path)
            if unet_checkpoint_path is not None
            else None,
            "postprocess_search_mode": postprocess_search_mode,
            "postprocess_adaptive_step": postprocess_adaptive_step,
            "timing_profile": timing_profile,
            "timing_factor": timing_factor,
            "checkpoint_every": checkpoint_every,
            "avoid_repeating": avoid_repeating,
            "memory_json_path": str(memory_json_path)
            if memory_json_path is not None
            else None,
        },
        "budget": {
            "deadline_seconds": deadline_seconds,
            "elapsed_seconds": elapsed_seconds,
        },
        "round_count": len(trajectory),
        "trajectory": trajectory,
        "best_candidate": best,
        "skipped_candidates": skipped_candidates,
        "memory_constraints": memory_constraints,
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
    metadata_gate_issues = _candidate_metadata_gate_issues(metadata)
    rules_passed = bool(judge_report.get("rules_passed")) and not metadata_gate_issues
    objective_signals = _objective_signals(judge_report)
    objective_signals["rules_passed"] = rules_passed
    result = {
        "round_index": round_index,
        "candidate_id": candidate_id,
        "submission_dir": str(submission_dir),
        "judge_report_path": judge_report.get("judge_report_path", str(judge_path)),
        "rules_passed": rules_passed,
        "score": score,
        "objective_signals": objective_signals,
        "judge_decision": judge_report.get("decision"),
        "metadata": metadata,
    }
    if metadata_gate_issues:
        result["candidate_gate_issues"] = metadata_gate_issues
    return result


def _candidate_metadata_gate_issues(metadata: dict[str, Any]) -> list[str]:
    audit = metadata.get("reference_audit")
    if isinstance(audit, dict):
        if (
            audit.get("uses_target_as_reference") is True
            or audit.get("submission_validity") == "local_validation_only"
        ):
            return ["target_reference_local_validation_only"]
    return []


def _build_candidate_runner(
    *,
    candidate_submission_dirs: Sequence[str | Path] | None,
    candidate_generator: str,
    hypothesis_blueprint_path: str | Path | None,
    hypothesis_reference_path: str | Path | None,
    hypothesis_reference_sample_limit: int | None,
    hypothesis_reference_sample_limits: Sequence[int] | None,
    hypothesis_candidate_options: Mapping[str, Any] | None,
    initial_submission_dir: Path,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    task2_test_path: str | Path | None,
    unet_checkpoint_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    postprocess_strategies: Sequence[str] | None,
    postprocess_blend_weights: Sequence[float] | None,
    postprocess_search_mode: str,
    postprocess_adaptive_step: float,
    avoid_set: set[tuple[str, float]],
    search_guidance: Sequence[Mapping[str, Any]],
    skipped_candidates: list[dict[str, Any]],
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
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
            timing_profile=timing_profile,
            timing_factor=timing_factor,
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
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "fast_baseline_family":
        return _fast_baseline_runner(
            task1_test_path=task1_test_path,
            task2_test_path=task2_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
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
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "linear_trend_spectral":
        return _linear_trend_spectral_runner(
            task1_test_path=task1_test_path,
            task2_test_path=task2_test_path,
            hypothesis_blueprint_path=hypothesis_blueprint_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab":
        return _hypothesis_lab_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit_schedule=[hypothesis_reference_sample_limit],
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_reference_search":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_reference_search requires hypothesis_reference_path"
            )
        sample_limits = list(hypothesis_reference_sample_limits or [64, 128, 256, 512])
        if not sample_limits:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_reference_search requires non-empty sample limits"
            )
        return _hypothesis_lab_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit_schedule=sample_limits,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_reference_early_patch":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_reference_early_patch requires hypothesis_reference_path"
            )
        sample_limits = list(hypothesis_reference_sample_limits or [64, 128, 256])
        return _hypothesis_lab_reference_early_patch_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit_schedule=sample_limits,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_early_search":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_early_search requires hypothesis_reference_path"
            )
        return _hypothesis_lab_early_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit=hypothesis_reference_sample_limit,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_early_transition_search":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_early_transition_search requires hypothesis_reference_path"
            )
        return _hypothesis_lab_early_transition_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit=hypothesis_reference_sample_limit,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_front25_dynamics_smoke":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_front25_dynamics_smoke requires hypothesis_reference_path"
            )
        return _hypothesis_lab_front25_dynamics_smoke_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit=hypothesis_reference_sample_limit,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_short_window_corrector_smoke":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_short_window_corrector_smoke requires hypothesis_reference_path"
            )
        return _hypothesis_lab_short_window_corrector_smoke_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit=hypothesis_reference_sample_limit,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_short_window_corrector_full":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_short_window_corrector_full requires hypothesis_reference_path"
            )
        return _hypothesis_lab_short_window_corrector_smoke_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit=hypothesis_reference_sample_limit,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
            search_budget="full",
        )
    if candidate_generator == "hypothesis_lab_short_window_corrector_patch_sources":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_short_window_corrector_patch_sources requires hypothesis_reference_path"
            )
        if task1_test_path is None or task1_target_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_short_window_corrector_patch_sources requires task1_test_path and task1_target_path"
            )
        return _hypothesis_lab_short_window_corrector_patch_sources_runner(
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            reference_path=Path(hypothesis_reference_path),
            reference_sample_limit=hypothesis_reference_sample_limit,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_transition_early_patch":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_transition_early_patch requires hypothesis_reference_path"
            )
        return _hypothesis_lab_transition_early_patch_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit=hypothesis_reference_sample_limit,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_middle_late_patch_search":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_middle_late_patch_search requires hypothesis_reference_path"
            )
        return _hypothesis_lab_middle_late_patch_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit=hypothesis_reference_sample_limit,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        if hypothesis_reference_path is None:
            raise HermesBatchOptimizationLoopError(
                "hypothesis_lab_early_subsegment_patch_search requires hypothesis_reference_path"
            )
        return _hypothesis_lab_early_subsegment_patch_runner(
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=hypothesis_reference_path,
            reference_sample_limit=hypothesis_reference_sample_limit,
            candidate_options=hypothesis_candidate_options,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "early_subsegment_patch_sources":
        if task1_test_path is None or task1_target_path is None:
            raise HermesBatchOptimizationLoopError(
                "early_subsegment_patch_sources requires task1_test_path and task1_target_path"
            )
        return _early_subsegment_patch_sources_runner(
            candidate_submission_dirs=candidate_submission_dirs,
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "checkpoint_front_window_adapter_patch_sources":
        if task1_test_path is None or task1_target_path is None:
            raise HermesBatchOptimizationLoopError(
                "checkpoint_front_window_adapter_patch_sources requires task1_test_path and task1_target_path"
            )
        return _checkpoint_front_window_adapter_patch_sources_runner(
            checkpoint_submission_dirs=candidate_submission_dirs,
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "checkpoint_window_gate":
        if task1_test_path is None:
            raise HermesBatchOptimizationLoopError(
                "checkpoint_window_gate requires task1_test_path"
            )
        return _checkpoint_window_gate_runner(
            checkpoint_submission_dirs=candidate_submission_dirs,
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "sem_window_blend":
        if task1_test_path is None:
            raise HermesBatchOptimizationLoopError(
                "sem_window_blend requires task1_test_path"
            )
        return _sem_window_blend_runner(
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "sem_window_gate_blend":
        if task1_test_path is None:
            raise HermesBatchOptimizationLoopError(
                "sem_window_gate_blend requires task1_test_path"
            )
        return _sem_window_gate_blend_runner(
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "sem_window_gate_blend_search":
        if task1_test_path is None or task1_target_path is None:
            raise HermesBatchOptimizationLoopError(
                "sem_window_gate_blend_search requires task1_test_path and task1_target_path"
            )
        return _sem_window_gate_blend_search_runner(
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "sem_window_gate_blend_multi_search":
        if task1_test_path is None or task1_target_path is None:
            raise HermesBatchOptimizationLoopError(
                "sem_window_gate_blend_multi_search requires task1_test_path and task1_target_path"
            )
        return _sem_window_gate_blend_multi_search_runner(
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    if candidate_generator == "sem_window_gate_blend_multi_early_search":
        if task1_test_path is None or task1_target_path is None:
            raise HermesBatchOptimizationLoopError(
                "sem_window_gate_blend_multi_early_search requires task1_test_path and task1_target_path"
            )
        return _sem_window_gate_blend_multi_search_runner(
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            search_guidance=search_guidance,
            selection_metric="early_segment_score",
            force=force,
        )
    if candidate_generator == "sem_window_gate_blend_multi_early_smoke":
        if task1_test_path is None or task1_target_path is None:
            raise HermesBatchOptimizationLoopError(
                "sem_window_gate_blend_multi_early_smoke requires task1_test_path and task1_target_path"
            )
        return _sem_window_gate_blend_multi_search_runner(
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            search_guidance=search_guidance,
            selection_metric="early_segment_score",
            search_budget="smoke",
            force=force,
        )
    if candidate_generator == "physics_residual_nudge":
        if task1_test_path is None:
            raise HermesBatchOptimizationLoopError(
                "physics_residual_nudge requires task1_test_path"
            )
        return _physics_residual_nudge_runner(
            initial_submission_dir=initial_submission_dir,
            task1_test_path=Path(task1_test_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
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


def _checkpoint_window_gate_runner(
    *,
    checkpoint_submission_dirs: Sequence[str | Path] | None,
    initial_submission_dir: Path,
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    checkpoint_dirs = [Path(path) for path in checkpoint_submission_dirs or []]
    if not checkpoint_dirs:
        raise HermesBatchOptimizationLoopError(
            "checkpoint_window_gate requires at least one candidate_submission_dir"
        )
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=initial_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    checkpoint_predictions = {
        checkpoint_dir: _load_submission_prediction(
            submission_dir=checkpoint_dir,
            prediction_dataset_key=prediction_dataset_key,
        )
        for checkpoint_dir in checkpoint_dirs
    }
    gate_specs = _checkpoint_window_gate_specs(
        checkpoint_submission_dirs=checkpoint_dirs,
        context=context,
        base_prediction=base_prediction,
        checkpoint_predictions=checkpoint_predictions,
    )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(gate_specs):
            return None
        spec = dict(gate_specs[round_index - 1])
        checkpoint_submission_dir = spec.pop("checkpoint_submission_dir")
        spec.pop("expected_selected_sample_count", None)
        return _build_checkpoint_window_gate_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_submission_dir=initial_submission_dir,
            checkpoint_submission_dir=checkpoint_submission_dir,
            task1_test_path=task1_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
            **spec,
        )

    return run_candidate


def _checkpoint_window_gate_specs(
    *,
    checkpoint_submission_dirs: Sequence[Path],
    context: np.ndarray,
    base_prediction: np.ndarray,
    checkpoint_predictions: Mapping[Path, np.ndarray],
) -> list[dict[str, Any]]:
    manual_specs: list[dict[str, Any]] = [
        {
            "feature_name": "checkpoint_jump35",
            "direction": "gt",
            "threshold": 0.0001483375398526754,
            "window_start": 35,
            "window_stop": 57,
        },
        {
            "feature_name": "checkpoint_jump35",
            "direction": "gt",
            "threshold": 0.00017371984224990293,
            "window_start": 35,
            "window_stop": 57,
        },
        {
            "feature_name": "context_velocity_energy",
            "direction": "gt",
            "threshold": 0.005039689082863085,
            "window_start": 35,
            "window_stop": 57,
        },
    ]
    windows = [(10, 20), (20, 35), (35, 57), (57, 105), (105, 200)]
    disagreement_feature_windows = [
        (105, 200),
        (57, 105),
        (35, 57),
        (20, 35),
        (10, 20),
    ]
    feature_names = [
        "checkpoint_jump35",
        "context_velocity_energy",
        "checkpoint_base_energy_ratio_35_57",
        *[
            f"base_checkpoint_disagreement_{window_start}_{window_stop}"
            for window_start, window_stop in disagreement_feature_windows
        ],
    ]
    quantiles = [0.10, 0.18, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90]
    buckets: list[list[dict[str, Any]]] = []
    seen: set[tuple[Path, str, str, float, int, int]] = set()
    for checkpoint_dir in checkpoint_submission_dirs:
        local_specs: list[dict[str, Any]] = []
        checkpoint_prediction = checkpoint_predictions[checkpoint_dir]
        if checkpoint_prediction.shape != base_prediction.shape:
            raise HermesBatchOptimizationLoopError(
                "checkpoint gate base and checkpoint predictions must have matching shapes"
            )
        for manual_spec in manual_specs:
            _append_checkpoint_window_gate_spec(
                specs=local_specs,
                seen=seen,
                checkpoint_dir=checkpoint_dir,
                context=context,
                base_prediction=base_prediction,
                checkpoint_prediction=checkpoint_prediction,
                **manual_spec,
            )
        for feature_name in feature_names:
            values = _checkpoint_window_gate_feature(
                context=context,
                base_prediction=base_prediction,
                checkpoint_prediction=checkpoint_prediction,
                feature_name=feature_name,
            )
            for quantile in quantiles:
                threshold = float(np.quantile(values, quantile))
                for window_start, window_stop in windows:
                    _append_checkpoint_window_gate_spec(
                        specs=local_specs,
                        seen=seen,
                        checkpoint_dir=checkpoint_dir,
                        context=context,
                        base_prediction=base_prediction,
                        checkpoint_prediction=checkpoint_prediction,
                        feature_name=feature_name,
                        direction="gt",
                        threshold=threshold,
                        window_start=window_start,
                        window_stop=window_stop,
                    )
        buckets.append(local_specs)
    specs: list[dict[str, Any]] = []
    max_bucket_length = max((len(bucket) for bucket in buckets), default=0)
    for index in range(max_bucket_length):
        for bucket in buckets:
            if index < len(bucket):
                specs.append(bucket[index])
    return specs


def _sem_window_blend_runner(
    *,
    initial_submission_dir: Path,
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    sem_param_grid: list[dict[str, Any]] = [
        {
            "viscosity": 0.001,
            "dt": 0.03,
            "element_count": 16,
            "polynomial_order": 8,
            "substeps": 4,
            "filter_strength": 0.02,
            "interface_penalty": 0.02,
        },
        {
            "viscosity": 0.001,
            "dt": 0.05,
            "element_count": 16,
            "polynomial_order": 8,
            "substeps": 4,
            "filter_strength": 0.02,
            "interface_penalty": 0.02,
        },
    ]
    sem_predictions = [
        sem_burgers_rollout(context=context, **params) for params in sem_param_grid
    ]
    windows = [(10, 20), (20, 35), (35, 57), (57, 105), (10, 57), (10, 105)]
    blend_weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    specs = [
        {
            "sem_prediction": sem_prediction,
            "sem_params": sem_params,
            "window_start": window_start,
            "window_stop": window_stop,
            "blend_weight": blend_weight,
        }
        for sem_prediction, sem_params in zip(sem_predictions, sem_param_grid, strict=True)
        for window_start, window_stop in windows
        for blend_weight in blend_weights
    ]

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(specs):
            return None
        return _build_sem_window_blend_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_submission_dir=initial_submission_dir,
            task1_test_path=task1_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
            **specs[round_index - 1],
        )

    return run_candidate


def _build_sem_window_blend_candidate(
    *,
    output_dir: Path,
    base_submission_dir: Path,
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    sem_prediction: np.ndarray,
    sem_params: dict[str, Any],
    window_start: int,
    window_stop: int,
    blend_weight: float,
    force: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=base_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    sem_array = np.asarray(sem_prediction, dtype=np.float32)
    if sem_array.shape != base_prediction.shape:
        raise HermesBatchOptimizationLoopError(
            "SEM window blend base and SEM predictions must have matching shapes"
        )
    if context.shape[0] != base_prediction.shape[0]:
        raise HermesBatchOptimizationLoopError(
            "SEM window blend context and prediction sample counts must match"
        )
    start_index = max(10, int(window_start))
    stop_index = min(int(window_stop), base_prediction.shape[1])
    if stop_index <= start_index:
        raise HermesBatchOptimizationLoopError(
            "SEM window blend requires window_stop > window_start"
        )
    weight = float(blend_weight)
    if weight < 0.0 or weight > 1.0:
        raise HermesBatchOptimizationLoopError(
            "SEM window blend weight must be between 0 and 1"
        )
    blended = np.array(base_prediction, copy=True)
    blended[:, start_index:stop_index, :] = (
        (1.0 - weight) * blended[:, start_index:stop_index, :]
        + weight * sem_array[:, start_index:stop_index, :]
    )
    blended[:, :10, :] = context
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "sem_window_blend_prediction.npz"
    np.savez_compressed(prediction_path, prediction=blended.astype(np.float32))
    elapsed = time.perf_counter() - start
    submission = build_competition_submission(
        task_id=1,
        test_path=task1_test_path,
        output_dir=output_dir / "submission_package",
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=elapsed,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    candidate_id = (
        "sem_window_blend"
        f"_dt{float(sem_params.get('dt', 0.0)):g}"
        f"_w{weight:g}"
        f"_{start_index}_{stop_index}"
    )
    report_path = output_dir / "sem_window_blend_candidate.json"
    report = {
        "schema_version": 1,
        "state_hint": "SEM_WINDOW_BLEND_CANDIDATE_READY",
        "candidate_id": candidate_id,
        "base_submission_dir": str(base_submission_dir),
        "sem_params": sem_params,
        "window_start": start_index,
        "window_stop": stop_index,
        "blend_weight": weight,
        "prediction_path": str(prediction_path),
        "submission_dir": submission["artifacts"]["submission_dir"],
        "elapsed_seconds": elapsed,
    }
    _write_json(report_path, report)
    return {
        "candidate_id": candidate_id,
        "submission_dir": submission["artifacts"]["submission_dir"],
        "source": "sem_window_blend",
        "candidate_report_path": str(report_path),
        "base_submission_dir": str(base_submission_dir),
        "sem_params": sem_params,
        "window_start": start_index,
        "window_stop": stop_index,
        "blend_weight": weight,
    }


def _sem_window_gate_blend_runner(
    *,
    initial_submission_dir: Path,
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=initial_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    sem_params = {
        "viscosity": 0.001,
        "dt": 0.05,
        "element_count": 16,
        "polynomial_order": 8,
        "substeps": 4,
        "filter_strength": 0.02,
        "interface_penalty": 0.02,
    }
    sem_prediction = sem_burgers_rollout(context=context, **sem_params)
    features = [
        "sem_base_disagreement_20_57",
        "sem_base_disagreement_10_105",
        "sem_base_disagreement_57_105",
        "context_velocity_energy",
    ]
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    windows = [(10, 105), (10, 57), (20, 57), (57, 105)]
    blend_weights = [0.1, 0.15, 0.2, 0.25]
    specs: list[dict[str, Any]] = []
    for feature_name in features:
        values = _sem_window_gate_feature(
            context=context,
            base_prediction=base_prediction,
            sem_prediction=sem_prediction,
            feature_name=feature_name,
        )
        for direction in ["lt", "gt"]:
            for quantile in quantiles:
                threshold = float(np.quantile(values, quantile))
                mask = values < threshold if direction == "lt" else values > threshold
                if int(np.sum(mask)) in {0, int(mask.shape[0])}:
                    continue
                for window_start, window_stop in windows:
                    for blend_weight in blend_weights:
                        specs.append(
                            {
                                "sem_prediction": sem_prediction,
                                "sem_params": sem_params,
                                "feature_name": feature_name,
                                "direction": direction,
                                "threshold": threshold,
                                "window_start": window_start,
                                "window_stop": window_stop,
                                "blend_weight": blend_weight,
                            }
                        )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(specs):
            return None
        return _build_sem_window_gate_blend_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_submission_dir=initial_submission_dir,
            task1_test_path=task1_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
            **specs[round_index - 1],
        )

    return run_candidate


def _sem_window_gate_blend_search_runner(
    *,
    initial_submission_dir: Path,
    task1_test_path: Path,
    task1_target_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=initial_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    sem_params = {
        "viscosity": 0.001,
        "dt": 0.05,
        "element_count": 16,
        "polynomial_order": 8,
        "substeps": 4,
        "filter_strength": 0.02,
        "interface_penalty": 0.02,
    }
    sem_prediction = sem_burgers_rollout(context=context, **sem_params)
    specs = _sem_window_gate_blend_search_specs(
        context=context,
        base_prediction=base_prediction,
        sem_prediction=sem_prediction,
    )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        if int(kwargs["round_index"]) != 1:
            return None
        return _build_sem_window_gate_blend_search_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_submission_dir=initial_submission_dir,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
            sem_prediction=sem_prediction,
            sem_params=sem_params,
            specs=specs,
        )

    return run_candidate


def _sem_window_gate_blend_multi_search_runner(
    *,
    initial_submission_dir: Path,
    task1_test_path: Path,
    task1_target_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
    search_guidance: Sequence[Mapping[str, Any]] | None = None,
    selection_metric: str = "segment_score",
    search_budget: str = "full",
) -> CandidateRunner:
    if search_budget not in {"full", "smoke"}:
        raise HermesBatchOptimizationLoopError(
            f"unsupported SEM multi-search budget: {search_budget}"
        )
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=initial_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    sem_param_grid = [] if search_budget == "smoke" else [
        {
            "source": "sem",
            "viscosity": 0.001,
            "dt": 0.03,
            "element_count": 16,
            "polynomial_order": 8,
            "substeps": 4,
            "filter_strength": 0.02,
            "interface_penalty": 0.02,
        },
        {
            "source": "sem",
            "viscosity": 0.001,
            "dt": 0.05,
            "element_count": 16,
            "polynomial_order": 8,
            "substeps": 4,
            "filter_strength": 0.02,
            "interface_penalty": 0.02,
        },
        {
            "source": "sem",
            "viscosity": 0.001,
            "dt": 0.05,
            "element_count": 16,
            "polynomial_order": 8,
            "substeps": 4,
            "filter_strength": 0.0,
            "interface_penalty": 0.02,
        },
    ]
    sem_candidates = [
        {
            "sem_prediction": sem_burgers_rollout(
                context=context,
                **{key: value for key, value in params.items() if key != "source"},
            ),
            "sem_params": params,
        }
        for params in sem_param_grid
    ]
    pseudospectral_param_grid = [
        {
            "source": "pseudospectral",
            "viscosity": 0.001,
            "dt": 0.05,
            "substeps": 4,
            "filter_strength": 0.0,
            "dealias_fraction": 2.0 / 3.0,
        },
        {
            "source": "pseudospectral",
            "viscosity": 0.001,
            "dt": 0.05,
            "substeps": 4,
            "filter_strength": 0.02,
            "dealias_fraction": 2.0 / 3.0,
        },
    ]
    if search_budget == "smoke":
        pseudospectral_param_grid = pseudospectral_param_grid[:1]
    for params in pseudospectral_param_grid:
        sem_candidates.append(
            {
                "sem_prediction": pseudospectral_burgers_rollout(
                    context=context,
                    viscosity=float(params["viscosity"]),
                    dt=float(params["dt"]),
                    substeps=int(params["substeps"]),
                    filter_strength=float(params["filter_strength"]),
                    dealias_fraction=float(params["dealias_fraction"]),
                ),
                "sem_params": params,
            }
        )
    specs = _sem_window_gate_blend_search_specs(
        context=context,
        base_prediction=base_prediction,
        sem_prediction=np.asarray(sem_candidates[0]["sem_prediction"]),
        search_budget=search_budget,
    )
    specs = _append_sem_window_memory_guidance_specs(
        specs=specs,
        search_guidance=search_guidance or [],
        context=context,
        base_prediction=base_prediction,
        sem_prediction=np.asarray(sem_candidates[0]["sem_prediction"]),
    )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        if int(kwargs["round_index"]) != 1:
            return None
        return _build_sem_window_gate_blend_multi_search_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_submission_dir=initial_submission_dir,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
            sem_candidates=sem_candidates,
            specs=specs,
            selection_metric=selection_metric,
        )

    return run_candidate


def _sem_window_gate_blend_search_specs(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    sem_prediction: np.ndarray,
    search_budget: str = "full",
) -> list[dict[str, Any]]:
    if search_budget not in {"full", "smoke"}:
        raise HermesBatchOptimizationLoopError(
            f"unsupported SEM gate spec budget: {search_budget}"
        )
    if search_budget == "smoke":
        features = [
            "sem_base_disagreement_20_57",
            "context_velocity_energy",
        ]
        quantiles = [0.1, 0.25, 0.4]
        windows = [(10, 20), (10, 35), (20, 35), (35, 57)]
        blend_weights = [0.05, 0.1, 0.2]
    else:
        features = [
            "sem_base_disagreement_20_57",
            "sem_base_disagreement_57_105",
            "sem_base_disagreement_10_105",
            "context_velocity_energy",
        ]
        quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        windows = [(10, 20), (20, 35), (35, 57), (57, 105), (10, 57), (10, 105)]
        blend_weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    specs: list[dict[str, Any]] = []
    for feature_name in features:
        values = _sem_window_gate_feature(
            context=context,
            base_prediction=base_prediction,
            sem_prediction=sem_prediction,
            feature_name=feature_name,
        )
        for direction in ["lt", "gt"]:
            for quantile in quantiles:
                threshold = float(np.quantile(values, quantile))
                mask = values < threshold if direction == "lt" else values > threshold
                selected_count = int(np.sum(mask))
                if selected_count <= 0 or selected_count >= int(mask.shape[0]):
                    continue
                for window_start, window_stop in windows:
                    for blend_weight in blend_weights:
                        specs.append(
                            {
                                "feature_name": feature_name,
                                "direction": direction,
                                "threshold": threshold,
                                "window_start": window_start,
                                "window_stop": window_stop,
                                "blend_weight": blend_weight,
                                "selected_sample_count": selected_count,
                            }
                        )
    return specs


def _append_sem_window_memory_guidance_specs(
    *,
    specs: Sequence[dict[str, Any]],
    search_guidance: Sequence[Mapping[str, Any]],
    context: np.ndarray,
    base_prediction: np.ndarray,
    sem_prediction: np.ndarray,
) -> list[dict[str, Any]]:
    result = [dict(spec) for spec in specs]
    seen = {
        (
            spec.get("feature_name"),
            spec.get("direction"),
            round(float(spec.get("threshold", 0.0)), 12),
            spec.get("window_start"),
            spec.get("window_stop"),
            round(float(spec.get("blend_weight", 0.0)), 12),
        )
        for spec in result
    }
    for guidance in search_guidance:
        if guidance.get("guidance_type") != "refine_gating_after_near_miss":
            continue
        if guidance.get("reason") != "near_miss_with_empty_gate_selection":
            continue
        if guidance.get("source") != "sem_window_gate_blend_multi_search":
            continue
        feature_name = guidance.get("feature_name")
        if not isinstance(feature_name, str) or not feature_name:
            feature_name = "sem_base_disagreement_20_57"
        try:
            values = _sem_window_gate_feature(
                context=context,
                base_prediction=base_prediction,
                sem_prediction=sem_prediction,
                feature_name=feature_name,
            )
        except HermesBatchOptimizationLoopError:
            continue
        windows = _guided_sem_windows(guidance)
        for direction in ["lt", "gt"]:
            for quantile in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
                threshold = float(np.quantile(values, quantile))
                mask = values < threshold if direction == "lt" else values > threshold
                selected_count = int(np.sum(mask))
                if selected_count <= 0 or selected_count >= int(mask.shape[0]):
                    continue
                for window_start, window_stop in windows:
                    for blend_weight in [0.01, 0.025, 0.05]:
                        key = (
                            feature_name,
                            direction,
                            round(threshold, 12),
                            window_start,
                            window_stop,
                            round(blend_weight, 12),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        result.append(
                            {
                                "feature_name": feature_name,
                                "direction": direction,
                                "threshold": threshold,
                                "window_start": window_start,
                                "window_stop": window_stop,
                                "blend_weight": blend_weight,
                                "selected_sample_count": selected_count,
                                "memory_guidance_reason": guidance.get("reason"),
                                "memory_guidance_action": guidance.get(
                                    "recommended_action"
                                ),
                            }
                        )
    return result


def _guided_sem_windows(guidance: Mapping[str, Any]) -> list[tuple[int, int]]:
    start = _int_or_default(guidance.get("window_start"), 10)
    stop = _int_or_default(guidance.get("window_stop"), 20)
    start = max(10, min(190, start))
    stop = max(start + 1, min(200, stop))
    windows = [
        (start, stop),
        (max(10, start - 5), stop),
        (start, min(200, stop + 15)),
    ]
    result: list[tuple[int, int]] = []
    for window in windows:
        if window[1] <= window[0]:
            continue
        if window not in result:
            result.append(window)
    return result


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _build_sem_window_gate_blend_multi_search_candidate(
    *,
    output_dir: Path,
    base_submission_dir: Path,
    task1_test_path: Path,
    task1_target_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    sem_candidates: Sequence[dict[str, Any]],
    specs: Sequence[dict[str, Any]],
    force: bool,
    selection_metric: str = "segment_score",
) -> dict[str, Any]:
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=base_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    target = load_hdf5_tensor(task1_target_path, dataset_key=dataset_key)
    if target.shape[1] == 1000:
        target = target[:, ::5, :]
    if target.shape[2] == 1024:
        target = target[:, :, ::4]
    baseline_score = _task1_segment_score_from_arrays(
        prediction=base_prediction,
        target=target,
    )
    best_score = _selection_metric_value(
        score=baseline_score,
        selection_metric=selection_metric,
    )
    memory_guidance_spec_count = sum(
        1 for spec in specs if spec.get("memory_guidance_reason")
    )
    best_spec: dict[str, Any] | None = None
    best_sem_prediction: np.ndarray | None = None
    best_sem_params: dict[str, Any] | None = None
    evaluated_count = 0
    for sem_candidate in sem_candidates:
        sem_prediction = np.asarray(sem_candidate["sem_prediction"], dtype=np.float32)
        sem_params = dict(sem_candidate.get("sem_params", {}))
        for spec in specs:
            evaluated_count += 1
            prediction = _sem_window_gate_blend_array(
                context=context,
                base_prediction=base_prediction,
                sem_prediction=sem_prediction,
                feature_name=str(spec["feature_name"]),
                direction=str(spec["direction"]),
                threshold=float(spec["threshold"]),
                window_start=int(spec["window_start"]),
                window_stop=int(spec["window_stop"]),
                blend_weight=float(spec["blend_weight"]),
            )
            score = _task1_segment_score_from_arrays(
                prediction=prediction,
                target=target,
            )
            selection_value = _selection_metric_value(
                score=score,
                selection_metric=selection_metric,
            )
            if selection_value > best_score:
                best_score = selection_value
                best_spec = dict(spec)
                best_sem_prediction = sem_prediction
                best_sem_params = sem_params
    if best_spec is None or best_sem_prediction is None or best_sem_params is None:
        best_sem_prediction = np.asarray(sem_candidates[0]["sem_prediction"], dtype=np.float32)
        best_sem_params = dict(sem_candidates[0].get("sem_params", {}))
        best_spec = {
            "feature_name": "sem_base_disagreement_20_57",
            "direction": "lt",
            "threshold": -1.0,
            "window_start": 10,
            "window_stop": 20,
            "blend_weight": 0.0,
            "selected_sample_count": 0,
        }
    candidate = _build_sem_window_gate_blend_candidate(
        output_dir=output_dir,
        base_submission_dir=base_submission_dir,
        task1_test_path=task1_test_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        sem_prediction=best_sem_prediction,
        sem_params=best_sem_params,
        feature_name=str(best_spec["feature_name"]),
        direction=str(best_spec["direction"]),
        threshold=float(best_spec["threshold"]),
        window_start=int(best_spec["window_start"]),
        window_stop=int(best_spec["window_stop"]),
        blend_weight=float(best_spec["blend_weight"]),
        force=force,
    )
    candidate["source"] = "sem_window_gate_blend_multi_search"
    candidate["sem_params"] = best_sem_params
    candidate["search_evaluated_count"] = evaluated_count
    candidate["memory_guidance_spec_count"] = memory_guidance_spec_count
    if best_spec.get("memory_guidance_reason"):
        candidate["memory_guidance_reason"] = best_spec.get("memory_guidance_reason")
    if best_spec.get("memory_guidance_action"):
        candidate["memory_guidance_action"] = best_spec.get("memory_guidance_action")
    final_score = _task1_segment_score_from_arrays(
        prediction=_load_submission_prediction(
            submission_dir=Path(candidate["submission_dir"]),
            prediction_dataset_key=prediction_dataset_key,
        ),
        target=target,
    )
    candidate["validation_segment_score"] = float(final_score["segment_score"])
    candidate["validation_selection_metric"] = selection_metric
    candidate["validation_selection_score"] = float(best_score)
    search_report_path = output_dir / "sem_window_gate_blend_multi_search_report.json"
    _write_json(
        search_report_path,
        {
            "schema_version": 1,
            "state_hint": "SEM_WINDOW_GATE_BLEND_MULTI_SEARCH_RAN",
            "candidate_id": candidate["candidate_id"],
            "search_evaluated_count": evaluated_count,
            "memory_guidance_spec_count": memory_guidance_spec_count,
            "validation_segment_score": float(final_score["segment_score"]),
            "validation_selection_metric": selection_metric,
            "validation_selection_score": float(best_score),
            "best_spec": best_spec,
            "best_sem_params": best_sem_params,
            "submission_dir": candidate["submission_dir"],
        },
    )
    candidate["search_report_path"] = str(search_report_path)
    return candidate


def _build_sem_window_gate_blend_search_candidate(
    *,
    output_dir: Path,
    base_submission_dir: Path,
    task1_test_path: Path,
    task1_target_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    sem_prediction: np.ndarray,
    sem_params: dict[str, Any],
    specs: Sequence[dict[str, Any]],
    force: bool,
) -> dict[str, Any]:
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=base_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    target = load_hdf5_tensor(task1_target_path, dataset_key=dataset_key)
    if target.shape[1] == 1000:
        target = target[:, ::5, :]
    if target.shape[2] == 1024:
        target = target[:, :, ::4]
    best_score = _task1_segment_score_from_arrays(
        prediction=base_prediction,
        target=target,
    )["segment_score"]
    best_spec: dict[str, Any] | None = None
    best_prediction: np.ndarray | None = None
    sem_array = np.asarray(sem_prediction, dtype=np.float32)
    for spec in specs:
        prediction = _sem_window_gate_blend_array(
            context=context,
            base_prediction=base_prediction,
            sem_prediction=sem_array,
            feature_name=str(spec["feature_name"]),
            direction=str(spec["direction"]),
            threshold=float(spec["threshold"]),
            window_start=int(spec["window_start"]),
            window_stop=int(spec["window_stop"]),
            blend_weight=float(spec["blend_weight"]),
        )
        score = _task1_segment_score_from_arrays(
            prediction=prediction,
            target=target,
        )["segment_score"]
        if score > best_score:
            best_score = score
            best_spec = dict(spec)
            best_prediction = prediction
    if best_spec is None or best_prediction is None:
        best_spec = {
            "feature_name": "sem_base_disagreement_20_57",
            "direction": "lt",
            "threshold": -1.0,
            "window_start": 10,
            "window_stop": 20,
            "blend_weight": 0.0,
            "selected_sample_count": 0,
        }
        best_prediction = np.array(base_prediction, copy=True)
    candidate = _build_sem_window_gate_blend_candidate(
        output_dir=output_dir,
        base_submission_dir=base_submission_dir,
        task1_test_path=task1_test_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        sem_prediction=sem_array,
        sem_params=sem_params,
        feature_name=str(best_spec["feature_name"]),
        direction=str(best_spec["direction"]),
        threshold=float(best_spec["threshold"]),
        window_start=int(best_spec["window_start"]),
        window_stop=int(best_spec["window_stop"]),
        blend_weight=float(best_spec["blend_weight"]),
        force=force,
    )
    candidate["source"] = "sem_window_gate_blend_search"
    candidate["search_evaluated_count"] = len(specs)
    candidate["validation_segment_score"] = float(best_score)
    candidate["selected_sample_count"] = int(best_spec.get("selected_sample_count", 0))
    search_report_path = output_dir / "sem_window_gate_blend_search_report.json"
    _write_json(
        search_report_path,
        {
            "schema_version": 1,
            "state_hint": "SEM_WINDOW_GATE_BLEND_SEARCH_RAN",
            "candidate_id": candidate["candidate_id"],
            "search_evaluated_count": len(specs),
            "validation_segment_score": float(best_score),
            "best_spec": best_spec,
            "submission_dir": candidate["submission_dir"],
        },
    )
    candidate["search_report_path"] = str(search_report_path)
    return candidate


def _build_sem_window_gate_blend_candidate(
    *,
    output_dir: Path,
    base_submission_dir: Path,
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    sem_prediction: np.ndarray,
    sem_params: dict[str, Any],
    feature_name: str,
    direction: str,
    threshold: float,
    window_start: int,
    window_stop: int,
    blend_weight: float,
    force: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=base_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    sem_array = np.asarray(sem_prediction, dtype=np.float32)
    if sem_array.shape != base_prediction.shape:
        raise HermesBatchOptimizationLoopError(
            "SEM gate blend base and SEM predictions must have matching shapes"
        )
    values = _sem_window_gate_feature(
        context=context,
        base_prediction=base_prediction,
        sem_prediction=sem_array,
        feature_name=feature_name,
    )
    if direction == "lt":
        mask = values < float(threshold)
    elif direction == "gt":
        mask = values > float(threshold)
    else:
        raise HermesBatchOptimizationLoopError(
            "SEM window gate direction must be 'lt' or 'gt'"
        )
    start_index = max(10, int(window_start))
    stop_index = min(int(window_stop), base_prediction.shape[1])
    if stop_index <= start_index:
        raise HermesBatchOptimizationLoopError(
            "SEM gate blend requires window_stop > window_start"
        )
    weight = float(blend_weight)
    if weight < 0.0 or weight > 1.0:
        raise HermesBatchOptimizationLoopError(
            "SEM gate blend weight must be between 0 and 1"
        )
    blended = np.array(base_prediction, copy=True)
    blended[mask, start_index:stop_index, :] = (
        (1.0 - weight) * blended[mask, start_index:stop_index, :]
        + weight * sem_array[mask, start_index:stop_index, :]
    )
    blended[:, :10, :] = context
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "sem_window_gate_blend_prediction.npz"
    np.savez_compressed(prediction_path, prediction=blended.astype(np.float32))
    elapsed = time.perf_counter() - start
    submission = build_competition_submission(
        task_id=1,
        test_path=task1_test_path,
        output_dir=output_dir / "submission_package",
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=elapsed,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    candidate_id = (
        "sem_window_gate_blend"
        f"_{feature_name}_{direction}_{float(threshold):g}"
        f"_w{weight:g}_{start_index}_{stop_index}"
    )
    report_path = output_dir / "sem_window_gate_blend_candidate.json"
    selected_count = int(np.sum(mask))
    report = {
        "schema_version": 1,
        "state_hint": "SEM_WINDOW_GATE_BLEND_CANDIDATE_READY",
        "candidate_id": candidate_id,
        "base_submission_dir": str(base_submission_dir),
        "sem_params": sem_params,
        "feature_name": feature_name,
        "direction": direction,
        "threshold": float(threshold),
        "selected_sample_count": selected_count,
        "window_start": start_index,
        "window_stop": stop_index,
        "blend_weight": weight,
        "prediction_path": str(prediction_path),
        "submission_dir": submission["artifacts"]["submission_dir"],
        "elapsed_seconds": elapsed,
    }
    _write_json(report_path, report)
    return {
        "candidate_id": candidate_id,
        "submission_dir": submission["artifacts"]["submission_dir"],
        "source": "sem_window_gate_blend",
        "candidate_report_path": str(report_path),
        "base_submission_dir": str(base_submission_dir),
        "sem_params": sem_params,
        "feature_name": feature_name,
        "direction": direction,
        "threshold": float(threshold),
        "selected_sample_count": selected_count,
        "window_start": start_index,
        "window_stop": stop_index,
        "blend_weight": weight,
    }


def _sem_window_gate_blend_array(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    sem_prediction: np.ndarray,
    feature_name: str,
    direction: str,
    threshold: float,
    window_start: int,
    window_stop: int,
    blend_weight: float,
) -> np.ndarray:
    values = _sem_window_gate_feature(
        context=context,
        base_prediction=base_prediction,
        sem_prediction=sem_prediction,
        feature_name=feature_name,
    )
    if direction == "lt":
        mask = values < float(threshold)
    elif direction == "gt":
        mask = values > float(threshold)
    else:
        raise HermesBatchOptimizationLoopError(
            "SEM window gate direction must be 'lt' or 'gt'"
        )
    start_index = max(10, int(window_start))
    stop_index = min(int(window_stop), base_prediction.shape[1])
    if stop_index <= start_index:
        return np.array(base_prediction, copy=True)
    weight = float(blend_weight)
    blended = np.array(base_prediction, copy=True)
    blended[mask, start_index:stop_index, :] = (
        (1.0 - weight) * blended[mask, start_index:stop_index, :]
        + weight * sem_prediction[mask, start_index:stop_index, :]
    )
    blended[:, :10, :] = context
    return blended.astype(np.float32, copy=False)


def _sem_window_gate_feature(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    sem_prediction: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    if feature_name == "context_velocity_energy":
        return np.mean(
            (context[:, -1, :].astype(np.float64) - context[:, -2, :].astype(np.float64))
            ** 2,
            axis=1,
        )
    if feature_name.startswith("sem_base_disagreement_"):
        parts = feature_name.removeprefix("sem_base_disagreement_").split("_")
        if len(parts) != 2:
            raise HermesBatchOptimizationLoopError(
                f"unsupported SEM gate feature: {feature_name}"
            )
        window_start, window_stop = int(parts[0]), int(parts[1])
        return np.mean(
            (
                sem_prediction[:, window_start:window_stop, :].astype(np.float64)
                - base_prediction[:, window_start:window_stop, :].astype(np.float64)
            )
            ** 2,
            axis=(1, 2),
        )
    raise HermesBatchOptimizationLoopError(f"unsupported SEM gate feature: {feature_name}")


def _task1_segment_score_from_arrays(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        raise HermesBatchOptimizationLoopError(
            f"prediction/target shape mismatch: {prediction_array.shape} != {target_array.shape}"
        )
    squared_error = (prediction_array - target_array) ** 2
    target_squared = target_array**2
    early = _rel_mse_segment(
        squared_error=squared_error,
        target_squared=target_squared,
        future_start=0,
        future_stop=EARLY_FUTURE_STOP,
    )
    middle = _rel_mse_segment(
        squared_error=squared_error,
        target_squared=target_squared,
        future_start=EARLY_FUTURE_STOP,
        future_stop=MIDDLE_FUTURE_STOP,
    )
    late = _late_segment(
        squared_error=squared_error,
        future_start=MIDDLE_FUTURE_STOP,
        future_stop=FUTURE_STEPS,
    )
    segment_score = float(
        0.25 * early["score"] + 0.25 * middle["score"] + 0.5 * late["score"]
    )
    return {
        "segment_score": segment_score,
        "segments": {"early": early, "middle": middle, "late": late},
    }


def _selection_metric_value(*, score: dict[str, Any], selection_metric: str) -> float:
    if selection_metric == "segment_score":
        return float(score["segment_score"])
    if selection_metric == "early_segment_score":
        return float(score["segments"]["early"]["score"])
    if selection_metric == "middle_segment_score":
        return float(score["segments"]["middle"]["score"])
    if selection_metric == "late_segment_score":
        return float(score["segments"]["late"]["score"])
    if selection_metric == "early_rel_mse":
        return -float(score["segments"]["early"]["rel_mse"])
    raise HermesBatchOptimizationLoopError(
        f"unsupported SEM selection_metric: {selection_metric}"
    )


def _physics_residual_nudge_runner(
    *,
    initial_submission_dir: Path,
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    specs = [
        {
            "correction_weight": correction_weight,
            "stop_step": stop_step,
            "viscosity": 0.001,
            "dt": 0.05,
            "dx": 1.0 / 256.0,
            "spectral_strength": 0.0,
        }
        for correction_weight in [0.002, 0.005, 0.01, 0.02, 0.04, 0.06]
        for stop_step in [20, 35, 57, 105]
    ]

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(specs):
            return None
        return _build_physics_residual_nudge_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_submission_dir=initial_submission_dir,
            task1_test_path=task1_test_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
            **specs[round_index - 1],
        )

    return run_candidate


def _build_physics_residual_nudge_candidate(
    *,
    output_dir: Path,
    base_submission_dir: Path,
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    correction_weight: float,
    stop_step: int,
    viscosity: float,
    dt: float,
    dx: float,
    spectral_strength: float,
    force: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=base_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    nudged = residual_guided_refinement(
        context=context,
        base_prediction=base_prediction,
        viscosity=float(viscosity),
        dt=float(dt),
        dx=float(dx),
        correction_weight=float(correction_weight),
        stop_step=int(stop_step),
        spectral_strength=float(spectral_strength),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "physics_residual_nudge_prediction.npz"
    np.savez_compressed(prediction_path, prediction=nudged.astype(np.float32))
    elapsed = time.perf_counter() - start
    submission = build_competition_submission(
        task_id=1,
        test_path=task1_test_path,
        output_dir=output_dir / "submission_package",
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=elapsed,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    candidate_id = (
        "physics_residual_nudge"
        f"_cw{float(correction_weight):g}"
        f"_stop{int(stop_step)}"
    )
    report_path = output_dir / "physics_residual_nudge_candidate.json"
    report = {
        "schema_version": 1,
        "state_hint": "PHYSICS_RESIDUAL_NUDGE_CANDIDATE_READY",
        "candidate_id": candidate_id,
        "base_submission_dir": str(base_submission_dir),
        "correction_weight": float(correction_weight),
        "stop_step": int(stop_step),
        "viscosity": float(viscosity),
        "dt": float(dt),
        "dx": float(dx),
        "spectral_strength": float(spectral_strength),
        "prediction_path": str(prediction_path),
        "submission_dir": submission["artifacts"]["submission_dir"],
        "elapsed_seconds": elapsed,
    }
    _write_json(report_path, report)
    return {
        "candidate_id": candidate_id,
        "submission_dir": submission["artifacts"]["submission_dir"],
        "source": "physics_residual_nudge",
        "candidate_report_path": str(report_path),
        "base_submission_dir": str(base_submission_dir),
        "correction_weight": float(correction_weight),
        "stop_step": int(stop_step),
        "viscosity": float(viscosity),
        "dt": float(dt),
        "dx": float(dx),
        "spectral_strength": float(spectral_strength),
    }


def _append_checkpoint_window_gate_spec(
    *,
    specs: list[dict[str, Any]],
    seen: set[tuple[Path, str, str, float, int, int]],
    checkpoint_dir: Path,
    context: np.ndarray,
    base_prediction: np.ndarray,
    checkpoint_prediction: np.ndarray,
    feature_name: str,
    direction: str,
    threshold: float,
    window_start: int,
    window_stop: int,
) -> None:
    mask, _feature_values = _checkpoint_window_gate_mask(
        context=context,
        base_prediction=base_prediction,
        checkpoint_prediction=checkpoint_prediction,
        feature_name=feature_name,
        direction=direction,
        threshold=threshold,
    )
    selected_count = int(np.sum(mask))
    if selected_count <= 0 or selected_count >= int(mask.shape[0]):
        return
    key = (
        checkpoint_dir,
        feature_name,
        direction,
        round(float(threshold), 15),
        int(window_start),
        int(window_stop),
    )
    if key in seen:
        return
    seen.add(key)
    specs.append(
        {
            "checkpoint_submission_dir": checkpoint_dir,
            "feature_name": feature_name,
            "direction": direction,
            "threshold": float(threshold),
            "window_start": int(window_start),
            "window_stop": int(window_stop),
            "expected_selected_sample_count": selected_count,
        }
    )


def _build_checkpoint_window_gate_candidate(
    *,
    output_dir: Path,
    base_submission_dir: Path,
    checkpoint_submission_dir: Path,
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    feature_name: str,
    direction: str,
    threshold: float,
    window_start: int,
    window_stop: int,
    force: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=base_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    checkpoint_prediction = _load_submission_prediction(
        submission_dir=checkpoint_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    if base_prediction.shape != checkpoint_prediction.shape:
        raise HermesBatchOptimizationLoopError(
            "checkpoint gate base and checkpoint predictions must have matching shapes"
        )
    if context.shape[0] != base_prediction.shape[0]:
        raise HermesBatchOptimizationLoopError(
            "checkpoint gate context and prediction sample counts must match"
        )
    mask, feature_values = _checkpoint_window_gate_mask(
        context=context,
        base_prediction=base_prediction,
        checkpoint_prediction=checkpoint_prediction,
        feature_name=feature_name,
        direction=direction,
        threshold=threshold,
    )
    gated = np.array(base_prediction, copy=True)
    gated[mask, window_start:window_stop, :] = checkpoint_prediction[
        mask, window_start:window_stop, :
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "checkpoint_window_gate_prediction.npz"
    np.savez_compressed(prediction_path, prediction=gated.astype(np.float32))
    elapsed = time.perf_counter() - start
    submission = build_competition_submission(
        task_id=1,
        test_path=task1_test_path,
        output_dir=output_dir / "submission_package",
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=elapsed,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    candidate_id = (
        "checkpoint_window_gate"
        f"_{feature_name}_{direction}_{float(threshold):g}"
        f"_{window_start}_{window_stop}"
    )
    report_path = output_dir / "checkpoint_window_gate_candidate.json"
    report = {
        "schema_version": 1,
        "state_hint": "CHECKPOINT_WINDOW_GATE_CANDIDATE_READY",
        "candidate_id": candidate_id,
        "base_submission_dir": str(base_submission_dir),
        "checkpoint_submission_dir": str(checkpoint_submission_dir),
        "feature_name": feature_name,
        "direction": direction,
        "threshold": float(threshold),
        "window_start": int(window_start),
        "window_stop": int(window_stop),
        "selected_sample_count": int(np.sum(mask)),
        "feature_summary": {
            "min": float(np.min(feature_values)),
            "median": float(np.median(feature_values)),
            "max": float(np.max(feature_values)),
        },
        "prediction_path": str(prediction_path),
        "submission_dir": submission["artifacts"]["submission_dir"],
        "elapsed_seconds": elapsed,
    }
    _write_json(report_path, report)
    return {
        "candidate_id": candidate_id,
        "submission_dir": submission["artifacts"]["submission_dir"],
        "source": "checkpoint_window_gate",
        "candidate_report_path": str(report_path),
        "base_submission_dir": str(base_submission_dir),
        "checkpoint_submission_dir": str(checkpoint_submission_dir),
        "feature_name": feature_name,
        "direction": direction,
        "threshold": float(threshold),
        "window_start": int(window_start),
        "window_stop": int(window_stop),
        "selected_sample_count": int(np.sum(mask)),
    }


def _checkpoint_window_gate_mask(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    checkpoint_prediction: np.ndarray,
    feature_name: str,
    direction: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = _checkpoint_window_gate_feature(
        context=context,
        base_prediction=base_prediction,
        checkpoint_prediction=checkpoint_prediction,
        feature_name=feature_name,
    )
    if direction == "gt":
        return values > float(threshold), values
    if direction == "lt":
        return values < float(threshold), values
    raise HermesBatchOptimizationLoopError(
        "checkpoint gate direction must be 'gt' or 'lt'"
    )


def _checkpoint_window_gate_feature(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    checkpoint_prediction: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    if feature_name == "checkpoint_jump35":
        return np.mean(
            (
                checkpoint_prediction[:, 35, :].astype(np.float64)
                - base_prediction[:, 34, :].astype(np.float64)
            )
            ** 2,
            axis=1,
        )
    if feature_name == "context_velocity_energy":
        return np.mean(
            (context[:, -1, :].astype(np.float64) - context[:, -2, :].astype(np.float64))
            ** 2,
            axis=1,
        )
    if feature_name == "checkpoint_base_energy_ratio_35_57":
        checkpoint_energy = np.mean(
            checkpoint_prediction[:, 35:57, :].astype(np.float64) ** 2,
            axis=(1, 2),
        )
        base_energy = np.mean(
            base_prediction[:, 35:57, :].astype(np.float64) ** 2,
            axis=(1, 2),
        )
        return checkpoint_energy / (base_energy + 1.0e-12)
    if feature_name.startswith("base_checkpoint_disagreement_"):
        parts = feature_name.removeprefix("base_checkpoint_disagreement_").split("_")
        if len(parts) != 2:
            raise HermesBatchOptimizationLoopError(
                f"unsupported checkpoint gate feature: {feature_name}"
            )
        window_start, window_stop = int(parts[0]), int(parts[1])
        return np.mean(
            (
                checkpoint_prediction[:, window_start:window_stop, :].astype(np.float64)
                - base_prediction[:, window_start:window_stop, :].astype(np.float64)
            )
            ** 2,
            axis=(1, 2),
        )
    raise HermesBatchOptimizationLoopError(
        f"unsupported checkpoint gate feature: {feature_name}"
    )


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
    timing_profile: str | None,
    timing_factor: float | None,
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
            timing_profile=timing_profile,
            timing_factor=timing_factor,
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
    timing_profile: str | None,
    timing_factor: float | None,
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
            timing_profile=timing_profile,
            timing_factor=timing_factor,
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
    timing_profile: str | None,
    timing_factor: float | None,
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
            timing_profile=timing_profile,
            timing_factor=timing_factor,
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
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    runtime_runner = _runtime_config_runner(
        unet_checkpoint_path=unet_checkpoint_path,
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
        configs=[(8, 1)],
    )
    fast_runner = _fast_baseline_runner(
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
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
    hypothesis_blueprint_path: str | Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    task_id, test_path = _single_task_context(
        task1_test_path=task1_test_path,
        task2_test_path=task2_test_path,
    )
    schedule = _linear_trend_spectral_schedule(hypothesis_blueprint_path)

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
        late_strength = params.get("late_spectral_strength")
        late_start = params.get("late_start")
        if isinstance(late_strength, int | float) and float(late_strength) > 0.0:
            start_index = int(late_start) if isinstance(late_start, int | float) else 105
            start_index = max(context.shape[1], min(prediction.shape[1], start_index))
            late_prediction = spectral_smooth_prediction(
                prediction=prediction,
                strength=float(late_strength),
            )
            prediction = prediction.copy()
            prediction[:, start_index:, :] = late_prediction[:, start_index:, :]
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
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
        candidate_id = (
            "linear_trend_spectral"
            f"_w{params['slope_window']}"
            f"_g{params['gain']:g}"
            f"_d{params['decay']:g}"
            f"_s{params['spectral_strength']:g}"
        )
        if "late_spectral_strength" in params:
            candidate_id += f"_ls{params['late_spectral_strength']:g}"
        if "late_start" in params:
            candidate_id += f"_lt{params['late_start']}"
        candidate_report_path = output_dir / "linear_trend_spectral_candidate.json"
        candidate_report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state_hint": "LINEAR_TREND_SPECTRAL_CANDIDATE_READY",
                    "candidate_id": candidate_id,
                    "params": params,
                    "hypothesis_blueprint_path": str(hypothesis_blueprint_path)
                    if hypothesis_blueprint_path is not None
                    else None,
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
            "hypothesis_blueprint_path": str(hypothesis_blueprint_path)
            if hypothesis_blueprint_path is not None
            else None,
            "candidate_report_path": str(candidate_report_path),
        }

    return run_candidate


def _hypothesis_lab_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path | None,
    reference_sample_limit_schedule: Sequence[int | None],
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab candidate generator requires task1_test_path"
        )
    if task1_target_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab candidate generator requires task1_target_path"
        )
    schedule = list(reference_sample_limit_schedule)

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index > len(schedule):
            return None
        reference_sample_limit = schedule[round_index - 1]
        output_dir = Path(kwargs["output_dir"])
        library_path = output_dir / "local_strategy_library.json"
        candidates_path = output_dir / "local_asset_candidates.json"
        build_local_strategy_library(output_path=library_path, force=force)
        build_local_asset_algorithm_candidates(
            local_strategy_library_path=library_path,
            checkpoint_paths=[],
            output_path=candidates_path,
            max_strategy_candidates=5,
            force=force,
        )
        lab = run_hypothesis_lab(
            manual_report_paths=[],
            algorithm_candidate_paths=[candidates_path],
            context_path=Path(task1_test_path),
            target_path=Path(task1_target_path),
            reference_path=Path(reference_path) if reference_path is not None else None,
            reference_sample_limit=reference_sample_limit,
            output_dir=output_dir,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
        sandbox = lab["sandbox"]
        best = sandbox["best_candidate"]
        verifier = sandbox.get("verifier", {})
        reference_audit = sandbox.get("reference_audit")
        return {
            "candidate_id": str(best["candidate_id"]),
            "submission_dir": lab["artifacts"]["best_submission_dir"],
            "source": "hypothesis_lab",
            "candidate_report_path": lab["artifacts"]["hypothesis_lab_report_path"],
            "score_delta_vs_baseline": verifier.get("score_delta_vs_baseline"),
            "hypothesis_lab_decision": verifier.get("decision"),
            "reference_audit": reference_audit if isinstance(reference_audit, dict) else None,
            "reference_sample_limit": reference_sample_limit,
            "params": best.get("params"),
        }

    return run_candidate


def _hypothesis_lab_early_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path,
    reference_sample_limit: int | None,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_early_search requires task1_test_path"
        )
    if task1_target_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_early_search requires task1_target_path"
        )
    used = False

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        nonlocal used
        if used:
            return None
        used = True
        output_dir = Path(kwargs["output_dir"])
        sandbox = run_blueprint_sandbox(
            blueprint=_early_refinement_blueprint(),
            context_path=Path(task1_test_path),
            target_path=Path(task1_target_path),
            reference_path=Path(reference_path),
            reference_sample_limit=reference_sample_limit,
            output_dir=output_dir,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
        best = sandbox["best_candidate"]
        verifier = sandbox.get("verifier", {})
        reference_audit = sandbox.get("reference_audit")
        segments = best.get("segments") if isinstance(best, dict) else None
        early_segment = (
            segments.get("early")
            if isinstance(segments, dict) and isinstance(segments.get("early"), dict)
            else None
        )
        return {
            "candidate_id": str(best["candidate_id"]),
            "submission_dir": sandbox["artifacts"]["best_submission_dir"],
            "source": "hypothesis_lab_early_search",
            "candidate_report_path": sandbox["artifacts"]["sandbox_report_path"],
            "hypothesis_lab_decision": verifier.get("decision"),
            "reference_audit": reference_audit if isinstance(reference_audit, dict) else None,
            "reference_sample_limit": reference_sample_limit,
            "selection_objective": verifier.get("selection_objective"),
            "early_segment_diagnosis": verifier.get("early_segment_diagnosis"),
            "early_segment": early_segment,
            "params": best.get("params"),
        }

    return run_candidate


def _hypothesis_lab_early_transition_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path,
    reference_sample_limit: int | None,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_early_transition_search requires task1_test_path"
        )
    if task1_target_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_early_transition_search requires task1_target_path"
        )
    used = False

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        nonlocal used
        if used:
            return None
        used = True
        output_dir = Path(kwargs["output_dir"])
        sandbox = run_blueprint_sandbox(
            blueprint=_early_transition_expert_blueprint(),
            context_path=Path(task1_test_path),
            target_path=Path(task1_target_path),
            reference_path=Path(reference_path),
            reference_sample_limit=reference_sample_limit,
            output_dir=output_dir,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
        best = sandbox["best_candidate"]
        verifier = sandbox.get("verifier", {})
        reference_audit = sandbox.get("reference_audit")
        segments = best.get("segments") if isinstance(best, dict) else None
        early_segment = (
            segments.get("early")
            if isinstance(segments, dict) and isinstance(segments.get("early"), dict)
            else None
        )
        return {
            "candidate_id": str(best["candidate_id"]),
            "submission_dir": sandbox["artifacts"]["best_submission_dir"],
            "source": "hypothesis_lab_early_transition_search",
            "candidate_report_path": sandbox["artifacts"]["sandbox_report_path"],
            "hypothesis_lab_decision": verifier.get("decision"),
            "reference_audit": reference_audit if isinstance(reference_audit, dict) else None,
            "reference_sample_limit": reference_sample_limit,
            "selection_objective": verifier.get("selection_objective"),
            "early_segment_diagnosis": verifier.get("early_segment_diagnosis"),
            "early_segment": early_segment,
            "params": best.get("params"),
        }

    return run_candidate


def _hypothesis_lab_front25_dynamics_smoke_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path,
    reference_sample_limit: int | None,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_front25_dynamics_smoke requires task1_test_path"
        )
    if task1_target_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_front25_dynamics_smoke requires task1_target_path"
        )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        if int(kwargs["round_index"]) != 1:
            return None
        return _run_hypothesis_blueprint_candidate(
            blueprint=_front25_dynamics_blueprint(),
            source="hypothesis_lab_front25_dynamics_smoke",
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            reference_path=Path(reference_path),
            reference_sample_limit=reference_sample_limit,
            output_dir=Path(kwargs["output_dir"]),
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )

    return run_candidate


def _hypothesis_lab_short_window_corrector_smoke_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path,
    reference_sample_limit: int | None,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
    search_budget: str = "smoke",
) -> CandidateRunner:
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_short_window_corrector_smoke requires task1_test_path"
        )
    if task1_target_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_short_window_corrector_smoke requires task1_target_path"
        )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        if int(kwargs["round_index"]) != 1:
            return None
        return _run_hypothesis_blueprint_candidate(
            blueprint=_short_window_corrector_blueprint(search_budget=search_budget),
            source=f"hypothesis_lab_short_window_corrector_{search_budget}",
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            reference_path=Path(reference_path),
            reference_sample_limit=reference_sample_limit,
            output_dir=Path(kwargs["output_dir"]),
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )

    return run_candidate


def _hypothesis_lab_short_window_corrector_patch_sources_runner(
    *,
    task1_test_path: Path,
    task1_target_path: Path,
    reference_path: Path,
    reference_sample_limit: int | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        if int(kwargs["round_index"]) != 1:
            return None
        output_dir = Path(kwargs["output_dir"])
        source = _run_hypothesis_blueprint_candidate(
            blueprint=_short_window_corrector_blueprint(search_budget="smoke"),
            source="hypothesis_lab_short_window_corrector_patch_sources_source",
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            reference_path=reference_path,
            reference_sample_limit=reference_sample_limit,
            output_dir=output_dir / "source",
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
        best_candidate = kwargs.get("best_candidate")
        if not (
            isinstance(best_candidate, dict)
            and best_candidate.get("rules_passed")
            and isinstance(best_candidate.get("submission_dir"), str)
        ):
            return source
        source_submission = source.get("submission_dir")
        if not isinstance(source_submission, str):
            return source
        return _build_early_subsegment_patch_candidate(
            output_dir=output_dir,
            base_candidate=best_candidate,
            source_candidates=[
                {
                    "candidate_id": str(source.get("candidate_id", "short_window_source")),
                    "submission_dir": source_submission,
                    "rules_passed": True,
                    "source": "hypothesis_lab_short_window_corrector_patch_sources",
                }
            ],
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )

    return run_candidate


def _hypothesis_lab_transition_early_patch_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path,
    reference_sample_limit: int | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    transition_runner = _hypothesis_lab_early_transition_runner(
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_transition_early_patch requires task1_test_path"
        )
    transition_candidate: dict[str, Any] | None = None

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        nonlocal transition_candidate
        round_index = int(kwargs["round_index"])
        if round_index == 1:
            transition_candidate = transition_runner(**kwargs)
            return transition_candidate
        if round_index != 2 or transition_candidate is None:
            return None
        trajectory = kwargs.get("trajectory")
        if not isinstance(trajectory, list):
            return None
        early_source = _best_early_source_candidate(trajectory)
        if early_source is None:
            return None
        if early_source.get("candidate_id") == transition_candidate.get("candidate_id"):
            return None
        return _build_early_segment_patch_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_candidate=transition_candidate,
            early_source_candidate=early_source,
            task1_test_path=Path(task1_test_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )

    return run_candidate


def _hypothesis_lab_middle_late_patch_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path,
    reference_sample_limit: int | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_middle_late_patch_search requires task1_test_path"
        )
    if task1_target_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_middle_late_patch_search requires task1_target_path"
        )
    base_candidate: dict[str, Any] | None = None

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        nonlocal base_candidate
        round_index = int(kwargs["round_index"])
        if round_index == 1:
            output_dir = Path(kwargs["output_dir"])
            sandbox = run_blueprint_sandbox(
                blueprint=_middle_late_retrieval_blueprint(),
                context_path=Path(task1_test_path),
                target_path=Path(task1_target_path),
                reference_path=Path(reference_path),
                reference_sample_limit=reference_sample_limit,
                output_dir=output_dir,
                submission_id=submission_id,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                force=force,
            )
            best = sandbox["best_candidate"]
            verifier = sandbox.get("verifier", {})
            reference_audit = sandbox.get("reference_audit")
            base_candidate = {
                "candidate_id": str(best["candidate_id"]),
                "submission_dir": sandbox["artifacts"]["best_submission_dir"],
                "source": "hypothesis_lab_middle_late_patch_search",
                "candidate_report_path": sandbox["artifacts"]["sandbox_report_path"],
                "hypothesis_lab_decision": verifier.get("decision"),
                "reference_audit": reference_audit
                if isinstance(reference_audit, dict)
                else None,
                "reference_sample_limit": reference_sample_limit,
                "selection_objective": verifier.get("selection_objective"),
                "params": best.get("params"),
            }
            return base_candidate
        if round_index != 2 or base_candidate is None:
            return None
        trajectory = kwargs.get("trajectory")
        if not isinstance(trajectory, list):
            return None
        early_source = _best_early_source_candidate(trajectory)
        if early_source is None:
            return None
        if early_source.get("candidate_id") == base_candidate.get("candidate_id"):
            return None
        return _build_early_segment_patch_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_candidate=base_candidate,
            early_source_candidate=early_source,
            task1_test_path=Path(task1_test_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )

    return run_candidate


def _hypothesis_lab_early_subsegment_patch_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path,
    reference_sample_limit: int | None,
    candidate_options: Mapping[str, Any] | None,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_early_subsegment_patch_search requires task1_test_path"
        )
    if task1_target_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_early_subsegment_patch_search requires task1_target_path"
        )
    stage_candidates: dict[str, dict[str, Any]] = {}

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        output_dir = Path(kwargs["output_dir"])
        if round_index == 1:
            candidate = _run_hypothesis_blueprint_candidate(
                blueprint=_early_refinement_blueprint(),
                source="hypothesis_lab_early_subsegment_patch_search",
                task1_test_path=Path(task1_test_path),
                task1_target_path=Path(task1_target_path),
                reference_path=Path(reference_path),
                reference_sample_limit=reference_sample_limit,
                output_dir=output_dir,
                submission_id=submission_id,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                force=force,
            )
            stage_candidates["early_refinement"] = candidate
            return candidate
        if round_index == 2:
            candidate = _run_hypothesis_blueprint_candidate(
                blueprint=_front25_dynamics_blueprint(),
                source="hypothesis_lab_early_subsegment_patch_search",
                task1_test_path=Path(task1_test_path),
                task1_target_path=Path(task1_target_path),
                reference_path=Path(reference_path),
                reference_sample_limit=reference_sample_limit,
                output_dir=output_dir,
                submission_id=submission_id,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                force=force,
            )
            stage_candidates["front25_dynamics"] = candidate
            return candidate
        if round_index == 3:
            candidate = _run_hypothesis_blueprint_candidate(
                blueprint=_early_transition_expert_blueprint(),
                source="hypothesis_lab_early_subsegment_patch_search",
                task1_test_path=Path(task1_test_path),
                task1_target_path=Path(task1_target_path),
                reference_path=Path(reference_path),
                reference_sample_limit=reference_sample_limit,
                output_dir=output_dir,
                submission_id=submission_id,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                force=force,
            )
            stage_candidates["early_transition"] = candidate
            return candidate
        if round_index == 4:
            candidate = _run_hypothesis_blueprint_candidate(
                blueprint=_middle_late_retrieval_blueprint(),
                source="hypothesis_lab_early_subsegment_patch_search",
                task1_test_path=Path(task1_test_path),
                task1_target_path=Path(task1_target_path),
                reference_path=Path(reference_path),
                reference_sample_limit=reference_sample_limit,
                output_dir=output_dir,
                submission_id=submission_id,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                force=force,
            )
            stage_candidates["middle_late"] = candidate
            return candidate
        if round_index != 5 or "middle_late" not in stage_candidates:
            return None
        trajectory = kwargs.get("trajectory")
        if not isinstance(trajectory, list):
            return None
        source_candidates = _valid_submission_candidates(trajectory)
        patch_base = stage_candidates["middle_late"]
        best_candidate = kwargs.get("best_candidate")
        if (
            isinstance(best_candidate, dict)
            and best_candidate.get("rules_passed")
            and isinstance(best_candidate.get("submission_dir"), str)
        ):
            patch_base = best_candidate
        return _build_early_subsegment_patch_candidate(
            output_dir=output_dir,
            base_candidate=patch_base,
            source_candidates=source_candidates,
            task1_test_path=Path(task1_test_path),
            task1_target_path=Path(task1_target_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            source_stability_guard=_candidate_option_bool(
                candidate_options,
                "source_stability_guard",
            ),
            handoff_smoothing_weight=_candidate_option_float(
                candidate_options,
                "handoff_smoothing_weight",
            ),
            force=force,
        )

    return run_candidate


def _early_subsegment_patch_sources_runner(
    *,
    candidate_submission_dirs: Sequence[str | Path] | None,
    task1_test_path: Path,
    task1_target_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    source_candidates = _source_submission_candidates(candidate_submission_dirs)
    if not source_candidates:
        raise HermesBatchOptimizationLoopError(
            "early_subsegment_patch_sources requires at least one candidate_submission_dir"
        )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        if int(kwargs["round_index"]) != 1:
            return None
        best_candidate = kwargs.get("best_candidate")
        if not (
            isinstance(best_candidate, dict)
            and best_candidate.get("rules_passed")
            and isinstance(best_candidate.get("submission_dir"), str)
        ):
            return None
        return _build_early_subsegment_patch_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_candidate=best_candidate,
            source_candidates=source_candidates,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )

    return run_candidate


def _checkpoint_front_window_adapter_patch_sources_runner(
    *,
    checkpoint_submission_dirs: Sequence[str | Path] | None,
    initial_submission_dir: Path,
    task1_test_path: Path,
    task1_target_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    checkpoint_dirs = [Path(path) for path in checkpoint_submission_dirs or []]
    if not checkpoint_dirs:
        raise HermesBatchOptimizationLoopError(
            "checkpoint_front_window_adapter_patch_sources requires at least one candidate_submission_dir"
        )
    context = load_hdf5_tensor(task1_test_path, dataset_key=dataset_key)[:, :10, :]
    base_prediction = _load_submission_prediction(
        submission_dir=initial_submission_dir,
        prediction_dataset_key=prediction_dataset_key,
    )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        if int(kwargs["round_index"]) != 1:
            return None
        best_candidate = kwargs.get("best_candidate")
        if not (
            isinstance(best_candidate, dict)
            and best_candidate.get("rules_passed")
            and isinstance(best_candidate.get("submission_dir"), str)
        ):
            return None
        output_dir = Path(kwargs["output_dir"])
        source_candidates = _materialize_checkpoint_front_window_adapter_sources(
            output_dir=output_dir / "checkpoint_front_window_adapter_sources",
            checkpoint_submission_dirs=checkpoint_dirs,
            context=context,
            base_prediction=base_prediction,
            task1_test_path=task1_test_path,
            prediction_dataset_key=prediction_dataset_key,
            dataset_key=dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
        candidate = _build_early_subsegment_patch_candidate(
            output_dir=output_dir,
            base_candidate=best_candidate,
            source_candidates=source_candidates,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
        manifest_path = output_dir / "checkpoint_front_window_adapter_sources.json"
        candidate["source_candidate_manifest_path"] = str(manifest_path)
        candidate["source_candidate_count"] = len(source_candidates)
        candidate["adapter_kinds"] = sorted(
            {
                str(item["adapter_kind"])
                for item in source_candidates
                if isinstance(item.get("adapter_kind"), str)
            }
        )
        return candidate

    return run_candidate


def _materialize_checkpoint_front_window_adapter_sources(
    *,
    output_dir: Path,
    checkpoint_submission_dirs: Sequence[Path],
    context: np.ndarray,
    base_prediction: np.ndarray,
    task1_test_path: Path,
    prediction_dataset_key: str,
    dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> list[dict[str, Any]]:
    source_candidates: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_dir in enumerate(checkpoint_submission_dirs):
        checkpoint_prediction = _load_submission_prediction(
            submission_dir=checkpoint_dir,
            prediction_dataset_key=prediction_dataset_key,
        )
        sources = build_checkpoint_front_window_adapter_sources(
            context=context,
            base_prediction=base_prediction,
            checkpoint_prediction=checkpoint_prediction,
            checkpoint_source=str(checkpoint_dir),
        )
        for source_index, source in enumerate(sources):
            metadata = source.get("metadata")
            if not isinstance(metadata, dict):
                continue
            candidate_id = str(source["candidate_id"])
            candidate_dir = (
                output_dir
                / f"checkpoint_{checkpoint_index:03d}"
                / f"source_{source_index:03d}_{candidate_id}"
            )
            candidate_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = candidate_dir / "prediction.npz"
            np.savez_compressed(prediction_path, prediction=source["prediction"])
            submission = build_competition_submission(
                task_id=1,
                test_path=task1_test_path,
                output_dir=candidate_dir / "submission_package",
                submission_id=submission_id,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                method="external_prediction",
                source_prediction_path=prediction_path,
                source_prediction_key="prediction",
                train_time=0.0,
                inference_time=0.0,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                force=force,
            )
            source_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "submission_dir": submission["artifacts"]["submission_dir"],
                    "rules_passed": True,
                    "source": "checkpoint_front_window_adapter_patch_sources",
                    "adapter_kind": metadata.get("adapter_kind"),
                    "checkpoint_source": metadata.get("checkpoint_source"),
                    "uses_target_as_reference": metadata.get("uses_target_as_reference"),
                    "window": metadata.get("window"),
                    "window_start": metadata.get("window_start"),
                    "window_stop": metadata.get("window_stop"),
                    "weight": metadata.get("weight"),
                    "metadata": metadata,
                }
            )
            manifest_rows.append(
                {
                    "candidate_id": candidate_id,
                    "submission_dir": submission["artifacts"]["submission_dir"],
                    "rules_passed": True,
                    "adapter_kind": metadata.get("adapter_kind"),
                    "checkpoint_source": metadata.get("checkpoint_source"),
                    "uses_target_as_reference": metadata.get("uses_target_as_reference"),
                    "window": metadata.get("window"),
                    "window_start": metadata.get("window_start"),
                    "window_stop": metadata.get("window_stop"),
                    "weight": metadata.get("weight"),
                    "mean_abs_checkpoint_delta": metadata.get(
                        "mean_abs_checkpoint_delta"
                    ),
                    "rmse_checkpoint_delta": metadata.get("rmse_checkpoint_delta"),
                }
            )
    manifest = {
        "schema_version": 1,
        "state_hint": "CHECKPOINT_FRONT_WINDOW_ADAPTER_SOURCES_READY",
        "source": "checkpoint_front_window_adapter_patch_sources",
        "uses_target_as_reference": False,
        "source_candidate_count": len(manifest_rows),
        "adapter_kinds": sorted(
            {
                str(row["adapter_kind"])
                for row in manifest_rows
                if isinstance(row.get("adapter_kind"), str)
            }
        ),
        "windows": sorted(
            {
                str(row["window"])
                for row in manifest_rows
                if isinstance(row.get("window"), str)
            }
        ),
        "source_candidates": manifest_rows,
    }
    _write_json(output_dir.parent / "checkpoint_front_window_adapter_sources.json", manifest)
    return source_candidates


def _source_submission_candidates(
    candidate_submission_dirs: Sequence[str | Path] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, submission_dir in enumerate(candidate_submission_dirs or []):
        path = Path(submission_dir)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        candidate_name = path.name or f"candidate_{index:03d}"
        if candidate_name == "submission" and path.parent.name:
            candidate_name = path.parent.name
        rows.append(
            {
                "candidate_id": f"source_{index:03d}_{candidate_name}",
                "submission_dir": str(path),
                "rules_passed": True,
                "source": "candidate_submission_dir",
            }
        )
    return rows


def _run_hypothesis_blueprint_candidate(
    *,
    blueprint: dict[str, Any],
    source: str,
    task1_test_path: Path,
    task1_target_path: Path,
    reference_path: Path,
    reference_sample_limit: int | None,
    output_dir: Path,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> dict[str, Any]:
    blueprint = _limit_blueprint_neighbor_count(
        blueprint=blueprint,
        reference_sample_limit=reference_sample_limit,
    )
    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=task1_test_path,
        target_path=task1_target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        output_dir=output_dir,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    best = sandbox["best_candidate"]
    verifier = sandbox.get("verifier", {})
    reference_audit = sandbox.get("reference_audit")
    segments = best.get("segments") if isinstance(best, dict) else None
    early_segment = (
        segments.get("early")
        if isinstance(segments, dict) and isinstance(segments.get("early"), dict)
        else None
    )
    return {
        "candidate_id": str(best["candidate_id"]),
        "submission_dir": sandbox["artifacts"]["best_submission_dir"],
        "source": source,
        "candidate_report_path": sandbox["artifacts"]["sandbox_report_path"],
        "hypothesis_lab_decision": verifier.get("decision"),
        "reference_audit": reference_audit if isinstance(reference_audit, dict) else None,
        "reference_sample_limit": reference_sample_limit,
        "selection_objective": verifier.get("selection_objective"),
        "early_segment_diagnosis": verifier.get("early_segment_diagnosis"),
        "early_segment": early_segment,
        "params": best.get("params"),
    }


def _limit_blueprint_neighbor_count(
    *,
    blueprint: dict[str, Any],
    reference_sample_limit: int | None,
) -> dict[str, Any]:
    if reference_sample_limit is None:
        return blueprint
    if reference_sample_limit < 1:
        return blueprint
    limited = copy.deepcopy(blueprint)
    grid = limited.get("parameter_grid")
    if not isinstance(grid, dict):
        return limited
    params = grid.get("candidate_params")
    if not isinstance(params, list):
        return limited
    for row in params:
        if not isinstance(row, dict):
            continue
        for key in ("k", "base_k"):
            k = row.get(key)
            if isinstance(k, int | float) and not isinstance(k, bool):
                row[key] = min(int(k), int(reference_sample_limit))
    return limited


def _valid_submission_candidates(trajectory: list[Any]) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for item in trajectory:
        if not isinstance(item, dict):
            continue
        if not item.get("rules_passed"):
            continue
        submission_dir = item.get("submission_dir")
        candidate_id = item.get("candidate_id")
        if not isinstance(submission_dir, str) or not isinstance(candidate_id, str):
            continue
        key = (candidate_id, submission_dir)
        if key in seen:
            continue
        seen.add(key)
        rows.append(item)
    return rows


def _hypothesis_lab_reference_early_patch_runner(
    *,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    reference_path: str | Path,
    reference_sample_limit_schedule: Sequence[int | None],
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> CandidateRunner:
    reference_runner = _hypothesis_lab_runner(
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        reference_path=reference_path,
        reference_sample_limit_schedule=reference_sample_limit_schedule,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    reference_round_count = len(list(reference_sample_limit_schedule))
    if task1_test_path is None:
        raise HermesBatchOptimizationLoopError(
            "hypothesis_lab_reference_early_patch requires task1_test_path"
        )

    def run_candidate(**kwargs: Any) -> dict[str, Any] | None:
        round_index = int(kwargs["round_index"])
        if round_index <= reference_round_count:
            return reference_runner(**kwargs)
        if round_index != reference_round_count + 1:
            return None
        best_candidate = kwargs.get("best_candidate")
        trajectory = kwargs.get("trajectory")
        if not isinstance(best_candidate, dict) or not isinstance(trajectory, list):
            return None
        early_source = _best_early_source_candidate(trajectory)
        if early_source is None:
            return None
        if early_source.get("candidate_id") == best_candidate.get("candidate_id"):
            return None
        return _build_early_segment_patch_candidate(
            output_dir=Path(kwargs["output_dir"]),
            base_candidate=best_candidate,
            early_source_candidate=early_source,
            task1_test_path=Path(task1_test_path),
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )

    return run_candidate


def _best_early_source_candidate(
    trajectory: list[Any],
) -> dict[str, Any] | None:
    candidates = [item for item in trajectory if isinstance(item, dict)]
    candidates = [
        item
        for item in candidates
        if item.get("rules_passed") and _candidate_signals(item).get("early_segment_score") is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: float(
            _candidate_signals(item).get("early_segment_score") or 0.0
        ),
    )


def _build_early_segment_patch_candidate(
    *,
    output_dir: Path,
    base_candidate: dict[str, Any],
    early_source_candidate: dict[str, Any],
    task1_test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    base_dir = _candidate_submission_dir(base_candidate)
    early_dir = _candidate_submission_dir(early_source_candidate)
    base_prediction = _load_submission_prediction(
        submission_dir=base_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    early_prediction = _load_submission_prediction(
        submission_dir=early_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    if base_prediction.shape != early_prediction.shape:
        raise HermesBatchOptimizationLoopError(
            "early patch source and base predictions must have matching shapes"
        )
    patched = np.array(base_prediction, copy=True)
    patched[:, 10:57, :] = early_prediction[:, 10:57, :]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "early_segment_patch_prediction.npz"
    np.savez_compressed(prediction_path, prediction=patched)
    elapsed = time.perf_counter() - start
    base_metadata = (
        base_candidate.get("metadata") if isinstance(base_candidate.get("metadata"), dict) else {}
    )
    reference_audit = (
        base_metadata.get("reference_audit")
        if isinstance(base_metadata.get("reference_audit"), dict)
        else None
    )
    if reference_audit is None and isinstance(base_candidate.get("reference_audit"), dict):
        reference_audit = base_candidate["reference_audit"]
    submission = build_competition_submission(
        task_id=1,
        test_path=task1_test_path,
        output_dir=output_dir / "submission_package",
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=elapsed,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    candidate_report_path = output_dir / "early_segment_patch_candidate.json"
    candidate_id = (
        "early_segment_patch"
        f"_{early_source_candidate.get('candidate_id', 'early')}"
        f"_into_{base_candidate.get('candidate_id', 'base')}"
    )
    report = {
        "schema_version": 1,
        "state_hint": "EARLY_SEGMENT_PATCH_CANDIDATE_READY",
        "candidate_id": candidate_id,
        "base_candidate_id": base_candidate.get("candidate_id"),
        "early_source_candidate_id": early_source_candidate.get("candidate_id"),
        "reference_audit": reference_audit,
        "copied_time_start": 10,
        "copied_time_stop": 57,
        "prediction_path": str(prediction_path),
        "submission_dir": submission["artifacts"]["submission_dir"],
        "elapsed_seconds": elapsed,
    }
    _write_json(candidate_report_path, report)
    return {
        "candidate_id": candidate_id,
        "submission_dir": submission["artifacts"]["submission_dir"],
        "source": "early_segment_patch",
        "candidate_report_path": str(candidate_report_path),
        "base_candidate_id": base_candidate.get("candidate_id"),
        "base_submission_dir": base_candidate.get("submission_dir"),
        "early_source_candidate_id": early_source_candidate.get("candidate_id"),
        "early_source_submission_dir": early_source_candidate.get("submission_dir"),
        "reference_audit": reference_audit,
        "copied_time_start": 10,
        "copied_time_stop": 57,
    }


def _build_early_subsegment_patch_candidate(
    *,
    output_dir: Path,
    base_candidate: dict[str, Any],
    source_candidates: Sequence[dict[str, Any]],
    task1_test_path: Path,
    task1_target_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    source_stability_guard: bool = False,
    handoff_smoothing_weight: float | None = None,
    force: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    base_dir = _candidate_submission_dir(base_candidate)
    base_prediction = _load_submission_prediction(
        submission_dir=base_dir,
        prediction_dataset_key=prediction_dataset_key,
    )
    target = load_hdf5_tensor(task1_target_path, dataset_key=dataset_key)
    if target.shape != base_prediction.shape:
        raise HermesBatchOptimizationLoopError(
            "early subsegment patch target and base predictions must have matching shapes"
        )
    loaded_sources = _load_patch_source_predictions(
        base_candidate=base_candidate,
        source_candidates=source_candidates,
        prediction_dataset_key=prediction_dataset_key,
        expected_shape=base_prediction.shape,
    )
    if not loaded_sources:
        raise HermesBatchOptimizationLoopError(
            "early subsegment patch requires at least one valid source candidate"
        )
    source_weight = _normalized_patch_source_weight(handoff_smoothing_weight)
    patched = np.array(base_prediction, copy=True)
    window_sources = []
    stable_source = (
        _best_stable_patch_source(loaded_sources=loaded_sources, target=target)
        if source_stability_guard
        else None
    )
    for time_start, time_stop in _early_subsegment_patch_windows():
        best = stable_source or min(
            loaded_sources,
            key=lambda item: _absolute_window_rel_mse(
                prediction=item["prediction"],
                target=target,
                time_start=time_start,
                time_stop=time_stop,
            ),
        )
        rel_mse = _absolute_window_rel_mse(
            prediction=best["prediction"],
            target=target,
            time_start=time_start,
            time_stop=time_stop,
        )
        source_window = best["prediction"][:, time_start:time_stop, :]
        if source_weight is None:
            patched[:, time_start:time_stop, :] = source_window
        else:
            base_window = base_prediction[:, time_start:time_stop, :]
            patched[:, time_start:time_stop, :] = (
                base_window * np.float32(1.0 - source_weight)
                + source_window * np.float32(source_weight)
            )
        window_sources.append(
            {
                "time_start": time_start,
                "time_stop": time_stop,
                "source_candidate_id": best["candidate_id"],
                "rel_mse": rel_mse,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "early_subsegment_patch_prediction.npz"
    np.savez_compressed(prediction_path, prediction=patched)
    elapsed = time.perf_counter() - start
    reference_audit = _reference_audit_from_candidate(base_candidate)
    submission = build_competition_submission(
        task_id=1,
        test_path=task1_test_path,
        output_dir=output_dir / "submission_package",
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=elapsed,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    candidate_id = f"early_subsegment_patch_into_{base_candidate.get('candidate_id', 'base')}"
    candidate_report_path = output_dir / "early_subsegment_patch_candidate.json"
    report = {
        "schema_version": 1,
        "state_hint": "EARLY_SUBSEGMENT_PATCH_CANDIDATE_READY",
        "candidate_id": candidate_id,
        "base_candidate_id": base_candidate.get("candidate_id"),
        "window_sources": window_sources,
        "reference_audit": reference_audit,
        "selection_target_path": str(task1_target_path),
        "selection_scope": "local_validation_window_selection",
        "source_stability_guard": bool(source_stability_guard),
        "handoff_smoothing_weight": source_weight,
        "prediction_path": str(prediction_path),
        "submission_dir": submission["artifacts"]["submission_dir"],
        "elapsed_seconds": elapsed,
    }
    _write_json(candidate_report_path, report)
    return {
        "candidate_id": candidate_id,
        "submission_dir": submission["artifacts"]["submission_dir"],
        "source": "early_subsegment_patch",
        "candidate_report_path": str(candidate_report_path),
        "base_candidate_id": base_candidate.get("candidate_id"),
        "base_submission_dir": base_candidate.get("submission_dir"),
        "window_sources": window_sources,
        "reference_audit": reference_audit,
        "selection_scope": "local_validation_window_selection",
        "source_stability_guard": bool(source_stability_guard),
        "handoff_smoothing_weight": source_weight,
    }


def _best_stable_patch_source(
    *,
    loaded_sources: Sequence[dict[str, Any]],
    target: np.ndarray,
) -> dict[str, Any]:
    return min(
        loaded_sources,
        key=lambda item: statistics.mean(
            _absolute_window_rel_mse(
                prediction=item["prediction"],
                target=target,
                time_start=time_start,
                time_stop=time_stop,
            )
            for time_start, time_stop in _early_subsegment_patch_windows()
        ),
    )


def _normalized_patch_source_weight(value: float | None) -> float | None:
    if value is None:
        return None
    weight = float(value)
    if weight < 0.0 or weight > 1.0:
        raise HermesBatchOptimizationLoopError(
            "handoff_smoothing_weight must be between 0.0 and 1.0"
        )
    return weight


def _candidate_option_bool(
    options: Mapping[str, Any] | None,
    key: str,
) -> bool:
    if not isinstance(options, Mapping):
        return False
    return bool(options.get(key)) if isinstance(options.get(key), bool) else False


def _candidate_option_float(
    options: Mapping[str, Any] | None,
    key: str,
) -> float | None:
    if not isinstance(options, Mapping):
        return None
    value = options.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _load_patch_source_predictions(
    *,
    base_candidate: dict[str, Any],
    source_candidates: Sequence[dict[str, Any]],
    prediction_dataset_key: str,
    expected_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for candidate in [base_candidate, *source_candidates]:
        candidate_id = candidate.get("candidate_id")
        submission_dir = candidate.get("submission_dir")
        if not isinstance(candidate_id, str) or not isinstance(submission_dir, str):
            continue
        key = (candidate_id, submission_dir)
        if key in seen:
            continue
        seen.add(key)
        prediction = _load_submission_prediction(
            submission_dir=Path(submission_dir),
            prediction_dataset_key=prediction_dataset_key,
        )
        if prediction.shape != expected_shape:
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "submission_dir": submission_dir,
                "prediction": prediction,
            }
        )
    return rows


def _early_subsegment_patch_windows() -> list[tuple[int, int]]:
    return [(10, 20), (20, 35), (35, 57)]


def _absolute_window_rel_mse(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    time_start: int,
    time_stop: int,
) -> float:
    squared_error = (prediction[:, time_start:time_stop, :] - target[:, time_start:time_stop, :]) ** 2
    target_squared = target[:, time_start:time_stop, :] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_by_sample_time = (
            np.sum(squared_error, axis=2)
            / (np.sum(target_squared, axis=2) + 1e-12)
        )
    return float(np.mean(np.minimum(np.mean(rel_by_sample_time, axis=1), 5.0)))


def _reference_audit_from_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    reference_audit = (
        metadata.get("reference_audit")
        if isinstance(metadata.get("reference_audit"), dict)
        else None
    )
    if reference_audit is None and isinstance(candidate.get("reference_audit"), dict):
        reference_audit = candidate["reference_audit"]
    return reference_audit


def _load_submission_prediction(
    *,
    submission_dir: Path,
    prediction_dataset_key: str,
) -> np.ndarray:
    prediction_path = submission_dir / "task1_pred.hdf5"
    if not prediction_path.exists():
        raise HermesBatchOptimizationLoopError(
            f"task1 prediction not found for early patch: {prediction_path}"
        )
    h5py = __import__("h5py")
    with h5py.File(prediction_path, "r") as handle:
        if prediction_dataset_key not in handle:
            raise HermesBatchOptimizationLoopError(
                f"dataset {prediction_dataset_key!r} not found in {prediction_path}"
            )
        prediction = np.asarray(handle[prediction_dataset_key][:], dtype=np.float32)
    if prediction.ndim != 3 or prediction.shape[1] != 200 or prediction.shape[2] != 256:
        raise HermesBatchOptimizationLoopError(
            f"expected task1 prediction shape [N, 200, 256], got {prediction.shape}"
        )
    return prediction


def _early_refinement_blueprint() -> dict[str, Any]:
    return {
        "blueprint_id": "bp_task1_early_refinement_hermes_direct_001",
        "candidate_family": "task1_early_refinement",
        "search_mode": "early_segment_objective_search",
        "parameter_grid": {
            "candidate_params": _early_refinement_candidate_params(),
        },
        "sandbox": {
            "allowed_executor": "controlled_numpy_task1_early_refinement",
            "primary_metric": "task1_early_rel_mse_guarded_score",
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }


def _early_transition_expert_blueprint() -> dict[str, Any]:
    return {
        "blueprint_id": "bp_task1_early_transition_expert_hermes_direct_001",
        "candidate_family": "task1_early_transition_expert",
        "search_mode": "early_segment_transition_expert_search",
        "parameter_grid": {
            "candidate_params": _early_transition_expert_candidate_params(),
        },
        "sandbox": {
            "allowed_executor": "controlled_numpy_task1_early_transition_expert",
            "primary_metric": "task1_early_rel_mse_guarded_score",
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }


def _front25_dynamics_blueprint() -> dict[str, Any]:
    return {
        "blueprint_id": "bp_task1_front25_dynamics_hermes_direct_001",
        "candidate_family": "task1_front25_dynamics",
        "search_mode": "front25_short_dynamics_search",
        "parameter_grid": {
            "candidate_params": _front25_dynamics_candidate_params(
                search_budget="smoke"
            ),
        },
        "sandbox": {
            "allowed_executor": "controlled_numpy_task1_front25_dynamics",
            "primary_metric": "task1_early_rel_mse_guarded_score",
            "selection_objective": "early_rel_mse",
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }


def _short_window_corrector_blueprint(*, search_budget: str = "smoke") -> dict[str, Any]:
    return {
        "blueprint_id": "bp_task1_short_window_corrector_hermes_direct_001",
        "candidate_family": "task1_short_window_corrector",
        "search_mode": "short_window_corrector_early_middle_search",
        "parameter_grid": {
            "candidate_params": _short_window_corrector_candidate_params(
                search_budget=search_budget
            ),
        },
        "sandbox": {
            "allowed_executor": "controlled_numpy_task1_short_window_corrector",
            "primary_metric": "task1_early_rel_mse_guarded_score",
            "selection_objective": "early_rel_mse",
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }


def _middle_late_retrieval_blueprint() -> dict[str, Any]:
    return {
        "blueprint_id": "bp_task1_middle_late_retrieval_hermes_direct_001",
        "candidate_family": "task1_retrieval_refinement_local_search",
        "search_mode": "middle_late_base_search_for_early_patch",
        "parameter_grid": {
            "candidate_params": _middle_late_retrieval_candidate_params(),
        },
        "sandbox": {
            "allowed_executor": "controlled_numpy_retrieval_refinement",
            "primary_metric": "task1_middle_late_score",
            "selection_objective": "middle_late_score",
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }


def _middle_late_retrieval_candidate_params() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in [48, 56, 64, 72, 80, 96, 99]:
        for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rows.append(
                {
                    "k": k,
                    "mode": "affine",
                    "early_blend_steps": 0,
                    "persistence_weight_start": 0.0,
                    "spectral_strength": strength,
                }
            )
    for k in [64, 72, 96, 99]:
        for mode in ["affine", "delta", "raw"]:
            for strength in [0.0, 0.5]:
                rows.append(
                    {
                        "k": k,
                        "mode": mode,
                        "early_blend_steps": 0,
                        "persistence_weight_start": 0.0,
                        "spectral_strength": strength,
                    }
                )
    for k in [64, 72, 96, 99]:
        for early_steps in [47, 60, 80]:
            for weight in [0.25, 0.5]:
                for strength in [0.0, 0.5]:
                    rows.append(
                        {
                            "k": k,
                            "mode": "affine",
                            "early_blend_steps": early_steps,
                            "persistence_weight_start": weight,
                            "spectral_strength": strength,
                        }
                    )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int, float, float]] = set()
    for row in rows:
        key = (
            int(row["k"]),
            str(row["mode"]),
            int(row["early_blend_steps"]),
            float(row["persistence_weight_start"]),
            float(row["spectral_strength"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _front25_dynamics_candidate_params(
    *, search_budget: str = "full"
) -> list[dict[str, Any]]:
    if search_budget not in {"full", "smoke"}:
        raise HermesBatchOptimizationLoopError(
            f"unsupported front25 dynamics search_budget: {search_budget}"
        )
    if search_budget == "smoke":
        return [
            {
                "k": 32,
                "horizon_steps": 25,
                "ridge_alpha": 1.0e-5,
                "correction_weight": 0.75,
                "dynamics_mode": "velocity_feature_ridge",
                "feature_mode": "last_velocity_gradient",
                "distance_temperature": 1.0,
                "spectral_strength": 0.0,
                "viscosity": 0.001,
                "reference_time_stride": 1,
                "base_k": 96,
                "base_mode": "affine",
            },
            {
                "k": 64,
                "horizon_steps": 25,
                "ridge_alpha": 1.0e-5,
                "correction_weight": 0.75,
                "dynamics_mode": "hybrid",
                "feature_mode": "last_velocity_gradient",
                "distance_temperature": 1.0,
                "spectral_strength": 0.0,
                "viscosity": 0.001,
                "reference_time_stride": 1,
                "base_k": 96,
                "base_mode": "affine",
            },
            {
                "k": 64,
                "horizon_steps": 25,
                "ridge_alpha": 1.0e-5,
                "correction_weight": 0.5,
                "dynamics_mode": "pseudospectral",
                "feature_mode": "last_velocity_gradient",
                "distance_temperature": 1.0,
                "spectral_strength": 0.0,
                "viscosity": 0.001,
                "reference_time_stride": 1,
                "base_k": 96,
                "base_mode": "affine",
                "pseudospectral_dt": 0.03,
                "pseudospectral_substeps": 4,
                "pseudospectral_filter_strength": 0.01,
                "pseudospectral_dealias_fraction": 2.0 / 3.0,
            },
        ]
    rows: list[dict[str, Any]] = []
    for k in [32, 48, 64, 96]:
        for mode in [
            "delta_mean",
            "velocity_feature_ridge",
            "burgers_feature_ridge",
            "hybrid",
        ]:
            for ridge_alpha in [1.0e-8, 1.0e-5, 1.0e-3]:
                rows.append(
                    {
                        "k": k,
                        "horizon_steps": 25,
                        "ridge_alpha": ridge_alpha,
                        "correction_weight": 1.0,
                        "dynamics_mode": mode,
                        "feature_mode": "last_velocity_gradient",
                        "distance_temperature": 1.0,
                        "spectral_strength": 0.0,
                        "viscosity": 0.001,
                        "reference_time_stride": 1,
                        "base_k": 96,
                        "base_mode": "affine",
                    }
                )
    for k in [1, 4, 8, 16, 32, 64]:
        for mode in ["delta_mean", "velocity_feature_ridge", "hybrid"]:
            for weight in [0.75, 1.0]:
                rows.append(
                    {
                        "k": k,
                        "horizon_steps": 25,
                        "ridge_alpha": 1.0e-5,
                        "correction_weight": weight,
                        "dynamics_mode": mode,
                        "feature_mode": "last_velocity_gradient",
                        "distance_temperature": 1.0,
                        "spectral_strength": 0.0,
                        "viscosity": 0.001,
                        "reference_time_stride": 5,
                        "base_k": 96,
                        "base_mode": "affine",
                    }
                )
    for weight in [0.5, 0.75]:
        rows.append(
            {
                "k": 64,
                "horizon_steps": 25,
                "ridge_alpha": 1.0e-5,
                "correction_weight": weight,
                "dynamics_mode": "hybrid",
                "feature_mode": "last_velocity_gradient",
                "distance_temperature": 1.0,
                "spectral_strength": 0.0,
                "viscosity": 0.001,
                "reference_time_stride": 1,
                "base_k": 96,
                "base_mode": "affine",
            }
        )
    for dt in [0.03, 0.05]:
        for filter_strength in [0.0, 0.01, 0.02]:
            for weight in [0.25, 0.5, 0.75]:
                rows.append(
                    {
                        "k": 64,
                        "horizon_steps": 25,
                        "ridge_alpha": 1.0e-5,
                        "correction_weight": weight,
                        "dynamics_mode": "pseudospectral",
                        "feature_mode": "last_velocity_gradient",
                        "distance_temperature": 1.0,
                        "spectral_strength": 0.0,
                        "viscosity": 0.001,
                        "reference_time_stride": 1,
                        "base_k": 96,
                        "base_mode": "affine",
                        "pseudospectral_dt": dt,
                        "pseudospectral_substeps": 4,
                        "pseudospectral_filter_strength": filter_strength,
                        "pseudospectral_dealias_fraction": 2.0 / 3.0,
                    }
                )
    return rows


def _short_window_corrector_candidate_params(
    *, search_budget: str = "full"
) -> list[dict[str, Any]]:
    if search_budget not in {"smoke", "full"}:
        raise HermesBatchOptimizationLoopError(
            f"unsupported short window corrector search_budget: {search_budget}"
        )
    if search_budget == "smoke":
        return [
            {
                "start_step": 20,
                "stop_step": 35,
                "feature_count": 8,
                "ridge_alpha": 1.0e-4,
                "correction_weight": 0.75,
                "reference_time_stride": 1,
                "corrector_mode": "context_pod_ridge",
                "base_k": 96,
                "base_mode": "affine",
            },
            {
                "start_step": 35,
                "stop_step": 57,
                "feature_count": 8,
                "ridge_alpha": 1.0e-4,
                "correction_weight": 0.75,
                "reference_time_stride": 1,
                "corrector_mode": "context_pod_ridge",
                "base_k": 96,
                "base_mode": "affine",
            },
            {
                "start_step": 20,
                "stop_step": 57,
                "feature_count": 8,
                "ridge_alpha": 1.0e-4,
                "correction_weight": 0.5,
                "reference_time_stride": 1,
                "corrector_mode": "delta_pod_ridge",
                "base_k": 96,
                "base_mode": "affine",
            },
        ]
    rows: list[dict[str, Any]] = []
    for start, stop in [(10, 20), (20, 35), (35, 57), (10, 35), (20, 57)]:
        for feature_count in [4, 8]:
            for ridge_alpha in [1.0e-4]:
                for weight in [0.25, 0.5, 0.75, 1.0]:
                    for mode in ["context_pod_ridge", "delta_pod_ridge"]:
                        for stride in [1, 5]:
                            reduced_steps = (200 + stride - 1) // stride
                            if stop > reduced_steps:
                                continue
                            rows.append(
                                {
                                    "start_step": start,
                                    "stop_step": stop,
                                    "feature_count": feature_count,
                                    "ridge_alpha": ridge_alpha,
                                    "correction_weight": weight,
                                    "reference_time_stride": stride,
                                    "corrector_mode": mode,
                                    "base_k": 96,
                                    "base_mode": "affine",
                                }
                            )
    return rows


def _early_transition_expert_candidate_params() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in [32, 48, 64]:
        for mode in ["velocity_ridge", "delta_mean", "hybrid"]:
            for correction_weight in [0.5, 0.75, 1.0]:
                rows.append(
                    {
                        "k": k,
                        "mode": "affine",
                        "early_blend_steps": 0,
                        "persistence_weight_start": 0.0,
                        "spectral_strength": 0.0,
                        "horizon_steps": 47,
                        "ridge_alpha": 1e-3,
                        "correction_weight": correction_weight,
                        "transition_mode": mode,
                        "feature_mode": "last_velocity",
                    }
                )
    for ridge_alpha in [1e-4, 1e-3, 1e-2]:
        rows.append(
            {
                "k": 48,
                "mode": "affine",
                "early_blend_steps": 0,
                "persistence_weight_start": 0.0,
                "spectral_strength": 0.0,
                "horizon_steps": 47,
                "ridge_alpha": ridge_alpha,
                "correction_weight": 1.0,
                "transition_mode": "velocity_ridge",
                "feature_mode": "last_velocity",
            }
        )
    return rows


def _early_refinement_candidate_params() -> list[dict[str, Any]]:
    anchors = [
        "persistence",
        "retrieval",
        "linear_velocity",
        "damped_linear_velocity",
        "scaled_damped_velocity",
        "retrieval_derivative_match",
        "phase_aligned_retrieval",
        "mean_trend_bias",
        "second_order_velocity",
        "affine_context_calibration",
        "phase_derivative_hybrid",
    ]
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        for early_steps in [12, 20, 32, 47]:
            row = {
                "k": 48,
                "mode": "affine",
                "early_blend_steps": early_steps,
                "persistence_weight_start": 0.75,
                "spectral_strength": 0.25,
                "early_anchor": anchor,
                "handoff_curve": "linear",
            }
            if anchor == "scaled_damped_velocity":
                for velocity_scale in [0.25, 0.5, 0.75]:
                    scaled = dict(row)
                    scaled["velocity_scale"] = velocity_scale
                    rows.append(scaled)
            elif anchor in {"phase_aligned_retrieval", "phase_derivative_hybrid"}:
                for phase_shift_scale in [0.5, 1.0]:
                    shifted = dict(row)
                    shifted["phase_shift_scale"] = phase_shift_scale
                    rows.append(shifted)
            else:
                rows.append(row)
    for weight in [0.25, 0.5, 1.0]:
        rows.append(
            {
                "k": 48,
                "mode": "affine",
                "early_blend_steps": 47,
                "persistence_weight_start": weight,
                "spectral_strength": 0.0,
                "early_anchor": "retrieval_derivative_match",
                "handoff_curve": "cosine",
            }
        )
    for k in [48, 96]:
        for early_steps in [10, 25]:
            for weight in [0.25, 0.5]:
                for anchor in [
                    "linear_velocity",
                    "retrieval_derivative_match",
                    "phase_aligned_retrieval",
                ]:
                    row = {
                        "k": k,
                        "mode": "affine",
                        "early_blend_steps": early_steps,
                        "persistence_weight_start": weight,
                        "spectral_strength": 0.0,
                        "early_anchor": anchor,
                        "handoff_curve": "cosine",
                    }
                    if anchor == "phase_aligned_retrieval":
                        row["phase_shift_scale"] = 1.0
                    rows.append(row)
    return rows


def _linear_trend_spectral_schedule(
    hypothesis_blueprint_path: str | Path | None,
) -> list[dict[str, float | int]]:
    if hypothesis_blueprint_path is not None:
        blueprint = _load_linear_trend_spectral_blueprint(Path(hypothesis_blueprint_path))
        return _blueprint_schedule(blueprint)
    return [
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


def _load_linear_trend_spectral_blueprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HermesBatchOptimizationLoopError(f"hypothesis blueprint not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesBatchOptimizationLoopError(
            f"invalid hypothesis blueprint JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise HermesBatchOptimizationLoopError("hypothesis blueprint must be a JSON object")
    blueprints = payload.get("blueprints")
    if not isinstance(blueprints, list):
        raise HermesBatchOptimizationLoopError(
            "hypothesis blueprint file must include blueprints list"
        )
    for blueprint in blueprints:
        if (
            isinstance(blueprint, dict)
            and blueprint.get("candidate_family") == "linear_trend_spectral"
            and isinstance(blueprint.get("parameter_grid"), dict)
        ):
            return blueprint
    raise HermesBatchOptimizationLoopError(
        "no linear_trend_spectral blueprint found in hypothesis blueprint file"
    )


def _grid_schedule(grid: dict[str, Any]) -> list[dict[str, float | int]]:
    required = ("slope_window", "gain", "decay", "spectral_strength")
    values: dict[str, list[Any]] = {}
    for key in required:
        raw = grid.get(key)
        if not isinstance(raw, list) or not raw:
            raise HermesBatchOptimizationLoopError(
                f"linear_trend_spectral blueprint missing non-empty {key} list"
            )
        values[key] = raw
    late_strengths = grid.get("late_spectral_strength", [None])
    late_starts = grid.get("late_start", [None])
    if not isinstance(late_strengths, list) or not late_strengths:
        raise HermesBatchOptimizationLoopError(
            "linear_trend_spectral blueprint late_spectral_strength must be a list"
        )
    if not isinstance(late_starts, list) or not late_starts:
        raise HermesBatchOptimizationLoopError(
            "linear_trend_spectral blueprint late_start must be a list"
        )
    schedule = []
    for slope_window in values["slope_window"]:
        for gain in values["gain"]:
            for decay in values["decay"]:
                for strength in values["spectral_strength"]:
                    for late_strength in late_strengths:
                        for late_start in late_starts:
                            params = {
                                "slope_window": int(slope_window),
                                "gain": float(gain),
                                "decay": float(decay),
                                "spectral_strength": float(strength),
                            }
                            if late_strength is not None:
                                params["late_spectral_strength"] = float(late_strength)
                            if late_start is not None:
                                params["late_start"] = int(late_start)
                            schedule.append(params)
    return schedule


def _blueprint_schedule(blueprint: dict[str, Any]) -> list[dict[str, float | int]]:
    schedule: list[dict[str, float | int]] = []
    seen: set[tuple[int, float, float, float]] = set()
    seed_params = blueprint.get("seed_params")
    if isinstance(seed_params, list):
        for row in seed_params:
            if not isinstance(row, dict) or not isinstance(row.get("params"), dict):
                continue
            params = _normalize_linear_trend_spectral_params(row["params"])
            key = _linear_trend_spectral_param_key(params)
            if key not in seen:
                schedule.append(params)
                seen.add(key)
    for params in _grid_schedule(blueprint["parameter_grid"]):
        key = _linear_trend_spectral_param_key(params)
        if key not in seen:
            schedule.append(params)
            seen.add(key)
    if not schedule:
        raise HermesBatchOptimizationLoopError(
            "linear_trend_spectral blueprint produced an empty schedule"
        )
    return schedule


def _normalize_linear_trend_spectral_params(
    params: dict[str, Any],
) -> dict[str, float | int]:
    normalized: dict[str, float | int] = {
        "slope_window": int(params["slope_window"]),
        "gain": float(params["gain"]),
        "decay": float(params["decay"]),
        "spectral_strength": float(params["spectral_strength"]),
    }
    if "late_spectral_strength" in params:
        normalized["late_spectral_strength"] = float(params["late_spectral_strength"])
    if "late_start" in params:
        normalized["late_start"] = int(params["late_start"])
    return normalized


def _linear_trend_spectral_param_key(
    params: dict[str, float | int],
) -> tuple[int, float, float, float, float | None, int | None]:
    return (
        int(params["slope_window"]),
        float(params["gain"]),
        float(params["decay"]),
        float(params["spectral_strength"]),
        float(params["late_spectral_strength"])
        if "late_spectral_strength" in params
        else None,
        int(params["late_start"]) if "late_start" in params else None,
    )


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
        "early_segment_score": None,
        "early_rel_mse": None,
    }


def _objective_signals(judge_report: dict[str, Any]) -> dict[str, Any]:
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
        score = task.get("score")
        if isinstance(score, dict):
            timeout_zeroed = timeout_zeroed or bool(score.get("timeout_zeroed"))
            _append_number(segment_scores, score.get("segment_score"))
            _append_number(prediction_scores, score.get("prediction_accuracy_score"))
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
    payload = _load_memory_json(path)
    return _avoid_set_from_payload(payload)


def _avoid_set_from_payload(payload: dict[str, Any]) -> set[tuple[str, float]]:
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


def _load_search_guidance(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = _load_memory_json(path)
    return _search_guidance_from_payload(payload)


def _search_guidance_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    guidance = payload.get("search_guidance")
    if not isinstance(guidance, list):
        return []
    return [dict(item) for item in guidance if isinstance(item, dict)]


def _memory_constraints_summary(
    *,
    memory_json_path: str | Path | None,
    avoid_repeating: bool,
    memory_payload: dict[str, Any],
    avoid_count: int,
    search_guidance_count: int,
) -> dict[str, Any]:
    repair_actions = _repair_policy_actions_from_payload(memory_payload)
    return {
        "memory_json_path": str(memory_json_path) if memory_json_path is not None else None,
        "avoid_repeating": avoid_repeating,
        "avoid_count": avoid_count,
        "search_guidance_count": search_guidance_count,
        "repair_policy_recommended_action_count": len(repair_actions),
        "repair_policy_recommended_actions": repair_actions,
        "input_sources": _memory_input_sources(memory_payload),
    }


def _repair_policy_actions_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("repair_policy_recommended_actions")
    if not isinstance(actions, list):
        return []
    return [dict(item) for item in actions if isinstance(item, dict)]


def _memory_input_sources(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("input_sources")
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _load_memory_json(path: str | Path) -> dict[str, Any]:
    memory_file = Path(path)
    if not memory_file.exists():
        raise HermesBatchOptimizationLoopError(f"memory_json_path not found: {path}")
    try:
        payload = json.loads(memory_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesBatchOptimizationLoopError(f"invalid memory JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HermesBatchOptimizationLoopError("memory JSON must be an object")
    return payload


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
