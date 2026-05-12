"""Candidate generators for the batch optimization loop."""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.batch_optimization_common import BatchOptimizationLoopError
from pde_agent.core.batch_optimization_hypothesis_candidates import (
    _checkpoint_front_window_adapter_patch_sources_runner,
    _early_subsegment_patch_sources_runner,
    _hypothesis_lab_early_runner,
    _hypothesis_lab_early_subsegment_patch_runner,
    _hypothesis_lab_early_transition_runner,
    _hypothesis_lab_front25_dynamics_smoke_runner,
    _hypothesis_lab_middle_late_patch_runner,
    _hypothesis_lab_reference_early_patch_runner,
    _hypothesis_lab_runner,
    _hypothesis_lab_short_window_corrector_patch_sources_runner,
    _hypothesis_lab_short_window_corrector_smoke_runner,
    _hypothesis_lab_transition_early_patch_runner,
)
from pde_agent.core.batch_optimization_physics_candidates import (
    _checkpoint_window_gate_runner,
    _physics_residual_nudge_runner,
    _sem_window_blend_runner,
    _sem_window_gate_blend_multi_search_runner,
    _sem_window_gate_blend_runner,
    _sem_window_gate_blend_search_runner,
)
from pde_agent.core.batch_optimization_memory import (
    adaptive_weight_neighbors as _adaptive_weight_neighbors,
    best_postprocess_metadata as _best_postprocess_metadata,
    normalize_weight as _normalize_weight,
    record_skip as _record_skip,
)
from pde_agent.core.burgers_hdf5_io import load_hdf5_tensor
from pde_agent.core.competition_candidate_materialization import (
    materialize_competition_candidate,
)
from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.manual_operator_tuning import (
    linear_trend_prediction,
    spectral_smooth_prediction,
)
from pde_agent.core.postprocess_candidate import generate_postprocess_candidate
from pde_agent.core.unet_competition_prediction import predict_unet_competition

CandidateRunner = Callable[..., dict[str, Any] | None]

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
            raise BatchOptimizationLoopError(
                "hypothesis_lab_reference_search requires hypothesis_reference_path"
            )
        sample_limits = list(hypothesis_reference_sample_limits or [64, 128, 256, 512])
        if not sample_limits:
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
                "hypothesis_lab_short_window_corrector_patch_sources requires hypothesis_reference_path"
            )
        if task1_test_path is None or task1_target_path is None:
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
    raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError("postprocess strategies must not be empty")
    if not weight_values:
        raise BatchOptimizationLoopError("postprocess blend weights must not be empty")
    if adaptive_step <= 0.0:
        raise BatchOptimizationLoopError("postprocess adaptive step must be positive")
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
        candidate = generate_postprocess_candidate(
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(f"hypothesis blueprint not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchOptimizationLoopError(
            f"invalid hypothesis blueprint JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise BatchOptimizationLoopError("hypothesis blueprint must be a JSON object")
    blueprints = payload.get("blueprints")
    if not isinstance(blueprints, list):
        raise BatchOptimizationLoopError(
            "hypothesis blueprint file must include blueprints list"
        )
    for blueprint in blueprints:
        if (
            isinstance(blueprint, dict)
            and blueprint.get("candidate_family") == "linear_trend_spectral"
            and isinstance(blueprint.get("parameter_grid"), dict)
        ):
            return blueprint
    raise BatchOptimizationLoopError(
        "no linear_trend_spectral blueprint found in hypothesis blueprint file"
    )


def _grid_schedule(grid: dict[str, Any]) -> list[dict[str, float | int]]:
    required = ("slope_window", "gain", "decay", "spectral_strength")
    values: dict[str, list[Any]] = {}
    for key in required:
        raw = grid.get(key)
        if not isinstance(raw, list) or not raw:
            raise BatchOptimizationLoopError(
                f"linear_trend_spectral blueprint missing non-empty {key} list"
            )
        values[key] = raw
    late_strengths = grid.get("late_spectral_strength", [None])
    late_starts = grid.get("late_start", [None])
    if not isinstance(late_strengths, list) or not late_strengths:
        raise BatchOptimizationLoopError(
            "linear_trend_spectral blueprint late_spectral_strength must be a list"
        )
    if not isinstance(late_starts, list) or not late_starts:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
    raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "postprocess candidate generation requires exactly one task test path"
        )
    return provided[0]
