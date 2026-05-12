"""Physics and numerical-dynamics candidate families for batch optimization."""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.batch_optimization_common import BatchOptimizationLoopError
from pde_agent.core.burgers_hdf5_io import load_hdf5_tensor
from pde_agent.core.burgers_pseudospectral import pseudospectral_burgers_rollout
from pde_agent.core.burgers_residual_refinement import residual_guided_refinement
from pde_agent.core.burgers_spectral_element import sem_burgers_rollout
from pde_agent.core.competition_scoring import (
    EARLY_FUTURE_STOP,
    FUTURE_STEPS,
    MIDDLE_FUTURE_STOP,
    _late_segment,
    _rel_mse_segment,
)
from pde_agent.core.competition_submission import build_competition_submission

CandidateRunner = Callable[..., dict[str, Any] | None]

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
        raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "SEM window blend base and SEM predictions must have matching shapes"
        )
    if context.shape[0] != base_prediction.shape[0]:
        raise BatchOptimizationLoopError(
            "SEM window blend context and prediction sample counts must match"
        )
    start_index = max(10, int(window_start))
    stop_index = min(int(window_stop), base_prediction.shape[1])
    if stop_index <= start_index:
        raise BatchOptimizationLoopError(
            "SEM window blend requires window_stop > window_start"
        )
    weight = float(blend_weight)
    if weight < 0.0 or weight > 1.0:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
        except BatchOptimizationLoopError:
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "SEM window gate direction must be 'lt' or 'gt'"
        )
    start_index = max(10, int(window_start))
    stop_index = min(int(window_stop), base_prediction.shape[1])
    if stop_index <= start_index:
        raise BatchOptimizationLoopError(
            "SEM gate blend requires window_stop > window_start"
        )
    weight = float(blend_weight)
    if weight < 0.0 or weight > 1.0:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
    raise BatchOptimizationLoopError(f"unsupported SEM gate feature: {feature_name}")


def _task1_segment_score_from_arrays(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        raise BatchOptimizationLoopError(
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
    raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "checkpoint gate base and checkpoint predictions must have matching shapes"
        )
    if context.shape[0] != base_prediction.shape[0]:
        raise BatchOptimizationLoopError(
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
    raise BatchOptimizationLoopError(
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
            raise BatchOptimizationLoopError(
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
    raise BatchOptimizationLoopError(
        f"unsupported checkpoint gate feature: {feature_name}"
    )

def _load_submission_prediction(
    *,
    submission_dir: Path,
    prediction_dataset_key: str,
) -> np.ndarray:
    candidates = [
        submission_dir / "task1_pred.hdf5",
        submission_dir / "task2_pred.hdf5",
    ]
    for path in candidates:
        if path.exists():
            return load_hdf5_tensor(path, dataset_key=prediction_dataset_key)
    raise BatchOptimizationLoopError(
        f"submission prediction file not found in {submission_dir}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
