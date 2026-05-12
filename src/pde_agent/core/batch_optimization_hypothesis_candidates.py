"""Hypothesis Lab candidate families for batch optimization."""

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.batch_optimization_common import BatchOptimizationLoopError
from pde_agent.core.burgers_hdf5_io import load_hdf5_tensor
from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.hypothesis_lab import run_blueprint_sandbox, run_hypothesis_lab
from pde_agent.core.literature_expansion import build_local_asset_algorithm_candidates
from pde_agent.core.local_strategy_library import build_local_strategy_library
from pde_agent.core.task1_checkpoint_front_window_adapter import (
    build_checkpoint_front_window_adapter_sources,
)

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
        raise BatchOptimizationLoopError(
            "hypothesis_lab candidate generator requires task1_test_path"
        )
    if task1_target_path is None:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "hypothesis_lab_early_search requires task1_test_path"
        )
    if task1_target_path is None:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "hypothesis_lab_early_transition_search requires task1_test_path"
        )
    if task1_target_path is None:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "hypothesis_lab_front25_dynamics_smoke requires task1_test_path"
        )
    if task1_target_path is None:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "hypothesis_lab_short_window_corrector_smoke requires task1_test_path"
        )
    if task1_target_path is None:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "hypothesis_lab_middle_late_patch_search requires task1_test_path"
        )
    if task1_target_path is None:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "hypothesis_lab_early_subsegment_patch_search requires task1_test_path"
        )
    if task1_target_path is None:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            "early subsegment patch target and base predictions must have matching shapes"
        )
    loaded_sources = _load_patch_source_predictions(
        base_candidate=base_candidate,
        source_candidates=source_candidates,
        prediction_dataset_key=prediction_dataset_key,
        expected_shape=base_prediction.shape,
    )
    if not loaded_sources:
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
            f"task1 prediction not found for early patch: {prediction_path}"
        )
    h5py = __import__("h5py")
    with h5py.File(prediction_path, "r") as handle:
        if prediction_dataset_key not in handle:
            raise BatchOptimizationLoopError(
                f"dataset {prediction_dataset_key!r} not found in {prediction_path}"
            )
        prediction = np.asarray(handle[prediction_dataset_key][:], dtype=np.float32)
    if prediction.ndim != 3 or prediction.shape[1] != 200 or prediction.shape[2] != 256:
        raise BatchOptimizationLoopError(
            f"expected task1 prediction shape [N, 200, 256], got {prediction.shape}"
        )
    return prediction


def _early_refinement_blueprint() -> dict[str, Any]:
    return {
        "blueprint_id": "bp_task1_early_refinement_direct_001",
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
        "blueprint_id": "bp_task1_early_transition_expert_direct_001",
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
        "blueprint_id": "bp_task1_front25_dynamics_direct_001",
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
        "blueprint_id": "bp_task1_short_window_corrector_direct_001",
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
        "blueprint_id": "bp_task1_middle_late_retrieval_direct_001",
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
        raise BatchOptimizationLoopError(
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
        raise BatchOptimizationLoopError(
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

def _candidate_submission_dir(candidate: dict[str, Any]) -> Path:
    value = candidate.get("submission_dir")
    if not isinstance(value, str) or not value:
        raise BatchOptimizationLoopError("candidate missing submission_dir")
    return Path(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
