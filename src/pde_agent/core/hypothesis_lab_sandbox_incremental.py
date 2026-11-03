"""Incremental sandbox execution with checkpointing for Hypothesis Lab.

Allows parameter grids to be executed partially, resumed after interruption,
and limited to a subset of combinations for rapid exploration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.burgers_hdf5_io import load_hdf5_tensor
from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.hypothesis_lab_candidate_reporting import (
    _candidate_id,
    _candidate_metadata,
)
from pde_agent.core.hypothesis_lab_predictions import _prediction_from_blueprint_params
from pde_agent.core.hypothesis_lab_sandbox_reference import (
    _load_sandbox_reference,
    _reference_audit,
)
from pde_agent.core.hypothesis_lab_sandbox_runner import (
    SANDBOX_EXECUTABLE_FAMILIES,
    _iter_parameter_grid,
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
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession
from pde_agent.core.hypothesis_lab_utils import HypothesisLabError
from pde_agent.core.manual_operator_tuning import (
    make_persistence_prediction,
    score_burgers_segments,
)
from pde_agent.core.submission_judge import judge_submission


INCREMENTAL_SANDBOX_SCHEMA_VERSION = 1


class IncrementalSandboxError(RuntimeError):
    pass


@dataclass
class _ParamResult:
    params: dict[str, Any]
    candidate_id: str
    elapsed_seconds: float
    segment_score: float
    task1_prediction_score: float
    task1_total_score_proxy: float
    segments: dict[str, Any]
    metadata: dict[str, Any]
    prediction_path: Path


def run_incremental_blueprint_sandbox(
    session: HypothesisLabSession,
    *,
    blueprint: dict[str, Any],
    context_path: str | Path,
    target_path: str | Path,
    reference_path: str | Path | None = None,
    reference_sample_limit: int | None = None,
    submission_id: str,
    limit: int | None = None,
    resume: bool = True,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    artifact_id: str = "sandbox",
    force: bool = True,
) -> dict[str, Any]:
    """Run a blueprint sandbox incrementally with checkpointing.

    Args:
        blueprint: The executable blueprint
        context_path: Path to context HDF5
        target_path: Path to target HDF5
        reference_path: Optional reference HDF5
        reference_sample_limit: Optional sample limit for reference
        submission_id: Submission identifier
        limit: Max parameter combinations to evaluate (None = all)
        resume: Whether to resume from existing checkpoint
        timing_profile: Optional timing profile
        timing_factor: Optional timing factor
        artifact_id: Artifact ID for storing results
        force: Overwrite existing artifacts

    Returns:
        Sandbox report dict (same format as run_blueprint_sandbox)
    """
    family = str(blueprint.get("candidate_family"))
    if family not in SANDBOX_EXECUTABLE_FAMILIES:
        raise IncrementalSandboxError(
            f"unsupported blueprint candidate_family: {blueprint.get('candidate_family')}"
        )

    # Setup workspace
    sandbox_dir = session.workspace / "sandbox" / artifact_id
    checkpoint_path = sandbox_dir / "checkpoint.json"
    predictions_dir = sandbox_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load data
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

    # Generate full parameter grid
    all_params = list(_iter_parameter_grid(blueprint["parameter_grid"], family=family))
    if limit is not None:
        all_params = all_params[:limit]

    # Load checkpoint if resuming
    completed_results: list[_ParamResult] = []
    if resume and checkpoint_path.exists():
        completed_results = _load_checkpoint(
            checkpoint_path, predictions_dir, family=family
        )

    completed_param_keys = {_param_key(r.params) for r in completed_results}

    # Evaluate remaining params
    remaining_params = [
        p for p in all_params if _param_key(p) not in completed_param_keys
    ]

    for idx, params in enumerate(remaining_params):
        session.emit(
            event_type="sandbox_param_started",
            payload={
                "artifact_id": artifact_id,
                "param_index": len(completed_results),
                "total_params": len(all_params),
                "candidate_id": _candidate_id(params, family=family),
            },
            tags=["sandbox", "param"],
        )

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

        # Save prediction
        pred_path = predictions_dir / f"pred_{len(completed_results):04d}.npz"
        np.savez_compressed(pred_path, prediction=prediction)

        result = _ParamResult(
            params=params,
            candidate_id=_candidate_id(params, family=family),
            elapsed_seconds=elapsed,
            segment_score=score["segment_score"],
            task1_prediction_score=score["task1_prediction_score"],
            task1_total_score_proxy=_task1_total_score_proxy(
                prediction_score=score["task1_prediction_score"],
                inference_seconds=elapsed,
            ),
            segments=score["segments"],
            metadata=_candidate_metadata(
                family=family,
                context=context,
                target=target,
                reference=reference,
                reference_audit=reference_audit,
                params=params,
            ),
            prediction_path=pred_path,
        )
        completed_results.append(result)

        # Save checkpoint
        _save_checkpoint(checkpoint_path, completed_results, blueprint)

        session.emit(
            event_type="sandbox_param_completed",
            payload={
                "artifact_id": artifact_id,
                "param_index": len(completed_results) - 1,
                "total_params": len(all_params),
                "candidate_id": result.candidate_id,
                "score": result.task1_total_score_proxy,
            },
            tags=["sandbox", "param"],
        )

    # Build final report
    return _build_sandbox_report(
        blueprint=blueprint,
        family=family,
        context=context,
        context_path=context_path,
        target=target,
        target_path=target_path,
        reference=reference,
        baseline_prediction=baseline_prediction,
        baseline_score=baseline_score,
        results=completed_results,
        sandbox_dir=sandbox_dir,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )


def _build_sandbox_report(
    *,
    blueprint: dict[str, Any],
    family: str,
    context: np.ndarray,
    context_path: str | Path,
    target: np.ndarray,
    target_path: str | Path,
    reference: np.ndarray | None,
    baseline_prediction: np.ndarray,
    baseline_score: dict[str, Any],
    results: list[_ParamResult],
    sandbox_dir: Path,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> dict[str, Any]:
    """Build the final sandbox report from evaluated parameter results."""
    if not results:
        raise IncrementalSandboxError("no parameter results to build report from")

    # Sort and pick best
    selection_objective_name = _blueprint_selection_objective_name(
        blueprint=blueprint,
        family=family,
    )
    rows = []
    for r in results:
        row = {
            "candidate_id": r.candidate_id,
            "family": family,
            "params": r.params,
            "metadata": r.metadata,
            "elapsed_seconds": r.elapsed_seconds,
            "segment_score": r.segment_score,
            "task1_prediction_score": r.task1_prediction_score,
            "task1_total_score_proxy": r.task1_total_score_proxy,
            "segments": r.segments,
        }
        rows.append(row)

    rows.sort(
        key=lambda item: _candidate_selection_key(
            item,
            family=family,
            selection_objective=selection_objective_name,
        ),
        reverse=True,
    )
    best_row = rows[0]

    # Load best prediction
    best_result = next(r for r in results if r.candidate_id == best_row["candidate_id"])
    best_prediction = np.load(best_result.prediction_path)["prediction"]

    # Build submission and judge
    prediction_path = sandbox_dir / "best_prediction.npz"
    np.savez_compressed(prediction_path, prediction=best_prediction)

    submission = build_competition_submission(
        task_id=1,
        test_path=context_path,
        output_dir=sandbox_dir / "submission_package",
        submission_id=submission_id,
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=float(best_row["elapsed_seconds"]),
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    judge = judge_submission(
        submission_dir=submission["artifacts"]["submission_dir"],
        output_path=sandbox_dir / "judge_report.json",
        task1_test_path=context_path,
        task1_target_path=target_path,
    )

    context_preserved = _context_preserved(
        prediction=best_prediction,
        context=context,
    )
    baseline_total = _task1_total_score_proxy(
        prediction_score=float(baseline_score["task1_prediction_score"]),
        inference_seconds=0.0,
    )
    delta = float(judge["total_score_proxy"]) - baseline_total

    from pde_agent.core.hypothesis_lab_sandbox_runner import _candidate_pool_attribution
    candidate_pool_attribution = _candidate_pool_attribution(
        rows=rows,
        top_candidate_count=20,
        family=family,
    )

    metric_results = _sandbox_metric_results(
        judge=judge,
        baseline_score=baseline_score,
        best_segments=best_row["segments"],
        score_delta=delta,
        context_preserved=context_preserved,
        inference_seconds=float(best_row["elapsed_seconds"]),
        max_inference_seconds=float(blueprint["sandbox"]["max_inference_seconds"]),
        promotion_min_score_delta=float(blueprint["sandbox"]["promotion_min_score_delta"]),
    )

    # Family-specific metric extensions
    if family == "task1_physics_residual_refinement":
        metadata = best_row.get("metadata") if isinstance(best_row.get("metadata"), dict) else {}
        residual_delta = metadata.get("residual_rmse_delta")
        if isinstance(residual_delta, (int, float)):
            metric_results["physics_residual_rmse_delta"] = {
                "value": float(residual_delta),
                "threshold": 0.0,
                "passed": float(residual_delta) < 0.0,
            }
    if family == "task1_pi_deeponet_correction_head":
        metadata = best_row.get("metadata") if isinstance(best_row.get("metadata"), dict) else {}
        correction_rmse = metadata.get("learned_correction_rmse")
        if isinstance(correction_rmse, (int, float)):
            metric_results["pi_deeponet_active_correction"] = {
                "value": float(correction_rmse),
                "threshold": 1e-8,
                "passed": float(correction_rmse) > 1e-8,
            }

    multiview_judge = _task1_multiview_local_judge(
        prediction=best_prediction,
        baseline_prediction=baseline_prediction,
        target=target,
    )
    best_validation_score = judge["tasks"]["1"].get("validation_score")
    if not isinstance(best_validation_score, dict):
        best_validation_score = {
            "segment_score": best_row["segment_score"],
            "segments": best_row["segments"],
        }
    early_segment_diagnosis = _task1_early_segment_diagnosis(
        baseline_score=baseline_score,
        best_score=best_validation_score,
        prediction=best_prediction,
        baseline_prediction=baseline_prediction,
        target=target,
        candidates=rows,
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
        "inference_time": float(best_row["elapsed_seconds"]),
        "judge_decision": judge["decision"],
        "metric_results": metric_results,
        "multiview_judge": multiview_judge,
        "early_segment_diagnosis": early_segment_diagnosis,
        "candidate_budget": len(rows),
        "promotion_score": promotion_score,
        "pareto_gate": pareto_gate,
    }

    report = {
        "schema_version": 1,
        "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
        "blueprint_id": blueprint.get("blueprint_id"),
        "candidate_family": family,
        "submission_id": submission_id,
        "best_candidate": {
            "candidate_id": best_row["candidate_id"],
            "params": best_row["params"],
            "segment_score": best_row["segment_score"],
            "task1_prediction_score": best_row["task1_prediction_score"],
            "task1_total_score_proxy": best_row["task1_total_score_proxy"],
            "elapsed_seconds": best_row["elapsed_seconds"],
            "metadata": best_row["metadata"],
        },
        "top_candidates": rows[:20],
        "verifier": verifier,
        "candidate_pool_attribution": candidate_pool_attribution,
        "baseline": {
            "segment_score": baseline_score["segment_score"],
            "task1_prediction_score": baseline_score["task1_prediction_score"],
            "task1_total_score_proxy": baseline_total,
        },
        "artifacts": {
            "sandbox_report_path": str(sandbox_dir / "hypothesis_lab_sandbox_report.json"),
            "best_prediction_path": str(prediction_path),
            "submission_dir": submission["artifacts"]["submission_dir"],
            "judge_report_path": str(sandbox_dir / "judge_report.json"),
        },
        "next_command": "record-sandbox-outcome",
    }

    # Write report
    report_path = sandbox_dir / "hypothesis_lab_sandbox_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


# -- Checkpoint I/O --


def _save_checkpoint(
    checkpoint_path: Path,
    results: list[_ParamResult],
    blueprint: dict[str, Any],
) -> None:
    checkpoint = {
        "schema_version": INCREMENTAL_SANDBOX_SCHEMA_VERSION,
        "blueprint_id": blueprint.get("blueprint_id"),
        "candidate_family": blueprint.get("candidate_family"),
        "completed_count": len(results),
        "results": [
            {
                "params": r.params,
                "candidate_id": r.candidate_id,
                "elapsed_seconds": r.elapsed_seconds,
                "segment_score": r.segment_score,
                "task1_prediction_score": r.task1_prediction_score,
                "task1_total_score_proxy": r.task1_total_score_proxy,
                "segments": r.segments,
                "metadata": r.metadata,
                "prediction_path": str(r.prediction_path),
            }
            for r in results
        ],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(checkpoint, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(
    checkpoint_path: Path,
    predictions_dir: Path,
    family: str,
) -> list[_ParamResult]:
    if not checkpoint_path.exists():
        return []
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if data.get("schema_version") != INCREMENTAL_SANDBOX_SCHEMA_VERSION:
        return []

    results: list[_ParamResult] = []
    for entry in data.get("results", []):
        if not isinstance(entry, dict):
            continue
        pred_path = Path(entry.get("prediction_path", ""))
        if not pred_path.exists():
            pred_path = predictions_dir / pred_path.name
        results.append(
            _ParamResult(
                params=dict(entry.get("params", {})),
                candidate_id=str(entry.get("candidate_id", "")),
                elapsed_seconds=float(entry.get("elapsed_seconds", 0.0)),
                segment_score=float(entry.get("segment_score", 0.0)),
                task1_prediction_score=float(entry.get("task1_prediction_score", 0.0)),
                task1_total_score_proxy=float(entry.get("task1_total_score_proxy", 0.0)),
                segments=dict(entry.get("segments", {})),
                metadata=dict(entry.get("metadata", {})),
                prediction_path=pred_path,
            )
        )
    return results


def _param_key(params: dict[str, Any]) -> str:
    """Stable key for deduplicating parameter sets."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))
