"""Three-stage Hermes policy for runtime gate, accuracy refinement, and score arbitration."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_batch_optimization_loop import (
    run_hermes_batch_optimization_loop,
)
from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat


STATE_HINT = "HERMES_THREE_STAGE_POLICY_RAN"
HeartbeatRunner = Callable[..., dict[str, Any]]
BatchRunner = Callable[..., dict[str, Any]]


class HermesThreeStagePolicyError(RuntimeError):
    pass


def run_hermes_three_stage_policy(
    *,
    output_dir: str | Path,
    initial_submission_dir: str | Path,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    task2_test_path: str | Path | None = None,
    task2_target_path: str | Path | None = None,
    unet_checkpoint_path: str | Path | None = None,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    submission_id: str = "team_hermes",
    stage_deadline_seconds: float = 600.0,
    heartbeat_runner: HeartbeatRunner | None = None,
    batch_runner: BatchRunner | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_path = root / "hermes_three_stage_policy_report.json"
    if report_path.exists() and not force:
        raise HermesThreeStagePolicyError(
            f"three-stage policy report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    run_heartbeat = heartbeat_runner or run_hermes_heartbeat
    run_batch = batch_runner or run_hermes_batch_optimization_loop

    hybrid = run_heartbeat(
        output_dir=root / "01_hybrid_gate",
        initial_submission_dir=Path(initial_submission_dir),
        max_batches=1,
        batch_deadline_seconds=stage_deadline_seconds,
        max_rounds_per_batch=2,
        checkpoint_every=1,
        max_stagnant_batches=1,
        min_batch_score_improvement=1.0,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        learning_objective="auto",
        patience=2,
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
        task2_target_path=Path(task2_target_path)
        if task2_target_path is not None
        else None,
        unet_checkpoint_path=Path(unet_checkpoint_path)
        if unet_checkpoint_path is not None
        else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        candidate_generator="hybrid_gate",
        postprocess_search_mode="paired",
        postprocess_strategies=[],
        postprocess_blend_weights=[],
        postprocess_adaptive_step=0.125,
        submission_id=submission_id,
        force=True,
    )
    hybrid_best = _best_candidate(hybrid, "hybrid stage")

    accuracy = run_heartbeat(
        output_dir=root / "02_accuracy_refine",
        initial_submission_dir=Path(str(hybrid_best["submission_dir"])),
        max_batches=1,
        batch_deadline_seconds=stage_deadline_seconds,
        max_rounds_per_batch=12,
        checkpoint_every=4,
        max_stagnant_batches=1,
        min_batch_score_improvement=0.1,
        min_score_improvement=0.1,
        submit_score_threshold=120.0,
        learning_objective="accuracy",
        patience=6,
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
        task2_target_path=Path(task2_target_path)
        if task2_target_path is not None
        else None,
        unet_checkpoint_path=Path(unet_checkpoint_path)
        if unet_checkpoint_path is not None
        else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        candidate_generator="postprocess",
        postprocess_search_mode="adaptive",
        postprocess_strategies=["context_anchor_blend", "spectral_smooth"],
        postprocess_blend_weights=[0.125, 0.25, 0.375, 0.5, 0.625],
        postprocess_adaptive_step=0.0625,
        submission_id=submission_id,
        force=True,
    )
    accuracy_best = _best_candidate(accuracy, "accuracy stage")

    arbitration = run_batch(
        output_dir=root / "03_score_arbitration",
        initial_submission_dir=Path(str(accuracy_best["submission_dir"])),
        max_rounds=1,
        patience=1,
        min_score_improvement=0.001,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
        task2_target_path=Path(task2_target_path)
        if task2_target_path is not None
        else None,
        unet_checkpoint_path=Path(unet_checkpoint_path)
        if unet_checkpoint_path is not None
        else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        candidate_submission_dirs=[Path(str(hybrid_best["submission_dir"]))],
        candidate_generator="none",
        postprocess_search_mode="paired",
        postprocess_strategies=[],
        postprocess_blend_weights=[],
        postprocess_adaptive_step=0.125,
        submission_id=submission_id,
        deadline_seconds=stage_deadline_seconds,
        checkpoint_every=0,
        memory_json_path=None,
        avoid_repeating=False,
        force=True,
    )
    best = _best_candidate(arbitration, "score arbitration stage")
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "policy_report_path": str(report_path),
        "policy": "hybrid_then_accuracy_then_score",
        "stage_count": 3,
        "stages": [
            _stage_record("hybrid_gate", hybrid),
            _stage_record("accuracy_refine", accuracy),
            _stage_record("score_arbitration", arbitration),
        ],
        "best_candidate": best,
        "best_score": _score(best),
        "decision": "review_policy_result",
        "next_command": "run-hermes-three-stage-policy",
    }
    _write_json(report_path, report)
    return report


def _stage_record(name: str, report: dict[str, Any]) -> dict[str, Any]:
    candidate = _best_candidate(report, name)
    return {
        "name": name,
        "report_path": report.get("heartbeat_report_path") or report.get("loop_report_path"),
        "best_candidate_id": candidate.get("candidate_id"),
        "best_submission_dir": candidate.get("submission_dir"),
        "best_score": candidate.get("score"),
        "stop_reason": report.get("stop_reason"),
        "decision": report.get("decision"),
    }


def _best_candidate(report: dict[str, Any], context: str) -> dict[str, Any]:
    value = report.get("best_candidate")
    if not isinstance(value, dict):
        raise HermesThreeStagePolicyError(f"{context} missing best_candidate")
    if not value.get("submission_dir"):
        raise HermesThreeStagePolicyError(f"{context} best_candidate missing submission_dir")
    return value


def _score(candidate: dict[str, Any]) -> float:
    value = candidate.get("score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HermesThreeStagePolicyError("best candidate missing numeric score")
    return float(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
