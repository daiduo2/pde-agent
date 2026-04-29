"""Heartbeat runner for guarded long-horizon Hermes optimization batches."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_batch_optimization_loop import (
    run_hermes_batch_optimization_loop,
)
from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory


STATE_HINT = "HERMES_HEARTBEAT_RAN"
BatchRunner = Callable[..., dict[str, Any]]
MemoryBuilder = Callable[..., dict[str, Any]]


class HermesHeartbeatError(RuntimeError):
    pass


def run_hermes_heartbeat(
    *,
    output_dir: str | Path,
    initial_submission_dir: str | Path,
    max_batches: int = 1,
    batch_deadline_seconds: float = 16800.0,
    max_rounds_per_batch: int = 50,
    checkpoint_every: int = 5,
    max_stagnant_batches: int = 1,
    min_batch_score_improvement: float = 1.0,
    min_score_improvement: float = 1.0,
    submit_score_threshold: float = 120.0,
    patience: int = 10,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    task2_test_path: str | Path | None = None,
    task2_target_path: str | Path | None = None,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    candidate_generator: str = "postprocess",
    postprocess_search_mode: str = "adaptive",
    postprocess_strategies: list[str] | None = None,
    postprocess_blend_weights: list[float] | None = None,
    postprocess_adaptive_step: float = 0.125,
    submission_id: str = "team_hermes",
    batch_runner: BatchRunner | None = None,
    memory_builder: MemoryBuilder | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if max_batches < 1:
        raise HermesHeartbeatError("max_batches must be positive")
    if batch_deadline_seconds <= 0.0:
        raise HermesHeartbeatError("batch_deadline_seconds must be positive")
    if max_rounds_per_batch < 1:
        raise HermesHeartbeatError("max_rounds_per_batch must be positive")
    if max_stagnant_batches < 0:
        raise HermesHeartbeatError("max_stagnant_batches must be non-negative")
    root = Path(output_dir)
    report_path = root / "hermes_heartbeat_report.json"
    if report_path.exists() and not force:
        raise HermesHeartbeatError(f"heartbeat report already exists: {report_path}")
    root.mkdir(parents=True, exist_ok=True)

    run_batch = batch_runner or run_hermes_batch_optimization_loop
    run_memory = memory_builder or build_hermes_iteration_memory
    current_submission_dir = Path(initial_submission_dir)
    current_memory_json: Path | None = None
    batches: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score: float | None = None
    stagnant_count = 0
    stop_reason = "max_batches_exhausted"

    for batch_index in range(1, max_batches + 1):
        batch_root = root / f"batch_{batch_index:03d}"
        batch = run_batch(
            output_dir=batch_root,
            initial_submission_dir=current_submission_dir,
            max_rounds=max_rounds_per_batch,
            patience=patience,
            min_score_improvement=min_score_improvement,
            submit_score_threshold=submit_score_threshold,
            task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
            task1_target_path=Path(task1_target_path)
            if task1_target_path is not None
            else None,
            task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
            task2_target_path=Path(task2_target_path)
            if task2_target_path is not None
            else None,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
            candidate_generator=candidate_generator,
            postprocess_search_mode=postprocess_search_mode,
            postprocess_strategies=postprocess_strategies or [],
            postprocess_blend_weights=postprocess_blend_weights or [],
            postprocess_adaptive_step=postprocess_adaptive_step,
            submission_id=submission_id,
            deadline_seconds=batch_deadline_seconds,
            checkpoint_every=checkpoint_every,
            memory_json_path=current_memory_json,
            avoid_repeating=current_memory_json is not None,
            force=True,
        )
        candidate = _best_candidate(batch)
        memory = run_memory(
            batch_loop_report_path=Path(batch["loop_report_path"]),
            output_path=root / "memory" / f"batch_{batch_index:03d}_memory.md",
            memory_json_path=root / "memory" / f"batch_{batch_index:03d}_memory.json",
            submission_dir=candidate.get("submission_dir"),
            task_ids=None,
            force=True,
        )
        score = _score(candidate)
        improved = best_score is None or score - best_score >= min_batch_score_improvement
        if best_score is None or score > best_score:
            best_score = score
            best = candidate
        if improved:
            stagnant_count = 0
        else:
            stagnant_count += 1
        current_submission_dir = Path(str(candidate["submission_dir"]))
        current_memory_json = Path(memory["artifacts"]["memory_json_path"])
        batch_record = {
            "batch_index": batch_index,
            "batch_report_path": batch["loop_report_path"],
            "memory_json_path": str(current_memory_json),
            "memory_markdown_path": memory["artifacts"]["memory_markdown_path"],
            "best_candidate_id": candidate.get("candidate_id"),
            "best_submission_dir": candidate.get("submission_dir"),
            "best_score": score,
            "score_improved": improved,
            "batch_stop_reason": batch.get("stop_reason"),
            "batch_decision": batch.get("decision"),
        }
        batches.append(batch_record)
        if batch.get("decision") == "submit_best_candidate":
            stop_reason = "submit_threshold_met"
            break
        if stagnant_count >= max_stagnant_batches:
            stop_reason = "stagnation_detected"
            break

    if best is None or best_score is None:
        raise HermesHeartbeatError("heartbeat produced no best candidate")
    decision = _decision(stop_reason)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "heartbeat_report_path": str(report_path),
        "parameters": {
            "max_batches": max_batches,
            "batch_deadline_seconds": batch_deadline_seconds,
            "max_rounds_per_batch": max_rounds_per_batch,
            "checkpoint_every": checkpoint_every,
            "max_stagnant_batches": max_stagnant_batches,
        },
        "batch_count": len(batches),
        "batches": batches,
        "best_candidate": best,
        "best_score": best_score,
        "stop_reason": stop_reason,
        "decision": decision,
        "next_command": "run-hermes-heartbeat",
    }
    _write_json(report_path, report)
    return report


def _best_candidate(batch: dict[str, Any]) -> dict[str, Any]:
    candidate = batch.get("best_candidate")
    if not isinstance(candidate, dict):
        raise HermesHeartbeatError("batch report missing best_candidate")
    if not candidate.get("submission_dir"):
        raise HermesHeartbeatError("batch best_candidate missing submission_dir")
    return candidate


def _score(candidate: dict[str, Any]) -> float:
    value = candidate.get("score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HermesHeartbeatError("best candidate missing numeric score")
    return float(value)


def _decision(stop_reason: str) -> str:
    if stop_reason == "submit_threshold_met":
        return "submit_best_candidate"
    if stop_reason == "stagnation_detected":
        return "review_before_next_heartbeat"
    return "continue_next_heartbeat"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
