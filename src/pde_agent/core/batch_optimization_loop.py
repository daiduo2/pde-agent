"""Batch optimization controller that uses the local submission judge as objective."""


import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pde_agent.core.batch_optimization_candidates import (
    _build_candidate_runner,
)
from pde_agent.core.batch_optimization_common import BatchOptimizationLoopError
from pde_agent.core.batch_optimization_objective import (
    candidate_signals as _candidate_signals,
    decision as _decision,
    meets_submit_threshold as _meets_submit_threshold,
    objective_score as _objective_score,
    objective_signals as _objective_signals,
    score as _score,
    with_objective as _with_objective,
)
from pde_agent.core.batch_optimization_memory import (
    avoid_set_from_payload as _avoid_set_from_payload,
    load_memory_json as _load_memory_json,
    memory_constraints_summary as _memory_constraints_summary,
    search_guidance_from_payload as _search_guidance_from_payload,
)
from pde_agent.core.iteration_memory import build_iteration_memory
from pde_agent.core.submission_judge import judge_submission


STATE_HINT = "BATCH_OPTIMIZATION_LOOP_RAN"
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


def run_batch_optimization_loop(
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
    submission_id: str = "team_default",
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
        raise BatchOptimizationLoopError("max_rounds must be positive")
    if patience < 0:
        raise BatchOptimizationLoopError("patience must be non-negative")
    if min_score_improvement < 0.0:
        raise BatchOptimizationLoopError("min_score_improvement must be non-negative")
    if learning_objective not in SUPPORTED_LEARNING_OBJECTIVES:
        raise BatchOptimizationLoopError(
            f"unsupported learning_objective: {learning_objective}"
        )
    if deadline_seconds is not None and deadline_seconds <= 0.0:
        raise BatchOptimizationLoopError("deadline_seconds must be positive")
    if checkpoint_every < 0:
        raise BatchOptimizationLoopError("checkpoint_every must be non-negative")
    root = Path(output_dir)
    report_path = root / "batch_optimization_loop.json"
    if report_path.exists() and not force:
        raise BatchOptimizationLoopError(
            f"batch optimization report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)

    run_judge = judge or judge_submission
    run_memory_builder = memory_builder or build_iteration_memory
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
        raise BatchOptimizationLoopError("batch optimization produced no score")
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


def _candidate_submission_dir(candidate: dict[str, Any]) -> Path:
    value = candidate.get("submission_dir")
    if not isinstance(value, str) or not value:
        raise BatchOptimizationLoopError("candidate missing submission_dir")
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
