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
CANDIDATE_GENERATOR_REGISTRY = {
    "postprocess": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["global_score", "late_stability"],
    },
    "physics_residual_nudge": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["physical_residual", "late_stability"],
    },
    "sem_window_gate_blend_multi_early_smoke": {
        "cost_level": "medium",
        "auto_fallback_allowed": True,
        "requires_target": True,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["local_strategy_skill"],
    },
    "hypothesis_lab_front25_dynamics_smoke": {
        "cost_level": "medium",
        "auto_fallback_allowed": True,
        "factory_handoff_required": True,
        "requires_target": True,
        "requires_reference": True,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["front25_dynamics"],
    },
    "hypothesis_lab_early_subsegment_patch_search": {
        "cost_level": "medium",
        "auto_fallback_allowed": True,
        "factory_handoff_required": True,
        "requires_target": True,
        "requires_reference": True,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["residual_corrector", "spectral_physics"],
    },
    "sem_window_gate_blend": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["local_strategy_skill"],
    },
    "sem_window_blend": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment"],
        "saturation_blocked_by": ["local_strategy_skill"],
    },
    "fast_baseline_family": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["baseline_control", "runtime"],
    },
    "sem_window_gate_blend_multi_search": {
        "cost_level": "heavy",
        "auto_fallback_allowed": False,
        "requires_target": True,
        "requires_reference": False,
        "default_budget": "explicit",
        "targets_error_modes": ["early_segment", "transition_window"],
    },
}


class HermesHeartbeatError(RuntimeError):
    pass


def get_candidate_generator_registry() -> dict[str, dict[str, Any]]:
    return {
        name: dict(metadata)
        for name, metadata in CANDIDATE_GENERATOR_REGISTRY.items()
    }


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
    learning_objective: str = "score",
    learning_objective_schedule: list[str] | None = None,
    patience: int = 10,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    task2_test_path: str | Path | None = None,
    task2_target_path: str | Path | None = None,
    unet_checkpoint_path: str | Path | None = None,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    candidate_generator: str = "postprocess",
    candidate_generator_schedule: list[str] | None = None,
    doom_loop_report_path: str | Path | None = None,
    candidate_generator_fallback: str = "auto",
    candidate_generator_guard_overrides: list[str] | None = None,
    hypothesis_blueprint_path: str | Path | None = None,
    hypothesis_reference_path: str | Path | None = None,
    hypothesis_reference_sample_limit: int | None = None,
    hypothesis_reference_sample_limits: list[int] | None = None,
    hypothesis_candidate_options: dict[str, Any] | None = None,
    hypothesis_kpi_report_path: str | Path | None = None,
    repair_policy_summary_path: str | Path | None = None,
    saturation_probe_interval_batches: int = 0,
    postprocess_search_mode: str = "adaptive",
    postprocess_strategies: list[str] | None = None,
    postprocess_blend_weights: list[float] | None = None,
    postprocess_adaptive_step: float = 0.125,
    submission_id: str = "team_hermes",
    timing_profile: str | None = None,
    timing_factor: float | None = None,
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
    if saturation_probe_interval_batches < 0:
        raise HermesHeartbeatError(
            "saturation_probe_interval_batches must be non-negative"
        )
    _validate_schedule(
        name="learning_objective_schedule",
        values=learning_objective_schedule,
    )
    _validate_schedule(
        name="candidate_generator_schedule",
        values=candidate_generator_schedule,
    )
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
    best_objective_score: float | None = None
    stagnant_count = 0
    stop_reason = "max_batches_exhausted"
    guard = _load_doom_loop_guard(doom_loop_report_path)
    kpi = _load_hypothesis_kpi_report(hypothesis_kpi_report_path)
    repair_policy = _load_repair_policy_summary(repair_policy_summary_path)
    saturated_candidate_families = _saturated_candidate_families(kpi)
    persistent_search_guidance: list[dict[str, Any]] = _persistent_search_guidance(kpi)
    active_audit_repair_errors = _active_audit_repair_errors(kpi)
    repair_policy_recommended_actions = _repair_policy_recommended_actions(
        repair_policy
    )

    for batch_index in range(1, max_batches + 1):
        batch_root = root / f"batch_{batch_index:03d}"
        batch_input_memory_json = _materialize_batch_input_memory(
            root=root,
            batch_index=batch_index,
            current_memory_json=current_memory_json,
            persistent_search_guidance=persistent_search_guidance,
            repair_policy_recommended_actions=repair_policy_recommended_actions,
            repair_policy_summary_path=Path(repair_policy_summary_path)
            if repair_policy_summary_path is not None
            else None,
        )
        batch_learning_objective = _scheduled_value(
            fallback=learning_objective,
            schedule=learning_objective_schedule,
            batch_index=batch_index,
        )
        batch_candidate_generator = _scheduled_value(
            fallback=candidate_generator,
            schedule=candidate_generator_schedule,
            batch_index=batch_index,
        )
        original_candidate_generator = batch_candidate_generator
        batch_candidate_generator, search_guidance_guarded, search_guidance_policy = (
            _guard_candidate_generator_by_search_guidance(
                candidate_generator=batch_candidate_generator,
                persistent_search_guidance=persistent_search_guidance,
            )
        )
        batch_candidate_generator, guarded_candidate_generator, saturation_policy = (
            _guard_candidate_generator(
                candidate_generator=batch_candidate_generator,
                fallback=candidate_generator_fallback,
                guard=guard,
                kpi=kpi,
                saturated_candidate_families=saturated_candidate_families,
                candidate_generator_guard_overrides=(
                    candidate_generator_guard_overrides
                ),
                batch_index=batch_index,
                saturation_probe_interval_batches=saturation_probe_interval_batches,
            )
        )
        batch = run_batch(
            output_dir=batch_root,
            initial_submission_dir=current_submission_dir,
            max_rounds=max_rounds_per_batch,
            patience=patience,
            min_score_improvement=min_score_improvement,
            submit_score_threshold=submit_score_threshold,
            learning_objective=batch_learning_objective,
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
            candidate_generator=batch_candidate_generator,
            hypothesis_blueprint_path=Path(hypothesis_blueprint_path)
            if hypothesis_blueprint_path is not None
            else None,
            hypothesis_reference_path=Path(hypothesis_reference_path)
            if hypothesis_reference_path is not None
            else None,
            hypothesis_reference_sample_limit=hypothesis_reference_sample_limit,
            hypothesis_reference_sample_limits=hypothesis_reference_sample_limits,
            hypothesis_candidate_options=hypothesis_candidate_options,
            postprocess_search_mode=postprocess_search_mode,
            postprocess_strategies=postprocess_strategies or [],
            postprocess_blend_weights=postprocess_blend_weights or [],
            postprocess_adaptive_step=postprocess_adaptive_step,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            deadline_seconds=batch_deadline_seconds,
            checkpoint_every=checkpoint_every,
            memory_json_path=batch_input_memory_json,
            avoid_repeating=batch_input_memory_json is not None,
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
        objective_score = _objective_score(candidate)
        improved = (
            best_objective_score is None
            or objective_score - best_objective_score >= min_batch_score_improvement
        )
        if best_objective_score is None or objective_score > best_objective_score:
            best_objective_score = objective_score
            best_score = score
            best = candidate
        if improved:
            stagnant_count = 0
        else:
            stagnant_count += 1
        current_submission_dir = Path(str(candidate["submission_dir"]))
        current_memory_json = Path(memory["artifacts"]["memory_json_path"])
        persistent_search_guidance = _merge_search_guidance(
            persistent_search_guidance,
            _search_guidance_from_memory(current_memory_json),
        )
        batch_record = {
            "batch_index": batch_index,
            "batch_report_path": batch["loop_report_path"],
            "batch_input_memory_json_path": str(batch_input_memory_json)
            if batch_input_memory_json is not None
            else None,
            "memory_json_path": str(current_memory_json),
            "memory_markdown_path": memory["artifacts"]["memory_markdown_path"],
            "candidate_diagnostics": _batch_candidate_diagnostics(batch),
            "best_candidate_id": candidate.get("candidate_id"),
            "best_submission_dir": candidate.get("submission_dir"),
            "best_score": score,
            "best_objective": _objective_name(candidate),
            "best_objective_score": objective_score,
            "learning_objective": batch_learning_objective,
            "candidate_generator": batch_candidate_generator,
            "candidate_generator_metadata": _candidate_generator_metadata(
                batch_candidate_generator
            ),
            "original_candidate_generator": original_candidate_generator,
            "guarded_candidate_generator": (
                guarded_candidate_generator or search_guidance_guarded
            ),
            **search_guidance_policy,
            **saturation_policy,
            "repair_policy_recommended_actions": [
                dict(item) for item in repair_policy_recommended_actions
            ],
            "repair_policy_recommended_action_count": len(
                repair_policy_recommended_actions
            ),
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

    if best is None or best_score is None or best_objective_score is None:
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
            "submit_score_threshold": submit_score_threshold,
            "learning_objective": learning_objective,
            "learning_objective_schedule": learning_objective_schedule or [],
            "candidate_generator": candidate_generator,
            "candidate_generator_schedule": candidate_generator_schedule or [],
            "doom_loop_report_path": str(doom_loop_report_path)
            if doom_loop_report_path is not None
            else None,
            "candidate_generator_fallback": candidate_generator_fallback,
            "unet_checkpoint_path": str(unet_checkpoint_path)
            if unet_checkpoint_path is not None
            else None,
            "hypothesis_blueprint_path": str(hypothesis_blueprint_path)
            if hypothesis_blueprint_path is not None
            else None,
            "hypothesis_reference_path": str(hypothesis_reference_path)
            if hypothesis_reference_path is not None
            else None,
            "hypothesis_reference_sample_limit": hypothesis_reference_sample_limit,
            "hypothesis_reference_sample_limits": hypothesis_reference_sample_limits
            or [],
            "hypothesis_candidate_options": hypothesis_candidate_options or {},
            "hypothesis_kpi_report_path": str(hypothesis_kpi_report_path)
            if hypothesis_kpi_report_path is not None
            else None,
            "repair_policy_summary_path": str(repair_policy_summary_path)
            if repair_policy_summary_path is not None
            else None,
            "repair_policy_recommended_action_count": len(
                repair_policy_recommended_actions
            ),
            "saturated_candidate_families": saturated_candidate_families,
            "saturation_probe_interval_batches": saturation_probe_interval_batches,
            "candidate_generator_guard_overrides": (
                candidate_generator_guard_overrides or []
            ),
            "persistent_search_guidance_count": len(persistent_search_guidance),
            "active_audit_repair_error_count": len(active_audit_repair_errors),
            "timing_profile": timing_profile,
            "timing_factor": timing_factor,
        },
        "batch_count": len(batches),
        "batches": batches,
        "best_candidate": best,
        "best_score": best_score,
        "best_objective": _objective_name(best),
        "best_objective_score": best_objective_score,
        "persistent_search_guidance": persistent_search_guidance,
        "active_audit_repair_errors": active_audit_repair_errors,
        "repair_policy_recommended_actions": repair_policy_recommended_actions,
        "stop_reason": stop_reason,
        "decision": decision,
        "next_command": "run-hermes-heartbeat",
    }
    if guard is not None:
        report["guard"] = guard
    _write_json(report_path, report)
    return report


def _best_candidate(batch: dict[str, Any]) -> dict[str, Any]:
    candidate = batch.get("best_candidate")
    if not isinstance(candidate, dict):
        raise HermesHeartbeatError("batch report missing best_candidate")
    if not candidate.get("submission_dir"):
        raise HermesHeartbeatError("batch best_candidate missing submission_dir")
    return candidate


_DIAGNOSTIC_METADATA_KEYS = (
    "source",
    "candidate_report_path",
    "search_report_path",
    "search_evaluated_count",
    "memory_guidance_spec_count",
    "selected_sample_count",
    "validation_selection_metric",
    "validation_selection_score",
    "validation_segment_score",
    "feature_name",
    "threshold",
    "blend_weight",
    "window_start",
    "window_stop",
)


def _batch_candidate_diagnostics(batch: dict[str, Any]) -> dict[str, Any]:
    trajectory = batch.get("trajectory")
    if not isinstance(trajectory, list):
        return {
            "trajectory_count": 0,
            "attempted_candidate_count": 0,
            "improved_candidate_count": 0,
            "best_attempted_candidate_id": None,
            "best_attempted_score": None,
            "best_attempted_objective_score": None,
            "best_attempted_delta_vs_initial_objective": None,
            "best_attempted_metadata": {},
            "memory_constraints": _batch_memory_constraints(batch),
        }
    entries = [entry for entry in trajectory if isinstance(entry, dict)]
    attempted = entries[1:] if entries else []
    initial_objective = _optional_objective_score(entries[0]) if entries else None
    improved_count = sum(1 for entry in attempted if entry.get("improved") is True)
    scored_attempts = [
        entry for entry in attempted if _optional_objective_score(entry) is not None
    ]
    best_attempt = max(
        scored_attempts,
        key=lambda entry: _optional_objective_score(entry) or float("-inf"),
        default=None,
    )
    if best_attempt is None:
        best_objective = None
        best_score = None
        delta_vs_initial = None
        best_metadata: dict[str, Any] = {}
        best_candidate_id = None
    else:
        best_objective = _optional_objective_score(best_attempt)
        best_score = _optional_score(best_attempt)
        delta_vs_initial = (
            best_objective - initial_objective
            if best_objective is not None and initial_objective is not None
            else None
        )
        best_metadata = _diagnostic_metadata(best_attempt.get("metadata"))
        best_candidate_id = best_attempt.get("candidate_id")
    return {
        "trajectory_count": len(entries),
        "attempted_candidate_count": len(attempted),
        "improved_candidate_count": improved_count,
        "best_attempted_candidate_id": best_candidate_id,
        "best_attempted_score": best_score,
        "best_attempted_objective_score": best_objective,
        "best_attempted_delta_vs_initial_objective": delta_vs_initial,
        "best_attempted_metadata": best_metadata,
        "memory_constraints": _batch_memory_constraints(batch),
    }


def _batch_memory_constraints(batch: dict[str, Any]) -> dict[str, Any]:
    value = batch.get("memory_constraints")
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _optional_objective_score(candidate: dict[str, Any]) -> float | None:
    objective = candidate.get("objective")
    if isinstance(objective, dict):
        value = objective.get("objective_score")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return _optional_score(candidate)


def _optional_score(candidate: dict[str, Any]) -> float | None:
    value = candidate.get("score")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _diagnostic_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata[key]
        for key in _DIAGNOSTIC_METADATA_KEYS
        if key in metadata
    }


def _materialize_batch_input_memory(
    *,
    root: Path,
    batch_index: int,
    current_memory_json: Path | None,
    persistent_search_guidance: list[dict[str, Any]],
    repair_policy_recommended_actions: list[dict[str, Any]],
    repair_policy_summary_path: Path | None,
) -> Path | None:
    if current_memory_json is None and not repair_policy_recommended_actions:
        return None
    if current_memory_json is not None and not persistent_search_guidance and not repair_policy_recommended_actions:
        return current_memory_json
    payload = (
        _read_memory_payload(current_memory_json)
        if current_memory_json is not None
        else _seed_memory_payload()
    )
    merged_guidance = _merge_search_guidance(
        _search_guidance_from_payload(payload),
        persistent_search_guidance,
    )
    merged_actions = _merge_repair_policy_actions(
        _repair_policy_actions_from_payload(payload),
        repair_policy_recommended_actions,
    )
    changed = (
        merged_guidance != _search_guidance_from_payload(payload)
        or merged_actions != _repair_policy_actions_from_payload(payload)
        or current_memory_json is None
    )
    if not changed:
        return current_memory_json
    payload["search_guidance"] = merged_guidance
    payload["repair_policy_recommended_actions"] = merged_actions
    payload["repair_policy_recommended_action_count"] = len(merged_actions)
    if repair_policy_summary_path is not None:
        input_sources = _memory_input_sources(payload)
        input_sources["repair_policy_summary_path"] = str(repair_policy_summary_path)
        payload["input_sources"] = input_sources
    output_path = root / "memory" / f"batch_{batch_index:03d}_input_memory.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _read_memory_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesHeartbeatError(f"invalid heartbeat memory JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HermesHeartbeatError("heartbeat memory JSON must be an object")
    return payload


def _seed_memory_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": "HERMES_HEARTBEAT_INPUT_MEMORY_SEEDED",
        "search_guidance": [],
        "repair_policy_recommended_actions": [],
        "repair_policy_recommended_action_count": 0,
        "input_sources": {},
    }


def _search_guidance_from_memory(path: Path) -> list[dict[str, Any]]:
    return _search_guidance_from_payload(_read_memory_payload(path))


def _search_guidance_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    guidance = payload.get("search_guidance")
    if not isinstance(guidance, list):
        return []
    return [dict(item) for item in guidance if isinstance(item, dict)]


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


def _merge_repair_policy_actions(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*existing, *incoming]:
        try:
            key = json.dumps(item, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
    return merged


def _merge_search_guidance(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*existing, *incoming]:
        try:
            key = json.dumps(item, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
    return merged


def _guard_candidate_generator_by_search_guidance(
    *,
    candidate_generator: str,
    persistent_search_guidance: list[dict[str, Any]],
) -> tuple[str, bool, dict[str, Any]]:
    policy = {
        "search_guidance_guarded_candidate_generator": False,
        "search_guidance_guard_reason": "",
    }
    if not _sem_window_generator(candidate_generator):
        return candidate_generator, False, policy
    if not _has_guided_sem_window_exhaustion(persistent_search_guidance):
        return candidate_generator, False, policy
    policy.update(
        {
            "search_guidance_guarded_candidate_generator": True,
            "search_guidance_guard_reason": "guided_sem_window_search_exhausted",
        }
    )
    return "physics_residual_nudge", True, policy


def _sem_window_generator(candidate_generator: str) -> bool:
    return candidate_generator in {
        "sem_window_gate_blend_multi_search",
        "sem_window_gate_blend_multi_early_search",
        "sem_window_gate_blend_multi_early_smoke",
        "sem_window_gate_blend",
        "sem_window_blend",
    }


def _has_guided_sem_window_exhaustion(
    search_guidance: list[dict[str, Any]],
) -> bool:
    for item in search_guidance:
        if item.get("guidance_type") != "escalate_after_guided_search_exhaustion":
            continue
        if item.get("reason") != "memory_guided_specs_failed_to_improve":
            continue
        if item.get("recommended_action") != (
            "switch_gate_feature_family_or_try_new_dynamics_source"
        ):
            continue
        return True
    return False


def _score(candidate: dict[str, Any]) -> float:
    value = candidate.get("score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HermesHeartbeatError("best candidate missing numeric score")
    return float(value)


def _objective_score(candidate: dict[str, Any]) -> float:
    objective = candidate.get("objective")
    if isinstance(objective, dict):
        value = objective.get("objective_score")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return _score(candidate)


def _objective_name(candidate: dict[str, Any]) -> str:
    objective = candidate.get("objective")
    if isinstance(objective, dict):
        value = objective.get("selected_objective")
        if isinstance(value, str) and value:
            return value
    return "score"


def _decision(stop_reason: str) -> str:
    if stop_reason == "submit_threshold_met":
        return "submit_best_candidate"
    if stop_reason == "stagnation_detected":
        return "review_before_next_heartbeat"
    return "continue_next_heartbeat"


def _validate_schedule(*, name: str, values: list[str] | None) -> None:
    if values is None:
        return
    if not values:
        raise HermesHeartbeatError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise HermesHeartbeatError(f"{name} entries must be non-empty strings")


def _scheduled_value(
    *,
    fallback: str,
    schedule: list[str] | None,
    batch_index: int,
) -> str:
    if not schedule:
        return fallback
    return schedule[(batch_index - 1) % len(schedule)]


def _load_doom_loop_guard(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    guard_path = Path(path)
    if not guard_path.exists():
        raise HermesHeartbeatError(f"doom_loop_report_path does not exist: {guard_path}")
    try:
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesHeartbeatError(f"invalid doom loop guard JSON: {guard_path}") from exc
    if not isinstance(guard, dict):
        raise HermesHeartbeatError("doom loop guard must be a JSON object")
    if guard.get("state_hint") != "HYPOTHESIS_DOOM_LOOP_CHECKED":
        raise HermesHeartbeatError("doom loop guard has invalid state_hint")
    blocked = guard.get("blocked_candidate_families")
    if blocked is not None and not isinstance(blocked, list):
        raise HermesHeartbeatError("blocked_candidate_families must be a list")
    return {
        "state_hint": guard.get("state_hint"),
        "guard_active": bool(guard.get("guard_active")),
        "blocked_candidate_families": [
            item for item in blocked or [] if isinstance(item, str) and item
        ],
        "corrective_prompt": guard.get("corrective_prompt", ""),
    }


def _load_hypothesis_kpi_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    kpi_path = Path(path)
    if not kpi_path.exists():
        raise HermesHeartbeatError(
            f"hypothesis_kpi_report_path does not exist: {kpi_path}"
        )
    try:
        kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesHeartbeatError(f"invalid hypothesis KPI JSON: {kpi_path}") from exc
    if not isinstance(kpi, dict):
        raise HermesHeartbeatError("hypothesis KPI report must be a JSON object")
    if kpi.get("state_hint") != "HYPOTHESIS_KPI_REPORT_BUILT":
        raise HermesHeartbeatError("hypothesis KPI report has invalid state_hint")
    return kpi


def _load_repair_policy_summary(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    policy_path = Path(path)
    if not policy_path.exists():
        raise HermesHeartbeatError(
            f"repair_policy_summary_path does not exist: {policy_path}"
        )
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesHeartbeatError(
            f"invalid repair policy summary JSON: {policy_path}"
        ) from exc
    if not isinstance(policy, dict):
        raise HermesHeartbeatError("repair policy summary must be a JSON object")
    if policy.get("state_hint") != "HERMES_REPAIR_POLICY_SUMMARY_BUILT":
        raise HermesHeartbeatError("repair policy summary has invalid state_hint")
    return policy


def _guard_candidate_generator(
    *,
    candidate_generator: str,
    fallback: str,
    guard: dict[str, Any] | None,
    kpi: dict[str, Any] | None = None,
    saturated_candidate_families: list[str] | None = None,
    candidate_generator_guard_overrides: list[str] | None = None,
    batch_index: int = 1,
    saturation_probe_interval_batches: int = 0,
) -> tuple[str, bool, dict[str, Any]]:
    blocked = set(saturated_candidate_families or [])
    guard_overrides = {
        item
        for item in candidate_generator_guard_overrides or []
        if isinstance(item, str) and item
    }
    doom_blocked: set[str] = set()
    if guard is not None and guard.get("guard_active"):
        doom_blocked = set(guard.get("blocked_candidate_families", []))
        blocked.update(doom_blocked)
    blocked_by_saturation = _generator_blocked_by_saturation(
        candidate_generator,
        saturated_candidate_families=saturated_candidate_families or [],
    )
    saturation_policy = {
        "saturation_probe": False,
        "saturation_probe_reason": "",
        "directive_guard_override": False,
    }
    if (
        candidate_generator in guard_overrides
        and candidate_generator in doom_blocked
        and not blocked_by_saturation
    ):
        saturation_policy.update(
            {
                "directive_guard_override": True,
                "directive_guard_override_reason": "evidence_bounded_reprobe",
            }
        )
        return candidate_generator, False, saturation_policy
    if candidate_generator not in blocked and not blocked_by_saturation:
        return candidate_generator, False, saturation_policy
    if (
        blocked_by_saturation
        and candidate_generator not in doom_blocked
        and saturation_probe_interval_batches > 0
        and batch_index % saturation_probe_interval_batches == 0
    ):
        saturation_policy.update(
            {
                "saturation_probe": True,
                "saturation_probe_reason": (
                    "interval_reprobe_of_saturated_candidate_family"
                ),
            }
        )
        return candidate_generator, False, saturation_policy
    if not isinstance(fallback, str) or not fallback.strip():
        raise HermesHeartbeatError("candidate_generator_fallback must be non-empty")
    if fallback == "auto":
        return _auto_fallback_candidate_generator(
            candidate_generator=candidate_generator,
            blocked=blocked,
            kpi=kpi,
            saturated_candidate_families=saturated_candidate_families or [],
        ), True, saturation_policy
    if fallback in blocked:
        raise HermesHeartbeatError(
            "candidate_generator_fallback is blocked by doom loop guard"
        )
    return fallback, True, saturation_policy


def _saturated_candidate_families(kpi: dict[str, Any] | None) -> list[str]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("saturated_candidate_families")
    if not isinstance(values, list):
        return []
    return sorted({item for item in values if isinstance(item, str) and item})


def _persistent_search_guidance(kpi: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("persistent_search_guidance")
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, dict)]


def _repair_policy_recommended_actions(
    repair_policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(repair_policy, dict):
        return []
    values = repair_policy.get("recommended_actions")
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, dict)]


def _active_audit_repair_errors(kpi: dict[str, Any] | None) -> list[str]:
    if not isinstance(kpi, dict):
        return []
    values = kpi.get("active_audit_repair_errors")
    if not isinstance(values, list):
        return []
    return sorted({item for item in values if isinstance(item, str) and item})


def _generator_blocked_by_saturation(
    candidate_generator: str,
    *,
    saturated_candidate_families: list[str],
) -> bool:
    if not saturated_candidate_families:
        return False
    aliases = CANDIDATE_GENERATOR_REGISTRY.get(candidate_generator, {}).get(
        "saturation_blocked_by",
    )
    if not isinstance(aliases, list):
        return False
    saturated = set(saturated_candidate_families)
    return any(isinstance(alias, str) and alias in saturated for alias in aliases)


def _auto_fallback_candidate_generator(
    *,
    candidate_generator: str,
    blocked: set[str],
    kpi: dict[str, Any] | None,
    saturated_candidate_families: list[str],
) -> str:
    kpi_family = _best_positive_kpi_family(kpi)
    if (
        kpi_family is not None
        and _auto_fallback_allowed(kpi_family)
        and kpi_family != candidate_generator
        and kpi_family not in blocked
        and not _generator_blocked_by_saturation(
            kpi_family,
            saturated_candidate_families=saturated_candidate_families,
        )
    ):
        return kpi_family
    for candidate in _auto_fallback_generators():
        if candidate == candidate_generator or candidate in blocked:
            continue
        if _generator_blocked_by_saturation(
            candidate,
            saturated_candidate_families=saturated_candidate_families,
        ):
            continue
        return candidate
    raise HermesHeartbeatError(
        "no unblocked auto candidate_generator_fallback is available"
    )


def _auto_fallback_generators() -> list[str]:
    return [
        name
        for name, metadata in CANDIDATE_GENERATOR_REGISTRY.items()
        if metadata.get("auto_fallback_allowed") is True
        and metadata.get("factory_handoff_required") is not True
    ]


def _auto_fallback_allowed(candidate_generator: str) -> bool:
    metadata = CANDIDATE_GENERATOR_REGISTRY.get(candidate_generator, {})
    return (
        metadata.get("auto_fallback_allowed") is True
        and metadata.get("factory_handoff_required") is not True
    )


def _candidate_generator_metadata(candidate_generator: str) -> dict[str, Any]:
    return dict(CANDIDATE_GENERATOR_REGISTRY.get(candidate_generator, {}))


def _best_positive_kpi_family(kpi: dict[str, Any] | None) -> str | None:
    if not isinstance(kpi, dict):
        return None
    best = kpi.get("best_family_by_score_delta")
    if not isinstance(best, dict):
        return None
    family = best.get("candidate_family")
    best_delta = best.get("best_score_delta")
    mean_delta = best.get("mean_score_delta")
    positive_best = (
        isinstance(best_delta, int | float)
        and not isinstance(best_delta, bool)
        and float(best_delta) > 0.0
    )
    positive_mean = (
        isinstance(mean_delta, int | float)
        and not isinstance(mean_delta, bool)
        and float(mean_delta) > 0.0
    )
    if isinstance(family, str) and family and (positive_best or positive_mean):
        return family
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
