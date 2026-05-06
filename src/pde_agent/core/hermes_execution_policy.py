"""Adaptive Hermes execution policy driven by heartbeat trajectory reports."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat


DIAGNOSIS_STATE_HINT = "HERMES_EXECUTION_TRAJECTORY_DIAGNOSED"
POLICY_STATE_HINT = "HERMES_ADAPTIVE_HEARTBEAT_RAN"
HeartbeatRunner = Callable[..., dict[str, Any]]


class HermesExecutionPolicyError(RuntimeError):
    pass


def diagnose_heartbeat_report(
    *,
    heartbeat_report_path: str | Path,
    verified_blueprint_path: str | Path | None,
    output_path: str | Path | None,
    force: bool,
) -> dict[str, Any]:
    report_file = Path(heartbeat_report_path)
    report = _read_json(report_file)
    if report.get("state_hint") != "HERMES_HEARTBEAT_RAN":
        raise HermesExecutionPolicyError("heartbeat report has invalid state_hint")
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)

    trajectory_state, reason_codes = _trajectory_state(report)
    blueprint_file = Path(verified_blueprint_path) if verified_blueprint_path else None
    has_blueprint = _has_verified_linear_trend_blueprint(blueprint_file)
    already_using_blueprint = (
        _candidate_generator(report) == "linear_trend_spectral"
        and _path_equal(_report_blueprint_path(report), blueprint_file)
    )
    hypothesis_lab_search_policy = _hypothesis_lab_search_policy(report)
    if trajectory_state == "plateau" and has_blueprint and already_using_blueprint:
        recommended_action = "verified_blueprint_plateau"
        next_command = "run-hypothesis-lab"
        recommended_candidate_generator = "linear_trend_spectral"
    elif trajectory_state == "plateau" and has_blueprint:
        recommended_action = "switch_to_verified_blueprint"
        next_command = "run-hermes-heartbeat"
        recommended_candidate_generator = "linear_trend_spectral"
    elif (
        trajectory_state == "plateau"
        and _candidate_generator(report) == "hypothesis_lab"
        and hypothesis_lab_search_policy is not None
    ):
        recommended_action = "continue_hypothesis_lab_with_search_policy"
        next_command = "run-hypothesis-lab"
        recommended_candidate_generator = "hypothesis_lab"
    elif trajectory_state == "plateau":
        recommended_action = "run_hypothesis_lab"
        next_command = "run-hypothesis-lab"
        recommended_candidate_generator = None
    else:
        recommended_action = "continue_current_generator"
        next_command = "run-hermes-heartbeat"
        recommended_candidate_generator = _candidate_generator(report)

    diagnosis = {
        "schema_version": 1,
        "state_hint": DIAGNOSIS_STATE_HINT,
        "heartbeat_report_path": str(report_file),
        "trajectory_state": trajectory_state,
        "recommended_action": recommended_action,
        "recommended_candidate_generator": recommended_candidate_generator,
        "verified_blueprint_path": str(blueprint_file) if has_blueprint else None,
        "best_score": report.get("best_score"),
        "stop_reason": report.get("stop_reason"),
        "reason_codes": reason_codes,
        "evidence": {
            "batch_count": report.get("batch_count"),
            "candidate_generator": _candidate_generator(report),
            "last_batch": _last_batch(report),
        },
        "next_command": next_command,
    }
    if hypothesis_lab_search_policy is not None:
        diagnosis["hypothesis_lab_search_policy"] = hypothesis_lab_search_policy
    if output_file is not None:
        diagnosis["diagnosis_path"] = str(output_file)
        _write_json(output_file, diagnosis)
    return diagnosis


def run_hermes_adaptive_heartbeat(
    *,
    output_dir: str | Path,
    initial_submission_dir: str | Path,
    verified_blueprint_path: str | Path | None,
    max_policy_stages: int = 2,
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
    initial_candidate_generator: str = "postprocess",
    candidate_generator_schedule: list[str] | None = None,
    hypothesis_reference_path: str | Path | None = None,
    hypothesis_reference_sample_limit: int | None = None,
    hypothesis_reference_sample_limits: list[int] | None = None,
    postprocess_search_mode: str = "adaptive",
    postprocess_strategies: list[str] | None = None,
    postprocess_blend_weights: list[float] | None = None,
    postprocess_adaptive_step: float = 0.125,
    submission_id: str = "team_hermes",
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    heartbeat_runner: HeartbeatRunner | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if max_policy_stages < 1:
        raise HermesExecutionPolicyError("max_policy_stages must be positive")
    root = Path(output_dir)
    report_path = root / "hermes_adaptive_heartbeat_report.json"
    _guard_output(report_path, force=force)
    root.mkdir(parents=True, exist_ok=True)
    run_heartbeat = heartbeat_runner or run_hermes_heartbeat

    current_submission_dir = Path(initial_submission_dir)
    current_generator = initial_candidate_generator
    current_blueprint: Path | None = None
    verified_blueprint = Path(verified_blueprint_path) if verified_blueprint_path else None
    policy_events: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for stage_index in range(1, max_policy_stages + 1):
        heartbeat = run_heartbeat(
            output_dir=root / f"stage_{stage_index:03d}_{current_generator}",
            initial_submission_dir=current_submission_dir,
            max_batches=max_batches,
            batch_deadline_seconds=batch_deadline_seconds,
            max_rounds_per_batch=max_rounds_per_batch,
            checkpoint_every=checkpoint_every,
            max_stagnant_batches=max_stagnant_batches,
            min_batch_score_improvement=min_batch_score_improvement,
            min_score_improvement=min_score_improvement,
            submit_score_threshold=submit_score_threshold,
            learning_objective=learning_objective,
            learning_objective_schedule=learning_objective_schedule,
            patience=patience,
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
            candidate_generator=current_generator,
            candidate_generator_schedule=candidate_generator_schedule,
            hypothesis_blueprint_path=current_blueprint,
            hypothesis_reference_path=Path(hypothesis_reference_path)
            if hypothesis_reference_path is not None
            else None,
            hypothesis_reference_sample_limit=hypothesis_reference_sample_limit,
            hypothesis_reference_sample_limits=hypothesis_reference_sample_limits,
            postprocess_search_mode=postprocess_search_mode,
            postprocess_strategies=postprocess_strategies or [],
            postprocess_blend_weights=postprocess_blend_weights or [],
            postprocess_adaptive_step=postprocess_adaptive_step,
            submission_id=submission_id,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=True,
        )
        stage_reports.append(_stage_record(stage_index, heartbeat))
        candidate = _best_candidate(heartbeat)
        if best is None or _score(candidate) > _score(best):
            best = candidate
        diagnosis = diagnose_heartbeat_report(
            heartbeat_report_path=heartbeat["heartbeat_report_path"],
            verified_blueprint_path=verified_blueprint,
            output_path=root / "diagnoses" / f"stage_{stage_index:03d}.json",
            force=True,
        )
        event = {
            "stage_index": stage_index,
            "action": diagnosis["recommended_action"],
            "trajectory_state": diagnosis["trajectory_state"],
            "reason_codes": diagnosis["reason_codes"],
            "source_heartbeat_report_path": heartbeat["heartbeat_report_path"],
            "best_candidate_id": candidate.get("candidate_id"),
            "best_score": candidate.get("score"),
        }
        if isinstance(diagnosis.get("hypothesis_lab_search_policy"), dict):
            event["hypothesis_lab_search_policy"] = diagnosis[
                "hypothesis_lab_search_policy"
            ]
        policy_events.append(event)
        if diagnosis["recommended_action"] != "switch_to_verified_blueprint":
            break
        if current_generator == diagnosis["recommended_candidate_generator"]:
            break
        current_generator = str(diagnosis["recommended_candidate_generator"])
        current_blueprint = Path(str(diagnosis["verified_blueprint_path"]))
        current_submission_dir = Path(str(candidate["submission_dir"]))

    if best is None:
        raise HermesExecutionPolicyError("adaptive heartbeat produced no best candidate")
    _append_policy_log(submission_dir=Path(str(best["submission_dir"])), events=policy_events)
    next_command = _next_command(policy_events)
    next_hypothesis_lab_inputs = _next_hypothesis_lab_inputs(
        adaptive_report_path=report_path,
        best_submission_dir=Path(str(best["submission_dir"])),
        policy_events=policy_events,
    )
    report = {
        "schema_version": 1,
        "state_hint": POLICY_STATE_HINT,
        "adaptive_report_path": str(report_path),
        "stage_count": len(stage_reports),
        "stages": stage_reports,
        "policy_events": policy_events,
        "best_candidate": best,
        "best_score": _score(best),
        "parameters": {
            "hypothesis_reference_path": str(hypothesis_reference_path)
            if hypothesis_reference_path is not None
            else None,
            "hypothesis_reference_sample_limit": hypothesis_reference_sample_limit,
            "hypothesis_reference_sample_limits": hypothesis_reference_sample_limits
            or [],
            "timing_profile": timing_profile,
            "timing_factor": timing_factor,
            "learning_objective_schedule": learning_objective_schedule or [],
            "candidate_generator_schedule": candidate_generator_schedule or [],
        },
        "decision": "continue_next_adaptive_heartbeat",
        "next_command": next_command,
    }
    if next_hypothesis_lab_inputs is not None:
        report["next_hypothesis_lab_inputs"] = next_hypothesis_lab_inputs
    _write_json(report_path, report)
    return report


def _trajectory_state(report: dict[str, Any]) -> tuple[str, list[str]]:
    reason_codes = []
    stop_reason = report.get("stop_reason")
    if stop_reason == "stagnation_detected":
        reason_codes.append("stagnation_detected")
    if stop_reason == "candidate_runner_exhausted":
        reason_codes.append("candidate_runner_exhausted")
    batches = report.get("batches")
    if isinstance(batches, list) and batches:
        recent = [item for item in batches[-2:] if isinstance(item, dict)]
        if recent and not any(bool(item.get("score_improved")) for item in recent):
            reason_codes.append("recent_batches_not_improved")
    if reason_codes:
        return "plateau", reason_codes
    return "improving_or_unblocked", ["no_plateau_detected"]


def _has_verified_linear_trend_blueprint(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    payload = _read_json(path)
    if payload.get("state_hint") != "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED":
        return False
    blueprints = payload.get("blueprints")
    return isinstance(blueprints, list) and any(
        isinstance(item, dict)
        and item.get("candidate_family") == "linear_trend_spectral"
        for item in blueprints
    )


def _candidate_generator(report: dict[str, Any]) -> str | None:
    parameters = report.get("parameters")
    if isinstance(parameters, dict):
        value = parameters.get("candidate_generator")
        if isinstance(value, str):
            return value
    return None


def _hypothesis_lab_search_policy(report: dict[str, Any]) -> dict[str, Any] | None:
    for path in _hypothesis_lab_artifact_paths(report):
        payload = _read_json_if_available(path)
        if payload is None:
            continue
        policy = _search_policy_from_hypothesis_lab_payload(payload)
        if policy is not None:
            return {"source_path": str(path), "policy": policy}
    return None


def _hypothesis_lab_artifact_paths(report: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    candidates = [report.get("best_candidate")]
    batch = _read_last_batch_report(report)
    if isinstance(batch, dict):
        candidates.append(batch.get("best_candidate"))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for key in (
            "candidate_report_path",
            "hypothesis_lab_report_path",
            "sandbox_report_path",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                paths.append(Path(value))
    return _dedupe_paths(paths)


def _read_last_batch_report(report: dict[str, Any]) -> dict[str, Any] | None:
    last_batch = _last_batch(report)
    if not isinstance(last_batch, dict):
        return None
    value = last_batch.get("batch_report_path")
    if not isinstance(value, str) or not value:
        return None
    return _read_json_if_available(Path(value))


def _search_policy_from_hypothesis_lab_payload(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    sandbox = payload.get("sandbox") if isinstance(payload.get("sandbox"), dict) else payload
    attribution = sandbox.get("candidate_pool_attribution")
    if not isinstance(attribution, dict):
        return None
    policy = attribution.get("search_space_policy")
    return policy if isinstance(policy, dict) else None


def _read_json_if_available(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except HermesExecutionPolicyError:
        return None


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _report_blueprint_path(report: dict[str, Any]) -> Path | None:
    parameters = report.get("parameters")
    if isinstance(parameters, dict):
        value = parameters.get("hypothesis_blueprint_path")
        if isinstance(value, str) and value:
            return Path(value)
    return None


def _path_equal(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    return str(left) == str(right)


def _last_batch(report: dict[str, Any]) -> dict[str, Any] | None:
    batches = report.get("batches")
    if isinstance(batches, list) and batches and isinstance(batches[-1], dict):
        return batches[-1]
    return None


def _best_candidate(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report.get("best_candidate")
    if not isinstance(candidate, dict):
        raise HermesExecutionPolicyError("heartbeat report missing best_candidate")
    if not candidate.get("submission_dir"):
        raise HermesExecutionPolicyError("best_candidate missing submission_dir")
    return candidate


def _score(candidate: dict[str, Any]) -> float:
    value = candidate.get("score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HermesExecutionPolicyError("candidate missing numeric score")
    return float(value)


def _stage_record(stage_index: int, heartbeat: dict[str, Any]) -> dict[str, Any]:
    candidate = _best_candidate(heartbeat)
    return {
        "stage_index": stage_index,
        "heartbeat_report_path": heartbeat.get("heartbeat_report_path"),
        "candidate_generator": _candidate_generator(heartbeat),
        "stop_reason": heartbeat.get("stop_reason"),
        "decision": heartbeat.get("decision"),
        "best_candidate_id": candidate.get("candidate_id"),
        "best_submission_dir": candidate.get("submission_dir"),
        "best_score": candidate.get("score"),
    }


def _append_policy_log(*, submission_dir: Path, events: list[dict[str, Any]]) -> None:
    log_path = submission_dir / "task1_logs.log"
    if not log_path.exists():
        return
    text = log_path.read_text(encoding="utf-8")
    if "Hermes Adaptive Execution Policy" in text:
        return
    lines = [
        "",
        "Hermes Adaptive Execution Policy",
        "Trace:",
        "- Hermes diagnosed heartbeat trajectory reports before selecting the next generator.",
        "Experiment Tracking:",
    ]
    for event in events:
        lines.append(
            "- stage={stage} action={action} state={state} best={best} score={score}".format(
                stage=event["stage_index"],
                action=event["action"],
                state=event["trajectory_state"],
                best=event.get("best_candidate_id"),
                score=event.get("best_score"),
            )
        )
    log_path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _next_command(events: list[dict[str, Any]]) -> str:
    if events and events[-1].get("action") in {
        "verified_blueprint_plateau",
        "continue_hypothesis_lab_with_search_policy",
    }:
        return "run-hypothesis-lab"
    return "run-hermes-adaptive-heartbeat"


def _next_hypothesis_lab_inputs(
    *,
    adaptive_report_path: Path,
    best_submission_dir: Path,
    policy_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    log_path = best_submission_dir / "task1_logs.log"
    if not log_path.exists():
        return None
    inputs: dict[str, Any] = {
        "manual_report_paths": [],
        "adaptive_report_path": str(adaptive_report_path),
        "log_paths": [str(log_path)],
    }
    search_policy = _last_hypothesis_lab_search_policy(policy_events)
    if search_policy is not None:
        inputs["search_space_policy"] = search_policy
    return inputs


def _last_hypothesis_lab_search_policy(
    policy_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for event in reversed(policy_events):
        value = event.get("hypothesis_lab_search_policy")
        if isinstance(value, dict):
            return value
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HermesExecutionPolicyError(f"JSON artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesExecutionPolicyError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise HermesExecutionPolicyError("JSON artifact must be an object")
    return value


def _guard_output(path: Path | None, *, force: bool) -> None:
    if path is not None and path.exists() and not force:
        raise HermesExecutionPolicyError(f"artifact already exists: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
