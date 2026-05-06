"""Repeated audited Hermes runtime closed-loop chain."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_runtime_closed_loop import run_hermes_runtime_closed_loop
from pde_agent.core.hermes_runtime_closed_loop_audit import (
    audit_hermes_runtime_closed_loop,
)
from pde_agent.core.hermes_runtime_memory_bridge_smoke import (
    run_hermes_runtime_memory_bridge_smoke,
)
from pde_agent.core.hermes_hypothesis_lab_infrastructure_audit import (
    STATE_HINT as INFRASTRUCTURE_AUDIT_STATE_HINT,
)
from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore


STATE_HINT = "HERMES_RUNTIME_AUDIT_CHAIN_RAN"
REPAIR_TASKS_STATE_HINT = "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED"
REPAIR_TASK_RESOLUTION_STATE_HINT = "HERMES_AUDIT_REPAIR_TASK_RESOLVED"
REPAIR_TASK_RUN_STATE_HINT = "HERMES_AUDIT_REPAIR_TASK_RAN"
ACCEPTANCE_SMOKE_STATE_HINT = "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN"
RuntimeRunner = Callable[..., dict[str, Any]]
AuditRunner = Callable[..., dict[str, Any]]


class HermesRuntimeAuditChainError(RuntimeError):
    pass


def resolve_audit_repair_task(
    *,
    audit_repair_tasks_path: str | Path,
    output_path: str | Path,
    task_id: str | None = None,
    provided_inputs: dict[str, str] | None = None,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise HermesRuntimeAuditChainError(
            f"repair task resolution exists: {output_file}"
        )
    payload = _read_repair_tasks_payload(Path(audit_repair_tasks_path))
    task = _select_repair_task(payload, task_id=task_id)
    next_command = str(task.get("next_command", ""))
    suggested_args = task.get("suggested_command_args")
    if not isinstance(suggested_args, dict):
        suggested_args = {}
    else:
        suggested_args = dict(suggested_args)
    provided = provided_inputs or {}
    for key, value in provided.items():
        if key:
            suggested_args[key] = value
    missing_inputs = [
        item
        for item in task.get("missing_inputs", [])
        if isinstance(item, str) and item
        and not suggested_args.get(item)
    ]
    argv = _repair_task_argv(
        next_command=next_command,
        suggested_args=suggested_args,
    )
    command_string = shlex.join(argv)
    report = {
        "schema_version": 1,
        "state_hint": REPAIR_TASK_RESOLUTION_STATE_HINT,
        "audit_repair_tasks_path": str(audit_repair_tasks_path),
        "task_id": task.get("task_id"),
        "audit_error": task.get("audit_error"),
        "next_command": next_command,
        "argv": argv,
        "command_string": command_string,
        "missing_inputs": missing_inputs,
        "provided_inputs": provided,
        "ready_to_run": not missing_inputs and bool(next_command),
        "next_hint": next_command if not missing_inputs else "fill-repair-task-missing-inputs",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report["repair_task_resolution_path"] = str(output_file)
    return report


def run_resolved_audit_repair_task(
    *,
    audit_repair_tasks_path: str | Path,
    output_dir: str | Path,
    task_id: str | None = None,
    provided_inputs: dict[str, str] | None = None,
    runtime_runner: RuntimeRunner | None = None,
    audit_runner: AuditRunner | None = None,
    force: bool,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_path = root / "audit_repair_task_run_report.json"
    if report_path.exists() and not force:
        raise HermesRuntimeAuditChainError(
            f"audit repair task run report exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    resolution = resolve_audit_repair_task(
        audit_repair_tasks_path=audit_repair_tasks_path,
        output_path=root / "repair_task_resolution.json",
        task_id=task_id,
        provided_inputs=provided_inputs,
        force=True,
    )
    next_command = str(resolution.get("next_command") or "")
    if resolution.get("ready_to_run") is not True:
        report = _repair_task_run_report(
            report_path=report_path,
            decision="blocked_missing_inputs",
            resolution=resolution,
            runtime=None,
            audit=None,
            smoke=None,
            next_command="fill-repair-task-missing-inputs",
        )
        return report
    args = _resolved_repair_task_args(
        audit_repair_tasks_path=Path(audit_repair_tasks_path),
        task_id=task_id,
        provided_inputs=provided_inputs or {},
    )
    if next_command == "run-hermes-runtime-memory-bridge-smoke":
        smoke = run_hermes_runtime_memory_bridge_smoke(
            **_runtime_memory_bridge_smoke_kwargs_from_repair_args(args=args)
        )
        return _repair_task_run_report(
            report_path=report_path,
            decision=str(smoke.get("decision") or "unknown"),
            resolution=resolution,
            runtime=None,
            audit=None,
            smoke=smoke,
            next_command=str(
                smoke.get("next_command") or "use-hermes-runtime-closed-loop"
            ),
        )
    if next_command != "run-hermes-runtime-closed-loop":
        report = _repair_task_run_report(
            report_path=report_path,
            decision="unsupported_next_command",
            resolution=resolution,
            runtime=None,
            audit=None,
            smoke=None,
            next_command=next_command or "inspect-hermes-runtime-audit-chain",
        )
        return report

    run_runtime = runtime_runner or run_hermes_runtime_closed_loop
    run_audit = audit_runner or audit_hermes_runtime_closed_loop
    runtime = run_runtime(
        **_runtime_closed_loop_kwargs_from_repair_args(
            args=args,
            root=root,
            audit_repair_tasks_path=Path(audit_repair_tasks_path),
        )
    )
    audit_command = runtime.get("audit_command")
    if not isinstance(audit_command, dict):
        raise HermesRuntimeAuditChainError("runtime repair report missing audit_command")
    audit = run_audit(
        runtime_report_path=Path(str(audit_command["runtime_report_path"])),
        output_path=Path(str(audit_command["output_path"])),
        require_guard_effect=bool(audit_command.get("require_guard_effect", False)),
        require_learning_constraint_smoke=bool(
            audit_command.get("require_learning_constraint_smoke", False)
        ),
        require_runtime_memory_bridge_smoke=bool(
            audit_command.get("require_runtime_memory_bridge_smoke", False)
        ),
        force=True,
    )
    return _repair_task_run_report(
        report_path=report_path,
        decision=str(audit.get("decision") or "unknown"),
        resolution=resolution,
        runtime=runtime,
        audit=audit,
        smoke=None,
        next_command=str(audit.get("next_command") or "run-hermes-runtime-audit-chain"),
    )


def run_hermes_runtime_audit_chain(
    *,
    output_dir: str | Path,
    trace_path: str | Path,
    initial_submission_dir: str | Path,
    candidate_generator: str,
    candidate_generator_fallback: str,
    iterations: int,
    max_batches: int,
    recent_window: int,
    repeated_family_threshold: int,
    failed_family_threshold: int,
    min_score_delta: float,
    max_rounds_per_batch: int,
    batch_deadline_seconds: float,
    submit_score_threshold: float,
    learning_objective: str,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    task1_reference_path: str | Path | None,
    task2_test_path: str | Path | None,
    task2_target_path: str | Path | None,
    timing_profile: str | None,
    timing_factor: float | None,
    hypothesis_candidate_options: dict[str, Any] | None = None,
    research_directive_path: str | Path | None = None,
    audit_repair_tasks_path: str | Path | None = None,
    preflight_acceptance_smoke_path: str | Path | None = None,
    preflight_infrastructure_audit_path: str | Path | None = None,
    auto_repair: bool = False,
    runtime_runner: RuntimeRunner | None = None,
    audit_runner: AuditRunner | None = None,
    force: bool,
) -> dict[str, Any]:
    if iterations < 1:
        raise HermesRuntimeAuditChainError("iterations must be positive")
    root = Path(output_dir)
    report_path = root / "hermes_runtime_audit_chain_report.json"
    if report_path.exists() and not force:
        raise HermesRuntimeAuditChainError(f"audit chain report exists: {report_path}")
    root.mkdir(parents=True, exist_ok=True)
    run_runtime = runtime_runner or run_hermes_runtime_closed_loop
    run_audit = audit_runner or audit_hermes_runtime_closed_loop
    preflight_acceptance = _load_preflight_acceptance_smoke(
        Path(preflight_acceptance_smoke_path)
        if preflight_acceptance_smoke_path is not None
        else None
    )
    preflight_infrastructure = _load_preflight_infrastructure_audit(
        Path(preflight_infrastructure_audit_path)
        if preflight_infrastructure_audit_path is not None
        else None
    )
    learning_smoke_path = _discover_learning_constraint_smoke_report(root)
    memory_bridge_smoke_path = _discover_runtime_memory_bridge_smoke_report(root)
    current_submission_dir = Path(initial_submission_dir)
    iteration_reports: list[dict[str, Any]] = []
    failed_audit_count = 0

    for index in range(1, iterations + 1):
        iteration_dir = root / f"iter_{index:03d}"
        runtime = run_runtime(
            output_dir=iteration_dir,
            trace_path=Path(trace_path),
            initial_submission_dir=current_submission_dir,
            candidate_generator=candidate_generator,
            candidate_generator_fallback=candidate_generator_fallback,
            max_batches=max_batches,
            recent_window=recent_window,
            repeated_family_threshold=repeated_family_threshold,
            failed_family_threshold=failed_family_threshold,
            min_score_delta=min_score_delta,
            max_rounds_per_batch=max_rounds_per_batch,
            batch_deadline_seconds=batch_deadline_seconds,
            submit_score_threshold=submit_score_threshold,
            learning_objective=learning_objective,
            task1_test_path=Path(task1_test_path)
            if task1_test_path is not None
            else None,
            task1_target_path=Path(task1_target_path)
            if task1_target_path is not None
            else None,
            task1_reference_path=Path(task1_reference_path)
            if task1_reference_path is not None
            else None,
            task2_test_path=Path(task2_test_path)
            if task2_test_path is not None
            else None,
            task2_target_path=Path(task2_target_path)
            if task2_target_path is not None
            else None,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            hypothesis_candidate_options=hypothesis_candidate_options,
            research_directive_path=Path(research_directive_path)
            if research_directive_path is not None
            else None,
            audit_repair_tasks_path=Path(audit_repair_tasks_path)
            if audit_repair_tasks_path is not None
            else None,
            force=force,
        )
        audit_command = runtime.get("audit_command")
        if not isinstance(audit_command, dict):
            raise HermesRuntimeAuditChainError("runtime report missing audit_command")
        if learning_smoke_path is None:
            learning_smoke_path = _discover_learning_constraint_smoke_report(root)
        if memory_bridge_smoke_path is None:
            memory_bridge_smoke_path = _discover_runtime_memory_bridge_smoke_report(
                root
            )
        if learning_smoke_path is not None:
            _attach_learning_constraint_smoke(
                runtime=runtime,
                smoke_path=learning_smoke_path,
            )
            audit_command = runtime["audit_command"]
        if memory_bridge_smoke_path is not None:
            _attach_runtime_memory_bridge_smoke(
                runtime=runtime,
                smoke_path=memory_bridge_smoke_path,
            )
            audit_command = runtime["audit_command"]
        audit = run_audit(
            runtime_report_path=Path(str(audit_command["runtime_report_path"])),
            output_path=Path(str(audit_command["output_path"])),
            require_guard_effect=bool(audit_command.get("require_guard_effect", False)),
            require_learning_constraint_smoke=bool(
                audit_command.get("require_learning_constraint_smoke", False)
            ),
            require_runtime_memory_bridge_smoke=bool(
                audit_command.get("require_runtime_memory_bridge_smoke", False)
            ),
            force=True,
        )
        audit_decision = str(audit.get("decision"))
        best_submission_dir = _best_submission_dir(runtime)
        iteration_reports.append(
            {
                "iteration_index": index,
                "iteration_dir": str(iteration_dir),
                "initial_submission_dir": str(current_submission_dir),
                "runtime_report_path": _runtime_report_path(runtime),
                "audit_report_path": audit.get("audit_report_path"),
                "audit_decision": audit_decision,
                "audit_failed_check_count": _failed_check_count(audit),
                "audit_errors": _audit_errors(audit),
                "heartbeat_best_score": _heartbeat_best_score(runtime),
                "candidate_space_exhausted": _candidate_space_exhausted(runtime),
                "best_submission_dir": str(best_submission_dir)
                if best_submission_dir is not None
                else None,
            }
        )
        if audit_decision != "passed":
            failed_audit_count += 1
            break
        if best_submission_dir is not None:
            current_submission_dir = best_submission_dir

    decision = "passed" if failed_audit_count == 0 else "failed"
    score_delta = _chain_score_delta(iteration_reports)
    candidate_space_exhausted_count = sum(
        1 for item in iteration_reports if item.get("candidate_space_exhausted") is True
    )
    chain_learning_signals = _chain_learning_signals(
        failed_audit_count=failed_audit_count,
        candidate_space_exhausted_count=candidate_space_exhausted_count,
        score_delta=score_delta,
        completed_iteration_count=len(iteration_reports),
    )
    audit_errors = _chain_audit_errors(iteration_reports)
    repair_convergence = _repair_convergence(
        audit_repair_tasks_path=audit_repair_tasks_path,
        audit_errors=audit_errors,
    )
    if repair_convergence.get("decision") == "nonconverged":
        _append_unique(chain_learning_signals, "audit_repair_nonconvergence")
    artifacts: dict[str, Any] = {}
    if preflight_acceptance is not None:
        artifacts["preflight_acceptance_smoke_path"] = str(
            preflight_acceptance["path"]
        )
    if preflight_infrastructure is not None:
        artifacts["preflight_infrastructure_audit_path"] = str(
            preflight_infrastructure["path"]
        )
    repair_tasks = _audit_repair_tasks(
        audit_errors,
        root=root,
        trace_path=Path(trace_path),
        initial_submission_dir=current_submission_dir,
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        audit_repair_tasks_path=Path(audit_repair_tasks_path)
        if audit_repair_tasks_path is not None
        else None,
        candidate_generator=candidate_generator,
    )
    auto_repair_report: dict[str, Any] = {}
    if repair_tasks:
        repair_tasks_path = root / "audit_repair_tasks.json"
        _write_audit_repair_tasks(
            path=repair_tasks_path,
            chain_report_path=report_path,
            repair_tasks=repair_tasks,
        )
        artifacts["audit_repair_tasks_path"] = str(repair_tasks_path)
        if auto_repair:
            auto_repair_report = run_resolved_audit_repair_task(
                audit_repair_tasks_path=repair_tasks_path,
                output_dir=root / "auto_repair",
                task_id=None,
                provided_inputs=_auto_repair_inputs(
                    next_command=_first_repair_next_command(repair_tasks),
                    root=root,
                    audit_repair_tasks_path=repair_tasks_path,
                    trace_path=Path(trace_path),
                    initial_submission_dir=_auto_repair_initial_submission_dir(
                        iteration_reports=iteration_reports,
                        fallback=current_submission_dir,
                    ),
                    candidate_generator=candidate_generator,
                    candidate_generator_fallback=candidate_generator_fallback,
                    max_batches=max_batches,
                    recent_window=recent_window,
                    repeated_family_threshold=repeated_family_threshold,
                    failed_family_threshold=failed_family_threshold,
                    min_score_delta=min_score_delta,
                    max_rounds_per_batch=max_rounds_per_batch,
                    batch_deadline_seconds=batch_deadline_seconds,
                    submit_score_threshold=submit_score_threshold,
                    learning_objective=learning_objective,
                    task1_test_path=task1_test_path,
                    task1_target_path=task1_target_path,
                    task1_reference_path=task1_reference_path,
                    task2_test_path=task2_test_path,
                    task2_target_path=task2_target_path,
                    timing_profile=timing_profile,
                    timing_factor=timing_factor,
                ),
                runtime_runner=run_runtime,
                audit_runner=run_audit,
                force=True,
            )
            auto_repair_artifacts = auto_repair_report.get("artifacts")
            if isinstance(auto_repair_artifacts, dict):
                auto_repair_path = auto_repair_artifacts.get(
                    "repair_task_run_report_path"
                )
                if isinstance(auto_repair_path, str):
                    artifacts["auto_repair_report_path"] = auto_repair_path
            if auto_repair_report.get("decision") == "passed":
                decision = "repaired"
                _append_unique(chain_learning_signals, "audit_repair_auto_repaired")
            auto_repair_convergence = _repair_convergence_from_tasks(
                repair_tasks=repair_tasks,
                audit_errors=_auto_repair_audit_errors(auto_repair_report),
            )
            if auto_repair_convergence.get("decision") in {
                "converged",
                "nonconverged",
            }:
                repair_convergence = auto_repair_convergence
            if auto_repair_convergence.get("decision") == "converged":
                _append_unique(chain_learning_signals, "audit_repair_convergence")
            if auto_repair_convergence.get("decision") == "nonconverged":
                _append_unique(chain_learning_signals, "audit_repair_nonconvergence")
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "chain_report_path": str(report_path),
        "decision": decision,
        "iterations": iteration_reports,
        "artifacts": artifacts,
        "metrics": {
            "requested_iteration_count": iterations,
            "completed_iteration_count": len(iteration_reports),
            "failed_audit_count": failed_audit_count,
            "audit_repair_task_count": len(repair_tasks),
            "repair_nonconverged_error_count": len(
                repair_convergence.get("repeated_audit_errors", [])
            ),
            "auto_repair_attempted": bool(auto_repair_report),
            "auto_repair_decision": auto_repair_report.get("decision"),
            "preflight_acceptance_smoke_passed": preflight_acceptance is not None,
            "preflight_infrastructure_audit_passed": (
                preflight_infrastructure is not None
            ),
            "preflight_infrastructure_check_count": (
                preflight_infrastructure["check_count"]
                if preflight_infrastructure is not None
                else 0
            ),
            **_auto_repair_metric_summary(auto_repair_report),
            "candidate_space_exhausted_count": candidate_space_exhausted_count,
            "score_delta": score_delta,
            "last_heartbeat_best_score": iteration_reports[-1][
                "heartbeat_best_score"
            ]
            if iteration_reports
            else None,
        },
        "audit_errors": audit_errors,
        "repair_convergence": repair_convergence,
        "auto_repair": auto_repair_report,
        "chain_learning_signals": chain_learning_signals,
        "next_command": _next_command(
            decision=decision,
            repair_tasks=repair_tasks,
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    trace_payload = {
        "algorithm_candidate_id": "hermes_runtime_audit_chain",
        "candidate_family": "hermes_runtime_audit_chain",
        "outcome": decision,
        "requested_iteration_count": iterations,
        "completed_iteration_count": len(iteration_reports),
        "failed_audit_count": failed_audit_count,
        "last_heartbeat_best_score": report["metrics"]["last_heartbeat_best_score"],
        "score_delta": score_delta,
        "candidate_space_exhausted_count": candidate_space_exhausted_count,
        "audit_errors": audit_errors,
        "repair_convergence": repair_convergence,
        "audit_repair_task_count": len(repair_tasks),
        "chain_learning_signals": chain_learning_signals,
        "chain_report_path": str(report_path),
    }
    if preflight_acceptance is not None:
        trace_payload["preflight_acceptance_smoke_path"] = str(
            preflight_acceptance["path"]
        )
        trace_payload["preflight_acceptance_check_count"] = preflight_acceptance[
            "check_count"
        ]
    if preflight_infrastructure is not None:
        trace_payload["preflight_infrastructure_audit_path"] = str(
            preflight_infrastructure["path"]
        )
        trace_payload["preflight_infrastructure_check_count"] = (
            preflight_infrastructure["check_count"]
        )
        if preflight_infrastructure.get("session_id"):
            trace_payload["preflight_infrastructure_session_id"] = (
                preflight_infrastructure["session_id"]
            )
    if auto_repair_report:
        trace_payload["auto_repair_decision"] = auto_repair_report.get("decision")
        auto_metrics = _trace_auto_repair_metrics(auto_repair_report)
        if auto_metrics:
            trace_payload["auto_repair_metrics"] = auto_metrics
        auto_artifacts = _trace_auto_repair_artifacts(auto_repair_report)
        if auto_artifacts:
            trace_payload["auto_repair_artifacts"] = auto_artifacts
    if "audit_repair_tasks_path" in artifacts:
        trace_payload["audit_repair_tasks_path"] = artifacts["audit_repair_tasks_path"]
    trace_event = HypothesisLabTraceStore(trace_path).append_event(
        event_type="hermes_runtime_audit_chain",
        payload=trace_payload,
        run_id="local",
        source_artifact_path=report_path,
        tags=["hermes", "runtime_audit_chain"],
    )
    report["trace_event_id"] = trace_event["event_id"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _discover_learning_constraint_smoke_report(root: Path) -> Path | None:
    path = root / "learning_constraint_smoke" / "hypothesis_learning_constraint_smoke.json"
    return path if path.is_file() else None


def _discover_runtime_memory_bridge_smoke_report(root: Path) -> Path | None:
    path = (
        root
        / "runtime_memory_bridge_smoke"
        / "hermes_runtime_memory_bridge_smoke.json"
    )
    return path if path.is_file() else None


def _load_preflight_acceptance_smoke(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise HermesRuntimeAuditChainError(
            f"preflight acceptance smoke not found: {path}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeAuditChainError(
            f"preflight acceptance smoke is not valid JSON: {path}"
        ) from exc
    if report.get("state_hint") != ACCEPTANCE_SMOKE_STATE_HINT:
        raise HermesRuntimeAuditChainError(
            "preflight acceptance smoke has invalid state_hint"
        )
    checks = report.get("checks")
    failed_checks = [
        name
        for name, passed in checks.items()
        if isinstance(name, str) and passed is not True
    ] if isinstance(checks, dict) else []
    if report.get("decision") != "passed" or not isinstance(checks, dict) or failed_checks:
        detail = ",".join(sorted(failed_checks)) if failed_checks else "decision"
        raise HermesRuntimeAuditChainError(
            f"preflight acceptance smoke failed: {detail}"
        )
    return {
        "path": path,
        "check_count": len(checks),
    }


def _load_preflight_infrastructure_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise HermesRuntimeAuditChainError(
            f"preflight infrastructure audit not found: {path}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeAuditChainError(
            f"preflight infrastructure audit is not valid JSON: {path}"
        ) from exc
    if not isinstance(report, dict):
        raise HermesRuntimeAuditChainError(
            "preflight infrastructure audit must be a JSON object"
        )
    loaded = _passed_infrastructure_fixture_report(report, path=path)
    if loaded is not None:
        return loaded
    loaded = _passed_infrastructure_command_report(report, path=path)
    if loaded is not None:
        return loaded
    raise HermesRuntimeAuditChainError(
        "preflight infrastructure audit failed: decision"
    )


def _passed_infrastructure_fixture_report(
    report: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    infrastructure = report.get("infrastructure")
    if not isinstance(infrastructure, dict):
        return None
    if report.get("status") != "passed":
        return None
    if infrastructure.get("decision") != "passed":
        return None
    failed_count = infrastructure.get("failed_check_count")
    check_count = infrastructure.get("check_count")
    if failed_count != 0 or not isinstance(check_count, int) or check_count <= 0:
        return None
    return {
        "path": path,
        "check_count": check_count,
        "session_id": report.get("session_id"),
        "source_type": "hermes_fixture_session_audit",
    }


def _passed_infrastructure_command_report(
    report: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    if report.get("state_hint") != INFRASTRUCTURE_AUDIT_STATE_HINT:
        return None
    metrics = report.get("metrics")
    summary = report.get("summary")
    decision = report.get("decision")
    if isinstance(metrics, dict) and decision is None:
        decision = metrics.get("decision")
    failed_count = None
    check_count = None
    for source in (metrics, summary):
        if not isinstance(source, dict):
            continue
        if failed_count is None and isinstance(source.get("failed_check_count"), int):
            failed_count = source["failed_check_count"]
        if check_count is None and isinstance(source.get("check_count"), int):
            check_count = source["check_count"]
    if decision != "passed" or failed_count != 0:
        return None
    if not isinstance(check_count, int) or check_count <= 0:
        return None
    return {
        "path": path,
        "check_count": check_count,
        "session_id": None,
        "source_type": "infrastructure_command_report",
    }


def _attach_learning_constraint_smoke(
    *,
    runtime: dict[str, Any],
    smoke_path: Path,
) -> None:
    artifacts = runtime.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise HermesRuntimeAuditChainError("runtime artifacts must be an object")
    artifacts["learning_constraint_smoke_report_path"] = str(smoke_path)
    audit_command = runtime.setdefault("audit_command", {})
    if not isinstance(audit_command, dict):
        raise HermesRuntimeAuditChainError("runtime audit_command must be an object")
    audit_command["require_learning_constraint_smoke"] = True
    runtime_path = artifacts.get("runtime_loop_report_path")
    if isinstance(runtime_path, str) and runtime_path:
        Path(runtime_path).write_text(
            json.dumps(runtime, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _attach_runtime_memory_bridge_smoke(
    *,
    runtime: dict[str, Any],
    smoke_path: Path,
) -> None:
    artifacts = runtime.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise HermesRuntimeAuditChainError("runtime artifacts must be an object")
    artifacts["runtime_memory_bridge_smoke_report_path"] = str(smoke_path)
    audit_command = runtime.setdefault("audit_command", {})
    if not isinstance(audit_command, dict):
        raise HermesRuntimeAuditChainError("runtime audit_command must be an object")
    audit_command["require_runtime_memory_bridge_smoke"] = True
    runtime_path = artifacts.get("runtime_loop_report_path")
    if isinstance(runtime_path, str) and runtime_path:
        Path(runtime_path).write_text(
            json.dumps(runtime, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _runtime_report_path(runtime: dict[str, Any]) -> str | None:
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get("runtime_loop_report_path")
    return str(value) if value is not None else None


def _heartbeat_best_score(runtime: dict[str, Any]) -> float | None:
    metrics = runtime.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("heartbeat_best_score")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _candidate_space_exhausted(runtime: dict[str, Any]) -> bool:
    metrics = runtime.get("metrics")
    return (
        isinstance(metrics, dict)
        and metrics.get("candidate_space_exhausted") is True
    )


def _chain_score_delta(iterations: list[dict[str, Any]]) -> float | None:
    scores = [
        item.get("heartbeat_best_score")
        for item in iterations
        if isinstance(item.get("heartbeat_best_score"), int | float)
        and not isinstance(item.get("heartbeat_best_score"), bool)
    ]
    if len(scores) < 2:
        return None
    return float(scores[-1]) - float(scores[0])


def _chain_learning_signals(
    *,
    failed_audit_count: int,
    candidate_space_exhausted_count: int,
    score_delta: float | None,
    completed_iteration_count: int,
) -> list[str]:
    signals: list[str] = []
    if failed_audit_count:
        signals.append("audit_gate_failure")
    if candidate_space_exhausted_count:
        signals.append("candidate_space_exhaustion")
    if (
        completed_iteration_count > 1
        and isinstance(score_delta, float)
        and score_delta <= 0.0
    ):
        signals.append("score_stagnation")
    return signals


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _failed_check_count(audit: dict[str, Any]) -> int | None:
    summary = audit.get("summary")
    if not isinstance(summary, dict):
        return None
    value = summary.get("failed_check_count")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _audit_errors(audit: dict[str, Any]) -> list[str]:
    errors = audit.get("errors")
    if not isinstance(errors, list):
        return []
    return [item for item in errors if isinstance(item, str) and item]


def _chain_audit_errors(iterations: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for iteration in iterations:
        for item in iteration.get("audit_errors", []):
            if isinstance(item, str) and item and item not in errors:
                errors.append(item)
    return errors


def _audit_repair_tasks(
    audit_errors: list[str],
    *,
    root: Path,
    trace_path: Path,
    initial_submission_dir: Path | None = None,
    task1_test_path: Path | None = None,
    task1_target_path: Path | None = None,
    audit_repair_tasks_path: Path | None = None,
    candidate_generator: str | None = None,
) -> list[dict[str, Any]]:
    return [
        _audit_repair_task(
            error,
            root=root,
            trace_path=trace_path,
            initial_submission_dir=initial_submission_dir,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            audit_repair_tasks_path=audit_repair_tasks_path,
            candidate_generator=candidate_generator,
        )
        for error in audit_errors
    ]


def _next_command(
    *,
    decision: str,
    repair_tasks: list[dict[str, Any]],
) -> str:
    if decision in {"passed", "repaired"}:
        return "run-hermes-runtime-audit-chain"
    if repair_tasks:
        next_command = repair_tasks[0].get("next_command")
        if isinstance(next_command, str) and next_command:
            return next_command
    return "inspect-hermes-runtime-audit-chain"


def _first_repair_next_command(repair_tasks: list[dict[str, Any]]) -> str:
    if repair_tasks:
        next_command = repair_tasks[0].get("next_command")
        if isinstance(next_command, str):
            return next_command
    return ""


def _auto_repair_initial_submission_dir(
    *,
    iteration_reports: list[dict[str, Any]],
    fallback: Path,
) -> Path:
    if not iteration_reports:
        return fallback
    best = iteration_reports[-1].get("best_submission_dir")
    if isinstance(best, str) and best:
        return Path(best)
    initial = iteration_reports[-1].get("initial_submission_dir")
    if isinstance(initial, str) and initial:
        return Path(initial)
    return fallback


def _auto_repair_inputs(
    *,
    next_command: str,
    root: Path,
    audit_repair_tasks_path: Path,
    trace_path: Path,
    initial_submission_dir: Path,
    candidate_generator: str,
    candidate_generator_fallback: str,
    max_batches: int,
    recent_window: int,
    repeated_family_threshold: int,
    failed_family_threshold: int,
    min_score_delta: float,
    max_rounds_per_batch: int,
    batch_deadline_seconds: float,
    submit_score_threshold: float,
    learning_objective: str,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    task1_reference_path: str | Path | None,
    task2_test_path: str | Path | None,
    task2_target_path: str | Path | None,
    timing_profile: str | None,
    timing_factor: float | None,
) -> dict[str, str]:
    if next_command == "run-hermes-runtime-memory-bridge-smoke":
        return {"audit_repair_tasks_path": str(audit_repair_tasks_path)}
    values: dict[str, str] = {
        "output_dir": str(root / "auto_repair" / "runtime_closed_loop"),
        "trace_path": str(trace_path),
        "initial_submission_dir": str(initial_submission_dir),
        "candidate_generator": candidate_generator,
        "candidate_generator_fallback": candidate_generator_fallback,
        "max_batches": str(max_batches),
        "recent_window": str(recent_window),
        "repeated_family_threshold": str(repeated_family_threshold),
        "failed_family_threshold": str(failed_family_threshold),
        "min_score_delta": str(min_score_delta),
        "max_rounds_per_batch": str(max_rounds_per_batch),
        "batch_deadline_seconds": str(batch_deadline_seconds),
        "submit_score_threshold": str(submit_score_threshold),
        "learning_objective": learning_objective,
        "force": "true",
    }
    for key, value in {
        "task1_test_path": task1_test_path,
        "task1_target_path": task1_target_path,
        "task1_reference_path": task1_reference_path,
        "task2_test_path": task2_test_path,
        "task2_target_path": task2_target_path,
        "timing_profile": timing_profile,
        "timing_factor": timing_factor,
    }.items():
        if value is not None:
            values[key] = str(value)
    return values


def _repair_convergence(
    *,
    audit_repair_tasks_path: str | Path | None,
    audit_errors: list[str],
) -> dict[str, Any]:
    input_repair_errors = _input_repair_errors(audit_repair_tasks_path)
    repeated = sorted(set(input_repair_errors).intersection(audit_errors))
    if repeated:
        decision = "nonconverged"
    elif input_repair_errors:
        decision = "converged"
    else:
        decision = "not_applicable"
    return {
        "decision": decision,
        "input_repair_errors": input_repair_errors,
        "repeated_audit_errors": repeated,
    }


def _repair_convergence_from_tasks(
    *,
    repair_tasks: list[dict[str, Any]],
    audit_errors: list[str],
) -> dict[str, Any]:
    input_repair_errors = sorted(
        {
            str(task["audit_error"])
            for task in repair_tasks
            if isinstance(task, dict)
            and isinstance(task.get("audit_error"), str)
            and task.get("audit_error")
        }
    )
    repeated = sorted(set(input_repair_errors).intersection(audit_errors))
    if repeated:
        decision = "nonconverged"
    elif input_repair_errors:
        decision = "converged"
    else:
        decision = "not_applicable"
    return {
        "decision": decision,
        "input_repair_errors": input_repair_errors,
        "repeated_audit_errors": repeated,
    }


def _auto_repair_audit_errors(auto_repair_report: dict[str, Any]) -> list[str]:
    audit = auto_repair_report.get("audit")
    if not isinstance(audit, dict):
        return []
    return _audit_errors(audit)


def _auto_repair_metric_summary(auto_repair_report: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        auto_repair_report.get("metrics")
        if isinstance(auto_repair_report.get("metrics"), dict)
        else {}
    )
    summary: dict[str, Any] = {}
    for key in [
        "smoke_executed",
        "memory_bridge_passed",
        "repair_policy_recommended_action_count",
        "batch_memory_constraint_action_count",
    ]:
        if key in metrics:
            summary[f"auto_repair_{key}"] = metrics[key]
    return summary


def _trace_auto_repair_metrics(auto_repair_report: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        auto_repair_report.get("metrics")
        if isinstance(auto_repair_report.get("metrics"), dict)
        else {}
    )
    return {
        key: metrics[key]
        for key in [
            "smoke_executed",
            "memory_bridge_passed",
            "repair_policy_recommended_action_count",
            "batch_memory_constraint_action_count",
        ]
        if key in metrics
    }


def _trace_auto_repair_artifacts(auto_repair_report: dict[str, Any]) -> dict[str, str]:
    artifacts = (
        auto_repair_report.get("artifacts")
        if isinstance(auto_repair_report.get("artifacts"), dict)
        else {}
    )
    return {
        key: value
        for key, value in {
            "repair_task_run_report_path": artifacts.get(
                "repair_task_run_report_path"
            ),
            "smoke_report_path": artifacts.get("smoke_report_path"),
            "runtime_loop_report_path": artifacts.get("runtime_loop_report_path"),
            "audit_report_path": artifacts.get("audit_report_path"),
        }.items()
        if isinstance(value, str) and value
    }


def _resolved_repair_task_args(
    *,
    audit_repair_tasks_path: Path,
    task_id: str | None,
    provided_inputs: dict[str, str],
) -> dict[str, Any]:
    payload = _read_repair_tasks_payload(audit_repair_tasks_path)
    task = _select_repair_task(payload, task_id=task_id)
    args = task.get("suggested_command_args")
    merged = dict(args) if isinstance(args, dict) else {}
    for key, value in provided_inputs.items():
        if key:
            merged[key] = value
    return merged


def _runtime_closed_loop_kwargs_from_repair_args(
    *,
    args: dict[str, Any],
    root: Path,
    audit_repair_tasks_path: Path,
) -> dict[str, Any]:
    trace_path = _required_path_arg(args, "trace_path")
    initial_submission_dir = _required_path_arg(args, "initial_submission_dir")
    return {
        "output_dir": _path_arg(args.get("output_dir"), root / "runtime_repair"),
        "trace_path": trace_path,
        "initial_submission_dir": initial_submission_dir,
        "candidate_generator": str(args.get("candidate_generator") or "postprocess"),
        "candidate_generator_fallback": str(
            args.get("candidate_generator_fallback") or "auto"
        ),
        "max_batches": _int_arg(args.get("max_batches"), default=1),
        "recent_window": _int_arg(args.get("recent_window"), default=50),
        "repeated_family_threshold": _int_arg(
            args.get("repeated_family_threshold"),
            default=3,
        ),
        "failed_family_threshold": _int_arg(
            args.get("failed_family_threshold"),
            default=2,
        ),
        "min_score_delta": _float_arg(args.get("min_score_delta"), default=0.1),
        "max_rounds_per_batch": _int_arg(
            args.get("max_rounds_per_batch"),
            default=1,
        ),
        "batch_deadline_seconds": _float_arg(
            args.get("batch_deadline_seconds"),
            default=300.0,
        ),
        "submit_score_threshold": _float_arg(
            args.get("submit_score_threshold"),
            default=120.0,
        ),
        "learning_objective": str(args.get("learning_objective") or "score"),
        "task1_test_path": _optional_path_arg(args.get("task1_test_path")),
        "task1_target_path": _optional_path_arg(args.get("task1_target_path")),
        "task1_reference_path": _optional_path_arg(args.get("task1_reference_path")),
        "task2_test_path": _optional_path_arg(args.get("task2_test_path")),
        "task2_target_path": _optional_path_arg(args.get("task2_target_path")),
        "timing_profile": _optional_str_arg(args.get("timing_profile")),
        "timing_factor": _optional_float_arg(args.get("timing_factor")),
        "evidence_accumulation_plan_path": _optional_path_arg(
            args.get("evidence_accumulation_plan_path")
        ),
        "expansion_registry_summary_path": _optional_path_arg(
            args.get("expansion_registry_summary_path")
        ),
        "expansion_handoff_report_path": _optional_path_arg(
            args.get("expansion_handoff_report_path")
        ),
        "audit_repair_tasks_path": audit_repair_tasks_path,
        "force": _bool_arg(args.get("force"), default=True),
    }


def _runtime_memory_bridge_smoke_kwargs_from_repair_args(
    *,
    args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trace_path": _required_path_arg(args, "trace_path"),
        "output_dir": _required_path_arg(args, "output_dir"),
        "initial_submission_dir": _required_path_arg(args, "initial_submission_dir"),
        "task1_test_path": _required_path_arg(args, "task1_test_path"),
        "task1_target_path": _required_path_arg(args, "task1_target_path"),
        "audit_repair_tasks_path": _required_path_arg(
            args,
            "audit_repair_tasks_path",
        ),
        "candidate_generator": str(args.get("candidate_generator") or "postprocess"),
        "force": _bool_arg(args.get("force"), default=True),
    }


def _repair_task_run_report(
    *,
    report_path: Path,
    decision: str,
    resolution: dict[str, Any],
    runtime: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    smoke: dict[str, Any] | None,
    next_command: str,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "repair_task_run_report_path": str(report_path),
        "repair_task_resolution_path": resolution.get("repair_task_resolution_path"),
    }
    if isinstance(runtime, dict):
        runtime_artifacts = runtime.get("artifacts")
        if isinstance(runtime_artifacts, dict):
            runtime_path = runtime_artifacts.get("runtime_loop_report_path")
            if isinstance(runtime_path, str):
                artifacts["runtime_loop_report_path"] = runtime_path
    if isinstance(audit, dict):
        audit_path = audit.get("audit_report_path")
        if isinstance(audit_path, str):
            artifacts["audit_report_path"] = audit_path
    if isinstance(smoke, dict):
        smoke_artifacts = smoke.get("artifacts")
        if isinstance(smoke_artifacts, dict):
            smoke_path = smoke_artifacts.get("smoke_report_path")
            if isinstance(smoke_path, str):
                artifacts["smoke_report_path"] = smoke_path
            runtime_path = smoke_artifacts.get("runtime_loop_report_path")
            if isinstance(runtime_path, str):
                artifacts["runtime_loop_report_path"] = runtime_path
            audit_path = smoke_artifacts.get("runtime_closed_loop_audit_path")
            if isinstance(audit_path, str):
                artifacts["audit_report_path"] = audit_path
    metrics = {
        "ready_to_run": resolution.get("ready_to_run") is True,
        "runtime_executed": isinstance(runtime, dict),
        "audit_executed": isinstance(audit, dict),
        "smoke_executed": isinstance(smoke, dict),
        "audit_failed_check_count": _failed_check_count(audit or {}),
    }
    if isinstance(smoke, dict):
        smoke_metrics = (
            smoke.get("metrics") if isinstance(smoke.get("metrics"), dict) else {}
        )
        memory_bridge = (
            smoke.get("memory_bridge")
            if isinstance(smoke.get("memory_bridge"), dict)
            else {}
        )
        metrics["memory_bridge_passed"] = memory_bridge.get("passed") is True
        metrics["repair_policy_recommended_action_count"] = _int_metric(
            smoke_metrics.get("repair_policy_recommended_action_count")
        )
        metrics["batch_memory_constraint_action_count"] = _int_metric(
            smoke_metrics.get("batch_memory_constraint_action_count")
        )
    report = {
        "schema_version": 1,
        "state_hint": REPAIR_TASK_RUN_STATE_HINT,
        "repair_task_run_report_path": str(report_path),
        "decision": decision,
        "repair_task_resolution": resolution,
        "runtime": runtime or {},
        "audit": audit or {},
        "smoke": smoke or {},
        "artifacts": artifacts,
        "metrics": metrics,
        "next_command": next_command,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _required_path_arg(args: dict[str, Any], key: str) -> Path:
    value = args.get(key)
    if value is None or value == "":
        raise HermesRuntimeAuditChainError(f"repair task missing required arg: {key}")
    return Path(str(value))


def _path_arg(value: Any, default: Path) -> Path:
    if value is None or value == "":
        return default
    return Path(str(value))


def _optional_path_arg(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _optional_str_arg(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_float_arg(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return _float_arg(value, default=0.0)


def _int_arg(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise HermesRuntimeAuditChainError("boolean is not a valid integer arg")
    return int(value)


def _int_metric(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    return 0


def _float_arg(value: Any, *, default: float) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise HermesRuntimeAuditChainError("boolean is not a valid float arg")
    return float(value)


def _bool_arg(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _input_repair_errors(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    file = Path(path)
    if not file.exists():
        return []
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    if payload.get("state_hint") != REPAIR_TASKS_STATE_HINT:
        return []
    tasks = payload.get("repair_tasks")
    if not isinstance(tasks, list):
        return []
    errors = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        audit_error = task.get("audit_error")
        if isinstance(audit_error, str) and audit_error and audit_error not in errors:
            errors.append(audit_error)
    return sorted(errors)


def _read_repair_tasks_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HermesRuntimeAuditChainError(f"audit repair tasks not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeAuditChainError(
            f"audit repair tasks invalid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise HermesRuntimeAuditChainError("audit repair tasks must be a JSON object")
    if payload.get("state_hint") != REPAIR_TASKS_STATE_HINT:
        raise HermesRuntimeAuditChainError("audit repair tasks invalid state_hint")
    return payload


def _select_repair_task(
    payload: dict[str, Any],
    *,
    task_id: str | None,
) -> dict[str, Any]:
    tasks = payload.get("repair_tasks")
    if not isinstance(tasks, list) or not tasks:
        raise HermesRuntimeAuditChainError("audit repair tasks are empty")
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if task_id is None or task.get("task_id") == task_id:
            return task
    raise HermesRuntimeAuditChainError(f"audit repair task not found: {task_id}")


def _repair_task_argv(
    *,
    next_command: str,
    suggested_args: dict[str, Any],
) -> list[str]:
    argv = ["uv", "run", "--frozen", "pde-agent", next_command]
    for key, value in suggested_args.items():
        if value is None:
            continue
        option = _repair_task_option_name(next_command=next_command, key=str(key))
        if isinstance(value, bool):
            if value:
                argv.append(option)
            continue
        if isinstance(value, list):
            for item in value:
                argv.extend([option, str(item)])
            continue
        argv.extend([option, str(value)])
    return argv


def _repair_task_option_name(
    *,
    next_command: str,
    key: str,
) -> str:
    option_key = key
    if (
        next_command == "run-hypothesis-learning-constraint-smoke"
        and key == "checkpoint_paths"
    ):
        option_key = "checkpoint_path"
    return "--" + option_key.replace("_", "-")


def _audit_repair_task(
    audit_error: str,
    *,
    root: Path,
    trace_path: Path,
    initial_submission_dir: Path | None = None,
    task1_test_path: Path | None = None,
    task1_target_path: Path | None = None,
    audit_repair_tasks_path: Path | None = None,
    candidate_generator: str | None = None,
) -> dict[str, Any]:
    mapping = {
        "learning_constraint_enforcement": {
            "repair_scope": "candidate_plan_and_heartbeat_schedule",
            "recommended_action": (
                "rebuild_candidate_plan_or_adjust_heartbeat_schedule_to_respect_learning_constraints"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        },
        "persistent_search_guidance_consistency": {
            "repair_scope": "runtime_memory_bridge",
            "recommended_action": (
                "rebuild_kpi_candidate_plan_and_heartbeat_trace_with_consistent_persistent_search_guidance"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        },
        "cluster_saturation_consistency": {
            "repair_scope": "candidate_cluster_saturation_memory",
            "recommended_action": (
                "rebuild_candidate_plan_from_latest_expansion_handoff_and_kpi_saturation_signals"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        },
        "heartbeat_saturation_policy": {
            "repair_scope": "heartbeat_candidate_generator_policy",
            "recommended_action": (
                "switch_blocked_generator_to_registry_allowed_fallback_or_mark_valid_probe"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        },
        "memory_bridge_inputs": {
            "repair_scope": "runtime_input_memory_bridge",
            "recommended_action": (
                "reload_input_expansion_handoff_and_registry_summary_before_candidate_planning"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        },
        "repair_nonconvergence_consistency": {
            "repair_scope": "runtime_repair_nonconvergence_signal_bridge",
            "recommended_action": (
                "rebuild_kpi_candidate_plan_runtime_trace_with_consistent_repair_nonconvergence_signals"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        },
        "repair_convergence_consistency": {
            "repair_scope": "runtime_repair_convergence_signal_bridge",
            "recommended_action": (
                "rebuild_kpi_candidate_plan_runtime_trace_with_consistent_repair_convergence_signals"
            ),
            "next_command": "run-hermes-runtime-closed-loop",
        },
        "learning_constraint_smoke_gate": {
            "repair_scope": "learning_constraint_smoke_artifact_bridge",
            "recommended_action": (
                "rerun_hypothesis_learning_constraint_smoke_and_attach_passed_report_before_runtime_audit"
            ),
            "next_command": "run-hypothesis-learning-constraint-smoke",
        },
        "runtime_memory_bridge_smoke_gate": {
            "repair_scope": "runtime_memory_bridge_smoke_artifact_bridge",
            "recommended_action": (
                "rerun_hermes_runtime_memory_bridge_smoke_and_attach_passed_report_before_runtime_audit"
            ),
            "next_command": "run-hermes-runtime-memory-bridge-smoke",
        },
    }
    details = mapping.get(
        audit_error,
        {
            "repair_scope": "runtime_audit_artifacts",
            "recommended_action": "inspect_failed_audit_gate_and_regenerate_runtime_artifacts",
            "next_command": "inspect-hermes-runtime-audit-chain",
        },
    )
    return {
        "task_id": f"repair_{audit_error}",
        "audit_error": audit_error,
        "priority": "high",
        **details,
        **_audit_repair_command_context(
            audit_error=audit_error,
            root=root,
            trace_path=trace_path,
            initial_submission_dir=initial_submission_dir,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            audit_repair_tasks_path=audit_repair_tasks_path,
            candidate_generator=candidate_generator,
        ),
    }


def _audit_repair_command_context(
    *,
    audit_error: str,
    root: Path,
    trace_path: Path,
    initial_submission_dir: Path | None = None,
    task1_test_path: Path | None = None,
    task1_target_path: Path | None = None,
    audit_repair_tasks_path: Path | None = None,
    candidate_generator: str | None = None,
) -> dict[str, Any]:
    if audit_error == "learning_constraint_smoke_gate":
        local_strategy_library_path = _discover_local_strategy_library(root)
        suggested_command_args: dict[str, Any] = {
            "trace_path": str(trace_path),
            "output_dir": str(root / "learning_constraint_smoke"),
            "candidate_generator": "postprocess",
            "checkpoint_paths": [],
            "min_viable_candidate_count": 999,
            "force": True,
        }
        missing_inputs = []
        if local_strategy_library_path is None:
            missing_inputs.append("local_strategy_library_path")
        else:
            suggested_command_args["local_strategy_library_path"] = str(
                local_strategy_library_path
            )
        return {
            "suggested_command_args": suggested_command_args,
            "missing_inputs": missing_inputs,
        }
    if audit_error == "runtime_memory_bridge_smoke_gate":
        suggested_command_args = {
            "trace_path": str(trace_path),
            "output_dir": str(root / "runtime_memory_bridge_smoke"),
            "candidate_generator": candidate_generator or "postprocess",
            "force": True,
        }
        missing_inputs = []
        required_paths = {
            "initial_submission_dir": initial_submission_dir,
            "task1_test_path": task1_test_path,
            "task1_target_path": task1_target_path,
            "audit_repair_tasks_path": audit_repair_tasks_path,
        }
        for key, value in required_paths.items():
            if value is None:
                missing_inputs.append(key)
            else:
                suggested_command_args[key] = str(value)
        return {
            "suggested_command_args": suggested_command_args,
            "missing_inputs": missing_inputs,
        }
    return {}


def _discover_local_strategy_library(root: Path) -> Path | None:
    for candidate in [
        root / "local_strategy_library.json",
        root.parent / "local_strategy_library.json",
        Path("runs/local_strategy_library.json"),
    ]:
        if candidate.is_file():
            return candidate
    return None


def _write_audit_repair_tasks(
    *,
    path: Path,
    chain_report_path: Path,
    repair_tasks: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": 1,
        "state_hint": REPAIR_TASKS_STATE_HINT,
        "chain_report_path": str(chain_report_path),
        "repair_tasks": repair_tasks,
        "metrics": {"repair_task_count": len(repair_tasks)},
        "next_command": repair_tasks[0]["next_command"]
        if repair_tasks
        else "run-hermes-runtime-audit-chain",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _best_submission_dir(runtime: dict[str, Any]) -> Path | None:
    heartbeat = runtime.get("heartbeat")
    if not isinstance(heartbeat, dict):
        return None
    best = heartbeat.get("best_candidate")
    if not isinstance(best, dict):
        return None
    value = best.get("submission_dir")
    if not isinstance(value, str) or not value:
        return None
    return Path(value)
