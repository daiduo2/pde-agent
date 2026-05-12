"""Main iteration runner for FNO training closed loops."""

import json
from pathlib import Path
from typing import Any

from pde_agent.core.fno_training_agent_interface import (
    build_fno_training_iteration_report,
)
from pde_agent.core.fno_training_agent_interface import (
    build_fno_training_closed_loop_contract as _build_contract,
)
from pde_agent.core.fno_training_checkpoint import (
    checkpoint_training_loop,
)
from pde_agent.core.fno_training_convergence import check_training_convergence
from pde_agent.core.fno_training_memory import (
    build_training_iteration_memory,
    load_training_iteration_memory,
    merge_training_memory_with_hypothesis_feedback,
)
from pde_agent.core.json_output import error_payload, success_payload
from pde_agent.core.task1_short_window_training import (
    plan_fno_training_candidates,
    run_fno_training_optimization_loop,
)


STATE_HINT = "FNO_TRAINING_ITERATION_REPORTED"
CONTRACT_STATE_HINT = "FNO_TRAINING_CONTRACT_BUILT"


class FnoTrainingClosedLoopError(RuntimeError):
    pass


def run_fno_training_iteration(
    *,
    workspace_dir: str | Path,
    source_path: str | Path,
    memory_path: str | Path | None = None,
    evidence_paths: list[str | Path] | None = None,
    dataset_key: str = "tensor",
    train_sample_limit: int = 64,
    val_sample_limit: int = 32,
    context_steps: int = 10,
    time_reduction: int = 5,
    space_reduction: int = 4,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
    max_candidates: int = 3,
    modes: list[int] | None = None,
    widths: list[int] | None = None,
    depths: list[int] | None = None,
    loss_types: list[str] | None = None,
    forecast_steps_values: list[int] | None = None,
    train_steps_values: list[int] | None = None,
    normalizations: list[str] | None = None,
    target_score: float | None = None,
    patience: int = 3,
    checkpoint_every: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(workspace_dir)
    root.mkdir(parents=True, exist_ok=True)

    previous_memory = None
    iteration_index = 0
    if memory_path is not None:
        previous_memory = load_training_iteration_memory(memory_path)
        if previous_memory is not None:
            iteration_index = int(previous_memory.get("iteration_index", 0)) + 1

    evidence_paths = _resolve_evidence_paths(
        evidence_paths=evidence_paths,
        workspace_dir=root,
        memory_path=memory_path,
        previous_memory=previous_memory,
    )

    plan_path = root / f"fno_training_plan_{iteration_index:04d}.json"
    optimization_dir = root / f"optimization_{iteration_index:04d}"
    memory_output_path = root / f"fno_training_memory_{iteration_index:04d}.json"
    convergence_path = root / f"fno_training_convergence_{iteration_index:04d}.json"
    report_path = root / "fno_training_iteration_report.json"

    def _run_body(
        *,
        max_candidates_override: int | None = None,
    ) -> dict[str, Any]:
        blocked_families: list[str] = []
        search_guidance: list[dict[str, Any]] = []
        budget_adjustment: dict[str, Any] = {}
        if previous_memory is not None:
            family_outcomes = previous_memory.get("family_outcomes", {})
            blocked_families = [
                family
                for family, outcome in family_outcomes.items()
                if isinstance(outcome, dict) and outcome.get("status") == "blocked"
            ]
            search_guidance = previous_memory.get("search_guidance", [])
            convergence_history = previous_memory.get("convergence_history", {})
            no_improvement = convergence_history.get("no_improvement_count", 0)
            if no_improvement >= 2:
                budget_adjustment["train_steps_delta"] = 2
            decision_policy = previous_memory.get("decision_policy", {})
            if isinstance(decision_policy, dict):
                policy_budget = decision_policy.get("budget_adjustments")
                if isinstance(policy_budget, dict):
                    budget_adjustment.update(policy_budget)
        plan = plan_fno_training_candidates(
            evidence_paths=[str(p) for p in evidence_paths],
            output_path=str(plan_path),
            max_candidates=max_candidates_override or max_candidates,
            modes=modes or [4, 8],
            widths=widths or [16, 32],
            depths=depths or [1],
            loss_types=loss_types or ["mse", "early_weighted_mse"],
            forecast_steps_values=forecast_steps_values or [12, 20],
            train_steps_values=train_steps_values or [2, 4],
            normalizations=normalizations or ["none"],
            blocked_families=blocked_families or None,
            search_guidance=search_guidance or None,
            budget_adjustment=budget_adjustment or None,
            force=force,
        )

        opt_report = run_fno_training_optimization_loop(
            candidate_plan_path=str(plan_path),
            source_path=str(source_path),
            output_dir=str(optimization_dir),
            dataset_key=dataset_key,
            train_sample_limit=train_sample_limit,
            val_sample_limit=val_sample_limit,
            context_steps=context_steps,
            time_reduction=time_reduction,
            space_reduction=space_reduction,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + iteration_index,
            force=force,
        )

        eval_report = evaluate_fno_training_iteration(opt_report=opt_report)

        memory = build_training_iteration_memory(
            optimization_report=opt_report,
            iteration_index=iteration_index,
            previous_memory=previous_memory,
        )

        convergence = check_training_convergence(
            memory=memory,
            target_score=target_score,
            patience=patience,
        )

        summary = _build_summary(
            eval_report=eval_report,
            previous_best_score=_float_or_none(
                previous_memory.get("best_so_far", {}).get("score") if previous_memory else None
            ),
        )

        iteration_result = {
            "iteration_index": iteration_index,
            "summary": summary,
            "convergence": convergence,
            "memory": memory,
            "artifacts": {
                "optimization_report_path": str(optimization_dir / "fno_training_optimization_report.json"),
                "memory_path": str(memory_output_path),
                "convergence_path": str(convergence_path),
                "report_path": str(report_path),
            },
        }

        report = build_fno_training_iteration_report(iteration_result=iteration_result)

        _write_json(memory_output_path, memory, force=force)
        _write_json(convergence_path, convergence, force=force)
        _write_json(report_path, report, force=force)

        if checkpoint_every > 0 and (iteration_index + 1) % checkpoint_every == 0:
            checkpoint_training_loop(
                workspace_dir=root,
                iteration_index=iteration_index,
                memory=memory,
                convergence=convergence,
                best_config=summary.get("config"),
                trajectory=memory.get("trajectory_summary", []),
                force=force,
            )

        final_report_path: str | None = None
        if convergence.get("decision") == "converged":
            final_report = build_fno_training_final_report(
                memory_path=memory_output_path,
                workspace_dir=root,
                force=force,
            )
            final_report_path = str(root / "fno_training_final_report.json")

        artifacts: dict[str, Any] = {
            "optimization_report_path": str(optimization_dir / "fno_training_optimization_report.json"),
            "memory_path": str(memory_output_path),
            "convergence_path": str(convergence_path),
            "report_path": str(report_path),
            "plan_path": str(plan_path),
        }
        if final_report_path is not None:
            artifacts["final_report_path"] = final_report_path

        return success_payload(
            run_id=root.name or "fno-training",
            command="run-fno-training-iteration",
            state_hint=STATE_HINT,
            artifacts=artifacts,
            metrics={
                "iteration_index": iteration_index,
                "best_score": summary.get("best_score"),
                "score_delta_vs_previous": summary.get("score_delta_vs_previous"),
                "convergence_decision": convergence.get("decision"),
                "candidate_count": len(opt_report.get("candidate_records", [])),
            },
            next_hint=_next_hint(convergence.get("decision", "continue")),
        )

    try:
        return _run_body()
    except Exception as exc:
        recovered = False
        # Retry once with degraded parameters
        try:
            return _run_body(max_candidates_override=max(1, max_candidates - 1))
        except Exception:
            pass
        # Emit error report so the agent can decide next action
        error_memory = _build_error_memory(
            iteration_index=iteration_index,
            previous_memory=previous_memory,
            error_message=str(exc),
            error_type=type(exc).__name__,
        )
        convergence = _build_error_convergence(
            error_message=str(exc),
            error_type=type(exc).__name__,
            recovered=recovered,
        )
        _write_json(memory_output_path, error_memory, force=force)
        _write_json(convergence_path, convergence, force=force)
        summary = {
            "best_candidate_id": None,
            "best_score": None,
            "score_delta_vs_previous": None,
            "model_family": None,
            "config": None,
        }
        iteration_result = {
            "iteration_index": iteration_index,
            "summary": summary,
            "convergence": convergence,
            "memory": error_memory,
            "artifacts": {
                "optimization_report_path": str(optimization_dir / "fno_training_optimization_report.json"),
                "memory_path": str(memory_output_path),
                "convergence_path": str(convergence_path),
                "report_path": str(report_path),
            },
        }
        report = build_fno_training_iteration_report(iteration_result=iteration_result)
        _write_json(report_path, report, force=force)
        return success_payload(
            run_id=root.name or "fno-training",
            command="run-fno-training-iteration",
            state_hint=STATE_HINT,
            artifacts={
                "optimization_report_path": str(optimization_dir / "fno_training_optimization_report.json"),
                "memory_path": str(memory_output_path),
                "convergence_path": str(convergence_path),
                "report_path": str(report_path),
                "plan_path": str(plan_path),
                "recovery_actions": [
                    {
                        "action": "retry_with_degraded_params",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "suggested_params": {
                            "max_candidates": max(1, max_candidates - 1),
                            "modes": [4],
                            "widths": [16],
                            "train_steps_values": [2],
                        },
                    },
                    {
                        "action": "escalate_to_hypothesis_lab",
                        "reason": "iteration_recovery_exhausted",
                    },
                ],
            },
            metrics={
                "iteration_index": iteration_index,
                "best_score": None,
                "score_delta_vs_previous": None,
                "convergence_decision": "error",
                "candidate_count": 0,
            },
            next_hint="iteration_failed_review_error_and_decide",
        )


def evaluate_fno_training_iteration(
    *,
    opt_report: dict[str, Any],
) -> dict[str, Any]:
    if opt_report.get("state_hint") != "FNO_TRAINING_OPTIMIZATION_LOOP_RAN":
        raise FnoTrainingClosedLoopError("optimization report has invalid state_hint")
    records = opt_report.get("candidate_records", [])
    selected = opt_report.get("selected_candidate", {})
    if not records:
        raise FnoTrainingClosedLoopError("optimization report has no candidate records")
    best_record = selected if isinstance(selected, dict) and selected.get("score") else max(
        records,
        key=lambda item: float(item.get("score", float("-inf")))
        if isinstance(item.get("score"), int | float)
        else float("-inf"),
    )
    scores = [
        float(r["score"]) for r in records
        if isinstance(r.get("score"), int | float)
    ]
    return {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_ITERATION_EVALUATED",
        "best_candidate_id": best_record.get("candidate_id"),
        "best_score": best_record.get("score"),
        "best_model_family": best_record.get("model_family"),
        "best_config": best_record.get("config"),
        "candidate_count": len(records),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }


def build_fno_training_closed_loop_contract(
    *,
    workspace_dir: str | Path,
    source_path: str | Path,
    output_path: str | Path | None = None,
    dataset_key: str = "tensor",
    baseline_prediction_path: str | Path | None = None,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    required_commands: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    return _build_contract(
        workspace_dir=workspace_dir,
        source_path=source_path,
        output_path=output_path,
        dataset_key=dataset_key,
        baseline_prediction_path=baseline_prediction_path,
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        required_commands=required_commands,
        force=force,
    )


def _resolve_evidence_paths(
    *,
    evidence_paths: list[str | Path] | None,
    workspace_dir: Path,
    memory_path: str | Path | None,
    previous_memory: dict[str, Any] | None,
) -> list[Path]:
    if evidence_paths:
        return [Path(p) for p in evidence_paths]
    paths: list[Path] = []
    if previous_memory is not None:
        if memory_path is not None:
            memory_file = Path(memory_path)
            if memory_file.is_file():
                paths.append(memory_file)
        prev_artifacts = previous_memory.get("source_optimization_report_path")
        if isinstance(prev_artifacts, str):
            evidence_path = Path(prev_artifacts)
            if evidence_path.is_file():
                paths.append(evidence_path)
    for candidate in ["fno_training_optimization_evidence.json"]:
        candidate_path = workspace_dir / candidate
        if candidate_path.is_file():
            paths.append(candidate_path)
    health_feedback_path = workspace_dir / "fno_training_health_feedback.json"
    if health_feedback_path.is_file():
        paths.append(health_feedback_path)
    return paths


def _build_summary(
    *,
    eval_report: dict[str, Any],
    previous_best_score: float | None,
) -> dict[str, Any]:
    best_score = _float_or_none(eval_report.get("best_score"))
    score_delta = None
    if best_score is not None and previous_best_score is not None:
        score_delta = round(best_score - previous_best_score, 6)
    return {
        "best_candidate_id": eval_report.get("best_candidate_id"),
        "best_score": best_score,
        "score_delta_vs_previous": score_delta,
        "model_family": eval_report.get("best_model_family"),
        "config": eval_report.get("best_config"),
    }


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FnoTrainingClosedLoopError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _build_error_memory(
    *,
    iteration_index: int,
    previous_memory: dict[str, Any] | None,
    error_message: str,
    error_type: str,
) -> dict[str, Any]:
    best_so_far = previous_memory.get("best_so_far") if previous_memory else None
    return {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_MEMORY_BUILT",
        "iteration_index": iteration_index,
        "source_optimization_report_path": None,
        "best_so_far": best_so_far or {},
        "trajectory_summary": [],
        "avoid_repeating": [
            {
                "candidate_id": f"error_{iteration_index}",
                "model_family": "unknown",
                "loss_type": "unknown",
                "config": {},
                "reason": f"iteration_error:{error_type}",
            }
        ],
        "family_outcomes": {},
        "positive_signals": [],
        "search_guidance": [
            {
                "guidance_type": "degraded_retry",
                "reason": f"Previous iteration failed with {error_type}: {error_message}",
                "recommended_action": "reduce_candidates_and_train_steps",
            }
        ],
        "convergence_history": previous_memory.get("convergence_history", {}) if previous_memory else {},
        "next_hints": [
            f"Iteration {iteration_index} failed with {error_type}. "
            "Consider retry with degraded parameters or escalate.",
        ],
    }


def _build_error_convergence(
    *,
    error_message: str,
    error_type: str,
    recovered: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_CONVERGENCE_CHECKED",
        "decision": "error",
        "reason": f"iteration_failed:{error_type}: {error_message}",
        "confidence": 0.0,
        "recovered": recovered,
        "recommended_actions": [
            {
                "action": "retry_with_degraded_params",
                "reason": error_message,
                "params": {},
            },
            {
                "action": "escalate_to_hypothesis_lab",
                "reason": "iteration_recovery_exhausted",
            },
        ],
        "blocked_families": [],
        "triggers": [
            {
                "trigger": "iteration_failure",
                "error_type": error_type,
                "error_message": error_message,
            }
        ],
    }


def _next_hint(decision: str) -> str:
    if decision == "continue":
        return "run-fno-training-iteration again with the updated memory"
    if decision == "escalate":
        return "escalate to hypothesis lab for strategy discovery"
    if decision == "converged":
        return "training converged; review best candidate artifacts"
    if decision == "stagnant":
        return "training stagnant; consider escalation or parameter expansion"
    if decision == "doom_loop":
        return "doom loop detected; escalate to hypothesis lab immediately"
    if decision == "error":
        return "iteration failed; review recovery_actions and decide next step"
    return "run-fno-training-iteration again"


FINAL_REPORT_STATE_HINT = "FNO_TRAINING_FINAL_REPORT_BUILT"
HEALTH_REPORT_STATE_HINT = "FNO_TRAINING_HEALTH_REPORT_BUILT"


def build_fno_training_health_report(
    *,
    workspace_dir: str | Path,
    output_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace_dir).expanduser().resolve()
    if not workspace.is_dir():
        raise FnoTrainingClosedLoopError(f"workspace does not exist: {workspace}")
    memory_files = sorted(workspace.glob("fno_training_memory_*.json"))
    plan_files = sorted(workspace.glob("fno_training_plan_*.json"))
    convergence_files = sorted(workspace.glob("fno_training_convergence_*.json"))
    memories = [_load_json(path) for path in memory_files]
    plans = [_load_json(path) for path in plan_files]
    convergences = [_load_json(path) for path in convergence_files]
    memories = [item for item in memories if isinstance(item, dict)]
    plans = [item for item in plans if isinstance(item, dict)]
    convergences = [item for item in convergences if isinstance(item, dict)]

    score_history = [
        _float_or_none(item.get("current_best", {}).get("score"))
        for item in memories
        if isinstance(item.get("current_best"), dict)
    ]
    score_history = [score for score in score_history if score is not None]
    best_scores = [
        _float_or_none(item.get("best_so_far", {}).get("score"))
        for item in memories
        if isinstance(item.get("best_so_far"), dict)
    ]
    best_scores = [score for score in best_scores if score is not None]
    family_counts = _family_counts(plans)
    rationale_counts = _rationale_counts(plans)
    anchor_status_counts = _anchor_status_counts(memories)
    convergence_counts = _convergence_counts(convergences)
    latest_convergence = convergences[-1] if convergences else {}
    issues = _health_issues(
        score_history=score_history,
        family_counts=family_counts,
        rationale_counts=rationale_counts,
        anchor_status_counts=anchor_status_counts,
        convergence_counts=convergence_counts,
        latest_convergence=latest_convergence,
    )
    report = {
        "schema_version": 1,
        "state_hint": HEALTH_REPORT_STATE_HINT,
        "workspace_dir": str(workspace),
        "iteration_count": len(memories),
        "score": {
            "current_best_history": score_history,
            "global_best": max(best_scores) if best_scores else None,
            "latest_current_best": score_history[-1] if score_history else None,
            "improvement_from_first": (
                score_history[-1] - score_history[0]
                if len(score_history) >= 2
                else None
            ),
        },
        "exploration": {
            "candidate_family_counts": family_counts,
            "rationale_counts": rationale_counts,
            "anchor_status_counts": anchor_status_counts,
            "convergence_decision_counts": convergence_counts,
        },
        "health": {
            "status": "healthy" if not issues else "needs_attention",
            "issues": issues,
        },
        "supervisor_questions": [
            "Is the agent improving score without collapsing to one family?",
            "Are anchors being challenged as well as reused?",
            "Did convergence request hypothesis-lab escalation?",
        ],
        "next_hint": "review health.status and let the agent continue or escalate",
    }
    out_file = Path(output_path) if output_path is not None else workspace / "fno_training_health_report.json"
    _write_json(out_file, report, force=force)
    feedback_path = workspace / "fno_training_health_feedback.json"
    _write_json(
        feedback_path,
        build_fno_training_health_feedback(health_report=report),
        force=True,
    )
    return report


def build_fno_training_health_feedback(
    *,
    health_report: dict[str, Any],
) -> dict[str, Any]:
    if health_report.get("state_hint") != HEALTH_REPORT_STATE_HINT:
        raise FnoTrainingClosedLoopError("health report has invalid state_hint")
    guidance: list[dict[str, Any]] = []
    issues = health_report.get("health", {}).get("issues", [])
    if not isinstance(issues, list):
        issues = []
    issue_codes = {
        item.get("code") for item in issues
        if isinstance(item, dict)
    }
    family_counts = (
        health_report.get("exploration", {}).get("candidate_family_counts", {})
        if isinstance(health_report.get("exploration"), dict)
        else {}
    )
    dominant_family = _dominant_family(family_counts)
    if "single_family_collapse" in issue_codes or _family_bias_is_high(family_counts):
        alternatives = [
            family for family in ["fno_ar1", "conv1d_residual", "fno_lite"]
            if family != dominant_family
        ]
        guidance.append(
            {
                "guidance_type": "operator_training_policy",
                "next_action": "rebalance_candidate_families",
                "required_axis_changes": ["model_family"],
                "preferred_model_families": alternatives,
                "preferred_loss_types": ["early_weighted_mse", "horizon_weighted_mse"],
                "budget_adjustments": {},
                "reason_codes": ["health_family_bias"],
            }
        )
    if "convergence_warning" in issue_codes:
        convergence_issue = _first_issue(issues, "convergence_warning")
        issue_evidence = (
            convergence_issue.get("evidence", {})
            if isinstance(convergence_issue, dict)
            else {}
        )
        blocked_families = _string_list(issue_evidence.get("blocked_families"))
        preferred_families = [
            family for family in ["fno_ar1", "conv1d_residual", "fno_lite"]
            if family not in set(blocked_families)
        ]
        if not preferred_families:
            preferred_families = ["fno_ar1", "conv1d_residual"]
        next_action = (
            "request_new_dynamics_source"
            if blocked_families
            else "expand_search_radius"
        )
        guidance.append(
            {
                "guidance_type": "operator_training_policy",
                "next_action": next_action,
                "required_axis_changes": ["model_family", "loss_type", "capacity"],
                "preferred_model_families": preferred_families,
                "preferred_loss_types": ["horizon_weighted_mse", "early_weighted_mse"],
                "budget_adjustments": {"train_steps_delta": 2},
                "reason_codes": ["health_convergence_warning"],
                "blocked_families": blocked_families,
                "hypothesis_lab_requests": [
                    {
                        "request_type": "expand_candidate_family",
                        "reason": "health_convergence_warning",
                        "blocked_families": blocked_families,
                    }
                ],
            }
        )
    if "anchor_not_challenged" in issue_codes:
        guidance.append(
            {
                "guidance_type": "operator_training_policy",
                "next_action": "challenge_anchor",
                "required_axis_changes": ["model_family"],
                "preferred_model_families": ["fno_ar1", "conv1d_residual"],
                "reason_codes": ["health_anchor_not_challenged"],
            }
        )
    if "latest_iteration_not_improving" in issue_codes:
        score = health_report.get("score", {})
        score_context = score if isinstance(score, dict) else {}
        guidance.append(
            {
                "guidance_type": "operator_training_policy",
                "next_action": "repair_regression",
                "required_axis_changes": ["loss_type"],
                "preferred_loss_types": [
                    "early_weighted_mse",
                    "horizon_weighted_mse",
                    "spectral_mse",
                    "mse",
                ],
                "budget_adjustments": {},
                "reason_codes": ["health_latest_iteration_not_improving"],
                "score_context": {
                    "latest_current_best": score_context.get("latest_current_best"),
                    "global_best": score_context.get("global_best"),
                },
            }
        )
    return {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_HEALTH_FEEDBACK_BUILT",
        "source_health_report_path": str(
            Path(str(health_report.get("workspace_dir", "."))) / "fno_training_health_report.json"
        ),
        "health_status": health_report.get("health", {}).get("status"),
        "search_guidance": guidance,
        "next_command": "run-fno-training-iteration",
    }


def build_fno_training_final_report(
    *,
    memory_path: str | Path,
    workspace_dir: str | Path,
    output_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    memory_file = Path(memory_path).expanduser().resolve()
    workspace = Path(workspace_dir).expanduser().resolve()
    out_file = Path(output_path) if output_path is not None else workspace / "fno_training_final_report.json"

    memory = _load_json(memory_file)
    if memory is None:
        raise FnoTrainingClosedLoopError(f"could not load memory: {memory_file}")

    best_so_far = memory.get("best_so_far", {}) if isinstance(memory, dict) else {}
    trajectory = memory.get("trajectory_summary", []) if isinstance(memory, dict) else []
    family_outcomes = memory.get("family_outcomes", {}) if isinstance(memory, dict) else {}

    best_candidate = best_so_far if best_so_far else (
        trajectory[-1] if trajectory else {}
    )

    provenance = {
        "iteration_count": int(memory.get("iteration_index", 0)) + 1,
        "trajectory_length": len(trajectory),
        "blocked_family_count": len(
            [f for f, o in family_outcomes.items() if isinstance(o, dict) and o.get("status") == "blocked"]
        ),
    }

    report = {
        "schema_version": 1,
        "state_hint": FINAL_REPORT_STATE_HINT,
        "workspace_dir": str(workspace),
        "memory_path": str(memory_file),
        "best_candidate": {
            "candidate_id": best_candidate.get("candidate_id"),
            "model_family": best_candidate.get("model_family"),
            "score": best_candidate.get("score"),
            "config": best_candidate.get("config"),
        },
        "provenance": provenance,
        "artifacts": {
            "memory_path": str(memory_file),
            "convergence_path": str(workspace / f"fno_training_convergence_{memory.get('iteration_index', 0):04d}.json"),
            "optimization_report_path": memory.get("source_optimization_report_path"),
        },
        "agent_instructions": [
            "Training loop has converged.",
            "Review best_candidate config and artifacts.",
            "Run build-competition-submission or record-promoted-algorithm-strategy-proposal if satisfied.",
        ],
    }

    _write_json(out_file, report, force=force)
    return report


def _family_counts(plans: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        configs = plan.get("candidate_configs")
        if not isinstance(configs, list):
            continue
        for config in configs:
            if not isinstance(config, dict):
                continue
            family = config.get("model_family")
            if isinstance(family, str) and family:
                counts[family] = counts.get(family, 0) + 1
    return counts


def _rationale_counts(plans: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        configs = plan.get("candidate_configs")
        if not isinstance(configs, list):
            continue
        for config in configs:
            rationales = config.get("rationale")
            if not isinstance(rationales, list):
                continue
            for rationale in rationales:
                if isinstance(rationale, str) and rationale:
                    counts[rationale] = counts.get(rationale, 0) + 1
    return counts


def _anchor_status_counts(memories: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    latest = memories[-1] if memories else {}
    anchors = latest.get("training_memory_anchors") if isinstance(latest, dict) else []
    if not isinstance(anchors, list):
        return counts
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        status = str(anchor.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _convergence_counts(convergences: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for convergence in convergences:
        decision = convergence.get("decision")
        if isinstance(decision, str) and decision:
            counts[decision] = counts.get(decision, 0) + 1
    return counts


def _health_issues(
    *,
    score_history: list[float],
    family_counts: dict[str, int],
    rationale_counts: dict[str, int],
    anchor_status_counts: dict[str, int],
    convergence_counts: dict[str, int],
    latest_convergence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    total_candidates = sum(family_counts.values())
    if total_candidates >= 6 and len(family_counts) <= 1:
        issues.append(
            {
                "code": "single_family_collapse",
                "severity": "medium",
                "evidence": {"family_counts": family_counts},
            }
        )
    if (
        rationale_counts.get("training_memory_anchor_reuse", 0) > 0
        and rationale_counts.get("training_memory_anchor_family_challenge", 0) == 0
    ):
        issues.append(
            {
                "code": "anchor_not_challenged",
                "severity": "medium",
                "evidence": {"rationale_counts": rationale_counts},
            }
        )
    if anchor_status_counts.get("challenged", 0) or anchor_status_counts.get("deprecated", 0):
        issues.append(
            {
                "code": "anchor_regression_detected",
                "severity": "low",
                "evidence": {"anchor_status_counts": anchor_status_counts},
            }
        )
    if convergence_counts.get("doom_loop", 0) or convergence_counts.get("stagnant", 0):
        latest = latest_convergence if isinstance(latest_convergence, dict) else {}
        issues.append(
            {
                "code": "convergence_warning",
                "severity": "high",
                "evidence": {
                    "convergence_decision_counts": convergence_counts,
                    "latest_decision": latest.get("decision"),
                    "latest_reason": latest.get("reason"),
                    "blocked_families": _string_list(latest.get("blocked_families")),
                    "recommended_actions": latest.get("recommended_actions", []),
                },
            }
        )
    if convergence_counts.get("error", 0):
        issues.append(
            {
                "code": "iteration_errors_detected",
                "severity": "high",
                "evidence": {"convergence_decision_counts": convergence_counts},
            }
        )
    if convergence_counts and not score_history:
        issues.append(
            {
                "code": "no_successful_score_observed",
                "severity": "high",
                "evidence": {"convergence_decision_counts": convergence_counts},
            }
        )
    if len(score_history) >= 4 and score_history[-1] <= max(score_history[:-1]):
        issues.append(
            {
                "code": "latest_iteration_not_improving",
                "severity": "low",
                "evidence": {
                    "latest_current_best": score_history[-1],
                    "previous_best_current_score": max(score_history[:-1]),
                },
            }
        )
    return issues


def _first_issue(issues: list[Any], code: str) -> dict[str, Any] | None:
    for issue in issues:
        if isinstance(issue, dict) and issue.get("code") == code:
            return issue
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _dominant_family(family_counts: Any) -> str | None:
    if not isinstance(family_counts, dict) or not family_counts:
        return None
    valid = {
        str(family): int(count)
        for family, count in family_counts.items()
        if isinstance(count, int | float) and not isinstance(count, bool)
    }
    if not valid:
        return None
    return max(valid, key=lambda family: valid[family])


def _family_bias_is_high(family_counts: Any) -> bool:
    if not isinstance(family_counts, dict):
        return False
    counts = [
        int(count) for count in family_counts.values()
        if isinstance(count, int | float) and not isinstance(count, bool)
    ]
    total = sum(counts)
    if total < 8 or not counts:
        return False
    return max(counts) / total >= 0.8


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None
