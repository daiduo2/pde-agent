from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_utils import (
    HypothesisLabError,
    _guard_output,
    _read_json,
    _string_list_from_any,
    _write_json,
    _write_optional_json,
    SANDBOX_STATE_HINT,
)
from pde_agent.core.literature_expansion import (
    FAILURE_MEMORY_STATE_HINT,
    record_literature_candidate_outcome,
)
from pde_agent.core.strategy_growth import record_strategy_proposal


def _lazy_normalize_evidence_params(params: dict[str, Any]) -> dict[str, Any] | None:
    from pde_agent.core.hypothesis_lab import _normalize_evidence_params
    return _normalize_evidence_params(params)


def _algorithm_candidate_cards(candidate_file: Path) -> list[dict[str, Any]]:
    report = _read_json(candidate_file)
    if report.get("state_hint") != "ALGORITHM_CANDIDATES_COMPILED":
        raise HypothesisLabError("algorithm candidates artifact has invalid state_hint")
    candidates = report.get("algorithm_candidates")
    if not isinstance(candidates, list):
        raise HypothesisLabError("algorithm_candidates must be a list")
    cards = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        card = _algorithm_candidate_card(candidate_file, candidate)
        if card is not None:
            cards.append(card)
    return cards


def _pareto_memory_cards(memory_file: Path) -> list[dict[str, Any]]:
    memory = _read_json(memory_file)
    if memory.get("state_hint") != FAILURE_MEMORY_STATE_HINT:
        raise HypothesisLabError("failure memory has invalid state_hint")
    cards = []
    for index, event in enumerate(memory.get("pareto_learning_events", [])):
        if not isinstance(event, dict):
            continue
        card = _pareto_memory_card(memory_file=memory_file, event=event, index=index)
        if card is not None:
            cards.append(card)
    return cards


def _pareto_memory_card(
    memory_file: Path,
    event: dict[str, Any],
    index: int = 0,
) -> dict[str, Any] | None:
    family = str(event.get("candidate_family", ""))
    if family not in {
        "linear_trend_spectral",
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
        "task1_early_refinement",
        "task1_early_transition_expert",
        "task1_retrieval_refinement",
        "task1_retrieval_refinement_local_search",
    }:
        return None
    params = event.get("candidate_params")
    if not isinstance(params, dict) or _lazy_normalize_evidence_params(params) is None:
        return None
    outcome = str(event.get("outcome", "unknown"))
    pareto_status = str(event.get("pareto_status", "unknown"))
    candidate_id = str(event.get("candidate_id") or f"pareto_event_{index}")
    diagnosis = _pareto_memory_early_diagnosis(event)
    avoid_directives = _string_list_from_any(event.get("avoid_directives"))
    retry_directives = _string_list_from_any(event.get("retry_directives"))
    effective_axes = _string_list_from_any(event.get("effective_axes"))
    negative_candidate_evidence = [
        item
        for item in event.get("negative_candidate_evidence", [])
        if isinstance(item, dict)
    ]
    return {
        "card_id": f"pareto_memory_{memory_file.stem}_{index:03d}",
        "source_type": "pareto_learning_memory",
        "source_path": str(memory_file),
        "claim": (
            f"Pareto memory for {family} recorded outcome={outcome} "
            f"and status={pareto_status} for {candidate_id}."
        ),
        "procedural_heuristic": (
            "Use promoted axes directly, keep evidence-only axes under guarded "
            "local search, and avoid repeating parameters that only improve a "
            "single objective without passing the Pareto gate."
        ),
        "preconditions": [
            "pareto learning memory exists",
            f"candidate_family={family}",
            f"pareto_status={pareto_status}",
        ],
        "implementation_hooks": [family],
        "expected_observables": [
            "candidate generation changes after reading Pareto memory",
            "effective axes are reused only under guardrails",
        ],
        "known_risks": avoid_directives
        + [
            f"deprioritize_{item.get('axis')}:{item.get('value')}"
            for item in negative_candidate_evidence
            if item.get("axis") and item.get("value")
        ]
        or ["memory replay can overfit a single sandbox window"],
        "evidence_status": outcome,
        "evidence": {
            "source_kind": "pareto_memory",
            "params": params,
            "candidate_id": candidate_id,
            "pareto_status": pareto_status,
            "outcome": outcome,
            "decision": event.get("decision"),
            "effective_axes": effective_axes,
            "retry_directives": retry_directives,
            "avoid_directives": avoid_directives,
            "candidate_pool_attribution": event.get("candidate_pool_attribution")
            if isinstance(event.get("candidate_pool_attribution"), dict)
            else {},
            "negative_candidate_evidence": negative_candidate_evidence,
            "early_segment_diagnosis": diagnosis,
            "task1_total_score_proxy": event.get("best_segment_score"),
            "segment_score": event.get("best_segment_score"),
        },
    }


def _pareto_memory_early_diagnosis(event: dict[str, Any]) -> dict[str, Any]:
    directives = [
        {
            "directive": directive,
            "target_family": str(event.get("candidate_family", "")),
            "rationale": "replayed from Pareto learning memory",
        }
        for directive in _string_list_from_any(event.get("action_directives"))
    ]
    effective_axes = _string_list_from_any(event.get("effective_axes"))
    recommended_axis = effective_axes[0] if effective_axes else None
    return {
        "schema_version": 2,
        "state_hint": "TASK1_EARLY_SEGMENT_DIAGNOSED",
        "decision": "early_segment_bottleneck",
        "reason_codes": _string_list_from_any(event.get("diagnosis_reason_codes")),
        "bottleneck_type": event.get("bottleneck_type"),
        "candidate_trend_attribution": {
            "primary_metric": "early_rel_mse",
            "recommended_axis": recommended_axis,
        },
        "error_taxonomy": _pareto_memory_error_taxonomy(event),
        "action_directives": directives,
        "objective_recommendation": {
            "primary": "early_rel_mse",
            "next_candidate_family": "task1_early_refinement",
        },
    }


def _pareto_memory_error_taxonomy(event: dict[str, Any]) -> dict[str, Any]:
    bottleneck = event.get("bottleneck_type")
    taxonomy = {}
    if isinstance(bottleneck, str) and bottleneck:
        taxonomy[bottleneck] = {"triggered": True, "source": "pareto_memory"}
    return taxonomy


def _algorithm_candidate_card(
    candidate_file: Path,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        return None
    hooks = _algorithm_candidate_hypothesis_hooks(candidate)
    if not hooks:
        return None
    candidate_id = str(candidate.get("algorithm_candidate_id", "")).strip()
    if not candidate_id:
        return None
    grounding = candidate.get("source_grounding")
    if not isinstance(grounding, dict):
        grounding = {}
    algorithm_card_id = str(
        grounding.get("algorithm_card_id") or candidate.get("parent_algorithm_card_id")
    ).strip()
    candidate_family = str(candidate.get("candidate_family", "")).strip()
    candidate_key = "|".join(
        [
            algorithm_card_id or "unknown_card",
            candidate_family or "unknown_family",
        ]
    )
    return {
        "card_id": f"algorithm_candidate_{candidate_id}",
        "source_type": "algorithm_candidate",
        "source_path": str(candidate_file),
        "claim": str(candidate.get("hypothesis", "")).strip()
        or "Local algorithm candidate should be tested in Hypothesis Lab.",
        "procedural_heuristic": str(candidate.get("mechanism", "")).strip()
        or "Run the bounded local sandbox before promotion.",
        "preconditions": [
            "Hermes local asset candidate queue selected this strategy",
            str(candidate.get("target_problem", "PDE_Burgers")),
        ],
        "implementation_hooks": hooks,
        "expected_observables": _candidate_string_list(
            candidate.get("expected_observables")
        )
        or ["sandbox score improves"],
        "known_risks": _candidate_string_list(candidate.get("risk_controls"))
        or ["candidate may not transfer"],
        "evidence_status": "candidate_ready",
        "evidence": {
            "algorithm_candidate_id": candidate_id,
            "candidate_key": candidate_key,
            "candidate_family": candidate_family,
            "model_family": candidate.get("model_family"),
            "source_algorithm_card_id": algorithm_card_id,
            "source_grounding": grounding,
        },
    }


def _algorithm_candidate_hypothesis_hooks(candidate: dict[str, Any]) -> list[str]:
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        return []
    hooks = _candidate_hypothesis_hooks(executor.get("implementation_hooks"))
    if hooks:
        return hooks
    family_to_hook = {
        "local_strategy_skill": None,
        "lightweight_residual_corrector": "learned_residual_correction",
        "spectral_rollout_stabilization": "linear_trend_spectral",
    }
    candidate_family = str(candidate.get("candidate_family", "")).strip()
    hook = family_to_hook.get(candidate_family)
    if hook is None:
        return []
    cli_command = str(executor.get("cli_command", "")).strip()
    if cli_command != "run-blueprint-sandbox":
        return []
    return _candidate_hypothesis_hooks([hook])


def _candidate_hypothesis_hooks(value: Any) -> list[str]:
    aliases = {
        "predict-task1-retrieval-prior": "task1_retrieval_prior",
        "hypothesis_lab_front25_dynamics_smoke": "task1_front25_dynamics",
    }
    supported = {
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
        "task1_retrieval_refinement_local_search",
        "task1_early_refinement",
        "task1_early_transition_expert",
        "linear_trend_spectral",
        "linear_trend_spectral_escape",
        "linear_trend_late_spectral_ramp",
        "learned_residual_correction",
        "external_residual_blend",
        "task1_sem_burgers_rollout",
        "task1_physics_residual_refinement",
        "task1_pi_deeponet_correction_head",
        "task1_front25_dynamics",
    }
    hooks = []
    seen: set[str] = set()
    for item in _candidate_string_list(value):
        item = aliases.get(item, item)
        if item not in supported or item in seen:
            continue
        hooks.append(item)
        seen.add(item)
    return hooks


def _candidate_string_list(value: Any) -> list[str]:
    return _string_list_from_any(value)


def _record_algorithm_candidate_outcome(
    *,
    cards: dict[str, Any],
    hypotheses: dict[str, Any],
    executable_blueprint: dict[str, Any],
    sandbox: dict[str, Any],
    failure_memory_output_path: str | Path | None,
    force: bool,
) -> dict[str, Any] | None:
    if failure_memory_output_path is None:
        return None
    evidence = _executed_algorithm_candidate_evidence(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
    )
    if evidence is None:
        return None
    verifier = sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
    decision = str(verifier.get("decision", "unknown"))
    outcome = "passed" if decision == "promote_to_strategy_proposal" else "failed"
    reason_codes = [
        f"hypothesis_lab_decision_{decision}",
        f"score_delta_{float(verifier.get('score_delta_vs_baseline', 0.0)):.6g}",
        f"best_segment_score_{float(verifier.get('best_segment_score', 0.0)):.6g}",
    ]
    output_file = Path(failure_memory_output_path)
    record = record_literature_candidate_outcome(
        algorithm_candidate_id=str(evidence["algorithm_candidate_id"]),
        candidate_key=str(evidence["candidate_key"]),
        outcome=outcome,
        reason_codes=reason_codes,
        output_path=output_file,
        force=force and not output_file.exists(),
    )
    return {
        "algorithm_candidate_id": evidence["algorithm_candidate_id"],
        "candidate_key": evidence["candidate_key"],
        "outcome": outcome,
        "reason_codes": reason_codes,
        "failure_memory_path": str(output_file),
        "outcome_count": len(record.get("outcomes", [])),
    }


def _record_pareto_failure_memory(
    *,
    sandbox: dict[str, Any],
    failure_memory_output_path: str | Path | None,
    algorithm_candidate_evidence: dict[str, Any] | None,
    force: bool,
) -> dict[str, Any] | None:
    if failure_memory_output_path is None:
        return None
    output_file = Path(failure_memory_output_path)
    memory = _load_or_init_pareto_failure_memory(output_file, force=force)
    event = _pareto_learning_event(
        sandbox=sandbox,
        algorithm_candidate_evidence=algorithm_candidate_evidence,
    )
    events = memory.setdefault("pareto_learning_events", [])
    if not isinstance(events, list):
        events = []
        memory["pareto_learning_events"] = events
    events.append(event)
    _update_pareto_learning_summary(memory=memory, event=event)
    _write_json(output_file, memory)
    return {
        "failure_memory_path": str(output_file),
        "event_count": len(events),
        "last_event": event,
    }


def _load_or_init_pareto_failure_memory(path: Path, *, force: bool) -> dict[str, Any]:
    _ = force
    if path.exists():
        memory = _read_json(path)
        if memory.get("state_hint") != FAILURE_MEMORY_STATE_HINT:
            raise HypothesisLabError("failure memory has invalid state_hint")
        return memory
    return {
        "schema_version": 1,
        "state_hint": FAILURE_MEMORY_STATE_HINT,
        "failed_algorithm_candidate_ids": [],
        "failed_candidate_keys": [],
        "failed_candidate_families": [],
        "outcomes": [],
    }


def _pareto_learning_event(
    *,
    sandbox: dict[str, Any],
    algorithm_candidate_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    verifier = sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
    gate = (
        verifier.get("pareto_promotion_gate")
        if isinstance(verifier.get("pareto_promotion_gate"), dict)
        else {}
    )
    diagnosis = (
        verifier.get("early_segment_diagnosis")
        if isinstance(verifier.get("early_segment_diagnosis"), dict)
        else {}
    )
    best = (
        sandbox.get("best_candidate")
        if isinstance(sandbox.get("best_candidate"), dict)
        else {}
    )
    candidate_pool_attribution = (
        sandbox.get("candidate_pool_attribution")
        if isinstance(sandbox.get("candidate_pool_attribution"), dict)
        else {}
    )
    candidate_family = str(sandbox.get("candidate_family", "unknown"))
    decision = str(verifier.get("decision") or gate.get("decision") or "unknown")
    pareto_status = str(gate.get("pareto_status", "unknown"))
    objective_axes = _pareto_memory_axes(gate.get("objective_axes"))
    event = {
        "schema_version": 1,
        "source": "hypothesis_lab_sandbox",
        "sandbox_report_path": str(
            sandbox.get("artifacts", {}).get("sandbox_report_path", "")
        ),
        "blueprint_id": str(sandbox.get("blueprint_id", "unknown")),
        "candidate_family": candidate_family,
        "candidate_id": str(best.get("candidate_id", "unknown")),
        "candidate_params": best.get("params") if isinstance(best.get("params"), dict) else {},
        "algorithm_candidate_id": None,
        "candidate_key": None,
        "decision": decision,
        "pareto_status": pareto_status,
        "outcome": _pareto_memory_outcome(decision),
        "reason_codes": _string_list_from_any(gate.get("reason_codes")),
        "score_delta_vs_baseline": verifier.get("score_delta_vs_baseline"),
        "best_segment_score": verifier.get("best_segment_score"),
        "bottleneck_type": diagnosis.get("bottleneck_type"),
        "diagnosis_reason_codes": _string_list_from_any(diagnosis.get("reason_codes")),
        "action_directives": _pareto_action_directive_names(diagnosis),
        "objective_axes": objective_axes,
        "effective_axes": _pareto_effective_axes(objective_axes),
        "guardrail_regressions": _pareto_guardrail_regressions(objective_axes),
        "candidate_pool_attribution": candidate_pool_attribution,
        "negative_candidate_evidence": _candidate_pool_negative_evidence(
            candidate_pool_attribution
        ),
        "retry_directives": _pareto_retry_directives(
            decision=decision,
            objective_axes=objective_axes,
        ),
        "avoid_directives": _pareto_avoid_directives(
            decision=decision,
            reason_codes=_string_list_from_any(gate.get("reason_codes")),
        ),
    }
    if isinstance(algorithm_candidate_evidence, dict):
        event["algorithm_candidate_id"] = algorithm_candidate_evidence.get(
            "algorithm_candidate_id"
        )
        event["candidate_key"] = algorithm_candidate_evidence.get("candidate_key")
    return event


def _pareto_reason_codes(sandbox: dict[str, Any]) -> list[str]:
    codes = []
    verifier = sandbox.get("verifier")
    if isinstance(verifier, dict):
        rc = verifier.get("reason_codes")
        if isinstance(rc, list):
            codes.extend(str(item) for item in rc if isinstance(item, str))
    return codes


def _candidate_pool_negative_evidence(
    attribution: dict[str, Any],
) -> list[dict[str, Any]]:
    axis = str(attribution.get("axis", ""))
    if axis != "early_anchor":
        return []
    by_anchor = attribution.get("by_early_anchor")
    if not isinstance(by_anchor, dict):
        return []
    evidence = []
    for anchor in _string_list_from_any(attribution.get("non_top_anchors")):
        payload = by_anchor.get(anchor)
        if not isinstance(payload, dict):
            continue
        evidence.append(
            {
                "axis": "early_anchor",
                "value": anchor,
                "reason": "generated_but_absent_from_top_candidates",
                "candidate_count": int(payload.get("candidate_count", 0)),
                "top_candidate_count": int(payload.get("top_candidate_count", 0)),
                "best_early_rel_mse": payload.get("best_early_rel_mse"),
                "best_segment_score": payload.get("best_segment_score"),
            }
        )
    return evidence


def _pareto_memory_axes(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    axes = {}
    for axis_name, axis in value.items():
        if not isinstance(axis, dict):
            continue
        axes[str(axis_name)] = {
            "value": axis.get("value"),
            "threshold": axis.get("threshold"),
            "passed": bool(axis.get("passed")),
            "improved": bool(axis.get("improved")),
            "metric": axis.get("metric"),
            "direction": axis.get("direction"),
        }
    return axes


def _pareto_memory_outcome(decision: str) -> str:
    return {
        "promote_to_strategy_proposal": "promoted",
        "keep_as_evidence": "evidence_only",
        "continue_local_search": "needs_local_search",
        "reject_or_revise": "rejected",
    }.get(decision, "unknown")


def _pareto_action_directive_names(diagnosis: dict[str, Any]) -> list[str]:
    directives = diagnosis.get("action_directives")
    if not isinstance(directives, list):
        return []
    return [
        str(item.get("directive"))
        for item in directives
        if isinstance(item, dict) and isinstance(item.get("directive"), str)
    ]


def _pareto_effective_axes(objective_axes: dict[str, dict[str, Any]]) -> list[str]:
    return [
        str(axis_name)
        for axis_name, axis_data in objective_axes.items()
        if isinstance(axis_data, dict) and axis_data.get("improved")
    ]


def _pareto_guardrail_regressions(
    objective_axes: dict[str, dict[str, Any]],
) -> list[str]:
    regressions = []
    for axis_name in ["middle_score", "late_rmse"]:
        axis = objective_axes.get(axis_name)
        if isinstance(axis, dict) and axis.get("passed") is False:
            regressions.append(axis_name)
    return regressions


def _pareto_retry_directives(
    *,
    decision: str,
    objective_axes: dict[str, dict[str, Any]],
) -> list[str]:
    if decision == "continue_local_search":
        return ["repair_guardrails_before_promotion"]
    if decision == "keep_as_evidence":
        return ["continue_guarded_local_search"]
    if decision == "reject_or_revise":
        return ["revise_candidate_family_or_axis"]
    if (
        decision == "promote_to_strategy_proposal"
        and objective_axes.get("early_rel_mse", {}).get("improved") is True
    ):
        return ["reuse_effective_axes_under_pareto_gate"]
    return []


def _pareto_avoid_directives(*, decision: str, reason_codes: list[str]) -> list[str]:
    directives = []
    if decision == "keep_as_evidence":
        directives.append("do_not_promote_primary_axis_without_total_score")
    if "guardrail_regression" in reason_codes:
        directives.append("do_not_repeat_guardrail_regressing_params")
    if decision == "reject_or_revise":
        directives.append("do_not_repeat_without_new_mechanism")
    return directives


def _update_pareto_learning_summary(
    *,
    memory: dict[str, Any],
    event: dict[str, Any],
) -> None:
    summary = memory.setdefault("pareto_learning_summary", {})
    if not isinstance(summary, dict):
        summary = {}
        memory["pareto_learning_summary"] = summary
    family = str(event["candidate_family"])
    family_summary = summary.setdefault(
        family,
        {
            "promoted_count": 0,
            "evidence_only_count": 0,
            "needs_local_search_count": 0,
            "rejected_count": 0,
            "effective_axes": [],
            "guardrail_regressions": [],
            "last_pareto_status": None,
        },
    )
    outcome = str(event.get("outcome", "unknown"))
    count_key = {
        "promoted": "promoted_count",
        "evidence_only": "evidence_only_count",
        "needs_local_search": "needs_local_search_count",
        "rejected": "rejected_count",
    }.get(outcome)
    if count_key is not None:
        family_summary[count_key] = int(family_summary.get(count_key, 0)) + 1
    for axis in event.get("effective_axes", []):
        if isinstance(axis, str):
            _append_unique(family_summary, "effective_axes", axis)
    for axis in event.get("guardrail_regressions", []):
        if isinstance(axis, str):
            _append_unique(family_summary, "guardrail_regressions", axis)
    for item in event.get("negative_candidate_evidence", []):
        if not isinstance(item, dict):
            continue
        if item.get("axis") == "early_anchor" and isinstance(item.get("value"), str):
            _append_unique(family_summary, "ineffective_anchors", str(item["value"]))
    family_summary["last_pareto_status"] = event.get("pareto_status")


def _append_unique(payload: dict[str, Any], key: str, value: str) -> None:
    items = payload.setdefault(key, [])
    if not isinstance(items, list):
        items = []
        payload[key] = items
    if value not in items:
        items.append(value)


def _record_hypothesis_strategy_growth_artifacts(
    *,
    cards: dict[str, Any],
    hypotheses: dict[str, Any],
    executable_blueprint: dict[str, Any],
    sandbox: dict[str, Any],
    output_dir: Path,
    force: bool,
) -> dict[str, Any] | None:
    verifier = sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
    if verifier.get("decision") != "promote_to_strategy_proposal":
        return None
    card = _executed_algorithm_candidate_card(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
    )
    if card is None:
        return None
    evidence = card["evidence"]
    source_type = (
        "external_retrieval"
        if evidence.get("candidate_family") != "local_strategy_skill"
        else "agent_hypothesis"
    )
    proposal_path = _fresh_output(output_dir / "strategy_proposal.json", force=force)
    evidence_path = _fresh_output(
        output_dir / "strategy_evidence_summary.json",
        force=force,
    )
    strategy_id = str(evidence["algorithm_candidate_id"])
    proposal = record_strategy_proposal(
        strategy_id=strategy_id,
        title=f"Hypothesis Lab verified {strategy_id}",
        source_type=source_type,
        problem_context=(
            "Hypothesis Lab promoted a Hermes algorithm candidate after "
            "bounded 1D Burgers sandbox verification."
        ),
        hypothesis=str(card["claim"]),
        mechanism=str(card["procedural_heuristic"]),
        implementation_plan={
            "allowed_commands": ["run-hypothesis-lab"],
            "expected_artifacts": [
                "hypothesis_lab_report.json",
                "hypothesis_lab_sandbox_report.json",
                "verified_blueprints.json",
            ],
            "algorithm_candidate_id": strategy_id,
            "candidate_key": evidence["candidate_key"],
            "source_grounding": evidence.get("source_grounding", {}),
            "blueprint_id": executable_blueprint.get("blueprint_id"),
            "sandbox_report_path": sandbox["artifacts"]["sandbox_report_path"],
            "verified_blueprint_path": str(output_dir / "verified_blueprints.json"),
        },
        expected_observables=_candidate_string_list(card.get("expected_observables"))
        or ["sandbox score improves"],
        known_risks=_candidate_string_list(card.get("known_risks"))
        or ["single-window sandbox evidence may not generalize"],
        output_path=proposal_path,
    )
    summary = _hypothesis_lab_strategy_evidence_summary(
        strategy_id=strategy_id,
        evidence_path=evidence_path,
        sandbox=sandbox,
    )
    _write_json(evidence_path, summary)
    return {
        "strategy_id": strategy_id,
        "state_hint": proposal["state_hint"],
        "artifacts": {
            "strategy_proposal_path": str(proposal_path),
            "strategy_evidence_summary_path": str(evidence_path),
        },
        "next_command": proposal["next_command"],
    }


def _hypothesis_lab_strategy_evidence_summary(
    *,
    strategy_id: str,
    evidence_path: Path,
    sandbox: dict[str, Any],
) -> dict[str, Any]:
    verifier = sandbox["verifier"]
    score_delta = float(verifier["score_delta_vs_baseline"])
    best = sandbox["best_candidate"]
    return {
        "schema_version": 1,
        "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "source_state_hint": SANDBOX_STATE_HINT,
        "source_sandbox_report_path": sandbox["artifacts"]["sandbox_report_path"],
        "evidence_summary_path": str(evidence_path),
        "best_strategy_id": strategy_id,
        "evaluation_window_count": 1,
        "mean_score_delta": score_delta,
        "score_delta_std": 0.0,
        "long_window_score_delta": score_delta,
        "physics_residual_delta": 0.0,
        "inference_time_regression": float(verifier["inference_time"]),
        "failure_rate": 0.0
        if verifier["decision"] == "promote_to_strategy_proposal"
        else 1.0,
        "candidate_records": [
            {
                "strategy_id": strategy_id,
                "candidate_id": best["candidate_id"],
                "score": best["task1_total_score_proxy"],
                "score_delta": score_delta,
                "segment_score": best["segment_score"],
                "params": best["params"],
            }
        ],
        "provenance_notes": [
            "single_window_hypothesis_lab_sandbox",
            "requires_multi_window_promotion_before_skill_candidate",
        ],
        "next_command": "evaluate-strategy-promotion",
    }


def _executed_algorithm_candidate_evidence(
    *,
    cards: dict[str, Any],
    hypotheses: dict[str, Any],
    executable_blueprint: dict[str, Any],
) -> dict[str, Any] | None:
    card = _executed_algorithm_candidate_card(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
    )
    if not isinstance(card, dict):
        return None
    evidence = card.get("evidence")
    if not isinstance(evidence, dict):
        return None
    return evidence


def _executed_algorithm_candidate_card(
    *,
    cards: dict[str, Any],
    hypotheses: dict[str, Any],
    executable_blueprint: dict[str, Any],
) -> dict[str, Any] | None:
    hypothesis_id = executable_blueprint.get("hypothesis_id")
    source_card_ids: list[str] = []
    for hypothesis in hypotheses.get("hypotheses", []):
        if not isinstance(hypothesis, dict):
            continue
        if hypothesis.get("hypothesis_id") != hypothesis_id:
            continue
        source_card_ids = [
            str(item)
            for item in hypothesis.get("source_card_ids", [])
            if isinstance(item, str)
        ]
        break
    if not source_card_ids:
        return None
    source_ids = set(source_card_ids)
    for card in cards.get("cards", []):
        if not isinstance(card, dict) or card.get("card_id") not in source_ids:
            continue
        evidence = card.get("evidence")
        if not isinstance(evidence, dict):
            continue
        if evidence.get("algorithm_candidate_id") and evidence.get("candidate_key"):
            return card
    return None


def _fresh_output(path: Path, *, force: bool) -> Path:
    if path.exists() and force:
        path.unlink()
    return path
