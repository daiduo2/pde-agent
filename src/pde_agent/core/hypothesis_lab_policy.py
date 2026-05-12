"""Evidence-to-policy bridge for Hypothesis Lab agent loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HYPOTHESIS_LAB_NEXT_POLICY_BUILT"


class HypothesisLabPolicyError(RuntimeError):
    pass


def build_hypothesis_lab_next_policy(
    *,
    output_path: str | Path,
    failure_memory_path: str | Path | None = None,
    hypothesis_lab_report_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise HypothesisLabPolicyError(
            f"hypothesis lab next policy already exists: {output_file}"
        )
    event = _latest_event(
        failure_memory_path=Path(failure_memory_path)
        if failure_memory_path is not None
        else None,
        hypothesis_lab_report_path=Path(hypothesis_lab_report_path)
        if hypothesis_lab_report_path is not None
        else None,
    )
    policy = _policy_from_event(event)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "policy_path": str(output_file),
        "inputs": {
            "failure_memory_path": str(failure_memory_path)
            if failure_memory_path is not None
            else None,
            "hypothesis_lab_report_path": str(hypothesis_lab_report_path)
            if hypothesis_lab_report_path is not None
            else None,
        },
        "source_event": event,
        "policy": policy,
        "next_command": policy["next_command"],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _latest_event(
    *,
    failure_memory_path: Path | None,
    hypothesis_lab_report_path: Path | None,
) -> dict[str, Any]:
    if failure_memory_path is not None:
        memory = _read_json(_require_file(failure_memory_path, "failure memory"))
        events = [
            item
            for item in memory.get("pareto_learning_events", [])
            if isinstance(item, dict)
        ]
        if events:
            return dict(events[-1])
        outcomes = [
            item for item in memory.get("outcomes", []) if isinstance(item, dict)
        ]
        if outcomes:
            return {
                "source": "failure_memory_outcome",
                "outcome": outcomes[-1].get("outcome"),
                "algorithm_candidate_id": outcomes[-1].get("algorithm_candidate_id"),
                "candidate_key": outcomes[-1].get("candidate_key"),
                "reason_codes": outcomes[-1].get("reason_codes", []),
            }
    if hypothesis_lab_report_path is not None:
        report = _read_json(_require_file(hypothesis_lab_report_path, "hypothesis lab report"))
        pareto = report.get("pareto_memory")
        if isinstance(pareto, dict) and isinstance(pareto.get("last_event"), dict):
            return dict(pareto["last_event"])
        sandbox = report.get("sandbox")
        if isinstance(sandbox, dict):
            verifier = sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
            return {
                "source": "hypothesis_lab_report",
                "candidate_family": sandbox.get("candidate_family"),
                "candidate_id": (sandbox.get("best_candidate") or {}).get("candidate_id")
                if isinstance(sandbox.get("best_candidate"), dict)
                else None,
                "decision": verifier.get("decision"),
                "score_delta_vs_baseline": verifier.get("score_delta_vs_baseline"),
                "best_segment_score": verifier.get("best_segment_score"),
                "bottleneck_type": (
                    verifier.get("early_segment_diagnosis") or {}
                ).get("bottleneck_type")
                if isinstance(verifier.get("early_segment_diagnosis"), dict)
                else None,
            }
    raise HypothesisLabPolicyError(
        "no usable failure memory or hypothesis lab report evidence found"
    )


def _policy_from_event(event: dict[str, Any]) -> dict[str, Any]:
    decision = str(event.get("decision") or "")
    outcome = str(event.get("outcome") or "")
    family = str(event.get("candidate_family") or "")
    bottleneck = str(event.get("bottleneck_type") or "")
    action_directives = _strings(event.get("action_directives"))
    retry_directives = _strings(event.get("retry_directives"))
    avoid_directives = _strings(event.get("avoid_directives"))
    reason_codes = _strings(event.get("reason_codes"))
    if decision == "promote_to_strategy_proposal" or outcome == "promoted":
        next_action = "promote_or_accumulate_evidence"
        next_command = "evaluate-strategy-promotion"
        candidate_generator = _candidate_generator_for_family(family)
    elif "do_not_repeat_without_new_mechanism" in avoid_directives:
        next_action = "expand_new_mechanism"
        next_command = "generate-task1-candidate-families"
        candidate_generator = "auto"
    elif (
        "continue_guarded_local_search" in retry_directives
        or "revise_candidate_family_or_axis" in retry_directives
    ):
        next_action = "retry_with_policy_constraints"
        next_command = "plan-hypothesis-candidates"
        candidate_generator = _candidate_generator_for_family(family)
    else:
        next_action = "collect_more_evidence"
        next_command = "run-hypothesis-lab"
        candidate_generator = _candidate_generator_for_family(family)
    return {
        "next_action": next_action,
        "next_command": next_command,
        "candidate_family": family,
        "candidate_generator": candidate_generator,
        "learning_objective": _learning_objective(bottleneck=bottleneck),
        "required_constraints": {
            "avoid_algorithm_candidate_ids": _optional_list(
                event.get("algorithm_candidate_id")
            ),
            "avoid_candidate_keys": _optional_list(event.get("candidate_key")),
            "avoid_directives": avoid_directives,
            "retry_directives": retry_directives,
            "action_directives": action_directives,
            "reason_codes": reason_codes,
        },
        "preferred_error_modes": _preferred_error_modes(bottleneck=bottleneck),
        "hypothesis_candidate_options": _candidate_options_from_event(event),
        "promotion_gate": {
            "min_score_delta": 0.01,
            "require_context_preserved": True,
            "require_middle_late_guardrails": True,
        },
    }


def _candidate_generator_for_family(family: str) -> str:
    mapping = {
        "task1_front25_dynamics": "hypothesis_lab_front25_dynamics_smoke",
        "task1_early_refinement": "hypothesis_lab_early_subsegment_patch_search",
        "task1_early_transition_expert": "hypothesis_lab_early_subsegment_patch_search",
        "task1_retrieval_refinement": "hypothesis_lab_early_subsegment_patch_search",
        "linear_trend_spectral": "sem_window_blend",
    }
    return mapping.get(family, "auto")


def _learning_objective(*, bottleneck: str) -> str:
    if bottleneck in {"initial_derivative_mismatch", "phase_shift"}:
        return "early_segment"
    if bottleneck in {"high_frequency_noise", "over_smoothing"}:
        return "physics_and_spectral_guarded_score"
    return "score"


def _preferred_error_modes(*, bottleneck: str) -> list[str]:
    modes = []
    if bottleneck:
        modes.append(bottleneck)
    if bottleneck in {"initial_derivative_mismatch", "phase_shift"}:
        modes.extend(["early_segment", "transition_window"])
    return _dedupe(modes)


def _candidate_options_from_event(event: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    directives = _strings(event.get("action_directives"))
    if directives:
        options["action_directives"] = directives
    effective_axes = _strings(event.get("effective_axes"))
    if effective_axes:
        options["focus_axes"] = effective_axes
    attribution = event.get("candidate_pool_attribution")
    if isinstance(attribution, dict):
        axis = attribution.get("axis")
        if isinstance(axis, str) and axis:
            options["candidate_pool_axis"] = axis
    params = event.get("candidate_params")
    if isinstance(params, dict) and params:
        options["anchor_params"] = params
    return options


def _optional_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) and value else []


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise HypothesisLabPolicyError(f"{path} must contain a JSON object")
    return payload


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise HypothesisLabPolicyError(f"{label} not found: {path}")
    return path
