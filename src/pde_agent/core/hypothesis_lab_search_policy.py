from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_utils import (
    HypothesisLabError,
    SANDBOX_STATE_HINT,
    _string_list_from_any,
)


def _lazy_normalize_evidence_params(params: dict[str, Any]) -> dict[str, Any] | None:
    # Lazy import to avoid circular dependency with hypothesis_lab.py
    from pde_agent.core.hypothesis_lab import _normalize_evidence_params

    return _normalize_evidence_params(params)


def _adaptive_plateau_card(
    report_file: Path,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    if report.get("state_hint") != "ADAPTIVE_HEARTBEAT_RAN":
        return None
    events = [item for item in report.get("policy_events", []) if isinstance(item, dict)]
    plateau_events = [
        item
        for item in events
        if item.get("trajectory_state") == "plateau"
        or item.get("action") == "verified_blueprint_plateau"
    ]
    if not plateau_events:
        return None
    last = plateau_events[-1]
    best_candidate_id = str(last.get("best_candidate_id", "unknown"))
    reason_codes = [
        str(item)
        for event in plateau_events
        for item in event.get("reason_codes", [])
        if isinstance(item, str)
    ]
    if last.get("action") == "verified_blueprint_plateau":
        reason_codes.append("verified_blueprint_plateau")
    escaped_once = "s1.5" in best_candidate_id or "spectral_strength': 1.5" in str(
        last.get("metadata", "")
    )
    return {
        "card_id": f"adaptive_plateau_{report_file.stem}",
        "source_type": "adaptive_heartbeat_report",
        "source_path": str(report_file),
        "claim": (
            "The verified linear trend spectral blueprint reached a local "
            "plateau and should trigger a wider controlled escape search."
        ),
        "procedural_heuristic": (
            "Keep the controlled NumPy operator, but expand slope windows, "
            "weaken the gain, and include unsmoothed plus stronger spectral "
            "settings instead of repeating the verified seed."
        ),
        "preconditions": [
            "adaptive heartbeat plateau",
            "verified blueprint reuse already attempted",
            "candidate generator is linear_trend_spectral",
        ],
        "implementation_hooks": [
            "linear_trend_late_spectral_ramp"
            if escaped_once
            else "linear_trend_spectral_escape"
        ],
        "expected_observables": [
            "new top candidate outside previous seed parameters",
            "non-negative score delta versus persistence",
            "runtime remains below inference budget",
        ],
        "known_risks": sorted(set(reason_codes)),
        "evidence_status": "plateaued",
        "evidence": {
            "candidate_id": best_candidate_id,
            "task1_total_fast_proxy": last.get("best_score"),
            "action": last.get("action"),
            "stage_index": last.get("stage_index"),
            "plateau_escape_already_used": escaped_once,
        },
    }


def _adaptive_hypothesis_lab_search_policy_card(
    report_file: Path,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    if report.get("state_hint") != "ADAPTIVE_HEARTBEAT_RAN":
        return None
    policy_bundle = _adaptive_hypothesis_lab_search_policy(report)
    if policy_bundle is None:
        return None
    policy = policy_bundle["policy"]
    hook = _adaptive_search_policy_hook(policy)
    if hook is None:
        return None
    params = _adaptive_best_candidate_params(report)
    normalized = _lazy_normalize_evidence_params(params) if isinstance(params, dict) else None
    if normalized is None:
        return None
    event = _last_hypothesis_lab_policy_event(report)
    reason_codes = []
    if isinstance(event, dict):
        reason_codes.extend(
            str(item) for item in event.get("reason_codes", []) if isinstance(item, str)
        )
    reason_codes.extend(
        str(action.get("reason"))
        for action in policy.get("recommended_actions", [])
        if isinstance(action, dict) and isinstance(action.get("reason"), str)
    )
    return {
        "card_id": f"adaptive_hypothesis_lab_search_policy_{report_file.stem}",
        "source_type": "adaptive_hypothesis_lab_search_policy",
        "source_path": str(report_file),
        "claim": (
            "Hermes plateaued after a Hypothesis Lab run, but the sandbox "
            "attribution identified which early anchors should stay in focus, "
            "which deserve rescue probes, and which should be deprioritized."
        ),
        "procedural_heuristic": (
            "Keep the Task 1 early-refinement executor centered on the latest "
            "Hypothesis Lab candidate, then rank the next grid with the recorded "
            "focus/rescue/deprioritize policy instead of restarting an unweighted "
            "anchor sweep."
        ),
        "preconditions": [
            "adaptive heartbeat plateau",
            "previous candidate generator was hypothesis_lab",
            "candidate pool attribution includes a search-space policy",
        ],
        "implementation_hooks": [hook],
        "expected_observables": [
            "focus values remain inside the next candidate budget",
            "deprioritized values are searched later unless they are rescue probes",
            "Hermes avoids repeating a uniform axis sweep",
        ],
        "known_risks": sorted(set(reason_codes)),
        "evidence_status": "search_policy_ready",
        "evidence": {
            "candidate_id": _adaptive_best_candidate_id(report),
            "params": normalized,
            "candidate_pool_attribution": {"search_space_policy": policy},
            "search_directives": _search_directives_from_candidate_pool_policy(policy),
            "source_hypothesis_lab_report_path": policy_bundle.get("source_path"),
            "task1_total_fast_proxy": _adaptive_best_score(report, event),
            "task1_total_score_proxy": _adaptive_best_score(report, event),
        },
    }


def _adaptive_search_policy_hook(policy: dict[str, Any]) -> str | None:
    axis = str(policy.get("axis", ""))
    if axis == "early_anchor":
        return "task1_early_refinement"
    if axis in {
        "k",
        "mode",
        "early_blend_steps",
        "persistence_weight_start",
        "spectral_strength",
    }:
        return "task1_retrieval_refinement_local_search"
    return None


def _search_directives_from_candidate_pool_policy(
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    axis = str(policy.get("axis", ""))
    if axis != "mode":
        return []
    focus_values = _string_list_from_any(policy.get("focus_values"))
    deprioritize_values = _string_list_from_any(policy.get("deprioritize_values"))
    if not focus_values and not deprioritize_values:
        return []
    return [
        {
            "axis": "mode",
            "direction": "focus",
            "values": focus_values,
            "avoid_values": deprioritize_values,
            "reason": "candidate_pool_search_policy",
        }
    ]


def _adaptive_hypothesis_lab_search_policy(
    report: dict[str, Any],
) -> dict[str, Any] | None:
    inputs = report.get("next_hypothesis_lab_inputs")
    if isinstance(inputs, dict):
        bundle = inputs.get("search_space_policy")
        if isinstance(bundle, dict) and isinstance(bundle.get("policy"), dict):
            return bundle
    event = _last_hypothesis_lab_policy_event(report)
    if isinstance(event, dict):
        bundle = event.get("hypothesis_lab_search_policy")
        if isinstance(bundle, dict) and isinstance(bundle.get("policy"), dict):
            return bundle
    return None


def _last_hypothesis_lab_policy_event(report: dict[str, Any]) -> dict[str, Any] | None:
    events = report.get("policy_events")
    if not isinstance(events, list):
        return None
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("hypothesis_lab_search_policy"), dict):
            return event
    return None


def _adaptive_best_candidate_params(report: dict[str, Any]) -> dict[str, Any] | None:
    candidate = report.get("best_candidate")
    if not isinstance(candidate, dict):
        return None
    params = candidate.get("params")
    if isinstance(params, dict):
        return params
    metadata = candidate.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("params"), dict):
        return metadata["params"]
    return None


def _adaptive_best_candidate_id(report: dict[str, Any]) -> str:
    candidate = report.get("best_candidate")
    if isinstance(candidate, dict) and candidate.get("candidate_id"):
        return str(candidate["candidate_id"])
    event = _last_hypothesis_lab_policy_event(report)
    if isinstance(event, dict) and event.get("best_candidate_id"):
        return str(event["best_candidate_id"])
    return "unknown"


def _adaptive_best_score(
    report: dict[str, Any],
    event: dict[str, Any] | None,
) -> float | None:
    if isinstance(event, dict) and isinstance(event.get("best_score"), int | float):
        return float(event["best_score"])
    candidate = report.get("best_candidate")
    if isinstance(candidate, dict) and isinstance(candidate.get("score"), int | float):
        return float(candidate["score"])
    if isinstance(report.get("best_score"), int | float):
        return float(report["best_score"])
    return None


def _adaptive_retrieval_refinement_card(
    report_file: Path,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    if report.get("state_hint") != SANDBOX_STATE_HINT:
        return None
    family = str(report.get("candidate_family", ""))
    blueprint_id = str(report.get("blueprint_id", ""))
    if (
        family
        not in {
            "task1_early_refinement",
            "task1_retrieval_refinement",
            "task1_retrieval_refinement_local_search",
        }
        and "task1_retrieval_refinement" not in blueprint_id
        and "task1_early_refinement" not in blueprint_id
    ):
        return None
    best = report.get("best_candidate")
    if not isinstance(best, dict):
        return None
    best_params = best.get("params")
    if not isinstance(best_params, dict):
        return None
    normalized = _lazy_normalize_evidence_params(best_params)
    if normalized is None or "k" not in normalized:
        return None
    top_params = []
    top_evidence = []
    seen = {json.dumps(normalized, sort_keys=True)}
    best_score = _candidate_score(best)
    scored_candidates = [
        {
            "candidate_id": best.get("candidate_id"),
            "params": normalized,
            "score": best_score,
            "segment_score": best.get("segment_score"),
            "task1_total_score_proxy": best.get("task1_total_score_proxy"),
        }
    ]
    top_candidates = report.get("top_candidates")
    if isinstance(top_candidates, list):
        for candidate in top_candidates:
            if not isinstance(candidate, dict):
                continue
            params = candidate.get("params")
            if not isinstance(params, dict):
                continue
            normalized_top = _lazy_normalize_evidence_params(params)
            if normalized_top is None or "k" not in normalized_top:
                continue
            key = json.dumps(normalized_top, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            top_params.append(normalized_top)
            score = _candidate_score(candidate)
            top_evidence.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "params": normalized_top,
                    "score": score,
                    "segment_score": candidate.get("segment_score"),
                    "task1_total_score_proxy": candidate.get(
                        "task1_total_score_proxy"
                    ),
                }
            )
            scored_candidates.append(top_evidence[-1])
    search_directives = _retrieval_refinement_search_directives(scored_candidates)
    verifier = report.get("verifier")
    early_segment_diagnosis = None
    if isinstance(verifier, dict) and isinstance(
        verifier.get("early_segment_diagnosis"), dict
    ):
        early_segment_diagnosis = verifier["early_segment_diagnosis"]
    hooks = ["task1_retrieval_refinement_local_search"]
    if (
        isinstance(early_segment_diagnosis, dict)
        and early_segment_diagnosis.get("decision") == "early_segment_bottleneck"
    ):
        hooks.insert(0, "task1_early_refinement")
    return {
        "card_id": f"adaptive_retrieval_refinement_{report_file.stem}",
        "source_type": "adaptive_retrieval_refinement_sandbox",
        "source_path": str(report_file),
        "claim": (
            "A verified retrieval-refinement sandbox found a stronger local "
            "center, so Hermes should move the next local search around the "
            "observed best parameters instead of repeating a fixed hand-written grid."
        ),
        "procedural_heuristic": (
            "Use the sandbox best candidate as the local-search center, then "
            "perturb neighbor count, early blend length, persistence weight, "
            "and spectral strength under the same controlled NumPy executor."
        ),
        "preconditions": [
            "verified retrieval-refinement sandbox report exists",
            "best candidate preserves the first 10 context steps",
            "next iteration can reuse the same validation target as a sandbox",
        ],
        "implementation_hooks": hooks,
        "expected_observables": [
            "candidate grid is centered on the latest sandbox best parameters",
            "new candidates include immediate score-neighborhood perturbations",
            "Hermes does not repeat only the original hand-written center",
        ],
        "known_risks": [
            "validation retrieval can overfit",
            "too narrow a local search can miss a different basin",
        ],
        "evidence_status": "local_search_center_ready",
        "evidence": {
            "candidate_id": best.get("candidate_id"),
            "params": normalized,
            "top_params": top_params,
            "top_candidates": top_evidence,
            "search_directives": search_directives,
            "early_segment_diagnosis": early_segment_diagnosis,
            "task1_total_score_proxy": best.get("task1_total_score_proxy"),
            "task1_total_fast_proxy": best.get("task1_total_fast_proxy"),
            "segment_score": best.get("segment_score"),
            "blueprint_id": blueprint_id,
        },
    }


def _candidate_score(candidate: dict[str, Any]) -> float | None:
    for key in ["task1_total_score_proxy", "task1_total_fast_proxy", "segment_score"]:
        value = candidate.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _retrieval_refinement_search_directives(
    scored_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    directives = []
    axes = [
        "k",
        "early_blend_steps",
        "persistence_weight_start",
        "spectral_strength",
    ]
    rows = [
        item
        for item in scored_candidates
        if isinstance(item.get("params"), dict)
        and isinstance(item.get("score"), int | float)
    ]
    for axis in axes:
        axis_rows = [
            (float(item["params"][axis]), float(item["score"]))
            for item in rows
            if axis in item["params"]
        ]
        distinct_values = sorted({value for value, _score in axis_rows})
        if len(distinct_values) < 3:
            continue
        low_value = distinct_values[0]
        high_value = distinct_values[-1]
        low_score = max(score for value, score in axis_rows if value == low_value)
        high_score = max(score for value, score in axis_rows if value == high_value)
        if high_score > low_score:
            directives.append(
                {
                    "axis": axis,
                    "direction": "increase",
                    "reason": "top_candidate_score_trend",
                }
            )
        elif low_score > high_score:
            directives.append(
                {
                    "axis": axis,
                    "direction": "decrease",
                    "reason": "top_candidate_score_trend",
                }
            )
    return directives


def _scientific_log_card(log_file: Path) -> dict[str, Any] | None:
    if not log_file.exists():
        raise HypothesisLabError(f"log artifact not found: {log_file}")
    text = log_file.read_text(encoding="utf-8")
    lower = text.lower()
    if "verified_blueprint_plateau" not in lower and "plateaued" not in lower:
        return None
    if "linear_trend_spectral" not in lower and "linear trend spectral" not in lower:
        return None
    candidate_id = _first_log_best_candidate(text)
    late_ramp_seen = "_ls" in lower and "_lt" in lower
    return {
        "card_id": f"scientific_log_{log_file.stem}",
        "source_type": "scientific_log",
        "source_path": str(log_file),
        "claim": (
            "Verified blueprint reuse plateaued in the scientific log, so "
            "Hermes should explore a controlled linear trend spectral escape."
        ),
        "procedural_heuristic": (
            "Convert the log reflection into a structured memory card before "
            "choosing the next hypothesis, so the next batch does not repeat "
            "the same verified blueprint."
        ),
        "preconditions": [
            "task1 agent log contains plateau evidence",
            "linear trend spectral strategy has already been reused",
        ],
        "implementation_hooks": [
            "linear_trend_late_spectral_ramp"
            if late_ramp_seen
            else "linear_trend_spectral_escape"
        ],
        "expected_observables": [
            "hypothesis generator prioritizes plateau_escape",
            "knowledge card is visible in hypothesis sources",
        ],
        "known_risks": ["verified_blueprint_plateau", "repeating prior seed"],
        "evidence_status": "observed_plateau",
        "evidence": {
            "candidate_id": candidate_id,
            "summary": "adaptive policy log reported verified blueprint plateau",
            "late_spectral_ramp_seen": late_ramp_seen,
        },
    }


def _first_log_best_candidate(text: str) -> str | None:
    for line in text.splitlines():
        marker = "candidate="
        if marker not in line:
            continue
        return line.split(marker, 1)[1].split()[0].strip()
    for line in text.splitlines():
        marker = "best="
        if marker not in line:
            continue
        return line.split(marker, 1)[1].split()[0].strip()
    return None
