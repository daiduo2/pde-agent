"""Failure taxonomy for hypothesis-lab sandbox post-mortem diagnosis.

Provides a structured classification of failure modes into four categories:
PHYSICAL, NUMERICAL, STATISTICAL, and OPERATIONAL.
"""

from __future__ import annotations

from typing import Any


FAILURE_TYPES: dict[str, dict[str, str]] = {
    "PHYSICAL": {
        "shock_capturing_failure": "Discontinuity smearing or overshoot",
        "energy_dissipation_anomaly": "Unphysical energy loss/gain",
        "boundary_condition_violation": "BC not respected",
        "wave_propagation_error": "Phase or group velocity wrong",
    },
    "NUMERICAL": {
        "instability_amplification": "Growing oscillations",
        "time_step_sensitivity": "Results change drastically with dt",
        "spatial_aliasing": "Grid too coarse for features",
        "solver_convergence_failure": "Nonlinear solver did not converge",
    },
    "STATISTICAL": {
        "distribution_shift": "Train/test mismatch",
        "covariate_mismatch": "Input features unlike training",
        "overfitting_artifact": "Memorization pattern visible",
        "sampling_bias": "Systematic sample selection issue",
    },
    "OPERATIONAL": {
        "timeout_risk": "Inference near or over budget",
        "memory_exhaustion": "OOM during rollout",
        "context_corruption": "Context prefix altered",
        "reference_mismatch": "Retrieval reference incompatible",
        "solver_convergence_failure": "Nonlinear solver did not converge",
    },
}


ROOT_CAUSE_TO_TYPE: dict[str, tuple[str, str]] = {
    "early_shock_smearing": ("PHYSICAL", "shock_capturing_failure"),
    "late_drift": ("PHYSICAL", "energy_dissipation_anomaly"),
    "context_corrupted": ("OPERATIONAL", "context_corruption"),
    "reference_mismatch": ("OPERATIONAL", "reference_mismatch"),
    "numerical_instability": ("NUMERICAL", "instability_amplification"),
    "timeout_risk": ("OPERATIONAL", "timeout_risk"),
    "early_score_low": ("STATISTICAL", "distribution_shift"),
    "early_rel_mse_high": ("STATISTICAL", "covariate_mismatch"),
    "early_weaker_than_middle": ("PHYSICAL", "wave_propagation_error"),
    "early_weaker_than_late": ("PHYSICAL", "energy_dissipation_anomaly"),
    "initial_derivative_mismatch": ("NUMERICAL", "time_step_sensitivity"),
    "phase_shift": ("PHYSICAL", "wave_propagation_error"),
    "amplitude_bias": ("STATISTICAL", "overfitting_artifact"),
    "over_smoothing": ("PHYSICAL", "shock_capturing_failure"),
    "high_frequency_noise": ("NUMERICAL", "spatial_aliasing"),
    "guardrail_regression": ("OPERATIONAL", "reference_mismatch"),
    "physics_residual_not_improved": ("PHYSICAL", "energy_dissipation_anomaly"),
    "pi_deeponet_correction_inactive": ("OPERATIONAL", "solver_convergence_failure"),
    "no_pareto_axis_improved": ("STATISTICAL", "sampling_bias"),
    "rules_passed_failed": ("OPERATIONAL", "context_corruption"),
    "context_preserved_failed": ("OPERATIONAL", "context_corruption"),
    "inference_time_budget_failed": ("OPERATIONAL", "timeout_risk"),
}


def classify_failure(sandbox_result: dict[str, Any]) -> dict[str, Any]:
    """Classify a sandbox result into the failure taxonomy.

    Returns a dict with:
        - primary_type: one of PHYSICAL, NUMERICAL, STATISTICAL, OPERATIONAL
        - primary_subtype: specific subtype key
        - root_cause_hint: human-readable root cause string
        - confidence: 'high' | 'medium' | 'low'
    """
    verifier = sandbox_result.get("verifier", {})
    decision = verifier.get("decision", "")
    metric_results = verifier.get("metric_results", {})
    early_diag = verifier.get("early_segment_diagnosis", {})
    reason_codes = early_diag.get("reason_codes", [])
    pareto_gate = verifier.get("pareto_promotion_gate", {})
    gate_reasons = pareto_gate.get("reason_codes", [])

    # Gather all reason codes
    all_reasons: list[str] = []
    all_reasons.extend(reason_codes)
    all_reasons.extend(gate_reasons)

    # Check hard constraint failures
    for name in ["rules_passed", "context_preserved", "inference_time_budget"]:
        mr = metric_results.get(name, {})
        if mr.get("passed") is False:
            all_reasons.append(f"{name}_failed")

    # If promoted, it's not a failure
    if decision == "promote_to_strategy_proposal":
        return {
            "primary_type": "NONE",
            "primary_subtype": "ok",
            "root_cause_hint": "ok",
            "confidence": "high",
        }

    # Score-based heuristics for root cause
    root_cause_hint = _infer_root_cause_hint_from_reasons(all_reasons)
    if root_cause_hint == "unknown" and sandbox_result.get("post_mortem"):
        pm = sandbox_result["post_mortem"]
        root_cause_hint = pm.get("root_cause_hint", "unknown")

    # Map to taxonomy
    mapped = ROOT_CAUSE_TO_TYPE.get(root_cause_hint)
    if mapped is None:
        # Fallback: try to map from reason codes directly
        for reason in all_reasons:
            mapped = ROOT_CAUSE_TO_TYPE.get(reason)
            if mapped is not None:
                root_cause_hint = reason
                break

    # Normalize synthetic reason codes to canonical root causes
    if root_cause_hint == "inference_time_budget_failed":
        mapped = ROOT_CAUSE_TO_TYPE.get("timeout_risk")
        root_cause_hint = "timeout_risk"
    elif root_cause_hint == "context_preserved_failed":
        mapped = ROOT_CAUSE_TO_TYPE.get("context_corrupted")
        root_cause_hint = "context_corrupted"
    elif root_cause_hint == "rules_passed_failed":
        mapped = ROOT_CAUSE_TO_TYPE.get("reference_mismatch")
        root_cause_hint = "reference_mismatch"

    if mapped is None:
        # Final fallback based on decision
        if decision in {"reject_or_revise", "dominated_or_invalid"}:
            mapped = ("STATISTICAL", "sampling_bias")
            root_cause_hint = "no_pareto_axis_improved"
        else:
            mapped = ("OPERATIONAL", "reference_mismatch")
            root_cause_hint = "reference_mismatch"

    primary_type, primary_subtype = mapped

    # Confidence based on evidence richness
    confidence = "low"
    if len(all_reasons) >= 2:
        confidence = "high"
    elif len(all_reasons) == 1:
        confidence = "medium"

    return {
        "primary_type": primary_type,
        "primary_subtype": primary_subtype,
        "root_cause_hint": root_cause_hint,
        "confidence": confidence,
    }


def diagnose_session_family(
    session: Any,
    family: str,
) -> dict[str, Any]:
    """Diagnose a family across all sandbox artifacts in a session.

    Returns aggregate failure taxonomy counts and root causes.
    """
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    if not isinstance(session, HypothesisLabSession):
        return {
            "family": family,
            "failure_types": {},
            "root_causes": [],
            "sandbox_count": 0,
        }

    failure_types: dict[str, int] = {}
    root_causes: list[str] = []
    sandbox_count = 0

    for artifact in session.list_artifacts("sandbox"):
        payload = artifact.get("payload", {})
        best = payload.get("best_candidate", {})
        if best.get("candidate_family") != family:
            continue
        sandbox_count += 1
        classification = classify_failure(payload)
        ptype = classification["primary_type"]
        if ptype != "NONE":
            failure_types[ptype] = failure_types.get(ptype, 0) + 1
        rch = classification["root_cause_hint"]
        if rch and rch != "ok" and rch not in root_causes:
            root_causes.append(rch)

    return {
        "family": family,
        "failure_types": failure_types,
        "root_causes": root_causes,
        "sandbox_count": sandbox_count,
    }


def _infer_root_cause_hint_from_reasons(reasons: list[str]) -> str:
    """Pick the most informative root cause from a list of reason codes."""
    if not reasons:
        return "unknown"
    priority_order = [
        "context_corrupted",
        "reference_mismatch",
        "early_shock_smearing",
        "late_drift",
        "numerical_instability",
        "timeout_risk",
        "inference_time_budget_failed",
        "initial_derivative_mismatch",
        "phase_shift",
        "amplitude_bias",
        "over_smoothing",
        "high_frequency_noise",
        "early_score_low",
        "early_rel_mse_high",
        "early_weaker_than_middle",
        "early_weaker_than_late",
        "guardrail_regression",
        "physics_residual_not_improved",
        "pi_deeponet_correction_inactive",
        "no_pareto_axis_improved",
        "rules_passed_failed",
        "context_preserved_failed",
    ]

    for candidate in priority_order:
        if candidate in reasons:
            return candidate
    return reasons[0]
