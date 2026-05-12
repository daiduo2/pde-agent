"""Candidate parameter construction for Hypothesis Lab blueprints."""

from typing import Any

from pde_agent.core.hypothesis_lab_utils import (
    EARLY_REFINEMENT_ANCHORS,
    HypothesisLabError,
    _clamp_float,
    _clamp_int,
    _clamp_task1_retrieval_k,
    _normalize_retrieval_refinement_params,
)
from pde_agent.core.hypothesis_lab_candidate_pool import (
    _apply_early_memory_policy,
    _dedupe_values,
    _directed_retrieval_refinement_params,
    _early_memory_policy,
    _early_refinement_signals,
    _filter_retrieval_refinement_params_by_directives,
    _retrieval_refinement_center,
    _retrieval_refinement_search_directives_from_evidence,
)
from pde_agent.core.task1_early_transition_expert import (
    early_transition_expert_params_from_any,
)
from pde_agent.core.task1_front25_dynamics import front25_dynamics_params_from_any


def _early_transition_expert_parameter_grid(
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    candidate_params = grid.get("candidate_params")
    if isinstance(candidate_params, list):
        rows = []
        for row in candidate_params:
            if not isinstance(row, dict):
                raise HypothesisLabError(
                    "early transition expert candidate_params entries must be objects"
                )
            rows.append(_normalize_early_transition_expert_params(row))
        if not rows:
            raise HypothesisLabError(
                "early transition expert candidate_params must not be empty"
            )
        return rows
    required = {
        "k": grid.get("k"),
        "horizon_steps": grid.get("horizon_steps", [47]),
        "ridge_alpha": grid.get("ridge_alpha", [1e-3]),
        "correction_weight": grid.get("correction_weight"),
        "transition_mode": grid.get("transition_mode", ["velocity_ridge"]),
    }
    for key, value in required.items():
        if not isinstance(value, list) or not value:
            raise HypothesisLabError(
                f"early transition expert blueprint missing {key} list"
            )
    rows = []
    for k in required["k"]:
        for horizon_steps in required["horizon_steps"]:
            for ridge_alpha in required["ridge_alpha"]:
                for correction_weight in required["correction_weight"]:
                    for transition_mode in required["transition_mode"]:
                        rows.append(
                            _normalize_early_transition_expert_params(
                                {
                                    "k": k,
                                    "mode": grid.get("mode", ["affine"])[0],
                                    "early_blend_steps": grid.get(
                                        "early_blend_steps",
                                        [0],
                                    )[0],
                                    "persistence_weight_start": grid.get(
                                        "persistence_weight_start",
                                        [0.0],
                                    )[0],
                                    "spectral_strength": grid.get(
                                        "spectral_strength",
                                        [0.0],
                                    )[0],
                                    "horizon_steps": horizon_steps,
                                    "ridge_alpha": ridge_alpha,
                                    "correction_weight": correction_weight,
                                    "transition_mode": transition_mode,
                                }
                            )
                        )
    return rows


def _default_retrieval_refinement_params() -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = [
        {
            "k": 20,
            "mode": "affine",
            "early_blend_steps": 0,
            "persistence_weight_start": 0.0,
            "spectral_strength": 0.0,
        },
        {
            "k": 48,
            "mode": "affine",
            "early_blend_steps": 0,
            "persistence_weight_start": 0.0,
            "spectral_strength": 0.0,
        },
    ]
    for k in [32, 48]:
        for early_steps in [20, 40]:
            for weight in [0.25, 0.5]:
                for strength in [0.5, 1.0]:
                    params.append(
                        {
                            "k": k,
                            "mode": "affine",
                            "early_blend_steps": early_steps,
                            "persistence_weight_start": weight,
                            "spectral_strength": strength,
                        }
                    )
    params.extend(
        [
            {
                "k": 48,
                "mode": "delta",
                "early_blend_steps": 20,
                "persistence_weight_start": 0.5,
                "spectral_strength": 0.5,
            },
            {
                "k": 48,
                "mode": "raw",
                "early_blend_steps": 20,
                "persistence_weight_start": 0.5,
                "spectral_strength": 0.5,
            },
        ]
    )
    return params


def _local_retrieval_refinement_params(
    evidence_params: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    center = _retrieval_refinement_center(evidence_params)
    directives = _retrieval_refinement_search_directives_from_evidence(
        evidence_params
    )
    k0 = int(center["k"])
    mode0 = str(center["mode"])
    early0 = int(center["early_blend_steps"])
    weight0 = float(center["persistence_weight_start"])
    strength0 = float(center["spectral_strength"])
    params: list[dict[str, Any]] = [center]
    for k in [k0 - 8, k0 - 4, k0, k0 + 4, k0 + 8]:
        params.append(
            {
                "k": _clamp_int(k, 4, 128),
                "mode": mode0,
                "early_blend_steps": early0,
                "persistence_weight_start": weight0,
                "spectral_strength": strength0,
            }
        )
    for early_steps in [early0 - 5, early0, early0 + 5]:
        for weight in [
            weight0 - 0.1,
            weight0 - 0.05,
            weight0,
            weight0 + 0.05,
            weight0 + 0.1,
        ]:
            params.append(
                {
                    "k": k0,
                    "mode": mode0,
                    "early_blend_steps": _clamp_int(early_steps, 0, 80),
                    "persistence_weight_start": _clamp_float(weight, 0.0, 1.0),
                    "spectral_strength": strength0,
                }
            )
    for strength in [strength0 - 0.25, strength0, strength0 + 0.25, strength0 + 0.5]:
        params.append(
            {
                "k": k0,
                "mode": mode0,
                "early_blend_steps": early0,
                "persistence_weight_start": weight0,
                "spectral_strength": _clamp_float(strength, 0.0, 2.0),
            }
        )
    for directive in directives:
        axis = directive["axis"]
        direction = directive["direction"]
        if direction not in {"increase", "decrease"}:
            continue
        params.extend(
            _directed_retrieval_refinement_params(
                center=center,
                axis=axis,
                direction=direction,
            )
        )
    for mode in ["delta", "raw"]:
        params.append(
            {
                "k": k0,
                "mode": mode,
                "early_blend_steps": early0,
                "persistence_weight_start": weight0,
                "spectral_strength": strength0,
            }
        )
    deduped = []
    seen: set[tuple[int, str, int, float, float]] = set()
    for row in params:
        normalized = _normalize_retrieval_refinement_params(row)
        key = (
            int(normalized["k"]),
            str(normalized["mode"]),
            int(normalized["early_blend_steps"]),
            float(normalized["persistence_weight_start"]),
            float(normalized["spectral_strength"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return _filter_retrieval_refinement_params_by_directives(
        params=deduped,
        center=center,
        directives=directives,
    )


def _early_refinement_params(
    evidence_params: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    center = _retrieval_refinement_center(evidence_params)
    signals = _early_refinement_signals(evidence_params)
    memory_policy = _early_memory_policy(evidence_params)
    k0 = int(center["k"])
    mode0 = str(center["mode"])
    strength0 = float(center["spectral_strength"])
    early0 = int(center["early_blend_steps"])
    weight0 = float(center["persistence_weight_start"])
    primary_anchor = str(center.get("early_anchor", "persistence"))
    if primary_anchor not in EARLY_REFINEMENT_ANCHORS:
        primary_anchor = "persistence"
    modes = [mode0]
    if signals["prioritize_mode"]:
        modes.extend(["affine", "delta", "raw"])
    modes = _dedupe_values(modes)
    anchors = [primary_anchor, "persistence", "retrieval", "linear_velocity"]
    if signals["expand_linear_velocity"]:
        anchors.extend(
            [
                "damped_linear_velocity",
                "scaled_damped_velocity",
                "retrieval_derivative_match",
                "second_order_velocity",
            ]
        )
    if signals["expand_phase_alignment"]:
        anchors.append("phase_aligned_retrieval")
    if signals["expand_bias_calibration"]:
        anchors.append("mean_trend_bias")
        anchors.append("affine_context_calibration")
    if signals["expand_compound_calibration"]:
        anchors.extend(["phase_derivative_hybrid", "affine_context_calibration"])
    anchors = _dedupe_values(anchors)
    spectral_strengths = [strength0]
    if signals["reduce_smoothing"]:
        spectral_strengths.extend([0.0, 0.25, 0.5])
    if signals["increase_smoothing"]:
        spectral_strengths.extend([0.75, 1.0, 1.5])
    spectral_strengths = [
        float(value)
        for value in _dedupe_values(
            [_clamp_float(float(value), 0.0, 2.0) for value in spectral_strengths]
        )
    ]
    handoff_curves = ["linear"]
    if signals["expand_handoff_curves"]:
        handoff_curves.extend(["cosine", "quadratic"])
    handoff_curves = [str(value) for value in _dedupe_values(handoff_curves)]
    phase_shift_scales = [1.0]
    if signals["expand_phase_alignment"]:
        phase_shift_scales.insert(0, 0.5)
    k_offsets = [-8, 0, 8, 16]
    if signals["increase_k"]:
        k_offsets.append(24)
    if signals["decrease_k"]:
        k_offsets.insert(0, -16)
    k_values = [_clamp_task1_retrieval_k(k0 + offset) for offset in k_offsets]
    k_values = [int(value) for value in _dedupe_values(k_values)]
    early_steps_values = [
        _clamp_int(early0 - 4, 4, 80),
        early0,
        _clamp_int(early0 + 4, 4, 80),
        32,
    ]
    early_steps_values = [int(value) for value in _dedupe_values(early_steps_values)]
    weight_values = [
        _clamp_float(weight0 - 0.25, 0.0, 1.0),
        weight0,
        _clamp_float(weight0 + 0.25, 0.0, 1.0),
    ]
    weight_values = [float(value) for value in _dedupe_values(weight_values)]
    rows: list[dict[str, Any]] = []
    core_anchors = [primary_anchor]
    if primary_anchor != "persistence":
        core_anchors.append("persistence")
    for k in k_values:
        for early_steps in early_steps_values:
            for weight in weight_values:
                for anchor in core_anchors:
                    rows.append(
                        {
                            "k": k,
                            "mode": mode0,
                            "early_blend_steps": early_steps,
                            "persistence_weight_start": weight,
                            "spectral_strength": strength0,
                            "early_anchor": anchor,
                            "handoff_curve": "linear",
                        }
                    )
    for mode in modes:
        for k in _dedupe_values(
            [
                _clamp_task1_retrieval_k(k0),
                _clamp_task1_retrieval_k(k0 + 8),
                _clamp_task1_retrieval_k(k0 + 16),
            ]
        ):
            for early_steps in _dedupe_values([early0, _clamp_int(early0 + 4, 4, 80)]):
                rows.append(
                    {
                        "k": int(k),
                        "mode": mode,
                        "early_blend_steps": early_steps,
                        "persistence_weight_start": weight0,
                        "spectral_strength": strength0,
                        "early_anchor": primary_anchor,
                        "handoff_curve": "linear",
                    }
                )
    probe_ks = _dedupe_values(
        [_clamp_task1_retrieval_k(k0), _clamp_task1_retrieval_k(k0 + 8)]
    )
    probe_steps = _dedupe_values([early0, _clamp_int(early0 + 4, 4, 80)])
    for anchor in anchors:
        for k in probe_ks:
            for early_steps in probe_steps:
                rows.append(
                    {
                        "k": int(k),
                        "mode": mode0,
                        "early_blend_steps": int(early_steps),
                        "persistence_weight_start": weight0,
                        "spectral_strength": strength0,
                        "early_anchor": anchor,
                        "handoff_curve": "linear",
                    }
                )
    mechanism_anchors = [
        anchor
        for anchor in anchors
        if anchor
        in {
            "phase_aligned_retrieval",
            "mean_trend_bias",
            "second_order_velocity",
            "scaled_damped_velocity",
            "affine_context_calibration",
            "phase_derivative_hybrid",
        }
    ]
    for anchor in mechanism_anchors:
        for curve in handoff_curves:
            for k in probe_ks:
                row = {
                    "k": int(k),
                    "mode": mode0,
                    "early_blend_steps": _clamp_int(early0 + 4, 4, 80),
                    "persistence_weight_start": _clamp_float(
                        max(weight0, 0.5),
                        0.0,
                        1.0,
                    ),
                    "spectral_strength": strength0,
                    "early_anchor": anchor,
                    "handoff_curve": curve,
                }
                if anchor in {"phase_aligned_retrieval", "phase_derivative_hybrid"}:
                    for scale in phase_shift_scales:
                        shifted = dict(row)
                        shifted["phase_shift_scale"] = float(scale)
                        rows.append(shifted)
                elif anchor == "scaled_damped_velocity":
                    for scale in [0.25, 0.5, 0.75]:
                        scaled = dict(row)
                        scaled["velocity_scale"] = float(scale)
                        rows.append(scaled)
                else:
                    rows.append(row)
    for strength in spectral_strengths:
        for k in probe_ks:
            rows.append(
                {
                    "k": int(k),
                    "mode": mode0,
                    "early_blend_steps": _clamp_int(early0 + 4, 4, 80),
                    "persistence_weight_start": weight0,
                    "spectral_strength": strength,
                    "early_anchor": primary_anchor,
                    "handoff_curve": "linear",
                }
            )
    deduped = []
    seen: set[tuple[int, str, int, float, float, str, str, float, float]] = set()
    for row in rows:
        normalized = _normalize_retrieval_refinement_params(row)
        key = (
            int(normalized["k"]),
            str(normalized["mode"]),
            int(normalized["early_blend_steps"]),
            float(normalized["persistence_weight_start"]),
            float(normalized["spectral_strength"]),
            str(normalized.get("early_anchor", "persistence")),
            str(normalized.get("handoff_curve", "linear")),
            float(normalized.get("phase_shift_scale", 1.0)),
            float(normalized.get("velocity_scale", 1.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return _apply_early_memory_policy(
        params=deduped,
        memory_policy=memory_policy,
    )[:160]


def _early_transition_expert_params(
    evidence_params: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    center = _retrieval_refinement_center(evidence_params)
    rows: list[dict[str, Any]] = []
    for k in _dedupe_values([int(center["k"]), 32, 48, 64]):
        for mode in ["velocity_ridge", "delta_mean", "hybrid"]:
            for correction_weight in [0.5, 0.75, 1.0]:
                rows.append(
                    {
                        "k": _clamp_task1_retrieval_k(int(k)),
                        "mode": str(center.get("mode", "affine")),
                        "early_blend_steps": 0,
                        "persistence_weight_start": 0.0,
                        "spectral_strength": 0.0,
                        "horizon_steps": 47,
                        "ridge_alpha": 1e-3,
                        "correction_weight": correction_weight,
                        "transition_mode": mode,
                        "feature_mode": "last_velocity",
                    }
                )
    for ridge_alpha in [1e-4, 1e-3, 1e-2]:
        rows.append(
            {
                "k": _clamp_task1_retrieval_k(int(center["k"])),
                "mode": str(center.get("mode", "affine")),
                "early_blend_steps": 0,
                "persistence_weight_start": 0.0,
                "spectral_strength": 0.0,
                "horizon_steps": 47,
                "ridge_alpha": ridge_alpha,
                "correction_weight": 1.0,
                "transition_mode": "velocity_ridge",
                "feature_mode": "last_velocity",
            }
        )
    deduped = []
    seen: set[tuple[int, str, int, float, float, str]] = set()
    for row in rows:
        normalized = _normalize_early_transition_expert_params(row)
        key = (
            int(normalized["k"]),
            str(normalized["transition_mode"]),
            int(normalized["horizon_steps"]),
            float(normalized["ridge_alpha"]),
            float(normalized["correction_weight"]),
            str(normalized["feature_mode"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:48]


def _normalize_early_transition_expert_params(row: dict[str, Any]) -> dict[str, Any]:
    base = _normalize_retrieval_refinement_params(
        {
            "k": row.get("k", row.get("base_k", 48)),
            "mode": row.get("mode", row.get("base_mode", "affine")),
            "early_blend_steps": row.get("early_blend_steps", 0),
            "persistence_weight_start": row.get("persistence_weight_start", 0.0),
            "spectral_strength": row.get("base_spectral_strength", 0.0),
        }
    )
    expert = early_transition_expert_params_from_any(row)
    normalized = {
        **base,
        **expert,
    }
    if "base_k" in row:
        normalized["base_k"] = int(row["base_k"])
    if "base_mode" in row:
        normalized["base_mode"] = str(row["base_mode"])
    return normalized


def _front25_dynamics_candidate_params(
    evidence_params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    rows = [
        item.get("params", item) if isinstance(item, dict) else {}
        for item in evidence_params or []
    ]
    rows.extend(
        [
            {
                "k": 32,
                "horizon_steps": 25,
                "ridge_alpha": 1.0e-5,
                "correction_weight": 0.75,
                "dynamics_mode": "velocity_feature_ridge",
                "feature_mode": "last_velocity_gradient",
                "distance_temperature": 1.0,
                "spectral_strength": 0.0,
                "viscosity": 0.001,
                "reference_time_stride": 1,
            },
            {
                "k": 64,
                "horizon_steps": 25,
                "ridge_alpha": 1.0e-5,
                "correction_weight": 0.75,
                "dynamics_mode": "hybrid",
                "feature_mode": "last_velocity_gradient",
                "distance_temperature": 1.0,
                "spectral_strength": 0.0,
                "viscosity": 0.001,
                "reference_time_stride": 1,
            },
            {
                "k": 64,
                "horizon_steps": 25,
                "ridge_alpha": 1.0e-5,
                "correction_weight": 0.5,
                "dynamics_mode": "pseudospectral",
                "feature_mode": "last_velocity_gradient",
                "distance_temperature": 1.0,
                "spectral_strength": 0.0,
                "viscosity": 0.001,
                "reference_time_stride": 1,
                "pseudospectral_dt": 0.03,
                "pseudospectral_substeps": 4,
                "pseudospectral_filter_strength": 0.01,
                "pseudospectral_dealias_fraction": 2.0 / 3.0,
            },
        ]
    )
    deduped = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = front25_dynamics_params_from_any(row)
        key = tuple(sorted(normalized.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:16]
