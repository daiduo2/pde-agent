"""Candidate metadata and ID helpers for Hypothesis Lab sandbox rows."""

from typing import Any

import numpy as np

from pde_agent.core.burgers_residual_refinement import (
    burgers_residual_summary,
    physics_residual_refinement_params_from_any,
    residual_guided_refinement,
)
from pde_agent.core.burgers_spectral_element import sem_params_from_any
from pde_agent.core.hypothesis_lab_anchors import _max_retrieval_k
from pde_agent.core.hypothesis_lab_candidate_pool import _base_reference_metadata
from pde_agent.core.hypothesis_lab_predictions import (
    _base_prediction_for_physics_residual_refinement,
    _prediction_from_task1_retrieval_refinement,
)
from pde_agent.core.hypothesis_lab_utils import _normalize_retrieval_refinement_params
from pde_agent.core.manual_operator_tuning import make_persistence_prediction
from pde_agent.core.pi_deeponet_correction_head import (
    fit_pi_deeponet_correction_head,
    pi_deeponet_params_from_any,
)
from pde_agent.core.task1_early_transition_expert import (
    early_transition_expert_params_from_any,
    fit_task1_early_transition_expert,
)
from pde_agent.core.task1_front25_dynamics import (
    fit_task1_front25_dynamics,
    front25_dynamics_params_from_any,
)
from pde_agent.core.task1_short_window_corrector import (
    fit_task1_short_window_corrector,
    short_window_corrector_params_from_any,
)


def _candidate_metadata(
    *,
    family: str,
    context: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    reference_audit: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    if family == "task1_sem_burgers_rollout":
        sem_params = sem_params_from_any(params)
        return {
            "executor": "sem_burgers_rollout",
            "viscosity": sem_params["viscosity"],
            "dt": sem_params["dt"],
            "element_count": sem_params["element_count"],
            "polynomial_order": sem_params["polynomial_order"],
            "substeps": sem_params["substeps"],
            "filter_strength": sem_params["filter_strength"],
        }
    if family == "task1_physics_residual_refinement":
        residual_params = physics_residual_refinement_params_from_any(params)
        base_prediction = _base_prediction_for_physics_residual_refinement(
            context=context,
            target=target,
            reference=reference,
            params=residual_params,
        )
        candidate_prediction = residual_guided_refinement(
            context=context,
            base_prediction=base_prediction,
            viscosity=float(residual_params["viscosity"]),
            dt=float(residual_params["dt"]),
            dx=float(residual_params["dx"]),
            correction_weight=float(residual_params["correction_weight"]),
            stop_step=int(residual_params["stop_step"]),
            spectral_strength=float(residual_params["residual_spectral_strength"]),
        )
        base_summary = burgers_residual_summary(
            base_prediction,
            viscosity=float(residual_params["viscosity"]),
            dt=float(residual_params["dt"]),
            dx=float(residual_params["dx"]),
        )
        candidate_summary = burgers_residual_summary(
            candidate_prediction,
            viscosity=float(residual_params["viscosity"]),
            dt=float(residual_params["dt"]),
            dx=float(residual_params["dx"]),
        )
        return {
            "executor": "residual_guided_refinement",
            "base_family": residual_params["base_family"],
            **_base_reference_metadata(
                base_family=str(residual_params["base_family"]),
                reference=reference,
                reference_audit=reference_audit,
            ),
            "viscosity": residual_params["viscosity"],
            "dt": residual_params["dt"],
            "dx": residual_params["dx"],
            "correction_weight": residual_params["correction_weight"],
            "stop_step": residual_params["stop_step"],
            "residual_spectral_strength": residual_params[
                "residual_spectral_strength"
            ],
            "base_residual_rmse": base_summary["residual_rmse"],
            "candidate_residual_rmse": candidate_summary["residual_rmse"],
            "residual_rmse_delta": (
                float(candidate_summary["residual_rmse"])
                - float(base_summary["residual_rmse"])
            ),
            "base_early_residual_rmse": base_summary["early_residual_rmse"],
            "candidate_early_residual_rmse": candidate_summary[
                "early_residual_rmse"
            ],
        }
    if family == "task1_pi_deeponet_correction_head":
        pi_params = pi_deeponet_params_from_any(params)
        base_prediction = _base_prediction_for_physics_residual_refinement(
            context=context,
            target=target,
            reference=reference,
            params=pi_params,
        )
        result = fit_pi_deeponet_correction_head(
            context=context,
            base_prediction=base_prediction,
            target=target,
            ridge_alpha=float(pi_params["ridge_alpha"]),
            correction_weight=float(pi_params["correction_weight"]),
            physics_weight=float(pi_params["physics_weight"]),
            viscosity=float(pi_params["viscosity"]),
            dt=float(pi_params["dt"]),
            dx=float(pi_params["dx"]),
        )
        metadata = dict(result["metadata"])
        metadata["base_family"] = pi_params["base_family"]
        metadata.update(
            _base_reference_metadata(
                base_family=str(pi_params["base_family"]),
                reference=reference,
                reference_audit=reference_audit,
            )
        )
        return metadata
    if family == "task1_early_transition_expert":
        if reference.shape[1] >= 200:
            base_prediction = _prediction_from_task1_retrieval_refinement(
                context=context,
                reference=reference,
                params=_normalize_retrieval_refinement_params(
                    {
                        "k": params.get("base_k", params.get("k", 48)),
                        "mode": params.get("base_mode", params.get("mode", "affine")),
                        "early_blend_steps": params.get("early_blend_steps", 0),
                        "persistence_weight_start": params.get(
                            "persistence_weight_start",
                            0.0,
                        ),
                        "spectral_strength": params.get("base_spectral_strength", 0.0),
                    }
                ),
            )
        else:
            base_prediction = make_persistence_prediction(context)
        result = fit_task1_early_transition_expert(
            context=context,
            base_prediction=base_prediction,
            reference=reference,
            **early_transition_expert_params_from_any(params),
        )
        metadata = dict(result["metadata"])
        metadata.update(
            {
                "early_objective": "early_rel_mse",
                "reference_path": reference_audit["reference_path"],
                "uses_target_as_reference": reference_audit[
                    "uses_target_as_reference"
                ],
                "submission_validity": reference_audit["submission_validity"],
            }
        )
        return metadata
    if family == "task1_front25_dynamics":
        if reference.shape[1] >= 200:
            base_prediction = _prediction_from_task1_retrieval_refinement(
                context=context,
                reference=reference,
                params=_normalize_retrieval_refinement_params(
                    {
                        "k": params.get("base_k", params.get("k", 64)),
                        "mode": params.get("base_mode", params.get("mode", "affine")),
                        "early_blend_steps": params.get("early_blend_steps", 0),
                        "persistence_weight_start": params.get(
                            "persistence_weight_start",
                            0.0,
                        ),
                        "spectral_strength": params.get("base_spectral_strength", 0.0),
                    }
                ),
            )
        else:
            base_prediction = make_persistence_prediction(context)
        result = fit_task1_front25_dynamics(
            context=context,
            base_prediction=base_prediction,
            reference=reference,
            **front25_dynamics_params_from_any(params),
        )
        metadata = dict(result["metadata"])
        metadata.update(
            {
                "early_objective": "early_rel_mse",
                "reference_path": reference_audit["reference_path"],
                "uses_target_as_reference": reference_audit[
                    "uses_target_as_reference"
                ],
                "submission_validity": reference_audit["submission_validity"],
            }
        )
        return metadata
    if family == "task1_short_window_corrector":
        if reference.shape[1] >= 200:
            base_k = min(int(params.get("base_k", 96)), _max_retrieval_k(context, reference))
            base_prediction = _prediction_from_task1_retrieval_refinement(
                context=context,
                reference=reference,
                params=_normalize_retrieval_refinement_params(
                    {
                        "k": base_k,
                        "mode": params.get("base_mode", "affine"),
                        "early_blend_steps": params.get("early_blend_steps", 0),
                        "persistence_weight_start": params.get(
                            "persistence_weight_start",
                            0.0,
                        ),
                        "spectral_strength": params.get("base_spectral_strength", 0.0),
                    }
                ),
            )
        else:
            base_prediction = make_persistence_prediction(context)
        result = fit_task1_short_window_corrector(
            context=context,
            base_prediction=base_prediction,
            reference=reference,
            **short_window_corrector_params_from_any(params),
        )
        metadata = dict(result["metadata"])
        metadata.update(
            {
                "early_objective": "early_rel_mse",
                "reference_path": reference_audit["reference_path"],
                "uses_target_as_reference": reference_audit[
                    "uses_target_as_reference"
                ],
                "submission_validity": reference_audit["submission_validity"],
            }
        )
        return metadata
    if family not in {
        "task1_retrieval_prior",
        "task1_retrieval_refinement",
        "task1_retrieval_refinement_local_search",
        "task1_early_refinement",
    }:
        return {}
    metadata = {
        "leave_one_out": bool(
            context.shape[0] == reference.shape[0]
            and np.allclose(context, reference[:, : context.shape[1], :], atol=1e-6)
        ),
        "reference_count": int(reference.shape[0]),
        "reference_path": reference_audit["reference_path"],
        "uses_target_as_reference": reference_audit["uses_target_as_reference"],
        "submission_validity": reference_audit["submission_validity"],
        "k": int(params["k"]),
        "mode": str(params.get("mode", "affine")),
    }
    if family in {
        "task1_retrieval_refinement",
        "task1_retrieval_refinement_local_search",
        "task1_early_refinement",
    }:
        metadata.update(
            {
                "early_blend_steps": int(params["early_blend_steps"]),
                "persistence_weight_start": float(
                    params["persistence_weight_start"]
                ),
                "spectral_strength": float(params.get("spectral_strength", 0.0)),
                "handoff_curve": str(params.get("handoff_curve", "linear")),
            }
        )
        if "phase_shift_scale" in params:
            metadata["phase_shift_scale"] = float(params["phase_shift_scale"])
        if "velocity_scale" in params:
            metadata["velocity_scale"] = float(params["velocity_scale"])
    if family == "task1_early_refinement":
        metadata.update(
            {
                "early_anchor": str(params.get("early_anchor", "persistence")),
                "early_objective": "early_rel_mse",
            }
        )
    return metadata


def _candidate_id(params: dict[str, Any], *, family: str | None = None) -> str:
    if family == "task1_short_window_corrector":
        corrector_params = short_window_corrector_params_from_any(params)
        return (
            "task1_short_window_corrector"
            f"_{corrector_params['corrector_mode']}"
            f"_w{int(corrector_params['start_step'])}_{int(corrector_params['stop_step'])}"
            f"_fc{int(corrector_params['feature_count'])}"
            f"_cw{float(corrector_params['correction_weight']):g}"
            f"_ra{float(corrector_params['ridge_alpha']):g}"
            f"_ts{int(corrector_params['reference_time_stride'])}"
        )
    if family == "task1_front25_dynamics":
        dynamics_params = front25_dynamics_params_from_any(params)
        return (
            "task1_front25_dynamics"
            f"_k{int(dynamics_params['k'])}"
            f"_{dynamics_params['dynamics_mode']}"
            f"_h{int(dynamics_params['horizon_steps'])}"
            f"_cw{float(dynamics_params['correction_weight']):g}"
            f"_ra{float(dynamics_params['ridge_alpha']):g}"
            f"_ts{int(dynamics_params['reference_time_stride'])}"
        )
    if family == "task1_early_transition_expert":
        expert_params = early_transition_expert_params_from_any(params)
        return (
            "task1_early_transition_expert"
            f"_k{int(expert_params['k'])}"
            f"_{expert_params['transition_mode']}"
            f"_h{int(expert_params['horizon_steps'])}"
            f"_cw{float(expert_params['correction_weight']):g}"
            f"_ra{float(expert_params['ridge_alpha']):g}"
        )
    if family == "task1_pi_deeponet_correction_head":
        pi_params = pi_deeponet_params_from_any(params)
        return (
            "task1_pi_deeponet_correction_head"
            f"_{pi_params['base_family']}"
            f"_cw{float(pi_params['correction_weight']):g}"
            f"_pa{float(pi_params['ridge_alpha']):g}"
            f"_pw{float(pi_params['physics_weight']):g}"
        )
    if family == "task1_physics_residual_refinement":
        residual_params = physics_residual_refinement_params_from_any(params)
        return (
            "task1_physics_residual_refinement"
            f"_{residual_params['base_family']}"
            f"_cw{float(residual_params['correction_weight']):g}"
            f"_stop{int(residual_params['stop_step'])}"
            f"_rs{float(residual_params['residual_spectral_strength']):g}"
        )
    if family == "task1_sem_burgers_rollout":
        sem_params = sem_params_from_any(params)
        return (
            "task1_sem_burgers_rollout"
            f"_e{sem_params['element_count']}"
            f"_p{sem_params['polynomial_order']}"
            f"_sub{sem_params['substeps']}"
            f"_nu{sem_params['viscosity']:g}"
            f"_dt{sem_params['dt']:g}"
            f"_f{sem_params['filter_strength']:g}"
        )
    if "early_blend_steps" in params and "persistence_weight_start" in params:
        prefix = (
            "task1_early_refinement"
            if family == "task1_early_refinement"
            else "task1_retrieval_refinement"
        )
        suffix = ""
        if family == "task1_early_refinement":
            suffix = f"_a{params.get('early_anchor', 'persistence')}"
            handoff_curve = str(params.get("handoff_curve", "linear"))
            if handoff_curve != "linear":
                suffix += f"_h{handoff_curve}"
            if "phase_shift_scale" in params:
                suffix += f"_ps{float(params['phase_shift_scale']):g}"
            if "velocity_scale" in params:
                suffix += f"_vs{float(params['velocity_scale']):g}"
        return (
            f"{prefix}"
            f"_k{int(params['k'])}"
            f"_{params['mode']}"
            f"_eb{int(params['early_blend_steps'])}"
            f"_pw{float(params['persistence_weight_start']):g}"
            f"_s{float(params.get('spectral_strength', 0.0)):g}"
            f"{suffix}"
        )
    if "k" in params and "mode" in params:
        return f"task1_retrieval_prior_k{int(params['k'])}_{params['mode']}"
    if "residual_shrinkage" in params:
        return (
            "learned_residual"
            f"_sh{params['residual_shrinkage']:g}"
            f"_rs{params.get('residual_spectral_strength', 0.0):g}"
        )
    candidate = (
        "linear_trend_spectral"
        f"_w{params['slope_window']}"
        f"_g{params['gain']:g}"
        f"_d{params['decay']:g}"
        f"_s{params['spectral_strength']:g}"
    )
    if "late_spectral_strength" in params:
        candidate += f"_ls{params['late_spectral_strength']:g}"
    if "late_start" in params:
        candidate += f"_lt{params['late_start']}"
    return candidate
