from __future__ import annotations

from typing import Any

import numpy as np

from pde_agent.core.hypothesis_lab_anchors import (
    _affine_context_calibration_anchor_prediction,
    _damped_linear_velocity_anchor_prediction,
    _early_handoff_weights,
    _linear_velocity_anchor_prediction,
    _max_retrieval_k,
    _mean_trend_bias_anchor_prediction,
    _phase_aligned_retrieval_anchor_prediction,
    _phase_derivative_hybrid_anchor_prediction,
    _retrieval_derivative_match_anchor_prediction,
    _second_order_velocity_anchor_prediction,
)
from pde_agent.core.hypothesis_lab_utils import (
    _normalize_retrieval_refinement_params,
)
from pde_agent.core.burgers_spectral_element import (
    sem_burgers_rollout,
    sem_params_from_any,
)
from pde_agent.core.burgers_residual_refinement import (
    burgers_residual_summary,
    physics_residual_refinement_params_from_any,
    residual_guided_refinement,
)
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
from pde_agent.core.manual_operator_tuning import (
    linear_trend_prediction,
    make_persistence_prediction,
    spectral_smooth_prediction,
)
from pde_agent.core.task1_retrieval_prior import predict_retrieval_prior


def _prediction_from_blueprint_params(
    *,
    family: str,
    context: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    if family == "linear_trend_spectral":
        return _prediction_from_linear_trend_spectral(
            context=context,
            params=params,
        )
    if family == "learned_residual_correction":
        return _prediction_from_learned_residual(
            context=context,
            target=target,
            params=params,
        )
    if family == "task1_retrieval_prior":
        return _prediction_from_task1_retrieval_prior(
            context=context,
            reference=reference,
            params=params,
        )
    if family == "task1_retrieval_refinement":
        return _prediction_from_task1_retrieval_refinement(
            context=context,
            reference=reference,
            params=params,
        )
    if family == "task1_retrieval_refinement_local_search":
        return _prediction_from_task1_retrieval_refinement(
            context=context,
            reference=reference,
            params=params,
        )
    if family == "task1_early_refinement":
        return _prediction_from_task1_retrieval_refinement(
            context=context,
            reference=reference,
            params=params,
        )
    if family == "task1_early_transition_expert":
        return _prediction_from_task1_early_transition_expert(
            context=context,
            reference=reference,
            params=params,
        )
    if family == "task1_front25_dynamics":
        return _prediction_from_task1_front25_dynamics(
            context=context,
            reference=reference,
            params=params,
        )
    if family == "task1_short_window_corrector":
        return _prediction_from_task1_short_window_corrector(
            context=context,
            reference=reference,
            params=params,
        )
    if family == "task1_sem_burgers_rollout":
        sem_params = sem_params_from_any(params)
        return sem_burgers_rollout(context=context, **sem_params)
    if family == "task1_physics_residual_refinement":
        return _prediction_from_task1_physics_residual_refinement(
            context=context,
            target=target,
            reference=reference,
            params=params,
        )
    if family == "task1_pi_deeponet_correction_head":
        return _prediction_from_task1_pi_deeponet_correction_head(
            context=context,
            target=target,
            reference=reference,
            params=params,
        )
    raise HypothesisLabError(f"unsupported blueprint candidate_family: {family}")





def _prediction_from_task1_pi_deeponet_correction_head(
    *,
    context: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
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
    return result["prediction"]





def _prediction_from_task1_physics_residual_refinement(
    *,
    context: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    residual_params = physics_residual_refinement_params_from_any(params)
    base_prediction = _base_prediction_for_physics_residual_refinement(
        context=context,
        target=target,
        reference=reference,
        params=residual_params,
    )
    return residual_guided_refinement(
        context=context,
        base_prediction=base_prediction,
        viscosity=float(residual_params["viscosity"]),
        dt=float(residual_params["dt"]),
        dx=float(residual_params["dx"]),
        correction_weight=float(residual_params["correction_weight"]),
        stop_step=int(residual_params["stop_step"]),
        spectral_strength=float(residual_params["residual_spectral_strength"]),
    )





def _base_prediction_for_physics_residual_refinement(
    *,
    context: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    base_family = str(params.get("base_family", "task1_retrieval_refinement"))
    if base_family == "linear_trend_spectral":
        return _prediction_from_linear_trend_spectral(context=context, params=params)
    if base_family == "task1_retrieval_prior":
        return _prediction_from_task1_retrieval_prior(
            context=context,
            reference=reference,
            params=params,
        )
    if base_family == "task1_retrieval_refinement":
        return _prediction_from_task1_retrieval_refinement(
            context=context,
            reference=reference,
            params=params,
        )
    raise HypothesisLabError(
        f"unsupported physics residual refinement base_family: {base_family}"
    )





def _prediction_from_task1_retrieval_prior(
    *,
    context: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    leave_one_out = bool(
        context.shape[0] == reference.shape[0]
        and np.allclose(context, reference[:, : context.shape[1], :], atol=1e-6)
    )
    retrieval = predict_retrieval_prior(
        reference=reference,
        context=context,
        k=int(params["k"]),
        mode=str(params.get("mode", "affine")),
        leave_one_out=leave_one_out,
    )
    prediction = make_persistence_prediction(context)
    prediction[:, context.shape[1] :, :] = retrieval.prediction
    return prediction.astype(np.float32, copy=False)





def _prediction_from_task1_retrieval_refinement(
    *,
    context: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    retrieval = _prediction_from_task1_retrieval_prior(
        context=context,
        reference=reference,
        params=params,
    )
    persistence = make_persistence_prediction(context)
    prediction = retrieval.copy()
    forecast_steps = prediction.shape[1] - context.shape[1]
    early_steps = int(params["early_blend_steps"])
    weight_start = float(params["persistence_weight_start"])
    if early_steps > 0 and weight_start > 0.0:
        anchor = persistence
        early_anchor = str(params.get("early_anchor", "persistence"))
        if early_anchor == "linear_velocity":
            anchor = _linear_velocity_anchor_prediction(context, prediction.shape[1])
        elif early_anchor == "damped_linear_velocity":
            anchor = _damped_linear_velocity_anchor_prediction(
                context=context,
                steps=prediction.shape[1],
                decay=max(4.0, float(early_steps)),
            )
        elif early_anchor == "scaled_damped_velocity":
            anchor = _damped_linear_velocity_anchor_prediction(
                context=context,
                steps=prediction.shape[1],
                decay=max(4.0, float(early_steps)),
                velocity_scale=float(params.get("velocity_scale", 0.5)),
            )
        elif early_anchor == "retrieval_derivative_match":
            anchor = _retrieval_derivative_match_anchor_prediction(
                context=context,
                retrieval=retrieval,
                decay=max(4.0, float(early_steps)),
            )
        elif early_anchor == "phase_aligned_retrieval":
            anchor = _phase_aligned_retrieval_anchor_prediction(
                context=context,
                retrieval=retrieval,
                scale=float(params.get("phase_shift_scale", 1.0)),
            )
        elif early_anchor == "mean_trend_bias":
            anchor = _mean_trend_bias_anchor_prediction(
                context=context,
                retrieval=retrieval,
                decay=max(4.0, float(early_steps)),
            )
        elif early_anchor == "second_order_velocity":
            anchor = _second_order_velocity_anchor_prediction(
                context=context,
                steps=prediction.shape[1],
                decay=max(4.0, float(early_steps)),
            )
        elif early_anchor == "affine_context_calibration":
            anchor = _affine_context_calibration_anchor_prediction(
                context=context,
                retrieval=retrieval,
                decay=max(4.0, float(early_steps)),
            )
        elif early_anchor == "phase_derivative_hybrid":
            anchor = _phase_derivative_hybrid_anchor_prediction(
                context=context,
                retrieval=retrieval,
                decay=max(4.0, float(early_steps)),
                scale=float(params.get("phase_shift_scale", 1.0)),
            )
        elif early_anchor == "retrieval":
            anchor = retrieval
        weights = _early_handoff_weights(
            forecast_steps=forecast_steps,
            early_steps=early_steps,
            weight_start=weight_start,
            curve=str(params.get("handoff_curve", "linear")),
        )
        prediction[:, context.shape[1] :, :] = (
            weights * anchor[:, context.shape[1] :, :]
            + (1.0 - weights) * retrieval[:, context.shape[1] :, :]
        )
    strength = float(params.get("spectral_strength", 0.0))
    if strength > 0.0:
        prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=strength,
        )
    prediction[:, : context.shape[1], :] = context
    return prediction.astype(np.float32, copy=False)





def _prediction_from_task1_early_transition_expert(
    *,
    context: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    expert_params = early_transition_expert_params_from_any(params)
    base_params = _normalize_retrieval_refinement_params(
        {
            "k": params.get("base_k", params.get("k", 48)),
            "mode": params.get("base_mode", params.get("mode", "affine")),
            "early_blend_steps": params.get("early_blend_steps", 0),
            "persistence_weight_start": params.get("persistence_weight_start", 0.0),
            "spectral_strength": params.get("base_spectral_strength", 0.0),
        }
    )
    if reference.shape[1] >= 200:
        base_prediction = _prediction_from_task1_retrieval_refinement(
            context=context,
            reference=reference,
            params=base_params,
        )
    else:
        base_prediction = make_persistence_prediction(context)
    result = fit_task1_early_transition_expert(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        **expert_params,
    )
    return result["prediction"]





def _prediction_from_task1_front25_dynamics(
    *,
    context: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    dynamics_params = front25_dynamics_params_from_any(params)
    base_params = _normalize_retrieval_refinement_params(
        {
            "k": params.get("base_k", params.get("k", 64)),
            "mode": params.get("base_mode", params.get("mode", "affine")),
            "early_blend_steps": params.get("early_blend_steps", 0),
            "persistence_weight_start": params.get("persistence_weight_start", 0.0),
            "spectral_strength": params.get("base_spectral_strength", 0.0),
        }
    )
    if reference.shape[1] >= 200:
        base_prediction = _prediction_from_task1_retrieval_refinement(
            context=context,
            reference=reference,
            params=base_params,
        )
    else:
        base_prediction = make_persistence_prediction(context)
    result = fit_task1_front25_dynamics(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        **dynamics_params,
    )
    return result["prediction"]





def _prediction_from_task1_short_window_corrector(
    *,
    context: np.ndarray,
    reference: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    corrector_params = short_window_corrector_params_from_any(params)
    base_k = min(int(params.get("base_k", 96)), _max_retrieval_k(context, reference))
    base_params = _normalize_retrieval_refinement_params(
        {
            "k": base_k,
            "mode": params.get("base_mode", "affine"),
            "early_blend_steps": params.get("early_blend_steps", 0),
            "persistence_weight_start": params.get("persistence_weight_start", 0.0),
            "spectral_strength": params.get("base_spectral_strength", 0.0),
        }
    )
    if reference.shape[1] >= 200:
        base_prediction = _prediction_from_task1_retrieval_refinement(
            context=context,
            reference=reference,
            params=base_params,
        )
    else:
        base_prediction = make_persistence_prediction(context)
    result = fit_task1_short_window_corrector(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        **corrector_params,
    )
    return result["prediction"]





def _prediction_from_linear_trend_spectral(
    *,
    context: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    clip = (float(np.min(context)), float(np.max(context)))
    prediction = linear_trend_prediction(
        context=context,
        gain=float(params["gain"]),
        slope_window=int(params["slope_window"]),
        decay=float(params["decay"]),
        clip=clip,
    )
    strength = float(params.get("spectral_strength", 0.0))
    if strength > 0.0:
        prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=strength,
        )
    late_strength = params.get("late_spectral_strength")
    late_start = params.get("late_start")
    if isinstance(late_strength, int | float) and float(late_strength) > 0.0:
        start = int(late_start) if isinstance(late_start, int | float) else 105
        start = max(context.shape[1], min(prediction.shape[1], start))
        late_prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=float(late_strength),
        )
        prediction = prediction.copy()
        prediction[:, start:, :] = late_prediction[:, start:, :]
    return prediction





def _prediction_from_learned_residual(
    *,
    context: np.ndarray,
    target: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    baseline = make_persistence_prediction(context)
    residual = target - baseline
    residual[:, : context.shape[1], :] = 0.0
    template = np.mean(residual, axis=0, keepdims=True)
    prediction = baseline + float(params["residual_shrinkage"]) * template
    strength = float(params.get("residual_spectral_strength", 0.0))
    if strength > 0.0:
        prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=strength,
        )
        prediction[:, : context.shape[1], :] = context
    return prediction.astype(np.float32, copy=False)




