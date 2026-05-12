"""Hypothesis Lab design submodule."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from pde_agent.core.hypothesis_lab_utils import (
    EARLY_HANDOFF_CURVES,
    EARLY_REFINEMENT_ANCHORS,
    HypothesisLabError,
    _clamp_float,
    _clamp_int,
    _clamp_task1_retrieval_k,
    _guard_output,
    _normalize_retrieval_refinement_params,
    _read_json,
    _require_state,
    _string_list_from_any,
    _write_json,
    _write_optional_json,
)
from pde_agent.core.hypothesis_lab_candidate_pool import (
    _apply_early_memory_policy,
    _candidate_pool_action,
    _candidate_pool_attribution,
    _candidate_pool_policy_from_evidence,
    _candidate_pool_search_space_policy,
    _candidate_early_rel_mse,
    _dedupe_strings,
    _dedupe_values,
    _directed_retrieval_refinement_params,
    _early_memory_policy,
    _early_memory_rank_key,
    _early_param_sort_key,
    _early_params_match,
    _early_refinement_experiment_plan,
    _early_refinement_signals,
    _filter_retrieval_refinement_params_by_directives,
    _finite_float_or_none,
    _retrieval_refinement_avoid_rules_from_directives,
    _retrieval_refinement_center,
    _retrieval_refinement_experiment_plan,
    _retrieval_refinement_search_directives_from_evidence,
    _should_avoid_exact_memory_params,
)
from pde_agent.core.hypothesis_lab_candidate_params import (
    _default_retrieval_refinement_params,
    _early_refinement_params,
    _early_transition_expert_parameter_grid,
    _early_transition_expert_params,
    _front25_dynamics_candidate_params,
    _local_retrieval_refinement_params,
)

HYPOTHESES_STATE_HINT = "HYPOTHESIS_LAB_HYPOTHESES_GENERATED"
BLUEPRINTS_STATE_HINT = "HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED"

def design_algorithm_blueprints(
    *,
    hypotheses: dict[str, Any],
    output_path: str | Path | None,
    force: bool,
) -> dict[str, Any]:
    from pde_agent.core.literature_algorithm_lab import (
        audit_executable_blueprint_payload,
    )

    _require_state(hypotheses, HYPOTHESES_STATE_HINT)
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    blueprints = []
    for hypothesis in hypotheses.get("hypotheses", []):
        if not isinstance(hypothesis, dict):
            continue
        if hypothesis.get("candidate_family") == "official_checkpoint_calibration":
            blueprints.append(
                {
                    "blueprint_id": "bp_official_checkpoint_calibration_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "official_checkpoint_calibration",
                    "search_mode": "checkpoint_calibration",
                    "checkpoint_candidates": hypothesis.get("checkpoint_candidates", []),
                    "algorithm": {
                        "summary": (
                            "Evaluate official PDEBench checkpoints as a distinct "
                            "model family, then calibrate horizon blending or run "
                            "short fine-tuning only if local judge evidence beats "
                            "postprocessing baselines."
                        ),
                        "pseudocode": [
                            "load official checkpoint through FNO/UNet backend",
                            "roll out first 10 context steps to 200-step prediction",
                            "sweep horizon calibration and blend only under budget",
                            "promote only with multi-window score and runtime evidence",
                        ],
                    },
                    "sandbox": {
                        "allowed_executor": "official_checkpoint_candidate_pipeline",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                        "executable_in_hypothesis_lab": False,
                    },
                    "execution_plan": {
                        "cli_command": "run-checkpoint-family-sandbox",
                        "required_options": [
                            "--checkpoint-path",
                            "--task-id",
                            "--test-path",
                            "--target-path",
                            "--output-dir",
                        ],
                        "optional_options": [
                            "--calibration-weight",
                            "--fno-finetune-steps",
                            "--submission-id",
                            "--dataset-key",
                            "--prediction-dataset-key",
                            "--force",
                        ],
                        "handoff_reason": (
                            "Checkpoint candidates require the audited Torch "
                            "checkpoint adapters and may run short FNO fine-tuning, "
                            "so Hypothesis Lab emits an executable handoff instead "
                            "of running them inside the lightweight numpy sandbox."
                        ),
                    },
                    "next_command": "run-checkpoint-family-sandbox",
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_sem_burgers_rollout":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_sem_burgers_rollout_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_sem_burgers_rollout",
                    "search_mode": "physics_numeric_rollout",
                    "algorithm": {
                        "summary": (
                            "Advance fixed-nu 1D Burgers from the last context "
                            "frame with a lightweight spectral-element style "
                            "collocation executor."
                        ),
                        "pseudocode": [
                            "state = context[:, -1, :]",
                            "for each requested future step:",
                            "  substep RK4 on u_t = -u u_x + nu u_xx",
                            "  compute derivatives with element-local spectral grids",
                            "  blend element interfaces and optionally filter modes",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": [
                            {
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "element_count": 16,
                                "polynomial_order": 8,
                                "substeps": 8,
                                "filter_strength": 0.02,
                                "interface_penalty": 0.02,
                            },
                            {
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "element_count": 32,
                                "polynomial_order": 8,
                                "substeps": 8,
                                "filter_strength": 0.05,
                                "interface_penalty": 0.04,
                            },
                        ]
                    },
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_sem_burgers_rollout",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_physics_residual_refinement":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_physics_residual_refinement_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_physics_residual_refinement",
                    "search_mode": "physics_residual_refinement",
                    "algorithm": {
                        "summary": (
                            "Score a strong Task 1 data-prior trajectory with the "
                            "Burgers residual and apply a bounded one-step physics "
                            "nudge over early and middle horizons."
                        ),
                        "pseudocode": [
                            "base = retrieval_refinement(context, reference_library)",
                            "for t in selected future window:",
                            "  physics_next = u[t-1] + dt*(-u*u_x + nu*u_xx)",
                            "  u[t] = blend(base[t], physics_next, correction_weight)",
                            "record base and candidate Burgers residual summaries",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": [
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                                "correction_weight": 0.0,
                                "stop_step": 57,
                                "residual_spectral_strength": 0.0,
                            },
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                                "correction_weight": 0.02,
                                "stop_step": 57,
                                "residual_spectral_strength": 0.0,
                            },
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                                "correction_weight": 0.05,
                                "stop_step": 105,
                                "residual_spectral_strength": 0.0,
                            },
                        ]
                    },
                    "sandbox": {
                        "allowed_executor": (
                            "controlled_numpy_burgers_residual_refinement"
                        ),
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_pi_deeponet_correction_head":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_pi_deeponet_correction_head_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_pi_deeponet_correction_head",
                    "search_mode": "pi_deeponet_correction_head",
                    "algorithm": {
                        "summary": (
                            "Fit a low-rank branch-trunk correction field over a "
                            "strong Task 1 base trajectory and keep the context "
                            "prefix exact."
                        ),
                        "pseudocode": [
                            "base = retrieval_refinement(context, reference_library)",
                            "branch = statistics(context)",
                            "trunk = polynomial/sinusoidal features(t, x)",
                            "solve ridge regression for correction target - base",
                            "prediction = base + correction_weight * correction",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": [
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "ridge_alpha": 1e-4,
                                "correction_weight": 0.1,
                                "physics_weight": 0.0,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                            },
                            {
                                "base_family": "task1_retrieval_refinement",
                                "k": 48,
                                "mode": "affine",
                                "early_blend_steps": 20,
                                "persistence_weight_start": 0.5,
                                "spectral_strength": 0.5,
                                "ridge_alpha": 1e-3,
                                "correction_weight": 0.25,
                                "physics_weight": 0.02,
                                "viscosity": 0.001,
                                "dt": 0.05,
                                "dx": 1.0 / 256.0,
                            },
                        ]
                    },
                    "sandbox": {
                        "allowed_executor": (
                            "controlled_numpy_pi_deeponet_correction_head"
                        ),
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_early_refinement":
            candidate_params = _early_refinement_params(
                hypothesis.get("evidence_params", [])
            )
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_early_refinement_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_early_refinement",
                    "search_mode": "early_segment_objective_search",
                    "algorithm": {
                        "summary": (
                            "Run the retrieval-refinement executor with the early "
                            "segment as a first-class objective, testing short "
                            "horizon anchors such as persistence, retrieval-only, "
                            "and linear-velocity extrapolation while guarding "
                            "middle and late windows."
                        ),
                        "pseudocode": [
                            "center = best retrieval-refinement evidence params",
                            "retrieval = knn_retrieval(context, reference, k, mode)",
                            "anchor = early_anchor(context, retrieval)",
                            "blend first early_blend_steps toward anchor",
                            "score early rel-mse plus middle/late guardrails",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": candidate_params,
                    },
                    "experiment_plan": _early_refinement_experiment_plan(
                        evidence_params=hypothesis.get("evidence_params", []),
                        candidate_budget=len(candidate_params),
                    ),
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_task1_early_refinement",
                        "primary_metric": "task1_early_rel_mse_guarded_score",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_early_transition_expert":
            candidate_params = _early_transition_expert_params(
                hypothesis.get("evidence_params", [])
            )
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_early_transition_expert_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_early_transition_expert",
                    "search_mode": "early_segment_transition_expert_search",
                    "algorithm": {
                        "summary": (
                            "Fit a local reference-conditioned transition map for "
                            "the first 47 predicted frames, using the base "
                            "retrieval-refinement candidate only as the middle and "
                            "late guardrail trajectory."
                        ),
                        "pseudocode": [
                            "base = retrieval_refinement(context, reference, base params)",
                            "neighbors = retrieve reference contexts by last frame and velocity",
                            "expert = local ridge map from context velocity to future delta",
                            "splice expert into frames 10:57 with correction_weight",
                            "score early rel-mse with middle/late guardrails",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": candidate_params,
                    },
                    "experiment_plan": _early_refinement_experiment_plan(
                        evidence_params=hypothesis.get("evidence_params", []),
                        candidate_budget=len(candidate_params),
                    ),
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": (
                            "controlled_numpy_task1_early_transition_expert"
                        ),
                        "primary_metric": "task1_early_rel_mse_guarded_score",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_retrieval_prior":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_retrieval_prior_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_retrieval_prior",
                    "search_mode": "affine_knn_trajectory_prior",
                    "algorithm": {
                        "summary": (
                            "Retrieve full Burgers trajectories whose first 10 "
                            "frames match the query context, then affine-normalize "
                            "their future windows to the query statistics."
                        ),
                        "pseudocode": [
                            "reference = validation trajectories",
                            "neighbors = top-k z-normalized context matches",
                            "future = affine_denormalize(weighted_neighbor_futures)",
                            "prediction = concat(query_context, future)",
                        ],
                    },
                    "parameter_grid": {
                        "k": [8, 12, 20],
                        "mode": ["affine"],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_retrieval_prior",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_retrieval_refinement":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_retrieval_refinement_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_retrieval_refinement",
                    "search_mode": "evidence_focused_coarse_to_fine",
                    "algorithm": {
                        "summary": (
                            "Blend a nearest-neighbor retrieval future with a "
                            "persistence fallback over the early horizon, then "
                            "optionally smooth predicted future frames while "
                            "restoring the context prefix exactly."
                        ),
                        "pseudocode": [
                            "retrieval = knn_retrieval(context, reference, k, mode)",
                            "fallback = persistence(context)",
                            "w = linear_decay(persistence_weight_start, early_blend_steps)",
                            "future = w * fallback_future + (1 - w) * retrieval_future",
                            "future = optional_spectral_smooth(future)",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": _default_retrieval_refinement_params(),
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_retrieval_refinement",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_retrieval_refinement_local_search":
            candidate_params = _local_retrieval_refinement_params(
                hypothesis.get("evidence_params", [])
            )
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_retrieval_refinement_local_search_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_retrieval_refinement_local_search",
                    "search_mode": "evidence_centered_local_search",
                    "algorithm": {
                        "summary": (
                            "Search tightly around the promoted Task 1 retrieval "
                            "refinement, preserving the same bounded NumPy executor "
                            "while perturbing the strongest observed parameters."
                        ),
                        "pseudocode": [
                            "center = best verified sandbox retrieval-refinement params",
                            "sweep nearby k, early blend, persistence weight, smoothing",
                            "retain raw/delta sentinels at the center",
                            "prediction[:10] = context",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": candidate_params,
                    },
                    "experiment_plan": _retrieval_refinement_experiment_plan(
                        evidence_params=hypothesis.get("evidence_params", []),
                        candidate_budget=len(candidate_params),
                    ),
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_retrieval_refinement",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "learned_residual_correction":
            blueprints.append(
                {
                    "blueprint_id": "bp_learned_residual_correction_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "learned_residual_correction",
                    "search_mode": "horizon_mean_residual",
                    "algorithm": {
                        "summary": (
                            "Fit a horizon-wise mean residual over local windows and "
                            "apply a shrinkage factor to persistence or checkpoint "
                            "rollouts while preserving context steps."
                        ),
                        "pseudocode": [
                            "baseline = persistence(context)",
                            "residual[t, x] = mean(target - baseline, axis=samples)",
                            "future = baseline + shrinkage * residual",
                            "future = optional spectral_smooth(future, strength)",
                        ],
                    },
                    "parameter_grid": {
                        "residual_shrinkage": [0.25, 0.5, 0.75],
                        "residual_spectral_strength": [0.0, 0.5],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_residual_operator",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "task1_front25_dynamics":
            blueprints.append(
                {
                    "blueprint_id": "bp_task1_front25_dynamics_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "task1_front25_dynamics",
                    "search_mode": "front25_short_dynamics_search",
                    "algorithm": {
                        "summary": (
                            "Fit a bounded front-window dynamics source from "
                            "reference trajectories and apply it only to the "
                            "early predicted segment."
                        ),
                        "pseudocode": [
                            "match legal reference trajectories to context",
                            "fit short-horizon velocity or gradient features",
                            "replace only the front horizon with guarded dynamics",
                            "preserve the first 10 context frames exactly",
                        ],
                    },
                    "parameter_grid": {
                        "candidate_params": _front25_dynamics_candidate_params(
                            hypothesis.get("evidence_params", [])
                        ),
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_task1_front25_dynamics",
                        "primary_metric": "task1_early_rel_mse_guarded_score",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "linear_trend_spectral":
            blueprints.append(
                {
                    "blueprint_id": "bp_linear_trend_spectral_grid_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "linear_trend_spectral",
                    "search_mode": "evidence_refine",
                    "algorithm": {
                        "summary": (
                            "Estimate a damped negative linear trend from the "
                            "context window and optionally apply spectral smoothing "
                            "to the predicted future."
                        ),
                        "pseudocode": [
                            "slope = (u[-1] - u[-1-window]) / window",
                            "future[t] = u[-1] + gain * t * exp(-t/decay) * slope",
                            "future = spectral_smooth(future, strength)",
                        ],
                    },
                    "parameter_grid": {
                        "slope_window": [4, 5, 6],
                        "gain": [-0.15, -0.125, -0.1, -0.075],
                        "decay": [40.0, 50.0, 60.0, 70.0],
                        "spectral_strength": [0.5, 0.75, 1.0],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_operator",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "linear_trend_spectral_escape":
            blueprints.append(
                {
                    "blueprint_id": "bp_linear_trend_spectral_escape_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "linear_trend_spectral",
                    "search_mode": "plateau_escape",
                    "algorithm": {
                        "summary": (
                            "Use the same controlled damped-trend spectral "
                            "operator, but search outside the verified blueprint's "
                            "local basin after repeated plateau evidence."
                        ),
                        "pseudocode": [
                            "slope = (u[-1] - u[-1-window]) / window",
                            "future[t] = u[-1] + gain * t * exp(-t/decay) * slope",
                            "future = optional spectral_smooth(future, strength)",
                        ],
                    },
                    "parameter_grid": {
                        "slope_window": [2, 3, 4, 5, 8],
                        "gain": [-0.2, -0.15, -0.1, -0.075, -0.05, -0.025],
                        "decay": [20.0, 40.0, 70.0, 100.0],
                        "spectral_strength": [0.0, 0.25, 0.5, 1.0, 1.5],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_operator",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
        if hypothesis.get("candidate_family") == "linear_trend_late_spectral_ramp":
            blueprints.append(
                {
                    "blueprint_id": "bp_linear_trend_late_spectral_ramp_001",
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "candidate_family": "linear_trend_spectral",
                    "search_mode": "late_spectral_ramp",
                    "algorithm": {
                        "summary": (
                            "Reuse the plateau_escape damped trend candidate, then "
                            "apply extra spectral smoothing only to later future "
                            "steps to target long-horizon RMSE without changing "
                            "the context prefix."
                        ),
                        "pseudocode": [
                            "future = damped_linear_trend(context, base_params)",
                            "future = spectral_smooth(future, base_strength)",
                            "late_future = spectral_smooth(future, late_strength)",
                            "future[t >= late_start] = late_future[t >= late_start]",
                        ],
                    },
                    "parameter_grid": {
                        "slope_window": [5],
                        "gain": [-0.075, -0.05],
                        "decay": [70.0, 100.0],
                        "spectral_strength": [1.5],
                        "late_spectral_strength": [1.75, 2.0, 2.25],
                        "late_start": [60, 80, 100],
                    },
                    "seed_params": hypothesis.get("evidence_params", []),
                    "sandbox": {
                        "allowed_executor": "controlled_numpy_operator",
                        "primary_metric": "task1_total_score_proxy",
                        "promotion_min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                    },
                    "risks": hypothesis.get("known_risks", []),
                }
            )
    audit = audit_executable_blueprint_payload(
        {
            "schema_version": 1,
            "state_hint": BLUEPRINTS_STATE_HINT,
            "blueprints": blueprints,
        }
    )
    result = {
        "schema_version": 1,
        "state_hint": BLUEPRINTS_STATE_HINT,
        "blueprint_count": len(blueprints),
        "blueprints": blueprints,
        "executable_blueprint_audit": {
            "state_hint": audit["state_hint"],
            "decision": audit["decision"],
            "issue_count": audit["issue_count"],
            "issues": audit["issues"],
            "normalized_blueprint_count": audit["normalized_blueprint_count"],
            "next_command": audit["next_command"],
        },
        "next_command": "run-blueprint-sandbox",
    }
    _write_optional_json(output_file, result)
    return result
