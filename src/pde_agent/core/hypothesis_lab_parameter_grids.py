from __future__ import annotations

from typing import Any

from pde_agent.core.hypothesis_lab_utils import (
    HypothesisLabError,
    _normalize_retrieval_refinement_params,
)
from pde_agent.core.burgers_spectral_element import sem_params_from_any
from pde_agent.core.burgers_residual_refinement import (
    physics_residual_refinement_params_from_any,
)
from pde_agent.core.pi_deeponet_correction_head import pi_deeponet_params_from_any
from pde_agent.core.task1_short_window_corrector import (
    short_window_corrector_params_from_any,
)
from pde_agent.core.task1_front25_dynamics import front25_dynamics_params_from_any


def _residual_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    shrinkages = grid.get("residual_shrinkage")
    strengths = grid.get("residual_spectral_strength", [0.0])
    if not isinstance(shrinkages, list) or not shrinkages:
        raise HypothesisLabError("residual blueprint missing residual_shrinkage list")
    if not isinstance(strengths, list) or not strengths:
        raise HypothesisLabError(
            "residual blueprint residual_spectral_strength must be a list"
        )
    return [
        {
            "residual_shrinkage": float(shrinkage),
            "residual_spectral_strength": float(strength),
        }
        for shrinkage in shrinkages
        for strength in strengths
    ]


def _retrieval_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    ks = grid.get("k")
    modes = grid.get("mode", ["affine"])
    if not isinstance(ks, list) or not ks:
        raise HypothesisLabError("retrieval blueprint missing k list")
    if not isinstance(modes, list) or not modes:
        raise HypothesisLabError("retrieval blueprint mode must be a list")
    return [{"k": int(k), "mode": str(mode)} for k in ks for mode in modes]


def _retrieval_refinement_parameter_grid(
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    candidate_params = grid.get("candidate_params")
    if isinstance(candidate_params, list):
        rows = []
        for row in candidate_params:
            if not isinstance(row, dict):
                raise HypothesisLabError(
                    "retrieval refinement candidate_params entries must be objects"
                )
            rows.append(_normalize_retrieval_refinement_params(row))
        if not rows:
            raise HypothesisLabError(
                "retrieval refinement candidate_params must not be empty"
            )
        return rows
    ks = grid.get("k")
    modes = grid.get("mode")
    early_steps = grid.get("early_blend_steps")
    weight_starts = grid.get("persistence_weight_start")
    strengths = grid.get("spectral_strength", [0.0])
    required = {
        "k": ks,
        "mode": modes,
        "early_blend_steps": early_steps,
        "persistence_weight_start": weight_starts,
        "spectral_strength": strengths,
    }
    for key, value in required.items():
        if not isinstance(value, list) or not value:
            raise HypothesisLabError(f"retrieval refinement blueprint missing {key} list")
    rows = []
    for k in ks:
        for mode in modes:
            for steps in early_steps:
                for weight in weight_starts:
                    for strength in strengths:
                        rows.append(
                            _normalize_retrieval_refinement_params(
                                {
                                    "k": k,
                                    "mode": mode,
                                    "early_blend_steps": steps,
                                    "persistence_weight_start": weight,
                                    "spectral_strength": strength,
                                }
                            )
                        )
    return rows


def _sem_burgers_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    candidate_params = grid.get("candidate_params")
    if isinstance(candidate_params, list):
        rows = []
        for row in candidate_params:
            if not isinstance(row, dict):
                raise HypothesisLabError(
                    "SEM Burgers candidate_params entries must be objects"
                )
            rows.append(sem_params_from_any(row))
        if not rows:
            raise HypothesisLabError("SEM Burgers candidate_params must not be empty")
        return rows
    required = {
        "viscosity": grid.get("viscosity"),
        "dt": grid.get("dt"),
        "element_count": grid.get("element_count"),
        "polynomial_order": grid.get("polynomial_order"),
        "substeps": grid.get("substeps"),
        "filter_strength": grid.get("filter_strength", [0.0]),
    }
    for key, value in required.items():
        if not isinstance(value, list) or not value:
            raise HypothesisLabError(f"SEM Burgers blueprint missing {key} list")
    return [
        sem_params_from_any(
            {
                "viscosity": viscosity,
                "dt": dt,
                "element_count": element_count,
                "polynomial_order": polynomial_order,
                "substeps": substeps,
                "filter_strength": filter_strength,
            }
        )
        for viscosity in required["viscosity"]
        for dt in required["dt"]
        for element_count in required["element_count"]
        for polynomial_order in required["polynomial_order"]
        for substeps in required["substeps"]
        for filter_strength in required["filter_strength"]
    ]


def _front25_dynamics_parameter_grid(
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    candidate_params = grid.get("candidate_params")
    if isinstance(candidate_params, list):
        rows = []
        for row in candidate_params:
            if not isinstance(row, dict):
                raise HypothesisLabError(
                    "front25 dynamics candidate_params entries must be objects"
                )
            normalized = front25_dynamics_params_from_any(row)
            for key in [
                "base_k",
                "base_mode",
                "early_blend_steps",
                "persistence_weight_start",
            ]:
                if key in row:
                    normalized[key] = row[key]
            rows.append(normalized)
        if not rows:
            raise HypothesisLabError(
                "front25 dynamics candidate_params must not be empty"
            )
        return rows
    ks = grid.get("k")
    modes = grid.get("dynamics_mode")
    if not isinstance(ks, list) or not ks:
        raise HypothesisLabError("front25 dynamics blueprint missing k list")
    if not isinstance(modes, list) or not modes:
        raise HypothesisLabError(
            "front25 dynamics blueprint missing dynamics_mode list"
        )
    feature_modes = grid.get("feature_mode", ["last_velocity_gradient"])
    horizon_steps = grid.get("horizon_steps", [25])
    ridge_alphas = grid.get("ridge_alpha", [1e-4])
    correction_weights = grid.get("correction_weight", [1.0])
    rows = []
    for k in ks:
        for mode in modes:
            for feature_mode in feature_modes:
                for horizon in horizon_steps:
                    for ridge_alpha in ridge_alphas:
                        for weight in correction_weights:
                            rows.append(
                                front25_dynamics_params_from_any(
                                    {
                                        "k": k,
                                        "dynamics_mode": mode,
                                        "feature_mode": feature_mode,
                                        "horizon_steps": horizon,
                                        "ridge_alpha": ridge_alpha,
                                        "correction_weight": weight,
                                    }
                                )
                            )
    return rows


def _short_window_corrector_parameter_grid(
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    candidate_params = grid.get("candidate_params")
    if isinstance(candidate_params, list):
        rows = []
        for row in candidate_params:
            if not isinstance(row, dict):
                raise HypothesisLabError(
                    "short window corrector candidate_params entries must be objects"
                )
            normalized = short_window_corrector_params_from_any(row)
            for key in [
                "base_k",
                "base_mode",
                "early_blend_steps",
                "persistence_weight_start",
                "base_spectral_strength",
            ]:
                if key in row:
                    normalized[key] = row[key]
            rows.append(normalized)
        if not rows:
            raise HypothesisLabError(
                "short window corrector candidate_params must not be empty"
            )
        return rows
    starts = grid.get("start_step", [20])
    stops = grid.get("stop_step", [35])
    feature_counts = grid.get("feature_count", [24])
    ridge_alphas = grid.get("ridge_alpha", [1e-4])
    correction_weights = grid.get("correction_weight", [1.0])
    modes = grid.get("corrector_mode", ["context_pod_ridge"])
    required = {
        "start_step": starts,
        "stop_step": stops,
        "feature_count": feature_counts,
        "ridge_alpha": ridge_alphas,
        "correction_weight": correction_weights,
        "corrector_mode": modes,
    }
    for key, value in required.items():
        if not isinstance(value, list) or not value:
            raise HypothesisLabError(
                f"short window corrector blueprint missing {key} list"
            )
    rows = []
    for start in starts:
        for stop in stops:
            for feature_count in feature_counts:
                for ridge_alpha in ridge_alphas:
                    for weight in correction_weights:
                        for mode in modes:
                            rows.append(
                                short_window_corrector_params_from_any(
                                    {
                                        "start_step": start,
                                        "stop_step": stop,
                                        "feature_count": feature_count,
                                        "ridge_alpha": ridge_alpha,
                                        "correction_weight": weight,
                                        "corrector_mode": mode,
                                        "reference_time_stride": grid.get(
                                            "reference_time_stride",
                                            [1],
                                        )[0],
                                    }
                                )
                            )
    return rows


def _physics_residual_refinement_parameter_grid(
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    candidate_params = grid.get("candidate_params")
    if isinstance(candidate_params, list):
        rows = []
        for row in candidate_params:
            if not isinstance(row, dict):
                raise HypothesisLabError(
                    "physics residual candidate_params entries must be objects"
                )
            rows.append(physics_residual_refinement_params_from_any(row))
        if not rows:
            raise HypothesisLabError(
                "physics residual candidate_params must not be empty"
            )
        return rows
    required = {
        "base_family": grid.get("base_family", ["task1_retrieval_refinement"]),
        "viscosity": grid.get("viscosity", [0.001]),
        "dt": grid.get("dt", [0.05]),
        "dx": grid.get("dx", [1.0 / 256.0]),
        "correction_weight": grid.get("correction_weight"),
        "stop_step": grid.get("stop_step"),
        "residual_spectral_strength": grid.get("residual_spectral_strength", [0.0]),
    }
    for key, value in required.items():
        if not isinstance(value, list) or not value:
            raise HypothesisLabError(
                f"physics residual blueprint missing {key} list"
            )
    rows = []
    for base_family in required["base_family"]:
        for viscosity in required["viscosity"]:
            for dt in required["dt"]:
                for dx in required["dx"]:
                    for correction_weight in required["correction_weight"]:
                        for stop_step in required["stop_step"]:
                            for residual_strength in required[
                                "residual_spectral_strength"
                            ]:
                                row = {
                                    "base_family": str(base_family),
                                    "viscosity": float(viscosity),
                                    "dt": float(dt),
                                    "dx": float(dx),
                                    "correction_weight": float(correction_weight),
                                    "stop_step": int(stop_step),
                                    "residual_spectral_strength": float(
                                        residual_strength
                                    ),
                                }
                                if row["base_family"] == "linear_trend_spectral":
                                    row.update(
                                        {
                                            "slope_window": int(
                                                grid.get("slope_window", [5])[0]
                                            ),
                                            "gain": float(grid.get("gain", [-0.075])[0]),
                                            "decay": float(grid.get("decay", [70.0])[0]),
                                            "spectral_strength": float(
                                                grid.get("spectral_strength", [0.0])[0]
                                            ),
                                        }
                                    )
                                else:
                                    row.update(
                                        _normalize_retrieval_refinement_params(
                                            {
                                                "k": grid.get("k", [48])[0],
                                                "mode": grid.get("mode", ["affine"])[0],
                                                "early_blend_steps": grid.get(
                                                    "early_blend_steps",
                                                    [20],
                                                )[0],
                                                "persistence_weight_start": grid.get(
                                                    "persistence_weight_start",
                                                    [0.5],
                                                )[0],
                                                "spectral_strength": grid.get(
                                                    "spectral_strength",
                                                    [0.5],
                                                )[0],
                                            }
                                        )
                                    )
                                rows.append(
                                    physics_residual_refinement_params_from_any(row)
                                )
    return rows


def _pi_deeponet_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    candidate_params = grid.get("candidate_params")
    if isinstance(candidate_params, list):
        rows = []
        for row in candidate_params:
            if not isinstance(row, dict):
                raise HypothesisLabError(
                    "PI-DeepONet candidate_params entries must be objects"
                )
            rows.append(pi_deeponet_params_from_any(row))
        if not rows:
            raise HypothesisLabError("PI-DeepONet candidate_params must not be empty")
        return rows
    required = {
        "base_family": grid.get("base_family", ["task1_retrieval_refinement"]),
        "ridge_alpha": grid.get("ridge_alpha"),
        "correction_weight": grid.get("correction_weight"),
        "physics_weight": grid.get("physics_weight", [0.0]),
        "viscosity": grid.get("viscosity", [0.001]),
        "dt": grid.get("dt", [0.05]),
        "dx": grid.get("dx", [1.0 / 256.0]),
    }
    for key, value in required.items():
        if not isinstance(value, list) or not value:
            raise HypothesisLabError(f"PI-DeepONet blueprint missing {key} list")
    rows = []
    for base_family in required["base_family"]:
        for ridge_alpha in required["ridge_alpha"]:
            for correction_weight in required["correction_weight"]:
                for physics_weight in required["physics_weight"]:
                    for viscosity in required["viscosity"]:
                        for dt in required["dt"]:
                            for dx in required["dx"]:
                                row = {
                                    "base_family": str(base_family),
                                    "ridge_alpha": float(ridge_alpha),
                                    "correction_weight": float(correction_weight),
                                    "physics_weight": float(physics_weight),
                                    "viscosity": float(viscosity),
                                    "dt": float(dt),
                                    "dx": float(dx),
                                }
                                if row["base_family"] == "linear_trend_spectral":
                                    row.update(
                                        {
                                            "slope_window": int(
                                                grid.get("slope_window", [5])[0]
                                            ),
                                            "gain": float(grid.get("gain", [0.0])[0]),
                                            "decay": float(grid.get("decay", [70.0])[0]),
                                            "spectral_strength": float(
                                                grid.get("spectral_strength", [0.0])[0]
                                            ),
                                        }
                                    )
                                else:
                                    row.update(
                                        _normalize_retrieval_refinement_params(
                                            {
                                                "k": grid.get("k", [48])[0],
                                                "mode": grid.get("mode", ["affine"])[0],
                                                "early_blend_steps": grid.get(
                                                    "early_blend_steps",
                                                    [20],
                                                )[0],
                                                "persistence_weight_start": grid.get(
                                                    "persistence_weight_start",
                                                    [0.5],
                                                )[0],
                                                "spectral_strength": grid.get(
                                                    "spectral_strength",
                                                    [0.5],
                                                )[0],
                                            }
                                        )
                                    )
                                rows.append(pi_deeponet_params_from_any(row))
    return rows
