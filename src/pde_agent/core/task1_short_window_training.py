"""Task 1 short-window trainable dynamics loop for the Claude SDK agent."""


import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pde_agent.core.local_windows import build_local_windows
from pde_agent.core.task1_fno_training_policy import (
    _any_guardrail_evaluated,
    _anchor_family_challenge_candidates,
    _anchor_neighborhood_candidates,
    _anchor_only_int_values,
    _anchor_train_step_values,
    _attempted_fno_config_signatures,
    _candidate_rationale,
    _cap_positive_int_values,
    _decision_reasons,
    _dedupe,
    _diversified_model_family_values,
    _fno_candidate_record,
    _fno_config_signature,
    _fno_directive_action,
    _fno_directive_reason_codes,
    _fno_graft_evidence_summary,
    _fno_model_family_values,
    _fno_optimization_constraints,
    _fno_planning_preferences,
    _fno_regression_repair_candidates,
    _first_available_loss,
    _first_available_string,
    _fno_selection_components,
    _fno_selection_reason_codes,
    _fno_selection_score,
    _fno_training_directive,
    _guardrail_quality,
    _has_expanded_capacity_axis,
    _has_positive_late_safe_graft_evidence,
    _loss_type_list,
    _medium_training_budget_values,
    _merge_preferred_ints,
    _merge_preferred_strings,
    _normalization_list,
    _normalize_fno_candidate_config,
    _operator_policy_from_search_guidance,
    _ordered_values,
    _prefers_medium_training_budget,
    _prioritize_new_dynamics_families,
    _preferred_probe_forecast,
    _select_fno_candidate,
    _short_window_metrics,
    _string_list,
    _training_constraints,
    _training_decision,
)
from pde_agent.core.task1_short_window_common import (
    FAMILY_SELECTION_STATE_HINT,
    STATE_HINT,
    SUPPORTED_FNO_LOSS_TYPES,
    SUPPORTED_MODEL_FAMILIES,
    SUPPORTED_NORMALIZATION_TYPES,
    Task1ShortWindowTrainingError,
)
from pde_agent.core.task1_short_window_family_policy import (
    family_record as _family_record,
    family_selection_constraints as _family_selection_constraints,
    family_training_directive as _family_training_directive,
    select_family as _select_family,
)
from pde_agent.core.task1_short_window_hypothesis_bridge import (
    convert_hypothesis_algorithm_evidence_for_candidate_promotion,
    materialize_training_windows_for_hypothesis_sandbox,
)
from pde_agent.core.task1_short_window_models import (
    build_model as _build_model,
    denormalize_prediction as _denormalize_prediction,
    normalize_windows as _normalize_windows,
    parameter_count as _parameter_count,
    training_loss as _training_loss,
)


def run_task1_short_window_training_loop(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str = "tensor",
    train_sample_limit: int = 64,
    val_sample_limit: int = 32,
    context_steps: int = 10,
    forecast_steps: int = 47,
    time_reduction: int = 5,
    space_reduction: int = 4,
    model_family: str = "conv1d_residual",
    fno_modes: int = 8,
    fno_width: int = 32,
    fno_depth: int = 1,
    loss_type: str = "mse",
    normalization: str = "none",
    train_steps: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    torch = _require_torch()
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise Task1ShortWindowTrainingError(
            "model_family must be one of: "
            + ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        )
    _positive_int(train_sample_limit, "train_sample_limit")
    _positive_int(val_sample_limit, "val_sample_limit")
    _positive_int(context_steps, "context_steps")
    _positive_int(forecast_steps, "forecast_steps")
    _positive_int(train_steps, "train_steps")
    _positive_int(batch_size, "batch_size")
    _positive_int(fno_modes, "fno_modes")
    _positive_int(fno_width, "fno_width")
    _positive_int(fno_depth, "fno_depth")
    if loss_type not in SUPPORTED_FNO_LOSS_TYPES:
        raise Task1ShortWindowTrainingError(
            "loss_type must be one of: " + ", ".join(sorted(SUPPORTED_FNO_LOSS_TYPES))
        )
    if normalization not in SUPPORTED_NORMALIZATION_TYPES:
        raise Task1ShortWindowTrainingError(
            "normalization must be one of: "
            + ", ".join(sorted(SUPPORTED_NORMALIZATION_TYPES))
        )
    lr = _positive_float(learning_rate, "learning_rate")

    root = Path(output_dir)
    artifacts = {
        "train_windows_path": root / "train_windows.npz",
        "val_windows_path": root / "val_windows.npz",
        "checkpoint_path": root / "checkpoint.pt",
        "config_path": root / "config.json",
        "training_log_path": root / "training_log.json",
        "prediction_path": root / "short_window_prediction.npz",
        "evidence_path": root / "hypothesis_lab_training_evidence.json",
        "report_path": root / "task1_short_window_training_report.json",
    }
    _guard_artifacts(artifacts.values(), force=force)
    root.mkdir(parents=True, exist_ok=True)

    train_windows = build_local_windows(
        source_path=source_path,
        output_path=artifacts["train_windows_path"],
        dataset_key=dataset_key,
        start_index=0,
        sample_limit=train_sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        force=force,
    )
    val_windows = build_local_windows(
        source_path=source_path,
        output_path=artifacts["val_windows_path"],
        dataset_key=dataset_key,
        start_index=train_sample_limit,
        sample_limit=val_sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        force=force,
    )
    train_context, train_target = _load_windows(artifacts["train_windows_path"])
    val_context, val_target = _load_windows(artifacts["val_windows_path"])
    train_context_model, train_target_model, train_norm = _normalize_windows(
        context=train_context,
        target=train_target,
        normalization=normalization,
    )
    val_context_model, val_target_model, val_norm = _normalize_windows(
        context=val_context,
        target=val_target,
        normalization=normalization,
    )

    torch.manual_seed(int(seed))
    model = _build_model(
        torch=torch,
        model_family=model_family,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        space_points=int(train_context.shape[2]),
        hidden_channels=fno_width if model_family in {"fno_ar1", "fno_lite"} else 32,
        modes=fno_modes,
        depth=fno_depth,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_context_t = torch.as_tensor(train_context_model, dtype=torch.float32)
    train_target_t = torch.as_tensor(train_target_model, dtype=torch.float32)
    val_context_t = torch.as_tensor(val_context_model, dtype=torch.float32)
    val_target_t = torch.as_tensor(val_target_model, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    loss_curve = {"train": [], "val": []}
    start_time = time.perf_counter()
    for _ in range(train_steps):
        model.train()
        indices = rng.choice(
            train_context.shape[0],
            size=min(batch_size, train_context.shape[0]),
            replace=False,
        )
        batch_context = train_context_t[indices]
        batch_target = train_target_t[indices]
        optimizer.zero_grad()
        prediction = model(batch_context)
        train_loss = _training_loss(
            torch=torch,
            prediction=prediction,
            target=batch_target,
            loss_type=loss_type,
        )
        if not torch.isfinite(train_loss):
            raise Task1ShortWindowTrainingError("training loss is non-finite")
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_context_t)
            val_loss = _training_loss(
                torch=torch,
                prediction=val_prediction,
                target=val_target_t,
                loss_type=loss_type,
            )
        loss_curve["train"].append(float(train_loss.detach().cpu().item()))
        loss_curve["val"].append(float(val_loss.detach().cpu().item()))
    train_runtime = time.perf_counter() - start_time
    with torch.no_grad():
        prediction_model = model(val_context_t).detach().cpu().numpy().astype(np.float32)
    prediction = _denormalize_prediction(
        prediction=prediction_model,
        stats=val_norm,
        normalization=normalization,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_family": model_family,
                "operator_type": model.operator_type,
                "modes": fno_modes,
                "width": fno_width,
                "depth": fno_depth,
                "context_steps": context_steps,
                "forecast_steps": forecast_steps,
                "space_points": int(train_context.shape[2]),
                "train_steps": train_steps,
                "loss_type": loss_type,
                "normalization": normalization,
                "learning_rate": lr,
                "seed": int(seed),
            },
        },
        artifacts["checkpoint_path"],
    )
    np.savez_compressed(
        artifacts["prediction_path"],
        prediction=prediction,
        context=val_context,
        target=val_target,
    )
    config = {
        "source_path": str(source_path),
        "dataset_key": dataset_key,
        "model_family": model_family,
        "fno_modes": fno_modes,
        "fno_width": fno_width,
        "fno_depth": fno_depth,
        "loss_type": loss_type,
        "normalization": normalization,
        "context_steps": context_steps,
        "forecast_steps": forecast_steps,
        "time_reduction": time_reduction,
        "space_reduction": space_reduction,
        "train_sample_limit": train_sample_limit,
        "val_sample_limit": val_sample_limit,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": int(seed),
    }
    metrics = _short_window_metrics(prediction=prediction, target=val_target)
    decision = _training_decision(loss_curve=loss_curve, metrics=metrics)
    constraints = _training_constraints(
        model_family=model_family,
        decision=decision,
        metrics=metrics,
    )
    evidence = {
        "schema_version": 1,
        "state_hint": "TASK1_SHORT_WINDOW_TRAINING_EVIDENCE_RECORDED",
        "source_type": "task1_short_window_training",
        "model_family": model_family,
        "loss": {
            "name": loss_type,
            "initial_train": loss_curve["train"][0],
            "final_train": loss_curve["train"][-1],
            "initial_val": loss_curve["val"][0],
            "final_val": loss_curve["val"][-1],
        },
        "normalization": normalization,
        "horizon": {
            "context_steps": context_steps,
            "forecast_steps": forecast_steps,
        },
        "train_steps": train_steps,
        "metrics": metrics,
        "decision": decision,
        "reason_codes": _decision_reasons(decision=decision, metrics=metrics),
        "next_generation_constraints": constraints,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "data": {"train": train_windows, "val": val_windows},
        "model": {
            "model_family": model_family,
            "operator_type": model.operator_type,
            "config": {
                "modes": fno_modes,
                "width": fno_width,
                "depth": fno_depth,
                "loss_type": loss_type,
                "normalization": normalization,
                "forecast_steps": forecast_steps,
                "train_steps": train_steps,
            },
            "parameter_count": _parameter_count(model),
        },
        "training": {
            "train_steps": train_steps,
            "runtime_seconds": train_runtime,
            "loss_curve": loss_curve,
        },
        "metrics": metrics,
        "evidence": evidence,
        "next_generation_constraints": constraints,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "next_command": "run-task1-short-window-training-loop"
        if decision != "abandon_model_family"
        else "build-hypothesis-algorithm-design",
    }
    artifacts["config_path"].write_text(
        json.dumps(config, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["training_log_path"].write_text(
        json.dumps(
            {
                "loss_curve": loss_curve,
                "runtime_seconds": train_runtime,
                "decision": decision,
                "metrics": metrics,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts["evidence_path"].write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_task1_short_window_family_selection_loop(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str = "tensor",
    model_families: list[str] | None = None,
    train_sample_limit: int = 64,
    val_sample_limit: int = 32,
    context_steps: int = 10,
    forecast_steps: int = 47,
    time_reduction: int = 5,
    space_reduction: int = 4,
    train_steps: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    families = model_families or ["conv1d_residual", "fno_lite"]
    if not families:
        raise Task1ShortWindowTrainingError("model_families must not be empty")
    unknown = sorted(set(families) - SUPPORTED_MODEL_FAMILIES)
    if unknown:
        raise Task1ShortWindowTrainingError(
            "unsupported model families: " + ", ".join(unknown)
        )
    root = Path(output_dir)
    artifacts = {
        "comparison_report_path": root / "task1_short_window_family_selection_report.json",
        "comparison_evidence_path": root / "hypothesis_lab_family_selection_evidence.json",
    }
    _guard_artifacts(artifacts.values(), force=force)
    root.mkdir(parents=True, exist_ok=True)

    reports = []
    for index, family in enumerate(families):
        family_dir = root / f"family_{index:02d}_{family}"
        reports.append(
            run_task1_short_window_training_loop(
                source_path=source_path,
                output_dir=family_dir,
                dataset_key=dataset_key,
                train_sample_limit=train_sample_limit,
                val_sample_limit=val_sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                model_family=family,
                train_steps=train_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + index,
                force=force,
            )
        )
    records = [_family_record(report) for report in reports]
    selected = _select_family(records)
    directive = _family_training_directive(selected=selected, records=records)
    constraints = _family_selection_constraints(directive=directive)
    evidence = {
        "schema_version": 1,
        "state_hint": "TASK1_SHORT_WINDOW_FAMILY_SELECTION_EVIDENCE_RECORDED",
        "source_type": "task1_short_window_family_selection",
        "selected_family": selected["model_family"],
        "selection_reason_codes": selected["selection_reason_codes"],
        "training_directive": directive,
        "family_records": records,
        "next_generation_constraints": constraints,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    report = {
        "schema_version": 1,
        "state_hint": FAMILY_SELECTION_STATE_HINT,
        "source_path": str(source_path),
        "data_protocol": {
            "dataset_key": dataset_key,
            "context_steps": context_steps,
            "forecast_steps": forecast_steps,
            "time_reduction": time_reduction,
            "space_reduction": space_reduction,
            "train_sample_limit": train_sample_limit,
            "val_sample_limit": val_sample_limit,
        },
        "family_records": records,
        "selected_family": selected["model_family"],
        "selection_reason_codes": selected["selection_reason_codes"],
        "training_directive": directive,
        "evidence": evidence,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "next_command": "build-hypothesis-algorithm-design",
    }
    artifacts["comparison_evidence_path"].write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["comparison_report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def plan_fno_training_candidates(
    *,
    evidence_paths: list[str | Path],
    output_path: str | Path | None,
    max_candidates: int,
    modes: list[int],
    widths: list[int],
    depths: list[int],
    loss_types: list[str],
    forecast_steps_values: list[int],
    train_steps_values: list[int],
    normalizations: list[str] | None = None,
    blocked_families: list[str] | None = None,
    search_guidance: list[dict[str, Any]] | None = None,
    budget_adjustment: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    _positive_int(max_candidates, "max_candidates")
    mode_values = _positive_int_list(modes, "modes")
    width_values = _positive_int_list(widths, "widths")
    depth_values = _positive_int_list(depths, "depths")
    forecast_values = _positive_int_list(forecast_steps_values, "forecast_steps_values")
    train_values = _positive_int_list(train_steps_values, "train_steps_values")
    loss_values = _loss_type_list(loss_types)
    normalization_values = _normalization_list(normalizations or ["none"])
    evidences = [_load_json(Path(path), "FNO training evidence") for path in evidence_paths]
    attempted = _attempted_fno_config_signatures(evidences)
    preferences = _fno_planning_preferences(evidences)
    operator_policy = _operator_policy_from_search_guidance(search_guidance or [])
    if operator_policy:
        policy_losses = [
            value
            for value in _string_list(operator_policy.get("preferred_loss_types"))
            if value in SUPPORTED_FNO_LOSS_TYPES
        ]
        loss_values = _dedupe([*policy_losses, *loss_values])
        normalization_values = _merge_preferred_strings(
            normalization_values,
            operator_policy.get("preferred_normalizations", []),
        )
        forecast_values = _merge_preferred_ints(
            forecast_values,
            operator_policy.get("preferred_forecast_steps", []),
        )
        policy_budget = operator_policy.get("budget_adjustments")
        if isinstance(policy_budget, dict):
            budget_adjustment = {
                **policy_budget,
                **(budget_adjustment or {}),
            }
        preferences = dict(preferences)
        preferences["loss_types"] = _dedupe(
            [*policy_losses, *preferences.get("loss_types", [])]
        )
        policy_families = [
            value
            for value in _string_list(operator_policy.get("preferred_model_families"))
            if value in SUPPORTED_MODEL_FAMILIES
        ]
        preferences["model_families"] = _dedupe(
            [*policy_families, *preferences.get("model_families", [])]
        )
        if operator_policy.get("next_action") == "request_new_dynamics_source":
            preferences["requires_new_dynamics_source"] = True
        required_axes = set(_string_list(operator_policy.get("required_axis_changes")))
        if "model_family" in required_axes:
            preferences["requires_model_family_change"] = True
    early_middle_source_mode = (
        forecast_values
        and max(forecast_values) <= 95
        and 190 not in {int(value) for value in forecast_values}
    )
    if early_middle_source_mode:
        preferences = dict(preferences)
        preferences["forecast_steps"] = [
            value for value in preferences.get("forecast_steps", []) if int(value) <= 95
        ]
    if not early_middle_source_mode:
        mode_values = _merge_preferred_ints(mode_values, preferences["modes"])
        width_values = _merge_preferred_ints(width_values, preferences["widths"])
        depth_values = _merge_preferred_ints(depth_values, preferences["depths"])
        forecast_values = _merge_preferred_ints(
            forecast_values,
            preferences["forecast_steps"],
        )
        train_values = _merge_preferred_ints(train_values, preferences["train_steps"])
        normalization_values = _merge_preferred_strings(
            normalization_values,
            preferences["normalizations"],
        )
    rows = []
    candidates = []
    model_family_values = _fno_model_family_values(preferences)
    if preferences.get("requires_model_family_change"):
        model_family_values = _diversified_model_family_values(model_family_values)
    ordered_forecasts = _ordered_values(forecast_values, preferences["forecast_steps"])
    ordered_losses = _ordered_values(loss_values, preferences["loss_types"])
    ordered_modes = _ordered_values(mode_values, preferences["modes"])
    ordered_widths = _ordered_values(width_values, preferences["widths"])
    ordered_depths = _ordered_values(depth_values, preferences["depths"])
    ordered_train_steps = _ordered_values(train_values, preferences["train_steps"])
    ordered_normalizations = _ordered_values(
        normalization_values,
        preferences["normalizations"],
    )
    repair_mode = bool(
        preferences.get("regression_mode")
        or preferences.get("capacity_expansion_regression_memory")
        or operator_policy.get("next_action") == "repair_regression"
    )
    health_repair_mode = bool(preferences.get("health_requested_regression_repair"))
    if not health_repair_mode:
        for anchor_config in preferences.get("anchor_configs", []):
            if isinstance(anchor_config, dict):
                candidates.append(
                    {
                        **anchor_config,
                        "rationale": [
                            "training_memory_anchor_reuse",
                            "new_global_best_training_candidate",
                        ],
                    }
                )
                candidates.extend(_anchor_family_challenge_candidates(anchor_config))
                candidates.extend(_anchor_neighborhood_candidates(anchor_config))
    if repair_mode:
        new_dynamics_candidates = []
        if "fno_ar1" in model_family_values:
            new_dynamics_candidates.append(
                {
                    "model_family": "fno_ar1",
                    "modes": max(mode_values),
                    "width": max(width_values),
                    "depth": max(depth_values),
                    "loss_type": ordered_losses[0],
                    "normalization": ordered_normalizations[0],
                    "forecast_steps": _preferred_probe_forecast(
                        ordered_forecasts,
                        preferences,
                    ),
                    "train_steps": min(max(train_values), 100),
                    "rationale": [
                        "new_dynamics_family_short_horizon_probe",
                        "autoregressive_rollout_source",
                    ],
                }
            )
        repair_candidates = _fno_regression_repair_candidates(
            preferences=preferences,
            loss_values=ordered_losses,
        )
        if "conv1d_residual" in model_family_values:
            new_dynamics_candidates.append(
                {
                    "model_family": "conv1d_residual",
                    "modes": min(mode_values),
                    "width": min(width_values),
                    "depth": 1,
                    "loss_type": ordered_losses[0],
                    "normalization": ordered_normalizations[0],
                    "forecast_steps": _preferred_probe_forecast(
                        ordered_forecasts,
                        preferences,
                    ),
                    "train_steps": min(max(train_values), 100),
                    "rationale": [
                        "new_dynamics_family_short_horizon_probe",
                        "local_residual_convolution_source",
                    ],
                }
            )
        if preferences.get("requires_new_dynamics_source"):
            candidates.extend(new_dynamics_candidates)
            candidates.extend(repair_candidates)
        else:
            candidates.extend(repair_candidates)
            candidates.extend(new_dynamics_candidates)
        if health_repair_mode:
            for anchor_config in preferences.get("anchor_configs", []):
                if isinstance(anchor_config, dict):
                    candidates.extend(_anchor_family_challenge_candidates(anchor_config))
                    candidates.extend(_anchor_neighborhood_candidates(anchor_config))
    elif not early_middle_source_mode:
        candidates.append(
            {
                "model_family": "fno_lite",
                "modes": max(mode_values),
                "width": max(width_values),
                "depth": max(depth_values),
                "loss_type": ordered_losses[0],
                "normalization": ordered_normalizations[0],
                "forecast_steps": max(forecast_values),
                "train_steps": max(train_values),
                "rationale": [
                    "capacity_axis_coverage",
                    "full_horizon_middle_late_guardrail",
                ],
            }
        )
    elif _has_expanded_capacity_axis(
        mode_values=mode_values,
        width_values=width_values,
        depth_values=depth_values,
        train_values=train_values,
    ):
        candidates.append(
            {
                "model_family": "fno_lite",
                "modes": max(mode_values),
                "width": max(width_values),
                "depth": max(depth_values),
                "loss_type": _first_available_loss(
                    ordered_losses,
                    preferred=("horizon_weighted_mse", "early_weighted_mse"),
                ),
                "normalization": _first_available_string(
                    ordered_normalizations,
                    preferred=("sample_standard", "none"),
                ),
                "forecast_steps": min(ordered_forecasts),
                "train_steps": max(train_values),
                "rationale": [
                    "early_middle_capacity_axis_coverage",
                    "positive_graft_expand_capacity_within_guard",
                ],
            }
        )
    if repair_mode:
        mode_values = _anchor_only_int_values(preferences, "modes", mode_values)
        width_values = _anchor_only_int_values(preferences, "widths", width_values)
        train_values = _anchor_train_step_values(preferences, train_values)
        if (
            preferences.get("requires_new_dynamics_source")
            or preferences.get("requires_model_family_change")
        ):
            model_family_values = _prioritize_new_dynamics_families(model_family_values)
        else:
            model_family_values = ["fno_lite"]
        ordered_modes = _ordered_values(mode_values, preferences["modes"])
        ordered_widths = _ordered_values(width_values, preferences["widths"])
        ordered_train_steps = _ordered_values(train_values, preferences["train_steps"])
    coverage_count = max(
        len(ordered_forecasts),
        len(ordered_normalizations),
        len(ordered_losses),
        min(2, len(ordered_modes)),
        min(2, len(ordered_widths)),
        min(2, len(ordered_depths)),
        min(2, len(ordered_train_steps)),
    )
    for index in range(coverage_count):
        model_family = model_family_values[index % len(model_family_values)]
        forecast_steps = ordered_forecasts[index % len(ordered_forecasts)]
        loss_type = ordered_losses[index % len(ordered_losses)]
        normalization_value = ordered_normalizations[index % len(ordered_normalizations)]
        modes_value = ordered_modes[index % min(2, len(ordered_modes))]
        width = ordered_widths[index % min(2, len(ordered_widths))]
        depth = ordered_depths[index % min(2, len(ordered_depths))]
        train_steps = ordered_train_steps[index % min(2, len(ordered_train_steps))]
        candidates.append(
            {
                "model_family": model_family,
                "modes": modes_value,
                "width": width,
                "depth": depth,
                "loss_type": loss_type,
                "normalization": normalization_value,
                "forecast_steps": forecast_steps,
                "train_steps": train_steps,
                "rationale": _candidate_rationale(
                    loss_type=loss_type,
                    forecast_steps=forecast_steps,
                    preferences=preferences,
                ),
            }
        )
    for model_family in model_family_values:
        for forecast_steps in ordered_forecasts:
            for loss_type in ordered_losses:
                for normalization_value in ordered_normalizations:
                    for modes_value in ordered_modes:
                        for width in ordered_widths:
                            for depth in ordered_depths:
                                for train_steps in ordered_train_steps:
                                    candidates.append(
                                        {
                                            "model_family": model_family,
                                            "modes": modes_value,
                                            "width": width,
                                            "depth": depth,
                                            "loss_type": loss_type,
                                            "normalization": normalization_value,
                                            "forecast_steps": forecast_steps,
                                            "train_steps": train_steps,
                                            "rationale": _candidate_rationale(
                                                loss_type=loss_type,
                                                forecast_steps=forecast_steps,
                                                preferences=preferences,
                                            ),
                                        }
                                    )
    seen = set()
    for config in candidates:
        signature = _fno_config_signature(config)
        if signature in seen:
            continue
        seen.add(signature)
        if _fno_config_signature(config) in attempted:
            continue
        config = dict(config)
        config["candidate_id"] = f"fno_train_candidate_{len(rows) + 1:03d}"
        rows.append(config)
        if len(rows) >= max_candidates:
            break

    applied_constraints: dict[str, Any] = {}
    if blocked_families:
        blocked_set = set(blocked_families)
        rows = [
            row for row in rows
            if row.get("model_family") not in blocked_set
        ]
        applied_constraints["blocked_families"] = list(blocked_set)
    if budget_adjustment:
        train_steps_delta = budget_adjustment.get("train_steps_delta")
        if isinstance(train_steps_delta, int):
            for row in rows:
                current = row.get("train_steps", 2)
                if isinstance(current, int):
                    row["train_steps"] = max(1, current + train_steps_delta)
            applied_constraints["train_steps_delta"] = train_steps_delta
        width_delta = budget_adjustment.get("width_delta")
        if isinstance(width_delta, int):
            for row in rows:
                current = row.get("width", 16)
                if isinstance(current, int):
                    row["width"] = max(4, current + width_delta)
            applied_constraints["width_delta"] = width_delta
    rows = [_normalize_fno_candidate_config(row) for row in rows]
    if search_guidance:
        guidance_types = set()
        for guidance in search_guidance:
            if isinstance(guidance, dict):
                guidance_type = guidance.get("guidance_type")
                if guidance_type:
                    guidance_types.add(guidance_type)
        if guidance_types:
            applied_constraints["search_guidance_types"] = sorted(guidance_types)
    if operator_policy:
        applied_constraints["operator_policy"] = {
            "next_action": operator_policy.get("next_action"),
            "required_axis_changes": operator_policy.get("required_axis_changes", []),
            "reason_codes": operator_policy.get("reason_codes", []),
        }
    if preferences.get("health_feedback_actions"):
        applied_constraints["health_feedback_actions"] = list(
            preferences["health_feedback_actions"]
        )

    report = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
        "evidence_paths": [str(path) for path in evidence_paths],
        "candidate_configs": rows,
        "planning_preferences": preferences,
        "early_middle_source_mode": early_middle_source_mode,
        "avoided_config_count": len(attempted),
        "applied_constraints": applied_constraints,
        "next_command": "run-fno-training-optimization-loop",
    }
    _write_optional_json(output_file, report)
    return report


def run_fno_training_optimization_loop(
    *,
    candidate_plan_path: str | Path,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    time_reduction: int,
    space_reduction: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    plan = _load_json(Path(candidate_plan_path), "FNO training candidate plan")
    if plan.get("state_hint") != "FNO_TRAINING_CANDIDATES_PLANNED":
        raise Task1ShortWindowTrainingError("candidate plan has invalid state_hint")
    root = Path(output_dir)
    artifacts = {
        "optimization_report_path": root / "fno_training_optimization_report.json",
    }
    _guard_artifacts(artifacts.values(), force=force)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, config in enumerate(plan.get("candidate_configs", [])):
        if not isinstance(config, dict):
            continue
        candidate_dir = root / f"candidate_{index:03d}_{config['candidate_id']}"
        report = run_task1_short_window_training_loop(
            source_path=source_path,
            output_dir=candidate_dir,
            dataset_key=dataset_key,
            train_sample_limit=train_sample_limit,
            val_sample_limit=val_sample_limit,
            context_steps=context_steps,
            forecast_steps=int(config["forecast_steps"]),
            time_reduction=time_reduction,
            space_reduction=space_reduction,
            model_family=str(config.get("model_family", "fno_lite")),
            fno_modes=int(config["modes"]),
            fno_width=int(config["width"]),
            fno_depth=int(config["depth"]),
            loss_type=str(config["loss_type"]),
            normalization=str(config.get("normalization", "none")),
            train_steps=int(config["train_steps"]),
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + index,
            force=force,
        )
        record = _fno_candidate_record(config=config, report=report)
        records.append(record)
    if not records:
        raise Task1ShortWindowTrainingError("candidate plan produced no runnable records")
    selected = _select_fno_candidate(records)
    directive = _fno_training_directive(selected=selected, records=records)
    report = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
        "candidate_plan_path": str(candidate_plan_path),
        "candidate_records": records,
        "selected_candidate": selected,
        "training_directive": directive,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "next_command": "record-fno-training-optimization-evidence",
    }
    artifacts["optimization_report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def record_fno_training_optimization_evidence(
    *,
    optimization_report_path: str | Path,
    output_path: str | Path | None,
    graft_report_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    report = _load_json(Path(optimization_report_path), "FNO optimization report")
    if report.get("state_hint") != "FNO_TRAINING_OPTIMIZATION_LOOP_RAN":
        raise Task1ShortWindowTrainingError("optimization report has invalid state_hint")
    selected = _select_fno_candidate(report["candidate_records"])
    directive = report["training_directive"]
    graft_evidence = (
        _fno_graft_evidence_summary(Path(graft_report_path))
        if graft_report_path is not None
        else None
    )
    evidence = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
        "source_type": "fno_training_optimization",
        "optimization_report_path": str(optimization_report_path),
        "graft_report_path": str(graft_report_path) if graft_report_path is not None else None,
        "graft_evidence": graft_evidence,
        "selected_candidate": selected,
        "training_directive": directive,
        "candidate_records": report["candidate_records"],
        "next_generation_constraints": _fno_optimization_constraints(
            selected=selected,
            directive=directive,
            records=report["candidate_records"],
            graft_evidence=graft_evidence,
        ),
        "next_command": "build-hypothesis-algorithm-design",
    }
    _write_optional_json(output_file, evidence)
    return evidence


def build_fno_training_harness_contract(
    *,
    output_path: str | Path | None,
    workspace_dir: str | Path,
    source_path: str | Path,
    baseline_prediction_path: str | Path | None = None,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    previous_evidence_paths: list[str | Path] | None = None,
    dataset_key: str = "tensor",
    train_sample_limit: int = 16,
    val_sample_limit: int = 8,
    context_steps: int = 10,
    time_reduction: int = 5,
    space_reduction: int = 4,
    batch_size: int = 4,
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
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    root = Path(workspace_dir)
    source = Path(source_path)
    evidence_paths = [Path(path) for path in previous_evidence_paths or []]
    mode_values = _positive_int_list(modes or [4, 8], "modes")
    width_values = _positive_int_list(widths or [16, 32], "widths")
    depth_values = _positive_int_list(depths or [1], "depths")
    forecast_values = _positive_int_list(
        forecast_steps_values or [12, 20],
        "forecast_steps_values",
    )
    horizon_guard = _source_horizon_guard(
        source_path=source,
        dataset_key=dataset_key,
        context_steps=context_steps,
        time_reduction=time_reduction,
    )
    if horizon_guard is not None:
        max_forecast_steps = int(horizon_guard["max_forecast_steps"])
        forecast_values = [
            value for value in forecast_values if int(value) <= max_forecast_steps
        ]
        if not forecast_values:
            raise Task1ShortWindowTrainingError(
                "forecast_steps_values has no values compatible with "
                f"source horizon after reduction; max_forecast_steps={max_forecast_steps}"
            )
    budget_repair = _repair_training_sample_budget(
        source_path=source,
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        batch_size=batch_size,
    )
    train_sample_limit = int(budget_repair["resolved_train_sample_limit"])
    val_sample_limit = int(budget_repair["resolved_val_sample_limit"])
    batch_size = int(budget_repair["resolved_batch_size"])
    train_values = _positive_int_list(train_steps_values or [1, 2], "train_steps_values")
    loss_values = _loss_type_list(
        loss_types
        or ["mse", "early_weighted_mse", "horizon_weighted_mse", "spectral_mse"]
    )
    normalization_values = _normalization_list(normalizations or ["none"])
    budget = {
        "max_candidates": max_candidates,
        "train_sample_limit": train_sample_limit,
        "val_sample_limit": val_sample_limit,
        "context_steps": context_steps,
        "time_reduction": time_reduction,
        "space_reduction": space_reduction,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "normalizations": normalization_values,
        "training_contract_repair": budget_repair,
    }
    data = {
        "source_path": str(source),
        "dataset_key": dataset_key,
        "baseline_prediction_path": str(baseline_prediction_path)
        if baseline_prediction_path is not None
        else None,
        "task1_test_path": str(task1_test_path) if task1_test_path is not None else None,
        "task1_target_path": str(task1_target_path)
        if task1_target_path is not None
        else None,
    }
    graft_inputs = _resolve_fno_graft_inputs(
        baseline_prediction_path=baseline_prediction_path,
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        dataset_key=dataset_key,
    )
    if graft_inputs is not None:
        data["graft_contract_repair"] = graft_inputs["repair"]
    paths = {
        "candidate_plan_path": str(root / "fno_training_plan.json"),
        "optimization_dir": str(root / "optimization"),
        "optimization_report_path": str(
            root / "optimization" / "fno_training_optimization_report.json"
        ),
        "optimization_evidence_path": str(root / "fno_training_optimization_evidence.json"),
        "algorithm_design_path": str(root / "hypothesis_algorithm_design.json"),
        "algorithm_design_validation_path": str(root / "hypothesis_validation_result.json"),
        "fno_graft_dir": str(root / "fno_graft_evaluation"),
        "fno_graft_report_path": str(
            root / "fno_graft_evaluation" / "task1_fno_graft_evaluator_report.json"
        ),
    }
    actions = [
        {
            "action_id": "plan_fno_training_candidates",
            "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
            "command": _join_command(
                [
                    "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent plan-fno-training-candidates",
                    f"--output-path {paths['candidate_plan_path']}",
                    *[
                        f"--evidence-path {path}"
                        for path in evidence_paths
                    ],
                    f"--max-candidates {max_candidates}",
                    *[f"--mode {value}" for value in mode_values],
                    *[f"--width {value}" for value in width_values],
                    *[f"--depth {value}" for value in depth_values],
                    *[f"--loss-type {value}" for value in loss_values],
                    *[f"--normalization {value}" for value in normalization_values],
                    *[
                        f"--forecast-steps-value {value}"
                        for value in forecast_values
                    ],
                    *[f"--train-steps-value {value}" for value in train_values],
                    "--force",
                ]
            ),
            "writes": [paths["candidate_plan_path"]],
            "next_action_options": ["run_fno_training_optimization", "stop_and_report"],
        },
        {
            "action_id": "run_fno_training_optimization",
            "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
            "command": _join_command(
                [
                    "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent run-fno-training-optimization-loop",
                    f"--candidate-plan-path {paths['candidate_plan_path']}",
                    f"--source-path {source}",
                    f"--output-dir {paths['optimization_dir']}",
                    f"--dataset-key {dataset_key}",
                    f"--train-sample-limit {train_sample_limit}",
                    f"--val-sample-limit {val_sample_limit}",
                    f"--context-steps {context_steps}",
                    f"--time-reduction {time_reduction}",
                    f"--space-reduction {space_reduction}",
                    f"--batch-size {batch_size}",
                    f"--learning-rate {learning_rate}",
                    f"--seed {seed}",
                    "--force",
                ]
            ),
            "reads": [paths["candidate_plan_path"], str(source)],
            "writes": [paths["optimization_report_path"]],
            "next_action_options": [
                "run_task1_fno_graft_from_optimization_report",
                "record_fno_training_optimization_evidence",
                "stop_and_report",
            ],
        },
        *_fno_graft_harness_actions(
            paths=paths,
            graft_inputs=graft_inputs,
        ),
        {
            "action_id": "record_fno_training_optimization_evidence",
            "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
            "command": _join_command(
                [
                    "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent record-fno-training-optimization-evidence",
                    f"--optimization-report-path {paths['optimization_report_path']}",
                    f"--output-path {paths['optimization_evidence_path']}",
                    *(
                        [f"--graft-report-path {paths['fno_graft_report_path']}"]
                        if graft_inputs is not None
                        else []
                    ),
                    "--force",
                ]
            ),
            "reads": [
                paths["optimization_report_path"],
                *(
                    [paths["fno_graft_report_path"]]
                    if graft_inputs is not None
                    else []
                ),
            ],
            "writes": [paths["optimization_evidence_path"]],
            "next_action_options": [
                "build_hypothesis_algorithm_design",
                "plan_fno_training_candidates",
                "stop_and_report",
            ],
        },
        {
            "action_id": "build_hypothesis_algorithm_design",
            "state_hint": "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
            "command": _join_command(
                [
                    "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent build-hypothesis-algorithm-design",
                    f"--output-path {paths['algorithm_design_path']}",
                    f"--evidence-path {paths['optimization_evidence_path']}",
                    "--min-hypothesis-count 5",
                    "--min-mechanism-family-count 4",
                    "--force",
                ]
            ),
            "reads": [paths["optimization_evidence_path"]],
            "writes": [paths["algorithm_design_path"]],
            "next_action_options": [
                "validate_hypothesis_algorithm_design",
                "stop_and_report",
            ],
        },
        {
            "action_id": "validate_hypothesis_algorithm_design",
            "state_hint": "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED",
            "command": _join_command(
                [
                    "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent validate-hypothesis-algorithm-design",
                    f"--design-path {paths['algorithm_design_path']}",
                    f"--output-path {paths['algorithm_design_validation_path']}",
                    "--min-hypothesis-count 5",
                    "--min-mechanism-family-count 4",
                    "--force",
                ]
            ),
            "reads": [paths["algorithm_design_path"]],
            "writes": [paths["algorithm_design_validation_path"]],
            "next_action_options": ["stop_and_report"],
        },
        {
            "action_id": "stop_and_report",
            "state_hint": "CLAUDE_AGENT_SDK_REPORTED_DECISION",
            "command": None,
            "reason": "Use when evidence is sufficient, budget is exhausted, or an upstream action fails.",
        },
    ]
    report = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_HARNESS_CONTRACT_BUILT",
        "workspace_dir": str(root),
        "data_contract": data,
        "budget_contract": budget,
        "candidate_space": {
            "modes": mode_values,
            "widths": width_values,
            "depths": depth_values,
            "loss_types": loss_values,
            "normalizations": normalization_values,
            "forecast_steps_values": forecast_values,
            "train_steps_values": train_values,
        },
        "candidate_space_guards": {
            "source_horizon": horizon_guard,
        },
        "paths": paths,
        "previous_evidence_paths": [str(path) for path in evidence_paths],
        "available_actions": actions,
        "recommended_policy": {
            "not_a_fixed_pipeline": True,
            "choose_next_action_from_evidence": True,
            "do_not_probe_cli_help": True,
            "do_not_create_directories_with_shell": True,
            "stop_conditions": [
                "action_failed_without_clear_parameter_repair",
                "graft_evaluation_shows_no_positive_delta_after_multiple_training_sources",
                "validation_accepted",
                "budget_exhausted",
            ],
            "score_objective": (
                "Do not judge trained FNO as a standalone full replacement. "
                "Prefer a trained dynamics source only if graft evaluation improves "
                "the stronger baseline under late non-regression."
            ),
        },
        "next_command": (
            "Claude SDK agent should choose an available action from evidence, "
            "not blindly run all actions."
        ),
    }
    _write_optional_json(output_file, report)
    return report


def build_fno_training_iteration_contract_from_evidence(
    *,
    previous_evidence_path: str | Path,
    previous_evidence_paths: list[str | Path] | None = None,
    output_dir: str | Path,
    workspace_dir: str | Path,
    source_path: str | Path,
    baseline_prediction_path: str | Path | None = None,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    dataset_key: str = "tensor",
    train_sample_limit: int = 32,
    val_sample_limit: int = 16,
    context_steps: int = 10,
    time_reduction: int = 5,
    space_reduction: int = 4,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 0,
    max_candidates: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    evidence_paths = [
        Path(path)
        for path in (previous_evidence_paths or [previous_evidence_path])
    ]
    evidences = []
    for evidence_path in evidence_paths:
        evidence = _load_json(evidence_path, "FNO training optimization evidence")
        if evidence.get("state_hint") != "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED":
            raise Task1ShortWindowTrainingError("previous evidence has invalid state_hint")
        evidences.append(evidence)
    if max_candidates < 4:
        max_candidates = 4
    preferences = _fno_planning_preferences(evidences if isinstance(evidences, list) else [])
    mode_values = _merge_preferred_ints([4, 8], preferences["modes"])
    width_values = _merge_preferred_ints([16, 32], preferences["widths"])
    depth_values = _merge_preferred_ints([1], preferences["depths"])
    if (
        baseline_prediction_path is not None
        and task1_test_path is not None
        and task1_target_path is not None
    ):
        forecast_values = _merge_preferred_ints([47, 95], preferences["forecast_steps"])
        forecast_values = [value for value in forecast_values if int(value) <= 95]
    else:
        forecast_values = _merge_preferred_ints([47, 95, 190], preferences["forecast_steps"])
    train_values = _merge_preferred_ints([1, 2, 20], preferences["train_steps"])
    normalization_values = _merge_preferred_strings(
        ["none", "sample_standard"],
        preferences["normalizations"],
    )
    graft_evaluation_available = (
        baseline_prediction_path is not None
        and task1_test_path is not None
        and task1_target_path is not None
    )
    if (
        not graft_evaluation_available
        and _prefers_medium_training_budget(preferences)
    ):
        train_values = _medium_training_budget_values(train_values)
    if graft_evaluation_available:
        if _has_positive_late_safe_graft_evidence(evidences):
            max_candidates = max(max_candidates, 6)
            mode_values = _merge_preferred_ints(mode_values, [12, 16])
            width_values = _merge_preferred_ints(width_values, [48, 64])
            depth_values = _merge_preferred_ints(depth_values, [2, 3, 4])
            train_values = _merge_preferred_ints(train_values, [50, 100, 150])
            mode_values = _cap_positive_int_values(
                mode_values,
                maximum=16,
                fallback=[4, 8, 12, 16],
            )
            width_values = _cap_positive_int_values(
                width_values,
                maximum=64,
                fallback=[16, 32, 48, 64],
            )
            depth_values = _cap_positive_int_values(
                depth_values,
                maximum=4,
                fallback=[1, 2, 3, 4],
            )
            train_values = _cap_positive_int_values(
                train_values,
                maximum=150,
                fallback=[20, 50, 100, 150],
            )
        else:
            mode_values = _cap_positive_int_values(mode_values, maximum=8, fallback=[4, 8])
            width_values = _cap_positive_int_values(width_values, maximum=32, fallback=[16, 32])
            depth_values = _cap_positive_int_values(depth_values, maximum=2, fallback=[1, 2])
            train_values = _cap_positive_int_values(
                train_values,
                maximum=50,
                fallback=[1, 2, 20],
            )
    root = Path(output_dir)
    _guard_artifacts(
        [
            root / "fno_training_harness_contract.json",
            root / "claude_fno_training_iteration_prompt.md",
            root / "fno_training_iteration_contract.json",
        ],
        force=force,
    )
    root.mkdir(parents=True, exist_ok=True)
    harness_contract_path = root / "fno_training_harness_contract.json"
    prompt_path = root / "claude_fno_training_iteration_prompt.md"
    output_path = root / "fno_training_iteration_contract.json"
    harness = build_fno_training_harness_contract(
        output_path=harness_contract_path,
        workspace_dir=root,
        source_path=source_path,
        baseline_prediction_path=baseline_prediction_path,
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        previous_evidence_paths=evidence_paths,
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        max_candidates=max_candidates,
        modes=mode_values,
        widths=width_values,
        depths=depth_values,
        loss_types=["early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        forecast_steps_values=forecast_values,
        train_steps_values=train_values,
        normalizations=normalization_values,
        force=True,
    )
    prompt = _build_fno_training_iteration_prompt(
        workspace_dir=Path(workspace_dir),
        previous_evidence_path=evidence_paths[-1],
        previous_evidence_paths=evidence_paths,
        harness_contract_path=harness_contract_path,
        harness=harness,
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    selected = evidence.get("selected_candidate") if isinstance(evidence, dict) else {}
    report = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_ITERATION_CONTRACT_BUILT",
        "previous_evidence_path": str(evidence_paths[-1]),
        "previous_evidence_paths": [str(path) for path in evidence_paths],
        "agent_contract": {
            "autonomy_policy": {
                "agent_selects_candidates": True,
                "supervisor_must_not_pin_parameters": True,
                "use_evidence_not_manual_tuning": True,
                "use_available_actions_only": True,
            },
            "required_agent_behaviors": [
                "read_previous_evidence",
                "build_or_read_harness_contract",
                "plan_candidates_from_evidence",
                "run_bounded_optimization",
                "record_evidence",
                "build_and_validate_hypothesis_design",
                "summarize_next_constraints",
            ],
            "graft_evaluation_required_when_available": bool(
                graft_evaluation_available
            ),
            "fast_feedback_guard_active": graft_evaluation_available,
            "previous_selected_candidate": selected,
            "harness_state_hint": harness["state_hint"],
        },
        "artifacts": {
            "iteration_contract_path": str(output_path),
            "prompt_path": str(prompt_path),
            "harness_contract_path": str(harness_contract_path),
            "optimization_evidence_path": harness["paths"]["optimization_evidence_path"],
            "algorithm_design_path": harness["paths"]["algorithm_design_path"],
            "algorithm_design_validation_path": harness["paths"][
                "algorithm_design_validation_path"
            ],
        },
        "next_command": "run-claude-agent-sdk-runtime",
    }
    output_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _fno_graft_harness_actions(
    *,
    paths: dict[str, str],
    graft_inputs: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if graft_inputs is None:
        return []
    baseline_prediction_path = graft_inputs["baseline_prediction_path"]
    task1_test_path = graft_inputs["test_path"]
    task1_target_path = graft_inputs["target_path"]
    return [
        {
            "action_id": "run_task1_fno_graft_from_optimization_report",
            "state_hint": "TASK1_FNO_GRAFT_EVALUATOR_RAN",
            "command": _join_command(
                [
                    "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent run-task1-fno-graft-from-optimization-report",
                    f"--optimization-report-path {paths['optimization_report_path']}",
                    f"--baseline-prediction-path {baseline_prediction_path}",
                    f"--test-path {task1_test_path}",
                    f"--target-path {task1_target_path}",
                    f"--output-dir {paths['fno_graft_dir']}",
                    "--window 10:20",
                    "--window 10:35",
                    "--window 10:57",
                    "--window 20:57",
                    "--window 35:57",
                    "--window 57:105",
                    "--window 10:105",
                    "--blend-weight 0.25",
                    "--blend-weight 0.5",
                    "--blend-weight 0.75",
                    "--blend-weight 1.0",
                    "--late-non-regression-tolerance 0.1",
                    "--force",
                ]
            ),
            "reads": [
                paths["optimization_report_path"],
                str(baseline_prediction_path),
                str(task1_test_path),
                str(task1_target_path),
            ],
            "writes": [paths["fno_graft_report_path"]],
            "next_action_options": [
                "record_fno_training_optimization_evidence",
                "build_hypothesis_algorithm_design",
                "plan_fno_training_candidates",
                "stop_and_report",
            ],
            "selection_rule": (
                "Promote training-source work only when best_task_prediction_score_delta "
                "is positive and late_non_regression_passed is true."
            ),
            "contract_repair": graft_inputs["repair"],
        }
    ]


def _resolve_fno_graft_inputs(
    *,
    baseline_prediction_path: str | Path | None,
    task1_test_path: str | Path | None,
    task1_target_path: str | Path | None,
    dataset_key: str,
) -> dict[str, Any] | None:
    if (
        baseline_prediction_path is None
        or task1_test_path is None
        or task1_target_path is None
    ):
        return None
    baseline = Path(baseline_prediction_path)
    test = Path(task1_test_path)
    target = Path(task1_target_path)
    repair: dict[str, Any] = {
        "applied": False,
        "reason": "inputs_declared",
        "requested_test_path": str(test),
    }
    baseline_count = _hdf5_dataset_sample_count(baseline, dataset_key=dataset_key)
    test_count = _hdf5_dataset_sample_count(test, dataset_key=dataset_key)
    target_count = _hdf5_dataset_sample_count(target, dataset_key=dataset_key)
    if (
        baseline_count is not None
        and test_count is not None
        and target_count is not None
        and baseline_count != test_count
        and baseline_count == target_count
    ):
        repair = {
            "applied": True,
            "reason": "baseline_sample_count_matches_target_not_test",
            "requested_test_path": str(test),
            "resolved_test_path": str(target),
            "baseline_sample_count": baseline_count,
            "test_sample_count": test_count,
            "target_sample_count": target_count,
        }
        test = target
    return {
        "baseline_prediction_path": str(baseline),
        "test_path": str(test),
        "target_path": str(target),
        "repair": repair,
    }


def _hdf5_dataset_sample_count(path: Path, *, dataset_key: str) -> int | None:
    if not path.is_file():
        return None
    try:
        with h5py.File(path, "r") as handle:
            if dataset_key not in handle:
                return None
            shape = handle[dataset_key].shape
            if not shape:
                return None
            return int(shape[0])
    except OSError:
        return None


def _repair_training_sample_budget(
    *,
    source_path: Path,
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    batch_size: int,
) -> dict[str, Any]:
    requested_train = int(train_sample_limit)
    requested_val = int(val_sample_limit)
    requested_batch = int(batch_size)
    source_sample_count = _hdf5_dataset_sample_count(source_path, dataset_key=dataset_key)
    resolved_train = requested_train
    resolved_val = requested_val
    resolved_batch = requested_batch
    applied = False
    reason = "source_sample_count_unavailable"
    if source_sample_count is not None:
        reason = "budget_within_source_sample_count"
        minimum_val = 1
        if requested_train + requested_val > source_sample_count:
            applied = True
            reason = "train_val_sample_limits_exceed_source_sample_count"
            resolved_val = min(requested_val, max(minimum_val, source_sample_count // 3))
            resolved_train = source_sample_count - resolved_val
            if resolved_train < 1:
                resolved_train = max(1, source_sample_count - 1)
                resolved_val = max(1, source_sample_count - resolved_train)
        if resolved_batch > resolved_train:
            applied = True
            reason = (
                "batch_size_exceeds_resolved_train_sample_limit"
                if reason == "budget_within_source_sample_count"
                else reason
            )
            resolved_batch = max(1, resolved_train)
    return {
        "applied": applied,
        "reason": reason,
        "source_sample_count": source_sample_count,
        "requested_train_sample_limit": requested_train,
        "requested_val_sample_limit": requested_val,
        "requested_batch_size": requested_batch,
        "resolved_train_sample_limit": resolved_train,
        "resolved_val_sample_limit": resolved_val,
        "resolved_batch_size": resolved_batch,
    }


def _write_hdf5_tensor(path: Path, array: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=array)


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        cleaned.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(cleaned).split("_") if part)


def _float_or_default_local(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _build_fno_training_iteration_prompt(
    *,
    workspace_dir: Path,
    previous_evidence_path: Path,
    previous_evidence_paths: list[Path],
    harness_contract_path: Path,
    harness: dict[str, Any],
) -> str:
    resolved_workspace = workspace_dir.expanduser().resolve()
    evidence_lines = [f"- previous evidence: {path}" for path in previous_evidence_paths]
    graft_action_available = any(
        action.get("action_id") == "run_task1_fno_graft_from_optimization_report"
        for action in harness.get("available_actions", [])
        if isinstance(action, dict)
    )
    graft_lines = []
    if graft_action_available:
        graft_lines = [
            f"- graft report path: {harness['paths']['fno_graft_report_path']}",
            "",
            "## Task-specific scoring objective",
            "The trained FNO is not being judged as a full replacement for the current strong baseline.",
            "Treat it as an early/middle dynamics source to graft onto the baseline prediction.",
            "After bounded optimization, run the graft evaluation action before deciding whether the training source helped.",
            "Promote the direction only when best_task_prediction_score_delta is positive and late_non_regression_passed is true.",
            "",
        ]
    return "\n".join(
        [
            "# PDEAgent autonomous FNO training iteration",
            "",
            "You are running one evidence-driven neural-operator training iteration.",
            "Your job is to use the previous evidence and the harness contract to choose the next bounded experiment.",
            "Do not patch source files. Do not create directories manually. Do not probe CLI help.",
            "Do not hand-pick hyperparameters outside the contract; select from contract actions and candidate space.",
            "Use the paths below exactly as relative paths from the current working directory; do not rewrite them to historical absolute repository paths.",
            "Previous evidence files can be large. Do not Read an entire evidence file first. Use Grep for selected_candidate, selection_components, graft_evidence, next_generation_constraints, and training_directive, then use bounded Read offsets around those matches.",
            "",
            f"- workspace: {workspace_dir}",
            f"- current working directory: {resolved_workspace}",
            *evidence_lines,
            f"- harness contract: {harness_contract_path}",
            f"- candidate plan path: {harness['paths']['candidate_plan_path']}",
            f"- optimization evidence path: {harness['paths']['optimization_evidence_path']}",
            *graft_lines,
            "",
            "## Required autonomous behavior",
            "1. Inspect previous evidence with Grep plus bounded Read, and identify the active bottleneck from selected_candidate, selection_components, graft_evidence, next_generation_constraints, and training_directive.",
            "2. Read the harness contract and choose available actions from it. The contract is not a blind fixed pipeline.",
            "3. Plan candidates from evidence using the provided candidate space. Preserve early/middle/late guardrails.",
            "4. Run one bounded optimization. If graft evaluation is available, run it and use its delta/non-regression result as the primary training-source evidence.",
            "5. Record FNO optimization evidence, then build and validate a hypothesis algorithm design.",
            "6. Stop if an action fails without a clear parameter repair, or after validation is written.",
            "",
            "## Reporting requirements",
            "- Report generated artifact paths.",
            "- Report selected candidate id, config, fno_selection_score, and selection_components when available.",
            "- Report graft best_task_prediction_score_delta and late_non_regression_passed when graft evaluation is available.",
            "- Report whether early, middle, and late metrics were evaluated.",
            "- Report next_generation_constraints for the next iteration.",
        ]
    )


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["context"].astype(np.float32), data["target"].astype(np.float32)


def _source_horizon_guard(
    *,
    source_path: Path,
    dataset_key: str,
    context_steps: int,
    time_reduction: int,
) -> dict[str, Any] | None:
    if not source_path.exists():
        return None
    try:
        with h5py.File(source_path, "r") as handle:
            if dataset_key not in handle:
                return None
            shape = tuple(int(axis) for axis in handle[dataset_key].shape)
    except OSError:
        return None
    if len(shape) < 2:
        return None
    reduced_time_steps = (shape[1] + int(time_reduction) - 1) // int(time_reduction)
    max_forecast_steps = max(0, reduced_time_steps - int(context_steps))
    return {
        "source_path": str(source_path),
        "dataset_key": dataset_key,
        "raw_time_steps": shape[1],
        "time_reduction": int(time_reduction),
        "reduced_time_steps": int(reduced_time_steps),
        "context_steps": int(context_steps),
        "max_forecast_steps": int(max_forecast_steps),
        "filtered_forecast_steps_above_max": True,
    }



def _guard_artifacts(paths: Any, *, force: bool) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not force:
        raise Task1ShortWindowTrainingError(
            "short-window training artifacts already exist: "
            + ", ".join(str(path) for path in existing)
        )


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Task1ShortWindowTrainingError(f"{label} must be a positive integer")
    return value


def _positive_float(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0.0:
        raise Task1ShortWindowTrainingError(f"{label} must be positive")
    return float(value)


def _positive_int_list(values: list[int], label: str) -> list[int]:
    if not values:
        raise Task1ShortWindowTrainingError(f"{label} must not be empty")
    rows = []
    for value in values:
        rows.append(_positive_int(int(value), label))
    return sorted(set(rows))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise Task1ShortWindowTrainingError(f"missing {label}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise Task1ShortWindowTrainingError(f"{label} must be a JSON object")
    return data


def _guard_output(output_path: Path | None, *, force: bool) -> None:
    if output_path is not None and output_path.exists() and not force:
        raise Task1ShortWindowTrainingError(f"output already exists: {output_path}")


def _write_optional_json(output_path: Path | None, payload: dict[str, Any]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _join_command(parts: list[str]) -> str:
    return " ".join(str(part) for part in parts if str(part))


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise Task1ShortWindowTrainingError(
            "torch is required for task1 short-window training"
        ) from exc
    return torch
