"""Task 1 short-window training planning functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
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
from pde_agent.core.task1_short_window_training_helpers import (
    _guard_artifacts,
    _guard_output,
    _load_json,
    _positive_int,
    _positive_int_list,
    _write_optional_json,
)
from pde_agent.core.task1_short_window_training_loops import (
    run_task1_short_window_training_loop,
)

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
