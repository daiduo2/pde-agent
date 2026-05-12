"""Task 1 short-window training contracts functions."""

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
    _build_fno_training_iteration_prompt,
    _guard_artifacts,
    _guard_output,
    _hdf5_dataset_sample_count,
    _join_command,
    _load_json,
    _positive_int_list,
    _repair_training_sample_budget,
    _source_horizon_guard,
    _write_optional_json,
)

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
