"""Command functions for fno domain."""

from __future__ import annotations

from pathlib import Path

from pde_agent.commands.hypothesis_lab_runtime_misc import _parse_json_objects
from pde_agent.core.json_output import success_payload
from pde_agent.core.task1_short_window_training import (
    FAMILY_SELECTION_STATE_HINT as TASK1_SHORT_WINDOW_FAMILY_SELECTION_STATE_HINT,
    STATE_HINT as TASK1_SHORT_WINDOW_TRAINING_STATE_HINT,
)
from pde_agent.core.task1_short_window_training import (
    build_fno_training_harness_contract,
    plan_fno_training_candidates,
    record_fno_training_optimization_evidence,
    run_fno_training_optimization_loop,
    run_task1_short_window_family_selection_loop,
    run_task1_short_window_training_loop,
)

def build_fno_training_harness_contract_command(
    *,
    output_path: str,
    workspace_dir: str,
    source_path: str,
    baseline_prediction_path: str | None = None,
    task1_test_path: str | None = None,
    task1_target_path: str | None = None,
    previous_evidence_paths: list[str],
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    time_reduction: int,
    space_reduction: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_candidates: int,
    modes: list[int],
    widths: list[int],
    depths: list[int],
    loss_types: list[str],
    normalizations: list[str] | None = None,
    forecast_steps_values: list[int],
    train_steps_values: list[int],
    force: bool,
) -> dict[str, object]:
    report = build_fno_training_harness_contract(
        output_path=Path(output_path),
        workspace_dir=Path(workspace_dir),
        source_path=Path(source_path),
        baseline_prediction_path=Path(baseline_prediction_path) if baseline_prediction_path else None,
        task1_test_path=Path(task1_test_path) if task1_test_path else None,
        task1_target_path=Path(task1_target_path) if task1_target_path else None,
        previous_evidence_paths=[Path(path) for path in previous_evidence_paths],
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
        modes=modes,
        widths=widths,
        depths=depths,
        loss_types=loss_types,
        normalizations=normalizations,
        forecast_steps_values=forecast_steps_values,
        train_steps_values=train_steps_values,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="build-fno-training-harness-contract",
        state_hint="FNO_TRAINING_HARNESS_CONTRACT_BUILT",
        artifacts={
            "harness_contract_path": output_path,
            "available_actions": report["available_actions"],
            "budget_contract": report["budget_contract"],
            "data_contract": report["data_contract"],
        },
        metrics={
            "action_count": len(report["available_actions"]),
            "max_candidates": report["budget_contract"]["max_candidates"],
            "train_sample_limit": report["budget_contract"]["train_sample_limit"],
            "val_sample_limit": report["budget_contract"]["val_sample_limit"],
        },
        next_hint="choose-action-from-harness-contract",
    )

def plan_fno_training_candidates_command(
    *,
    evidence_paths: list[str],
    output_path: str,
    max_candidates: int,
    modes: list[int],
    widths: list[int],
    depths: list[int],
    loss_types: list[str],
    normalizations: list[str] | None = None,
    forecast_steps_values: list[int],
    train_steps_values: list[int],
    search_guidance: list[str] | None = None,
    force: bool,
) -> dict[str, object]:
    parsed_search_guidance = _parse_json_objects(
        values=search_guidance or [],
        label="search_guidance",
    )
    report = plan_fno_training_candidates(
        evidence_paths=[Path(path) for path in evidence_paths],
        output_path=Path(output_path),
        max_candidates=max_candidates,
        modes=modes,
        widths=widths,
        depths=depths,
        loss_types=loss_types,
        normalizations=normalizations,
        forecast_steps_values=forecast_steps_values,
        train_steps_values=train_steps_values,
        search_guidance=parsed_search_guidance,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="plan-fno-training-candidates",
        state_hint="FNO_TRAINING_CANDIDATES_PLANNED",
        artifacts={"candidate_plan_path": output_path},
        metrics={
            "candidate_count": len(report["candidate_configs"]),
            "avoided_config_count": report["avoided_config_count"],
        },
        next_hint=report["next_command"],
    )

def run_fno_training_optimization_loop_command(
    *,
    candidate_plan_path: str,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    time_reduction: int,
    space_reduction: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    force: bool,
) -> dict[str, object]:
    report = run_fno_training_optimization_loop(
        candidate_plan_path=Path(candidate_plan_path),
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="run-fno-training-optimization-loop",
        state_hint="FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
        artifacts={
            "optimization_report_path": report["artifacts"][
                "optimization_report_path"
            ],
        },
        metrics={
            "candidate_count": len(report["candidate_records"]),
            "selected_candidate_id": report["selected_candidate"]["candidate_id"],
            "directive_action": report["training_directive"]["action"],
            "best_early_score": report["selected_candidate"]["early_score"],
        },
        next_hint=report["next_command"],
    )

def record_fno_training_optimization_evidence_command(
    *,
    optimization_report_path: str,
    output_path: str,
    graft_report_path: str | None = None,
    force: bool,
) -> dict[str, object]:
    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=Path(optimization_report_path),
        output_path=Path(output_path),
        graft_report_path=Path(graft_report_path) if graft_report_path else None,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="record-fno-training-optimization-evidence",
        state_hint="FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
        artifacts={"fno_training_optimization_evidence_path": output_path},
        metrics={
            "selected_candidate_id": evidence["selected_candidate"]["candidate_id"],
            "directive_action": evidence["training_directive"]["action"],
        },
        next_hint=evidence["next_command"],
    )

def run_task1_short_window_training_loop_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    model_family: str,
    normalization: str = "none",
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    force: bool,
) -> dict[str, object]:
    report = run_task1_short_window_training_loop(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        model_family=model_family,
        normalization=normalization,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        force=force,
    )
    loss_curve = report["training"]["loss_curve"]
    early = report["metrics"]["early"]
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="run-task1-short-window-training-loop",
        state_hint=TASK1_SHORT_WINDOW_TRAINING_STATE_HINT,
        artifacts={
            "report_path": report["artifacts"]["report_path"],
            "checkpoint_path": report["artifacts"]["checkpoint_path"],
            "config_path": report["artifacts"]["config_path"],
            "training_log_path": report["artifacts"]["training_log_path"],
            "prediction_path": report["artifacts"]["prediction_path"],
            "evidence_path": report["artifacts"]["evidence_path"],
        },
        metrics={
            "model_family": report["model"]["model_family"],
            "parameter_count": report["model"]["parameter_count"],
            "train_steps": report["training"]["train_steps"],
            "final_train_loss": loss_curve["train"][-1],
            "final_val_loss": loss_curve["val"][-1],
            "early_rel_mse": early["rel_mse"],
            "early_score": early["score"],
            "decision": report["evidence"]["decision"],
        },
        next_hint=report["next_command"],
    )

def run_task1_short_window_family_selection_loop_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    model_families: list[str],
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    force: bool,
) -> dict[str, object]:
    report = run_task1_short_window_family_selection_loop(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        model_families=model_families,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="run-task1-short-window-family-selection-loop",
        state_hint=TASK1_SHORT_WINDOW_FAMILY_SELECTION_STATE_HINT,
        artifacts={
            "comparison_report_path": report["artifacts"]["comparison_report_path"],
            "comparison_evidence_path": report["artifacts"][
                "comparison_evidence_path"
            ],
        },
        metrics={
            "selected_family": report["selected_family"],
            "family_count": len(report["family_records"]),
            "directive_action": report["training_directive"]["action"],
            "best_early_score": max(
                float(item["early_score"]) for item in report["family_records"]
            ),
        },
        next_hint=report["next_command"],
    )

def build_fno_training_iteration_contract_command(
    *,
    previous_evidence_path: str,
    previous_evidence_paths: list[str] | None = None,
    output_dir: str,
    workspace_dir: str,
    source_path: str,
    baseline_prediction_path: str | None = None,
    task1_test_path: str | None = None,
    task1_target_path: str | None = None,
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    time_reduction: int,
    space_reduction: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_candidates: int,
    force: bool,
) -> dict[str, object]:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_iteration_contract_from_evidence,
    )

    report = build_fno_training_iteration_contract_from_evidence(
        previous_evidence_path=Path(previous_evidence_path),
        previous_evidence_paths=[Path(path) for path in previous_evidence_paths or []],
        output_dir=Path(output_dir),
        workspace_dir=Path(workspace_dir),
        source_path=Path(source_path),
        baseline_prediction_path=Path(baseline_prediction_path) if baseline_prediction_path else None,
        task1_test_path=Path(task1_test_path) if task1_test_path else None,
        task1_target_path=Path(task1_target_path) if task1_target_path else None,
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
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="build-fno-training-iteration-contract",
        state_hint="FNO_TRAINING_ITERATION_CONTRACT_BUILT",
        artifacts={
            "iteration_contract_path": report["artifacts"]["iteration_contract_path"],
            "prompt_path": report["artifacts"]["prompt_path"],
            "harness_contract_path": report["artifacts"]["harness_contract_path"],
        },
        metrics={
            "candidate_count": len(report.get("agent_contract", {}).get("required_agent_behaviors", [])),
            "max_candidates": report["agent_contract"].get("previous_selected_candidate", {}).get("max_candidates", max_candidates)
            if isinstance(report.get("agent_contract", {}).get("previous_selected_candidate"), dict)
            else max_candidates,
        },
        next_hint=report["next_command"],
    )
