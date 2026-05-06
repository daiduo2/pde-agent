"""Task 1 short-window trainable dynamics loop for Hermes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.local_windows import build_local_windows


STATE_HINT = "TASK1_SHORT_WINDOW_TRAINING_LOOP_RAN"
FAMILY_SELECTION_STATE_HINT = "TASK1_SHORT_WINDOW_FAMILY_SELECTION_RAN"
SUPPORTED_MODEL_FAMILIES = {"conv1d_residual", "fno_lite"}
SUPPORTED_FNO_LOSS_TYPES = {"mse", "early_weighted_mse", "spectral_mse"}


class Task1ShortWindowTrainingError(RuntimeError):
    pass


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

    torch.manual_seed(int(seed))
    model = _build_model(
        torch=torch,
        model_family=model_family,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        space_points=int(train_context.shape[2]),
        hidden_channels=fno_width if model_family == "fno_lite" else 32,
        modes=fno_modes,
        depth=fno_depth,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_context_t = torch.as_tensor(train_context, dtype=torch.float32)
    train_target_t = torch.as_tensor(train_target, dtype=torch.float32)
    val_context_t = torch.as_tensor(val_context, dtype=torch.float32)
    val_target_t = torch.as_tensor(val_target, dtype=torch.float32)
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
        prediction = model(val_context_t).detach().cpu().numpy().astype(np.float32)

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
    evidences = [_load_json(Path(path), "FNO training evidence") for path in evidence_paths]
    attempted = _attempted_fno_config_signatures(evidences)
    preferences = _fno_planning_preferences(evidences)
    rows = []
    index = 1
    for loss_type in _ordered_values(loss_values, preferences["loss_types"]):
        for forecast_steps in _ordered_values(forecast_values, preferences["forecast_steps"]):
            for modes_value in _ordered_values(mode_values, preferences["modes"]):
                for width in _ordered_values(width_values, preferences["widths"]):
                    for depth in _ordered_values(depth_values, preferences["depths"]):
                        for train_steps in _ordered_values(train_values, preferences["train_steps"]):
                            config = {
                                "candidate_id": f"fno_train_candidate_{index:03d}",
                                "model_family": "fno_lite",
                                "modes": modes_value,
                                "width": width,
                                "depth": depth,
                                "loss_type": loss_type,
                                "forecast_steps": forecast_steps,
                                "train_steps": train_steps,
                                "rationale": _candidate_rationale(
                                    loss_type=loss_type,
                                    forecast_steps=forecast_steps,
                                    preferences=preferences,
                                ),
                            }
                            if _fno_config_signature(config) in attempted:
                                continue
                            rows.append(config)
                            index += 1
                            if len(rows) >= max_candidates:
                                break
                        if len(rows) >= max_candidates:
                            break
                    if len(rows) >= max_candidates:
                        break
                if len(rows) >= max_candidates:
                    break
            if len(rows) >= max_candidates:
                break
        if len(rows) >= max_candidates:
            break
    report = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
        "evidence_paths": [str(path) for path in evidence_paths],
        "candidate_configs": rows,
        "planning_preferences": preferences,
        "avoided_config_count": len(attempted),
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
            model_family="fno_lite",
            fno_modes=int(config["modes"]),
            fno_width=int(config["width"]),
            fno_depth=int(config["depth"]),
            loss_type=str(config["loss_type"]),
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
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    report = _load_json(Path(optimization_report_path), "FNO optimization report")
    if report.get("state_hint") != "FNO_TRAINING_OPTIMIZATION_LOOP_RAN":
        raise Task1ShortWindowTrainingError("optimization report has invalid state_hint")
    selected = report["selected_candidate"]
    directive = report["training_directive"]
    evidence = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
        "source_type": "fno_training_optimization",
        "optimization_report_path": str(optimization_report_path),
        "selected_candidate": selected,
        "training_directive": directive,
        "candidate_records": report["candidate_records"],
        "next_generation_constraints": _fno_optimization_constraints(
            selected=selected,
            directive=directive,
            records=report["candidate_records"],
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
    train_values = _positive_int_list(train_steps_values or [1, 2], "train_steps_values")
    loss_values = _loss_type_list(
        loss_types or ["mse", "early_weighted_mse", "spectral_mse"]
    )
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
    }
    data = {
        "source_path": str(source),
        "dataset_key": dataset_key,
    }
    paths = {
        "candidate_plan_path": str(root / "fno_training_plan.json"),
        "optimization_dir": str(root / "optimization"),
        "optimization_report_path": str(
            root / "optimization" / "fno_training_optimization_report.json"
        ),
        "optimization_evidence_path": str(root / "fno_training_optimization_evidence.json"),
        "algorithm_design_path": str(root / "hypothesis_algorithm_design.json"),
        "algorithm_design_validation_path": str(root / "hypothesis_validation_result.json"),
    }
    actions = [
        {
            "action_id": "plan_fno_training_candidates",
            "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
            "command": _join_command(
                [
                    "uv run --frozen pde-agent plan-fno-training-candidates",
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
                    "uv run --frozen pde-agent run-fno-training-optimization-loop",
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
                "record_fno_training_optimization_evidence",
                "stop_and_report",
            ],
        },
        {
            "action_id": "record_fno_training_optimization_evidence",
            "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
            "command": _join_command(
                [
                    "uv run --frozen pde-agent record-fno-training-optimization-evidence",
                    f"--optimization-report-path {paths['optimization_report_path']}",
                    f"--output-path {paths['optimization_evidence_path']}",
                    "--force",
                ]
            ),
            "reads": [paths["optimization_report_path"]],
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
                    "uv run --frozen pde-agent build-hypothesis-algorithm-design",
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
                    "uv run --frozen pde-agent validate-hypothesis-algorithm-design",
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
            "state_hint": "HERMES_REPORTED_DECISION",
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
            "forecast_steps_values": forecast_values,
            "train_steps_values": train_values,
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
                "validation_accepted",
                "budget_exhausted",
            ],
        },
        "next_command": "Hermes should choose an available action, not blindly run all actions.",
    }
    _write_optional_json(output_file, report)
    return report


class _Conv1DResidualRollout:
    def __init__(
        self,
        *,
        torch: Any,
        context_steps: int,
        forecast_steps: int,
        hidden_channels: int,
    ) -> None:
        self._torch = torch
        self.operator_type = "conv1d_residual"
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(context_steps, hidden_channels, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_channels, forecast_steps, kernel_size=1),
        )

    def parameters(self) -> Any:
        return self.network.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.network.state_dict()

    def train(self) -> None:
        self.network.train()

    def eval(self) -> None:
        self.network.eval()

    def __call__(self, context: Any) -> Any:
        raw = self.network(context)
        anchor = context[:, -1:, :]
        return anchor + raw


class _FNOLiteRollout:
    def __init__(
        self,
        *,
        torch: Any,
        context_steps: int,
        forecast_steps: int,
        space_points: int,
        hidden_channels: int,
        modes: int = 8,
        depth: int = 1,
    ) -> None:
        self._torch = torch
        self.operator_type = "spectral_conv1d"
        mode_count = max(1, min(int(modes), (int(space_points) // 2) + 1))
        self.lift = torch.nn.Conv1d(context_steps, hidden_channels, kernel_size=1)
        self.spectral_layers = [
            _SpectralConv1D(torch=torch, channels=hidden_channels, modes=mode_count)
            for _ in range(depth)
        ]
        self.local_layers = [
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in range(depth)
        ]
        self.project = torch.nn.Conv1d(hidden_channels, forecast_steps, kernel_size=1)
        self.activation = torch.nn.GELU()

    def parameters(self) -> Any:
        return (
            list(self.lift.parameters())
            + list(self.project.parameters())
            + [
                parameter
                for layer in self.local_layers
                for parameter in layer.parameters()
            ]
            + [layer.weight for layer in self.spectral_layers]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "lift": self.lift.state_dict(),
            "spectral_layers": [
                layer.state_dict() for layer in self.spectral_layers
            ],
            "local_layers": [layer.state_dict() for layer in self.local_layers],
            "project": self.project.state_dict(),
        }

    def train(self) -> None:
        self.lift.train()
        for layer in self.local_layers:
            layer.train()
        self.project.train()

    def eval(self) -> None:
        self.lift.eval()
        for layer in self.local_layers:
            layer.eval()
        self.project.eval()

    def __call__(self, context: Any) -> Any:
        hidden = self.activation(self.lift(context))
        for spectral_layer, local_layer in zip(self.spectral_layers, self.local_layers):
            hidden = self.activation(spectral_layer(hidden) + local_layer(hidden))
        raw = self.project(hidden)
        anchor = context[:, -1:, :]
        return anchor + raw


class _SpectralConv1D:
    def __init__(self, *, torch: Any, channels: int, modes: int) -> None:
        self._torch = torch
        self.channels = int(channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, channels * channels)
        self.weight = torch.nn.Parameter(
            scale
            * torch.randn(
                channels,
                channels,
                self.modes,
                dtype=torch.cfloat,
            )
        )

    def parameters(self) -> Any:
        return [self.weight]

    def state_dict(self) -> dict[str, Any]:
        return {"weight": self.weight}

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def __call__(self, value: Any) -> Any:
        torch = self._torch
        spectrum = torch.fft.rfft(value, dim=-1)
        output_spectrum = torch.zeros_like(spectrum)
        modes = min(self.modes, spectrum.shape[-1])
        output_spectrum[:, :, :modes] = torch.einsum(
            "bim,iom->bom",
            spectrum[:, :, :modes],
            self.weight[:, :, :modes],
        )
        return torch.fft.irfft(output_spectrum, n=value.shape[-1], dim=-1)


def _build_model(
    *,
    torch: Any,
    model_family: str,
    context_steps: int,
    forecast_steps: int,
    space_points: int,
        hidden_channels: int,
        modes: int,
        depth: int,
) -> Any:
    if model_family == "conv1d_residual":
        return _Conv1DResidualRollout(
            torch=torch,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            hidden_channels=hidden_channels,
        )
    if model_family == "fno_lite":
        return _FNOLiteRollout(
            torch=torch,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            space_points=space_points,
            hidden_channels=hidden_channels,
            modes=modes,
            depth=depth,
        )
    raise Task1ShortWindowTrainingError(f"unsupported model_family: {model_family}")


def _family_record(report: dict[str, Any]) -> dict[str, Any]:
    loss_curve = report["training"]["loss_curve"]
    middle = report["metrics"]["middle_guardrail"]
    return {
        "model_family": report["model"]["model_family"],
        "operator_type": report["model"]["operator_type"],
        "parameter_count": report["model"]["parameter_count"],
        "runtime_seconds": report["training"]["runtime_seconds"],
        "train_steps": report["training"]["train_steps"],
        "loss_curve": loss_curve,
        "final_train_loss": float(loss_curve["train"][-1]),
        "final_val_loss": float(loss_curve["val"][-1]),
        "early_rel_mse": float(report["metrics"]["early"]["rel_mse"]),
        "early_score": float(report["metrics"]["early"]["score"]),
        "middle_guardrail": middle,
        "decision": report["evidence"]["decision"],
        "reason_codes": report["evidence"]["reason_codes"],
        "artifacts": report["artifacts"],
    }


def _select_family(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        records,
        key=lambda item: (
            -float(item.get("early_score", 0.0)),
            float(item.get("final_val_loss", float("inf"))),
            float(item.get("runtime_seconds", float("inf"))),
        ),
    )
    selected = dict(ranked[0])
    selected["selection_reason_codes"] = _selection_reason_codes(
        selected=selected,
        records=records,
    )
    return selected


def _selection_reason_codes(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> list[str]:
    reasons = ["best_early_score"]
    early_scores = [float(item.get("early_score", 0.0)) for item in records]
    if early_scores and max(early_scores) - min(early_scores) < 1e-9:
        reasons.append("tie_broken_by_val_loss_or_runtime")
    if selected.get("middle_guardrail", {}).get("evaluated"):
        reasons.append("middle_guardrail_available")
    if selected.get("decision") == "continue_training":
        reasons.append("selected_family_wants_more_training")
    return reasons


def _family_training_directive(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    action = _family_directive_action(selected=selected, records=records)
    return {
        "action": action,
        "target_family": selected["model_family"],
        "reason_codes": selected["selection_reason_codes"],
        "next_command": "run-task1-short-window-training-loop"
        if action in {"continue_training", "change_loss_or_horizon"}
        else "build-hypothesis-algorithm-design",
    }


def _family_directive_action(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> str:
    if all(item.get("decision") == "abandon_model_family" for item in records):
        return "abandon_candidate_family"
    if selected.get("decision") == "change_loss_or_horizon":
        return "change_loss_or_horizon"
    if selected.get("decision") == "continue_training":
        return "continue_training"
    return "expand_model_family"


def _family_selection_constraints(
    *, directive: dict[str, Any]
) -> dict[str, list[str]]:
    target = str(directive["target_family"])
    constraints = {
        "training_model_families": [],
        "prefer_training_objectives": [],
        "refine_training_horizons": [],
        "avoid_training_model_families": [],
    }
    if directive["action"] == "abandon_candidate_family":
        constraints["avoid_training_model_families"].append(target)
    else:
        constraints["training_model_families"].append(target)
        constraints["prefer_training_objectives"].append("short_window_mse")
    if directive["action"] == "change_loss_or_horizon":
        constraints["prefer_training_objectives"].extend(
            ["early_weighted_mse", "rollout_consistency"]
        )
        constraints["refine_training_horizons"].append("early_window")
    if "middle_guardrail_available" in directive.get("reason_codes", []):
        constraints["refine_training_horizons"].append("middle_guardrail")
    return {key: _dedupe(value) for key, value in constraints.items()}


def _training_loss(*, torch: Any, prediction: Any, target: Any, loss_type: str) -> Any:
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(prediction, target)
    if loss_type == "early_weighted_mse":
        weights = torch.linspace(
            1.5,
            0.75,
            steps=prediction.shape[1],
            dtype=prediction.dtype,
            device=prediction.device,
        ).view(1, -1, 1)
        return torch.mean(weights * (prediction - target) ** 2)
    if loss_type == "spectral_mse":
        base = torch.nn.functional.mse_loss(prediction, target)
        pred_spec = torch.fft.rfft(prediction, dim=-1)
        target_spec = torch.fft.rfft(target, dim=-1)
        spectral = torch.mean(torch.abs(pred_spec - target_spec) ** 2)
        return base + 0.01 * spectral
    raise Task1ShortWindowTrainingError(f"unsupported loss_type: {loss_type}")


def _fno_candidate_record(
    *, config: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    base = _family_record(report)
    base["candidate_id"] = str(config["candidate_id"])
    base["config"] = {
        "modes": int(config["modes"]),
        "width": int(config["width"]),
        "depth": int(config["depth"]),
        "loss_type": str(config["loss_type"]),
        "forecast_steps": int(config["forecast_steps"]),
        "train_steps": int(config["train_steps"]),
    }
    return base


def _select_fno_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        records,
        key=lambda item: (
            -float(item["early_score"]),
            float(item["final_val_loss"]),
            float(item["runtime_seconds"]),
        ),
    )
    selected = dict(ranked[0])
    selected["selection_reason_codes"] = _fno_selection_reason_codes(
        selected=selected,
        records=records,
    )
    return selected


def _fno_selection_reason_codes(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> list[str]:
    reasons = ["best_early_score"]
    if selected.get("middle_guardrail", {}).get("evaluated"):
        reasons.append("middle_guardrail_available")
    if selected.get("config", {}).get("loss_type") != "mse":
        reasons.append("non_default_loss_selected")
    if len({item["config"]["loss_type"] for item in records}) > 1:
        reasons.append("loss_family_compared")
    return reasons


def _fno_training_directive(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    action = _fno_directive_action(selected=selected, records=records)
    return {
        "action": action,
        "target_family": "fno_lite"
        if action != "fallback_to_conv1d"
        else "conv1d_residual",
        "selected_candidate_id": selected["candidate_id"],
        "reason_codes": selected["selection_reason_codes"]
        + _fno_directive_reason_codes(action=action, selected=selected),
        "next_command": "plan-fno-training-candidates"
        if action
        in {
            "continue_current_fno_family",
            "change_loss",
            "change_horizon",
            "change_architecture",
        }
        else "run-task1-short-window-family-selection-loop",
    }


def _fno_directive_action(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> str:
    if float(selected["early_rel_mse"]) >= 4.5:
        return "fallback_to_conv1d"
    if selected["config"]["loss_type"] == "mse" and float(selected["early_rel_mse"]) >= 1.0:
        return "change_loss"
    if float(selected["early_rel_mse"]) >= 1.0:
        return "change_horizon"
    if _architecture_search_exhausted(records):
        return "expand_new_model_family"
    if selected.get("decision") == "continue_training":
        return "continue_current_fno_family"
    return "change_architecture"


def _fno_directive_reason_codes(
    *, action: str, selected: dict[str, Any]
) -> list[str]:
    reasons = [action]
    if float(selected["early_rel_mse"]) >= 1.0:
        reasons.append("early_rel_mse_high")
    else:
        reasons.append("early_signal_learned")
    return reasons


def _architecture_search_exhausted(records: list[dict[str, Any]]) -> bool:
    return len({(item["config"]["modes"], item["config"]["width"], item["config"]["depth"]) for item in records}) >= 6


def _fno_optimization_constraints(
    *,
    selected: dict[str, Any],
    directive: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    config = selected["config"]
    constraints = {
        "training_model_families": ["fno_lite"]
        if directive["action"] != "fallback_to_conv1d"
        else ["conv1d_residual"],
        "prefer_training_objectives": [],
        "refine_training_horizons": [],
        "avoid_training_model_families": [],
        "fno_training_configs": [item["config"] for item in records],
    }
    if directive["action"] == "change_loss":
        constraints["prefer_training_objectives"].extend(
            ["early_weighted_mse", "spectral_mse"]
        )
    else:
        constraints["prefer_training_objectives"].append(str(config["loss_type"]))
    if directive["action"] == "change_horizon":
        constraints["refine_training_horizons"].append("shorter_early_window")
    if directive["action"] == "fallback_to_conv1d":
        constraints["avoid_training_model_families"].append("fno_lite")
    if selected.get("middle_guardrail", {}).get("evaluated"):
        constraints["refine_training_horizons"].append("middle_guardrail")
    constraints["prefer_training_objectives"] = _dedupe(
        constraints["prefer_training_objectives"]
    )
    constraints["refine_training_horizons"] = _dedupe(
        constraints["refine_training_horizons"]
    )
    constraints["avoid_training_model_families"] = _dedupe(
        constraints["avoid_training_model_families"]
    )
    return constraints


def _attempted_fno_config_signatures(evidences: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    signatures = set()
    for evidence in evidences:
        for record in evidence.get("family_records", []) or []:
            if isinstance(record, dict) and isinstance(record.get("config"), dict):
                signatures.add(_fno_config_signature(record["config"]))
        for record in evidence.get("candidate_records", []) or []:
            if isinstance(record, dict) and isinstance(record.get("config"), dict):
                signatures.add(_fno_config_signature(record["config"]))
        constraints = evidence.get("next_generation_constraints")
        if isinstance(constraints, dict):
            for config in constraints.get("fno_training_configs", []) or []:
                if isinstance(config, dict):
                    signatures.add(_fno_config_signature(config))
    return signatures


def _fno_config_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(config.get("modes", 0)),
        int(config.get("width", 0)),
        int(config.get("depth", 0)),
        str(config.get("loss_type", "mse")),
        int(config.get("forecast_steps", 0)),
        int(config.get("train_steps", 0)),
    )


def _fno_planning_preferences(evidences: list[dict[str, Any]]) -> dict[str, Any]:
    preferences = {
        "loss_types": [],
        "forecast_steps": [],
        "modes": [],
        "widths": [],
        "depths": [],
        "train_steps": [],
    }
    for evidence in evidences:
        constraints = evidence.get("next_generation_constraints")
        if not isinstance(constraints, dict):
            continue
        objectives = _string_list(constraints.get("prefer_training_objectives"))
        for objective in objectives:
            if objective in SUPPORTED_FNO_LOSS_TYPES:
                preferences["loss_types"].append(objective)
            if objective == "short_window_mse":
                preferences["loss_types"].append("mse")
        horizons = _string_list(constraints.get("refine_training_horizons"))
        if any("shorter" in horizon or "early" in horizon for horizon in horizons):
            preferences["forecast_steps"].extend([12, 20])
        selected = evidence.get("selected_candidate")
        if isinstance(selected, dict) and isinstance(selected.get("config"), dict):
            config = selected["config"]
            preferences["modes"].append(int(config.get("modes", 0)) + 4)
            preferences["widths"].append(int(config.get("width", 0)))
            preferences["depths"].append(int(config.get("depth", 0)))
            preferences["train_steps"].append(int(config.get("train_steps", 0)) + 1)
    return {
        key: _dedupe([str(item) for item in value]) for key, value in preferences.items()
    }


def _candidate_rationale(
    *, loss_type: str, forecast_steps: int, preferences: dict[str, Any]
) -> list[str]:
    reasons = []
    if str(loss_type) in preferences.get("loss_types", []):
        reasons.append("preferred_by_previous_evidence")
    if str(forecast_steps) in preferences.get("forecast_steps", []):
        reasons.append("matches_refined_horizon_constraint")
    if not reasons:
        reasons.append("coverage_candidate")
    return reasons


def _ordered_values(values: list[Any], preferred_values: list[Any]) -> list[Any]:
    preferred = [str(item) for item in preferred_values]
    return sorted(
        values,
        key=lambda item: (
            0 if str(item) in preferred else 1,
            preferred.index(str(item)) if str(item) in preferred else 999,
            item,
        ),
    )


def _short_window_metrics(
    *, prediction: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    early_stop = min(47, prediction.shape[1])
    middle_start = min(47, prediction.shape[1])
    middle_stop = min(95, prediction.shape[1])
    early_rel = _rel_mse(prediction[:, :early_stop], target[:, :early_stop])
    metrics = {
        "early": {
            "evaluated": early_stop > 0,
            "future_start": 0,
            "future_stop": int(early_stop),
            "rel_mse": early_rel,
            "score": 100.0 * float(np.exp(-20.0 * early_rel)),
        },
        "middle_guardrail": {"evaluated": middle_stop > middle_start},
    }
    if middle_stop > middle_start:
        middle_rel = _rel_mse(
            prediction[:, middle_start:middle_stop],
            target[:, middle_start:middle_stop],
        )
        metrics["middle_guardrail"].update(
            {
                "future_start": int(middle_start),
                "future_stop": int(middle_stop),
                "rel_mse": middle_rel,
                "score": 100.0 * float(np.exp(-10.0 * middle_rel)),
            }
        )
    return metrics


def _training_decision(
    *, loss_curve: dict[str, list[float]], metrics: dict[str, Any]
) -> str:
    initial = float(loss_curve["val"][0])
    final = float(loss_curve["val"][-1])
    early_rel = float(metrics["early"]["rel_mse"])
    if final < initial or early_rel < 1.0:
        return "continue_training"
    if early_rel < 5.0:
        return "change_loss_or_horizon"
    return "abandon_model_family"


def _training_constraints(
    *, model_family: str, decision: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    constraints = {
        "training_model_families": [model_family],
        "prefer_training_objectives": [],
        "refine_training_horizons": [],
        "avoid_training_model_families": [],
    }
    if decision == "continue_training":
        constraints["prefer_training_objectives"].append("short_window_mse")
        constraints["refine_training_horizons"].append("early_window")
    elif decision == "change_loss_or_horizon":
        constraints["prefer_training_objectives"].extend(
            ["early_weighted_mse", "rollout_consistency"]
        )
        constraints["refine_training_horizons"].append("shorter_early_window")
    else:
        constraints["avoid_training_model_families"].append(model_family)
    if metrics["middle_guardrail"]["evaluated"]:
        constraints["refine_training_horizons"].append("middle_guardrail")
    return constraints


def _decision_reasons(*, decision: str, metrics: dict[str, Any]) -> list[str]:
    reasons = [decision]
    if float(metrics["early"]["rel_mse"]) >= 1.0:
        reasons.append("early_rel_mse_high")
    else:
        reasons.append("early_signal_learned")
    if metrics["middle_guardrail"]["evaluated"]:
        reasons.append("middle_guardrail_available")
    return reasons


def _rel_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sum((prediction - target) ** 2, axis=(1, 2))
    denominator = np.sum(target**2, axis=(1, 2)) + 1e-12
    return float(np.mean(np.minimum(numerator / denominator, 5.0)))


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["context"].astype(np.float32), data["target"].astype(np.float32)


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


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


def _loss_type_list(values: list[str]) -> list[str]:
    if not values:
        raise Task1ShortWindowTrainingError("loss_types must not be empty")
    rows = []
    for value in values:
        text = str(value)
        if text not in SUPPORTED_FNO_LOSS_TYPES:
            raise Task1ShortWindowTrainingError(f"unsupported loss_type: {text}")
        rows.append(text)
    return _dedupe(rows)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


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


def _dedupe(values: list[str]) -> list[str]:
    rows = []
    seen = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows
