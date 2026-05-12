"""Task 1 short-window training loops functions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
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
from pde_agent.core.task1_short_window_models import (
    build_model as _build_model,
    denormalize_prediction as _denormalize_prediction,
    normalize_windows as _normalize_windows,
    parameter_count as _parameter_count,
    training_loss as _training_loss,
)
from pde_agent.core.task1_short_window_training_helpers import (
    _guard_artifacts,
    _load_windows,
    _positive_float,
    _positive_int,
    _require_torch,
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
