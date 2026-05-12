"""Task 1 short-window training helpers functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np
from pde_agent.core.task1_short_window_common import (
    FAMILY_SELECTION_STATE_HINT,
    STATE_HINT,
    SUPPORTED_FNO_LOSS_TYPES,
    SUPPORTED_MODEL_FAMILIES,
    SUPPORTED_NORMALIZATION_TYPES,
    Task1ShortWindowTrainingError,
)
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
