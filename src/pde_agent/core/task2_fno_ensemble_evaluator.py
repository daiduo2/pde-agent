"""Ensemble multiple front-window models for Task 2 graft evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.manual_operator_tuning import score_burgers_segments
from pde_agent.core.task2_fno_data import load_task2_data, normalize_per_sample
from pde_agent.core.task2_fno_graft_evaluator import (
    _build_model_from_config,
    _load_checkpoint,
    _require_torch,
)

STATE_HINT = "TASK2_FNO_ENSEMBLE_EVALUATOR_RAN"


def _run_inference(
    model: Any,
    context: np.ndarray,
    nu: np.ndarray | None,
    device: Any,
) -> np.ndarray:
    torch = _require_torch()
    ctx_n, _, stats = normalize_per_sample(context, context)
    ctx_t = torch.as_tensor(ctx_n, dtype=torch.float32).to(device)

    model.eval_mode()
    with torch.no_grad():
        if hasattr(model, "use_film") and model.use_film:
            nu_t = torch.as_tensor(nu, dtype=torch.float32).to(device) if nu is not None else None
            pred_t = model(ctx_t, cond=nu_t)
        else:
            pred_t = model(ctx_t)

    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).to(device)
    std = torch.as_tensor(stats["std"], dtype=torch.float32).to(device)
    pred = (pred_t * std + mean).cpu().numpy()
    return pred


def _score_prediction(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    full = np.concatenate([target[:, :10, :], prediction], axis=1).astype(np.float32)
    return score_burgers_segments(prediction=full, target=target)


def run_task2_fno_ensemble_evaluator(
    *,
    base_checkpoint_path: str | Path,
    front_checkpoints: list[dict[str, Any]],
    output_dir: str | Path,
    val_data_path: str | Path | None = None,
    windows: list[tuple[int, int]] | None = None,
    blend_weights: list[float] | None = None,
    residual_scales: list[float] | None = None,
    late_non_regression_tolerance: float = 1.0,
    device: str = "mps",
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate ensemble front models grafted onto base backend.

    front_checkpoints: list of {"path": str, "weight": float}
    """
    root = Path(output_dir)
    report_path = root / "task2_fno_ensemble_evaluator_report.json"
    if not force and report_path.exists():
        raise RuntimeError(f"output exists and force=False: {root}")
    root.mkdir(parents=True, exist_ok=True)

    torch = _require_torch()
    device_obj = torch.device(device)

    # Load base model
    base_ckpt = _load_checkpoint(base_checkpoint_path)
    base_model = _build_model_from_config(base_ckpt["config"], device_obj)
    base_model.load_state_dict(base_ckpt["model_state_dict"])

    # Load validation data
    _, _, val_context, val_target, val_nu = load_task2_data(val_path=val_data_path)

    # Run base inference
    base_pred = _run_inference(base_model, val_context, val_nu, device_obj)
    base_score = _score_prediction(base_pred, val_target)

    # Load and run all front models
    front_preds = []
    front_scores = []
    front_configs = []
    for fc in front_checkpoints:
        ckpt = _load_checkpoint(fc["path"])
        model = _build_model_from_config(ckpt["config"], device_obj)
        model.load_state_dict(ckpt["model_state_dict"])
        pred = _run_inference(model, val_context, val_nu, device_obj)
        front_preds.append(pred)
        front_configs.append({"path": str(fc["path"]), "weight": fc.get("weight", 1.0)})

        # Score standalone (padded)
        if pred.shape[1] < base_pred.shape[1]:
            padded = np.array(base_pred, copy=True, dtype=np.float32)
            padded[:, :pred.shape[1], :] = pred
            front_scores.append(_score_prediction(padded, val_target))
        else:
            front_scores.append(_score_prediction(pred, val_target))

    # Compute weighted ensemble prediction
    weights = np.array([fc.get("weight", 1.0) for fc in front_checkpoints], dtype=np.float32)
    weights = weights / weights.sum()

    # Ensemble must agree on shape; use minimum steps across all fronts
    min_steps = min(p.shape[1] for p in front_preds)
    ensemble_pred = np.zeros_like(base_pred, dtype=np.float32)
    for i, pred in enumerate(front_preds):
        ensemble_pred[:, :min_steps, :] += weights[i] * pred[:, :min_steps, :]

    # Score ensemble standalone
    if min_steps < base_pred.shape[1]:
        ensemble_padded = np.array(base_pred, copy=True, dtype=np.float32)
        ensemble_padded[:, :min_steps, :] = ensemble_pred[:, :min_steps, :]
        ensemble_score = _score_prediction(ensemble_padded, val_target)
    else:
        ensemble_score = _score_prediction(ensemble_pred, val_target)

    np.savez_compressed(root / "base_prediction.npz", prediction=base_pred)
    np.savez_compressed(root / "ensemble_prediction.npz", prediction=ensemble_pred)

    if windows is None:
        windows = [(0, 47), (0, 95), (47, 95)]
    if blend_weights is None:
        blend_weights = [0.25, 0.5, 0.75]
    if residual_scales is None:
        residual_scales = [0.25, 0.5, 0.75]

    rows = []

    # Direct replacement with ensemble front
    for start, stop in windows:
        stop = min(stop, min_steps)
        if start >= stop:
            continue
        candidate = np.array(base_pred, copy=True, dtype=np.float32)
        candidate[:, start:stop, :] = ensemble_pred[:, start:stop, :]
        score = _score_prediction(candidate, val_target)
        late_delta = float(score["segments"]["late"]["score"]) - float(base_score["segments"]["late"]["score"])
        total_delta = float(score["segment_score"]) - float(base_score["segment_score"])
        rows.append({
            "candidate_id": f"ensemble_direct_w{start}_{stop}",
            "strategy": "direct",
            "window_start": start,
            "window_stop": stop,
            "segment_score": score["segment_score"],
            "task2_proxy": score["segment_score"] * 1.5,
            "segment_score_delta": total_delta,
            "late_score_delta": late_delta,
            "late_non_regression_passed": late_delta >= -float(late_non_regression_tolerance),
            "segments": score["segments"],
        })

    # Blend strategy with ensemble front
    for start, stop in windows:
        stop = min(stop, min_steps)
        if start >= stop:
            continue
        for weight in blend_weights:
            candidate = np.array(base_pred, copy=True, dtype=np.float32)
            candidate[:, start:stop, :] = (
                (1.0 - weight) * base_pred[:, start:stop, :]
                + weight * ensemble_pred[:, start:stop, :]
            )
            score = _score_prediction(candidate, val_target)
            late_delta = float(score["segments"]["late"]["score"]) - float(base_score["segments"]["late"]["score"])
            total_delta = float(score["segment_score"]) - float(base_score["segment_score"])
            rows.append({
                "candidate_id": f"ensemble_blend_w{start}_{stop}_a{weight:g}",
                "strategy": "blend",
                "window_start": start,
                "window_stop": stop,
                "blend_weight": weight,
                "segment_score": score["segment_score"],
                "task2_proxy": score["segment_score"] * 1.5,
                "segment_score_delta": total_delta,
                "late_score_delta": late_delta,
                "late_non_regression_passed": late_delta >= -float(late_non_regression_tolerance),
                "segments": score["segments"],
            })

    # Residual strategy with ensemble front
    for start, stop in windows:
        stop = min(stop, min_steps)
        if start >= stop:
            continue
        for scale in residual_scales:
            candidate = np.array(base_pred, copy=True, dtype=np.float32)
            delta = ensemble_pred[:, start:stop, :] - base_pred[:, start:stop, :]
            candidate[:, start:stop, :] = base_pred[:, start:stop, :] + scale * delta
            score = _score_prediction(candidate, val_target)
            late_delta = float(score["segments"]["late"]["score"]) - float(base_score["segments"]["late"]["score"])
            total_delta = float(score["segment_score"]) - float(base_score["segment_score"])
            rows.append({
                "candidate_id": f"ensemble_residual_w{start}_{stop}_s{scale:g}",
                "strategy": "residual",
                "window_start": start,
                "window_stop": stop,
                "residual_scale": scale,
                "segment_score": score["segment_score"],
                "task2_proxy": score["segment_score"] * 1.5,
                "segment_score_delta": total_delta,
                "late_score_delta": late_delta,
                "late_non_regression_passed": late_delta >= -float(late_non_regression_tolerance),
                "segments": score["segments"],
            })

    ranked = sorted(
        rows,
        key=lambda item: (
            not bool(item["late_non_regression_passed"]),
            -float(item["segment_score_delta"]),
            -float(item["segment_score"]),
        ),
    )
    best = ranked[0]

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "inputs": {
            "base_checkpoint_path": str(base_checkpoint_path),
            "front_checkpoints": front_configs,
        },
        "base_score": base_score,
        "ensemble_standalone_score": ensemble_score,
        "front_standalone_scores": front_scores,
        "best_candidate": best,
        "candidate_records": rows,
        "late_non_regression_tolerance": float(late_non_regression_tolerance),
        "artifacts": {
            "output_dir": str(root),
            "report_path": str(report_path),
            "base_prediction": str(root / "base_prediction.npz"),
            "ensemble_prediction": str(root / "ensemble_prediction.npz"),
        },
        "next_command": "evaluate-strategy-promotion",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
