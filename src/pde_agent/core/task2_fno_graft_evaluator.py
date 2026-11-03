"""Evaluate FNO front-window grafts for Task 2.

Combines a front-segment specialist model (0-95 steps) with a base model
(0-190 steps) using direct replacement, blend, and residual strategies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.manual_operator_tuning import score_burgers_segments
from pde_agent.core.task2_fno_data import load_task2_data, normalize_per_sample

STATE_HINT = "TASK2_FNO_GRAFT_EVALUATOR_RAN"


class Task2FnoGraftEvaluatorError(RuntimeError):
    pass


def _require_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise Task2FnoGraftEvaluatorError("torch is required") from exc


def _load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    torch = _require_torch()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise Task2FnoGraftEvaluatorError("checkpoint missing config")
    if "model_state_dict" not in ckpt:
        raise Task2FnoGraftEvaluatorError("checkpoint missing model_state_dict")
    return ckpt


def _build_model_from_config(config: dict[str, Any], device: Any) -> Any:
    torch = _require_torch()
    from pde_agent.core.task2_fno_model import FNOLiteRollout
    from pde_agent.core.task2_fno_model_v2 import FNOLiteRolloutV2

    forecast_steps = config.get("forecast_steps", 190)
    use_v2 = config.get("use_v2", False)
    use_film = config.get("use_film", False)
    use_coord = config.get("use_coord", False)
    use_residual = config.get("use_residual", False)
    use_groupnorm = config.get("use_groupnorm", False)
    dropout = config.get("dropout", 0.0)

    if use_v2 or use_film or use_coord or use_residual or use_groupnorm or dropout > 0:
        model = FNOLiteRolloutV2(
            torch=torch,
            context_steps=10,
            forecast_steps=forecast_steps,
            space_points=256,
            hidden_channels=config["hidden_channels"],
            modes=config["modes"],
            depth=config["depth"],
            device=device,
            use_film=use_film,
            use_coord=use_coord,
            use_residual=use_residual,
            use_groupnorm=use_groupnorm,
            dropout=dropout,
        )
    else:
        model = FNOLiteRollout(
            torch=torch,
            context_steps=10,
            forecast_steps=forecast_steps,
            space_points=256,
            hidden_channels=config["hidden_channels"],
            modes=config["modes"],
            depth=config["depth"],
            device=device,
        )
    return model


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


def run_task2_fno_graft_evaluator(
    *,
    base_checkpoint_path: str | Path,
    front_checkpoint_path: str | Path,
    output_dir: str | Path,
    val_data_path: str | Path | None = None,
    windows: list[tuple[int, int]] | None = None,
    blend_weights: list[float] | None = None,
    residual_scales: list[float] | None = None,
    late_non_regression_tolerance: float = 1.0,
    device: str = "mps",
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_path = root / "task2_fno_graft_evaluator_report.json"
    if not force and report_path.exists():
        raise Task2FnoGraftEvaluatorError(f"output exists and force=False: {root}")
    root.mkdir(parents=True, exist_ok=True)

    torch = _require_torch()
    device_obj = torch.device(device)

    # Load models
    base_ckpt = _load_checkpoint(base_checkpoint_path)
    front_ckpt = _load_checkpoint(front_checkpoint_path)

    base_model = _build_model_from_config(base_ckpt["config"], device_obj)
    base_model.load_state_dict(base_ckpt["model_state_dict"])

    front_model = _build_model_from_config(front_ckpt["config"], device_obj)
    front_model.load_state_dict(front_ckpt["model_state_dict"])

    # Load validation data
    _, _, val_context, val_target, val_nu = load_task2_data(val_path=val_data_path)

    # Run inference
    base_pred = _run_inference(base_model, val_context, val_nu, device_obj)
    front_pred = _run_inference(front_model, val_context, val_nu, device_obj)

    base_steps = base_pred.shape[1]
    front_steps = front_pred.shape[1]

    # Score base standalone
    base_score = _score_prediction(base_pred, val_target)

    # For front standalone score, pad missing steps with base predictions
    # so it forms a valid 200-step trajectory for scoring
    if front_pred.shape[1] < base_pred.shape[1]:
        front_padded = np.array(base_pred, copy=True, dtype=np.float32)
        front_padded[:, :front_pred.shape[1], :] = front_pred
        front_score = _score_prediction(front_padded, val_target)
    else:
        front_score = _score_prediction(front_pred, val_target)

    np.savez_compressed(root / "base_prediction.npz", prediction=base_pred)
    np.savez_compressed(root / "front_prediction.npz", prediction=front_pred)

    # Default windows and weights
    if windows is None:
        windows = [(0, 47), (0, 95), (47, 95)]
    if blend_weights is None:
        blend_weights = [0.25, 0.5, 0.75]
    if residual_scales is None:
        residual_scales = [0.25, 0.5, 0.75]

    rows: list[dict[str, Any]] = []

    # Direct replacement strategy
    for start, stop in windows:
        stop = min(stop, front_steps)
        if start >= stop:
            continue
        candidate = np.array(base_pred, copy=True, dtype=np.float32)
        candidate[:, start:stop, :] = front_pred[:, start:stop, :]
        score = _score_prediction(candidate, val_target)
        late_delta = float(score["segments"]["late"]["score"]) - float(base_score["segments"]["late"]["score"])
        total_delta = float(score["segment_score"]) - float(base_score["segment_score"])
        rows.append({
            "candidate_id": f"direct_w{start}_{stop}",
            "strategy": "direct",
            "window_start": start,
            "window_stop": stop,
            "blend_weight": None,
            "residual_scale": None,
            "segment_score": score["segment_score"],
            "task2_proxy": score["segment_score"] * 1.5,
            "segment_score_delta": total_delta,
            "late_score_delta": late_delta,
            "late_non_regression_passed": late_delta >= -float(late_non_regression_tolerance),
            "segments": score["segments"],
        })

    # Blend strategy
    for start, stop in windows:
        stop = min(stop, front_steps)
        if start >= stop:
            continue
        for weight in blend_weights:
            candidate = np.array(base_pred, copy=True, dtype=np.float32)
            candidate[:, start:stop, :] = (
                (1.0 - weight) * base_pred[:, start:stop, :]
                + weight * front_pred[:, start:stop, :]
            )
            score = _score_prediction(candidate, val_target)
            late_delta = float(score["segments"]["late"]["score"]) - float(base_score["segments"]["late"]["score"])
            total_delta = float(score["segment_score"]) - float(base_score["segment_score"])
            rows.append({
                "candidate_id": f"blend_w{start}_{stop}_a{weight:g}",
                "strategy": "blend",
                "window_start": start,
                "window_stop": stop,
                "blend_weight": weight,
                "residual_scale": None,
                "segment_score": score["segment_score"],
                "task2_proxy": score["segment_score"] * 1.5,
                "segment_score_delta": total_delta,
                "late_score_delta": late_delta,
                "late_non_regression_passed": late_delta >= -float(late_non_regression_tolerance),
                "segments": score["segments"],
            })

    # Residual correction strategy: front predicts delta from base
    for start, stop in windows:
        stop = min(stop, front_steps)
        if start >= stop:
            continue
        for scale in residual_scales:
            candidate = np.array(base_pred, copy=True, dtype=np.float32)
            delta = front_pred[:, start:stop, :] - base_pred[:, start:stop, :]
            candidate[:, start:stop, :] = base_pred[:, start:stop, :] + scale * delta
            score = _score_prediction(candidate, val_target)
            late_delta = float(score["segments"]["late"]["score"]) - float(base_score["segments"]["late"]["score"])
            total_delta = float(score["segment_score"]) - float(base_score["segment_score"])
            rows.append({
                "candidate_id": f"residual_w{start}_{stop}_s{scale:g}",
                "strategy": "residual",
                "window_start": start,
                "window_stop": stop,
                "blend_weight": None,
                "residual_scale": scale,
                "segment_score": score["segment_score"],
                "task2_proxy": score["segment_score"] * 1.5,
                "segment_score_delta": total_delta,
                "late_score_delta": late_delta,
                "late_non_regression_passed": late_delta >= -float(late_non_regression_tolerance),
                "segments": score["segments"],
            })

    if not rows:
        raise Task2FnoGraftEvaluatorError("no graft candidates were generated")

    # Rank: prioritize late non-regression, then total improvement, then absolute score
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
        "objective": (
            "Use a trained front-segment FNO as an early/middle dynamics graft source "
            "on top of a stronger base model, while protecting late-window performance."
        ),
        "inputs": {
            "base_checkpoint_path": str(base_checkpoint_path),
            "front_checkpoint_path": str(front_checkpoint_path),
            "val_data_path": str(val_data_path) if val_data_path else None,
        },
        "base_score": base_score,
        "front_score": front_score,
        "best_candidate": best,
        "candidate_records": rows,
        "late_non_regression_tolerance": float(late_non_regression_tolerance),
        "artifacts": {
            "output_dir": str(root),
            "report_path": str(report_path),
            "base_prediction": str(root / "base_prediction.npz"),
            "front_prediction": str(root / "front_prediction.npz"),
        },
        "next_command": "evaluate-strategy-promotion",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
