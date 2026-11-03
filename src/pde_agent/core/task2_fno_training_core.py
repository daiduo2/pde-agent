"""Task 2 FNO training core with standardized interface and memory bridge.

Wraps model training in a contract that the autonomous loop can consume.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.manual_operator_tuning import score_burgers_segments
from pde_agent.core.task2_fno_data import (
    burgers_physics_loss,
    create_datasets,
    load_task2_data,
    normalize_per_sample,
)
from pde_agent.core.task2_fno_model import FNOLiteRollout
from pde_agent.core.task2_fno_model_v2 import FNOLiteRolloutV2

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

STATE_HINT = "TASK2_FNO_TRAINING_CORE_RAN"


class Task2FnoTrainingCoreError(RuntimeError):
    pass


def _require_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise Task2FnoTrainingCoreError("torch is required") from exc


def _time_weighted_mse(torch, pred, target, time_weight=None):
    """MSE with optional time decay weighting (later timesteps weighted more)."""
    if time_weight is None or time_weight <= 0:
        return torch.nn.functional.mse_loss(pred, target)
    B, T, X = pred.shape
    weights = torch.exp(time_weight * torch.linspace(0.0, 1.0, T, device=pred.device))
    weights = weights.view(1, T, 1) / weights.sum()
    return (weights * (pred - target) ** 2).sum() / B


def _spectral_gradient_loss(torch, pred, target, grad_weight=0.0):
    """L2 loss on spatial gradients in Fourier space."""
    if grad_weight is None or grad_weight <= 0:
        return torch.tensor(0.0, device=pred.device)
    pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
    target_grad = target[:, :, 1:] - target[:, :, :-1]
    pred_spec = torch.fft.rfft(pred_grad, dim=-1)
    target_spec = torch.fft.rfft(target_grad, dim=-1)
    return grad_weight * torch.mean(torch.abs(pred_spec - target_spec) ** 2)


def _temporal_difference_loss(torch, pred, temporal_weight=0.0):
    """Penalize temporal discontinuities (second-order diff in time)."""
    if temporal_weight is None or temporal_weight <= 0:
        return torch.tensor(0.0, device=pred.device)
    dt_pred = pred[:, 1:, :] - pred[:, :-1, :]
    return temporal_weight * torch.mean(dt_pred ** 2)


def train_task2_fno_model(
    *,
    hidden_channels: int = 96,
    modes: int = 48,
    depth: int = 5,
    batch_size: int = 32,
    train_steps: int = 10000,
    learning_rate: float = 1e-3,
    physics_weight: float = 0.01,
    eval_every: int = 1000,
    save_every: int = 5000,
    device: str = "mps",
    seed: int = 42,
    output_dir: str | Path = "/tmp/task2_fno_training",
    checkpoint_path: str | Path | None = None,
    resume_lr: float | None = None,
    train_data_paths: list[str | Path] | None = None,
    val_data_path: str | Path | None = None,
    force: bool = False,
    memory: Any | None = None,
    session_id: str | None = None,
    # V2 model flags
    use_v2: bool = False,
    use_film: bool = False,
    use_coord: bool = False,
    use_residual: bool = False,
    use_groupnorm: bool = False,
    dropout: float = 0.0,
    # Forecast steps (for front-window specialist models)
    forecast_steps: int = 190,
    # Auxiliary loss weights
    time_weight: float = 0.0,
    grad_weight: float = 0.0,
    temporal_weight: float = 0.0,
) -> dict[str, Any]:
    """Train a Task 2 FNO model with standardized report.

    Args:
        hidden_channels: Hidden channel count.
        modes: Fourier modes.
        depth: Number of spectral layers.
        batch_size: Training batch size.
        train_steps: Total training steps.
        learning_rate: Adam LR.
        physics_weight: Burgers residual loss weight.
        eval_every: Evaluate every N steps.
        save_every: Save checkpoint every N steps.
        device: "mps", "cuda", or "cpu".
        seed: Random seed.
        output_dir: Directory for artifacts.
        checkpoint_path: Optional path to resume from.
        resume_lr: Optional override LR when resuming.
        train_data_paths: Override default training HDF5 paths.
        val_data_path: Override default validation HDF5 path.
        force: Overwrite existing output.
        memory: Optional HypothesisLabMemory for cross-session recording.
        session_id: Session ID for memory recording.
        use_v2: Use FNOLiteRolloutV2 (enhanced model).
        use_film: Enable FiLM + nu_estimator conditioning.
        use_coord: Enable coordinate channel injection.
        use_residual: Enable residual output (learn increment).
        use_groupnorm: Enable GroupNorm per block.
        dropout: Dropout rate (0 to disable).
        forecast_steps: Number of forecast steps to predict (default 190).
        time_weight: Time-decay MSE weight (0 to disable).
        grad_weight: Spectral gradient loss weight (0 to disable).
        temporal_weight: Temporal smoothness loss weight (0 to disable).

    Returns:
        Standardized report dict with segment_score, task2_proxy, etc.
    """
    torch = _require_torch()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    if not force and (root / "training_report.json").exists() and checkpoint_path is None:
        raise Task2FnoTrainingCoreError(f"output exists and force=False: {root}")

    # Load data
    train_full, train_nu, val_context, val_target, val_nu = load_task2_data(
        train_paths=train_data_paths,
        val_path=val_data_path,
    )
    (train_ctx, train_tgt, train_nu_tr), (int_val_ctx, int_val_tgt, int_val_nu), (
        ext_val_ctx,
        ext_val_tgt,
        ext_val_nu,
    ) = create_datasets(train_full, train_nu, val_context, val_target, val_nu)

    # Slice targets to forecast_steps if needed
    if forecast_steps < train_tgt.shape[1]:
        train_tgt = train_tgt[:, :forecast_steps, :]
    if forecast_steps < int_val_tgt.shape[1]:
        int_val_tgt = int_val_tgt[:, :forecast_steps, :]
    if forecast_steps < ext_val_tgt.shape[1]:
        ext_val_tgt = ext_val_tgt[:, :forecast_steps, :]

    # Normalize
    train_ctx_n, train_tgt_n, _ = normalize_per_sample(train_ctx, train_tgt)
    int_val_ctx_n, int_val_tgt_n, _ = normalize_per_sample(int_val_ctx, int_val_tgt)
    ext_val_ctx_n, ext_val_tgt_n, ext_val_stats = normalize_per_sample(ext_val_ctx, ext_val_tgt)

    device_obj = torch.device(device)
    train_ctx_t = torch.as_tensor(train_ctx_n, dtype=torch.float32).to(device_obj)
    train_tgt_t = torch.as_tensor(train_tgt_n, dtype=torch.float32).to(device_obj)
    train_nu_t = torch.as_tensor(train_nu_tr, dtype=torch.float32).to(device_obj)
    int_val_ctx_t = torch.as_tensor(int_val_ctx_n, dtype=torch.float32).to(device_obj)
    int_val_tgt_t = torch.as_tensor(int_val_tgt_n, dtype=torch.float32).to(device_obj)
    int_val_nu_t = torch.as_tensor(int_val_nu, dtype=torch.float32).to(device_obj)
    ext_val_ctx_t = torch.as_tensor(ext_val_ctx_n, dtype=torch.float32).to(device_obj)
    ext_val_tgt_t = torch.as_tensor(ext_val_tgt_n, dtype=torch.float32).to(device_obj)
    ext_val_nu_t = torch.as_tensor(ext_val_nu, dtype=torch.float32).to(device_obj)

    ext_mean_t = torch.as_tensor(ext_val_stats["mean"], dtype=torch.float32).to(device_obj)
    ext_std_t = torch.as_tensor(ext_val_stats["std"], dtype=torch.float32).to(device_obj)

    # Build model (V2 if any enhanced feature is requested)
    if use_v2 or use_film or use_coord or use_residual or use_groupnorm or dropout > 0:
        model = FNOLiteRolloutV2(
            torch=torch,
            context_steps=10,
            forecast_steps=forecast_steps,
            space_points=256,
            hidden_channels=hidden_channels,
            modes=modes,
            depth=depth,
            device=device_obj,
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
            hidden_channels=hidden_channels,
            modes=modes,
            depth=depth,
            device=device_obj,
        )

    config = {
        "hidden_channels": hidden_channels,
        "modes": modes,
        "depth": depth,
        "batch_size": batch_size,
        "train_steps": train_steps,
        "learning_rate": learning_rate,
        "physics_weight": physics_weight,
        "device": device,
        "seed": seed,
        "forecast_steps": forecast_steps,
        "use_v2": use_v2,
        "use_film": use_film,
        "use_coord": use_coord,
        "use_residual": use_residual,
        "use_groupnorm": use_groupnorm,
        "dropout": dropout,
        "time_weight": time_weight,
        "grad_weight": grad_weight,
        "temporal_weight": temporal_weight,
    }

    start_step = 0
    best_ext_val_loss = float("inf")
    best_score = 0.0
    best_state = None
    best_global_step = 0

    # Resume from checkpoint
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "config" in ckpt:
            config.update(ckpt["config"])
        if "best_ext_val_loss" in ckpt:
            best_ext_val_loss = ckpt["best_ext_val_loss"]
        if "best_score" in ckpt:
            best_score = ckpt["best_score"]
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    lr = resume_lr if resume_lr is not None else learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = train_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )

    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    dx = 0.00390625
    dt = 0.05

    for step in range(total_steps):
        model.train_mode()

        indices = rng.choice(
            len(train_ctx_t), size=min(batch_size, len(train_ctx_t)), replace=False
        )
        batch_ctx = train_ctx_t[indices]
        batch_tgt = train_tgt_t[indices]
        batch_nu = train_nu_t[indices]

        batch_mean = batch_ctx.mean(dim=(1, 2), keepdim=True)
        batch_std = batch_ctx.std(dim=(1, 2), keepdim=True) + 1e-8

        optimizer.zero_grad()
        if use_film:
            pred = model(batch_ctx, cond=batch_nu)
        else:
            pred = model(batch_ctx)

        data_loss = _time_weighted_mse(torch, pred, batch_tgt, time_weight)

        physics_loss_val = torch.tensor(0.0, device=device_obj)
        if physics_weight > 0:
            pred_phys = pred * batch_std + batch_mean
            ctx_phys = batch_ctx * batch_std + batch_mean
            full_traj = torch.cat([ctx_phys, pred_phys], dim=1)
            physics_loss_val = burgers_physics_loss(torch, full_traj, batch_nu, dx, dt)

        grad_loss = _spectral_gradient_loss(torch, pred, batch_tgt, grad_weight)
        temporal_loss = _temporal_difference_loss(torch, pred, temporal_weight)

        total_loss = data_loss + physics_weight * physics_loss_val + grad_loss + temporal_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_every == 0 or step == 0:
            model.eval_mode()
            with torch.no_grad():
                if use_film:
                    int_pred = model(int_val_ctx_t, cond=int_val_nu_t)
                    ext_pred = model(ext_val_ctx_t, cond=ext_val_nu_t)
                else:
                    int_pred = model(int_val_ctx_t)
                    ext_pred = model(ext_val_ctx_t)
                int_val_loss = torch.nn.functional.mse_loss(int_pred, int_val_tgt_t).item()
                ext_val_loss = torch.nn.functional.mse_loss(ext_pred, ext_val_tgt_t).item()

            ext_pred_np = (ext_pred * ext_std_t + ext_mean_t).cpu().numpy()
            full_prediction = np.concatenate([ext_val_ctx, ext_pred_np], axis=1).astype(np.float32)

            if forecast_steps == 190:
                score = score_burgers_segments(prediction=full_prediction, target=val_target)
                t2_proxy = score["segment_score"] * 1.5
                seg_score = score["segment_score"]
            else:
                score = None
                t2_proxy = 0.0
                seg_score = 0.0

            global_step = start_step + step + 1
            if ext_val_loss < best_ext_val_loss:
                best_ext_val_loss = ext_val_loss
                best_score = t2_proxy if forecast_steps == 190 else ext_val_loss
                best_state = model.state_dict()
                best_global_step = global_step
                ckpt_path = root / "best_checkpoint.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "score": {
                            "segment_score": seg_score,
                            "task2_proxy": t2_proxy,
                        },
                        "step": global_step,
                        "best_ext_val_loss": best_ext_val_loss,
                        "best_score": best_score,
                    },
                    ckpt_path,
                )

            elapsed = time.perf_counter() - start_time
            if forecast_steps == 190:
                log_line = (
                    f"Step {global_step:6d} | data={data_loss.item():.6f} | "
                    f"phys={physics_loss_val.item():.6f} | ext_val={ext_val_loss:.6f} | "
                    f"seg={seg_score:.2f} | t2={t2_proxy:.2f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | time={elapsed:.0f}s"
                )
            else:
                log_line = (
                    f"Step {global_step:6d} | data={data_loss.item():.6f} | "
                    f"phys={physics_loss_val.item():.6f} | ext_val={ext_val_loss:.6f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | time={elapsed:.0f}s"
                )
            if grad_weight > 0:
                log_line += f" | grad={grad_loss.item():.6f}"
            if temporal_weight > 0:
                log_line += f" | temp={temporal_loss.item():.6f}"
            print(log_line)

        if (step + 1) % save_every == 0 and best_state is not None:
            global_step = start_step + step + 1
            ckpt_path = root / f"checkpoint_step{global_step:06d}.pt"
            torch.save(
                {
                    "model_state_dict": best_state,
                    "step": global_step,
                    "best_ext_val_loss": best_ext_val_loss,
                    "best_score": best_score,
                },
                ckpt_path,
            )

    runtime = time.perf_counter() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval_mode()
    with torch.no_grad():
        if use_film:
            ext_pred = model(ext_val_ctx_t, cond=ext_val_nu_t)
        else:
            ext_pred = model(ext_val_ctx_t)
        ext_pred_np = (ext_pred * ext_std_t + ext_mean_t).cpu().numpy()

    full_prediction = np.concatenate([ext_val_ctx, ext_pred_np], axis=1).astype(np.float32)
    if forecast_steps == 190:
        score = score_burgers_segments(prediction=full_prediction, target=val_target)
        t2_proxy = score["segment_score"] * 1.5
        seg_score = score["segment_score"]
        segments = score["segments"]
    else:
        score = None
        t2_proxy = 0.0
        seg_score = 0.0
        segments = {}

    param_count = sum(p.numel() for p in model.parameters())

    model_type = "fno_lite_norm_physics"
    if use_v2 or use_film or use_coord or use_residual or use_groupnorm:
        model_type = "fno_lite_v2"

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "model_type": model_type,
        "parameters": param_count,
        "config": config,
        "train_steps": train_steps,
        "best_global_step": best_global_step,
        "runtime_seconds": runtime,
        "segment_score": seg_score,
        "task2_proxy": t2_proxy,
        "segments": segments,
        "best_ext_val_loss": best_ext_val_loss,
        "artifacts": {
            "output_dir": str(root),
            "best_checkpoint": str(root / "best_checkpoint.pt"),
            "best_prediction": str(root / "best_prediction.npz"),
            "training_report": str(root / "training_report.json"),
        },
        "next_hint": "use score to compare against other task2 candidates",
    }

    (root / "training_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(root / "best_prediction.npz", prediction=full_prediction)

    # Hypothesis-lab memory bridge
    if memory is not None:
        family = "fno_lite_task2"
        if use_v2 or use_film or use_coord or use_residual or use_groupnorm:
            family = "fno_lite_v2_task2"
        outcome = {
            "candidate_family": family,
            "candidate_id": f"fno_task2_{hidden_channels}h_{modes}m_{depth}d_pw{physics_weight}",
            "task2_proxy": t2_proxy,
            "segment_score": seg_score,
            "best_ext_val_loss": best_ext_val_loss,
            "config": config,
            "runtime_seconds": runtime,
            "parameters": param_count,
        }
        memory.record_experiment(
            session_id=session_id or str(root.name),
            outcome=outcome,
        )

    return report
