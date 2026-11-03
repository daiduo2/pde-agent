"""Task 2 data loading, normalization, and physics loss."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Task2FnoDataError(RuntimeError):
    pass


# Standard Task 2 Burgers discretization
dx = 0.00390625
dt = 0.05


def load_task2_data(
    train_paths: list[str | Path] | None = None,
    val_path: str | Path | None = None,
):
    """Load Task 2 training and validation data from HDF5.

    Args:
        train_paths: List of 3 training HDF5 paths. Defaults to official paths.
        val_path: Validation HDF5 path. Defaults to official path.

    Returns:
        (train, train_nu, val_context, val_target, val_nu)
    """
    h5py = __import__("h5py")

    if train_paths is None:
        base = "data/official/data_and_sample_submission/train_val_test_init"
        train_paths = [f"{base}/task2_part{i}_train.h5" for i in range(3)]
    if val_path is None:
        val_path = "data/official/data_and_sample_submission/train_val_test_init/task2_val.h5"

    train_parts = []
    nu_parts = []
    for path in train_paths:
        path_obj = Path(path)
        if not path_obj.is_file():
            raise Task2FnoDataError(f"missing training file: {path_obj}")
        with h5py.File(path_obj, "r") as f:
            tensor = f["tensor"][:]
            nu = f["nu"][:]
            train_parts.append(tensor[:, 3:203, :])
            nu_parts.append(nu)
    train = np.concatenate(train_parts, axis=0).astype(np.float32)
    train_nu = np.concatenate(nu_parts, axis=0).astype(np.float32)

    val_path_obj = Path(val_path)
    if not val_path_obj.is_file():
        raise Task2FnoDataError(f"missing validation file: {val_path_obj}")
    with h5py.File(val_path_obj, "r") as f:
        val_tensor = f["tensor"][:].astype(np.float32)
        val_nu = f["nu"][:].astype(np.float32)
    val_context = val_tensor[:, :10, :]
    val_target = val_tensor[:, :200, :]

    return train, train_nu, val_context, val_target, val_nu


def create_datasets(
    train, train_nu, val_context, val_target, val_nu, train_val_split=0.9
):
    """Split training data into train / internal val / external val."""
    N_train = int(len(train) * train_val_split)
    train_ctx = train[:N_train, :10, :]
    train_tgt = train[:N_train, 10:, :]
    train_nu_split = train_nu[:N_train]

    val_ctx = train[N_train:, :10, :]
    val_tgt = train[N_train:, 10:, :]
    val_nu_split = train_nu[N_train:]

    ext_val_ctx = val_context
    ext_val_tgt = val_target[:, 10:, :]

    return (
        (train_ctx, train_tgt, train_nu_split),
        (val_ctx, val_tgt, val_nu_split),
        (ext_val_ctx, ext_val_tgt, val_nu),
    )


def normalize_per_sample(context, target):
    """Per-sample normalization using context mean/std.

    Returns normalized context, normalized target, and stats dict.
    """
    mean = context.mean(axis=(1, 2), keepdims=True)
    std = context.std(axis=(1, 2), keepdims=True) + 1e-8
    return (context - mean) / std, (target - mean) / std, {"mean": mean, "std": std}


def burgers_physics_loss(torch, prediction_full, nu, dx=dx, dt=dt):
    """Compute Burgers equation residual on full trajectory.

    prediction_full: (B, T, X) where T is total time steps
    nu: (B,) viscosity values
    """
    B, T, X = prediction_full.shape

    u_t = (prediction_full[:, 2:, :] - prediction_full[:, :-2, :]) / (2.0 * dt)

    u_right = torch.roll(prediction_full, shifts=-1, dims=2)
    u_left = torch.roll(prediction_full, shifts=1, dims=2)
    u_x = (u_right - u_left) / (2.0 * dx)
    u_xx = (u_right - 2.0 * prediction_full + u_left) / (dx * dx)

    u_x_interior = u_x[:, 1:-1, :]
    u_xx_interior = u_xx[:, 1:-1, :]
    u_interior = prediction_full[:, 1:-1, :]

    nu_expanded = nu.view(B, 1, 1)
    residual = u_t + u_interior * u_x_interior - nu_expanded * u_xx_interior

    return torch.mean(residual ** 2)
