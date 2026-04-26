"""Execute a minimal scheduled-sampling finetune for prepared FNO candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.fno_backend import (
    FNOBackendError,
    _extract_state_dict,
    _infer_pdebench_fno1d_config,
    _load_checkpoint_with_metadata,
    _load_windows,
    _make_pdebench_fno1d_class,
    _require_file,
    _require_torch,
)


class FNOFinetuneCandidateError(RuntimeError):
    pass


def execute_fno_finetune_candidate(
    *,
    candidate_plan_path: str | Path,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    max_train_steps: int = 1,
    learning_rate: float = 1e-4,
    force: bool = False,
) -> dict[str, Any]:
    torch = _require_torch()
    candidate_plan_file = _require_candidate_file(
        Path(candidate_plan_path), "candidate execution plan"
    )
    windows_file = _require_candidate_file(Path(windows_path), "windows")
    checkpoint_file = _require_candidate_file(Path(checkpoint_path), "checkpoint")
    candidate_plan = _load_json_object(candidate_plan_file, "candidate execution plan")
    candidate = _validate_candidate_plan(candidate_plan)
    parameters = candidate["parameters"]
    train_steps = _positive_int(max_train_steps, "max_train_steps")
    lr = _positive_float(learning_rate, "learning_rate")

    workspace = candidate_plan["workspace"]
    artifacts_dir = Path(workspace["artifacts_dir"])
    logs_dir = Path(workspace["logs_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "finetuned_checkpoint_path": artifacts_dir / "finetuned_fno_checkpoint.pt",
        "candidate_prediction_path": artifacts_dir / "candidate_prediction.npz",
        "candidate_execution_report_path": artifacts_dir
        / "candidate_execution_report.json",
    }
    _prepare_output_paths(paths=paths, force=force)

    try:
        context, target = _load_windows(windows_file)
        loaded = _load_checkpoint_with_metadata(checkpoint_file, torch=torch)
        state = _extract_state_dict(loaded["checkpoint"])
        if not isinstance(state, dict):
            raise FNOFinetuneCandidateError("unsupported checkpoint state for FNO finetune")
        config = _infer_pdebench_fno1d_config(state, torch=torch)
        model = _build_model(torch=torch, state=state, config=config)
        loss_trace = _finetune_model(
            torch=torch,
            model=model,
            config=config,
            context=context,
            target=target,
            rollout_steps=int(parameters["rollout_steps"]),
            teacher_forcing_decay=float(parameters["teacher_forcing_decay"]),
            max_train_steps=train_steps,
            learning_rate=lr,
        )
        prediction = _rollout_model(
            torch=torch,
            model=model,
            config=config,
            context=context,
            forecast_steps=target.shape[1],
        )
    except FNOBackendError as exc:
        raise FNOFinetuneCandidateError(str(exc)) from exc

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "finetune_metadata": {
                "source_checkpoint_path": str(checkpoint_file),
                "candidate_plan_path": str(candidate_plan_file),
                "candidate_id": candidate["experiment_id"],
                "max_train_steps": train_steps,
                "learning_rate": lr,
                "loss_trace": loss_trace,
            },
        },
        paths["finetuned_checkpoint_path"],
    )
    with paths["candidate_prediction_path"].open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction.astype(np.float32),
            metadata_json=json.dumps(
                {
                    "model_type": "fno",
                    "adapter_type": "pdebench_fno1d_finetuned",
                    "candidate_id": candidate["experiment_id"],
                    "source_checkpoint_path": str(checkpoint_file),
                    "finetuned_checkpoint_path": str(
                        paths["finetuned_checkpoint_path"]
                    ),
                },
                allow_nan=False,
                sort_keys=True,
            ),
        )

    report = _build_report(
        candidate_plan_file=candidate_plan_file,
        windows_file=windows_file,
        checkpoint_file=checkpoint_file,
        paths=paths,
        candidate=candidate,
        train_steps=train_steps,
        learning_rate=lr,
        loss_trace=loss_trace,
        prediction_shape=prediction.shape,
    )
    paths["candidate_execution_report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_model(*, torch: Any, state: dict[str, Any], config: Any) -> Any:
    model_class = _make_pdebench_fno1d_class(torch)
    model = model_class(config)
    model.load_state_dict(state)
    return model


def _finetune_model(
    *,
    torch: Any,
    model: Any,
    config: Any,
    context: np.ndarray,
    target: np.ndarray,
    rollout_steps: int,
    teacher_forcing_decay: float,
    max_train_steps: int,
    learning_rate: float,
) -> list[float]:
    if context.shape[1] < config.history_steps:
        raise FNOFinetuneCandidateError(
            "context_steps must be >= official FNO history steps"
        )
    horizon = min(int(rollout_steps), int(target.shape[1]))
    if horizon < 1:
        raise FNOFinetuneCandidateError("rollout_steps must include at least one step")
    teacher_forcing_ratio = min(max(float(teacher_forcing_decay), 0.0), 1.0)
    device = "cpu"
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    context_tensor = torch.as_tensor(context, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    grid = _grid(torch=torch, batch_size=context.shape[0], space_points=context.shape[2])

    loss_trace: list[float] = []
    for _ in range(max_train_steps):
        optimizer.zero_grad()
        history = context_tensor[:, -config.history_steps :, :]
        losses = []
        for time_index in range(horizon):
            model_input = torch.cat([history.permute(0, 2, 1), grid], dim=-1)
            prediction = model(model_input)
            target_step = target_tensor[:, time_index, :]
            losses.append(torch.nn.functional.mse_loss(prediction, target_step))
            blended_next = (
                teacher_forcing_ratio * target_step
                + (1.0 - teacher_forcing_ratio) * prediction.detach()
            )
            history = torch.cat([history[:, 1:, :], blended_next.unsqueeze(1)], dim=1)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FNOFinetuneCandidateError("training loss is non-finite")
        loss.backward()
        optimizer.step()
        loss_trace.append(float(loss.detach().cpu().item()))
    return loss_trace


def _rollout_model(
    *,
    torch: Any,
    model: Any,
    config: Any,
    context: np.ndarray,
    forecast_steps: int,
) -> np.ndarray:
    model.eval()
    device = "cpu"
    history = torch.as_tensor(
        context[:, -config.history_steps :, :],
        dtype=torch.float32,
        device=device,
    )
    grid = _grid(torch=torch, batch_size=context.shape[0], space_points=context.shape[2])
    predictions: list[Any] = []
    with torch.no_grad():
        for _ in range(forecast_steps):
            model_input = torch.cat([history.permute(0, 2, 1), grid], dim=-1)
            next_step = model(model_input)
            if not torch.isfinite(next_step).all():
                raise FNOFinetuneCandidateError("prediction contains non-finite values")
            predictions.append(next_step)
            history = torch.cat([history[:, 1:, :], next_step.unsqueeze(1)], dim=1)
    prediction = torch.stack(predictions, dim=1).detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(prediction)):
        raise FNOFinetuneCandidateError("prediction contains non-finite values")
    return prediction


def _grid(*, torch: Any, batch_size: int, space_points: int) -> Any:
    return (
        torch.linspace(0.0, 1.0, space_points, dtype=torch.float32)
        .view(1, space_points, 1)
        .repeat(batch_size, 1, 1)
    )


def _validate_candidate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("state_hint") != "CANDIDATE_EXECUTION_PREPARED":
        raise FNOFinetuneCandidateError(
            "candidate plan must include state_hint CANDIDATE_EXECUTION_PREPARED"
        )
    candidate = plan.get("candidate")
    if not isinstance(candidate, dict):
        raise FNOFinetuneCandidateError("candidate plan must include candidate object")
    if candidate.get("intervention") != "use_scheduled_sampling_rollout_finetune":
        raise FNOFinetuneCandidateError(
            f"unsupported intervention: {candidate.get('intervention')}"
        )
    parameters = candidate.get("parameters")
    if not isinstance(parameters, dict):
        raise FNOFinetuneCandidateError("candidate must include parameters object")
    if "rollout_steps" not in parameters or "teacher_forcing_decay" not in parameters:
        raise FNOFinetuneCandidateError(
            "scheduled sampling candidate requires rollout_steps and "
            "teacher_forcing_decay"
        )
    workspace = plan.get("workspace")
    if not isinstance(workspace, dict) or not isinstance(
        workspace.get("artifacts_dir"), str
    ):
        raise FNOFinetuneCandidateError(
            "candidate plan must include workspace.artifacts_dir"
        )
    return candidate


def _prepare_output_paths(*, paths: dict[str, Path], force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FNOFinetuneCandidateError(f"candidate output already exists: {names}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _build_report(
    *,
    candidate_plan_file: Path,
    windows_file: Path,
    checkpoint_file: Path,
    paths: dict[str, Path],
    candidate: dict[str, Any],
    train_steps: int,
    learning_rate: float,
    loss_trace: list[float],
    prediction_shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": "FNO_FINETUNE_CANDIDATE_EXECUTED",
        "candidate_plan_path": str(candidate_plan_file),
        "windows_path": str(windows_file),
        "source_checkpoint_path": str(checkpoint_file),
        "candidate": candidate,
        "train": {
            "train_steps": train_steps,
            "learning_rate": learning_rate,
            "loss_trace": loss_trace,
        },
        "prediction_shape": [int(axis) for axis in prediction_shape],
        "artifacts": {name: str(path) for name, path in paths.items()},
        "execution_status": "completed",
        "next_command": "score-local-rollout",
    }


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FNOFinetuneCandidateError(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise FNOFinetuneCandidateError(f"{label} must be a positive number")
    return float(value)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FNOFinetuneCandidateError(
            f"failed to parse {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise FNOFinetuneCandidateError(f"{label} JSON must be an object")
    return data


def _require_candidate_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FNOFinetuneCandidateError(f"missing {label} path: {path}")
    if not path.is_file():
        raise FNOFinetuneCandidateError(f"{label} path is not a file: {path}")
    return path
