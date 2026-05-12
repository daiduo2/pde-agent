"""Evaluate FNO early/middle grafts against a stronger Task 1 baseline."""


import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pde_agent.core.competition_scoring import score_competition_prediction
from pde_agent.core.competition_submission import CONTEXT_STEPS, PREDICTION_STEPS, SPACE_POINTS
from pde_agent.core.task1_short_window_training import (
    Task1ShortWindowTrainingError,
    _build_model,
    _denormalize_prediction,
    _normalize_windows,
)


STATE_HINT = "TASK1_FNO_GRAFT_EVALUATOR_RAN"


class Task1FNOGraftEvaluatorError(RuntimeError):
    pass


def run_task1_fno_graft_evaluator(
    *,
    baseline_prediction_path: str | Path,
    checkpoint_path: str | Path,
    test_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    target_dataset_key: str = "tensor",
    windows: list[tuple[int, int]] | None = None,
    blend_weights: list[float] | None = None,
    late_non_regression_tolerance: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    artifacts = {
        "source_prediction_path": root / "fno_source_prediction.hdf5",
        "best_prediction_path": root / "best_graft_prediction.hdf5",
        "report_path": root / "task1_fno_graft_evaluator_report.json",
    }
    _guard_outputs(artifacts.values(), force=force)
    root.mkdir(parents=True, exist_ok=True)

    baseline = _load_hdf5_tensor(
        Path(baseline_prediction_path),
        dataset_key=prediction_dataset_key,
        label="baseline prediction",
    )
    target = _load_hdf5_tensor(
        Path(target_path),
        dataset_key=target_dataset_key,
        label="target",
    )
    context = _load_context(Path(test_path), dataset_key=dataset_key)
    baseline = _normalize_prediction_shape(baseline, label="baseline prediction")
    target = _normalize_prediction_shape(target, label="target")
    _validate_pair(baseline=baseline, target=target, context=context)

    source = predict_task1_fno_checkpoint(
        checkpoint_path=checkpoint_path,
        context=context,
    )
    _write_hdf5_tensor(artifacts["source_prediction_path"], source)
    baseline_score = _score_array(
        prediction=baseline,
        target=target,
        output_dir=root / "baseline_score",
    )
    source_score = _score_array(
        prediction=source,
        target=target,
        output_dir=root / "source_score",
    )

    rows: list[dict[str, Any]] = []
    for start, stop in _normalized_windows(windows):
        for weight in _normalized_weights(blend_weights):
            candidate = np.array(baseline, copy=True, dtype=np.float32)
            candidate[:, start:stop, :] = (
                (1.0 - weight) * baseline[:, start:stop, :]
                + weight * source[:, start:stop, :]
            )
            score = _score_array(
                prediction=candidate,
                target=target,
                output_dir=root / "candidate_scores" / f"w{start}_{stop}_a{weight:g}",
            )
            late_delta = (
                float(score["segments"]["late"]["score"])
                - float(baseline_score["segments"]["late"]["score"])
            )
            total_delta = float(score["task_prediction_score"]) - float(
                baseline_score["task_prediction_score"]
            )
            rows.append(
                {
                    "candidate_id": f"fno_graft_w{start}_{stop}_a{weight:g}",
                    "window_start": start,
                    "window_stop": stop,
                    "blend_weight": weight,
                    "task_prediction_score": score["task_prediction_score"],
                    "segment_score": score["segment_score"],
                    "task_prediction_score_delta": total_delta,
                    "late_score_delta": late_delta,
                    "late_non_regression_passed": late_delta
                    >= -float(late_non_regression_tolerance),
                    "segments": score["segments"],
                    "score_path": score.get("score_path"),
                }
            )
    if not rows:
        raise Task1FNOGraftEvaluatorError("no graft candidates were generated")
    ranked = sorted(
        rows,
        key=lambda item: (
            not bool(item["late_non_regression_passed"]),
            -float(item["task_prediction_score_delta"]),
            -float(item["task_prediction_score"]),
        ),
    )
    best = ranked[0]
    best_prediction = _materialize_candidate(
        baseline=baseline,
        source=source,
        start=int(best["window_start"]),
        stop=int(best["window_stop"]),
        weight=float(best["blend_weight"]),
    )
    _write_hdf5_tensor(artifacts["best_prediction_path"], best_prediction)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "objective": (
            "Use a trained FNO as an early/middle dynamics graft source on top of "
            "a stronger baseline, while protecting late-window performance."
        ),
        "inputs": {
            "baseline_prediction_path": str(baseline_prediction_path),
            "checkpoint_path": str(checkpoint_path),
            "test_path": str(test_path),
            "target_path": str(target_path),
            "dataset_key": dataset_key,
        },
        "baseline_score": baseline_score,
        "source_score": source_score,
        "best_candidate": best,
        "candidate_records": rows,
        "late_non_regression_tolerance": float(late_non_regression_tolerance),
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "next_command": "build-competition-submission or continue FNO graft search",
    }
    artifacts["report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_task1_fno_graft_from_optimization_report(
    *,
    optimization_report_path: str | Path,
    baseline_prediction_path: str | Path,
    test_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    target_dataset_key: str = "tensor",
    windows: list[tuple[int, int]] | None = None,
    blend_weights: list[float] | None = None,
    late_non_regression_tolerance: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    report_path = Path(optimization_report_path)
    if not report_path.is_file():
        raise Task1FNOGraftEvaluatorError(
            f"optimization report not found: {report_path}"
        )
    optimization_report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = optimization_report.get("selected_candidate")
    if not isinstance(selected, dict):
        raise Task1FNOGraftEvaluatorError("optimization report missing selected_candidate")
    artifacts = selected.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(
        artifacts.get("checkpoint_path"), str
    ):
        raise Task1FNOGraftEvaluatorError(
            "selected candidate missing artifacts.checkpoint_path"
        )
    graft_report = run_task1_fno_graft_evaluator(
        baseline_prediction_path=baseline_prediction_path,
        checkpoint_path=artifacts["checkpoint_path"],
        test_path=test_path,
        target_path=target_path,
        output_dir=output_dir,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        target_dataset_key=target_dataset_key,
        windows=windows,
        blend_weights=blend_weights,
        late_non_regression_tolerance=late_non_regression_tolerance,
        force=force,
    )
    graft_report["optimization_report_path"] = str(report_path)
    graft_report["selected_training_candidate"] = selected
    graft_report["artifacts"]["optimization_report_path"] = str(report_path)
    Path(graft_report["artifacts"]["report_path"]).write_text(
        json.dumps(graft_report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return graft_report


def predict_task1_fno_checkpoint(
    *,
    checkpoint_path: str | Path,
    context: np.ndarray,
) -> np.ndarray:
    torch = _require_torch()
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise Task1FNOGraftEvaluatorError(f"checkpoint not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    if not isinstance(metadata, dict):
        raise Task1FNOGraftEvaluatorError("checkpoint metadata missing")
    model_family = str(metadata.get("model_family", "fno_lite"))
    forecast_steps = int(metadata.get("forecast_steps", PREDICTION_STEPS - CONTEXT_STEPS))
    max_future_steps = PREDICTION_STEPS - CONTEXT_STEPS
    if forecast_steps <= 0 or forecast_steps > max_future_steps:
        raise Task1FNOGraftEvaluatorError(
            "graft source checkpoint forecast_steps must be in 1..190"
        )
    normalized_context, dummy_target, stats = _normalize_windows(
        context=context.astype(np.float32),
        target=np.zeros(
            (context.shape[0], forecast_steps, context.shape[2]),
            dtype=np.float32,
        ),
        normalization=str(metadata.get("normalization", "none")),
    )
    _ = dummy_target
    model = _build_model(
        torch=torch,
        model_family=model_family,
        context_steps=int(metadata.get("context_steps", CONTEXT_STEPS)),
        forecast_steps=forecast_steps,
        space_points=int(metadata.get("space_points", SPACE_POINTS)),
        hidden_channels=int(metadata.get("width", 32)),
        modes=int(metadata.get("modes", 8)),
        depth=int(metadata.get("depth", 1)),
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise Task1FNOGraftEvaluatorError("checkpoint model_state_dict missing")
    _load_model_state(model=model, state=state)
    model.eval()
    with torch.no_grad():
        prediction_model = model(
            torch.as_tensor(normalized_context, dtype=torch.float32)
        ).detach().cpu().numpy()
    future = _denormalize_prediction(
        prediction=prediction_model.astype(np.float32),
        stats=stats,
        normalization=str(metadata.get("normalization", "none")),
    )
    prediction = np.empty(
        (context.shape[0], PREDICTION_STEPS, SPACE_POINTS),
        dtype=np.float32,
    )
    prediction[:, :CONTEXT_STEPS, :] = context
    prediction[:, CONTEXT_STEPS:, :] = context[:, -1:, :]
    prediction[:, CONTEXT_STEPS : CONTEXT_STEPS + forecast_steps, :] = future[
        :, :forecast_steps, :
    ]
    return prediction


def _load_model_state(*, model: Any, state: dict[str, Any]) -> None:
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(state)
        return
    for name, sub_state in state.items():
        target = getattr(model, name, None)
        if hasattr(target, "load_state_dict"):
            target.load_state_dict(sub_state)
        elif name == "spectral_layers":
            for layer, layer_state in zip(model.spectral_layers, sub_state):
                if hasattr(layer, "load_state_dict"):
                    layer.load_state_dict(layer_state)
                elif (
                    isinstance(layer_state, dict)
                    and "weight" in layer_state
                    and hasattr(layer, "weight")
                ):
                    layer.weight.data.copy_(layer_state["weight"])
                else:
                    raise Task1FNOGraftEvaluatorError(
                        "cannot load spectral layer state"
                    )
        elif name == "local_layers":
            for layer, layer_state in zip(model.local_layers, sub_state):
                layer.load_state_dict(layer_state)
        else:
            raise Task1FNOGraftEvaluatorError(f"cannot load model state field: {name}")


def _score_array(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "prediction.hdf5"
    target_path = output_dir / "target.hdf5"
    score_path = output_dir / "score.json"
    _write_hdf5_tensor(pred_path, prediction)
    _write_hdf5_tensor(target_path, target)
    return score_competition_prediction(
        task_id=1,
        prediction_path=pred_path,
        target_path=target_path,
        output_path=score_path,
    )


def _materialize_candidate(
    *,
    baseline: np.ndarray,
    source: np.ndarray,
    start: int,
    stop: int,
    weight: float,
) -> np.ndarray:
    candidate = np.array(baseline, copy=True, dtype=np.float32)
    candidate[:, start:stop, :] = (
        (1.0 - weight) * baseline[:, start:stop, :]
        + weight * source[:, start:stop, :]
    )
    return candidate


def _load_hdf5_tensor(path: Path, *, dataset_key: str, label: str) -> np.ndarray:
    if not path.is_file():
        raise Task1FNOGraftEvaluatorError(f"{label} not found: {path}")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise Task1FNOGraftEvaluatorError(
                f"{label} missing dataset {dataset_key!r}: {path}"
            )
        return np.asarray(handle[dataset_key][:], dtype=np.float32)


def _load_context(path: Path, *, dataset_key: str) -> np.ndarray:
    data = _load_hdf5_tensor(path, dataset_key=dataset_key, label="test")
    if data.ndim != 3 or data.shape[1] < CONTEXT_STEPS:
        raise Task1FNOGraftEvaluatorError("test tensor must have rank 3 and >=10 steps")
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    if data.shape[2] != SPACE_POINTS:
        raise Task1FNOGraftEvaluatorError("test tensor spatial width must be 256 or 1024")
    return np.asarray(data[:, :CONTEXT_STEPS, :], dtype=np.float32)


def _normalize_prediction_shape(array: np.ndarray, *, label: str) -> np.ndarray:
    if array.ndim != 3:
        raise Task1FNOGraftEvaluatorError(f"{label} must be rank 3")
    if array.shape[1] == 1000:
        array = array[:, ::5, :]
    if array.shape[1] > PREDICTION_STEPS:
        array = array[:, :PREDICTION_STEPS, :]
    if array.shape[2] == 1024:
        array = array[:, :, ::4]
    if array.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise Task1FNOGraftEvaluatorError(
            f"{label} must have shape [N, 200, 256] after reduction"
        )
    if not np.isfinite(array).all():
        raise Task1FNOGraftEvaluatorError(f"{label} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _validate_pair(
    *,
    baseline: np.ndarray,
    target: np.ndarray,
    context: np.ndarray,
) -> None:
    if baseline.shape != target.shape:
        raise Task1FNOGraftEvaluatorError("baseline and target shapes differ")
    if baseline.shape[0] != context.shape[0]:
        raise Task1FNOGraftEvaluatorError("baseline and context sample counts differ")
    if not np.allclose(baseline[:, :CONTEXT_STEPS, :], context, atol=1e-3):
        raise Task1FNOGraftEvaluatorError("baseline first 10 steps do not match context")


def _normalized_windows(windows: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    rows = windows or [(10, 20), (10, 35), (10, 57), (20, 57), (35, 57), (57, 105)]
    out = []
    for start, stop in rows:
        start_i, stop_i = int(start), int(stop)
        if start_i < CONTEXT_STEPS or stop_i <= start_i or stop_i > PREDICTION_STEPS:
            raise Task1FNOGraftEvaluatorError("invalid graft window")
        out.append((start_i, stop_i))
    return out


def _normalized_weights(weights: list[float] | None) -> list[float]:
    rows = weights or [0.25, 0.5, 0.75, 1.0]
    out = []
    for value in rows:
        weight = float(value)
        if weight < 0.0 or weight > 1.0:
            raise Task1FNOGraftEvaluatorError("blend weights must be in [0, 1]")
        out.append(weight)
    return out


def _write_hdf5_tensor(path: Path, array: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=np.asarray(array, dtype=np.float32))


def _guard_outputs(paths: Any, *, force: bool) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not force:
        raise Task1FNOGraftEvaluatorError(
            "graft evaluator artifacts already exist: "
            + ", ".join(str(path) for path in existing)
        )


def _require_torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise Task1ShortWindowTrainingError("torch is required") from exc
