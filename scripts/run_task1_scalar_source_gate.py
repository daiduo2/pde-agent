"""Run Task 1 scalar source-gate selection on local validation artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.hermes_local_judge import judge_hermes_submission
from pde_agent.core.manual_operator_tuning import load_hdf5_tensor
from pde_agent.core.task1_scalar_source_gate import fit_task1_scalar_source_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-submission-dir", required=True)
    parser.add_argument("--task1-test-path", required=True)
    parser.add_argument("--task1-target-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-report-path")
    parser.add_argument("--source-pred-path", action="append", default=[])
    parser.add_argument("--top-per-segment", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--min-selected", type=int, default=2)
    parser.add_argument(
        "--search-preset",
        choices=["compact", "balanced", "wide"],
        default="compact",
    )
    parser.add_argument("--timing-profile", default="mac_mini_m4_local_dev")
    parser.add_argument("--submission-id", default="team_hermes")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    context = load_hdf5_tensor(Path(args.task1_test_path), dataset_key="tensor")[:, :10, :]
    target = load_hdf5_tensor(Path(args.task1_target_path), dataset_key="tensor")
    base_prediction_path = Path(args.base_submission_dir) / "task1_pred.hdf5"
    base_prediction = _load_prediction(base_prediction_path)
    source_paths = _source_paths_from_args(
        explicit_paths=[Path(path) for path in args.source_pred_path],
        oracle_report_path=Path(args.oracle_report_path)
        if args.oracle_report_path
        else None,
        top_per_segment=int(args.top_per_segment),
    )
    source_predictions = [
        (str(path), _load_prediction(path))
        for path in source_paths
        if path.exists() and path != base_prediction_path
    ]
    preset = _search_preset(str(args.search_preset))
    result = fit_task1_scalar_source_gate(
        context=context,
        base_prediction=base_prediction,
        source_predictions=source_predictions,
        target=target,
        max_steps=int(args.max_steps),
        min_selected=int(args.min_selected),
        **preset,
    )
    search_elapsed = time.perf_counter() - started
    prediction_path = output_dir / "scalar_source_gate_prediction.npz"
    np.savez_compressed(prediction_path, prediction=result["prediction"])
    submission = build_competition_submission(
        task_id=1,
        test_path=Path(args.task1_test_path),
        output_dir=output_dir / "submission_package",
        submission_id=str(args.submission_id),
        method="external_prediction",
        source_prediction_path=prediction_path,
        source_prediction_key="prediction",
        train_time=search_elapsed,
        inference_time=0.0,
        timing_profile=str(args.timing_profile),
        force=bool(args.force),
    )
    judge = judge_hermes_submission(
        submission_dir=submission["artifacts"]["submission_dir"],
        output_path=output_dir / "judge.json",
        task1_test_path=Path(args.task1_test_path),
        task1_target_path=Path(args.task1_target_path),
    )
    report = {
        "schema_version": 1,
        "state_hint": "TASK1_SCALAR_SOURCE_GATE_RAN",
        "base_submission_dir": str(args.base_submission_dir),
        "source_count": len(source_predictions),
        "source_paths": [label for label, _prediction in source_predictions],
        "selected_specs": result["selected_specs"],
        "search_preset": str(args.search_preset),
        "score": result["score"],
        "submission_dir": submission["artifacts"]["submission_dir"],
        "judge_report_path": str(output_dir / "judge.json"),
        "total_score_proxy": judge["total_score_proxy"],
        "search_elapsed_seconds": search_elapsed,
        "recorded_train_time_seconds": search_elapsed,
        "recorded_inference_time_seconds": 0.0,
        "local_validation_warning": (
            "Gate thresholds are selected on the local validation target. "
            "This is a deployable selector shape, but its learned thresholds "
            "must be treated as validation-tuned hyperparameters."
        ),
    }
    (output_dir / "scalar_source_gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def _source_paths_from_args(
    *,
    explicit_paths: list[Path],
    oracle_report_path: Path | None,
    top_per_segment: int,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for path in explicit_paths:
        if path not in seen:
            paths.append(path)
            seen.add(path)
    if oracle_report_path is None:
        return paths
    report = json.loads(oracle_report_path.read_text(encoding="utf-8"))
    selections = report.get("selections")
    if not isinstance(selections, dict):
        return paths
    for segment in ["early", "middle", "late"]:
        segment_report = selections.get(segment)
        if not isinstance(segment_report, dict):
            continue
        sources = segment_report.get("sources")
        if not isinstance(sources, dict):
            continue
        ranked = sorted(
            sources.items(),
            key=lambda item: int(item[1]) if isinstance(item[1], int) else 0,
            reverse=True,
        )
        for source, _count in ranked[:top_per_segment]:
            if source == "base":
                continue
            path = Path(str(source))
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _search_preset(name: str) -> dict[str, Any]:
    if name == "compact":
        return {
            "windows": [(10, 57), (35, 57), (57, 105), (105, 200)],
            "feature_names": [
                "context_velocity_energy",
                "source_base_disagreement",
                "source_base_energy_ratio",
            ],
            "quantiles": [0.10, 0.25, 0.50, 0.75, 0.90],
            "directions": ["gt", "lt"],
            "blend_weights": [1.0, 0.75],
        }
    if name == "balanced":
        return {
            "windows": [
                (10, 20),
                (20, 35),
                (35, 57),
                (10, 57),
                (57, 80),
                (80, 105),
                (57, 105),
                (105, 200),
            ],
            "feature_names": [
                "context_velocity_energy",
                "context_gradient_energy",
                "source_base_disagreement",
                "source_base_energy_ratio",
                "source_jump_start",
            ],
            "quantiles": [0.10, 0.18, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90],
            "directions": ["gt", "lt"],
            "blend_weights": [1.0, 0.75, 0.5],
        }
    return {}


def _load_prediction(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            return np.asarray(data["prediction"], dtype=np.float32)
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["tensor"][:], dtype=np.float32)


if __name__ == "__main__":
    main()
