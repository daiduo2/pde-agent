from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pde_agent.core.burgers_hdf5_io import load_hdf5_tensor
from pde_agent.core.competition_submission import build_competition_submission
from pde_agent.core.submission_judge import judge_submission
from pde_agent.core.manual_operator_tuning import (
    linear_trend_prediction,
    spectral_smooth_prediction,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a manually tuned lightweight operator candidate."
    )
    parser.add_argument("--context-path", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submission-id", default="manual_operator_tuning")
    parser.add_argument("--slope-window", type=int, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--spectral-strength", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    context = load_hdf5_tensor(args.context_path)
    if context.shape[1] != 10:
        context = context[:, :10, :]
    target = load_hdf5_tensor(args.target_path)

    clip = (float(np.min(context)), float(np.max(context)))
    start = time.perf_counter()
    prediction = linear_trend_prediction(
        context=context,
        gain=args.gain,
        slope_window=args.slope_window,
        decay=args.decay,
        clip=clip,
    )
    if args.spectral_strength > 0.0:
        prediction = spectral_smooth_prediction(
            prediction=prediction,
            strength=args.spectral_strength,
        )
    elapsed = time.perf_counter() - start

    source_path = root / "manual_candidate_prediction.npz"
    np.savez_compressed(source_path, prediction=prediction)
    candidate_report_path = root / "manual_candidate_report.json"
    candidate_report = {
        "schema_version": 1,
        "state_hint": "MANUAL_OPERATOR_CANDIDATE_READY",
        "candidate": {
            "candidate_id": (
                "manual_linear_trend_spectral"
                if args.spectral_strength > 0.0
                else "manual_linear_trend"
            ),
            "family": "linear_trend_spectral"
            if args.spectral_strength > 0.0
            else "linear_trend",
            "params": {
                "slope_window": args.slope_window,
                "gain": args.gain,
                "decay": args.decay,
                "spectral_strength": args.spectral_strength,
            },
        },
        "artifacts": {
            "candidate_prediction_path": str(source_path),
        },
        "train": {
            "elapsed_seconds": 0.0,
        },
        "inference": {
            "elapsed_seconds": elapsed,
        },
    }
    candidate_report_path.write_text(
        json.dumps(candidate_report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    submission = build_competition_submission(
        task_id=1,
        test_path=args.context_path,
        output_dir=root / "submission_package",
        submission_id=args.submission_id,
        method="external_prediction",
        source_prediction_path=source_path,
        source_prediction_key="prediction",
        train_time=0.0,
        inference_time=elapsed,
        force=True,
    )
    judge = judge_submission(
        submission_dir=submission["artifacts"]["submission_dir"],
        output_path=root / "judge_report.json",
        task1_test_path=args.context_path,
        task1_target_path=args.target_path,
    )
    result = {
        "candidate_report_path": str(candidate_report_path),
        "submission_dir": submission["artifacts"]["submission_dir"],
        "judge_report_path": str(root / "judge_report.json"),
        "inference_time": elapsed,
        "judge_total_score_proxy": judge["total_score_proxy"],
        "judge_decision": judge["decision"],
        "segment_score": judge["tasks"]["1"]["score"]["segment_score"],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
