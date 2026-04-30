from __future__ import annotations

import argparse
import json
from pathlib import Path

from pde_agent.core.manual_operator_tuning import (
    evaluate_manual_candidate_grid,
    load_hdf5_tensor,
    write_report_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a manual Burgers operator tuning grid on validation tensors."
    )
    parser.add_argument("--context-path", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-prediction", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    context = load_hdf5_tensor(args.context_path)
    if context.shape[1] != 10:
        context = context[:, :10, :]
    target = load_hdf5_tensor(args.target_path)
    external_predictions = {}
    for spec in args.external_prediction:
        name, raw_path = spec.split("=", 1)
        external_predictions[name] = load_hdf5_tensor(Path(raw_path))

    report = evaluate_manual_candidate_grid(
        context=context,
        target=target,
        external_predictions=external_predictions,
        top_k=args.top_k,
    )
    write_report_artifacts(report=report, output_dir=args.output_dir)
    print(json.dumps(report["best_candidate"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
