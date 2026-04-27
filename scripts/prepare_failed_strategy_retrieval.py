from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUN_ID = "failed_strategy_retrieval_001"


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "hermes_failed_strategy_retrieval"


def prepare_runtime(*, repo_root: Path, output_dir: Path) -> dict[str, Path]:
    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_evidence_path = output_dir / "failed_strategy_evidence_summary.json"
    _write_failed_evidence(failed_evidence_path)
    run_state = root / "run_state.yaml"
    run_state.write_text(
        render_template(
            repo_root=repo_root,
            output_dir=output_dir.resolve(),
            failed_evidence_path=failed_evidence_path.resolve(),
        ),
        encoding="utf-8",
    )
    return {
        "run_state": run_state,
        "output_dir": output_dir,
        "failed_evidence_path": failed_evidence_path,
    }


def render_template(
    *,
    repo_root: Path,
    output_dir: Path,
    failed_evidence_path: Path,
) -> str:
    root = fixture_root(repo_root)
    template = (root / "run_state.template.yaml").read_text(encoding="utf-8")
    replacements = {
        "{{REPO_ROOT}}": str(repo_root),
        "{{FIXTURE_ROOT}}": str(root),
        "{{RUN_ID}}": RUN_ID,
        "{{OUTPUT_DIR}}": str(output_dir),
        "{{FAILED_EVIDENCE_PATH}}": str(failed_evidence_path),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def _write_failed_evidence(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "model_family": "DeepONet",
        "source_state_hint": "DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        "evaluation_window_count": 4,
        "mean_score_delta": -0.6,
        "score_delta_std": 0.5,
        "long_window_score_delta": -0.3,
        "physics_residual_delta": 0.02,
        "inference_time_regression": 0.01,
        "failure_rate": 0.75,
        "best_strategy_id": "supervised_h8",
        "strategy_ranking": [
            {
                "rank": 1,
                "strategy_id": "supervised_h8",
                "training_strategy": "branch_trunk_supervised",
                "record_count": 4,
                "win_count": 1,
                "win_rate": 0.25,
                "mean_score": 29.0,
                "mean_score_delta": -0.6,
                "best_score": 31.0,
                "failure_rate": 0.75,
                "dominant_error_patterns": ["long_horizon_drift"],
                "physics_issues": ["burgers_residual_high"],
            }
        ],
    }
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hermes failed-strategy retrieval runtime files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pde-agent-hermes-failed-strategy-retrieval-001"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    )
    outputs = prepare_runtime(repo_root=repo_root, output_dir=output_dir)
    print(f"generated {outputs['run_state']}")
    print(f"failed_evidence_path {outputs['failed_evidence_path']}")
    print(f"output_dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
