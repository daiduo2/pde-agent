from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUN_ID = "fno_strategy_growth_001"


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "hermes_fno_strategy_growth"


def render_template(
    *,
    repo_root: Path,
    fno_sweep_summary_path: Path,
    output_dir: Path,
    long_window_threshold: int,
) -> str:
    root = fixture_root(repo_root)
    template = (root / "run_state.template.yaml").read_text(encoding="utf-8")
    replacements = {
        "{{REPO_ROOT}}": str(repo_root),
        "{{FIXTURE_ROOT}}": str(root),
        "{{RUN_ID}}": RUN_ID,
        "{{FNO_SWEEP_SUMMARY_PATH}}": str(fno_sweep_summary_path),
        "{{OUTPUT_DIR}}": str(output_dir),
        "{{LONG_WINDOW_THRESHOLD}}": str(long_window_threshold),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def prepare_runtime(
    *,
    repo_root: Path,
    fno_sweep_summary_path: Path,
    output_dir: Path,
    long_window_threshold: int,
) -> dict[str, Path]:
    if not fno_sweep_summary_path.exists():
        raise SystemExit(f"FNO sweep summary does not exist: {fno_sweep_summary_path}")
    if not fno_sweep_summary_path.is_file():
        raise SystemExit(f"FNO sweep summary is not a file: {fno_sweep_summary_path}")
    if long_window_threshold <= 0:
        raise SystemExit("long_window_threshold must be positive")
    _validate_fno_sweep_summary_schema(fno_sweep_summary_path)

    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    run_state = root / "run_state.yaml"
    run_state.write_text(
        render_template(
            repo_root=repo_root,
            fno_sweep_summary_path=fno_sweep_summary_path.resolve(),
            output_dir=output_dir.resolve(),
            long_window_threshold=long_window_threshold,
        ),
        encoding="utf-8",
    )
    return {"run_state": run_state}


def _validate_fno_sweep_summary_schema(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FNO sweep summary must be valid JSON: {exc}") from exc

    winners = payload.get("comparison_winners")
    if not isinstance(winners, list) or not winners:
        raise SystemExit("FNO sweep summary must include non-empty comparison_winners")
    for index, winner in enumerate(winners):
        if not isinstance(winner, dict):
            raise SystemExit(f"comparison_winners[{index}] must be an object")
        delta = winner.get("winner_score_delta")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise SystemExit(
                f"comparison_winners[{index}].winner_score_delta must be numeric"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hermes FNO strategy-growth runtime files."
    )
    parser.add_argument(
        "--fno-sweep-summary-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pde-agent-hermes-fno-strategy-growth-20260427-001"),
    )
    parser.add_argument("--long-window-threshold", type=int, default=31)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    )
    outputs = prepare_runtime(
        repo_root=repo_root,
        fno_sweep_summary_path=args.fno_sweep_summary_path,
        output_dir=output_dir,
        long_window_threshold=args.long_window_threshold,
    )
    print(f"generated {outputs['run_state']}")
    print(f"fno_sweep_summary {args.fno_sweep_summary_path}")
    print(f"output_dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
