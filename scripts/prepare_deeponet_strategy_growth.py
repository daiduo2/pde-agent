from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

RUN_ID = "deeponet_strategy_growth_001"
EXPECTED_STATE_HINT = "DEEPONET_CANDIDATE_EVALUATED"


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "hermes_deeponet_strategy_growth"


def render_template(
    *,
    repo_root: Path,
    evaluation_report_paths: list[Path],
    output_dir: Path,
    long_window_threshold: int,
) -> str:
    root = fixture_root(repo_root)
    template = (root / "run_state.template.yaml").read_text(encoding="utf-8")
    evaluation_report_args = " ".join(
        f"--evaluation-report-path {shlex.quote(str(path))}"
        for path in evaluation_report_paths
    )
    replacements = {
        "{{REPO_ROOT}}": str(repo_root),
        "{{FIXTURE_ROOT}}": str(root),
        "{{RUN_ID}}": RUN_ID,
        "{{EVALUATION_REPORT_ARGS}}": evaluation_report_args,
        "{{OUTPUT_DIR}}": str(output_dir),
        "{{LONG_WINDOW_THRESHOLD}}": str(long_window_threshold),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def prepare_runtime(
    *,
    repo_root: Path,
    evaluation_report_paths: list[Path],
    output_dir: Path,
    long_window_threshold: int,
) -> dict[str, Path]:
    if not evaluation_report_paths:
        raise SystemExit("at least one evaluation report path is required")
    if long_window_threshold <= 0:
        raise SystemExit("long_window_threshold must be positive")
    resolved_reports = [path.resolve() for path in evaluation_report_paths]
    for path in resolved_reports:
        _validate_evaluation_report(path)

    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    run_state = root / "run_state.yaml"
    run_state.write_text(
        render_template(
            repo_root=repo_root,
            evaluation_report_paths=resolved_reports,
            output_dir=output_dir.resolve(),
            long_window_threshold=long_window_threshold,
        ),
        encoding="utf-8",
    )
    return {"run_state": run_state}


def _validate_evaluation_report(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"DeepONet evaluation report does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"DeepONet evaluation report is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"DeepONet evaluation report must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("DeepONet evaluation report must be a JSON object")
    if payload.get("state_hint") != EXPECTED_STATE_HINT:
        raise SystemExit(
            "DeepONet evaluation report must include state_hint "
            f"{EXPECTED_STATE_HINT}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hermes DeepONet strategy-growth runtime files."
    )
    parser.add_argument(
        "--evaluation-report-path",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pde-agent-hermes-deeponet-strategy-growth-20260427-001"),
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
        evaluation_report_paths=args.evaluation_report_path,
        output_dir=output_dir,
        long_window_threshold=args.long_window_threshold,
    )
    print(f"generated {outputs['run_state']}")
    for path in args.evaluation_report_path:
        print(f"evaluation_report {path}")
    print(f"output_dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
