from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RUN_ID = "deeponet_trainable_loop_001"


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "hermes_deeponet_trainable_loop"


def render_template(
    *,
    repo_root: Path,
    output_dir: Path,
    windows_path: Path,
    baseline_score_path: Path,
) -> str:
    root = fixture_root(repo_root)
    template = (root / "run_state.template.yaml").read_text(encoding="utf-8")
    replacements = {
        "{{REPO_ROOT}}": str(repo_root),
        "{{FIXTURE_ROOT}}": str(root),
        "{{RUN_ID}}": RUN_ID,
        "{{OUTPUT_DIR}}": str(output_dir),
        "{{WINDOWS_PATH}}": str(windows_path),
        "{{BASELINE_SCORE_PATH}}": str(baseline_score_path),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def prepare_runtime(*, repo_root: Path, output_dir: Path) -> dict[str, Path]:
    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    windows_path = inputs_dir / "windows.npz"
    baseline_score_path = inputs_dir / "baseline_score.json"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_state = root / "run_state.yaml"
    run_state.write_text(
        render_template(
            repo_root=repo_root,
            output_dir=output_dir.resolve(),
            windows_path=windows_path.resolve(),
            baseline_score_path=baseline_score_path.resolve(),
        ),
        encoding="utf-8",
    )
    return {
        "run_state": run_state,
        "windows_path": windows_path,
        "baseline_score_path": baseline_score_path,
    }


def _write_windows(path: Path) -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
    context = np.zeros((2, 2, 16), dtype=np.float32)
    target = np.zeros((2, 3, 16), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.15)
        trend = 0.02 * np.cos(x) + 0.005 * sample
        context[sample, 0, :] = base
        context[sample, 1, :] = base + trend
        for time_index in range(3):
            target[sample, time_index, :] = base + (time_index + 2) * trend
    np.savez(path, context=context, target=target)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hermes DeepONet trainable-loop runtime files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pde-agent-hermes-deeponet-trainable-loop-20260427-001"),
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
    print(f"windows_path {outputs['windows_path']}")
    print(f"baseline_score_path {outputs['baseline_score_path']}")
    print(f"output_dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
