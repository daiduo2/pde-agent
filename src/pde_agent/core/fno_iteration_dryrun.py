"""One-command FNO diagnostic-to-config dry-run orchestration."""


import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from pde_agent.core.experiment_config import generate_experiment_config_from_file
from pde_agent.core.fno_rollout_diagnosis import diagnose_fno_rollout
from pde_agent.core.next_experiment_plan import plan_next_experiment_from_fno_summary
from pde_agent.core.pde_hypothesis import select_pde_hypothesis_from_fno_summary


class FNOIterationDryRunError(RuntimeError):
    pass


def run_fno_iteration_dryrun(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    windows_file = _require_file(Path(windows_path), "windows")
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    output_root = Path(output_dir)
    paths = _artifact_paths(output_root)
    _prepare_output_paths(paths=paths, force=force)
    preexisting_artifacts = {path for path in paths.values() if path.exists()}

    try:
        diagnosis = diagnose_fno_rollout(
            windows_path=windows_file,
            checkpoint_path=checkpoint_file,
            output_dir=output_root,
            nu=nu,
            dt=dt,
            dx=dx,
            force=force,
        )
        hypothesis = select_pde_hypothesis_from_fno_summary(
            summary_path=paths["fno_rollout_summary_path"],
            output_path=paths["selected_pde_hypothesis_path"],
        )
        plan = plan_next_experiment_from_fno_summary(
            summary_path=paths["fno_rollout_summary_path"],
            hypothesis_path=paths["selected_pde_hypothesis_path"],
            output_path=paths["next_experiment_plan_path"],
        )
        config = generate_experiment_config_from_file(
            plan_path=paths["next_experiment_plan_path"],
            output_path=paths["experiment_config_path"],
        )
        manifest = _build_manifest(
            windows_file=windows_file,
            checkpoint_file=checkpoint_file,
            paths=paths,
            diagnosis=diagnosis,
            hypothesis=hypothesis,
            plan=plan,
            config=config,
        )
        paths["iteration_manifest_path"].write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        _cleanup_artifacts(paths, preexisting_artifacts=preexisting_artifacts)
        raise


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "fno_prediction_path": output_dir / "fno_prediction.npz",
        "fno_score_path": output_dir / "fno_score.json",
        "fno_rollout_error_path": output_dir / "fno_rollout_error.json",
        "fno_burgers_physics_path": output_dir / "fno_burgers_physics.json",
        "fno_rollout_summary_path": output_dir / "fno_rollout_summary.json",
        "selected_pde_hypothesis_path": output_dir / "selected_pde_hypothesis.json",
        "next_experiment_plan_path": output_dir / "next_experiment_plan.json",
        "experiment_config_path": output_dir / "experiment_config.json",
        "iteration_manifest_path": output_dir / "iteration_manifest.json",
    }


def _prepare_output_paths(*, paths: dict[str, Path], force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FNOIterationDryRunError(f"dry-run artifact already exists: {names}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _cleanup_artifacts(
    paths: dict[str, Path],
    *,
    preexisting_artifacts: set[Path],
) -> None:
    for path in paths.values():
        if path in preexisting_artifacts:
            continue
        with suppress(OSError):
            if path.exists():
                path.unlink()


def _build_manifest(
    *,
    windows_file: Path,
    checkpoint_file: Path,
    paths: dict[str, Path],
    diagnosis: dict[str, Any],
    hypothesis: dict[str, Any],
    plan: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": "FNO_ITERATION_DRYRUN_COMPLETED",
        "inputs": {
            "windows_path": str(windows_file),
            "checkpoint_path": str(checkpoint_file),
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
        "stages": {
            "diagnosis_state_hint": diagnosis["state_hint"],
            "hypothesis_state_hint": hypothesis["state_hint"],
            "plan_state_hint": plan["state_hint"],
            "config_state_hint": config["state_hint"],
        },
        "metrics": {
            "adapter_type": diagnosis["model"]["adapter_type"],
            "competition_like_score_short": diagnosis["score"][
                "competition_like_score_short"
            ],
            "dominant_error_pattern": diagnosis["error_diagnosis"][
                "dominant_error_pattern"
            ],
            "physics_issue": diagnosis["physics"]["physics_issue"],
            "selected_hypothesis": hypothesis["selected_hypothesis"],
            "selected_intervention": plan["selected_intervention"],
            "change_scope": plan["change_scope"],
            "candidate_count": len(config["candidate_experiments"]),
            "requires_training": config["requires_training"],
        },
        "next_command": config["next_command"],
    }


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FNOIterationDryRunError(f"missing {label} path: {path}")
    if not path.is_file():
        raise FNOIterationDryRunError(f"{label} path is not a file: {path}")
    return path
