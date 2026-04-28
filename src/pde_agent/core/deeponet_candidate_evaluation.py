"""Evaluate DeepONet candidate predictions with local diagnostics."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from pde_agent.core.burgers_physics_local import check_burgers_physics_npz
from pde_agent.core.burgers_residual_evaluator import evaluate_burgers_residual_file
from pde_agent.core.local_scoring import score_prediction_npz
from pde_agent.core.pde_hypothesis import select_pde_hypothesis_from_files
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz


class DeepONetCandidateEvaluationError(RuntimeError):
    pass


def evaluate_deeponet_candidate(
    *,
    candidate_report_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    baseline_score_path: str | Path | None = None,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    candidate_report_file = _require_file(
        Path(candidate_report_path), "DeepONet candidate execution report"
    )
    target_file = _require_file(Path(target_path), "target")
    candidate_report = _load_json_object(candidate_report_file, "candidate report")
    candidate = _validate_candidate_report(candidate_report)
    prediction_file = _require_file(
        Path(candidate_report["artifacts"]["candidate_prediction_path"]),
        "candidate prediction",
    )
    baseline_score_file = (
        _require_file(Path(baseline_score_path), "baseline score")
        if baseline_score_path is not None
        else None
    )

    output_root = Path(output_dir)
    paths = _artifact_paths(output_root)
    _prepare_output_paths(paths=paths, force=force)
    preexisting_artifacts = {path for path in paths.values() if path.exists()}

    try:
        score = score_prediction_npz(
            prediction_path=prediction_file,
            target_path=target_file,
            output_path=paths["score_path"],
        )
        error = diagnose_rollout_error_npz(
            prediction_path=prediction_file,
            target_path=target_file,
            score_path=paths["score_path"],
            output_path=paths["rollout_error_path"],
        )
        physics = check_burgers_physics_npz(
            prediction_path=prediction_file,
            target_path=target_file,
            output_path=paths["burgers_physics_path"],
            nu=nu,
            dt=dt,
            dx=dx,
        )
        physics_residual = evaluate_burgers_residual_file(
            trajectory_path=prediction_file,
            output_path=paths["burgers_residual_path"],
            array_key="prediction",
            nu=nu,
            dt=dt,
            dx=dx,
        )
        hypothesis = select_pde_hypothesis_from_files(
            score_path=paths["score_path"],
            error_diagnosis_path=paths["rollout_error_path"],
            physics_check_path=paths["burgers_physics_path"],
            output_path=paths["selected_hypothesis_path"],
        )
        report = _build_report(
            candidate_report_file=candidate_report_file,
            target_file=target_file,
            prediction_file=prediction_file,
            paths=paths,
            candidate=candidate,
            inputs=candidate_report.get("inputs"),
            score=score,
            error=error,
            physics=physics,
            physics_residual=physics_residual,
            hypothesis=hypothesis,
            baseline_score_file=baseline_score_file,
        )
        paths["evaluation_report_path"].write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception:
        _cleanup_artifacts(paths, preexisting_artifacts=preexisting_artifacts)
        raise


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "score_path": output_dir / "candidate_score.json",
        "rollout_error_path": output_dir / "candidate_rollout_error.json",
        "burgers_physics_path": output_dir / "candidate_burgers_physics.json",
        "burgers_residual_path": output_dir / "candidate_burgers_residual.json",
        "selected_hypothesis_path": output_dir
        / "candidate_selected_pde_hypothesis.json",
        "evaluation_report_path": output_dir / "candidate_evaluation_report.json",
    }


def _prepare_output_paths(*, paths: dict[str, Path], force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise DeepONetCandidateEvaluationError(
            f"DeepONet candidate evaluation artifact already exists: {names}"
        )
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


def _build_report(
    *,
    candidate_report_file: Path,
    target_file: Path,
    prediction_file: Path,
    paths: dict[str, Path],
    candidate: dict[str, Any],
    inputs: Any,
    score: dict[str, Any],
    error: dict[str, Any],
    physics: dict[str, Any],
    physics_residual: dict[str, Any],
    hypothesis: dict[str, Any],
    baseline_score_file: Path | None,
) -> dict[str, Any]:
    input_payload = inputs if isinstance(inputs, dict) else {}
    return {
        "schema_version": 1,
        "state_hint": "DEEPONET_CANDIDATE_EVALUATED",
        "candidate_report_path": str(candidate_report_file),
        "target_path": str(target_file),
        "prediction_path": str(prediction_file),
        "inputs": {
            "forecast_steps": _positive_int(
                input_payload.get("forecast_steps"), "inputs.forecast_steps"
            ),
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
        "candidate": {
            "experiment_id": candidate["experiment_id"],
            "start_index": candidate["start_index"],
            "train_steps": candidate["train_steps"],
            "training_strategy": candidate["training_strategy"],
        },
        "score": {
            "competition_like_score_short": score["competition_like_score_short"],
            "rel_mse": score["rel_mse"],
            "rmse": score["rmse"],
        },
        "error_diagnosis": {
            "dominant_error_pattern": error["dominant_error_pattern"],
            "late_rel_mse": error["late_rel_mse"],
            "time_error_slope": error["time_error_slope"],
            "shock_region_error_ratio": error["shock_region_error_ratio"],
        },
        "physics": {
            "physics_issue": physics["physics_issue"],
            "residual_ratio": physics["residual_ratio"],
            "total_variation_ratio": physics["total_variation_ratio"],
            "gradient_error_ratio": physics["gradient_error_ratio"],
        },
        "physics_residual": {
            "physics_issue": physics_residual["physics_issue"],
            "residual_rmse": physics_residual["residual_rmse"],
            "mass_drift": physics_residual["mass_drift"],
            "energy_growth_ratio": physics_residual["energy_growth_ratio"],
            "high_frequency_energy_ratio": physics_residual[
                "high_frequency_energy_ratio"
            ],
        },
        "selected_hypothesis": {
            "state_hint": hypothesis["state_hint"],
            "selected_hypothesis": hypothesis["selected_hypothesis"],
            "next_experiment_scope": hypothesis["next_experiment_scope"],
        },
        "baseline_comparison": _baseline_comparison(
            baseline_score_file=baseline_score_file,
            candidate_score=score["competition_like_score_short"],
        ),
        "next_command": "build-deeponet-evidence-sweep",
    }


def _baseline_comparison(
    *,
    baseline_score_file: Path | None,
    candidate_score: float,
) -> dict[str, Any] | None:
    if baseline_score_file is None:
        return None
    baseline = _load_json_object(baseline_score_file, "baseline score")
    baseline_score = baseline.get("competition_like_score_short")
    if not isinstance(baseline_score, int | float):
        raise DeepONetCandidateEvaluationError(
            "baseline score must include numeric competition_like_score_short"
        )
    baseline_value = float(baseline_score)
    candidate_value = float(candidate_score)
    return {
        "baseline_score_path": str(baseline_score_file),
        "baseline_score": baseline_value,
        "candidate_score": candidate_value,
        "score_delta": candidate_value - baseline_value,
        "improved": candidate_value > baseline_value,
    }


def _validate_candidate_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("state_hint") != "DEEPONET_CANDIDATE_EXECUTED":
        raise DeepONetCandidateEvaluationError(
            "candidate report must include state_hint DEEPONET_CANDIDATE_EXECUTED"
        )
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise DeepONetCandidateEvaluationError("candidate report must include candidate")
    if not isinstance(candidate.get("experiment_id"), str):
        raise DeepONetCandidateEvaluationError(
            "candidate must include string experiment_id"
        )
    if not isinstance(candidate.get("training_strategy"), str):
        raise DeepONetCandidateEvaluationError(
            "candidate must include string training_strategy"
        )
    start_index = _positive_or_zero_int(candidate.get("start_index"), "start_index")
    train_steps = _positive_int(candidate.get("train_steps"), "train_steps")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(
        artifacts.get("candidate_prediction_path"), str
    ):
        raise DeepONetCandidateEvaluationError(
            "candidate report must include artifacts.candidate_prediction_path"
        )
    return {
        "experiment_id": candidate["experiment_id"],
        "start_index": start_index,
        "train_steps": train_steps,
        "training_strategy": candidate["training_strategy"],
    }


def _positive_or_zero_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise DeepONetCandidateEvaluationError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise DeepONetCandidateEvaluationError(f"{label} must be a positive integer")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DeepONetCandidateEvaluationError(
            f"failed to parse {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise DeepONetCandidateEvaluationError(f"{label} JSON must be an object")
    return data


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise DeepONetCandidateEvaluationError(f"missing {label} path: {path}")
    if not path.is_file():
        raise DeepONetCandidateEvaluationError(f"{label} path is not a file: {path}")
    return path
