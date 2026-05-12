"""PDE hypothesis selection from local diagnostic artifacts."""


import json
from pathlib import Path
from typing import Any


class PDEHypothesisError(RuntimeError):
    pass


def select_pde_hypothesis_from_files(
    *,
    score_path: str | Path,
    error_diagnosis_path: str | Path,
    physics_check_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    score_file = Path(score_path)
    error_file = Path(error_diagnosis_path)
    physics_file = Path(physics_check_path)

    score = _load_json_object(score_file, "score")
    error = _load_json_object(error_file, "error diagnosis")
    physics = _load_json_object(physics_file, "physics check")

    result = select_pde_hypothesis(score=score, error=error, physics=physics)
    result.update(
        {
            "score_path": str(score_file),
            "error_diagnosis_path": str(error_file),
            "physics_check_path": str(physics_file),
        }
    )

    if output_path is not None:
        hypothesis_file = Path(output_path)
        result["hypothesis_path"] = str(hypothesis_file)
        hypothesis_file.parent.mkdir(parents=True, exist_ok=True)
        hypothesis_file.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return result


def select_pde_hypothesis_from_fno_summary(
    *,
    summary_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    summary_file = Path(summary_path)
    summary = _load_json_object(summary_file, "FNO rollout summary")
    state_hint = summary.get("state_hint")
    if state_hint != "FNO_ROLLOUT_DIAGNOSED":
        raise PDEHypothesisError(
            "FNO rollout summary must include state_hint "
            f"'FNO_ROLLOUT_DIAGNOSED', got {state_hint!r}"
        )

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PDEHypothesisError("FNO rollout summary must include object 'artifacts'")

    result = select_pde_hypothesis_from_files(
        score_path=_require_artifact_path(
            artifacts, "score_json_path", "FNO rollout summary"
        ),
        error_diagnosis_path=_require_artifact_path(
            artifacts, "rollout_error_json_path", "FNO rollout summary"
        ),
        physics_check_path=_require_artifact_path(
            artifacts, "burgers_physics_json_path", "FNO rollout summary"
        ),
        output_path=output_path,
    )
    result["summary_path"] = str(summary_file)
    if output_path is not None:
        hypothesis_file = Path(output_path)
        result["hypothesis_path"] = str(hypothesis_file)
        hypothesis_file.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def select_pde_hypothesis(
    *,
    score: dict[str, Any],
    error: dict[str, Any],
    physics: dict[str, Any],
) -> dict[str, Any]:
    pattern = _require_string(error, "dominant_error_pattern", "error diagnosis")
    issue = _require_string(physics, "physics_issue", "physics check")
    selected = _select(pattern=pattern, issue=issue)
    details = _HYPOTHESIS_DETAILS[selected]

    score_value = score.get("competition_like_score_short")
    score_clause = (
        f", score={score_value}" if isinstance(score_value, int | float) else ""
    )

    return {
        "selected_hypothesis": selected,
        "rationale": (
            f"Selected from dominant_error_pattern={pattern} and "
            f"physics_issue={issue}{score_clause}."
        ),
        "expected_effect": details["expected_effect"],
        "risk": details["risk"],
        "next_experiment_scope": details["next_experiment_scope"],
        "required_artifacts": details["required_artifacts"],
        "rejected_hypotheses": details["rejected_hypotheses"],
        "state_hint": "PDE_HYPOTHESIS_SELECTED",
    }


def _select(*, pattern: str, issue: str) -> str:
    if pattern in {"overdamped_rollout", "overdamped"} or issue == "overdissipation":
        return "add_spectral_consistency_loss"
    if pattern == "long_horizon_drift":
        return "use_scheduled_sampling_rollout_finetune"
    if issue == "residual_mismatch":
        return "add_residual_correction_head"
    if issue == "nonphysical_oscillation" or pattern == "oscillatory_error":
        return "apply_low_pass_stabilization"
    if pattern == "shock_region_error":
        return "add_gradient_weighted_loss"
    return "keep_baseline_and_collect_stronger_evidence"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise PDEHypothesisError(f"missing {label} path: {path}")
    if not path.is_file():
        raise PDEHypothesisError(f"{label} path is not a file: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PDEHypothesisError(f"failed to parse {label} JSON {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise PDEHypothesisError(f"{label} JSON must be an object")
    return data


def _require_string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise PDEHypothesisError(f"{label} must include string field '{field}'")
    return value


def _require_artifact_path(
    artifacts: dict[Any, Any], field: str, label: str
) -> str:
    value = artifacts.get(field)
    if not isinstance(value, str):
        raise PDEHypothesisError(f"{label} artifacts must include string '{field}'")
    return value


_COMMON_ARTIFACTS = [
    "local rollout score JSON",
    "rollout error diagnosis JSON",
    "Burgers physics check JSON",
]

_HYPOTHESIS_DETAILS: dict[str, dict[str, Any]] = {
    "add_spectral_consistency_loss": {
        "expected_effect": "reduce high-frequency damping while preserving rollout stability",
        "risk": "may amplify oscillations near shocks if overweighted",
        "next_experiment_scope": "single short finetune with spectral loss weight sweep",
        "required_artifacts": _COMMON_ARTIFACTS
        + ["spectrum comparison", "rollout stability check"],
        "rejected_hypotheses": [
            "apply_low_pass_stabilization",
            "keep_baseline_and_collect_stronger_evidence",
        ],
    },
    "use_scheduled_sampling_rollout_finetune": {
        "expected_effect": "reduce exposure-bias drift over later rollout steps",
        "risk": "can destabilize early-step accuracy if schedule ramps too quickly",
        "next_experiment_scope": "short rollout finetune with scheduled sampling ramp",
        "required_artifacts": _COMMON_ARTIFACTS
        + ["early/middle/late error breakdown"],
        "rejected_hypotheses": [
            "add_spectral_consistency_loss",
            "keep_baseline_and_collect_stronger_evidence",
        ],
    },
    "add_residual_correction_head": {
        "expected_effect": "improve PDE residual alignment without replacing the base rollout",
        "risk": "may overfit finite-difference proxy residuals on local windows",
        "next_experiment_scope": "residual-head ablation against unchanged baseline",
        "required_artifacts": _COMMON_ARTIFACTS + ["residual ratio comparison"],
        "rejected_hypotheses": [
            "apply_low_pass_stabilization",
            "use_scheduled_sampling_rollout_finetune",
        ],
    },
    "apply_low_pass_stabilization": {
        "expected_effect": "suppress nonphysical oscillations during rollout",
        "risk": "may worsen overdamping and shock smearing",
        "next_experiment_scope": "low-pass threshold sweep with shock-region error audit",
        "required_artifacts": _COMMON_ARTIFACTS
        + ["oscillation evidence", "shock-region error ratio"],
        "rejected_hypotheses": [
            "add_spectral_consistency_loss",
            "add_gradient_weighted_loss",
        ],
    },
    "add_gradient_weighted_loss": {
        "expected_effect": "improve accuracy near sharp gradients and shock-like regions",
        "risk": "may trade off smooth-region accuracy if gradients dominate the objective",
        "next_experiment_scope": "gradient-weighted loss ablation on shock-heavy windows",
        "required_artifacts": _COMMON_ARTIFACTS + ["shock-region error evidence"],
        "rejected_hypotheses": [
            "apply_low_pass_stabilization",
            "keep_baseline_and_collect_stronger_evidence",
        ],
    },
    "keep_baseline_and_collect_stronger_evidence": {
        "expected_effect": "avoid speculative tuning until diagnostics identify a dominant failure mode",
        "risk": "delays potentially useful model changes",
        "next_experiment_scope": "collect additional diagnostic windows and rerun scoring",
        "required_artifacts": _COMMON_ARTIFACTS
        + ["additional local windows", "repeat diagnostics"],
        "rejected_hypotheses": [
            "add_spectral_consistency_loss",
            "use_scheduled_sampling_rollout_finetune",
        ],
    },
}
