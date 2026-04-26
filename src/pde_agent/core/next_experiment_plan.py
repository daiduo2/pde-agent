"""Deterministic next-experiment planning from PDE diagnostic artifacts."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


class NextExperimentPlanError(RuntimeError):
    pass


def plan_next_experiment_from_files(
    *,
    score_path: str | Path,
    error_diagnosis_path: str | Path,
    physics_check_path: str | Path,
    hypothesis_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    score_file = Path(score_path)
    error_file = Path(error_diagnosis_path)
    physics_file = Path(physics_check_path)
    hypothesis_file = Path(hypothesis_path)

    score = _load_json_object(score_file, "score")
    error = _load_json_object(error_file, "error diagnosis")
    physics = _load_json_object(physics_file, "physics check")
    hypothesis = _load_json_object(hypothesis_file, "hypothesis")

    plan = plan_next_experiment(
        score=score,
        error=error,
        physics=physics,
        hypothesis=hypothesis,
    )
    plan.update(
        {
            "score_path": str(score_file),
            "error_diagnosis_path": str(error_file),
            "physics_check_path": str(physics_file),
            "hypothesis_path": str(hypothesis_file),
        }
    )

    if output_path is not None:
        plan_file = Path(output_path)
        plan["next_experiment_plan_path"] = str(plan_file)
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(
            json.dumps(plan, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return plan


def plan_next_experiment_from_fno_summary(
    *,
    summary_path: str | Path,
    hypothesis_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    summary_file = Path(summary_path)
    summary = _load_json_object(summary_file, "FNO rollout summary")
    state_hint = summary.get("state_hint")
    if state_hint != "FNO_ROLLOUT_DIAGNOSED":
        raise NextExperimentPlanError(
            "FNO rollout summary must include state_hint "
            f"'FNO_ROLLOUT_DIAGNOSED', got {state_hint!r}"
        )

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise NextExperimentPlanError(
            "FNO rollout summary must include object 'artifacts'"
        )

    plan = plan_next_experiment_from_files(
        score_path=_require_artifact_path(
            artifacts, "score_json_path", "FNO rollout summary"
        ),
        error_diagnosis_path=_require_artifact_path(
            artifacts, "rollout_error_json_path", "FNO rollout summary"
        ),
        physics_check_path=_require_artifact_path(
            artifacts, "burgers_physics_json_path", "FNO rollout summary"
        ),
        hypothesis_path=hypothesis_path,
        output_path=output_path,
    )
    plan["summary_path"] = str(summary_file)
    if output_path is not None:
        plan_file = Path(output_path)
        plan["next_experiment_plan_path"] = str(plan_file)
        plan_file.write_text(
            json.dumps(plan, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return plan


def plan_next_experiment(
    *,
    score: dict[str, Any],
    error: dict[str, Any],
    physics: dict[str, Any],
    hypothesis: dict[str, Any],
) -> dict[str, Any]:
    selected_hypothesis = _require_string(
        hypothesis, "selected_hypothesis", "hypothesis"
    )
    dominant_pattern = _require_string(
        error, "dominant_error_pattern", "error diagnosis"
    )
    physics_issue = _require_string(physics, "physics_issue", "physics check")

    selected_intervention = _select_intervention(
        selected_hypothesis=selected_hypothesis,
        dominant_pattern=dominant_pattern,
        physics_issue=physics_issue,
    )
    policy = deepcopy(_INTERVENTION_POLICIES[selected_intervention])
    score_value = _number_or_none(score.get("competition_like_score_short"))

    return {
        "schema_version": 1,
        "state_hint": "NEXT_EXPERIMENT_PLANNED",
        "selected_intervention": selected_intervention,
        "source_hypothesis": selected_hypothesis,
        "failure_mode": {
            "dominant_error_pattern": dominant_pattern,
            "physics_issue": physics_issue,
            "score_band": _score_band(score_value),
            "evidence": {
                "competition_like_score_short": score_value,
                "rel_mse": _number_or_none(score.get("rel_mse")),
                "rmse": _number_or_none(score.get("rmse")),
                "residual_ratio": _number_or_none(physics.get("residual_ratio")),
                "total_variation_ratio": _number_or_none(
                    physics.get("total_variation_ratio")
                ),
            },
        },
        "change_scope": policy["change_scope"],
        "intervention_plan": {
            "description": policy["description"],
            "allowed_actions": policy["allowed_actions"],
            "forbidden_actions": policy["forbidden_actions"],
            "candidate_parameters": policy["candidate_parameters"],
        },
        "budget": policy["budget"],
        "controls": {
            "baseline_required": True,
            "baseline_artifacts": [
                "previous prediction NPZ",
                "previous score JSON",
                "previous physics check JSON",
            ],
        },
        "acceptance_criteria": policy["acceptance_criteria"],
        "rejection_criteria": policy["rejection_criteria"],
        "required_next_artifacts": [
            "candidate prediction NPZ",
            "candidate score JSON",
            "candidate rollout error diagnosis JSON",
            "candidate Burgers physics check JSON",
            "candidate selected hypothesis JSON",
        ],
        "next_command": "generate-experiment-config",
        "rationale": (
            "Selected because the previous hypothesis, dominant error pattern, "
            "and physics check indicate "
            f"{selected_intervention} is the narrowest supported intervention."
        ),
    }


def _select_intervention(
    *,
    selected_hypothesis: str,
    dominant_pattern: str,
    physics_issue: str,
) -> str:
    if (
        physics_issue == "nonphysical_oscillation"
        or selected_hypothesis == "apply_low_pass_stabilization"
    ):
        return "apply_low_pass_stabilization"
    if dominant_pattern == "long_horizon_drift":
        return "use_scheduled_sampling_rollout_finetune"
    if dominant_pattern in {"overdamped_rollout", "overdamped"} or physics_issue in {
        "overdissipation",
        "high_frequency_suppression",
    }:
        return "add_spectral_consistency_loss"
    if dominant_pattern == "shock_region_error":
        return "add_gradient_weighted_loss"
    if physics_issue == "residual_mismatch":
        return "add_residual_correction_head"
    if selected_hypothesis in _INTERVENTION_POLICIES:
        return selected_hypothesis
    return "keep_baseline_and_collect_stronger_evidence"


def _score_band(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 95.0:
        return "excellent"
    if score >= 85.0:
        return "high"
    if score >= 60.0:
        return "medium"
    return "low"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise NextExperimentPlanError(f"missing {label} path: {path}")
    if not path.is_file():
        raise NextExperimentPlanError(f"{label} path is not a file: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NextExperimentPlanError(
            f"failed to parse {label} JSON {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise NextExperimentPlanError(f"{label} JSON must be an object")
    return data


def _require_string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise NextExperimentPlanError(f"{label} must include string field '{field}'")
    return value


def _require_artifact_path(artifacts: dict[Any, Any], field: str, label: str) -> str:
    value = artifacts.get(field)
    if not isinstance(value, str):
        raise NextExperimentPlanError(
            f"{label} artifacts must include string '{field}'"
        )
    return value


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


_POSTPROCESS_BUDGET = {
    "max_experiment_count": 3,
    "max_train_time_seconds": 0,
    "max_inference_time_seconds": 120,
    "requires_training": False,
}

_TRAINING_BUDGET = {
    "max_experiment_count": 2,
    "max_train_time_seconds": 3600,
    "max_inference_time_seconds": 120,
    "requires_training": True,
}

_DIAGNOSTIC_BUDGET = {
    "max_experiment_count": 3,
    "max_train_time_seconds": 0,
    "max_inference_time_seconds": 120,
    "requires_training": False,
}

_COMMON_FORBIDDEN_ACTIONS = [
    "change training data split",
    "submit without baseline comparison",
    "overwrite previous artifacts",
]

_INTERVENTION_POLICIES: dict[str, dict[str, Any]] = {
    "apply_low_pass_stabilization": {
        "change_scope": "postprocess_or_inference_only",
        "description": (
            "Sweep low-pass thresholds to suppress nonphysical oscillations "
            "while auditing shock-region error."
        ),
        "allowed_actions": [
            "add deterministic low-pass postprocessing",
            "sweep spectral cutoff",
            "compare against unchanged prediction",
        ],
        "forbidden_actions": _COMMON_FORBIDDEN_ACTIONS,
        "candidate_parameters": {
            "low_pass_cutoff_ratio": [0.5, 0.625, 0.75],
            "preserve_mean": [True],
        },
        "budget": _POSTPROCESS_BUDGET,
        "acceptance_criteria": [
            "competition_like_score_short improves by at least 1.0",
            "total_variation_ratio moves closer to 1.0",
            "shock-region error does not worsen by more than 5%",
        ],
        "rejection_criteria": [
            "score decreases by more than 0.5",
            "physics issue changes to overdissipation",
            "prediction contains non-finite values",
        ],
    },
    "add_gradient_weighted_loss": {
        "change_scope": "training_loss_only",
        "description": (
            "Sweep gradient-weighted loss settings to reduce shock-region error "
            "without broad score regression."
        ),
        "allowed_actions": [
            "add gradient-weighted loss term",
            "sweep gradient weight",
            "compare shock-region diagnostics against baseline",
        ],
        "forbidden_actions": _COMMON_FORBIDDEN_ACTIONS,
        "candidate_parameters": {
            "gradient_weight": [0.5, 1.0, 2.0],
            "shock_quantile": [0.85, 0.9],
        },
        "budget": _TRAINING_BUDGET,
        "acceptance_criteria": [
            "shock-region error evidence improves",
            "competition_like_score_short does not regress by more than 0.5",
        ],
        "rejection_criteria": [
            "smooth-region accuracy collapses",
            "physics diagnostics identify new oscillation",
            "prediction contains non-finite values",
        ],
    },
    "use_scheduled_sampling_rollout_finetune": {
        "change_scope": "training_schedule_only",
        "description": (
            "Run short scheduled-sampling rollout finetunes to reduce late-step "
            "exposure-bias drift."
        ),
        "allowed_actions": [
            "add scheduled sampling finetune schedule",
            "sweep rollout finetune length",
            "compare early and late rollout error",
        ],
        "forbidden_actions": _COMMON_FORBIDDEN_ACTIONS,
        "candidate_parameters": {
            "teacher_forcing_decay": [0.95, 0.9, 0.85],
            "rollout_steps": [10, 20, 31],
            "training_strategy": [
                "scheduled_sampling",
                "late_weighted_scheduled_sampling",
            ],
        },
        "budget": _TRAINING_BUDGET,
        "acceptance_criteria": [
            "late rollout error improves",
            "early rollout error does not collapse",
            "competition_like_score_short improves by at least 1.0",
        ],
        "rejection_criteria": [
            "early-step error worsens by more than 5%",
            "late error remains the dominant pattern",
            "prediction contains non-finite values",
        ],
    },
    "add_spectral_consistency_loss": {
        "change_scope": "training_loss_only",
        "description": (
            "Add spectral consistency loss to restore suppressed high-frequency "
            "energy while monitoring oscillation risk."
        ),
        "allowed_actions": [
            "add spectral consistency loss",
            "sweep spectral loss weight",
            "compare spectrum alignment against baseline",
        ],
        "forbidden_actions": _COMMON_FORBIDDEN_ACTIONS,
        "candidate_parameters": {
            "spectral_loss_weight": [0.01, 0.05, 0.1],
            "spectral_band": ["mid", "high"],
        },
        "budget": _TRAINING_BUDGET,
        "acceptance_criteria": [
            "spectral energy alignment improves",
            "competition_like_score_short does not regress by more than 0.5",
        ],
        "rejection_criteria": [
            "oscillation diagnostics grow",
            "total_variation_ratio moves farther from 1.0",
            "prediction contains non-finite values",
        ],
    },
    "add_residual_correction_head": {
        "change_scope": "model_head_only",
        "description": (
            "Add a narrow residual correction head to improve PDE residual "
            "alignment without replacing the base rollout."
        ),
        "allowed_actions": [
            "add residual correction head",
            "sweep residual head width",
            "compare residual ratio against baseline",
        ],
        "forbidden_actions": _COMMON_FORBIDDEN_ACTIONS,
        "candidate_parameters": {
            "residual_head_width": [16, 32],
            "residual_loss_weight": [0.01, 0.05],
        },
        "budget": _TRAINING_BUDGET,
        "acceptance_criteria": [
            "residual_ratio improves",
            "competition_like_score_short does not regress by more than 0.5",
        ],
        "rejection_criteria": [
            "score decreases by more than 0.5",
            "residual proxy overfits local windows",
            "prediction contains non-finite values",
        ],
    },
    "keep_baseline_and_collect_stronger_evidence": {
        "change_scope": "data_diagnostic_only",
        "description": (
            "Keep the baseline unchanged and rerun diagnostics on additional "
            "local windows until a stable failure mode is identified."
        ),
        "allowed_actions": [
            "collect additional diagnostic windows",
            "rerun scoring and physics checks",
            "compare dominant failure modes across windows",
        ],
        "forbidden_actions": _COMMON_FORBIDDEN_ACTIONS,
        "candidate_parameters": {
            "sample_limit": [16, 32, 64],
            "start_index": [0, 128, 512],
        },
        "budget": _DIAGNOSTIC_BUDGET,
        "acceptance_criteria": [
            "repeated diagnostics identify a stable failure mode",
            "near-perfect baseline remains stable across windows",
        ],
        "rejection_criteria": [
            "diagnostics remain inconclusive",
            "score variance across windows is too high for a single intervention",
        ],
    },
}
