from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_common import (
    LiteratureAlgorithmLabError,
    guard_output,
    read_json,
    require_file,
    slug,
    write_json,
)


ALGORITHM_CANDIDATES_STATE_HINT = "ALGORITHM_CANDIDATES_COMPILED"
SANDBOX_PLAN_STATE_HINT = "ALGORITHM_CANDIDATE_SANDBOX_PLANNED"
FNO_CANDIDATE_ADAPTED_STATE_HINT = "FNO_ARCHITECTURE_CANDIDATE_ADAPTED"

HOOK_TO_FAMILY = {
    "official_checkpoint_calibration": {
        "candidate_family": "official_checkpoint_family",
        "model_family": "checkpoint",
        "cli_command": "run-checkpoint-family-sandbox",
        "sandbox_level": 1,
    },
    "learned_residual_correction": {
        "candidate_family": "lightweight_residual_corrector",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "spectral_rollout_stabilization": {
        "candidate_family": "spectral_rollout_stabilization",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "physics_residual_regularization": {
        "candidate_family": "physics_residual_regularization",
        "model_family": "FNO",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 1,
    },
    "scheduled_rollout_training": {
        "candidate_family": "scheduled_rollout_training",
        "model_family": "FNO",
        "cli_command": "execute-fno-finetune-candidate",
        "sandbox_level": 1,
    },
    "operator_correction_postprocess": {
        "candidate_family": "operator_correction_postprocess",
        "model_family": "postprocess",
        "cli_command": "run-hypothesis-lab",
        "sandbox_level": 0,
    },
}

HOOK_ALIASES = {
    "physics_informed_loss": "physics_residual_regularization",
    "pinn_residual_loss": "physics_residual_regularization",
    "pi_deeponet_physics_loss": "physics_residual_regularization",
    "scheduled_sampling": "scheduled_rollout_training",
    "rollout_training": "scheduled_rollout_training",
    "autoregressive_rollout_training": "scheduled_rollout_training",
    "postprocess_operator_correction": "operator_correction_postprocess",
    "residual_blend_postprocess": "operator_correction_postprocess",
}


def compile_algorithm_candidates(
    *,
    algorithm_cards_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    cards_report = read_json(
        require_file(Path(algorithm_cards_path), "algorithm cards")
    )
    if cards_report.get("state_hint") != "ALGORITHM_CARDS_EXTRACTED":
        raise LiteratureAlgorithmLabError(
            "algorithm cards report has invalid state_hint"
        )
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    candidates = []
    for card in cards_report.get("algorithm_cards", []):
        if not isinstance(card, dict):
            raise LiteratureAlgorithmLabError("algorithm card must be an object")
        seen_families: set[str] = set()
        for hook in _text_list(card.get("implementation_hooks"), "implementation_hooks"):
            normalized_hook = _normalize_algorithm_hook(hook)
            family_spec = HOOK_TO_FAMILY.get(normalized_hook)
            if family_spec is None:
                candidates.append(_blocked_candidate(card=card, hook=hook))
                continue
            family = str(family_spec["candidate_family"])
            if family in seen_families:
                continue
            seen_families.add(family)
            candidate_id = _algorithm_candidate_id(
                card_id=str(card["algorithm_card_id"]),
                family=family,
            )
            candidate = {
                "algorithm_candidate_id": candidate_id,
                "parent_algorithm_card_id": card["algorithm_card_id"],
                "candidate_family": family_spec["candidate_family"],
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "model_family": family_spec["model_family"],
                "hypothesis": card["claim"],
                "mechanism": card["core_mechanism"],
                "literature_hook": hook,
                "normalized_hook": normalized_hook,
                "executor_spec": _executor_spec(str(family_spec["cli_command"])),
                "parameter_grid": _default_parameter_grid(family),
                "benchmark_plan": {
                    "judge": "local_judge",
                    "primary_metric": "task1_total_score_proxy",
                    "sandbox_level": family_spec["sandbox_level"],
                },
                "ablation_plan": {
                    "baseline": "persistence_or_current_best",
                    "contrast": "candidate_disabled_or_weight_zero",
                },
                "promotion_criteria": {
                    "min_score_delta": 0.01,
                    "max_inference_seconds": 120.0,
                    "requires_ablation": True,
                },
                "risk_controls": [
                    "preserve first 10 context steps",
                    "do not execute external repo code",
                ],
                "source_grounding": {
                    "source_ids": card["source_ids"],
                    "algorithm_card_id": card["algorithm_card_id"],
                },
            }
            mutation_evidence = card.get("mutation_evidence")
            if isinstance(mutation_evidence, dict):
                candidate["mutation_evidence"] = mutation_evidence
            candidates.append(candidate)
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CANDIDATES_STATE_HINT,
        "algorithm_candidates_path": str(output_file),
        "algorithm_candidate_count": len(candidates),
        "algorithm_candidates": candidates,
        "next_command": "plan-algorithm-candidate-sandbox",
    }
    write_json(output_file, report)
    return report


def plan_candidate_sandbox(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = read_json(
        require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") not in {
        ALGORITHM_CANDIDATES_STATE_HINT,
        FNO_CANDIDATE_ADAPTED_STATE_HINT,
    }:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    selected = None
    for candidate in candidates_report.get("algorithm_candidates", []):
        if (
            isinstance(candidate, dict)
            and candidate.get("algorithm_candidate_id") == selected_id
        ):
            selected = candidate
            break
    if selected is None:
        raise LiteratureAlgorithmLabError(
            f"algorithm candidate not found: {selected_id}"
        )
    output_file = Path(output_path)
    guard_output(output_file, force=force)
    executor = selected.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    cli_command = _nonempty_text(executor.get("cli_command"), "cli_command")
    blocked = cli_command == "blocked_requires_executor_design"
    report = {
        "schema_version": 1,
        "state_hint": SANDBOX_PLAN_STATE_HINT,
        "sandbox_plan_path": str(output_file),
        "algorithm_candidates_path": str(Path(algorithm_candidates_path)),
        "algorithm_candidate_id": selected_id,
        "candidate_family": selected.get("candidate_family"),
        "selected_candidate": selected,
        "sandbox_level": _sandbox_level(str(selected.get("candidate_family"))),
        "executor": executor,
        "blocked": blocked,
        "reason_codes": ["requires_new_bounded_executor"] if blocked else [],
        "risk_controls": selected.get("risk_controls", []),
        "next_command": "create-executor-design-request" if blocked else cli_command,
    }
    write_json(output_file, report)
    return report


def _blocked_candidate(*, card: dict[str, Any], hook: str) -> dict[str, Any]:
    candidate_id = _algorithm_candidate_id(
        card_id=str(card["algorithm_card_id"]),
        family="blocked",
        hook=hook,
    )
    return {
        "algorithm_candidate_id": candidate_id,
        "parent_algorithm_card_id": card["algorithm_card_id"],
        "candidate_family": "unsupported",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": "hybrid",
        "hypothesis": card["claim"],
        "mechanism": card["core_mechanism"],
        "executor_spec": {
            "cli_command": "blocked_requires_executor_design",
            "unsupported_hook": hook,
            "required_options": [],
        },
        "parameter_grid": {},
        "benchmark_plan": {
            "judge": "local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 2,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "candidate_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": ["blocked until a bounded executor is designed"],
        "source_grounding": {
            "source_ids": card["source_ids"],
            "algorithm_card_id": card["algorithm_card_id"],
        },
    }


def _executor_spec(cli_command: str) -> dict[str, Any]:
    if cli_command == "run-checkpoint-family-sandbox":
        return {
            "cli_command": cli_command,
            "required_options": [
                "--checkpoint-path",
                "--task-id",
                "--test-path",
                "--target-path",
                "--output-dir",
            ],
        }
    if cli_command == "execute-fno-finetune-candidate":
        return {
            "cli_command": cli_command,
            "required_options": [
                "--candidate-plan-path",
                "--windows-path",
                "--checkpoint-path",
                "--max-train-steps",
            ],
        }
    return {
        "cli_command": cli_command,
        "required_options": [
            "--blueprint-path",
            "--test-path",
            "--target-path",
            "--output-dir",
        ],
    }


def _default_parameter_grid(candidate_family: str) -> dict[str, Any]:
    if candidate_family == "official_checkpoint_family":
        return {
            "calibration_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
            "fno_finetune_steps": [0, 1],
        }
    if candidate_family == "lightweight_residual_corrector":
        return {"residual_shrinkage": [0.25, 0.5, 0.75], "horizon_decay": [40.0, 80.0]}
    if candidate_family == "spectral_rollout_stabilization":
        return {"spectral_strength": [0.25, 0.5, 1.0], "cutoff_ratio": [0.5, 0.75]}
    if candidate_family == "physics_residual_regularization":
        return {"physics_weight": [0.001, 0.01, 0.1], "spectral_weight": [0.0, 0.1]}
    if candidate_family == "scheduled_rollout_training":
        return {"rollout_steps": [1, 2, 4], "teacher_forcing_ratio": [0.5, 0.75]}
    if candidate_family == "operator_correction_postprocess":
        return {
            "correction_strength": [0.25, 0.5, 0.75],
            "spectral_strength": [0.0, 0.5],
        }
    return {"candidate_enabled": [True]}


def _sandbox_level(candidate_family: str) -> int:
    if candidate_family in {
        "lightweight_residual_corrector",
        "spectral_rollout_stabilization",
        "operator_correction_postprocess",
    }:
        return 0
    if candidate_family in {
        "official_checkpoint_family",
        "physics_residual_regularization",
        "scheduled_rollout_training",
    }:
        return 1
    return 2


def _normalize_algorithm_hook(hook: str) -> str:
    normalized = hook.strip()
    return HOOK_ALIASES.get(normalized, normalized)


def _algorithm_candidate_id(
    *,
    card_id: str,
    family: str,
    hook: str | None = None,
) -> str:
    raw_slug = re.sub(r"[^a-zA-Z0-9]+", "_", card_id.lower()).strip("_")
    if len(raw_slug) <= 80:
        return f"candidate_{raw_slug or 'source'}_{family}"
    digest_source = "|".join([card_id, family, hook or ""])
    digest = _sha256_text(digest_source)[:8]
    return f"candidate_{slug(card_id)}_{family}_{digest}"


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise LiteratureAlgorithmLabError(f"{label} must be a non-empty list")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LiteratureAlgorithmLabError(
                f"{label} items must be non-empty strings"
            )
        result.append(item.strip())
    return result


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureAlgorithmLabError(f"{label} must be a non-empty string")
    return value.strip()
