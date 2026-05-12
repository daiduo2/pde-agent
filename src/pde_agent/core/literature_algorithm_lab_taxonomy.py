"""Literature algorithm lab taxonomy functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError
from pde_agent.core.literature_algorithm_lab_helpers import (
    _default_parameter_grid,
    _executor_spec,
    _find_by_id,
    _guard_output,
    _nonempty_text,
    _read_json,
    _require_file,
    _slug,
    _write_json,
)

ALGORITHM_CANDIDATES_STATE_HINT = "ALGORITHM_CANDIDATES_COMPILED"
EXECUTOR_TAXONOMY_STATE_HINT = "EXECUTOR_TAXONOMY_BUILT"
FNO_CANDIDATE_ADAPTED_STATE_HINT = "FNO_ARCHITECTURE_CANDIDATE_ADAPTED"
EXECUTOR_DESIGN_REQUEST_STATE_HINT = "EXECUTOR_DESIGN_REQUEST_CREATED"
BLOCKED_CANDIDATE_HANDOFF_STATE_HINT = "BLOCKED_CANDIDATE_HANDOFF_BUILT"
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
EXECUTOR_DESIGN_HOOKS = {
    "fourier_neural_operator_architecture": {
        "route": "requires_executor_design",
        "requested_executor_id": "bounded_fno_architecture_executor",
        "fallback_route": "official_checkpoint_family",
        "rationale": (
            "A paper-level FNO architecture claim needs bounded local implementation "
            "or an official-checkpoint adapter before sandbox execution."
        ),
    }
}

def build_executor_taxonomy(
    *,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    hook_mappings: dict[str, dict[str, Any]] = {}
    for hook, family_spec in HOOK_TO_FAMILY.items():
        hook_mappings[hook] = {
            "route": "existing_executor",
            "candidate_family": family_spec["candidate_family"],
            "model_family": family_spec["model_family"],
            "cli_command": family_spec["cli_command"],
            "sandbox_level": family_spec["sandbox_level"],
        }
    hook_mappings.update(EXECUTOR_DESIGN_HOOKS)
    report = {
        "schema_version": 1,
        "state_hint": EXECUTOR_TAXONOMY_STATE_HINT,
        "executor_taxonomy_path": str(output_file),
        "hook_mappings": hook_mappings,
        "route_counts": {
            "existing_executor": len(HOOK_TO_FAMILY),
            "requires_executor_design": len(EXECUTOR_DESIGN_HOOKS),
        },
        "next_command": "create-executor-design-request",
    }
    _write_json(output_file, report)
    return report

def adapt_fno_architecture_candidate(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    checkpoint_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") != ALGORITHM_CANDIDATES_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    candidate = _find_by_id(
        candidates_report.get("algorithm_candidates"),
        selected_id,
        "algorithm_candidate_id",
    )
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    if executor.get("unsupported_hook") != "fourier_neural_operator_architecture":
        raise LiteratureAlgorithmLabError(
            "FNO architecture adapter requires a fourier_neural_operator_architecture candidate"
        )
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint resource")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    source_grounding = candidate.get("source_grounding")
    if not isinstance(source_grounding, dict):
        source_grounding = {}
    adapted_source_grounding = {
        **source_grounding,
        "adapter_source_candidate_id": selected_id,
        "adapter_route": "official_checkpoint_family",
    }
    adapted_candidate = {
        "algorithm_candidate_id": f"{selected_id}_official_checkpoint_adapter",
        "parent_algorithm_card_id": candidate.get("parent_algorithm_card_id"),
        "candidate_family": "official_checkpoint_family",
        "target_problem": candidate.get("target_problem", "PDE_Burgers"),
        "target_task": candidate.get("target_task", "task1"),
        "model_family": "checkpoint",
        "hypothesis": candidate.get("hypothesis"),
        "mechanism": (
            f"{candidate.get('mechanism')} Adapted through a local official "
            "checkpoint-family sandbox instead of unbounded architecture code."
        ),
        "executor_spec": _executor_spec("run-checkpoint-family-sandbox"),
        "parameter_grid": _default_parameter_grid("official_checkpoint_family"),
        "resource_bindings": {"checkpoint_paths": [str(checkpoint_file)]},
        "benchmark_plan": {
            "judge": "local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 1,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "checkpoint_adapter_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": [
            "preserve first 10 context steps",
            "do not execute external repo code",
            "verify checkpoint tensor convention before promotion",
        ],
        "source_grounding": adapted_source_grounding,
    }
    report = {
        "schema_version": 1,
        "state_hint": FNO_CANDIDATE_ADAPTED_STATE_HINT,
        "adapted_candidate_path": str(output_file),
        "algorithm_candidates_path": str(output_file),
        "source_algorithm_candidate_id": selected_id,
        "algorithm_candidate_count": 1,
        "adapted_candidate": adapted_candidate,
        "algorithm_candidates": [adapted_candidate],
        "next_command": "plan-algorithm-candidate-sandbox",
    }
    _write_json(output_file, report)
    return report

def create_executor_design_request(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") != ALGORITHM_CANDIDATES_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    candidate = _find_by_id(
        candidates_report.get("algorithm_candidates"),
        _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    if executor.get("cli_command") != "blocked_requires_executor_design":
        raise LiteratureAlgorithmLabError(
            "executor design request requires a blocked candidate"
        )
    hook = _nonempty_text(executor.get("unsupported_hook"), "unsupported_hook")
    hook_spec = EXECUTOR_DESIGN_HOOKS.get(hook, {})
    requested_executor_id = str(
        hook_spec.get("requested_executor_id", f"bounded_{_slug(hook)}_executor")
    )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    strategy_id = f"external_fno_executor_design_{_slug(str(candidate['algorithm_candidate_id']))}"
    hypothesis = _nonempty_text(candidate.get("hypothesis"), "hypothesis")
    mechanism = _nonempty_text(candidate.get("mechanism"), "mechanism")
    source_grounding = candidate.get("source_grounding")
    if not isinstance(source_grounding, dict):
        source_grounding = {}
    report = {
        "schema_version": 1,
        "state_hint": EXECUTOR_DESIGN_REQUEST_STATE_HINT,
        "executor_design_request_path": str(output_file),
        "algorithm_candidate_id": candidate["algorithm_candidate_id"],
        "candidate_family": candidate.get("candidate_family"),
        "unsupported_hook": hook,
        "requested_executor_id": requested_executor_id,
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "source_grounding": source_grounding,
        "reason_codes": [
            "unsupported_algorithm_hook",
            "requires_bounded_executor",
        ],
        "executor_contract": {
            "inputs": [
                "algorithm_candidate_path",
                "validation_windows_path",
                "optional_checkpoint_path",
                "output_dir",
            ],
            "outputs": [
                "candidate_prediction_path",
                "candidate_execution_report_path",
                "sandbox_result_path",
            ],
            "forbidden_actions": [
                "execute external repository code directly",
                "mutate source training data",
                "skip first-10-step identity validation",
            ],
        },
        "minimal_experiment": {
            "level": 2,
            "description": (
                "Implement a bounded executor adapter, run it on local Burgers "
                "validation windows, then compare against persistence and current best."
            ),
        },
        "risk_controls": list(candidate.get("risk_controls", []))
        + [
            "require local sandbox score before promotion",
            "record failure even when executor design is abandoned",
        ],
        "recommended_intermediate_actions": [
            "search official FNO checkpoint resource",
            "map checkpoint resource to run-checkpoint-family-sandbox when available",
            "otherwise design bounded_fno_architecture_executor",
        ],
        "strategy_proposal_fields": {
            "strategy_id": strategy_id,
            "title": "Design bounded FNO executor from literature candidate",
            "rationale": hypothesis,
            "implementation_plan": [
                "Translate the literature FNO mechanism into a bounded local executor contract.",
                "Prefer official checkpoint-family execution if a compatible checkpoint is available.",
                "Run a local Burgers sandbox and require score, runtime, and first-context identity evidence before promotion.",
            ],
            "expected_observables": [
                "higher local judge score than persistence or current best",
                "stable inference under the two-minute task budget",
            ],
            "known_risks": [
                "FNO tensor conventions may not match official checkpoints",
                "architecture reproduction may consume budget without improving long-horizon score",
            ],
            "source_type": "external_retrieval",
            "source_grounding": source_grounding,
        },
        "next_command": "build-blocked-candidate-handoff",
    }
    _write_json(output_file, report)
    return report

def build_blocked_candidate_handoff(
    *,
    executor_design_request_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    request = _read_json(
        _require_file(Path(executor_design_request_path), "executor design request")
    )
    if request.get("state_hint") != EXECUTOR_DESIGN_REQUEST_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "executor design request has invalid state_hint"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    proposal_fields = request.get("strategy_proposal_fields")
    if not isinstance(proposal_fields, dict):
        raise LiteratureAlgorithmLabError(
            "executor design request missing strategy_proposal_fields"
        )
    report = {
        "schema_version": 1,
        "state_hint": BLOCKED_CANDIDATE_HANDOFF_STATE_HINT,
        "blocked_candidate_handoff_path": str(output_file),
        "algorithm_candidate_id": request.get("algorithm_candidate_id"),
        "record_strategy_proposal_fields": proposal_fields,
        "blocked_evidence_summary": {
            "blocked": True,
            "requires_executor": request.get("requested_executor_id"),
            "reason_codes": request.get("reason_codes", []),
            "source_grounding": request.get("source_grounding", {}),
        },
        "next_command": "record-strategy-proposal",
    }
    _write_json(output_file, report)
    return report
