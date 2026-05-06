"""Hermes bridge from literature algorithm candidates into strategy growth."""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.checkpoint_family_sandbox import run_checkpoint_family_sandbox
from pde_agent.core.literature_algorithm_lab import (
    ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT,
    LITERATURE_REGISTRY_EVENT_STATE_HINT,
    adapt_fno_architecture_candidate,
    build_blocked_candidate_handoff,
    build_executor_taxonomy,
    create_bounded_executor_design_spec,
    create_executor_design_request,
    evaluate_algorithm_candidate_promotion,
    materialize_bounded_executor_template,
    plan_candidate_sandbox,
    prepare_bounded_executor_authoring_task,
    record_literature_registry_event,
    update_literature_registry,
)
from pde_agent.core.local_strategy_library import build_local_strategy_library
from pde_agent.core.strategy_growth import record_strategy_proposal


STATE_HINT = "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN"
HANDOFF_STATE_HINT = "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN"
QUEUE_STATE_HINT = "HERMES_EXPANSION_SANDBOX_QUEUE_RAN"
QUEUE_EVALUATION_STATE_HINT = "HERMES_EXPANSION_SANDBOX_QUEUE_EVALUATED"
BLOCKED_EXECUTOR_TASKS_STATE_HINT = (
    "HERMES_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED"
)
FAILURE_MEMORY_STATE_HINT = "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED"
LOCAL_ASSET_CANDIDATES_STATE_HINT = "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT"
REGISTRY_SUMMARY_STATE_HINT = "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT"
REFINEMENT_STATE_HINT = "HERMES_ALGORITHM_CANDIDATE_REFINEMENT_PREPARED"
PROMOTED_STRATEGY_PROPOSAL_STATE_HINT = "STRATEGY_PROPOSAL_RECORDED"
EVIDENCE_ACCUMULATION_PLAN_STATE_HINT = "CANDIDATE_EVIDENCE_ACCUMULATION_PLANNED"
SHORT_SANDBOX_COMMANDS = {"run-hypothesis-lab", "run-blueprint-sandbox"}
CheckpointFamilyRunner = Callable[..., dict[str, Any]]
LocalStrategyRunner = Callable[..., dict[str, Any]]


class HermesLiteratureExpansionError(RuntimeError):
    pass


def run_candidate_expansion_handoff(
    *,
    candidate_plan_path: str | Path,
    workspace_dir: str | Path,
    local_strategy_library_path: str | Path | None = None,
    checkpoint_paths: list[str | Path] | None = None,
    max_strategy_candidates: int | None = None,
    failure_memory_path: str | Path | None = None,
    registry_summary_path: str | Path | None = None,
    force: bool,
) -> dict[str, Any]:
    plan_file = Path(candidate_plan_path)
    plan = _read_json(plan_file)
    if plan.get("state_hint") != "HYPOTHESIS_CANDIDATE_PLAN_BUILT":
        raise HermesLiteratureExpansionError("candidate plan has invalid state_hint")
    if plan.get("expansion_required") is not True:
        raise HermesLiteratureExpansionError("candidate plan does not require expansion")
    handoff = plan.get("expansion_handoff")
    if not isinstance(handoff, dict):
        raise HermesLiteratureExpansionError("candidate plan missing expansion_handoff")
    recommended_inputs = handoff.get("recommended_inputs")
    if not isinstance(recommended_inputs, dict):
        recommended_inputs = {}
    workspace = Path(workspace_dir)
    report_path = workspace / "candidate_expansion_handoff_report.json"
    _guard_output(report_path, force=force)
    workspace.mkdir(parents=True, exist_ok=True)

    library_path = Path(
        local_strategy_library_path
        or recommended_inputs.get("local_strategy_library_path")
        or workspace / "local_strategy_library.json"
    )
    if not library_path.exists():
        build_local_strategy_library(output_path=library_path, force=True)
    requested_checkpoints = [
        Path(path)
        for path in (
            checkpoint_paths
            if checkpoint_paths is not None
            else recommended_inputs.get("checkpoint_paths", [])
        )
        if isinstance(path, str | Path)
    ]
    existing_checkpoints = [path for path in requested_checkpoints if path.exists()]
    missing_checkpoints = [path for path in requested_checkpoints if not path.exists()]
    max_candidates = int(
        max_strategy_candidates
        if max_strategy_candidates is not None
        else recommended_inputs.get("max_strategy_candidates", 5)
    )
    candidates = build_local_asset_algorithm_candidates(
        local_strategy_library_path=library_path,
        checkpoint_paths=existing_checkpoints,
        output_path=workspace / "local_asset_algorithm_candidates.json",
        max_strategy_candidates=max_candidates,
        force=force,
    )
    registry_summary = (
        _read_registry_summary(Path(registry_summary_path))
        if registry_summary_path is not None
        else {}
    )
    learning_constraints = _merge_registry_summary_learning_constraints(
        handoff.get("learning_constraints"),
        registry_summary=registry_summary,
    )
    candidates, skipped_by_constraints = _apply_learning_constraints_to_candidates(
        candidates,
        constraints=learning_constraints,
        registry_summary=registry_summary,
    )
    expansion = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates["algorithm_candidates_path"],
        workspace_dir=workspace / "literature_strategy_expansion",
        checkpoint_paths=existing_checkpoints,
        failure_memory_path=failure_memory_path,
        registry_summary_path=registry_summary_path,
        force=force,
    )
    expansion_exhaustion = _handoff_expansion_exhaustion(
        skipped_by_constraints=skipped_by_constraints,
        registry_summary=registry_summary,
    )
    expansion_exhaustion = _merge_prior_expansion_exhaustion(
        expansion_exhaustion,
        prior_expansion_exhaustion=plan.get("prior_expansion_exhaustion"),
    )
    next_command = (
        expansion_exhaustion["recommended_next_command"]
        if expansion_exhaustion.get("exhausted") is True
        and not _expansion_has_viable_actions(expansion)
        else expansion["next_command"]
    )
    report = {
        "schema_version": 1,
        "state_hint": HANDOFF_STATE_HINT,
        "handoff_report_path": str(report_path),
        "candidate_plan_path": str(plan_file),
        "artifacts": {
            "local_strategy_library_path": str(library_path),
            "algorithm_candidates_path": candidates["algorithm_candidates_path"],
            "expansion_report_path": expansion["expansion_report_path"],
        },
        "metrics": {
            "existing_checkpoint_count": len(existing_checkpoints),
            "missing_checkpoint_count": len(missing_checkpoints),
            "algorithm_candidate_count": candidates["algorithm_candidate_count"],
            "learning_constraint_skipped_count": len(skipped_by_constraints),
            "expansion_action_count": expansion["action_count"],
            "expansion_executable_count": expansion["executable_count"],
            "expansion_proposal_count": expansion["proposal_count"],
        },
        "missing_checkpoint_paths": [str(path) for path in missing_checkpoints],
        "handoff": handoff,
        "learning_constraint_skipped_candidates": skipped_by_constraints,
        "candidate_generation": candidates,
        "expansion": expansion,
        "expansion_exhaustion": expansion_exhaustion,
        "next_command": next_command,
    }
    _write_json(report_path, report)
    return report


def _expansion_has_viable_actions(expansion: dict[str, Any]) -> bool:
    for key in ["action_count", "executable_count", "proposal_count"]:
        value = expansion.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return False


def _apply_learning_constraints_to_candidates(
    candidates: dict[str, Any],
    *,
    constraints: Any,
    registry_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(constraints, dict):
        return candidates, []
    avoid = {
        item
        for item in constraints.get("avoid_candidate_families", [])
        if isinstance(item, str) and item
    }
    avoid_ids = {
        item
        for item in constraints.get("avoid_algorithm_candidate_ids", [])
        if isinstance(item, str) and item
    }
    saturated = {
        item
        for item in constraints.get("saturated_candidate_families", [])
        if isinstance(item, str) and item
    }
    if not avoid and not avoid_ids and not saturated:
        return candidates, []
    original = _candidate_list(candidates)
    kept: list[dict[str, Any]] = []
    generated_mutations: list[dict[str, Any]] = []
    for candidate in original:
        family = candidate.get("candidate_family")
        candidate_id = candidate.get("algorithm_candidate_id")
        if family in saturated:
            continue
        constrained = family in avoid or candidate_id in avoid_ids
        if not constrained or _candidate_satisfies_mutation_constraint(
            candidate,
            constraints=constraints,
        ):
            kept.append(candidate)
            continue
        mutation = _mutate_candidate_for_learning_constraint(
            candidate,
            constraints=constraints,
            registry_summary=registry_summary or {},
        )
        if mutation is not None:
            kept.append(mutation)
            generated_mutations.append(mutation)
    skipped = [
        {
            "algorithm_candidate_id": candidate.get("algorithm_candidate_id"),
            "candidate_family": candidate.get("candidate_family"),
            "reason": _learning_constraint_reason_for_candidate(
                candidate,
                constraints=constraints,
            ),
            "mutation_generated": any(
                mutation.get("mutation_evidence", {}).get(
                    "parent_algorithm_candidate_id"
                )
                == candidate.get("algorithm_candidate_id")
                for mutation in generated_mutations
                if isinstance(mutation.get("mutation_evidence"), dict)
            ),
        }
        for candidate in original
        if (
            candidate.get("candidate_family") in avoid
            or candidate.get("algorithm_candidate_id") in avoid_ids
            or candidate.get("candidate_family") in saturated
        )
        and (
            candidate.get("candidate_family") in saturated
            or not _candidate_satisfies_mutation_constraint(
                candidate,
                constraints=constraints,
            )
        )
    ]
    if not skipped:
        return candidates, []
    updated = dict(candidates)
    updated["algorithm_candidates"] = kept
    updated["algorithm_candidate_count"] = len(kept)
    updated["learning_constraints"] = constraints
    updated["learning_constraint_skipped_candidates"] = skipped
    _write_json(Path(str(updated["algorithm_candidates_path"])), updated)
    return updated, skipped


def _merge_registry_summary_learning_constraints(
    constraints: Any,
    *,
    registry_summary: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(constraints) if isinstance(constraints, dict) else {}
    by_candidate = merged.get("reason_codes_by_algorithm_candidate_id")
    if not isinstance(by_candidate, dict):
        by_candidate = {}
    else:
        by_candidate = {
            str(candidate_id): _string_list(reason_codes)
            for candidate_id, reason_codes in by_candidate.items()
            if isinstance(candidate_id, str)
        }

    changed = False
    for action in _object_items(registry_summary.get("candidate_action_queue")):
        if action.get("recommended_action") != "generate_variant_or_ablation":
            continue
        candidate_id = action.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            continue
        constraint_candidate_id = _registry_variant_constraint_candidate_id(action)
        _append_unique_constraint_value(
            merged,
            "avoid_algorithm_candidate_ids",
            constraint_candidate_id,
        )
        _append_unique_constraint_value(
            merged,
            "mutation_required_for_algorithm_candidate_ids",
            constraint_candidate_id,
        )
        reason_codes = _registry_variant_reason_codes(action)
        if reason_codes:
            existing = by_candidate.get(constraint_candidate_id, [])
            for reason_code in reason_codes:
                _append_unique_value(existing, reason_code)
                _append_unique_constraint_value(merged, "reason_codes", reason_code)
            by_candidate[constraint_candidate_id] = existing
        changed = True

    if by_candidate:
        merged["reason_codes_by_algorithm_candidate_id"] = by_candidate
    if changed:
        return merged
    return merged


def _append_unique_constraint_value(
    constraints: dict[str, Any],
    key: str,
    value: str,
) -> None:
    values = _string_list(constraints.get(key))
    _append_unique_value(values, value)
    constraints[key] = values


def _registry_variant_reason_codes(action: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for reason in _string_list(action.get("failure_codes")):
        _append_unique_value(reasons, reason)
    reason = action.get("reason")
    if isinstance(reason, str) and reason.strip():
        _append_unique_value(reasons, reason)
    variant_constraints = action.get("variant_constraints")
    if isinstance(variant_constraints, dict):
        for reason in _string_list(variant_constraints.get("target_failure_codes")):
            _append_unique_value(reasons, reason)
    return reasons


def _registry_variant_constraint_candidate_id(action: dict[str, Any]) -> str:
    candidate_id = _text(action.get("candidate_id"), "candidate_id")
    if action.get("candidate_family") != "local_strategy_skill":
        return candidate_id
    return _local_strategy_base_candidate_id_from_variant(candidate_id)


def _local_strategy_base_candidate_id_from_variant(candidate_id: str) -> str:
    for axis in _local_strategy_mutation_axes():
        suffix = f"_{axis['id_suffix']}"
        if candidate_id.endswith(suffix):
            return candidate_id[: -len(suffix)]
    return candidate_id


def _learning_constraint_reason(constraints: dict[str, Any]) -> str:
    reason_codes = constraints.get("reason_codes")
    if isinstance(reason_codes, list):
        for reason in reason_codes:
            if isinstance(reason, str) and reason:
                return reason
    return "learning_constraint_avoid_family"


def _learning_constraint_reason_for_candidate(
    candidate: dict[str, Any],
    *,
    constraints: dict[str, Any],
) -> str:
    saturated = {
        item
        for item in constraints.get("saturated_candidate_families", [])
        if isinstance(item, str) and item
    }
    if candidate.get("candidate_family") in saturated:
        return "candidate_family_saturated"
    return _learning_constraint_reason(constraints)


def _candidate_satisfies_mutation_constraint(
    candidate: dict[str, Any],
    *,
    constraints: dict[str, Any],
) -> bool:
    mutation_required = {
        item
        for item in constraints.get("mutation_required_for_families", [])
        if isinstance(item, str) and item
    }
    mutation_required_ids = {
        item
        for item in constraints.get("mutation_required_for_algorithm_candidate_ids", [])
        if isinstance(item, str) and item
    }
    family = candidate.get("candidate_family")
    candidate_id = candidate.get("algorithm_candidate_id")
    if (
        (not isinstance(family, str) or family not in mutation_required)
        and candidate_id not in mutation_required_ids
    ):
        return False
    evidence = candidate.get("mutation_evidence")
    if not isinstance(evidence, dict):
        return False
    mutation_type = evidence.get("mutation_type")
    return isinstance(mutation_type, str) and bool(mutation_type.strip())


def _mutate_candidate_for_learning_constraint(
    candidate: dict[str, Any],
    *,
    constraints: dict[str, Any],
    registry_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    mutation_required = {
        item
        for item in constraints.get("mutation_required_for_families", [])
        if isinstance(item, str) and item
    }
    mutation_required_ids = {
        item
        for item in constraints.get("mutation_required_for_algorithm_candidate_ids", [])
        if isinstance(item, str) and item
    }
    if (
        candidate.get("candidate_family") not in mutation_required
        and candidate.get("algorithm_candidate_id") not in mutation_required_ids
    ):
        return None
    if candidate.get("candidate_family") == "official_checkpoint_family":
        return _mutate_checkpoint_candidate_for_learning_constraint(candidate)
    if (
        candidate.get("candidate_family") == "local_strategy_skill"
        and _constraints_request_local_strategy_variant(constraints)
    ):
        return _mutate_local_strategy_candidate_for_learning_constraint(
            candidate,
            registry_summary=registry_summary or {},
        )
    return None


def _constraints_request_local_strategy_variant(constraints: dict[str, Any]) -> bool:
    reasons = set(_string_list(constraints.get("reason_codes")))
    by_candidate = constraints.get("reason_codes_by_algorithm_candidate_id")
    if isinstance(by_candidate, dict):
        for values in by_candidate.values():
            reasons.update(_string_list(values))
    if "local_strategy_failed_legal_retest_requires_variant" in reasons:
        return True
    return bool(reasons.intersection({"score_regressed", "ablation_missing_or_weak"}))


def _mutate_local_strategy_candidate_for_learning_constraint(
    candidate: dict[str, Any],
    *,
    registry_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    parent_id = _text(candidate.get("algorithm_candidate_id"), "algorithm_candidate_id")
    previous_mutation_types = _registry_avoided_local_strategy_mutation_types(
        registry_summary or {},
        parent_id=parent_id,
    )
    axis = _select_local_strategy_mutation_axis(previous_mutation_types)
    if axis is None:
        return None
    mutated = dict(candidate)
    mutated["algorithm_candidate_id"] = f"{parent_id}_{axis['id_suffix']}"
    mutated["hypothesis"] = axis["hypothesis"]
    mutated["mechanism"] = axis["mechanism"]
    executor = (
        dict(candidate.get("executor_spec"))
        if isinstance(candidate.get("executor_spec"), dict)
        else {}
    )
    hooks = _string_list(executor.get("implementation_hooks"))
    hook = str(axis["implementation_hook"])
    if hook not in hooks:
        hooks.append(hook)
    executor["implementation_hooks"] = hooks
    mutated["executor_spec"] = executor
    parameter_grid = (
        dict(candidate.get("parameter_grid"))
        if isinstance(candidate.get("parameter_grid"), dict)
        else {}
    )
    for key, value in axis["parameter_grid"].items():
        parameter_grid[key] = value
    mutated["parameter_grid"] = parameter_grid
    mutation_evidence = {
        "parent_candidate_family": "local_strategy_skill",
        "parent_algorithm_candidate_id": parent_id,
        "mutation_type": axis["mutation_type"],
        "reason": "local_strategy_failed_legal_retest_requires_variant",
    }
    if previous_mutation_types:
        mutation_evidence["previous_mutation_types"] = previous_mutation_types
    mutated["mutation_evidence"] = mutation_evidence
    risks = _string_list(candidate.get("risk_controls"))
    risks.extend(axis["risk_controls"])
    mutated["risk_controls"] = risks
    return mutated


def _select_local_strategy_mutation_axis(
    previous_mutation_types: list[str],
) -> dict[str, Any] | None:
    previous = set(previous_mutation_types)
    for axis in _local_strategy_mutation_axes():
        mutation_type = str(axis["mutation_type"])
        if mutation_type not in previous:
            return axis
    return None


def _registry_avoided_local_strategy_mutation_types(
    registry_summary: dict[str, Any],
    *,
    parent_id: str,
) -> list[str]:
    mutation_types: list[str] = []
    axes = _local_strategy_mutation_axes()
    for item in _object_items(registry_summary.get("avoid_repetition")):
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.startswith(
            f"{parent_id}_"
        ):
            continue
        explicit = item.get("mutation_type")
        if isinstance(explicit, str) and explicit:
            _append_unique_value(mutation_types, explicit)
            continue
        for index, axis in enumerate(axes):
            if candidate_id.endswith(str(axis["id_suffix"])):
                for implied_axis in axes[: index + 1]:
                    _append_unique_value(
                        mutation_types,
                        str(implied_axis["mutation_type"]),
                    )
                break
    return mutation_types


def _append_unique_value(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _handoff_expansion_exhaustion(
    *,
    skipped_by_constraints: list[dict[str, Any]],
    registry_summary: dict[str, Any],
) -> dict[str, Any]:
    exhausted_families: list[str] = []
    exhausted_candidate_ids: list[str] = []
    exhausted_node_ids: list[str] = []
    attempted_mutation_types: list[str] = []
    all_local_axes = [str(axis["mutation_type"]) for axis in _local_strategy_mutation_axes()]
    for skipped in skipped_by_constraints:
        if skipped.get("candidate_family") != "local_strategy_skill":
            continue
        if skipped.get("mutation_generated") is not False:
            continue
        parent_id = skipped.get("algorithm_candidate_id")
        if not isinstance(parent_id, str) or not parent_id:
            continue
        previous = _registry_avoided_local_strategy_mutation_types(
            registry_summary,
            parent_id=parent_id,
        )
        if set(all_local_axes).issubset(previous):
            _append_unique_value(exhausted_candidate_ids, parent_id)
            _append_unique_value(
                exhausted_node_ids,
                _local_strategy_node_id_from_candidate_id(parent_id),
            )
            for mutation_type in all_local_axes:
                _append_unique_value(attempted_mutation_types, mutation_type)
    if not exhausted_candidate_ids and not exhausted_families:
        return {"exhausted": False}
    return {
        "exhausted": True,
        "reason": "local_strategy_mutation_axes_exhausted",
        "exhausted_candidate_families": exhausted_families,
        "exhausted_algorithm_candidate_ids": exhausted_candidate_ids,
        "exhausted_algorithm_node_ids": exhausted_node_ids,
        "attempted_mutation_types": attempted_mutation_types,
        "recommended_next_command": "plan-hypothesis-candidates",
        "recommended_learning_objective": (
            "switch_candidate_family_after_axis_exhaustion"
        ),
    }


def _merge_prior_expansion_exhaustion(
    expansion_exhaustion: dict[str, Any],
    *,
    prior_expansion_exhaustion: Any,
) -> dict[str, Any]:
    if (
        not isinstance(prior_expansion_exhaustion, dict)
        or prior_expansion_exhaustion.get("exhausted") is not True
    ):
        return expansion_exhaustion
    if expansion_exhaustion.get("exhausted") is not True:
        return dict(prior_expansion_exhaustion)
    merged = dict(expansion_exhaustion)
    for key in [
        "exhausted_candidate_families",
        "exhausted_algorithm_candidate_ids",
        "exhausted_algorithm_node_ids",
        "attempted_mutation_types",
    ]:
        values = _string_list(prior_expansion_exhaustion.get(key))
        for value in _string_list(expansion_exhaustion.get(key)):
            _append_unique_value(values, value)
        if values or key in merged:
            merged[key] = values
    for key in ["reason", "recommended_next_command", "recommended_learning_objective"]:
        if key not in merged and isinstance(prior_expansion_exhaustion.get(key), str):
            merged[key] = prior_expansion_exhaustion[key]
    return merged


def _local_strategy_node_id_from_candidate_id(candidate_id: str) -> str:
    if candidate_id.startswith("candidate_local_strategy_"):
        return "local_strategy_" + candidate_id.removeprefix(
            "candidate_local_strategy_"
        )
    return candidate_id


def _local_strategy_mutation_axes() -> list[dict[str, Any]]:
    return [
        {
            "id_suffix": "mutation_robust_reference_ablation_v0",
            "mutation_type": "robust_reference_ablation_retry",
            "implementation_hook": "robust_reference_ablation",
            "hypothesis": (
                "The parent local strategy improved a narrow legal-reference "
                "window but failed on a wider window; a robust "
                "reference-ablation retry should prefer smoother handoffs and "
                "require wider-window evidence before promotion."
            ),
            "mechanism": (
                "Reuse the bounded local strategy executor, but mark this "
                "candidate as a mutation that must search a wider legal "
                "reference window and report whether each early sub-window "
                "source remains stable under ablation."
            ),
            "parameter_grid": {
                "reference_sample_limits": [128],
                "requires_wider_reference_ablation": [True],
            },
            "risk_controls": [
                "mutation must not repeat the narrow-window-only promotion path"
            ],
        },
        {
            "id_suffix": "mutation_source_stability_ablation_v0",
            "mutation_type": "source_stability_ablation_retry",
            "implementation_hook": "source_stability_ablation",
            "hypothesis": (
                "The robust reference-ablation retry did not improve the legal "
                "window; the next test should isolate whether early-window "
                "source switching is unstable across reference samples."
            ),
            "mechanism": (
                "Reuse the bounded local strategy executor, but require source "
                "stability diagnostics for each early subsegment before any "
                "handoff patch is accepted."
            ),
            "parameter_grid": {
                "reference_sample_limits": [128],
                "source_stability_guard": [True],
            },
            "risk_controls": [
                "mutation must report per-source stability before promotion"
            ],
        },
        {
            "id_suffix": "mutation_handoff_smoothing_ablation_v0",
            "mutation_type": "handoff_smoothing_ablation_retry",
            "implementation_hook": "handoff_smoothing_ablation",
            "hypothesis": (
                "If source stability is not sufficient, the remaining failure "
                "mode may be a discontinuous early-to-middle handoff; a "
                "smoothing ablation should test conservative blend weights."
            ),
            "mechanism": (
                "Reuse the bounded local strategy executor with explicit "
                "handoff blend weights so that the sandbox can measure whether "
                "the early segment improves without degrading the middle "
                "segment."
            ),
            "parameter_grid": {
                "reference_sample_limits": [128],
                "handoff_smoothing_weights": [0.25, 0.5],
            },
            "risk_controls": [
                "mutation must not trade early score for middle-segment regression"
            ],
        },
    ]


def _mutate_checkpoint_candidate_for_learning_constraint(
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    resources = (
        candidate.get("resource_bindings")
        if isinstance(candidate.get("resource_bindings"), dict)
        else {}
    )
    checkpoint_paths = _string_list(resources.get("checkpoint_paths"))
    if not checkpoint_paths:
        return None
    checkpoint_path = Path(checkpoint_paths[0])
    if _checkpoint_family(checkpoint_path) != "FNO":
        return None
    parent_id = _text(candidate.get("algorithm_candidate_id"), "algorithm_candidate_id")
    mutated = dict(candidate)
    mutated["algorithm_candidate_id"] = f"{parent_id}_mutation_finetune_calibration_v0"
    mutated["hypothesis"] = (
        "A direct official FNO checkpoint underperformed the current Hermes best; "
        "a bounded one-step finetune plus narrower calibration retry may rescue "
        "checkpoint dynamics without repeating the failed direct checkpoint route."
    )
    mutated["mechanism"] = (
        "Reuse the audited checkpoint-family sandbox, but require FNO finetuning "
        "on local supervised windows and sweep only conservative calibration "
        "weights before promotion."
    )
    mutated["parameter_grid"] = {
        "calibration_weights": [0.1, 0.25, 0.5],
        "fno_finetune_steps": [1],
    }
    mutated["mutation_evidence"] = {
        "parent_candidate_family": "official_checkpoint_family",
        "parent_algorithm_candidate_id": parent_id,
        "mutation_type": "checkpoint_finetune_calibration_retry",
        "reason": "responds_to_deprioritized_checkpoint_family",
    }
    risks = [
        item
        for item in candidate.get("risk_controls", [])
        if isinstance(item, str) and item
    ]
    risks.append("finetune retry must beat current best, not only direct checkpoint")
    mutated["risk_controls"] = risks
    return mutated


def run_hermes_literature_strategy_expansion(
    *,
    algorithm_candidates_path: str | Path,
    workspace_dir: str | Path,
    checkpoint_paths: list[str | Path] | None = None,
    failure_memory_path: str | Path | None = None,
    registry_summary_path: str | Path | None = None,
    force: bool,
) -> dict[str, Any]:
    candidates_file = Path(algorithm_candidates_path)
    candidates_report = _read_json(candidates_file)
    if candidates_report.get("state_hint") not in {
        "ALGORITHM_CANDIDATES_COMPILED",
        "FNO_ARCHITECTURE_CANDIDATE_ADAPTED",
    }:
        raise HermesLiteratureExpansionError(
            "algorithm candidates artifact has invalid state_hint"
        )
    workspace = Path(workspace_dir)
    report_path = workspace / "hermes_literature_strategy_expansion.json"
    _guard_output(report_path, force=force)
    workspace.mkdir(parents=True, exist_ok=True)
    build_executor_taxonomy(
        output_path=workspace / "executor_taxonomy.json",
        force=force,
    )

    checkpoints = [Path(path) for path in checkpoint_paths or []]
    failure_memory = (
        _read_failure_memory(Path(failure_memory_path))
        if failure_memory_path is not None
        else {}
    )
    registry_summary = (
        _read_registry_summary(Path(registry_summary_path))
        if registry_summary_path is not None
        else {}
    )
    candidate_rankings = _rank_candidates(
        _candidate_list(candidates_report),
        checkpoint_paths=checkpoints,
        failure_memory=failure_memory,
        registry_summary=registry_summary,
    )
    skipped_candidates = [
        ranking
        for ranking in candidate_rankings
        if ranking.get("skip_reason")
    ]
    actions: list[dict[str, Any]] = []
    candidates_by_id = {
        candidate["algorithm_candidate_id"]: candidate
        for candidate in _candidate_list(candidates_report)
    }
    for ranking in candidate_rankings:
        if ranking.get("skip_reason"):
            continue
        candidate = candidates_by_id[str(ranking["algorithm_candidate_id"])]
        action = _expand_candidate(
            candidate=candidate,
            candidates_file=candidates_file,
            workspace=workspace,
            checkpoint_paths=checkpoints,
            force=force,
        )
        actions.append(action)

    executable_count = sum(1 for action in actions if action.get("executable") is True)
    proposal_count = sum(
        1 for action in actions if action.get("strategy_proposal_path")
    )
    sandbox_queue = [
        str(action["sandbox_plan_path"])
        for action in actions
        if action.get("sandbox_plan_path")
    ]
    strategy_growth_queue = [
        str(action["strategy_proposal_path"])
        for action in actions
        if action.get("strategy_proposal_path")
    ]
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "expansion_report_path": str(report_path),
        "algorithm_candidates_path": str(candidates_file),
        "workspace_dir": str(workspace),
        "action_count": len(actions),
        "executable_count": executable_count,
        "proposal_count": proposal_count,
        "skipped_count": len(skipped_candidates),
        "candidate_rankings": candidate_rankings,
        "skipped_candidates": skipped_candidates,
        "sandbox_queue": sandbox_queue,
        "strategy_growth_queue": strategy_growth_queue,
        "selection_policy": {
            "ordered_by": [
                "skip failed-memory candidates",
                "skip registry avoid-repetition candidates",
                "prefer checkpoint adapters when a resource is available",
                "then existing bounded executors",
                "then executor-design proposals",
            ],
            "failure_memory_path": str(failure_memory_path)
            if failure_memory_path is not None
            else None,
            "registry_summary_path": str(registry_summary_path)
            if registry_summary_path is not None
            else None,
        },
        "actions": actions,
        "next_command": _next_command(
            actions,
            skipped_count=len(skipped_candidates),
        ),
    }
    recovery_plan = _recovery_plan(actions=actions, skipped_candidates=skipped_candidates)
    if recovery_plan is not None:
        report["recovery_plan"] = recovery_plan
    _write_json(report_path, report)
    return report


def run_expansion_sandbox_queue(
    *,
    expansion_report_path: str | Path,
    context_path: str | Path,
    target_path: str | Path,
    initial_submission_dir: str | Path | None = None,
    output_dir: str | Path,
    submission_id: str,
    max_items: int | None = None,
    failure_memory_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    reference_sample_limit: int | None = None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    selected_algorithm_candidate_id: str | None = None,
    checkpoint_family_runner: CheckpointFamilyRunner | None = None,
    local_strategy_runner: LocalStrategyRunner | None = None,
    force: bool,
) -> dict[str, Any]:
    from pde_agent.core.hypothesis_lab import run_hypothesis_lab
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    expansion_file = _require_file(Path(expansion_report_path), "expansion report")
    expansion = _read_json(expansion_file)
    if expansion.get("state_hint") != STATE_HINT:
        raise HermesLiteratureExpansionError("expansion report has invalid state_hint")
    sandbox_queue = _string_list(expansion.get("sandbox_queue"))
    filtered_queue = _filter_sandbox_queue_by_candidate_id(
        sandbox_queue=sandbox_queue,
        selected_algorithm_candidate_id=selected_algorithm_candidate_id,
    )
    limit = len(filtered_queue) if max_items is None else max_items
    if limit < 0:
        raise HermesLiteratureExpansionError("max_items must be >= 0")
    root = Path(output_dir)
    report_path = root / "expansion_sandbox_queue_report.json"
    _guard_output(report_path, force=force)
    root.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for index, plan_ref in enumerate(filtered_queue[:limit]):
        plan_file = _require_file(Path(plan_ref), "sandbox plan")
        plan = _read_json(plan_file)
        item_dir = root / f"{index:03d}_{_slug(str(plan.get('algorithm_candidate_id', 'candidate')))}"
        try:
            item = _run_expansion_sandbox_item(
                plan=plan,
                plan_file=plan_file,
                item_dir=item_dir,
                context_path=Path(context_path),
                target_path=Path(target_path),
                initial_submission_dir=Path(initial_submission_dir)
                if initial_submission_dir is not None
                else None,
                submission_id=submission_id,
                failure_memory_path=Path(failure_memory_path)
                if failure_memory_path is not None
                else None,
                reference_path=Path(reference_path) if reference_path is not None else None,
                reference_sample_limit=reference_sample_limit,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                run_hypothesis_lab=run_hypothesis_lab,
                checkpoint_family_runner=checkpoint_family_runner
                or run_checkpoint_family_sandbox,
                local_strategy_runner=local_strategy_runner
                or run_hermes_batch_optimization_loop,
                force=force,
            )
        except Exception as exc:
            item = _error_expansion_sandbox_item(
                plan=plan,
                plan_file=plan_file,
                failure_memory_path=Path(failure_memory_path)
                if failure_memory_path is not None
                else None,
                error=exc,
            )
        items.append(item)

    executed_count = sum(1 for item in items if item.get("status") == "executed")
    skipped_count = sum(1 for item in items if item.get("status") == "skipped")
    error_count = sum(1 for item in items if item.get("status") == "error")
    failed_count = sum(1 for item in items if item.get("outcome") == "failed")
    passed_count = sum(1 for item in items if item.get("outcome") == "passed")
    report = {
        "schema_version": 1,
        "state_hint": QUEUE_STATE_HINT,
        "queue_report_path": str(report_path),
        "expansion_report_path": str(expansion_file),
        "failure_memory_path": str(failure_memory_path)
        if failure_memory_path is not None
        else None,
        "selected_algorithm_candidate_id": selected_algorithm_candidate_id,
        "metrics": {
            "queued_count": len(sandbox_queue),
            "candidate_filter_applied": selected_algorithm_candidate_id is not None,
            "candidate_filter_match_count": len(filtered_queue),
            "attempted_count": len(items),
            "executed_count": executed_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "passed_count": passed_count,
            "failed_count": failed_count,
        },
        "items": items,
        "next_command": _sandbox_queue_next_command(items),
    }
    _write_json(report_path, report)
    return report


def _filter_sandbox_queue_by_candidate_id(
    *,
    sandbox_queue: list[str],
    selected_algorithm_candidate_id: str | None,
) -> list[str]:
    if selected_algorithm_candidate_id is None:
        return sandbox_queue
    candidate_id = _text(
        selected_algorithm_candidate_id,
        "selected_algorithm_candidate_id",
    )
    matches: list[str] = []
    for plan_ref in sandbox_queue:
        plan = _read_json(_require_file(Path(plan_ref), "sandbox plan"))
        if plan.get("algorithm_candidate_id") == candidate_id:
            matches.append(plan_ref)
    if not matches:
        raise HermesLiteratureExpansionError(
            f"selected algorithm candidate not found in sandbox queue: {candidate_id}"
        )
    return matches


def _run_expansion_sandbox_item(
    *,
    plan: dict[str, Any],
    plan_file: Path,
    item_dir: Path,
    context_path: Path,
    target_path: Path,
    initial_submission_dir: Path | None,
    submission_id: str,
    failure_memory_path: Path | None,
    reference_path: Path | None,
    reference_sample_limit: int | None,
    timing_profile: str | None,
    timing_factor: float | None,
    run_hypothesis_lab: Any,
    checkpoint_family_runner: CheckpointFamilyRunner,
    local_strategy_runner: LocalStrategyRunner,
    force: bool,
) -> dict[str, Any]:
    if plan.get("state_hint") != "ALGORITHM_CANDIDATE_SANDBOX_PLANNED":
        raise HermesLiteratureExpansionError("sandbox plan has invalid state_hint")
    candidate_id = _text(plan.get("algorithm_candidate_id"), "algorithm_candidate_id")
    candidate = plan.get("selected_candidate")
    if not isinstance(candidate, dict):
        return _skip_expansion_sandbox_item(
            plan=plan,
            plan_file=plan_file,
            failure_memory_path=failure_memory_path,
            reason_code="sandbox_plan_missing_selected_candidate",
            outcome="blocked",
            force=force,
        )
    executor = plan.get("executor")
    if not isinstance(executor, dict):
        raise HermesLiteratureExpansionError("sandbox plan executor must be an object")
    cli_command = _text(executor.get("cli_command"), "cli_command")
    if bool(plan.get("blocked")):
        return _skip_expansion_sandbox_item(
            plan=plan,
            plan_file=plan_file,
            failure_memory_path=failure_memory_path,
            reason_code="sandbox_plan_blocked",
            outcome="blocked",
            force=force,
        )
    if cli_command == "run-checkpoint-family-sandbox":
        return _run_checkpoint_family_sandbox_item(
            plan=plan,
            plan_file=plan_file,
            item_dir=item_dir,
            candidate=candidate,
            context_path=context_path,
            target_path=target_path,
            submission_id=submission_id,
            failure_memory_path=failure_memory_path,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            checkpoint_family_runner=checkpoint_family_runner,
            force=force,
        )
    if _is_bounded_local_strategy_candidate(
        candidate=candidate,
        cli_command=cli_command,
    ):
        if initial_submission_dir is None:
            return _skip_expansion_sandbox_item(
                plan=plan,
                plan_file=plan_file,
                failure_memory_path=failure_memory_path,
                reason_code="local_strategy_missing_initial_submission_dir",
                outcome="blocked",
                force=force,
            )
        return _run_bounded_local_strategy_sandbox_item(
            plan=plan,
            plan_file=plan_file,
            item_dir=item_dir,
            candidate=candidate,
            context_path=context_path,
            target_path=target_path,
            initial_submission_dir=initial_submission_dir,
            submission_id=submission_id,
            failure_memory_path=failure_memory_path,
            reference_path=reference_path,
            reference_sample_limit=reference_sample_limit,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            local_strategy_runner=local_strategy_runner,
            force=force,
        )
    if cli_command not in SHORT_SANDBOX_COMMANDS:
        return _skip_expansion_sandbox_item(
            plan=plan,
            plan_file=plan_file,
            failure_memory_path=failure_memory_path,
            reason_code=f"unsupported_for_auto_short_sandbox:{cli_command}",
            outcome="blocked",
            force=force,
        )
    item_dir.mkdir(parents=True, exist_ok=True)
    selected_candidates_path = item_dir / "selected_algorithm_candidates.json"
    _guard_output(selected_candidates_path, force=force)
    _write_json(
        selected_candidates_path,
        {
            "schema_version": 1,
            "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
            "algorithm_candidates_path": str(selected_candidates_path),
            "algorithm_candidate_count": 1,
            "algorithm_candidates": [candidate],
            "next_command": "run-hypothesis-lab",
        },
    )
    lab = run_hypothesis_lab(
        manual_report_paths=[],
        adaptive_report_paths=[],
        log_paths=[],
        checkpoint_paths=[],
        literature_paths=[],
        algorithm_candidate_paths=[selected_candidates_path],
        failure_memory_paths=[],
        failure_memory_output_path=failure_memory_path,
        context_path=context_path,
        target_path=target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        output_dir=item_dir / "hypothesis_lab",
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    sandbox = lab.get("sandbox") if isinstance(lab.get("sandbox"), dict) else {}
    verifier = (
        sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
    )
    decision = str(verifier.get("decision", "unknown"))
    outcome = "passed" if decision == "promote_to_strategy_proposal" else "failed"
    return {
        "status": "executed",
        "outcome": outcome,
        "algorithm_candidate_id": candidate_id,
        "candidate_family": plan.get("candidate_family"),
        "sandbox_plan_path": str(plan_file),
        "selected_algorithm_candidates_path": str(selected_candidates_path),
        "sandbox_report_path": lab["artifacts"]["hypothesis_lab_report_path"],
        "decision": decision,
        "best_score": verifier.get("best_segment_score"),
        "score_delta_vs_baseline": verifier.get("score_delta_vs_baseline"),
        "next_command": "evaluate-expansion-sandbox-queue",
    }


def _run_checkpoint_family_sandbox_item(
    *,
    plan: dict[str, Any],
    plan_file: Path,
    item_dir: Path,
    candidate: dict[str, Any],
    context_path: Path,
    target_path: Path,
    submission_id: str,
    failure_memory_path: Path | None,
    timing_profile: str | None,
    timing_factor: float | None,
    checkpoint_family_runner: CheckpointFamilyRunner,
    force: bool,
) -> dict[str, Any]:
    item_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = _text(plan.get("algorithm_candidate_id"), "algorithm_candidate_id")
    selected_candidates_path = item_dir / "selected_algorithm_candidates.json"
    _guard_output(selected_candidates_path, force=force)
    _write_json(
        selected_candidates_path,
        {
            "schema_version": 1,
            "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
            "algorithm_candidates_path": str(selected_candidates_path),
            "algorithm_candidate_count": 1,
            "algorithm_candidates": [candidate],
            "next_command": "run-checkpoint-family-sandbox",
        },
    )
    checkpoint_paths = _checkpoint_paths_from_candidate(candidate)
    if not checkpoint_paths:
        return _skip_expansion_sandbox_item(
            plan=plan,
            plan_file=plan_file,
            failure_memory_path=failure_memory_path,
            reason_code="checkpoint_family_missing_checkpoint_paths",
            outcome="blocked",
            force=force,
        )
    checkpoint_report = checkpoint_family_runner(
        checkpoint_paths=checkpoint_paths,
        task_id=_task_id_from_candidate(candidate),
        test_path=context_path,
        target_path=target_path,
        output_dir=item_dir / "checkpoint_family_sandbox",
        submission_id=submission_id,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        calibration_weights=_calibration_weights_from_candidate(candidate),
        fno_finetune_steps=_fno_finetune_steps_from_candidate(candidate),
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    report_path = _checkpoint_family_report_path(checkpoint_report)
    best = (
        checkpoint_report.get("best_candidate")
        if isinstance(checkpoint_report.get("best_candidate"), dict)
        else {}
    )
    outcome = "passed" if best.get("rules_passed") is True else "failed"
    if failure_memory_path is not None:
        record_hermes_literature_candidate_outcome(
            algorithm_candidate_id=candidate_id,
            candidate_key=_plan_candidate_key(plan),
            outcome=outcome,
            reason_codes=["checkpoint_family_sandbox_executed"],
            output_path=failure_memory_path,
            force=False,
        )
    return {
        "status": "executed",
        "outcome": outcome,
        "algorithm_candidate_id": candidate_id,
        "candidate_family": plan.get("candidate_family"),
        "sandbox_plan_path": str(plan_file),
        "selected_algorithm_candidates_path": str(selected_candidates_path),
        "checkpoint_family_report_path": str(report_path),
        "decision": "checkpoint_family_sandbox",
        "best_score": best.get("score"),
        "score_delta_vs_baseline": None,
        "next_command": "evaluate-expansion-sandbox-queue",
    }


def _run_bounded_local_strategy_sandbox_item(
    *,
    plan: dict[str, Any],
    plan_file: Path,
    item_dir: Path,
    candidate: dict[str, Any],
    context_path: Path,
    target_path: Path,
    initial_submission_dir: Path,
    submission_id: str,
    failure_memory_path: Path | None,
    reference_path: Path | None,
    reference_sample_limit: int | None,
    timing_profile: str | None,
    timing_factor: float | None,
    local_strategy_runner: LocalStrategyRunner,
    force: bool,
) -> dict[str, Any]:
    item_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = _text(plan.get("algorithm_candidate_id"), "algorithm_candidate_id")
    selected_candidates_path = item_dir / "selected_algorithm_candidates.json"
    _guard_output(selected_candidates_path, force=force)
    _write_json(
        selected_candidates_path,
        {
            "schema_version": 1,
            "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
            "algorithm_candidates_path": str(selected_candidates_path),
            "algorithm_candidate_count": 1,
            "algorithm_candidates": [candidate],
            "next_command": "run-hermes-batch-optimization-loop",
        },
    )
    generator = _local_strategy_candidate_generator(candidate)
    loop_report = local_strategy_runner(
        output_dir=item_dir / "local_strategy_sandbox",
        initial_submission_dir=initial_submission_dir,
        max_rounds=_bounded_local_strategy_max_rounds(generator),
        patience=_bounded_local_strategy_max_rounds(generator),
        min_score_improvement=0.0,
        submit_score_threshold=999999.0,
        learning_objective=_bounded_local_strategy_objective(generator),
        task1_test_path=context_path,
        task1_target_path=target_path,
        task2_test_path=None,
        task2_target_path=None,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        candidate_submission_dirs=[],
        candidate_generator=generator,
        hypothesis_blueprint_path=None,
        hypothesis_reference_path=reference_path or target_path,
        hypothesis_reference_sample_limit=_bounded_local_strategy_reference_sample_limit(
            candidate=candidate,
            candidate_generator=generator,
            requested_limit=reference_sample_limit,
        ),
        hypothesis_reference_sample_limits=[],
        hypothesis_candidate_options=_local_strategy_candidate_options(candidate),
        postprocess_search_mode="paired",
        postprocess_strategies=[],
        postprocess_blend_weights=[],
        postprocess_adaptive_step=0.125,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        deadline_seconds=_bounded_local_strategy_deadline_seconds(generator),
        checkpoint_every=0,
        memory_json_path=None,
        avoid_repeating=False,
        force=force,
    )
    report_path = _local_strategy_loop_report_path(loop_report)
    best = (
        loop_report.get("best_candidate")
        if isinstance(loop_report.get("best_candidate"), dict)
        else {}
    )
    outcome = "passed" if best.get("rules_passed") is True else "failed"
    if failure_memory_path is not None:
        record_hermes_literature_candidate_outcome(
            algorithm_candidate_id=candidate_id,
            candidate_key=_plan_candidate_key(plan),
            outcome=outcome,
            reason_codes=["bounded_local_strategy_sandbox_executed", generator],
            output_path=failure_memory_path,
            force=False,
        )
    return {
        "status": "executed",
        "outcome": outcome,
        "algorithm_candidate_id": candidate_id,
        "candidate_family": plan.get("candidate_family"),
        "sandbox_plan_path": str(plan_file),
        "selected_algorithm_candidates_path": str(selected_candidates_path),
        "local_strategy_loop_report_path": str(report_path),
        "decision": "bounded_local_strategy_sandbox",
        "candidate_generator": generator,
        "best_score": best.get("score"),
        "score_delta_vs_baseline": None,
        "next_command": "evaluate-expansion-sandbox-queue",
    }


def _skip_expansion_sandbox_item(
    *,
    plan: dict[str, Any],
    plan_file: Path,
    failure_memory_path: Path | None,
    reason_code: str,
    outcome: str,
    force: bool,
) -> dict[str, Any]:
    candidate_id = _text(plan.get("algorithm_candidate_id"), "algorithm_candidate_id")
    candidate = plan.get("selected_candidate")
    candidate_key = _plan_candidate_key(plan)
    if failure_memory_path is not None:
        record_hermes_literature_candidate_outcome(
            algorithm_candidate_id=candidate_id,
            candidate_key=candidate_key,
            outcome=outcome,
            reason_codes=[reason_code],
            output_path=failure_memory_path,
            force=False,
        )
    return {
        "status": "skipped",
        "outcome": outcome,
        "algorithm_candidate_id": candidate_id,
        "candidate_family": plan.get("candidate_family"),
        "candidate_key": candidate_key,
        "sandbox_plan_path": str(plan_file),
        "reason_codes": [reason_code],
        "selected_candidate_present": isinstance(candidate, dict),
        "next_command": "review-expansion-sandbox-queue",
    }


def _error_expansion_sandbox_item(
    *,
    plan: dict[str, Any],
    plan_file: Path,
    failure_memory_path: Path | None,
    error: Exception,
) -> dict[str, Any]:
    candidate_id = _text(plan.get("algorithm_candidate_id"), "algorithm_candidate_id")
    candidate_key = _plan_candidate_key(plan)
    reason_codes = [
        "sandbox_item_exception",
        f"{error.__class__.__name__}",
    ]
    if failure_memory_path is not None:
        record_hermes_literature_candidate_outcome(
            algorithm_candidate_id=candidate_id,
            candidate_key=candidate_key,
            outcome="failed",
            reason_codes=reason_codes,
            output_path=failure_memory_path,
            force=False,
        )
    return {
        "status": "error",
        "outcome": "failed",
        "algorithm_candidate_id": candidate_id,
        "candidate_family": plan.get("candidate_family"),
        "candidate_key": candidate_key,
        "sandbox_plan_path": str(plan_file),
        "reason_codes": reason_codes,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "next_command": "review-expansion-sandbox-queue",
    }


def _plan_candidate_key(plan: dict[str, Any]) -> str:
    candidate = plan.get("selected_candidate")
    if not isinstance(candidate, dict):
        return f"unknown_card|{plan.get('candidate_family') or 'unknown_family'}"
    return _candidate_key(candidate)


def _sandbox_queue_next_command(items: list[dict[str, Any]]) -> str:
    if any(item.get("outcome") == "passed" for item in items):
        return "evaluate-expansion-sandbox-queue"
    if any(item.get("outcome") == "failed" for item in items):
        return "summarize-literature-registry-for-hermes"
    if any(item.get("status") == "skipped" for item in items):
        return "review-expansion-sandbox-queue"
    return "run-hermes-literature-strategy-expansion"


def evaluate_expansion_sandbox_queue(
    *,
    queue_report_path: str | Path,
    output_dir: str | Path,
    registry_path: str | Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    queue_file = _require_file(Path(queue_report_path), "sandbox queue report")
    queue = _read_json(queue_file)
    if queue.get("state_hint") != QUEUE_STATE_HINT:
        raise HermesLiteratureExpansionError("sandbox queue report has invalid state_hint")
    root = Path(output_dir)
    report_path = root / "expansion_sandbox_queue_evaluation.json"
    _guard_output(report_path, force=force)
    root.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    event_paths: list[str] = []
    candidate_paths: list[str] = []
    for index, item in enumerate(_object_items(queue.get("items"))):
        candidate_id = _text(item.get("algorithm_candidate_id"), "algorithm_candidate_id")
        item_dir = root / f"{index:03d}_{_slug(candidate_id)}"
        item_dir.mkdir(parents=True, exist_ok=True)
        if item.get("status") == "executed":
            selected_path = _require_file(
                Path(_text(
                    item.get("selected_algorithm_candidates_path"),
                    "selected_algorithm_candidates_path",
                )),
                "selected algorithm candidates",
            )
            evidence_path = item_dir / "sandbox_evidence.json"
            evidence = _materialize_queue_sandbox_evidence(
                queue_item=item,
                output_path=evidence_path,
                current_best_score=float(current_best_score),
                force=force,
            )
            promotion_path = item_dir / "candidate_promotion.json"
            promotion = evaluate_algorithm_candidate_promotion(
                algorithm_candidates_path=selected_path,
                algorithm_candidate_id=candidate_id,
                sandbox_evidence_paths=[evidence_path],
                output_path=promotion_path,
                current_best_score=float(current_best_score),
                force=force,
            )
            event_path = item_dir / "literature_registry_event.json"
            event = record_literature_registry_event(
                event_source_path=promotion_path,
                output_path=event_path,
                force=force,
            )
            candidate_paths.append(str(selected_path))
            event_paths.append(event["literature_registry_event_path"])
            items.append(
                {
                    "status": "evaluated",
                    "algorithm_candidate_id": candidate_id,
                    "candidate_family": item.get("candidate_family"),
                    "evidence_path": evidence["evidence_path"],
                    "promotion_path": promotion["algorithm_candidate_promotion_path"],
                    "registry_event_path": event["literature_registry_event_path"],
                    "decision": promotion["decision"],
                    "reason_codes": promotion.get("reason_codes", []),
                    "score": evidence["score"],
                    "score_delta_vs_current_best": evidence[
                        "score_delta_vs_current_best"
                    ],
                    "next_command": promotion["next_command"],
                }
            )
            continue
        event_path = item_dir / "literature_registry_event.json"
        event = _write_queue_item_registry_event(
            queue_item=item,
            output_path=event_path,
            force=force,
        )
        event_paths.append(event["literature_registry_event_path"])
        items.append(
            {
                "status": "registry_event_recorded",
                "algorithm_candidate_id": candidate_id,
                "candidate_family": item.get("candidate_family"),
                "registry_event_path": event["literature_registry_event_path"],
                "decision": event["decision"],
                "reason_codes": event["failure_codes"],
                "next_command": event["recommended_next"],
            }
        )

    registry = update_literature_registry(
        registry_path=Path(registry_path),
        algorithm_candidates_paths=_unique_paths(candidate_paths),
        event_paths=event_paths,
        force=force,
    )
    promotion_count = sum(1 for item in items if item.get("decision") == "promote")
    evaluated_count = sum(1 for item in items if item.get("status") == "evaluated")
    report = {
        "schema_version": 1,
        "state_hint": QUEUE_EVALUATION_STATE_HINT,
        "queue_evaluation_report_path": str(report_path),
        "queue_report_path": str(queue_file),
        "current_best_score": float(current_best_score),
        "artifacts": {
            "queue_evaluation_report_path": str(report_path),
            "registry_path": registry["registry_path"],
        },
        "metrics": {
            "item_count": len(items),
            "evidence_count": evaluated_count,
            "promotion_count": promotion_count,
            "registry_event_count": len(event_paths),
            "registry_candidate_count": registry["candidate_node_count"],
        },
        "items": items,
        "next_command": "record-strategy-proposal"
        if promotion_count
        else "summarize-literature-registry-for-hermes",
    }
    _write_json(report_path, report)
    return report


def _materialize_queue_sandbox_evidence(
    *,
    queue_item: dict[str, Any],
    output_path: Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    if isinstance(queue_item.get("checkpoint_family_report_path"), str):
        return _materialize_checkpoint_family_sandbox_evidence(
            queue_item=queue_item,
            output_path=output_path,
            current_best_score=current_best_score,
            force=force,
        )
    if isinstance(queue_item.get("local_strategy_loop_report_path"), str):
        return _materialize_local_strategy_sandbox_evidence(
            queue_item=queue_item,
            output_path=output_path,
            current_best_score=current_best_score,
            force=force,
        )
    lab_report_file = _require_file(
        Path(_text(queue_item.get("sandbox_report_path"), "sandbox_report_path")),
        "hypothesis lab report",
    )
    lab_report = _read_json(lab_report_file)
    sandbox = lab_report.get("sandbox") if isinstance(lab_report.get("sandbox"), dict) else {}
    best = (
        sandbox.get("best_candidate")
        if isinstance(sandbox.get("best_candidate"), dict)
        else {}
    )
    verifier = (
        sandbox.get("verifier") if isinstance(sandbox.get("verifier"), dict) else {}
    )
    score = _queue_evidence_score(best, verifier)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    candidate_id = _text(queue_item.get("algorithm_candidate_id"), "algorithm_candidate_id")
    evidence = {
        "schema_version": 1,
        "state_hint": "SANDBOX_EVIDENCE_RECORDED",
        "evidence_path": str(output_file),
        "evidence_id": f"evidence_{_slug(candidate_id)}",
        "candidate_id": candidate_id,
        "candidate_family": queue_item.get("candidate_family"),
        "source_report_path": str(lab_report_file),
        "rules_passed": bool(verifier.get("context_preserved", False)),
        "score": score,
        "segment_scores": _queue_segment_scores(best),
        "runtime": {"inference_seconds": _queue_inference_seconds(best)},
        "physics_metrics": {"residual_delta": 0.0},
        "ablation_metrics": {
            "score_delta": _float_or_default(
                verifier.get("score_delta_vs_baseline"),
                0.0,
            )
        },
        "score_delta_vs_current_best": score - float(current_best_score),
        "artifact_paths": _queue_artifact_paths(lab_report),
        "reproducibility": {"deterministic": True, "repeat_count": 1},
    }
    _write_json(output_file, evidence)
    return evidence


def _materialize_local_strategy_sandbox_evidence(
    *,
    queue_item: dict[str, Any],
    output_path: Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    loop_report_file = _require_file(
        Path(
            _text(
                queue_item.get("local_strategy_loop_report_path"),
                "local_strategy_loop_report_path",
            )
        ),
        "local strategy loop report",
    )
    loop_report = _read_json(loop_report_file)
    best = (
        loop_report.get("best_candidate")
        if isinstance(loop_report.get("best_candidate"), dict)
        else {}
    )
    selected = _select_local_strategy_evidence_candidate(
        loop_report=loop_report,
        best=best,
    )
    score = _float_or_default(selected.get("score"), 0.0)
    best_valid_score = _float_or_default(best.get("score"), 0.0)
    candidate_gate_issues = _local_strategy_candidate_gate_issues(selected)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    candidate_id = _text(queue_item.get("algorithm_candidate_id"), "algorithm_candidate_id")
    artifact_paths = [str(loop_report_file)]
    artifacts = loop_report.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ["best_submission_dir", "best_judge_report_path"]:
            value = artifacts.get(key)
            if isinstance(value, str) and value.strip():
                artifact_paths.append(value)
    evidence = {
        "schema_version": 1,
        "state_hint": "SANDBOX_EVIDENCE_RECORDED",
        "evidence_path": str(output_file),
        "evidence_id": f"evidence_{_slug(candidate_id)}",
        "candidate_id": candidate_id,
        "candidate_family": queue_item.get("candidate_family"),
        "source_report_path": str(loop_report_file),
        "rules_passed": bool(selected.get("rules_passed")),
        "score": score,
        "segment_scores": _local_strategy_segment_scores(selected),
        "runtime": {
            "inference_seconds": _local_strategy_inference_seconds(selected)
        },
        "physics_metrics": {"residual_delta": 0.0},
        "ablation_metrics": {"score_delta": score - float(current_best_score)},
        "score_delta_vs_current_best": score - float(current_best_score),
        "artifact_paths": artifact_paths,
        "reproducibility": {"deterministic": True, "repeat_count": 1},
    }
    if selected is not best:
        evidence["diagnostic_only"] = True
        evidence["best_valid_candidate_id"] = best.get("candidate_id")
        evidence["best_valid_score"] = best_valid_score
        evidence["diagnostic_candidate_id"] = selected.get("candidate_id")
    if candidate_gate_issues:
        evidence["candidate_gate_issues"] = candidate_gate_issues
    _write_json(output_file, evidence)
    return evidence


def _materialize_checkpoint_family_sandbox_evidence(
    *,
    queue_item: dict[str, Any],
    output_path: Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    checkpoint_report_file = _require_file(
        Path(
            _text(
                queue_item.get("checkpoint_family_report_path"),
                "checkpoint_family_report_path",
            )
        ),
        "checkpoint family sandbox report",
    )
    checkpoint_report = _read_json(checkpoint_report_file)
    if checkpoint_report.get("state_hint") != "CHECKPOINT_FAMILY_SANDBOX_RAN":
        raise HermesLiteratureExpansionError(
            "checkpoint family sandbox report has invalid state_hint"
        )
    best = (
        checkpoint_report.get("best_candidate")
        if isinstance(checkpoint_report.get("best_candidate"), dict)
        else {}
    )
    score = _float_or_default(best.get("score"), 0.0)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    candidate_id = _text(queue_item.get("algorithm_candidate_id"), "algorithm_candidate_id")
    artifact_paths = [str(checkpoint_report_file)]
    artifacts = checkpoint_report.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ["best_submission_dir", "best_judge_report_path"]:
            value = artifacts.get(key)
            if isinstance(value, str) and value.strip():
                artifact_paths.append(value)
    evidence = {
        "schema_version": 1,
        "state_hint": "SANDBOX_EVIDENCE_RECORDED",
        "evidence_path": str(output_file),
        "evidence_id": f"evidence_{_slug(candidate_id)}",
        "candidate_id": candidate_id,
        "candidate_family": queue_item.get("candidate_family"),
        "source_report_path": str(checkpoint_report_file),
        "rules_passed": bool(best.get("rules_passed")),
        "score": score,
        "segment_scores": _checkpoint_segment_scores(best),
        "runtime": {
            "inference_seconds": _float_or_default(best.get("inference_time"), 0.0)
        },
        "physics_metrics": {"residual_delta": 0.0},
        "ablation_metrics": {"score_delta": score - float(current_best_score)},
        "score_delta_vs_current_best": score - float(current_best_score),
        "artifact_paths": artifact_paths,
        "reproducibility": {"deterministic": True, "repeat_count": 1},
    }
    _write_json(output_file, evidence)
    return evidence


def _write_queue_item_registry_event(
    *,
    queue_item: dict[str, Any],
    output_path: Path,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    status = str(queue_item.get("status", "skipped"))
    outcome = str(queue_item.get("outcome", "blocked"))
    event_type = "candidate_blocked" if outcome == "blocked" else "candidate_rejected"
    decision = "blocked" if outcome == "blocked" else "reject"
    raw_reason_codes = _string_list(queue_item.get("reason_codes")) or [f"queue_{status}"]
    failure_codes = _queue_failure_codes(raw_reason_codes, outcome=outcome)
    report = {
        "schema_version": 1,
        "state_hint": LITERATURE_REGISTRY_EVENT_STATE_HINT,
        "literature_registry_event_path": str(output_file),
        "event_source_path": str(queue_item.get("sandbox_plan_path") or ""),
        "event_type": event_type,
        "algorithm_candidate_id": _text(
            queue_item.get("algorithm_candidate_id"),
            "algorithm_candidate_id",
        ),
        "candidate_family": queue_item.get("candidate_family"),
        "evidence_id": queue_item.get("sandbox_plan_path"),
        "decision": decision,
        "failure_codes": failure_codes,
        "raw_reason_codes": raw_reason_codes,
        "source_ids": [],
        "algorithm_card_id": None,
        "recommended_next": "design_executor_before_retry"
        if decision == "blocked"
        else "avoid_repeating_without_new_evidence",
        "retry_allowed": decision == "blocked",
        "metric_results": {},
        "next_command": "update-literature-registry",
    }
    _write_json(output_file, report)
    return report


def _queue_failure_codes(reason_codes: list[str], *, outcome: str) -> list[str]:
    lowered = " ".join(reason_codes).lower()
    if "unsupported" in lowered or "checkpoint" in lowered:
        return ["unsupported_executor"]
    if "runtime" in lowered or "timeout" in lowered:
        return ["runtime_exceeded"]
    if outcome == "blocked":
        return ["unsupported_executor"]
    return ["sandbox_rule_failed"]


def _queue_evidence_score(best: dict[str, Any], verifier: dict[str, Any]) -> float:
    for key in ["task1_total_score_proxy", "segment_score"]:
        value = best.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return _float_or_default(verifier.get("best_segment_score"), 0.0)


def _queue_segment_scores(best: dict[str, Any]) -> dict[str, Any]:
    segments = best.get("segments")
    if not isinstance(segments, dict):
        return {}
    return {
        key: value.get("score")
        for key, value in segments.items()
        if isinstance(value, dict) and isinstance(value.get("score"), int | float)
    }


def _queue_inference_seconds(best: dict[str, Any]) -> float:
    return _float_or_default(best.get("elapsed_seconds"), 0.0)


def _queue_artifact_paths(lab_report: dict[str, Any]) -> list[str]:
    artifacts = lab_report.get("artifacts")
    if not isinstance(artifacts, dict):
        return []
    paths = []
    for key in ["sandbox_report_path", "best_submission_dir", "best_prediction_path"]:
        value = artifacts.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value)
    return paths


def _checkpoint_paths_from_candidate(candidate: dict[str, Any]) -> list[Path]:
    resources = (
        candidate.get("resource_bindings")
        if isinstance(candidate.get("resource_bindings"), dict)
        else {}
    )
    return [Path(item) for item in _string_list(resources.get("checkpoint_paths"))]


def _task_id_from_candidate(candidate: dict[str, Any]) -> int:
    value = candidate.get("target_task")
    if isinstance(value, int) and value in {1, 2}:
        return value
    if isinstance(value, str):
        stripped = value.lower().replace("task", "").replace("-", "").strip()
        if stripped in {"1", "2"}:
            return int(stripped)
    return 1


def _calibration_weights_from_candidate(candidate: dict[str, Any]) -> list[float]:
    grid = (
        candidate.get("parameter_grid")
        if isinstance(candidate.get("parameter_grid"), dict)
        else {}
    )
    values = grid.get("calibration_weights")
    weights = [
        float(item)
        for item in values
        if isinstance(item, int | float) and not isinstance(item, bool)
    ] if isinstance(values, list) else []
    return weights or [0.0, 0.25, 0.5, 1.0]


def _fno_finetune_steps_from_candidate(candidate: dict[str, Any]) -> int:
    grid = (
        candidate.get("parameter_grid")
        if isinstance(candidate.get("parameter_grid"), dict)
        else {}
    )
    values = grid.get("fno_finetune_steps")
    if isinstance(values, list):
        numeric = [
            int(item)
            for item in values
            if isinstance(item, int | float) and not isinstance(item, bool)
        ]
        return max(numeric) if numeric else 0
    return 0


def _checkpoint_family_report_path(report: dict[str, Any]) -> Path:
    for value in [
        report.get("checkpoint_family_report_path"),
        report.get("artifacts", {}).get("checkpoint_family_report_path")
        if isinstance(report.get("artifacts"), dict)
        else None,
    ]:
        if isinstance(value, str) and value.strip():
            return Path(value)
    raise HermesLiteratureExpansionError(
        "checkpoint family sandbox report missing report path"
    )


def _checkpoint_segment_scores(best: dict[str, Any]) -> dict[str, Any]:
    segment_score = best.get("segment_score")
    if isinstance(segment_score, dict):
        return {
            key: value
            for key, value in segment_score.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    segments = best.get("segments")
    if isinstance(segments, dict):
        return {
            key: value.get("score")
            for key, value in segments.items()
            if isinstance(value, dict) and isinstance(value.get("score"), int | float)
        }
    return {}


def _is_bounded_local_strategy_candidate(
    *,
    candidate: dict[str, Any],
    cli_command: str,
) -> bool:
    if cli_command != "run-hermes-batch-optimization-loop":
        return False
    if candidate.get("candidate_family") != "local_strategy_skill":
        return False
    return _local_strategy_candidate_generator(candidate) is not None


def _local_strategy_candidate_generator(candidate: dict[str, Any]) -> str | None:
    executor = (
        candidate.get("executor_spec")
        if isinstance(candidate.get("executor_spec"), dict)
        else {}
    )
    hooks = set(_string_list(executor.get("implementation_hooks")))
    if "hypothesis_lab_early_subsegment_patch_search" in hooks:
        return "hypothesis_lab_early_subsegment_patch_search"
    return None


def _bounded_local_strategy_max_rounds(candidate_generator: str | None) -> int:
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        return 5
    return 1


def _bounded_local_strategy_objective(candidate_generator: str | None) -> str:
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        return "early_accuracy"
    return "score"


def _bounded_local_strategy_reference_sample_limit(
    *,
    candidate: dict[str, Any] | None = None,
    candidate_generator: str | None,
    requested_limit: int | None,
) -> int | None:
    if requested_limit is not None:
        return requested_limit
    candidate_limit = _candidate_reference_sample_limit(candidate)
    if candidate_limit is not None:
        return candidate_limit
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        return 64
    return None


def _candidate_reference_sample_limit(candidate: dict[str, Any] | None) -> int | None:
    if not isinstance(candidate, dict):
        return None
    grid = candidate.get("parameter_grid")
    if not isinstance(grid, dict):
        return None
    values = grid.get("reference_sample_limits")
    if not isinstance(values, list):
        return None
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _local_strategy_candidate_options(candidate: dict[str, Any]) -> dict[str, Any]:
    grid = candidate.get("parameter_grid")
    if not isinstance(grid, dict):
        return {}
    options: dict[str, Any] = {}
    source_stability = _first_bool_grid_value(grid.get("source_stability_guard"))
    if source_stability is not None:
        options["source_stability_guard"] = source_stability
    smoothing_weight = _first_float_grid_value(grid.get("handoff_smoothing_weights"))
    if smoothing_weight is not None:
        options["handoff_smoothing_weight"] = smoothing_weight
    return options


def _first_bool_grid_value(value: Any) -> bool | None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, bool):
            return item
    return None


def _first_float_grid_value(value: Any) -> float | None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, int | float) and not isinstance(item, bool):
            return float(item)
    return None


def _bounded_local_strategy_deadline_seconds(candidate_generator: str | None) -> float:
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        return 300.0
    return 120.0


def _local_strategy_loop_report_path(report: dict[str, Any]) -> Path:
    for value in [
        report.get("loop_report_path"),
        report.get("artifacts", {}).get("loop_report_path")
        if isinstance(report.get("artifacts"), dict)
        else None,
    ]:
        if isinstance(value, str) and value.strip():
            return Path(value)
    raise HermesLiteratureExpansionError("local strategy loop missing report path")


def _select_local_strategy_evidence_candidate(
    *,
    loop_report: dict[str, Any],
    best: dict[str, Any],
) -> dict[str, Any]:
    best_score = _float_or_default(best.get("score"), -1e30)
    diagnostic_candidates = []
    trajectory = loop_report.get("trajectory")
    if isinstance(trajectory, list):
        for item in trajectory:
            if not isinstance(item, dict):
                continue
            if "target_reference_local_validation_only" not in (
                _local_strategy_candidate_gate_issues(item)
            ):
                continue
            if _float_or_default(item.get("score"), -1e30) <= best_score:
                continue
            diagnostic_candidates.append(item)
    if not diagnostic_candidates:
        return best
    return max(
        diagnostic_candidates,
        key=lambda item: _float_or_default(item.get("score"), -1e30),
    )


def _local_strategy_candidate_gate_issues(candidate: dict[str, Any]) -> list[str]:
    issues = []
    value = candidate.get("candidate_gate_issues")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip() and item not in issues:
                issues.append(item.strip())
    return issues


def _local_strategy_segment_scores(best: dict[str, Any]) -> dict[str, Any]:
    segment_score = best.get("segment_score")
    if isinstance(segment_score, dict):
        return {
            key: value
            for key, value in segment_score.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    segments = best.get("segments")
    if isinstance(segments, dict):
        return {
            key: value.get("score")
            for key, value in segments.items()
            if isinstance(value, dict) and isinstance(value.get("score"), int | float)
        }
    signals = best.get("objective_signals")
    if isinstance(signals, dict):
        early = signals.get("early_segment_score")
        if isinstance(early, int | float) and not isinstance(early, bool):
            return {"early": float(early)}
    return {}


def _local_strategy_inference_seconds(best: dict[str, Any]) -> float:
    signals = best.get("objective_signals")
    if isinstance(signals, dict):
        value = signals.get("inference_time")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return _float_or_default(best.get("inference_time"), 0.0)


def _unique_paths(paths: list[str]) -> list[Path]:
    seen: set[str] = set()
    unique = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(Path(path))
    return unique


def prepare_expansion_blocked_executor_authoring_tasks(
    *,
    queue_report_path: str | Path,
    output_dir: str | Path,
    context_path: str | Path,
    target_path: str | Path,
    submission_id: str,
    max_tasks: int | None = None,
    force: bool,
) -> dict[str, Any]:
    queue_file = _require_file(Path(queue_report_path), "sandbox queue report")
    queue = _read_json(queue_file)
    if queue.get("state_hint") != QUEUE_STATE_HINT:
        raise HermesLiteratureExpansionError("sandbox queue report has invalid state_hint")
    limit = max_tasks if max_tasks is not None else len(_object_items(queue.get("items")))
    if limit < 0:
        raise HermesLiteratureExpansionError("max_tasks must be >= 0")
    root = Path(output_dir)
    report_path = root / "blocked_executor_authoring_tasks.json"
    _guard_output(report_path, force=force)
    root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for item in _blocked_queue_items(queue)[:limit]:
        task = _prepare_blocked_executor_authoring_task(
            queue_item=item,
            output_dir=root / _slug(_text(
                item.get("algorithm_candidate_id"),
                "algorithm_candidate_id",
            )),
            context_path=Path(context_path),
            target_path=Path(target_path),
            submission_id=submission_id,
            force=force,
        )
        tasks.append(task)
    report = {
        "schema_version": 1,
        "state_hint": BLOCKED_EXECUTOR_TASKS_STATE_HINT,
        "blocked_executor_tasks_path": str(report_path),
        "queue_report_path": str(queue_file),
        "metrics": {
            "blocked_item_count": len(_blocked_queue_items(queue)),
            "task_count": len(tasks),
        },
        "tasks": tasks,
        "next_command": "audit-bounded-executor-authoring-task"
        if tasks
        else "summarize-literature-registry-for-hermes",
    }
    _write_json(report_path, report)
    return report


def _blocked_queue_items(queue: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in _object_items(queue.get("items"))
        if item.get("status") == "skipped" and item.get("outcome") == "blocked"
    ]


def _prepare_blocked_executor_authoring_task(
    *,
    queue_item: dict[str, Any],
    output_dir: Path,
    context_path: Path,
    target_path: Path,
    submission_id: str,
    force: bool,
) -> dict[str, Any]:
    plan_file = _require_file(
        Path(_text(queue_item.get("sandbox_plan_path"), "sandbox_plan_path")),
        "blocked sandbox plan",
    )
    plan = _read_json(plan_file)
    candidate = plan.get("selected_candidate")
    if not isinstance(candidate, dict):
        raise HermesLiteratureExpansionError(
            "blocked sandbox plan missing selected_candidate"
        )
    candidate_id = _text(
        candidate.get("algorithm_candidate_id")
        or queue_item.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        executor = plan.get("executor") if isinstance(plan.get("executor"), dict) else {}
    cli_command = _text(executor.get("cli_command"), "cli_command")
    requested_executor_id = f"bounded_{_slug(cli_command)}_executor"
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "executor_design_request.json"
    design_spec_path = output_dir / "bounded_executor_design_spec.json"
    executor_dir = output_dir / "executor"
    authoring_path = output_dir / "bounded_executor_authoring_task.json"
    request = _blocked_queue_executor_design_request(
        queue_item=queue_item,
        candidate=candidate,
        plan_file=plan_file,
        requested_executor_id=requested_executor_id,
        cli_command=cli_command,
        output_path=request_path,
    )
    _guard_output(request_path, force=force)
    _write_json(request_path, request)
    design = create_bounded_executor_design_spec(
        executor_design_request_path=request_path,
        output_path=design_spec_path,
        force=force,
    )
    materialized = materialize_bounded_executor_template(
        design_spec_path=design["bounded_executor_design_spec_path"],
        output_dir=executor_dir,
        force=force,
    )
    authoring = prepare_bounded_executor_authoring_task(
        design_spec_path=design["bounded_executor_design_spec_path"],
        manifest_path=materialized["manifest_path"],
        executor_path=materialized["executor_path"],
        context_path=_authoring_context_path(
            context_path=context_path,
            target_path=target_path,
            output_path=output_dir / "context_fixture.hdf5",
            force=force,
        ),
        target_path=target_path,
        output_path=authoring_path,
        sandbox_output_dir=output_dir / "sandbox",
        audit_output_path=output_dir / "bounded_executor_audit.json",
        event_output_path=output_dir / "bounded_executor_registry_event.json",
        submission_id=submission_id,
        force=force,
    )
    return {
        "algorithm_candidate_id": candidate_id,
        "candidate_family": candidate.get("candidate_family")
        or queue_item.get("candidate_family"),
        "requested_executor_id": requested_executor_id,
        "executor_gap": cli_command,
        "executor_design_request_path": str(request_path),
        "design_spec_path": design["bounded_executor_design_spec_path"],
        "manifest_path": materialized["manifest_path"],
        "executor_path": materialized["executor_path"],
        "authoring_task_path": authoring["bounded_executor_authoring_task_path"],
        "next_command": authoring["next_command"],
    }


def _blocked_queue_executor_design_request(
    *,
    queue_item: dict[str, Any],
    candidate: dict[str, Any],
    plan_file: Path,
    requested_executor_id: str,
    cli_command: str,
    output_path: Path,
) -> dict[str, Any]:
    grounding = (
        candidate.get("source_grounding")
        if isinstance(candidate.get("source_grounding"), dict)
        else {}
    )
    return {
        "schema_version": 1,
        "state_hint": "EXECUTOR_DESIGN_REQUEST_CREATED",
        "executor_design_request_path": str(output_path),
        "algorithm_candidate_id": _text(
            candidate.get("algorithm_candidate_id")
            or queue_item.get("algorithm_candidate_id"),
            "algorithm_candidate_id",
        ),
        "candidate_family": candidate.get("candidate_family")
        or queue_item.get("candidate_family"),
        "unsupported_hook": cli_command,
        "requested_executor_id": requested_executor_id,
        "hypothesis": str(candidate.get("hypothesis") or "Bound local strategy executor."),
        "mechanism": str(
            candidate.get("mechanism")
            or f"Translate {cli_command} into a bounded predict(context, candidate, params) implementation."
        ),
        "source_grounding": grounding,
        "reason_codes": _queue_failure_codes(
            _string_list(queue_item.get("reason_codes")),
            outcome=str(queue_item.get("outcome", "blocked")),
        ),
        "executor_contract": {
            "inputs": ["context", "candidate", "params"],
            "outputs": ["prediction"],
            "forbidden_actions": [
                "execute the original heavy CLI command directly",
                "read target_path inside predict",
                "perform shell, network, or file IO",
            ],
        },
        "minimal_experiment": {
            "level": 2,
            "description": (
                "Implement a bounded executor adapter for the blocked queue item "
                "and run it through run-bounded-executor-sandbox."
            ),
        },
        "risk_controls": _string_list(candidate.get("risk_controls"))
        + [
            f"blocked_queue_reason:{code}"
            for code in _string_list(queue_item.get("reason_codes"))
        ],
        "queue_context": {
            "queue_status": queue_item.get("status"),
            "queue_outcome": queue_item.get("outcome"),
            "sandbox_plan_path": str(plan_file),
        },
        "next_command": "create-bounded-executor-design-spec",
    }


def build_local_asset_algorithm_candidates(
    *,
    local_strategy_library_path: str | Path,
    checkpoint_paths: list[str | Path] | None,
    output_path: str | Path,
    max_strategy_candidates: int,
    force: bool,
) -> dict[str, Any]:
    if max_strategy_candidates < 0:
        raise HermesLiteratureExpansionError("max_strategy_candidates must be >= 0")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    library = _read_json(Path(local_strategy_library_path))
    if library.get("state_hint") != "LOCAL_STRATEGY_LIBRARY_BUILT":
        raise HermesLiteratureExpansionError(
            "local strategy library has invalid state_hint"
        )
    entries = library.get("entries")
    if not isinstance(entries, list):
        raise HermesLiteratureExpansionError(
            "local strategy library entries must be a list"
        )
    candidates = []
    for checkpoint_path in checkpoint_paths or []:
        checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
        candidates.append(_checkpoint_asset_candidate(checkpoint_file))
    ranked_entries = sorted(
        [entry for entry in entries if isinstance(entry, dict)],
        key=lambda entry: (
            int(entry.get("priority", 999)),
            str(entry.get("strategy_id", "")),
        ),
    )
    for entry in ranked_entries[:max_strategy_candidates]:
        candidates.append(_local_strategy_asset_candidate(entry))
    if not candidates:
        raise HermesLiteratureExpansionError(
            "local asset candidate generation produced no candidates"
        )
    report = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "source_state_hint": LOCAL_ASSET_CANDIDATES_STATE_HINT,
        "algorithm_candidates_path": str(output_file),
        "local_strategy_library_path": str(local_strategy_library_path),
        "algorithm_candidate_count": len(candidates),
        "algorithm_candidates": candidates,
        "next_command": "run-hermes-literature-strategy-expansion",
    }
    _write_json(output_file, report)
    return report


def _authoring_context_path(
    *,
    context_path: Path,
    target_path: Path,
    output_path: Path,
    force: bool,
) -> Path:
    if context_path.resolve() != target_path.resolve():
        return context_path
    _guard_output(output_path, force=force)
    import h5py

    with h5py.File(target_path, "r") as source:
        dataset_name = _first_hdf5_tensor_dataset_name(source)
        data = source[dataset_name][()]
    if getattr(data, "ndim", 0) >= 3 and data.shape[1] > 10:
        data = data[:, :10, :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as target:
        target.create_dataset(dataset_name, data=data)
    return output_path


def _first_hdf5_dataset_name(handle: Any) -> str:
    for key in handle.keys():
        node = handle[key]
        if hasattr(node, "shape"):
            return str(key)
    raise HermesLiteratureExpansionError("HDF5 file contains no dataset")


def _first_hdf5_tensor_dataset_name(handle: Any) -> str:
    tensor = handle.get("tensor")
    if tensor is not None and getattr(tensor, "ndim", len(getattr(tensor, "shape", ()))) == 3:
        return "tensor"
    for key in handle.keys():
        node = handle[key]
        if len(getattr(node, "shape", ())) == 3:
            return str(key)
    return _first_hdf5_dataset_name(handle)


def record_hermes_literature_candidate_outcome(
    *,
    algorithm_candidate_id: str,
    candidate_key: str,
    outcome: str,
    reason_codes: list[str],
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    if output_file.exists() and not force:
        memory = _read_failure_memory(output_file)
    else:
        memory = {
            "schema_version": 1,
            "state_hint": FAILURE_MEMORY_STATE_HINT,
            "failed_algorithm_candidate_ids": [],
            "failed_candidate_keys": [],
            "failed_candidate_families": [],
            "outcomes": [],
        }
    selected_id = _text(algorithm_candidate_id, "algorithm_candidate_id")
    selected_key = _text(candidate_key, "candidate_key")
    normalized_outcome = _text(outcome, "outcome")
    if normalized_outcome not in {"failed", "passed", "blocked"}:
        raise HermesLiteratureExpansionError(f"unsupported outcome: {outcome}")
    outcomes = memory.setdefault("outcomes", [])
    if not isinstance(outcomes, list):
        outcomes = []
        memory["outcomes"] = outcomes
    outcomes.append(
        {
            "algorithm_candidate_id": selected_id,
            "candidate_key": selected_key,
            "outcome": normalized_outcome,
            "reason_codes": [item for item in reason_codes if isinstance(item, str)],
        }
    )
    if normalized_outcome == "failed":
        _append_unique(memory, "failed_algorithm_candidate_ids", selected_id)
        _append_unique(memory, "failed_candidate_keys", selected_key)
        for family in _failed_candidate_families_for_outcome(
            candidate_key=selected_key,
            reason_codes=reason_codes,
        ):
            _append_unique(memory, "failed_candidate_families", family)
    _write_json(output_file, memory)
    return memory


def summarize_literature_registry_for_hermes(
    *,
    registry_path: str | Path,
    output_path: str | Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    registry_file = Path(registry_path)
    registry = _read_json(registry_file)
    if registry.get("state_hint") != "LITERATURE_REGISTRY_UPDATED":
        raise HermesLiteratureExpansionError("literature registry has invalid state_hint")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)

    source_backlog = _registry_source_backlog(registry)
    candidate_actions = _registry_candidate_actions(
        registry,
        current_best_score=float(current_best_score),
    )
    learning_signals = _registry_learning_signals(
        candidate_actions,
        current_best_score=float(current_best_score),
    )
    family_summary = _registry_family_summary(candidate_actions)
    avoid_repetition = _registry_avoid_repetition(
        candidate_actions,
        current_best_score=float(current_best_score),
    )
    report = {
        "schema_version": 1,
        "state_hint": REGISTRY_SUMMARY_STATE_HINT,
        "registry_path": str(registry_file),
        "summary_path": str(output_file),
        "current_best_score": float(current_best_score),
        "source_backlog": source_backlog,
        "candidate_action_queue": candidate_actions,
        "family_summary": family_summary,
        "learning_signals": learning_signals,
        "avoid_repetition": avoid_repetition,
        "metrics": {
            "source_backlog_count": len(source_backlog),
            "candidate_action_count": len(candidate_actions),
            "candidate_family_count": len(family_summary),
            "learning_signal_count": len(learning_signals),
            "avoid_repetition_count": len(avoid_repetition),
        },
        "next_command": _summary_next_command(candidate_actions, source_backlog),
    }
    _write_json(output_file, report)
    return report


def refine_algorithm_candidate(
    *,
    registry_summary_path: str | Path,
    output_path: str | Path,
    algorithm_candidate_id: str | None = None,
    force: bool,
) -> dict[str, Any]:
    summary_file = _require_file(Path(registry_summary_path), "registry summary")
    summary = _read_json(summary_file)
    if summary.get("state_hint") != REGISTRY_SUMMARY_STATE_HINT:
        raise HermesLiteratureExpansionError("registry summary has invalid state_hint")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)

    action = _select_refinement_action(
        summary,
        algorithm_candidate_id=algorithm_candidate_id,
    )
    candidate_id = _text(action.get("candidate_id"), "candidate_id")
    candidate_family = action.get("candidate_family")
    candidate_generator = _refinement_candidate_generator(action)
    learning_objective = _refinement_learning_objective(action)
    reference_sample_limit = _refinement_reference_sample_limit(action)
    report = {
        "schema_version": 1,
        "state_hint": REFINEMENT_STATE_HINT,
        "candidate_refinement_path": str(output_file),
        "registry_summary_path": str(summary_file),
        "algorithm_candidate_id": candidate_id,
        "candidate_family": candidate_family,
        "refinement_type": "legal_reference_retest",
        "reason": action.get("reason"),
        "diagnostic_score": action.get("best_score"),
        "score_delta_vs_current_best": action.get("score_delta_vs_current_best"),
        "candidate_gate_issues": _string_list(action.get("candidate_gate_issues")),
        "executor": action.get("executor"),
        "candidate_generator": candidate_generator,
        "learning_objective": learning_objective,
        "legal_reference_constraints": {
            "reference_policy": "reference_path_must_be_distinct_from_target_path",
            "forbidden_reference_modes": [
                "target_as_reference",
                "local_validation_only",
            ],
            "required_checks": [
                "rules_passed_true",
                "context_prefix_preserved",
                "no_target_reference_local_validation_only_gate_issue",
            ],
        },
        "suggested_sandbox_args": {
            "reference_path_required": True,
            "reference_sample_limit": reference_sample_limit,
            "candidate_generator": candidate_generator,
            "learning_objective": learning_objective,
        },
        "success_criteria": [
            "rules_passed must be true",
            "candidate_gate_issues must not include target_reference_local_validation_only",
            "score_delta_vs_current_best should remain positive on legal reference",
        ],
        "next_command": "run-expansion-sandbox-queue",
    }
    _write_json(output_file, report)
    return report


def record_promoted_algorithm_strategy_proposal(
    *,
    candidate_promotion_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    promotion_file = _require_file(
        Path(candidate_promotion_path),
        "candidate promotion",
    )
    promotion = _read_json(promotion_file)
    if promotion.get("state_hint") != ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT:
        raise HermesLiteratureExpansionError(
            "candidate promotion has invalid state_hint"
        )
    if promotion.get("decision") != "promote" or not promotion.get(
        "promote_to_strategy_growth"
    ):
        raise HermesLiteratureExpansionError(
            "candidate promotion did not approve strategy growth"
        )
    candidates_path = _require_file(
        Path(
            _text(
                promotion.get("algorithm_candidates_path"),
                "algorithm_candidates_path",
            )
        ),
        "algorithm candidates",
    )
    candidates = _read_json(candidates_path)
    candidate_id = _text(
        promotion.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    candidate = _algorithm_candidate_by_id(candidates, candidate_id)
    output_file = Path(output_path)
    if output_file.exists():
        if not force:
            raise HermesLiteratureExpansionError(
                f"strategy proposal already exists: {output_file}"
            )
        output_file.unlink()
    return record_strategy_proposal(
        strategy_id=_strategy_id_from_promoted_candidate(candidate),
        title=_promoted_strategy_title(candidate),
        source_type=_strategy_source_type_from_candidate(candidate),
        problem_context=_promoted_strategy_problem_context(candidate, promotion),
        hypothesis=_text(candidate.get("hypothesis"), "candidate hypothesis"),
        mechanism=_text(candidate.get("mechanism"), "candidate mechanism"),
        implementation_plan=_promoted_strategy_implementation_plan(
            candidate=candidate,
            promotion=promotion,
            promotion_path=promotion_file,
        ),
        expected_observables=_promoted_strategy_observables(promotion),
        known_risks=_promoted_strategy_risks(candidate),
        output_path=output_file,
    )


def convert_candidate_promotion_to_strategy_evidence(
    *,
    candidate_promotion_path: str | Path,
    extra_candidate_promotion_paths: list[str | Path] | None = None,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    promotion_files = [
        _require_file(Path(candidate_promotion_path), "candidate promotion"),
        *[
            _require_file(Path(path), "candidate promotion")
            for path in (extra_candidate_promotion_paths or [])
        ],
    ]
    promotions = [_read_candidate_promotion(path) for path in promotion_files]
    primary = promotions[0]
    _validate_same_promoted_candidate(promotions)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    weighted_deltas = _promotion_weighted_deltas(promotions)
    failure_count = sum(1 for promotion in promotions if promotion.get("decision") != "promote")
    physics_deltas = [
        _float_or_default(_promotion_metrics(promotion).get("physics_residual_delta"), 0.0)
        for promotion in promotions
    ]
    evidence = {
        "schema_version": 1,
        "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "evidence_summary_path": str(output_file),
        "source_candidate_promotion_path": str(promotion_files[0]),
        "source_candidate_promotion_paths": [str(path) for path in promotion_files],
        "algorithm_candidate_id": primary.get("algorithm_candidate_id"),
        "candidate_family": primary.get("candidate_family"),
        "evaluation_window_count": len(weighted_deltas),
        "mean_score_delta": statistics.fmean(weighted_deltas),
        "score_delta_std": statistics.stdev(weighted_deltas)
        if len(weighted_deltas) > 1
        else 0.0,
        "long_window_score_delta": min(weighted_deltas),
        "physics_residual_delta": max(physics_deltas) if physics_deltas else 0.0,
        "inference_time_regression": 0.0,
        "failure_rate": failure_count / len(promotions),
        "candidate_promotion_metric_results": [
            _promotion_metrics(promotion) for promotion in promotions
        ],
        "candidate_promotion_reason_codes": [
            _string_list(promotion.get("reason_codes")) for promotion in promotions
        ],
        "sandbox_evidence_paths": _promotion_sandbox_evidence_paths(promotions),
        "next_command": "evaluate-strategy-promotion",
    }
    _write_json(output_file, evidence)
    return evidence


def plan_candidate_evidence_accumulation(
    *,
    strategy_promotion_report_path: str | Path,
    strategy_evidence_summary_path: str | Path | None = None,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    promotion_file = _require_file(
        Path(strategy_promotion_report_path),
        "strategy promotion report",
    )
    promotion = _read_json(promotion_file)
    if promotion.get("state_hint") != "STRATEGY_PROMOTION_EVALUATED":
        raise HermesLiteratureExpansionError(
            "strategy promotion report has invalid state_hint"
        )
    evidence_file = _require_file(
        Path(
            strategy_evidence_summary_path
            if strategy_evidence_summary_path is not None
            else _text(promotion.get("evidence_summary_path"), "evidence_summary_path")
        ),
        "strategy evidence summary",
    )
    evidence = _read_json(evidence_file)
    rubric_file = _require_file(
        Path(_text(promotion.get("experiment_rubric_path"), "experiment_rubric_path")),
        "experiment rubric",
    )
    rubric = _read_json(rubric_file)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    current_windows = max(
        int(_float_or_default(evidence.get("evaluation_window_count"), 0.0)),
        0,
    )
    required_windows = _rubric_min_evaluation_windows(rubric)
    missing_windows = max(required_windows - current_windows, 0)
    needs_accumulation = (
        "insufficient_evaluation_windows" in _string_list(promotion.get("reason_codes"))
        and missing_windows > 0
    )
    plan = {
        "schema_version": 1,
        "state_hint": EVIDENCE_ACCUMULATION_PLAN_STATE_HINT,
        "evidence_accumulation_plan_path": str(output_file),
        "strategy_promotion_report_path": str(promotion_file),
        "strategy_evidence_summary_path": str(evidence_file),
        "experiment_rubric_path": str(rubric_file),
        "strategy_id": promotion.get("strategy_id"),
        "algorithm_candidate_id": evidence.get("algorithm_candidate_id"),
        "decision": "accumulate_more_evidence"
        if needs_accumulation
        else "no_accumulation_required",
        "current_window_count": current_windows,
        "required_window_count": required_windows,
        "missing_window_count": missing_windows if needs_accumulation else 0,
        "source_candidate_promotion_paths": _evidence_source_candidate_promotion_paths(
            evidence
        ),
        "recommended_actions": _evidence_accumulation_actions(
            missing_windows=missing_windows if needs_accumulation else 0,
            algorithm_candidate_id=evidence.get("algorithm_candidate_id"),
            current_windows=current_windows,
        ),
        "next_command": "run-hermes-runtime-closed-loop"
        if needs_accumulation
        else promotion.get("next_command", "append-strategy-ledger"),
    }
    _write_json(output_file, plan)
    return plan


def _read_candidate_promotion(path: Path) -> dict[str, Any]:
    promotion = _read_json(path)
    if promotion.get("state_hint") != ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT:
        raise HermesLiteratureExpansionError(
            "candidate promotion has invalid state_hint"
        )
    return promotion


def _validate_same_promoted_candidate(promotions: list[dict[str, Any]]) -> None:
    if not promotions:
        raise HermesLiteratureExpansionError(
            "at least one candidate promotion is required"
        )
    candidate_id = promotions[0].get("algorithm_candidate_id")
    for promotion in promotions:
        if promotion.get("algorithm_candidate_id") != candidate_id:
            raise HermesLiteratureExpansionError(
                "candidate promotion artifacts must reference the same candidate"
            )


def _promotion_weighted_deltas(promotions: list[dict[str, Any]]) -> list[float]:
    deltas: list[float] = []
    for promotion in promotions:
        metrics = _promotion_metrics(promotion)
        delta = _float_or_default(metrics.get("score_delta_vs_current_best"), 0.0)
        count = max(int(_float_or_default(promotion.get("evidence_count"), 1.0)), 1)
        deltas.extend([delta] * count)
    return deltas or [0.0]


def _promotion_metrics(promotion: dict[str, Any]) -> dict[str, Any]:
    metrics = promotion.get("metric_results")
    return metrics if isinstance(metrics, dict) else {}


def _promotion_sandbox_evidence_paths(promotions: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for promotion in promotions:
        for path in _string_list(promotion.get("sandbox_evidence_paths")):
            if path not in paths:
                paths.append(path)
    return paths


def _rubric_min_evaluation_windows(rubric: dict[str, Any]) -> int:
    thresholds = rubric.get("promotion_thresholds")
    if not isinstance(thresholds, dict):
        return 1
    return max(
        int(_float_or_default(thresholds.get("min_evaluation_windows"), 1.0)),
        1,
    )


def _evidence_accumulation_actions(
    *,
    missing_windows: int,
    algorithm_candidate_id: Any,
    current_windows: int,
) -> list[dict[str, Any]]:
    if missing_windows <= 0:
        return []
    return [
        {
            "action": "run_legal_reference_retest",
            "algorithm_candidate_id": algorithm_candidate_id,
            "repeat_count": missing_windows,
            "next_command": "run-hermes-runtime-closed-loop",
            "reason": "insufficient_evaluation_windows",
            "suggested_sandbox_args": {
                "reference_sample_limit": _next_evidence_reference_sample_limit(
                    current_windows
                ),
            },
        }
    ]


def _next_evidence_reference_sample_limit(current_windows: int) -> int:
    return max(64 * (max(current_windows, 0) + 1), 64)


def _evidence_source_candidate_promotion_paths(evidence: dict[str, Any]) -> list[str]:
    paths = _string_list(evidence.get("source_candidate_promotion_paths"))
    if paths:
        return paths
    single = evidence.get("source_candidate_promotion_path")
    return [single.strip()] if isinstance(single, str) and single.strip() else []


def _select_refinement_action(
    summary: dict[str, Any],
    *,
    algorithm_candidate_id: str | None,
) -> dict[str, Any]:
    actions = [
        item
        for item in _object_items(summary.get("candidate_action_queue"))
        if item.get("recommended_action") == "refine_with_legal_reference"
    ]
    if algorithm_candidate_id is not None:
        candidate_id = _text(algorithm_candidate_id, "algorithm_candidate_id")
        actions = [item for item in actions if item.get("candidate_id") == candidate_id]
    if not actions:
        raise HermesLiteratureExpansionError(
            "registry summary has no legal-reference refinement action"
        )
    return actions[0]


def _algorithm_candidate_by_id(
    candidates_report: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    candidates = _object_items(candidates_report.get("algorithm_candidates"))
    adapted = candidates_report.get("adapted_candidate")
    if isinstance(adapted, dict):
        candidates.append(adapted)
    for candidate in candidates:
        if candidate.get("algorithm_candidate_id") == candidate_id:
            return candidate
    raise HermesLiteratureExpansionError(
        f"algorithm candidate not found: {candidate_id}"
    )


def _strategy_id_from_promoted_candidate(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_grounding")
    source_algorithm_id = None
    if isinstance(source, dict):
        source_algorithm_id = source.get("algorithm_card_id")
    value = (
        source_algorithm_id
        or candidate.get("parent_algorithm_card_id")
        or candidate.get("algorithm_candidate_id")
    )
    text = _text(value, "strategy source id")
    for prefix in [
        "local_strategy_",
        "algorithm_",
        "candidate_local_strategy_",
        "candidate_",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return _slug(text)


def _promoted_strategy_title(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_grounding")
    if isinstance(source, dict):
        title = source.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return _text(candidate.get("algorithm_candidate_id"), "algorithm_candidate_id")


def _strategy_source_type_from_candidate(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_grounding")
    source_type = source.get("source_type") if isinstance(source, dict) else None
    if source_type in {"local_canonical_skill", "local_library", "local_strategy"}:
        return "local_library"
    if source_type in {"arxiv", "openalex", "github", "paper", "external_retrieval"}:
        return "external_retrieval"
    if candidate.get("candidate_family") == "local_strategy_skill":
        return "local_library"
    return "agent_hypothesis"


def _promoted_strategy_problem_context(
    candidate: dict[str, Any],
    promotion: dict[str, Any],
) -> str:
    target_problem = candidate.get("target_problem") or "PDE_Burgers"
    target_task = candidate.get("target_task") or "task1"
    metrics = promotion.get("metric_results")
    score = metrics.get("score") if isinstance(metrics, dict) else None
    delta = (
        metrics.get("score_delta_vs_current_best")
        if isinstance(metrics, dict)
        else None
    )
    return (
        f"{target_problem} {target_task} candidate promoted by Hypothesis Lab "
        f"with score={score} and score_delta_vs_current_best={delta}."
    )


def _promoted_strategy_implementation_plan(
    *,
    candidate: dict[str, Any],
    promotion: dict[str, Any],
    promotion_path: Path,
) -> dict[str, Any]:
    executor = candidate.get("executor_spec")
    cli_command = executor.get("cli_command") if isinstance(executor, dict) else None
    allowed_commands = [cli_command] if isinstance(cli_command, str) and cli_command else []
    evidence_paths = [
        str(Path(path))
        for path in _string_list(promotion.get("sandbox_evidence_paths"))
    ]
    return {
        "algorithm_candidate_id": candidate.get("algorithm_candidate_id"),
        "candidate_family": candidate.get("candidate_family"),
        "candidate_promotion_path": str(promotion_path),
        "algorithm_candidates_path": promotion.get("algorithm_candidates_path"),
        "sandbox_evidence_paths": evidence_paths,
        "allowed_commands": allowed_commands,
        "implementation_hooks": _string_list(
            executor.get("implementation_hooks") if isinstance(executor, dict) else []
        ),
        "promotion_metric_results": promotion.get("metric_results", {}),
        "promotion_reason_codes": promotion.get("reason_codes", []),
    }


def _promoted_strategy_observables(promotion: dict[str, Any]) -> list[str]:
    metrics = promotion.get("metric_results")
    if not isinstance(metrics, dict):
        metrics = {}
    observables = [
        f"score={metrics.get('score')}",
        f"score_delta_vs_current_best={metrics.get('score_delta_vs_current_best')}",
        f"inference_seconds={metrics.get('inference_seconds')}",
        f"physics_residual_delta={metrics.get('physics_residual_delta')}",
        f"rules_passed={metrics.get('rules_passed')}",
    ]
    return [item for item in observables if not item.endswith("=None")]


def _promoted_strategy_risks(candidate: dict[str, Any]) -> list[str]:
    risks = _string_list(candidate.get("risk_controls"))
    if risks:
        return risks
    return ["promotion may not generalize beyond the sandbox reference split"]


def _refinement_candidate_generator(action: dict[str, Any]) -> str:
    if action.get("candidate_family") == "local_strategy_skill":
        return "hypothesis_lab_early_subsegment_patch_search"
    return "auto"


def _refinement_learning_objective(action: dict[str, Any]) -> str:
    if action.get("candidate_family") == "local_strategy_skill":
        return "early_accuracy"
    return "score"


def _refinement_reference_sample_limit(action: dict[str, Any]) -> int | None:
    if action.get("candidate_family") == "local_strategy_skill":
        return 64
    return None


def _registry_avoid_repetition(
    actions: list[dict[str, Any]],
    *,
    current_best_score: float,
) -> list[dict[str, Any]]:
    avoid = [
        {
            "candidate_id": action["candidate_id"],
            "reason": "sandbox_score_below_current_best",
            "best_score": action.get("best_score"),
            "current_best_score": current_best_score,
            "recommended_next": "only_retry_with_new_variant_or_ablation",
        }
        for action in actions
        if action.get("recommended_action") == "refine_or_ablate"
        and float(action.get("score_delta_vs_current_best", 0.0)) < 0.0
    ]
    for action in actions:
        if (
            action.get("reason") == "candidate_failure_history_blocks_retry"
            and action.get("retry_allowed") is False
        ):
            avoid.append(
                {
                    "candidate_id": action["candidate_id"],
                    "reason": action["reason"],
                    "failure_codes": action.get("failure_codes", []),
                    "recommended_next": action.get("recommended_next"),
                    "retry_allowed": False,
                }
            )
        elif action.get("recommended_action") == "generate_variant_or_ablation":
            avoid.append(
                {
                    "candidate_id": action["candidate_id"],
                    "reason": action["reason"],
                    "failure_codes": action.get("failure_codes", []),
                    "recommended_next": action.get("recommended_next"),
                    "retry_allowed": False,
                }
            )
    return avoid


def _rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    checkpoint_paths: list[Path],
    failure_memory: dict[str, Any],
    registry_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rankings = []
    failed_ids = set(_string_list(failure_memory.get("failed_algorithm_candidate_ids")))
    failed_keys = set(_string_list(failure_memory.get("failed_candidate_keys")))
    failed_families = set(_string_list(failure_memory.get("failed_candidate_families")))
    avoid_ids = {
        item["candidate_id"]
        for item in _object_items(registry_summary.get("avoid_repetition"))
        if isinstance(item.get("candidate_id"), str)
    }
    for index, candidate in enumerate(candidates):
        candidate_id = _text(
            candidate.get("algorithm_candidate_id"),
            "algorithm_candidate_id",
        )
        route, score = _candidate_route(candidate, checkpoint_paths=checkpoint_paths)
        key = _candidate_key(candidate)
        family = str(candidate.get("candidate_family", "")).strip()
        ranking = {
            "algorithm_candidate_id": candidate_id,
            "candidate_family": family,
            "candidate_key": key,
            "route": route,
            "selection_score": score,
            "source_order": index,
        }
        if (
            candidate_id in failed_ids
            or key in failed_keys
            or family in failed_families
        ):
            ranking["skip_reason"] = "failed_memory"
            ranking["selection_score"] = -1
        elif candidate_id in avoid_ids:
            ranking["skip_reason"] = "registry_avoid_repetition"
            ranking["selection_score"] = -1
        rankings.append(ranking)
    return sorted(
        rankings,
        key=lambda item: (
            1 if item.get("skip_reason") else 0,
            -float(item["selection_score"]),
            int(item["source_order"]),
        ),
    )


def _registry_source_backlog(registry: dict[str, Any]) -> list[dict[str, Any]]:
    backlog = []
    for source in registry.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_id = _text(source.get("source_id"), "source_id")
        parse_status = str(source.get("parse_status", "not_parsed"))
        card_ids = source.get("algorithm_card_ids")
        if parse_status in {"not_parsed", "unknown", "failed"}:
            backlog.append(
                {
                    "source_id": source_id,
                    "title": source.get("title"),
                    "status": "downloaded_not_parsed",
                    "next_command": "submit-mineru-parse",
                    "priority": 80,
                }
            )
        elif parse_status == "usable" and not card_ids:
            backlog.append(
                {
                    "source_id": source_id,
                    "title": source.get("title"),
                    "status": "parsed_without_algorithm_cards",
                    "next_command": "extract-algorithm-cards",
                    "priority": 70,
                }
            )
    return sorted(
        backlog,
        key=lambda item: (-int(item["priority"]), str(item["source_id"])),
    )


def _registry_candidate_actions(
    registry: dict[str, Any],
    *,
    current_best_score: float,
) -> list[dict[str, Any]]:
    actions = []
    algorithm_by_candidate = _registry_algorithm_by_candidate(registry)
    for candidate in registry.get("candidate_nodes", []):
        if not isinstance(candidate, dict):
            continue
        candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
        status = str(candidate.get("status", "compiled"))
        executor = str(candidate.get("executor", ""))
        algorithm = algorithm_by_candidate.get(candidate_id, {})
        action = {
            "candidate_id": candidate_id,
            "candidate_family": candidate.get("candidate_family"),
            "status": status,
            "executor": executor,
            "algorithm_node_id": algorithm.get("node_id"),
            "source_ids": candidate.get("source_ids", []),
        }
        failure = _latest_failure_history(candidate)
        if failure and _failure_requires_local_strategy_variant(
            candidate=candidate,
            failure=failure,
        ):
            failure_codes = _string_list(failure.get("failure_codes"))
            action.update(
                {
                    "recommended_action": "generate_variant_or_ablation",
                    "next_command": "run-candidate-expansion-handoff",
                    "priority": 75,
                    "reason": "local_strategy_failed_legal_retest_requires_variant",
                    "failure_codes": failure_codes,
                    "recommended_next": "generate_variant_with_new_ablation",
                    "retry_allowed": False,
                    "failure_artifact_path": failure.get("artifact_path"),
                    "variant_constraints": {
                        "parent_candidate_id": candidate_id,
                        "required_mutation_evidence": True,
                        "forbidden_retry_without_new_ablation": True,
                        "target_failure_codes": failure_codes,
                    },
                }
            )
        elif failure and failure.get("retry_allowed") is False:
            action.update(
                {
                    "recommended_action": "deprioritize",
                    "next_command": "review-literature-registry",
                    "priority": 20,
                    "reason": "candidate_failure_history_blocks_retry",
                    "failure_codes": failure.get("failure_codes", []),
                    "recommended_next": failure.get("recommended_next"),
                    "retry_allowed": False,
                    "failure_artifact_path": failure.get("artifact_path"),
                }
            )
        elif status == "needs_refinement" and _failure_has_code(
            failure,
            "target_leakage_risk",
        ):
            metrics = (
                failure.get("metric_results")
                if isinstance(failure.get("metric_results"), dict)
                else {}
            )
            action.update(
                {
                    "recommended_action": "refine_with_legal_reference",
                    "next_command": "refine-algorithm-candidate",
                    "priority": 95,
                    "reason": "target_reference_signal_requires_legal_retest",
                    "best_score": _optional_float(metrics.get("score")),
                    "score_delta_vs_current_best": _optional_float(
                        metrics.get("score_delta_vs_current_best")
                    ),
                    "candidate_gate_issues": _string_list(
                        metrics.get("candidate_gate_issues")
                    ),
                    "diagnostic_only": bool(metrics.get("diagnostic_only")),
                    "failure_codes": failure.get("failure_codes", []),
                    "recommended_next": failure.get("recommended_next"),
                    "retry_allowed": failure.get("retry_allowed"),
                    "failure_artifact_path": failure.get("artifact_path"),
                }
            )
        elif status == "compiled":
            action.update(
                {
                    "recommended_action": "run_sandbox",
                    "next_command": executor or "run-hypothesis-lab",
                    "priority": 100,
                    "reason": "candidate_has_no_sandbox_evidence",
                }
            )
        elif status == "sandbox_verified":
            evidence = _latest_sandbox_evidence(candidate)
            score = _optional_float(evidence.get("best_score"))
            delta = None if score is None else round(score - current_best_score, 6)
            should_promote = delta is not None and delta >= 0
            action.update(
                {
                    "recommended_action": (
                        "promote_strategy" if should_promote else "refine_or_ablate"
                    ),
                    "next_command": (
                        "record-strategy-proposal"
                        if should_promote
                        else "run-hermes-literature-strategy-expansion"
                    ),
                    "priority": 90 if should_promote else 80,
                    "reason": (
                        "sandbox_score_meets_or_exceeds_current_best"
                        if should_promote
                        else "sandbox_score_below_current_best"
                    ),
                    "best_score": score,
                    "score_delta_vs_current_best": delta,
                    "rules_passed": evidence.get("rules_passed"),
                    "evidence_artifact_path": evidence.get("artifact_path"),
                }
            )
        elif status == "blocked":
            action.update(
                {
                    "recommended_action": "design_executor",
                    "next_command": "create-executor-design-request",
                    "priority": 60,
                    "reason": "candidate_blocked_by_executor_gap",
                }
            )
        elif status == "promoted":
            promotion_event = _latest_candidate_event(
                registry=registry,
                candidate_id=candidate_id,
                event_type="candidate_promoted",
            )
            candidate_promotion_path = _candidate_promotion_path_from_event(
                promotion_event
            )
            action.update(
                {
                    "recommended_action": "record_strategy_proposal",
                    "next_command": "record-promoted-algorithm-strategy-proposal",
                    "priority": 85,
                    "reason": "candidate_promotion_recorded",
                    "candidate_promotion_path": candidate_promotion_path,
                    "registry_event_path": promotion_event.get("artifact_path"),
                }
            )
        elif status == "sandbox_failed":
            action.update(
                {
                    "recommended_action": "deprioritize",
                    "next_command": "review-sandbox-failure",
                    "priority": 20,
                    "reason": "candidate_failed_sandbox_rules",
                }
            )
        else:
            action.update(
                {
                    "recommended_action": "review",
                    "next_command": "review-literature-registry",
                    "priority": 10,
                    "reason": f"unhandled_candidate_status:{status}",
                }
            )
        actions.append(action)
    return sorted(
        actions,
        key=lambda item: (
            -int(item["priority"]),
            str(item["candidate_id"]),
        ),
    )


def _registry_learning_signals(
    actions: list[dict[str, Any]],
    *,
    current_best_score: float,
) -> list[dict[str, Any]]:
    signals = []
    for action in actions:
        if action.get("recommended_action") == "refine_or_ablate":
            signals.append(
                {
                    "signal": "sandbox_underperformed_current_best",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "best_score": action.get("best_score"),
                    "current_best_score": current_best_score,
                    "score_delta_vs_current_best": action.get(
                        "score_delta_vs_current_best"
                    ),
                    "recommended_learning_objective": (
                        "search_variant_or_ablation_before_retry"
                    ),
                }
            )
        elif action.get("recommended_action") == "generate_variant_or_ablation":
            signals.append(
                {
                    "signal": "local_strategy_variant_required",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "failure_codes": action.get("failure_codes", []),
                    "variant_constraints": action.get("variant_constraints", {}),
                    "recommended_learning_objective": (
                        "generate_mutated_strategy_with_new_ablation_axis"
                    ),
                }
            )
        elif action.get("recommended_action") == "refine_with_legal_reference":
            signals.append(
                {
                    "signal": "target_reference_signal_requires_legal_retest",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "best_score": action.get("best_score"),
                    "current_best_score": current_best_score,
                    "score_delta_vs_current_best": action.get(
                        "score_delta_vs_current_best"
                    ),
                    "candidate_gate_issues": action.get("candidate_gate_issues", []),
                    "recommended_learning_objective": (
                        "legal_reference_retest_before_promotion"
                    ),
                }
            )
        elif action.get("recommended_action") == "design_executor":
            signals.append(
                {
                    "signal": "executor_gap_blocks_algorithm_family",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "recommended_learning_objective": "build_bounded_executor",
                }
            )
        elif action.get("reason") == "candidate_failure_history_blocks_retry":
            signals.append(
                {
                    "signal": "candidate_failure_recorded",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "failure_codes": action.get("failure_codes", []),
                    "recommended_learning_objective": (
                        "avoid_repetition_until_new_evidence_or_variant"
                    ),
                }
            )
        elif action.get("recommended_action") == "run_sandbox":
            signals.append(
                {
                    "signal": "candidate_needs_first_sandbox_evidence",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "recommended_learning_objective": "measure_before_promotion",
                }
            )
    priority = {
        "target_reference_signal_requires_legal_retest": 110,
        "sandbox_underperformed_current_best": 100,
        "local_strategy_variant_required": 95,
        "candidate_failure_recorded": 90,
        "executor_gap_blocks_algorithm_family": 80,
        "candidate_needs_first_sandbox_evidence": 60,
    }
    return sorted(
        signals,
        key=lambda item: (
            -priority.get(str(item.get("signal")), 0),
            str(item.get("candidate_id")),
        ),
    )


def _registry_family_summary(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        family = str(action.get("candidate_family") or "unknown_family")
        grouped.setdefault(family, []).append(action)
    summaries = []
    for family, family_actions in grouped.items():
        status_counts: dict[str, int] = {}
        best_score = None
        best_candidate_id = None
        for action in family_actions:
            status = str(action.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            score = action.get("best_score")
            if isinstance(score, int | float) and not isinstance(score, bool):
                if best_score is None or float(score) > best_score:
                    best_score = float(score)
                    best_candidate_id = str(action["candidate_id"])
        top_action = sorted(
            family_actions,
            key=lambda item: (
                -int(item.get("priority", 0)),
                str(item.get("candidate_id")),
            ),
        )[0]
        summaries.append(
            {
                "candidate_family": family,
                "candidate_count": len(family_actions),
                "status_counts": dict(sorted(status_counts.items())),
                "best_score": best_score,
                "best_candidate_id": best_candidate_id,
                "recommended_action": top_action.get("recommended_action"),
                "next_command": top_action.get("next_command"),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -int(_family_priority(str(item.get("recommended_action")))),
            str(item["candidate_family"]),
        ),
    )


def _family_priority(recommended_action: str) -> int:
    return {
        "run_sandbox": 100,
        "promote_strategy": 90,
        "refine_or_ablate": 80,
        "generate_variant_or_ablation": 75,
        "design_executor": 60,
        "deprioritize": 20,
    }.get(recommended_action, 10)


def _registry_algorithm_by_candidate(
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {}
    for algorithm in registry.get("algorithm_nodes", []):
        if not isinstance(algorithm, dict):
            continue
        for candidate_id in algorithm.get("candidate_ids", []):
            if isinstance(candidate_id, str):
                result[candidate_id] = algorithm
    return result


def _latest_sandbox_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("sandbox_evidence")
    if not isinstance(evidence, list) or not evidence:
        return {}
    latest = evidence[-1]
    return latest if isinstance(latest, dict) else {}


def _latest_failure_history(candidate: dict[str, Any]) -> dict[str, Any]:
    history = candidate.get("failure_history")
    if not isinstance(history, list) or not history:
        return {}
    latest = history[-1]
    return latest if isinstance(latest, dict) else {}


def _latest_candidate_event(
    *,
    registry: dict[str, Any],
    candidate_id: str,
    event_type: str,
) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for event in _object_items(registry.get("events")):
        if (
            event.get("algorithm_candidate_id") == candidate_id
            and event.get("event_type") == event_type
        ):
            latest = event
    return latest


def _candidate_promotion_path_from_event(event: dict[str, Any]) -> str | None:
    value = event.get("event_source_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    artifact_path = event.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None
    artifact_file = Path(artifact_path)
    if not artifact_file.is_file():
        return artifact_path.strip()
    try:
        payload = _read_json(artifact_file)
    except Exception:
        return artifact_path.strip()
    value = payload.get("event_source_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return artifact_path.strip()


def _failure_has_code(failure: dict[str, Any], code: str) -> bool:
    codes = failure.get("failure_codes")
    return isinstance(codes, list) and code in codes


def _failure_requires_local_strategy_variant(
    *,
    candidate: dict[str, Any],
    failure: dict[str, Any],
) -> bool:
    if candidate.get("candidate_family") != "local_strategy_skill":
        return False
    if failure.get("retry_allowed") is not False:
        return False
    codes = set(_string_list(failure.get("failure_codes")))
    return bool(codes.intersection({"score_regressed", "ablation_missing_or_weak"}))


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _summary_next_command(
    actions: list[dict[str, Any]],
    source_backlog: list[dict[str, Any]],
) -> str:
    if actions:
        command = actions[0].get("next_command")
        if isinstance(command, str) and command.strip():
            return command
    if source_backlog:
        return str(source_backlog[0]["next_command"])
    return "run-hermes-literature-strategy-expansion"


def _candidate_route(
    candidate: dict[str, Any],
    *,
    checkpoint_paths: list[Path],
) -> tuple[str, float]:
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        return "invalid", 0.0
    cli_command = executor.get("cli_command")
    if (
        cli_command == "blocked_requires_executor_design"
        and executor.get("unsupported_hook") == "fourier_neural_operator_architecture"
        and checkpoint_paths
    ):
        return "checkpoint_adapter", 100.0
    if cli_command == "blocked_requires_executor_design":
        return "executor_design", 10.0
    return "existing_executor", 80.0


def _candidate_key(candidate: dict[str, Any]) -> str:
    grounding = candidate.get("source_grounding")
    if not isinstance(grounding, dict):
        grounding = {}
    card_id = grounding.get("algorithm_card_id")
    executor = candidate.get("executor_spec")
    hook = executor.get("unsupported_hook") if isinstance(executor, dict) else None
    family = candidate.get("candidate_family")
    mutation = candidate.get("mutation_evidence")
    mutation_type = (
        mutation.get("mutation_type")
        if isinstance(mutation, dict)
        and isinstance(mutation.get("mutation_type"), str)
        and mutation.get("mutation_type")
        else None
    )
    return "|".join(
        str(part)
        for part in [card_id or "unknown_card", hook or family or "unknown_family"]
        + ([mutation_type] if mutation_type else [])
    )


def _expand_candidate(
    *,
    candidate: dict[str, Any],
    candidates_file: Path,
    workspace: Path,
    checkpoint_paths: list[Path],
    force: bool,
) -> dict[str, Any]:
    candidate_id = _text(candidate.get("algorithm_candidate_id"), "algorithm_candidate_id")
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        raise HermesLiteratureExpansionError("candidate executor_spec must be an object")
    cli_command = _text(executor.get("cli_command"), "cli_command")
    slug = _slug(candidate_id)
    if (
        cli_command == "blocked_requires_executor_design"
        and executor.get("unsupported_hook") == "fourier_neural_operator_architecture"
        and checkpoint_paths
    ):
        adapted_path = workspace / f"{slug}_adapted_candidate.json"
        adapted = adapt_fno_architecture_candidate(
            algorithm_candidates_path=candidates_file,
            algorithm_candidate_id=candidate_id,
            checkpoint_path=checkpoint_paths[0],
            output_path=adapted_path,
            force=force,
        )
        adapted_candidate_id = adapted["adapted_candidate"]["algorithm_candidate_id"]
        sandbox_path = workspace / f"{slug}_sandbox_plan.json"
        sandbox = plan_candidate_sandbox(
            algorithm_candidates_path=adapted_path,
            algorithm_candidate_id=adapted_candidate_id,
            output_path=sandbox_path,
            force=force,
        )
        return {
            "algorithm_candidate_id": candidate_id,
            "action": "adapt_to_checkpoint_sandbox",
            "executable": True,
            "adapted_candidate_path": str(adapted_path),
            "adapted_algorithm_candidate_id": adapted_candidate_id,
            "sandbox_plan_path": str(sandbox_path),
            "next_command": sandbox["next_command"],
        }
    if cli_command == "blocked_requires_executor_design":
        request_path = workspace / f"{slug}_executor_design_request.json"
        request = create_executor_design_request(
            algorithm_candidates_path=candidates_file,
            algorithm_candidate_id=candidate_id,
            output_path=request_path,
            force=force,
        )
        handoff_path = workspace / f"{slug}_blocked_candidate_handoff.json"
        build_blocked_candidate_handoff(
            executor_design_request_path=request_path,
            output_path=handoff_path,
            force=force,
        )
        proposal_path = workspace / f"{slug}_strategy_proposal.json"
        proposal = _record_design_strategy_proposal(
            request=request,
            request_path=request_path,
            output_path=proposal_path,
        )
        return {
            "algorithm_candidate_id": candidate_id,
            "action": "record_executor_design_proposal",
            "executable": False,
            "executor_design_request_path": str(request_path),
            "blocked_candidate_handoff_path": str(handoff_path),
            "strategy_proposal_path": proposal["strategy_proposal_path"],
            "next_command": proposal["next_command"],
        }
    sandbox_path = workspace / f"{slug}_sandbox_plan.json"
    sandbox = plan_candidate_sandbox(
        algorithm_candidates_path=candidates_file,
        algorithm_candidate_id=candidate_id,
        output_path=sandbox_path,
        force=force,
    )
    return {
        "algorithm_candidate_id": candidate_id,
        "action": "plan_existing_executor_sandbox",
        "executable": not bool(sandbox.get("blocked")),
        "sandbox_plan_path": str(sandbox_path),
        "next_command": sandbox["next_command"],
    }


def _checkpoint_asset_candidate(checkpoint_file: Path) -> dict[str, Any]:
    family = _checkpoint_family(checkpoint_file)
    candidate_id = f"candidate_local_checkpoint_{_slug(checkpoint_file.stem)}"
    return {
        "algorithm_candidate_id": candidate_id,
        "parent_algorithm_card_id": f"local_checkpoint_{_slug(checkpoint_file.stem)}",
        "candidate_family": "official_checkpoint_family",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": "checkpoint",
        "hypothesis": (
            f"Evaluate local {family} checkpoint as a bounded candidate family "
            "after literature candidates are exhausted."
        ),
        "mechanism": (
            "Route the local checkpoint through the audited checkpoint-family "
            "sandbox and compare calibration weights against persistence."
        ),
        "executor_spec": {
            "cli_command": "run-checkpoint-family-sandbox",
            "required_options": [
                "--checkpoint-path",
                "--task-id",
                "--test-path",
                "--target-path",
                "--output-dir",
            ],
        },
        "parameter_grid": {
            "calibration_weights": [0.0, 0.25, 0.5, 1.0],
            "fno_finetune_steps": [0],
        },
        "resource_bindings": {"checkpoint_paths": [str(checkpoint_file)]},
        "benchmark_plan": {
            "judge": "hermes_local_judge",
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
            "verify checkpoint tensor convention before promotion",
        ],
        "source_grounding": {
            "source_ids": [str(checkpoint_file)],
            "algorithm_card_id": f"local_checkpoint_{_slug(checkpoint_file.stem)}",
        },
    }


def _local_strategy_asset_candidate(entry: dict[str, Any]) -> dict[str, Any]:
    strategy_id = _text(entry.get("strategy_id"), "strategy_id")
    hooks = _string_list(entry.get("implementation_hooks"))
    cli_command = _preferred_cli_hook(hooks)
    candidate = {
        "algorithm_candidate_id": f"candidate_local_strategy_{_slug(strategy_id)}",
        "parent_algorithm_card_id": f"local_strategy_{_slug(strategy_id)}",
        "candidate_family": "local_strategy_skill",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": _primary_model_family(entry),
        "hypothesis": _text(entry.get("hypothesis"), "hypothesis"),
        "mechanism": _text(entry.get("mechanism"), "mechanism"),
        "executor_spec": {
            "cli_command": cli_command,
            "required_options": [],
            "implementation_hooks": hooks,
        },
        "parameter_grid": {"candidate_enabled": [True]},
        "benchmark_plan": {
            "judge": "hermes_local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 2,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "local_strategy_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": _string_list(entry.get("known_risks")),
        "source_grounding": {
            "source_ids": [strategy_id],
            "algorithm_card_id": f"local_strategy_{_slug(strategy_id)}",
            "source_type": entry.get("source_type"),
            "title": entry.get("title"),
        },
    }
    mutation_evidence = entry.get("mutation_evidence")
    if isinstance(mutation_evidence, dict):
        candidate["mutation_evidence"] = mutation_evidence
    return candidate


def _preferred_cli_hook(hooks: list[str]) -> str:
    preferred = [
        "run-hypothesis-lab",
        "run-hermes-operator-tuning-loop",
        "run-burgers-spectral-viscosity-benchmark",
        "execute-fno-finetune-candidate",
        "predict-fno-competition",
        "predict-unet-competition",
    ]
    for hook in preferred:
        if hook in hooks:
            return hook
    return hooks[0] if hooks else "review-local-strategy-candidate"


def _primary_model_family(entry: dict[str, Any]) -> str:
    families = _string_list(entry.get("model_families"))
    return families[0] if families else "all"


def _checkpoint_family(checkpoint_file: Path) -> str:
    name = checkpoint_file.name.lower()
    if "unet" in name:
        return "UNet"
    if "fno" in name:
        return "FNO"
    return "NeuralOperator"


def _record_design_strategy_proposal(
    *,
    request: dict[str, Any],
    request_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    fields = request.get("strategy_proposal_fields")
    if not isinstance(fields, dict):
        raise HermesLiteratureExpansionError(
            "executor design request missing strategy_proposal_fields"
        )
    implementation_steps = fields.get("implementation_plan")
    if not isinstance(implementation_steps, list) or not implementation_steps:
        implementation_steps = [
            "Design bounded executor",
            "Run local sandbox",
            "Promote only with evidence",
        ]
    return record_strategy_proposal(
        strategy_id=_text(fields.get("strategy_id"), "strategy_id"),
        title=_text(fields.get("title"), "title"),
        source_type=str(fields.get("source_type", "external_retrieval")),
        problem_context=(
            "Hermes literature strategy expansion found a grounded algorithm "
            "candidate that cannot execute until a bounded local executor exists."
        ),
        hypothesis=_text(fields.get("rationale"), "rationale"),
        mechanism=_text(request.get("mechanism"), "mechanism"),
        implementation_plan={
            "steps": implementation_steps,
            "executor_design_request_path": str(request_path),
            "executor_contract": request.get("executor_contract", {}),
            "recommended_intermediate_actions": request.get(
                "recommended_intermediate_actions",
                [],
            ),
        },
        expected_observables=_string_list(fields.get("expected_observables")),
        known_risks=_string_list(fields.get("known_risks")),
        output_path=output_path,
    )


def _candidate_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("algorithm_candidates")
    if not isinstance(candidates, list):
        raise HermesLiteratureExpansionError("algorithm_candidates must be a list")
    result = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise HermesLiteratureExpansionError("algorithm candidate must be an object")
        result.append(candidate)
    return result


def _next_command(actions: list[dict[str, Any]], *, skipped_count: int = 0) -> str:
    for action in actions:
        if action.get("executable") is True:
            return str(action["next_command"])
    for action in actions:
        if action.get("strategy_proposal_path"):
            return str(action["next_command"])
    if skipped_count > 0:
        return "build-local-asset-algorithm-candidates"
    return "review-hermes-literature-expansion"


def _recovery_plan(
    *,
    actions: list[dict[str, Any]],
    skipped_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if actions or not skipped_candidates:
        return None
    return {
        "reason": "candidate_pool_exhausted",
        "next_command": "build-local-asset-algorithm-candidates",
        "recommended_inputs": {
            "local_strategy_library_path": "runs/local_strategy_library.json",
            "checkpoint_paths": [
                "data/pdebench/checkpoints/burgers_FNO.tar",
                "data/pdebench/checkpoints/burgers_Unet.tar",
            ],
            "max_strategy_candidates": 5,
        },
        "follow_up_command": "run-hermes-literature-strategy-expansion",
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _append_unique(memory: dict[str, Any], key: str, value: str) -> None:
    values = memory.setdefault(key, [])
    if not isinstance(values, list):
        values = []
        memory[key] = values
    if value not in values:
        values.append(value)


def _failed_candidate_families_for_outcome(
    *,
    candidate_key: str,
    reason_codes: list[str],
) -> list[str]:
    reason_text = " ".join(
        code.lower() for code in reason_codes if isinstance(code, str)
    )
    key_text = candidate_key.lower()
    if (
        "checkpoint_adapter" in reason_text
        or "checkpoint_weight" in reason_text
        or "best_is_persistence" in reason_text
        or (
            "fourier_neural_operator_architecture" in key_text
            and "checkpoint" in reason_text
        )
    ):
        return ["official_checkpoint_family"]
    return []


def _read_failure_memory(path: Path) -> dict[str, Any]:
    memory = _read_json(path)
    if memory.get("state_hint") != FAILURE_MEMORY_STATE_HINT:
        raise HermesLiteratureExpansionError("failure memory has invalid state_hint")
    return memory


def _read_registry_summary(path: Path) -> dict[str, Any]:
    summary = _read_json(path)
    if summary.get("state_hint") != REGISTRY_SUMMARY_STATE_HINT:
        raise HermesLiteratureExpansionError("registry summary has invalid state_hint")
    return summary


def _object_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float_or_default(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HermesLiteratureExpansionError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HermesLiteratureExpansionError("JSON artifact must be an object")
    return value


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise HermesLiteratureExpansionError(f"{label} not found: {path}")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise HermesLiteratureExpansionError(f"artifact already exists: {path}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HermesLiteratureExpansionError(f"{label} must be a non-empty string")
    return value.strip()


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "candidate"
