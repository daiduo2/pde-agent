"""Hermes bridge from literature algorithm candidates into strategy growth."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_lab import (
    adapt_fno_architecture_candidate,
    build_blocked_candidate_handoff,
    build_executor_taxonomy,
    create_executor_design_request,
    plan_candidate_sandbox,
)
from pde_agent.core.strategy_growth import record_strategy_proposal


STATE_HINT = "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN"
FAILURE_MEMORY_STATE_HINT = "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED"
LOCAL_ASSET_CANDIDATES_STATE_HINT = "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT"
REGISTRY_SUMMARY_STATE_HINT = "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT"


class HermesLiteratureExpansionError(RuntimeError):
    pass


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
    avoid_repetition = [
        {
            "candidate_id": action["candidate_id"],
            "reason": "sandbox_score_below_current_best",
            "best_score": action.get("best_score"),
            "current_best_score": float(current_best_score),
            "recommended_next": "only_retry_with_new_variant_or_ablation",
        }
        for action in candidate_actions
        if action.get("recommended_action") == "refine_or_ablate"
        and float(action.get("score_delta_vs_current_best", 0.0)) < 0.0
    ]
    report = {
        "schema_version": 1,
        "state_hint": REGISTRY_SUMMARY_STATE_HINT,
        "registry_path": str(registry_file),
        "summary_path": str(output_file),
        "current_best_score": float(current_best_score),
        "source_backlog": source_backlog,
        "candidate_action_queue": candidate_actions,
        "learning_signals": learning_signals,
        "avoid_repetition": avoid_repetition,
        "metrics": {
            "source_backlog_count": len(source_backlog),
            "candidate_action_count": len(candidate_actions),
            "learning_signal_count": len(learning_signals),
            "avoid_repetition_count": len(avoid_repetition),
        },
        "next_command": _summary_next_command(candidate_actions, source_backlog),
    }
    _write_json(output_file, report)
    return report


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
        if status == "compiled":
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
        elif action.get("recommended_action") == "design_executor":
            signals.append(
                {
                    "signal": "executor_gap_blocks_algorithm_family",
                    "candidate_id": action["candidate_id"],
                    "candidate_family": action.get("candidate_family"),
                    "recommended_learning_objective": "build_bounded_executor",
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
        "sandbox_underperformed_current_best": 100,
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
    return "|".join(
        str(part)
        for part in [card_id or "unknown_card", hook or family or "unknown_family"]
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
    return {
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
