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


class HermesLiteratureExpansionError(RuntimeError):
    pass


def run_hermes_literature_strategy_expansion(
    *,
    algorithm_candidates_path: str | Path,
    workspace_dir: str | Path,
    checkpoint_paths: list[str | Path] | None = None,
    failure_memory_path: str | Path | None = None,
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
    candidate_rankings = _rank_candidates(
        _candidate_list(candidates_report),
        checkpoint_paths=checkpoints,
        failure_memory=failure_memory,
    )
    skipped_candidates = [
        ranking
        for ranking in candidate_rankings
        if ranking.get("skip_reason") == "failed_memory"
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
                "prefer checkpoint adapters when a resource is available",
                "then existing bounded executors",
                "then executor-design proposals",
            ],
            "failure_memory_path": str(failure_memory_path)
            if failure_memory_path is not None
            else None,
        },
        "actions": actions,
        "next_command": _next_command(actions),
    }
    _write_json(report_path, report)
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
    _write_json(output_file, memory)
    return memory


def _rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    checkpoint_paths: list[Path],
    failure_memory: dict[str, Any],
) -> list[dict[str, Any]]:
    rankings = []
    failed_ids = set(_string_list(failure_memory.get("failed_algorithm_candidate_ids")))
    failed_keys = set(_string_list(failure_memory.get("failed_candidate_keys")))
    for index, candidate in enumerate(candidates):
        candidate_id = _text(
            candidate.get("algorithm_candidate_id"),
            "algorithm_candidate_id",
        )
        route, score = _candidate_route(candidate, checkpoint_paths=checkpoint_paths)
        key = _candidate_key(candidate)
        ranking = {
            "algorithm_candidate_id": candidate_id,
            "candidate_key": key,
            "route": route,
            "selection_score": score,
            "source_order": index,
        }
        if candidate_id in failed_ids or key in failed_keys:
            ranking["skip_reason"] = "failed_memory"
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


def _next_command(actions: list[dict[str, Any]]) -> str:
    for action in actions:
        if action.get("executable") is True:
            return str(action["next_command"])
    for action in actions:
        if action.get("strategy_proposal_path"):
            return str(action["next_command"])
    return "review-hermes-literature-expansion"


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


def _read_failure_memory(path: Path) -> dict[str, Any]:
    memory = _read_json(path)
    if memory.get("state_hint") != FAILURE_MEMORY_STATE_HINT:
        raise HermesLiteratureExpansionError("failure memory has invalid state_hint")
    return memory


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HermesLiteratureExpansionError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HermesLiteratureExpansionError("JSON artifact must be an object")
    return value


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
