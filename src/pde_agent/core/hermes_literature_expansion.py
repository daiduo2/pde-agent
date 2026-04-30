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


class HermesLiteratureExpansionError(RuntimeError):
    pass


def run_hermes_literature_strategy_expansion(
    *,
    algorithm_candidates_path: str | Path,
    workspace_dir: str | Path,
    checkpoint_paths: list[str | Path] | None = None,
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
    actions: list[dict[str, Any]] = []
    for candidate in _candidate_list(candidates_report):
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
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "expansion_report_path": str(report_path),
        "algorithm_candidates_path": str(candidates_file),
        "workspace_dir": str(workspace),
        "action_count": len(actions),
        "executable_count": executable_count,
        "proposal_count": proposal_count,
        "actions": actions,
        "next_command": _next_command(actions),
    }
    _write_json(report_path, report)
    return report


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
