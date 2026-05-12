from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.agent_runtime_shared import (
    AgentRuntimeError,
    CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_STATE_HINT,
    _write_json,
)


def build_claude_literature_candidate_closure_task(
    *,
    output_dir: Path,
    workspace_dir: Path,
    runtime_contract_path: Path,
    tool_catalog_path: Path,
    paper_parse_path: Path,
    operator_family: str,
    target_problem: str,
    registry_path: Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    workspace_dir = workspace_dir.expanduser().resolve()
    runtime_contract_path = runtime_contract_path.expanduser().resolve()
    tool_catalog_path = tool_catalog_path.expanduser().resolve()
    paper_parse_path = paper_parse_path.expanduser().resolve()
    registry_path = registry_path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_operator = operator_family.strip().lower()
    if normalized_operator not in {"fno", "deeponet", "unet"}:
        raise AgentRuntimeError(
            "operator_family must be one of: fno, deeponet, unet"
        )

    prompt_path = output_dir / "claude_agent_prompt.md"
    checklist_path = output_dir / "expected_artifacts.json"
    context_path = output_dir / "literature_extraction_context.json"
    skill_path = output_dir / "pde_literature_extraction_skill.json"
    operator_task_path = output_dir / f"{normalized_operator}_literature_adaptation_task.json"
    agent_cards_path = output_dir / "agent_algorithm_cards.json"
    cards_audit_path = output_dir / "algorithm_cards_v2_audit.json"
    candidates_path = output_dir / "algorithm_candidates.json"
    pipeline_workspace = output_dir / "literature_algorithm_v2"
    availability_path = output_dir / "runtime_availability.json"
    execution_issue_path = output_dir / "literature_closure_execution_issue.json"
    session_path = output_dir / "claude_agent_sdk_session.json"

    operator_task_command = {
        "fno": "prepare-fno-literature-adaptation-task",
        "deeponet": "prepare-deeponet-literature-adaptation-task",
        "unet": "prepare-unet-literature-adaptation-task",
    }[normalized_operator]
    required_commands = [
        "prepare-literature-extraction-context",
        "prepare-pde-literature-extraction-skill",
        operator_task_command,
        "record-agent-algorithm-cards",
        "validate-algorithm-cards-v2",
        "compile-algorithm-candidates",
        "run-literature-algorithm-v2-pipeline",
        "audit-agent-runtime-availability",
        "record-agent-runtime-issue",
    ]
    command_lines = [
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent prepare-literature-extraction-context "
            f"--paper-parse-path {paper_parse_path} "
            f"--output-path {context_path} --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent prepare-pde-literature-extraction-skill "
            f"--literature-extraction-context-path {context_path} "
            f"--output-path {skill_path} "
            f"--operator-family {normalized_operator} "
            f"--target-problem {target_problem} --force"
        ),
        (
            f"UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent {operator_task_command} "
            f"--literature-extraction-context-path {context_path} "
            f"--output-path {operator_task_path} --force"
        ),
        (
            f"Read {context_path}, {skill_path}, and {operator_task_path}. "
            "Use agent reasoning to construct a compact AlgorithmCard JSON object "
            "with state_hint ALGORITHM_CARDS_EXTRACTED. Then persist it only via "
            "the controlled record-agent-algorithm-cards command below; do not use "
            "Write. Every card must cite source evidence from context units and "
            "must not invent paper claims."
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent record-agent-algorithm-cards "
            f"--output-path {agent_cards_path} "
            "--cards-json '<compact_algorithm_cards_json>' --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent validate-algorithm-cards-v2 "
            f"--algorithm-cards-path {agent_cards_path} "
            f"--output-path {cards_audit_path} --force"
        ),
        (
            "If the validation decision is rejected, run record-agent-runtime-issue "
            "with failed-command validate-algorithm-cards-v2 and stop. Do not "
            "patch source files or retry with ungrounded claims."
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent compile-algorithm-candidates "
            f"--algorithm-cards-path {agent_cards_path} "
            f"--output-path {candidates_path} --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent run-literature-algorithm-v2-pipeline "
            f"--algorithm-cards-path {agent_cards_path} "
            f"--workspace-dir {pipeline_workspace} "
            f"--registry-path {registry_path} "
            f"--paper-parse-path {paper_parse_path} "
            f"--current-best-score {current_best_score} --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent audit-agent-runtime-availability "
            f"--output-path {availability_path} --runtime-name claude_agent_sdk "
            f"--tool-catalog-path {tool_catalog_path} "
            f"--canary-report-path {output_dir / 'claude_agent_sdk_session.json'} "
            + " ".join(f"--cli-command {cmd}" for cmd in required_commands)
            + " "
            + " ".join(f"--required-command {cmd}" for cmd in required_commands)
            + " --force"
        ),
        (
            "Only if one of the required commands fails, run "
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent record-agent-runtime-issue "
            f"--output-path {execution_issue_path} "
            "--failed-command <failed_command_name> "
            "--artifact-path <related_artifact_path> "
            "--stderr-summary <short_error_summary> "
            "--likely-schema-gap/--not-schema-gap "
            "--next-action stop_after_issue_recorded --force"
        ),
    ]
    prompt = "\n".join(
        [
            "# Claude Agent SDK Literature-to-Candidate Closure",
            "",
            "You are the Claude Agent SDK runtime for pde-agent.",
            "Use only `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent ...` Bash commands.",
            "Do not patch source files. Do not run discovery or probing commands such as `pde-agent --help`, `<command> --help`, shell pipelines, or ad hoc Python.",
            "Read the runtime contract and scientific tool catalog before acting.",
            "Treat this task directory as a persistent agent workspace, not a disposable run folder. Reuse the named artifacts below with `--force`; do not create sibling run directories, timestamped rerun directories, or ad hoc staging folders.",
            "",
            f"- workspace: {workspace_dir}",
            f"- persistent task workspace: {output_dir}",
            f"- expected runtime session artifact: {session_path}",
            f"- runtime contract: {runtime_contract_path}",
            f"- tool catalog: {tool_catalog_path}",
            f"- paper parse: {paper_parse_path}",
            f"- operator family: {normalized_operator}",
            f"- target problem: {target_problem}",
            f"- literature registry: {registry_path}",
            f"- current best score: {current_best_score}",
            "",
            "## Required Command Flow",
            *[f"{index}. {line}" for index, line in enumerate(command_lines, start=1)],
            "",
            "## Agent-Authored AlgorithmCard Rules",
            f"Persist exactly one JSON file at {agent_cards_path}, using only `record-agent-algorithm-cards`.",
            "Do not use Write. The JSON object must have state_hint `ALGORITHM_CARDS_EXTRACTED` and an `algorithm_cards` list.",
            "Each card must include AlgorithmCard v2 fields required by the extraction skill, including source_ids, evidence_spans, operator_target, mechanism_type, mathematical_object, implementation_requirements, expected_observables, failure_modes, minimal_experiment, and evidence_status.",
            "When passing JSON through `--cards-json`, use compact JSON and shell-safe quoting; avoid apostrophes in free-text strings when an equivalent wording works.",
            "Ground claims in source unit IDs and evidence spans from the prepared context. If the paper lacks a testable mechanism, write an empty card list with extraction_status `empty` and failure_code `no_algorithmic_claim`.",
            "Prefer candidates that can feed the existing PDE_Burgers sandbox and evidence gates; do not write runnable code here.",
            "",
            "## Closure Rules",
            "If validation rejects the cards, record one runtime issue and stop.",
            f"If the runtime or API fails before tool execution, record the failure at {execution_issue_path} when control returns to a pde-agent command.",
            "If pipeline promotion rejects all candidates, treat that as useful evidence and stop after the availability audit.",
            "Do not continue into another literature search or sandbox loop.",
            "",
            "Return a concise final summary with generated artifact paths, state hints, validation decision, candidate count, and registry path.",
        ]
    )
    checklist = {
        "state_hint": CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_STATE_HINT,
        "runtime": {"name": "claude_agent_sdk"},
        "operator_family": normalized_operator,
        "workspace_policy": {
            "persistence": "reuse_named_task_workspace",
            "task_workspace": str(output_dir),
            "expected_session_path": str(session_path),
            "forbidden_patterns": [
                "create sibling run directories",
                "create timestamped rerun directories",
                "use ad hoc staging folders outside expected_artifacts",
            ],
        },
        "required_commands": required_commands,
        "required_state_hints": [
            "LITERATURE_EXTRACTION_CONTEXT_PREPARED",
            "PDE_LITERATURE_EXTRACTION_SKILL_PREPARED",
            {
                "fno": "FNO_LITERATURE_ADAPTATION_TASK_PREPARED",
                "deeponet": "DEEPONET_LITERATURE_ADAPTATION_TASK_PREPARED",
                "unet": "UNET_LITERATURE_ADAPTATION_TASK_PREPARED",
            }[normalized_operator],
            "ALGORITHM_CARDS_EXTRACTED",
            "ALGORITHM_CARDS_V2_VALIDATED",
            "ALGORITHM_CANDIDATES_COMPILED",
            "LITERATURE_ALGORITHM_V2_PIPELINE_RAN",
            "AGENT_RUNTIME_AVAILABILITY_AUDITED",
        ],
        "expected_artifacts": {
            "claude_agent_sdk_session_path": str(session_path),
            "literature_extraction_context_path": str(context_path),
            "pde_literature_extraction_skill_path": str(skill_path),
            "operator_literature_adaptation_task_path": str(operator_task_path),
            "agent_algorithm_cards_path": str(agent_cards_path),
            "algorithm_card_v2_audit_path": str(cards_audit_path),
            "algorithm_candidates_path": str(candidates_path),
            "literature_algorithm_v2_workspace": str(pipeline_workspace),
            "literature_registry_path": str(registry_path),
            "availability_path": str(availability_path),
        },
        "failure_artifacts": {
            "execution_issue_path": str(execution_issue_path),
        },
    }
    prompt_path.write_text(prompt, encoding="utf-8")
    _write_json(checklist_path, checklist, force=force)
    return {
        "state_hint": CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_STATE_HINT,
        "runtime": {"name": "claude_agent_sdk"},
        "artifacts": {
            "prompt_path": str(prompt_path),
            "checklist_path": str(checklist_path),
        },
        "required_commands": required_commands,
        "expected_artifacts": checklist["expected_artifacts"],
        "failure_artifacts": checklist["failure_artifacts"],
    }
