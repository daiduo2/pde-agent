from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.agent_runtime_shared import (
    CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_STATE_HINT,
    _write_json,
)


def build_claude_hypothesis_lab_closure_task(
    *,
    output_dir: Path,
    workspace_dir: Path,
    runtime_contract_path: Path,
    tool_catalog_path: Path,
    candidate_family_sandbox_path: Path,
    context_path: Path,
    target_path: Path,
    reference_path: Path | None,
    force: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    workspace_dir = workspace_dir.expanduser().resolve()
    runtime_contract_path = runtime_contract_path.expanduser().resolve()
    tool_catalog_path = tool_catalog_path.expanduser().resolve()
    candidate_family_sandbox_path = candidate_family_sandbox_path.expanduser().resolve()
    context_path = context_path.expanduser().resolve()
    target_path = target_path.expanduser().resolve()
    reference_path = reference_path.expanduser().resolve() if reference_path else None
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = output_dir / "claude_agent_prompt.md"
    checklist_path = output_dir / "expected_artifacts.json"
    materialized_candidates_path = output_dir / "algorithm_candidates_materialized.json"
    hypothesis_lab_dir = output_dir / "hypothesis_lab_run"
    failure_memory_path = output_dir / "failure_memory.json"
    next_policy_path = output_dir / "hypothesis_lab_next_policy.json"
    candidate_plan_path = output_dir / "hypothesis_candidate_plan.json"
    availability_path = output_dir / "runtime_availability.json"
    execution_issue_path = output_dir / "closure_execution_issue.json"
    trace_path = output_dir / "hypothesis_lab_trace.jsonl"
    hypothesis_lab_report_path = hypothesis_lab_dir / "hypothesis_lab_report.json"

    required_commands = [
        "materialize-candidate-family-item",
        "run-hypothesis-lab",
        "build-hypothesis-lab-next-policy",
        "plan-hypothesis-candidates",
        "audit-agent-runtime-availability",
        "record-agent-runtime-issue",
    ]
    reference_arg = f"--reference-path {reference_path} " if reference_path else ""
    command_lines = [
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent materialize-candidate-family-item "
            f"--sandbox-report-path {candidate_family_sandbox_path} "
            f"--output-path {materialized_candidates_path} --force"
        ),
        (
            "Inspect the materialized candidates and choose the candidate-family "
            "item named by the artifact. Do not invent a candidate outside the "
            "sandbox report."
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent run-hypothesis-lab "
            f"--algorithm-candidates-path {materialized_candidates_path} "
            f"--context-path {context_path} "
            f"--target-path {target_path} "
            f"{reference_arg}"
            f"--output-dir {hypothesis_lab_dir} "
            f"--failure-memory-output-path {failure_memory_path} "
            f"--trace-path {trace_path} "
            "--submission-id claude_hypothesis_lab_closure --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent build-hypothesis-lab-next-policy "
            f"--failure-memory-path {failure_memory_path} "
            f"--hypothesis-lab-report-path {hypothesis_lab_report_path} "
            f"--output-path {next_policy_path} --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent plan-hypothesis-candidates "
            f"--output-path {candidate_plan_path} "
            f"--hypothesis-lab-next-policy-path {next_policy_path} "
            "--min-viable-candidate-count 1 --force"
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
            "# Claude Agent SDK Hypothesis Lab Closure",
            "",
            "You are the Claude Agent SDK runtime for pde-agent.",
            "Use only `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent ...` Bash commands.",
            "Do not patch source files. Do not create directories manually.",
            "Do not run discovery or probing commands such as `pde-agent --help`, `<command> --help`, score-local-rollout, shell pipelines, or ad hoc Python.",
            "Read the runtime contract, scientific tool catalog, and candidate-family sandbox before acting.",
            "",
            f"- workspace: {workspace_dir}",
            f"- runtime contract: {runtime_contract_path}",
            f"- tool catalog: {tool_catalog_path}",
            f"- candidate-family sandbox: {candidate_family_sandbox_path}",
            f"- context hdf5: {context_path}",
            f"- target hdf5: {target_path}",
            f"- reference hdf5: {reference_path}" if reference_path else "- reference hdf5: none",
            "",
            "## Required Command Flow",
            *[f"{index}. {line}" for index, line in enumerate(command_lines, start=1)],
            "",
            "## Closure Rules",
            "Treat negative sandbox evidence as useful learning, not as a reason to retry the same family.",
            "If a command fails because an artifact is missing or schema-incompatible, do not use Write. Call `record-agent-runtime-issue` once and stop.",
            "If `audit-agent-runtime-availability` fails, call `record-agent-runtime-issue` once and stop; do not try help commands or alternate diagnostics.",
            "Stop after building the next policy, next candidate plan, and availability audit, or after recording one runtime issue. Do not continue into another loop.",
            "",
            "Return a concise final summary with generated artifact paths, state hints, and the next policy action.",
        ]
    )
    checklist = {
        "state_hint": CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_STATE_HINT,
        "runtime": {"name": "claude_agent_sdk"},
        "required_commands": required_commands,
        "required_state_hints": [
            "ALGORITHM_CANDIDATES_COMPILED",
            "HYPOTHESIS_LAB_RAN",
            "HYPOTHESIS_LAB_NEXT_POLICY_BUILT",
            "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
            "AGENT_RUNTIME_AVAILABILITY_AUDITED",
        ],
        "expected_artifacts": {
            "materialized_candidates_path": str(materialized_candidates_path),
            "hypothesis_lab_report_path": str(hypothesis_lab_report_path),
            "failure_memory_path": str(failure_memory_path),
            "trace_path": str(trace_path),
            "next_policy_path": str(next_policy_path),
            "candidate_plan_path": str(candidate_plan_path),
            "availability_path": str(availability_path),
        },
        "failure_artifacts": {
            "execution_issue_path": str(execution_issue_path),
        },
    }
    prompt_path.write_text(prompt, encoding="utf-8")
    _write_json(checklist_path, checklist, force=force)
    return {
        "state_hint": CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_STATE_HINT,
        "runtime": {"name": "claude_agent_sdk"},
        "artifacts": {
            "prompt_path": str(prompt_path),
            "checklist_path": str(checklist_path),
        },
        "required_commands": required_commands,
        "expected_artifacts": checklist["expected_artifacts"],
        "failure_artifacts": checklist["failure_artifacts"],
    }
