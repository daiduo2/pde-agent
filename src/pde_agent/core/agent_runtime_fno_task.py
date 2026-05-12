from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.agent_runtime_shared import (
    CLAUDE_FNO_TRAINING_SMOKE_TASK_STATE_HINT,
    _write_json,
)


def build_claude_fno_training_smoke_task(
    *,
    output_dir: Path,
    workspace_dir: Path,
    runtime_contract_path: Path,
    tool_catalog_path: Path,
    source_path: Path,
    previous_evidence_paths: list[Path],
    force: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    workspace_dir = workspace_dir.expanduser().resolve()
    runtime_contract_path = runtime_contract_path.expanduser().resolve()
    tool_catalog_path = tool_catalog_path.expanduser().resolve()
    source_path = source_path.expanduser().resolve()
    previous_evidence_paths = [
        path.expanduser().resolve() for path in previous_evidence_paths
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "claude_agent_prompt.md"
    checklist_path = output_dir / "expected_artifacts.json"
    harness_contract_path = output_dir / "fno_training_harness_contract.json"
    optimization_dir = output_dir / "fno_training_optimization"
    optimization_evidence_path = output_dir / "fno_training_optimization_evidence.json"
    algorithm_design_path = output_dir / "hypothesis_algorithm_design.json"
    algorithm_validation_path = output_dir / "hypothesis_algorithm_validation.json"
    strategy_proposal_path = output_dir / "strategy_proposal.json"
    strategy_evidence_path = output_dir / "strategy_evidence_summary.json"
    availability_path = output_dir / "runtime_availability.json"
    writeback_failure_path = output_dir / "writeback_failure_report.json"

    previous_evidence_args = " ".join(
        f"--previous-evidence-path {path}" for path in previous_evidence_paths
    )
    explicit_evidence_args = " ".join(
        f"--evidence-path {path}"
        for path in [optimization_evidence_path, *previous_evidence_paths]
    )
    required_commands = [
        "build-fno-training-harness-contract",
        "run-fno-training-optimization-loop",
        "record-fno-training-optimization-evidence",
        "build-hypothesis-algorithm-design",
        "validate-hypothesis-algorithm-design",
        "record-promoted-algorithm-strategy-proposal",
        "convert-candidate-promotion-to-strategy-evidence",
        "audit-agent-runtime-availability",
    ]
    command_lines = [
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent build-fno-training-harness-contract "
            f"--output-path {harness_contract_path} "
            f"--workspace-dir {output_dir} "
            f"--source-path {source_path} "
            f"{previous_evidence_args} "
            "--max-candidates 2 --train-sample-limit 8 --val-sample-limit 4 "
            "--batch-size 2 --force"
        ),
        (
            "Select the run-fno-training-optimization-loop action from "
            f"{harness_contract_path}; use the candidate_plan_path and budget "
            "from that contract, not guessed CLI flags."
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent record-fno-training-optimization-evidence "
            f"--optimization-report-path {optimization_dir / 'fno_training_optimization_report.json'} "
            f"--output-path {optimization_evidence_path} --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent build-hypothesis-algorithm-design "
            f"--output-path {algorithm_design_path} {explicit_evidence_args} "
            "--min-hypothesis-count 5 --min-mechanism-family-count 4 --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent validate-hypothesis-algorithm-design "
            f"--design-path {algorithm_design_path} "
            f"--output-path {algorithm_validation_path} --force"
        ),
        (
            "If validation accepts a promoted or promising candidate, run "
            "record-promoted-algorithm-strategy-proposal and then "
            "convert-candidate-promotion-to-strategy-evidence. If no candidate "
            "is promotable, write the validation artifact and stop. If either "
            "writeback command fails, write a JSON failure report to "
            f"{writeback_failure_path} with the failed command, stderr summary, "
            "missing schema field if visible, and whether this is a schema gap; "
            "do not keep retrying or wait indefinitely."
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
    ]
    prompt = "\n".join(
        [
            "# Claude Agent SDK PDEAgent FNO Training Smoke",
            "",
            "You are the Claude Agent SDK runtime for pde-agent.",
            "Use only `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent ...` Bash commands.",
            "Do not patch source files. Do not create directories manually.",
            "Read the runtime contract and scientific tool catalog before acting.",
            "",
            f"- workspace: {workspace_dir}",
            f"- runtime contract: {runtime_contract_path}",
            f"- tool catalog: {tool_catalog_path}",
            f"- source data: {source_path}",
            "",
            "## Required Command Flow",
            *[f"{index}. {line}" for index, line in enumerate(command_lines, start=1)],
            "",
            "## Writeback Failure Protocol",
            (
                "If a promotion writeback command fails, write "
                f"{writeback_failure_path} as JSON with failed_command, "
                "stderr_summary, missing_schema_field if visible, schema_gap, "
                "and next_action; do not keep retrying or wait indefinitely."
            ),
            "",
            "Return a concise final summary with generated artifact paths and state hints.",
        ]
    )
    checklist = {
        "state_hint": CLAUDE_FNO_TRAINING_SMOKE_TASK_STATE_HINT,
        "runtime": {"name": "claude_agent_sdk"},
        "required_commands": required_commands,
        "required_state_hints": [
            "FNO_TRAINING_HARNESS_CONTRACT_BUILT",
            "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
            "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
            "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
            "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED",
            "STRATEGY_PROPOSAL_RECORDED",
            "STRATEGY_EVIDENCE_SUMMARY_BUILT",
            "AGENT_RUNTIME_AVAILABILITY_AUDITED",
        ],
        "expected_artifacts": {
            "harness_contract_path": str(harness_contract_path),
            "optimization_dir": str(optimization_dir),
            "optimization_evidence_path": str(optimization_evidence_path),
            "algorithm_design_path": str(algorithm_design_path),
            "algorithm_validation_path": str(algorithm_validation_path),
            "strategy_proposal_path": str(strategy_proposal_path),
            "strategy_evidence_path": str(strategy_evidence_path),
            "availability_path": str(availability_path),
        },
        "failure_artifacts": {
            "writeback_failure_report_path": str(writeback_failure_path),
        },
    }
    prompt_path.write_text(prompt, encoding="utf-8")
    _write_json(checklist_path, checklist, force=force)
    return {
        "state_hint": CLAUDE_FNO_TRAINING_SMOKE_TASK_STATE_HINT,
        "runtime": {"name": "claude_agent_sdk"},
        "artifacts": {
            "prompt_path": str(prompt_path),
            "checklist_path": str(checklist_path),
        },
        "required_commands": required_commands,
        "expected_artifacts": checklist["expected_artifacts"],
        "failure_artifacts": checklist["failure_artifacts"],
    }
