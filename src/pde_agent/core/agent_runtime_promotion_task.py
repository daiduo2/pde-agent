from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.agent_runtime_shared import (
    CLAUDE_PROMOTION_SMOKE_TASK_STATE_HINT,
    _write_json,
)


def build_claude_promotion_smoke_task(
    *,
    output_dir: Path,
    workspace_dir: Path,
    runtime_contract_path: Path,
    tool_catalog_path: Path,
    force: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    workspace_dir = workspace_dir.expanduser().resolve()
    runtime_contract_path = runtime_contract_path.expanduser().resolve()
    tool_catalog_path = tool_catalog_path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "claude_agent_prompt.md"
    checklist_path = output_dir / "expected_artifacts.json"
    candidates_path = output_dir / "selected_algorithm_candidates.json"
    evidence_path = output_dir / "sandbox_evidence.json"
    promotion_path = output_dir / "candidate_promotion.json"
    strategy_proposal_path = output_dir / "strategy_proposal.json"
    strategy_evidence_path = output_dir / "strategy_evidence_summary.json"
    availability_path = output_dir / "runtime_availability.json"
    writeback_failure_path = output_dir / "writeback_failure_report.json"
    candidate_id = "candidate_claude_sdk_promotion_fixture_v0"
    required_commands = [
        "record-promoted-algorithm-strategy-proposal",
        "convert-candidate-promotion-to-strategy-evidence",
        "audit-agent-runtime-availability",
    ]

    candidates_payload: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "algorithm_candidates_path": str(candidates_path),
        "algorithm_candidate_count": 1,
        "algorithm_candidates": [
            {
                "algorithm_candidate_id": candidate_id,
                "parent_algorithm_card_id": "claude_sdk_promotion_fixture_v0",
                "candidate_family": "runtime_migration_fixture",
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "hypothesis": (
                    "A Claude SDK-driven promotion branch can preserve sandbox "
                    "evidence and emit strategy-growth artifacts."
                ),
                "mechanism": (
                    "Convert a promoted candidate artifact into a strategy "
                    "proposal and rubric-compatible evidence summary."
                ),
                "executor_spec": {
                    "cli_command": "run-fno-training-optimization-loop",
                    "implementation_hooks": ["claude_sdk_promotion_fixture"],
                },
                "risk_controls": [
                    "fixture evidence is not a model-quality claim",
                    "downstream rubric must still evaluate evidence depth",
                ],
                "source_grounding": {
                    "algorithm_card_id": "claude_sdk_promotion_fixture_v0",
                    "source_type": "local_runtime_fixture",
                    "source_ids": ["claude_agent_sdk_promotion_smoke"],
                    "title": "Claude SDK promotion fixture",
                },
            }
        ],
    }
    evidence_payload: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": "SANDBOX_EVIDENCE_RECORDED",
        "evidence_id": "evidence_claude_sdk_promotion_fixture_v0",
        "candidate_id": candidate_id,
        "rules_passed": True,
        "score": 128.77,
        "score_delta_vs_current_best": 10.38,
        "runtime": {"inference_seconds": 0.7},
        "physics_metrics": {"residual_delta": 0.0},
    }
    promotion_payload: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED",
        "algorithm_candidate_promotion_path": str(promotion_path),
        "algorithm_candidates_path": str(candidates_path),
        "algorithm_candidate_id": candidate_id,
        "candidate_family": "runtime_migration_fixture",
        "sandbox_evidence_paths": [str(evidence_path)],
        "evidence_count": 1,
        "decision": "promote",
        "promote_to_strategy_growth": True,
        "reason_codes": ["all_rubric_checks_passed"],
        "metric_results": {
            "score": 128.77,
            "current_best_score": 118.39,
            "score_delta_vs_current_best": 10.38,
            "inference_seconds": 0.7,
            "physics_residual_delta": 0.0,
            "rules_passed": True,
        },
    }
    _write_json(candidates_path, candidates_payload, force=force)
    _write_json(evidence_path, evidence_payload, force=force)
    _write_json(promotion_path, promotion_payload, force=force)

    command_lines = [
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent record-promoted-algorithm-strategy-proposal "
            f"--candidate-promotion-path {promotion_path} "
            f"--output-path {strategy_proposal_path} --force"
        ),
        (
            "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent convert-candidate-promotion-to-strategy-evidence "
            f"--candidate-promotion-path {promotion_path} "
            f"--output-path {strategy_evidence_path} --force"
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
            "# Claude Agent SDK PDEAgent Promotion Smoke",
            "",
            "You are the Claude Agent SDK runtime for pde-agent.",
            "Use only `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent ...` Bash commands.",
            "Do not patch source files. Do not create directories manually.",
            "Read the runtime contract and scientific tool catalog before acting.",
            "",
            f"- workspace: {workspace_dir}",
            f"- runtime contract: {runtime_contract_path}",
            f"- tool catalog: {tool_catalog_path}",
            f"- candidate promotion fixture: {promotion_path}",
            "",
            "## Required Command Flow",
            *[f"{index}. {line}" for index, line in enumerate(command_lines, start=1)],
            "",
            "## Writeback Failure Protocol",
            (
                "If a promotion writeback command fails, write "
                f"{writeback_failure_path} as JSON with failed_command, "
                "stderr_summary, missing_schema_field if visible, schema_gap, "
                "and next_action. Treat missing required candidate fields as a "
                "schema gap. Do not keep retrying or wait indefinitely."
            ),
            "",
            "Return a concise final summary with generated artifact paths and state hints.",
        ]
    )
    checklist = {
        "state_hint": CLAUDE_PROMOTION_SMOKE_TASK_STATE_HINT,
        "runtime": {"name": "claude_agent_sdk"},
        "required_commands": required_commands,
        "required_state_hints": [
            "STRATEGY_PROPOSAL_RECORDED",
            "STRATEGY_EVIDENCE_SUMMARY_BUILT",
            "AGENT_RUNTIME_AVAILABILITY_AUDITED",
        ],
        "fixture_artifacts": {
            "candidates_path": str(candidates_path),
            "evidence_path": str(evidence_path),
            "candidate_promotion_path": str(promotion_path),
        },
        "failure_artifacts": {
            "writeback_failure_report_path": str(writeback_failure_path),
        },
        "expected_artifacts": {
            "strategy_proposal_path": str(strategy_proposal_path),
            "strategy_evidence_path": str(strategy_evidence_path),
            "availability_path": str(availability_path),
        },
    }
    prompt_path.write_text(prompt, encoding="utf-8")
    _write_json(checklist_path, checklist, force=force)
    return {
        "state_hint": CLAUDE_PROMOTION_SMOKE_TASK_STATE_HINT,
        "runtime": {"name": "claude_agent_sdk"},
        "artifacts": {
            "prompt_path": str(prompt_path),
            "checklist_path": str(checklist_path),
            "candidate_promotion_path": str(promotion_path),
        },
        "required_commands": required_commands,
        "expected_artifacts": checklist["expected_artifacts"],
    }
