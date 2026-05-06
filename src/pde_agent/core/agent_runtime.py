from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any


CONTRACT_STATE_HINT = "AGENT_RUNTIME_CONTRACT_BUILT"
CLAUDE_RUNTIME_STATE_HINT = "CLAUDE_AGENT_SDK_RUNTIME_RAN"
CLAUDE_FNO_TRAINING_SMOKE_TASK_STATE_HINT = "CLAUDE_FNO_TRAINING_SMOKE_TASK_BUILT"
CLAUDE_PROMOTION_SMOKE_TASK_STATE_HINT = "CLAUDE_PROMOTION_SMOKE_TASK_BUILT"


class AgentRuntimeError(RuntimeError):
    pass


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise AgentRuntimeError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_agent_runtime_contract(
    *,
    runtime_name: str,
    output_path: Path,
    workspace_dir: Path,
    tool_catalog_path: Path,
    required_commands: list[str],
    force: bool,
) -> dict[str, Any]:
    if runtime_name != "claude_agent_sdk":
        raise AgentRuntimeError(f"unsupported runtime: {runtime_name}")
    workspace_dir = workspace_dir.expanduser().resolve()
    tool_catalog_path = tool_catalog_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    contract: dict[str, Any] = {
        "state_hint": CONTRACT_STATE_HINT,
        "runtime": {
            "name": "claude_agent_sdk",
            "primary": True,
            "compatibility_backends": ["hermes"],
            "sdk_package": "claude-agent-sdk",
            "sdk_import": "claude_agent_sdk",
        },
        "workspace_dir": str(workspace_dir),
        "tool_catalog_path": str(tool_catalog_path),
        "required_commands": list(required_commands),
        "tool_policy": {
            "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
            "disallowed_tools": ["Edit", "Write", "NotebookEdit", "WebFetch"],
            "permission_mode": "dontAsk",
            "bash_guard_hook": "enabled",
        },
        "bash_policy": {
            "allowed_prefixes": ["uv run --frozen pde-agent"],
            "forbidden_patterns": [
                "mkdir ",
                "rm ",
                "python -c",
                "python3 -c",
                "pip install",
                "uv add",
            ],
        },
        "agent_guidance": (
            "Use the scientific tool catalog and pde-agent CLI artifacts as the "
            "execution surface. Do not patch project files or invent shell "
            "commands outside the allowed pde-agent prefix."
        ),
    }
    _write_json(output_path, contract, force=force)
    return contract


def build_bash_prefix_guard_hook(
    allowed_bash_prefixes: list[str],
) -> Any:
    async def guard_hook(
        input_data: dict[str, Any],
        tool_use_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        _ = tool_use_id, context
        tool_input = input_data.get("tool_input", {})
        command = tool_input.get("command") if isinstance(tool_input, dict) else None
        if not isinstance(command, str):
            return {}
        if any(command.startswith(prefix) for prefix in allowed_bash_prefixes):
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Bash command is outside allowed pde-agent prefixes."
                ),
            }
        }

    return guard_hook


def _import_claude_agent_sdk() -> Any:
    try:
        return importlib.import_module("claude_agent_sdk")
    except ModuleNotFoundError as exc:
        raise AgentRuntimeError(
            "Claude Agent SDK is not installed. Install optional package "
            "`claude-agent-sdk` so import `claude_agent_sdk` is available."
        ) from exc


async def _collect_sdk_messages(
    *,
    sdk_module: Any,
    prompt: str,
    workspace_dir: Path,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    allowed_bash_prefixes: list[str],
) -> list[str]:
    options_cls = getattr(sdk_module, "ClaudeAgentOptions")
    hook_matcher_cls = getattr(sdk_module, "HookMatcher", None)
    query = getattr(sdk_module, "query")
    hooks = {}
    if hook_matcher_cls is not None:
        hooks = {
            "PreToolUse": [
                hook_matcher_cls(
                    matcher="Bash",
                    hooks=[build_bash_prefix_guard_hook(allowed_bash_prefixes)],
                )
            ]
        }
    options = options_cls(
        cwd=str(workspace_dir),
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        permission_mode="dontAsk",
        hooks=hooks,
    )
    messages = []
    async for message in query(prompt=prompt, options=options):
        messages.append(str(message))
    return messages


def run_claude_agent_sdk_runtime(
    *,
    prompt: str,
    output_path: Path,
    workspace_dir: Path,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    allowed_bash_prefixes: list[str],
    dry_run: bool,
    force: bool,
    sdk_module: Any | None = "auto",
) -> dict[str, Any]:
    output_path = output_path.expanduser().resolve()
    workspace_dir = workspace_dir.expanduser().resolve()
    if dry_run:
        sdk_messages: list[str] = []
        status = "dry_run"
    else:
        module = _import_claude_agent_sdk() if sdk_module == "auto" else sdk_module
        if module is None:
            raise AgentRuntimeError(
                "Claude Agent SDK is not installed. Install optional package "
                "`claude-agent-sdk` so import `claude_agent_sdk` is available."
            )
        sdk_messages = asyncio.run(
            _collect_sdk_messages(
                sdk_module=module,
                prompt=prompt,
                workspace_dir=workspace_dir,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                allowed_bash_prefixes=allowed_bash_prefixes,
            )
        )
        status = "completed"

    session: dict[str, Any] = {
        "state_hint": CLAUDE_RUNTIME_STATE_HINT,
        "status": status,
        "runtime": {
            "name": "claude_agent_sdk",
            "sdk_package": "claude-agent-sdk",
            "sdk_import": "claude_agent_sdk",
        },
        "workspace_dir": str(workspace_dir),
        "prompt_preview": prompt[:200],
        "tool_policy": {
            "allowed_tools": list(allowed_tools),
            "disallowed_tools": list(disallowed_tools),
            "permission_mode": "dontAsk",
            "bash_guard_hook": "enabled",
        },
        "bash_policy": {
            "allowed_prefixes": list(allowed_bash_prefixes),
        },
        "sdk_message_count": len(sdk_messages),
        "sdk_messages": sdk_messages,
        "artifacts": {
            "session_path": str(output_path),
        },
    }
    _write_json(output_path, session, force=force)
    return session


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
            "uv run --frozen pde-agent build-fno-training-harness-contract "
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
            "uv run --frozen pde-agent record-fno-training-optimization-evidence "
            f"--optimization-report-path {optimization_dir / 'fno_training_optimization_report.json'} "
            f"--output-path {optimization_evidence_path} --force"
        ),
        (
            "uv run --frozen pde-agent build-hypothesis-algorithm-design "
            f"--output-path {algorithm_design_path} {explicit_evidence_args} "
            "--min-hypothesis-count 5 --min-mechanism-family-count 4 --force"
        ),
        (
            "uv run --frozen pde-agent validate-hypothesis-algorithm-design "
            f"--design-path {algorithm_design_path} "
            f"--output-path {algorithm_validation_path} --force"
        ),
        (
            "If validation accepts a promoted or promising candidate, run "
            "record-promoted-algorithm-strategy-proposal and then "
            "convert-candidate-promotion-to-strategy-evidence. If no candidate "
            "is promotable, write the validation artifact and stop."
        ),
        (
            "uv run --frozen pde-agent audit-agent-runtime-availability "
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
            "Use only `uv run --frozen pde-agent ...` Bash commands.",
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
    }


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
            "uv run --frozen pde-agent record-promoted-algorithm-strategy-proposal "
            f"--candidate-promotion-path {promotion_path} "
            f"--output-path {strategy_proposal_path} --force"
        ),
        (
            "uv run --frozen pde-agent convert-candidate-promotion-to-strategy-evidence "
            f"--candidate-promotion-path {promotion_path} "
            f"--output-path {strategy_evidence_path} --force"
        ),
        (
            "uv run --frozen pde-agent audit-agent-runtime-availability "
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
            "Use only `uv run --frozen pde-agent ...` Bash commands.",
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
