"""Agent interface: format iteration reports and build closed-loop tasks for Task 2 FNO."""

import json
from pathlib import Path
from typing import Any


STATE_HINT = "TASK2_FNO_TRAINING_ITERATION_REPORTED"
CONTRACT_STATE_HINT = "TASK2_FNO_TRAINING_CONTRACT_BUILT"


class Task2FnoTrainingAgentInterfaceError(RuntimeError):
    pass


def build_task2_fno_training_iteration_report(
    *,
    iteration_result: dict[str, Any],
) -> dict[str, Any]:
    iteration_index = iteration_result.get("iteration_index", 0)
    summary = iteration_result.get("summary", {})
    convergence = iteration_result.get("convergence", {})
    memory = iteration_result.get("memory", {})
    artifacts = iteration_result.get("artifacts", {})
    decision = convergence.get("decision", "continue")
    directives = _agent_directives(decision=decision, artifacts=artifacts)
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "iteration_index": iteration_index,
        "summary": {
            "best_candidate_id": summary.get("best_candidate_id"),
            "best_score": summary.get("best_score"),
            "score_delta_vs_previous": summary.get("score_delta_vs_previous"),
            "model_family": "fno_lite_task2",
            "config": summary.get("config"),
        },
        "convergence": {
            "decision": decision,
            "reason": convergence.get("reason"),
            "recommended_actions": convergence.get("recommended_actions", []),
        },
        "evidence_files": {
            "optimization_report": artifacts.get("optimization_report_path"),
            "memory_json": artifacts.get("memory_path"),
            "convergence_report": artifacts.get("convergence_path"),
            "checkpoint": artifacts.get("checkpoint_path"),
        },
        "agent_directives": directives,
        "next_command": "run-task2-fno-training-iteration",
    }


def build_task2_fno_training_closed_loop_contract(
    *,
    workspace_dir: str | Path,
    output_path: str | Path | None = None,
    required_commands: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    if output_file is not None and output_file.exists() and not force:
        raise Task2FnoTrainingAgentInterfaceError(
            f"output already exists: {output_file}"
        )
    workspace = Path(workspace_dir).expanduser().resolve()
    commands = required_commands or [
        "run-task2-fno-training-iteration",
        "build-task2-fno-training-contract",
        "check-task2-fno-training-convergence",
        "build-task2-fno-training-memory",
        "checkpoint-task2-fno-training-loop",
        "resume-task2-fno-training-loop",
        "escalate-task2-fno-training-to-hypothesis",
        "integrate-hypothesis-training-feedback",
    ]
    contract = {
        "state_hint": CONTRACT_STATE_HINT,
        "workspace_dir": str(workspace),
        "required_commands": commands,
        "tool_policy": {
            "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
            "disallowed_tools": ["Edit", "Write", "NotebookEdit", "WebFetch"],
            "permission_mode": "dontAsk",
        },
        "bash_policy": {
            "allowed_prefixes": [
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"
            ],
            "allowed_pde_agent_commands": commands,
        },
        "agent_guidance": (
            "Use the pde-agent CLI as the execution surface. "
            "After each iteration, read the iteration report JSON, "
            "check the convergence decision, and decide whether to continue, "
            "escalate to hypothesis lab, or stop. Do not patch project files."
        ),
        "checklist": [
            "Build contract: build-task2-fno-training-contract",
            "Run iteration: run-task2-fno-training-iteration --workspace-dir <dir>",
            "Read report: Read task2_fno_iteration_report.json",
            "Decide: continue / escalate / stop based on convergence.decision",
            "If escalate: run escalate-task2-fno-training-to-hypothesis",
            "If stop: report best candidate artifacts",
        ],
    }
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(contract, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return contract


def build_claude_task2_fno_training_closed_loop_task(
    *,
    workspace_dir: str | Path,
    output_dir: str | Path | None = None,
    max_iterations: int = 10,
    target_score: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace_dir).expanduser().resolve()
    out_dir = Path(output_dir) if output_dir is not None else workspace / "agent_task"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "claude_agent_prompt.md"
    contract_path = out_dir / "task2_fno_training_closed_loop_contract.json"
    tool_catalog_path = out_dir / "task2_fno_training_tool_catalog.json"
    if prompt_path.exists() and not force:
        raise Task2FnoTrainingAgentInterfaceError(f"prompt already exists: {prompt_path}")
    required_commands = [
        "run-task2-fno-training-iteration",
        "build-task2-fno-training-contract",
        "check-task2-fno-training-convergence",
        "build-task2-fno-training-memory",
        "checkpoint-task2-fno-training-loop",
        "resume-task2-fno-training-loop",
        "escalate-task2-fno-training-to-hypothesis",
        "integrate-hypothesis-training-feedback",
    ]
    contract = build_task2_fno_training_closed_loop_contract(
        workspace_dir=workspace,
        output_path=contract_path,
        required_commands=required_commands,
        force=force,
    )
    from pde_agent.core.task2_fno_training_tool_catalog import (
        build_task2_fno_training_tool_catalog,
        write_tool_catalog,
    )

    catalog = build_task2_fno_training_tool_catalog()
    write_tool_catalog(catalog, tool_catalog_path)
    target_clause = f"Target score: {target_score}. " if target_score is not None else ""
    prompt = f"""# Task 2 FNO Training Closed Loop Task

You are an autonomous training optimization agent. Your goal is to iteratively train and evaluate FNO (Fourier Neural Operator) models for Task 2 (Burgers' equation with varying viscosity), improving the task2_proxy score over multiple iterations.

## Workspace
- Workspace directory: {workspace}
- Model family: fno_lite_task2 (single family, no selection needed)
- Max iterations: {max_iterations}
- {target_clause}Stop when converged or max iterations reached.

## Task 2 Hyperparameters
- hidden_channels: width of hidden layers
- modes: number of Fourier modes
- depth: number of spectral layers
- physics_weight: weight for Burgers residual loss
- learning_rate: Adam learning rate
- train_steps: total training steps
- batch_size: training batch size

## Score Metric
- task2_proxy = segment_score * 1.5
- Higher is better. Target depends on competition baseline.

## Available Commands
You may only use commands from the pde-agent CLI with this exact bash prefix:
```
UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent <command>
```

Key commands:
- `run-task2-fno-training-iteration --workspace-dir {workspace}` — Run one complete training iteration
- `build-task2-fno-training-contract --workspace-dir {workspace}` — Build the runtime contract
- `checkpoint-task2-fno-training-loop --workspace-dir {workspace} --iteration-index N --memory-path ... --convergence-path ...` — Save checkpoint

## Loop Protocol

### Per Iteration
1. Run: `run-task2-fno-training-iteration --workspace-dir {workspace} [--memory-path <previous_memory>]`
2. Read the returned `task2_fno_iteration_report.json`
3. Look at `convergence.decision`:

### Decision Rules
- **`convergence.decision == "continue"`**
  -> Run the next iteration immediately. Pass the `--memory-path` from the report so the next iteration learns from this one.

- **`convergence.decision == "converged"`**
  -> Stop. Read the best candidate artifacts and report the final best score, config, and artifact paths.

- **`convergence.decision == "stagnant"`**
  -> You may either:
    a) Try one more iteration with expanded search (the report will suggest parameters), OR
    b) Escalate to hypothesis lab for new strategy discovery.

- **`convergence.decision == "doom_loop"`**
  -> Escalate immediately. Run `escalate-task2-fno-training-to-hypothesis`, then after it completes run `integrate-hypothesis-training-feedback`, then resume training.

- **`convergence.decision == "escalate"`**
  -> Run `escalate-task2-fno-training-to-hypothesis --workspace-dir {workspace} --memory-path <path>`, then `integrate-hypothesis-training-feedback`, then continue.

### Checkpointing
Every 2 iterations, run `checkpoint-task2-fno-training-loop` with the current memory and convergence paths. This saves state in case you need to resume.

### Constraints
- Do NOT edit project files.
- Do NOT invent commands outside the allowed pde-agent prefix.
- Do NOT download files or search the web.
- Only use: Read, Glob, Grep, Bash.
- In Bash, only use the exact `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent <command>` prefix.

### Getting Started
1. Build the contract: `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent build-task2-fno-training-contract --workspace-dir {workspace}`
2. Run the first iteration: `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent run-task2-fno-training-iteration --workspace-dir {workspace} --max-candidates 2`
3. Read the report and follow the decision rules above.
"""
    prompt_path.write_text(prompt, encoding="utf-8")
    return {
        "schema_version": 1,
        "state_hint": "CLAUDE_TASK2_FNO_TRAINING_CLOSED_LOOP_TASK_BUILT",
        "prompt": prompt,
        "required_commands": required_commands,
        "artifacts": {
            "contract_path": str(contract_path),
            "prompt_path": str(prompt_path),
            "tool_catalog_path": str(tool_catalog_path),
        },
        "max_iterations": max_iterations,
        "target_score": target_score,
    }


def _agent_directives(
    *,
    decision: str,
    artifacts: dict[str, Any],
) -> list[str]:
    directives = []
    if artifacts.get("memory_path"):
        directives.append(
            f"Read {artifacts['memory_path']} to understand search guidance."
        )
    if decision == "continue":
        directives.append(
            "Run next iteration with: run-task2-fno-training-iteration --workspace-dir <dir>"
        )
    elif decision == "escalate":
        directives.append(
            "Escalate with: escalate-task2-fno-training-to-hypothesis --workspace-dir <dir>"
        )
    elif decision in {"converged", "stagnant", "doom_loop"}:
        directives.append(
            "Review best candidate artifacts and report final results."
        )
    else:
        directives.append(
            "Run next iteration with: run-task2-fno-training-iteration --workspace-dir <dir>"
        )
    return directives
