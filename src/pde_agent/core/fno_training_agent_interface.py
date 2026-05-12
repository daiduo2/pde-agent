"""Agent interface: format iteration reports and build closed-loop tasks."""

import json
from pathlib import Path
from typing import Any


STATE_HINT = "FNO_TRAINING_ITERATION_REPORTED"
CONTRACT_STATE_HINT = "FNO_TRAINING_CONTRACT_BUILT"


class FnoTrainingAgentInterfaceError(RuntimeError):
    pass


def build_fno_training_iteration_report(
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
            "model_family": summary.get("model_family"),
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
        "next_command": "run-fno-training-iteration",
    }


def build_fno_training_closed_loop_contract(
    *,
    workspace_dir: str | Path,
    source_path: str | Path,
    output_path: str | Path | None = None,
    dataset_key: str = "tensor",
    baseline_prediction_path: str | Path | None = None,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    required_commands: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    if output_file is not None and output_file.exists() and not force:
        raise FnoTrainingAgentInterfaceError(
            f"output already exists: {output_file}"
        )
    workspace = Path(workspace_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    commands = required_commands or [
        "run-fno-training-iteration",
        "build-fno-training-closed-loop-contract",
        "check-fno-training-convergence",
        "build-fno-training-memory",
        "checkpoint-fno-training-loop",
        "resume-fno-training-loop",
        "escalate-fno-training-to-hypothesis",
        "integrate-hypothesis-training-feedback",
    ]
    contract = {
        "state_hint": CONTRACT_STATE_HINT,
        "workspace_dir": str(workspace),
        "source_path": str(source),
        "dataset_key": dataset_key,
        "baseline_prediction_path": str(baseline_prediction_path) if baseline_prediction_path else None,
        "task1_test_path": str(task1_test_path) if task1_test_path else None,
        "task1_target_path": str(task1_target_path) if task1_target_path else None,
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
            "Build contract: build-fno-training-closed-loop-contract",
            "Run iteration: run-fno-training-iteration --workspace-dir <dir>",
            "Read report: Read iteration_report.json",
            "Decide: continue / escalate / stop based on convergence.decision",
            "If escalate: run escalate-fno-training-to-hypothesis",
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


def build_claude_fno_training_closed_loop_task(
    *,
    workspace_dir: str | Path,
    source_path: str | Path,
    output_dir: str | Path | None = None,
    dataset_key: str = "tensor",
    max_iterations: int = 10,
    target_score: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    out_dir = Path(output_dir) if output_dir is not None else workspace / "agent_task"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "claude_agent_prompt.md"
    contract_path = out_dir / "fno_training_closed_loop_contract.json"
    tool_catalog_path = out_dir / "fno_training_tool_catalog.json"
    if prompt_path.exists() and not force:
        raise FnoTrainingAgentInterfaceError(f"prompt already exists: {prompt_path}")
    required_commands = [
        "run-fno-training-iteration",
        "build-fno-training-closed-loop-contract",
        "check-fno-training-convergence",
        "build-fno-training-memory",
        "checkpoint-fno-training-loop",
        "resume-fno-training-loop",
        "escalate-fno-training-to-hypothesis",
        "integrate-hypothesis-training-feedback",
    ]
    contract = build_fno_training_closed_loop_contract(
        workspace_dir=workspace,
        source_path=source,
        output_path=contract_path,
        dataset_key=dataset_key,
        required_commands=required_commands,
        force=force,
    )
    from pde_agent.core.fno_training_tool_catalog import (
        build_fno_training_tool_catalog,
        write_tool_catalog,
    )

    catalog = build_fno_training_tool_catalog()
    write_tool_catalog(catalog, tool_catalog_path)
    target_clause = f"Target score: {target_score}. " if target_score is not None else ""
    prompt = f"""# FNO Training Closed Loop Task

You are an autonomous training optimization agent. Your goal is to iteratively train and evaluate FNO (Fourier Neural Operator) models for Burgers' equation prediction, improving the score over multiple iterations.

## Workspace
- Workspace directory: {workspace}
- Source data: {source}
- Dataset key: {dataset_key}
- Max iterations: {max_iterations}
- {target_clause}Stop when converged or max iterations reached.

## Available Commands
You may only use commands from the pde-agent CLI with this exact bash prefix:
```
UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent <command>
```

Key commands:
- `run-fno-training-iteration --workspace-dir {workspace} --source-path {source}` — Run one complete training iteration
- `build-fno-training-closed-loop-contract --workspace-dir {workspace} --source-path {source}` — Build the runtime contract
- `checkpoint-fno-training-loop --workspace-dir {workspace} --iteration-index N --memory-path ... --convergence-path ...` — Save checkpoint

## Loop Protocol

### Per Iteration
1. Run: `run-fno-training-iteration --workspace-dir {workspace} --source-path {source} [--memory-path <previous_memory>]`
2. Read the returned `iteration_report.json`
3. Look at `convergence.decision`:

### Decision Rules
- **`convergence.decision == "continue"`**
  → Run the next iteration immediately. Pass the `--memory-path` from the report so the next iteration learns from this one.

- **`convergence.decision == "converged"`**
  → Stop. Read the best candidate artifacts and report the final best score, model family, and artifact paths.

- **`convergence.decision == "stagnant"`**
  → You may either:
    a) Try one more iteration with expanded search (the report will suggest parameters), OR
    b) Escalate to hypothesis lab for new strategy discovery.

- **`convergence.decision == "doom_loop"`**
  → Escalate immediately. Run `escalate-fno-training-to-hypothesis`, then after it completes run `integrate-hypothesis-training-feedback`, then resume training.

- **`convergence.decision == "escalate"`**
  → Run `escalate-fno-training-to-hypothesis --workspace-dir {workspace} --memory-path <path>`, then `integrate-hypothesis-training-feedback`, then continue.

### Checkpointing
Every 2 iterations, run `checkpoint-fno-training-loop` with the current memory and convergence paths. This saves state in case you need to resume.

### Constraints
- Do NOT edit project files.
- Do NOT invent commands outside the allowed pde-agent prefix.
- Do NOT download files or search the web.
- Only use: Read, Glob, Grep, Bash.
- In Bash, only use the exact `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent <command>` prefix.

### Getting Started
1. Build the contract: `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent build-fno-training-closed-loop-contract --workspace-dir {workspace} --source-path {source}`
2. Run the first iteration: `UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent run-fno-training-iteration --workspace-dir {workspace} --source-path {source} --max-candidates 2`
3. Read the report and follow the decision rules above.
"""
    prompt_path.write_text(prompt, encoding="utf-8")
    return {
        "schema_version": 1,
        "state_hint": "CLAUDE_FNO_TRAINING_CLOSED_LOOP_TASK_BUILT",
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
            "Run next iteration with: run-fno-training-iteration --workspace-dir <dir>"
        )
    elif decision == "escalate":
        directives.append(
            "Escalate with: escalate-fno-training-to-hypothesis --workspace-dir <dir>"
        )
    elif decision in {"converged", "stagnant", "doom_loop"}:
        directives.append(
            "Review best candidate artifacts and report final results."
        )
    else:
        directives.append(
            "Run next iteration with: run-fno-training-iteration --workspace-dir <dir>"
        )
    return directives
