"""Tool catalog for Task 2 FNO training closed-loop agent commands."""

import json
from pathlib import Path
from typing import Any


STATE_HINT = "TASK2_FNO_TRAINING_TOOL_CATALOG_BUILT"


def build_task2_fno_training_tool_catalog() -> dict[str, Any]:
    commands = [
        {
            "name": "run-task2-fno-training-iteration",
            "description": (
                "Run one complete Task 2 FNO training iteration: plan candidates, "
                "train, evaluate, build memory, check convergence, and report."
            ),
            "parameters": [
                {
                    "name": "workspace-dir",
                    "type": "string",
                    "required": True,
                    "description": "Directory for all iteration artifacts",
                },
                {
                    "name": "memory-path",
                    "type": "string",
                    "required": False,
                    "description": "Path to previous iteration memory JSON",
                },
                {
                    "name": "evidence-paths",
                    "type": "list[string]",
                    "required": False,
                    "description": "Paths to evidence JSON files",
                },
                {
                    "name": "max-candidates",
                    "type": "integer",
                    "required": False,
                    "default": 3,
                    "description": "Maximum candidates to plan",
                },
                {
                    "name": "target-score",
                    "type": "float",
                    "required": False,
                    "description": "Target task2_proxy score for convergence",
                },
                {
                    "name": "patience",
                    "type": "integer",
                    "required": False,
                    "default": 3,
                    "description": "Iterations without improvement before stagnation",
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                    "description": "Overwrite existing artifacts",
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_ITERATION_REPORTED",
                "status": "ok | error",
                "artifacts": {
                    "optimization_report_path": "string",
                    "memory_path": "string",
                    "convergence_path": "string",
                    "report_path": "string",
                },
                "metrics": {
                    "iteration_index": "integer",
                    "best_score": "float | null",
                    "convergence_decision": "string",
                },
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "run-task2-fno-training-iteration "
                "--workspace-dir ./ws --max-candidates 2"
            ),
        },
        {
            "name": "build-task2-fno-training-contract",
            "description": (
                "Build agent runtime contract with allowed commands, "
                "tool policy, and bash policy for the Task 2 training loop."
            ),
            "parameters": [
                {
                    "name": "workspace-dir",
                    "type": "string",
                    "required": True,
                    "description": "Workspace directory",
                },
                {
                    "name": "output-path",
                    "type": "string",
                    "required": False,
                    "description": "Contract output path",
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_CONTRACT_BUILT",
                "required_commands": "list[string]",
                "tool_policy": "object",
                "bash_policy": "object",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "build-task2-fno-training-contract "
                "--workspace-dir ./ws"
            ),
        },
        {
            "name": "check-task2-fno-training-convergence",
            "description": (
                "Check convergence from Task 2 training iteration memory. "
                "Returns decision: continue, converged, stagnant, doom_loop."
            ),
            "parameters": [
                {
                    "name": "memory-path",
                    "type": "string",
                    "required": True,
                    "description": "Path to iteration memory JSON",
                },
                {
                    "name": "target-score",
                    "type": "float",
                    "required": False,
                },
                {
                    "name": "patience",
                    "type": "integer",
                    "required": False,
                    "default": 3,
                },
                {
                    "name": "output-path",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_CONVERGENCE_CHECKED",
                "decision": "string",
                "reason": "string",
                "confidence": "float",
                "recommended_actions": "list[object]",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "check-task2-fno-training-convergence --memory-path ./mem.json"
            ),
        },
        {
            "name": "build-task2-fno-training-memory",
            "description": (
                "Build iteration memory from Task 2 optimization report."
            ),
            "parameters": [
                {
                    "name": "optimization-report-path",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "iteration-index",
                    "type": "integer",
                    "required": True,
                },
                {
                    "name": "previous-memory-path",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "output-path",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "build-task2-fno-training-memory "
                "--optimization-report-path ./opt.json --iteration-index 0"
            ),
        },
        {
            "name": "build-task2-fno-training-final-report",
            "description": (
                "Build final report from best Task 2 training memory."
            ),
            "parameters": [
                {
                    "name": "memory-path",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "workspace-dir",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "output-path",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_FINAL_REPORT_BUILT",
                "best_candidate": "object",
                "provenance": "object",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "build-task2-fno-training-final-report "
                "--memory-path ./mem.json --workspace-dir ./ws"
            ),
        },
        {
            "name": "build-task2-fno-training-health-report",
            "description": (
                "Build health report across all Task 2 training iterations."
            ),
            "parameters": [
                {
                    "name": "workspace-dir",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "output-path",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_HEALTH_REPORT_BUILT",
                "health": "object",
                "exploration": "object",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "build-task2-fno-training-health-report --workspace-dir ./ws"
            ),
        },
        {
            "name": "checkpoint-task2-fno-training-loop",
            "description": "Save full Task 2 loop state to checkpoint file.",
            "parameters": [
                {
                    "name": "workspace-dir",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "iteration-index",
                    "type": "integer",
                    "required": True,
                },
                {
                    "name": "memory-path",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "convergence-path",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_CHECKPOINTED",
                "checkpoint_path": "string",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "checkpoint-task2-fno-training-loop "
                "--workspace-dir ./ws --iteration-index 2 "
                "--memory-path ./mem.json --convergence-path ./conv.json"
            ),
        },
        {
            "name": "resume-task2-fno-training-loop",
            "description": "Resume Task 2 loop from checkpoint file.",
            "parameters": [
                {
                    "name": "checkpoint-path",
                    "type": "string",
                    "required": True,
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_RESUMED",
                "iteration_index": "integer",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "resume-task2-fno-training-loop --checkpoint-path ./chk.json"
            ),
        },
        {
            "name": "escalate-task2-fno-training-to-hypothesis",
            "description": (
                "Escalate Task 2 training loop to hypothesis lab for strategy discovery."
            ),
            "parameters": [
                {
                    "name": "workspace-dir",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "memory-path",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "output-path",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            ],
            "return_shape": {
                "state_hint": "TASK2_FNO_TRAINING_STRATEGY_ESCALATION_TRIGGERED",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "escalate-task2-fno-training-to-hypothesis "
                "--workspace-dir ./ws --memory-path ./mem.json"
            ),
        },
        {
            "name": "integrate-hypothesis-training-feedback",
            "description": (
                "Merge hypothesis lab feedback into Task 2 training memory."
            ),
            "parameters": [
                {
                    "name": "hypothesis-report-path",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "memory-path",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "output-path",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "force",
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            ],
            "return_shape": {
                "state_hint": "HYPOTHESIS_LAB_STRATEGY_FEEDBACK_INTEGRATED",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "integrate-hypothesis-training-feedback "
                "--hypothesis-report-path ./hyp.json --memory-path ./mem.json"
            ),
        },
    ]
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "commands": commands,
    }


def write_tool_catalog(catalog: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(catalog, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
