"""Tool catalog for FNO training closed-loop agent commands."""

import json
from pathlib import Path
from typing import Any


STATE_HINT = "FNO_TRAINING_TOOL_CATALOG_BUILT"


def build_fno_training_tool_catalog() -> dict[str, Any]:
    commands = [
        {
            "name": "run-fno-training-iteration",
            "description": (
                "Run one complete FNO training iteration: plan candidates, "
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
                    "name": "source-path",
                    "type": "string",
                    "required": True,
                    "description": "Path to HDF5 source data",
                },
                {
                    "name": "memory-path",
                    "type": "string",
                    "required": False,
                    "description": "Path to previous iteration memory JSON",
                },
                {
                    "name": "evidence-path",
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
                    "description": "Target score for convergence",
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
                "state_hint": "FNO_TRAINING_ITERATION_REPORTED",
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
                "run-fno-training-iteration "
                "--workspace-dir ./ws --source-path ./data.h5 --max-candidates 2"
            ),
        },
        {
            "name": "build-fno-training-closed-loop-contract",
            "description": (
                "Build agent runtime contract with allowed commands, "
                "tool policy, and bash policy for the training loop."
            ),
            "parameters": [
                {
                    "name": "workspace-dir",
                    "type": "string",
                    "required": True,
                    "description": "Workspace directory",
                },
                {
                    "name": "source-path",
                    "type": "string",
                    "required": True,
                    "description": "Source data path",
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
                "state_hint": "FNO_TRAINING_CONTRACT_BUILT",
                "required_commands": "list[string]",
                "tool_policy": "object",
                "bash_policy": "object",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "build-fno-training-closed-loop-contract "
                "--workspace-dir ./ws --source-path ./data.h5"
            ),
        },
        {
            "name": "check-fno-training-convergence",
            "description": (
                "Check convergence from training iteration memory. "
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
                "state_hint": "FNO_TRAINING_CONVERGENCE_CHECKED",
                "decision": "string",
                "reason": "string",
                "confidence": "float",
                "recommended_actions": "list[object]",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "check-fno-training-convergence --memory-path ./mem.json"
            ),
        },
        {
            "name": "build-fno-training-memory",
            "description": (
                "Build iteration memory from optimization report."
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
                "state_hint": "FNO_TRAINING_MEMORY_BUILT",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "build-fno-training-memory "
                "--optimization-report-path ./opt.json --iteration-index 0"
            ),
        },
        {
            "name": "checkpoint-fno-training-loop",
            "description": "Save full loop state to checkpoint file.",
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
                "state_hint": "FNO_TRAINING_CHECKPOINTED",
                "checkpoint_path": "string",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "checkpoint-fno-training-loop "
                "--workspace-dir ./ws --iteration-index 2 "
                "--memory-path ./mem.json --convergence-path ./conv.json"
            ),
        },
        {
            "name": "resume-fno-training-loop",
            "description": "Resume loop from checkpoint file.",
            "parameters": [
                {
                    "name": "checkpoint-path",
                    "type": "string",
                    "required": True,
                },
            ],
            "return_shape": {
                "state_hint": "FNO_TRAINING_RESUMED",
                "iteration_index": "integer",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "resume-fno-training-loop --checkpoint-path ./chk.json"
            ),
        },
        {
            "name": "escalate-fno-training-to-hypothesis",
            "description": (
                "Escalate training loop to hypothesis lab for strategy discovery."
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
                "state_hint": "FNO_TRAINING_STRATEGY_ESCALATION_TRIGGERED",
            },
            "example": (
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "escalate-fno-training-to-hypothesis "
                "--workspace-dir ./ws --memory-path ./mem.json"
            ),
        },
        {
            "name": "integrate-hypothesis-training-feedback",
            "description": (
                "Merge hypothesis lab feedback into training memory."
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
