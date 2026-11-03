"""CLI commands for failure strategy layer.

Thin wrappers around hypothesis_lab_failure_strategy core functions.
Each returns a JSON payload following the CLI contract.
"""

from __future__ import annotations

from typing import Any

from pde_agent.core.hypothesis_lab_memory import (
    HypothesisLabMemory,
    get_memory_path_from_env,
)
from pde_agent.core.json_output import success_payload


def extract_failure_rules_command(
    memory_path: str | None = None,
) -> dict[str, Any]:
    """Extract counterfactual rules from cross-session failure memory.

    Args:
        memory_path: Path to JSONL memory file. Uses env fallback if None.

    Returns:
        JSON payload with extracted rules.
    """
    path = memory_path or get_memory_path_from_env()
    memory = HypothesisLabMemory(path)
    rules = memory.get_failure_rules()

    return success_payload(
        run_id="global",
        command="extract-failure-rules",
        state_hint="FAILURE_RULES_EXTRACTED",
        artifacts={"rules": rules},
        metrics={"rule_count": len(rules)},
        next_hint="review rules and apply to next sandbox run",
    )
