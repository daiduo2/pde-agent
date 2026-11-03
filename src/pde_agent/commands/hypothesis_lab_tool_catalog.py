"""Thin CLI wrapper for the Hypothesis Lab tool catalog builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_tool_registry import (
    STATE_HINT,
    build_hypothesis_lab_tool_catalog_payload,
)
from pde_agent.core.json_output import success_payload


def build_hypothesis_lab_tool_catalog_command(
    *,
    output_path: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Build and optionally persist the Hypothesis Lab tool catalog.

    Args:
        output_path: Optional file path to write catalog JSON.
        force: Overwrite existing file if True.

    Returns:
        JSON payload conforming to the CLI contract.
    """
    catalog = build_hypothesis_lab_tool_catalog_payload(
        output_path=output_path,
        force=force,
    )

    artifacts: dict[str, object] = {
        "catalog": catalog,
        "tool_count": catalog["tool_count"],
    }
    if output_path is not None:
        artifacts["output_path"] = output_path

    return success_payload(
        run_id="hypothesis-lab-tool-catalog",
        command="build-hypothesis-lab-tool-catalog",
        state_hint=STATE_HINT,
        artifacts=artifacts,
        metrics={"tool_count": catalog["tool_count"]},
        next_hint="use catalog to configure agent runtime tool policy",
    )
