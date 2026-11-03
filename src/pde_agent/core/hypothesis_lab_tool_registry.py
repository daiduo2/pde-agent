"""Tool registry for Hypothesis Lab agent infrastructure primitives.

Maps the 15 agent-hypothesis-* CLI commands to ScientificToolSpec objects
so they are discoverable by agent runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pde_agent.core.json_output import success_payload
from pde_agent.core.scientific_tool_router import (
    ScientificToolSpec,
    VALID_RISK_LEVELS,
)

STATE_HINT = "HYPOTHESIS_LAB_TOOL_CATALOG_BUILT"


@dataclass(frozen=True)
class HypothesisLabToolSpec:
    """Thin wrapper around ScientificToolSpec for hypothesis-lab primitives."""

    spec: ScientificToolSpec

    def to_dict(self) -> dict[str, Any]:
        return self.spec.to_dict()


def build_hypothesis_lab_tool_catalog() -> list[ScientificToolSpec]:
    """Return ScientificToolSpec entries for all 15 hypothesis-lab primitives."""
    return [
        ScientificToolSpec(
            name="agent-hypothesis-create-session",
            command="agent-hypothesis-create-session",
            description="Create a new HypothesisLabSession in the given workspace.",
            risk_level="artifact_write",
            required_inputs=["workspace"],
            produced_artifacts=["session_path"],
            state_hints=["HYPOTHESIS_LAB_SESSION_CREATED"],
            agent_guidance="First step in any hypothesis-lab workflow. Creates the session directory.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-load-session",
            command="agent-hypothesis-load-session",
            description="Load an existing HypothesisLabSession from workspace.",
            risk_level="read_only",
            required_inputs=["workspace"],
            produced_artifacts=["session_path"],
            state_hints=["HYPOTHESIS_LAB_SESSION_LOADED"],
            agent_guidance="Use to resume work on an existing session.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-build-cards",
            command="agent-hypothesis-build-cards",
            description="Build knowledge cards from reports and inputs.",
            risk_level="artifact_write",
            required_inputs=["workspace"],
            produced_artifacts=["knowledge_cards"],
            state_hints=["HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT"],
            agent_guidance="Feed manual reports, adaptive reports, logs, and literature into structured knowledge cards.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-generate-hypotheses",
            command="agent-hypothesis-generate-hypotheses",
            description="Generate hypotheses from knowledge cards.",
            risk_level="artifact_write",
            required_inputs=["workspace"],
            produced_artifacts=["hypotheses"],
            state_hints=["HYPOTHESIS_LAB_HYPOTHESES_GENERATED"],
            agent_guidance="Run after build-cards to turn knowledge into testable hypotheses.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-design-blueprints",
            command="agent-hypothesis-design-blueprints",
            description="Design algorithm blueprints from hypotheses.",
            risk_level="artifact_write",
            required_inputs=["workspace"],
            produced_artifacts=["blueprints"],
            state_hints=["HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED"],
            agent_guidance="Run after generate-hypotheses to produce executable blueprint designs.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-run-sandbox",
            command="agent-hypothesis-run-sandbox",
            description="Run a blueprint through the sandbox verifier.",
            risk_level="sandbox",
            required_inputs=["workspace", "context_path", "target_path"],
            produced_artifacts=["sandbox_report"],
            state_hints=["HYPOTHESIS_LAB_SANDBOX_VERIFIED"],
            agent_guidance="Execute one blueprint against data. Bounded and safe.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-record-outcome",
            command="agent-hypothesis-record-outcome",
            description="Record sandbox outcome into session state.",
            risk_level="artifact_write",
            required_inputs=["workspace"],
            produced_artifacts=["outcome"],
            state_hints=["HYPOTHESIS_LAB_OUTCOME_RECORDED"],
            agent_guidance="Persist results after sandbox so they feed into next iteration.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-run-pipeline",
            command="agent-hypothesis-run-pipeline",
            description="Run the full cards-to-blueprints-to-sandbox pipeline.",
            risk_level="long_running",
            required_inputs=["workspace", "context_path", "target_path"],
            produced_artifacts=["hypothesis_lab_report"],
            state_hints=["HYPOTHESIS_LAB_RAN"],
            agent_guidance="Convenience command for the full pipeline. Use for batch runs.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-query-state",
            command="agent-hypothesis-query-state",
            description="Query current session state and next recommended action.",
            risk_level="read_only",
            required_inputs=["workspace"],
            produced_artifacts=[],
            state_hints=["HYPOTHESIS_LAB_STATE_QUERIED"],
            agent_guidance="Read-only introspection. Use anytime to understand session status.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-list-candidates",
            command="agent-hypothesis-list-candidates",
            description="List algorithm candidates in the session.",
            risk_level="read_only",
            required_inputs=["workspace"],
            produced_artifacts=[],
            state_hints=["HYPOTHESIS_LAB_CANDIDATES_LISTED"],
            agent_guidance="Filter by family or status to find candidates to sandbox.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-get-best-evidence",
            command="agent-hypothesis-get-best-evidence",
            description="Get the best-scoring evidence from the session.",
            risk_level="read_only",
            required_inputs=["workspace"],
            produced_artifacts=[],
            state_hints=["HYPOTHESIS_LAB_BEST_EVIDENCE_QUERIED"],
            agent_guidance="Use to compare against current best before deciding to promote.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-get-failure-pattern",
            command="agent-hypothesis-get-failure-pattern",
            description="Analyze failure patterns for a candidate family.",
            risk_level="read_only",
            required_inputs=["workspace", "family"],
            produced_artifacts=[],
            state_hints=["HYPOTHESIS_LAB_FAILURE_PATTERN_QUERIED"],
            agent_guidance="Use before retrying a family to understand why it failed.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-list-pending-blueprints",
            command="agent-hypothesis-list-pending-blueprints",
            description="List blueprints that have not yet been sandboxed.",
            risk_level="read_only",
            required_inputs=["workspace"],
            produced_artifacts=[],
            state_hints=["HYPOTHESIS_LAB_PENDING_BLUEPRINTS_LISTED"],
            agent_guidance="Use to find the next blueprint to run through sandbox.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-get-pareto-front",
            command="agent-hypothesis-get-pareto-front",
            description="Get the Pareto-optimal candidate set.",
            risk_level="read_only",
            required_inputs=["workspace"],
            produced_artifacts=[],
            state_hints=["HYPOTHESIS_LAB_PARETO_FRONT_QUERIED"],
            agent_guidance="Use to explore tradeoffs between score, speed, and complexity.",
        ),
        ScientificToolSpec(
            name="agent-hypothesis-summarize-events",
            command="agent-hypothesis-summarize-events",
            description="Summarize recent session events.",
            risk_level="read_only",
            required_inputs=["workspace"],
            produced_artifacts=[],
            state_hints=["HYPOTHESIS_LAB_EVENTS_SUMMARIZED"],
            agent_guidance="Use for quick session history review.",
        ),
    ]


def build_hypothesis_lab_tool_catalog_payload(
    *,
    output_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build the full catalog payload with schema metadata.

    Args:
        output_path: Optional path to write catalog JSON.
        force: Overwrite existing file if True.

    Returns:
        JSON-serializable catalog dict conforming to the CLI contract.
    """
    from pde_agent.core.scientific_tool_router import (
        ScientificToolRouterError,
        _guard_output,
        _write_optional_json,
    )

    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)

    tools = [spec.to_dict() for spec in build_hypothesis_lab_tool_catalog()]
    catalog = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "tool_count": len(tools),
        "tools": tools,
        "risk_levels": sorted(VALID_RISK_LEVELS),
    }
    _write_optional_json(output_file, catalog)
    return catalog
