"""Tests for hypothesis_lab_tool_registry and its CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pde_agent.core.hypothesis_lab_tool_registry import (
    STATE_HINT,
    HypothesisLabToolSpec,
    build_hypothesis_lab_tool_catalog,
    build_hypothesis_lab_tool_catalog_payload,
)
from pde_agent.core.scientific_tool_router import (
    VALID_RISK_LEVELS,
    ScientificToolRouterError,
    ScientificToolSpec,
)


class TestBuildHypothesisLabToolCatalog:
    def test_returns_exactly_15_tools(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        assert len(catalog) == 15

    def test_all_entries_are_scientific_tool_spec(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        for spec in catalog:
            assert isinstance(spec, ScientificToolSpec)

    def test_all_risk_levels_are_valid(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        for spec in catalog:
            assert spec.risk_level in VALID_RISK_LEVELS, (
                f"{spec.name} has invalid risk_level: {spec.risk_level}"
            )

    def test_read_only_queries_have_correct_risk_level(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        read_only_names = {
            "agent-hypothesis-query-state",
            "agent-hypothesis-list-candidates",
            "agent-hypothesis-get-best-evidence",
            "agent-hypothesis-get-failure-pattern",
            "agent-hypothesis-list-pending-blueprints",
            "agent-hypothesis-get-pareto-front",
            "agent-hypothesis-summarize-events",
            "agent-hypothesis-load-session",
        }
        for spec in catalog:
            if spec.name in read_only_names:
                assert spec.risk_level == "read_only", (
                    f"{spec.name} should be read_only"
                )

    def test_artifact_write_tools_have_correct_risk_level(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        artifact_write_names = {
            "agent-hypothesis-build-cards",
            "agent-hypothesis-generate-hypotheses",
            "agent-hypothesis-design-blueprints",
            "agent-hypothesis-record-outcome",
            "agent-hypothesis-create-session",
        }
        for spec in catalog:
            if spec.name in artifact_write_names:
                assert spec.risk_level == "artifact_write", (
                    f"{spec.name} should be artifact_write"
                )

    def test_sandbox_tool_has_correct_risk_level(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        sandbox_spec = next(
            (s for s in catalog if s.name == "agent-hypothesis-run-sandbox"),
            None,
        )
        assert sandbox_spec is not None
        assert sandbox_spec.risk_level == "sandbox"

    def test_long_running_tool_has_correct_risk_level(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        pipeline_spec = next(
            (s for s in catalog if s.name == "agent-hypothesis-run-pipeline"),
            None,
        )
        assert pipeline_spec is not None
        assert pipeline_spec.risk_level == "long_running"

    def test_all_tools_have_non_empty_required_inputs(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        for spec in catalog:
            assert len(spec.required_inputs) > 0, (
                f"{spec.name} must have at least one required input"
            )

    def test_all_tools_have_state_hints(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        for spec in catalog:
            assert len(spec.state_hints) > 0, (
                f"{spec.name} must have at least one state_hint"
            )

    def test_all_tools_have_agent_guidance(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        for spec in catalog:
            assert spec.agent_guidance, f"{spec.name} must have agent_guidance"

    def test_all_tools_are_active_by_default(self) -> None:
        catalog = build_hypothesis_lab_tool_catalog()
        for spec in catalog:
            assert spec.runtime_status == "active"

    def test_to_dict_validates_risk_level(self) -> None:
        bad_spec = ScientificToolSpec(
            name="bad-tool",
            command="bad-tool",
            description="test",
            risk_level="invalid_risk",
            required_inputs=[],
            produced_artifacts=[],
            state_hints=[],
            agent_guidance="test",
        )
        with pytest.raises(ScientificToolRouterError):
            bad_spec.to_dict()


class TestHypothesisLabToolSpec:
    def test_wraps_scientific_tool_spec(self) -> None:
        inner = ScientificToolSpec(
            name="test-tool",
            command="test-tool",
            description="A test tool.",
            risk_level="read_only",
            required_inputs=["x"],
            produced_artifacts=["y"],
            state_hints=["TEST"],
            agent_guidance="do it",
        )
        wrapper = HypothesisLabToolSpec(spec=inner)
        assert wrapper.to_dict() == inner.to_dict()


class TestBuildHypothesisLabToolCatalogPayload:
    def test_returns_catalog_with_schema_metadata(self) -> None:
        payload = build_hypothesis_lab_tool_catalog_payload()
        assert payload["schema_version"] == 1
        assert payload["state_hint"] == STATE_HINT
        assert payload["tool_count"] == 15
        assert len(payload["tools"]) == 15
        assert "risk_levels" in payload

    def test_writes_to_file_when_output_path_given(self, tmp_path: Path) -> None:
        output_file = tmp_path / "catalog.json"
        payload = build_hypothesis_lab_tool_catalog_payload(
            output_path=output_file,
            force=False,
        )
        assert output_file.exists()
        persisted = json.loads(output_file.read_text())
        assert persisted == payload

    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        output_file = tmp_path / "catalog.json"
        output_file.write_text("{}")
        with pytest.raises(ScientificToolRouterError):
            build_hypothesis_lab_tool_catalog_payload(
                output_path=output_file,
                force=False,
            )

    def test_allows_overwrite_with_force(self, tmp_path: Path) -> None:
        output_file = tmp_path / "catalog.json"
        output_file.write_text("{}")
        payload = build_hypothesis_lab_tool_catalog_payload(
            output_path=output_file,
            force=True,
        )
        assert json.loads(output_file.read_text()) == payload


class TestBuildHypothesisLabToolCatalogCommand:
    def test_returns_correct_json_structure(self) -> None:
        from pde_agent.commands.hypothesis_lab_tool_catalog import (
            build_hypothesis_lab_tool_catalog_command,
        )

        result = build_hypothesis_lab_tool_catalog_command()
        assert result["status"] == "ok"
        assert result["state_hint"] == STATE_HINT
        assert result["command"] == "build-hypothesis-lab-tool-catalog"
        assert result["artifacts"]["tool_count"] == 15
        assert result["metrics"]["tool_count"] == 15
        assert "next_hint" in result

    def test_includes_output_path_in_artifacts_when_given(self, tmp_path: Path) -> None:
        from pde_agent.commands.hypothesis_lab_tool_catalog import (
            build_hypothesis_lab_tool_catalog_command,
        )

        output_file = tmp_path / "catalog.json"
        result = build_hypothesis_lab_tool_catalog_command(
            output_path=str(output_file)
        )
        assert result["artifacts"]["output_path"] == str(output_file)

    def test_writes_file_when_output_path_given(self, tmp_path: Path) -> None:
        from pde_agent.commands.hypothesis_lab_tool_catalog import (
            build_hypothesis_lab_tool_catalog_command,
        )

        output_file = tmp_path / "catalog.json"
        result = build_hypothesis_lab_tool_catalog_command(
            output_path=str(output_file),
            force=False,
        )
        assert output_file.exists()
        persisted = json.loads(output_file.read_text())
        assert persisted["tool_count"] == 15
        assert persisted["state_hint"] == STATE_HINT
