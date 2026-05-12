"""TDD tests for FNO training tool catalog."""

import json
from pathlib import Path

import pytest


class TestToolCatalog:
    def test_catalog_has_required_commands(self) -> None:
        from pde_agent.core.fno_training_tool_catalog import build_fno_training_tool_catalog

        catalog = build_fno_training_tool_catalog()
        names = {cmd["name"] for cmd in catalog["commands"]}
        required = {
            "run-fno-training-iteration",
            "build-fno-training-closed-loop-contract",
            "check-fno-training-convergence",
            "build-fno-training-memory",
            "checkpoint-fno-training-loop",
            "resume-fno-training-loop",
            "escalate-fno-training-to-hypothesis",
            "integrate-hypothesis-training-feedback",
        }
        assert required.issubset(names), f"Missing commands: {required - names}"

    def test_catalog_command_structure(self) -> None:
        from pde_agent.core.fno_training_tool_catalog import build_fno_training_tool_catalog

        catalog = build_fno_training_tool_catalog()
        for cmd in catalog["commands"]:
            assert "name" in cmd
            assert "description" in cmd
            assert "parameters" in cmd
            assert "return_shape" in cmd
            assert isinstance(cmd["parameters"], list)
            assert isinstance(cmd["return_shape"], dict)

    def test_run_iteration_command_params(self) -> None:
        from pde_agent.core.fno_training_tool_catalog import build_fno_training_tool_catalog

        catalog = build_fno_training_tool_catalog()
        cmd = next(
            c for c in catalog["commands"] if c["name"] == "run-fno-training-iteration"
        )
        param_names = {p["name"] for p in cmd["parameters"]}
        assert "workspace-dir" in param_names
        assert "source-path" in param_names
        assert "memory-path" in param_names

    def test_catalog_schema_version(self) -> None:
        from pde_agent.core.fno_training_tool_catalog import build_fno_training_tool_catalog

        catalog = build_fno_training_tool_catalog()
        assert catalog["schema_version"] == 1
        assert catalog["state_hint"] == "FNO_TRAINING_TOOL_CATALOG_BUILT"

    def test_catalog_json_roundtrip(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_tool_catalog import (
            build_fno_training_tool_catalog,
            write_tool_catalog,
        )

        catalog = build_fno_training_tool_catalog()
        output_path = tmp_path / "catalog.json"
        write_tool_catalog(catalog, output_path)
        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["schema_version"] == 1
        assert len(loaded["commands"]) >= 8
