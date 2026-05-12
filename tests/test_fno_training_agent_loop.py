"""TDD tests for FNO training agent loop prompt and launcher.
Uses real Claude Code SDK input/output traces for verification."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


class TestAgentPrompt:
    def test_prompt_contains_decision_rules(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_interface import (
            build_claude_fno_training_closed_loop_task,
        )

        task = build_claude_fno_training_closed_loop_task(
            workspace_dir=str(tmp_path / "ws"),
            source_path="/tmp/data.h5",
            force=True,
        )
        prompt = task["prompt"]
        assert "continue" in prompt.lower()
        assert "escalate" in prompt.lower()
        assert "stop" in prompt.lower()
        assert "convergence.decision" in prompt

    def test_prompt_contains_command_examples(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_interface import (
            build_claude_fno_training_closed_loop_task,
        )

        task = build_claude_fno_training_closed_loop_task(
            workspace_dir=str(tmp_path / "ws"),
            source_path="/tmp/data.h5",
            force=True,
        )
        prompt = task["prompt"]
        assert "run-fno-training-iteration" in prompt
        assert "build-fno-training-closed-loop-contract" in prompt

    def test_prompt_contains_checkpoint_guidance(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_interface import (
            build_claude_fno_training_closed_loop_task,
        )

        task = build_claude_fno_training_closed_loop_task(
            workspace_dir=str(tmp_path / "ws"),
            source_path="/tmp/data.h5",
            force=True,
        )
        prompt = task["prompt"]
        assert "checkpoint" in prompt.lower()

    def test_task_has_required_commands(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_interface import (
            build_claude_fno_training_closed_loop_task,
        )

        task = build_claude_fno_training_closed_loop_task(
            workspace_dir=str(tmp_path / "ws"),
            source_path="/tmp/data.h5",
            force=True,
        )
        required = {
            "run-fno-training-iteration",
            "build-fno-training-closed-loop-contract",
            "check-fno-training-convergence",
            "checkpoint-fno-training-loop",
        }
        assert required.issubset(set(task["required_commands"]))

    def test_task_artifacts_paths(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_interface import (
            build_claude_fno_training_closed_loop_task,
        )

        task = build_claude_fno_training_closed_loop_task(
            workspace_dir=str(tmp_path / "ws"),
            source_path="/tmp/data.h5",
            force=True,
        )
        assert "contract_path" in task["artifacts"]
        assert "prompt_path" in task["artifacts"]


class TestSDKInputOutput:
    """Tests using real claude-agent-sdk runtime (dry-run mode)."""

    def test_sdk_dry_run_accepts_prompt_and_config(self, tmp_path: Path) -> None:
        from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime
        from pde_agent.core.fno_training_agent_interface import (
            build_claude_fno_training_closed_loop_task,
        )

        task = build_claude_fno_training_closed_loop_task(
            workspace_dir=str(tmp_path / "ws"),
            source_path=str(tmp_path / "data.h5"),
        )
        output_path = tmp_path / "session.json"
        session = run_claude_agent_sdk_runtime(
            prompt=task["prompt"],
            output_path=output_path,
            workspace_dir=tmp_path / "ws",
            allowed_tools=["Read", "Glob", "Grep", "Bash"],
            disallowed_tools=["Edit", "Write", "NotebookEdit", "WebFetch"],
            allowed_bash_prefixes=[
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"
            ],
            allowed_pde_agent_commands=task["required_commands"],
            dry_run=True,
            force=True,
        )
        assert session["status"] == "dry_run"
        assert session["runtime"]["sdk_package"] == "claude-agent-sdk"
        assert "Read" in session["tool_policy"]["allowed_tools"]
        assert "Edit" in session["tool_policy"]["disallowed_tools"]
        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["status"] == "dry_run"

    def test_sdk_session_has_prompt_preview(self, tmp_path: Path) -> None:
        from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime
        from pde_agent.core.fno_training_agent_interface import (
            build_claude_fno_training_closed_loop_task,
        )

        task = build_claude_fno_training_closed_loop_task(
            workspace_dir=str(tmp_path / "ws"),
            source_path=str(tmp_path / "data.h5"),
        )
        output_path = tmp_path / "session.json"
        session = run_claude_agent_sdk_runtime(
            prompt=task["prompt"],
            output_path=output_path,
            workspace_dir=tmp_path / "ws",
            allowed_tools=["Read", "Glob", "Grep", "Bash"],
            disallowed_tools=["Edit", "Write", "NotebookEdit", "WebFetch"],
            allowed_bash_prefixes=[
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"
            ],
            allowed_pde_agent_commands=task["required_commands"],
            dry_run=True,
            force=True,
        )
        assert "prompt_preview" in session
        assert len(session["prompt_preview"]) > 50
        assert "autonomous" in session["prompt_preview"].lower()

    def test_sdk_bash_policy_has_training_commands(self, tmp_path: Path) -> None:
        from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime
        from pde_agent.core.fno_training_agent_interface import (
            build_claude_fno_training_closed_loop_task,
        )

        task = build_claude_fno_training_closed_loop_task(
            workspace_dir=str(tmp_path / "ws"),
            source_path=str(tmp_path / "data.h5"),
        )
        output_path = tmp_path / "session.json"
        session = run_claude_agent_sdk_runtime(
            prompt=task["prompt"],
            output_path=output_path,
            workspace_dir=tmp_path / "ws",
            allowed_tools=["Read", "Glob", "Grep", "Bash"],
            disallowed_tools=["Edit", "Write", "NotebookEdit", "WebFetch"],
            allowed_bash_prefixes=[
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"
            ],
            allowed_pde_agent_commands=task["required_commands"],
            dry_run=True,
            force=True,
        )
        bash_commands = session["bash_policy"]["allowed_pde_agent_commands"]
        assert "run-fno-training-iteration" in bash_commands
        assert "checkpoint-fno-training-loop" in bash_commands


class TestAgentLoopLauncher:
    def test_loop_launcher_dry_run(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_loop import (
            run_claude_fno_training_closed_loop,
        )

        result = run_claude_fno_training_closed_loop(
            workspace_dir=tmp_path / "ws",
            source_path=tmp_path / "data.h5",
            max_iterations=2,
            dry_run=True,
            force=True,
        )
        assert result["status"] == "dry_run"
        assert result["state_hint"] == "FNO_TRAINING_CLOSED_LOOP_LAUNCHED"
        assert result["max_iterations"] == 2

    def test_loop_launcher_contract_path(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_loop import (
            run_claude_fno_training_closed_loop,
        )

        result = run_claude_fno_training_closed_loop(
            workspace_dir=tmp_path / "ws",
            source_path=tmp_path / "data.h5",
            max_iterations=1,
            dry_run=True,
            force=True,
        )
        assert "contract_path" in result["artifacts"]
        assert Path(result["artifacts"]["contract_path"]).exists()

    def test_loop_launcher_prompt_path(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_loop import (
            run_claude_fno_training_closed_loop,
        )

        result = run_claude_fno_training_closed_loop(
            workspace_dir=tmp_path / "ws",
            source_path=tmp_path / "data.h5",
            max_iterations=1,
            dry_run=True,
            force=True,
        )
        assert "prompt_path" in result["artifacts"]
        assert Path(result["artifacts"]["prompt_path"]).exists()

    def test_loop_launcher_writes_tool_catalog(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_loop import (
            run_claude_fno_training_closed_loop,
        )

        result = run_claude_fno_training_closed_loop(
            workspace_dir=tmp_path / "ws",
            source_path=tmp_path / "data.h5",
            max_iterations=1,
            dry_run=True,
            force=True,
        )
        assert "tool_catalog_path" in result["artifacts"]
        assert Path(result["artifacts"]["tool_catalog_path"]).exists()

    def test_loop_launcher_dry_run_skips_api_key_check(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_loop import (
            run_claude_fno_training_closed_loop,
        )

        result = run_claude_fno_training_closed_loop(
            workspace_dir=tmp_path / "ws",
            source_path=tmp_path / "data.h5",
            max_iterations=1,
            dry_run=True,
            force=True,
        )
        assert result["status"] == "dry_run"

    def test_loop_launcher_non_dry_run_requires_api_key(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_loop import (
            FnoTrainingAgentLoopError,
            run_claude_fno_training_closed_loop,
        )

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(FnoTrainingAgentLoopError) as exc_info:
                run_claude_fno_training_closed_loop(
                    workspace_dir=tmp_path / "ws",
                    source_path=tmp_path / "data.h5",
                    max_iterations=1,
                    dry_run=False,
                    force=True,
                )
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_build_sdk_env_includes_base_url_and_api_key(self) -> None:
        from pde_agent.core.fno_training_agent_loop import _build_sdk_env

        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "test-key",
                "ANTHROPIC_BASE_URL": "https://custom.example.com",
            },
            clear=True,
        ):
            env = _build_sdk_env()
        assert env["ANTHROPIC_API_KEY"] == "test-key"
        assert env["ANTHROPIC_BASE_URL"] == "https://custom.example.com"

    def test_build_sdk_env_omits_missing_vars(self) -> None:
        from pde_agent.core.fno_training_agent_loop import _build_sdk_env

        with patch.dict("os.environ", {}, clear=True):
            env = _build_sdk_env()
        assert "ANTHROPIC_BASE_URL" not in env
        assert "ANTHROPIC_API_KEY" not in env

    def test_loop_launcher_with_base_url_env(self, tmp_path: Path) -> None:
        from pde_agent.core.fno_training_agent_loop import (
            run_claude_fno_training_closed_loop,
        )

        with patch.dict(
            "os.environ",
            {"ANTHROPIC_BASE_URL": "https://proxy.example.com"},
        ):
            result = run_claude_fno_training_closed_loop(
                workspace_dir=tmp_path / "ws",
                source_path=tmp_path / "data.h5",
                max_iterations=1,
                dry_run=True,
                force=True,
            )
        assert result["status"] == "dry_run"
