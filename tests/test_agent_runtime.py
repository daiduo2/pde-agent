from __future__ import annotations

import inspect
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


def test_build_claude_agent_sdk_runtime_contract_restricts_to_pde_agent_tools(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import build_agent_runtime_contract

    contract = build_agent_runtime_contract(
        runtime_name="claude_agent_sdk",
        output_path=tmp_path / "runtime_contract.json",
        workspace_dir=Path("/repo"),
        tool_catalog_path=Path("/repo/runs/tool_catalog.json"),
        required_commands=[
            "build-fno-training-harness-contract",
            "run-fno-training-optimization-loop",
        ],
        force=False,
    )

    assert contract["state_hint"] == "AGENT_RUNTIME_CONTRACT_BUILT"
    assert contract["runtime"]["name"] == "claude_agent_sdk"
    assert contract["runtime"]["primary"] is True
    assert contract["runtime"]["compatibility_backends"] == []
    assert contract["runtime"]["legacy_backends"] == []
    assert contract["runtime"]["legacy_backend_policy"] == (
        "not exposed in active runtime contracts"
    )
    assert contract["runtime"]["sdk_package"] == "claude-agent-sdk"
    assert contract["runtime"]["sdk_import"] == "claude_agent_sdk"
    assert contract["tool_policy"]["allowed_tools"] == ["Read", "Glob", "Grep", "Bash"]
    assert contract["tool_policy"]["permission_mode"] == "dontAsk"
    assert contract["tool_policy"]["bash_guard_hook"] == "enabled"
    assert "Edit" in contract["tool_policy"]["disallowed_tools"]
    assert contract["bash_policy"]["allowed_prefixes"] == [
        "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"
    ]
    assert contract["bash_policy"]["allowed_pde_agent_commands"] == [
        "build-fno-training-harness-contract",
        "run-fno-training-optimization-loop",
    ]
    assert contract["bash_policy"]["forbidden_patterns"] == [
        "mkdir ",
        "rm ",
        "python -c",
        "python3 -c",
        "pip install",
        "uv add",
        ";",
        "&&",
        "||",
        "|",
        "$(",
        "`",
    ]
    assert contract["required_commands"] == [
        "build-fno-training-harness-contract",
        "run-fno-training-optimization-loop",
    ]
    active_contract_text = json.dumps(contract, sort_keys=True).lower()
    assert "hermes" not in active_contract_text
    assert "heartbeat" not in active_contract_text

    persisted = json.loads((tmp_path / "runtime_contract.json").read_text())
    assert persisted == contract




def test_cli_help_hides_legacy_commands_from_active_surface() -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run-claude-agent-sdk-runtime" in result.output
    assert "build-agent-research-directive" in result.output
    assert "build-agent-meta-research-directive" in result.output
    assert "hermes" not in result.output.lower()
    assert "run-hermes-heartbeat" not in result.output
    assert "run-hermes-adaptive-heartbeat" not in result.output


def test_bash_guard_rejects_wrapped_training_commands() -> None:
    from pde_agent.core.agent_runtime import _validate_bash_command

    assert (
        _validate_bash_command(
            command=(
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent "
                "run-fno-training-optimization-loop "
                "--candidate-plan-path runs/x.json --force"
            ),
            allowed_bash_prefixes=[
                "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"
            ],
            allowed_pde_agent_commands={"run-fno-training-optimization-loop"},
        )
        is None
    )
    assert (
        _validate_bash_command(
            command=(
                "uv run --frozen pde-agent run-fno-training-optimization-loop "
                "--candidate-plan-path runs/x.json --force"
            ),
            allowed_bash_prefixes=["uv run --frozen pde-agent"],
            allowed_pde_agent_commands={"run-fno-training-optimization-loop"},
        )
        is None
    )
    assert (
        _validate_bash_command(
            command=(
                "uv run --frozen pde-agent run-fno-training-optimization-loop "
                "--candidate-plan-path runs/x.json >/tmp/fno.log 2>&1"
            ),
            allowed_bash_prefixes=["uv run --frozen pde-agent"],
            allowed_pde_agent_commands={"run-fno-training-optimization-loop"},
        )
        is not None
    )
    assert (
        _validate_bash_command(
            command=(
                "uv run --frozen pde-agent run-fno-training-optimization-loop "
                "| tail -40"
            ),
            allowed_bash_prefixes=["uv run --frozen pde-agent"],
            allowed_pde_agent_commands={"run-fno-training-optimization-loop"},
        )
        is not None
    )


def test_claude_agent_sdk_runtime_dry_run_writes_session_artifact(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime

    session = run_claude_agent_sdk_runtime(
        prompt="Read the runtime contract and choose the next pde-agent action.",
        output_path=tmp_path / "session.json",
        workspace_dir=tmp_path,
        allowed_tools=["Read", "Glob", "Grep", "Bash"],
        disallowed_tools=["Edit", "Write"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=True,
        force=False,
    )

    assert session["state_hint"] == "CLAUDE_AGENT_SDK_RUNTIME_RAN"
    assert session["runtime"]["name"] == "claude_agent_sdk"
    assert session["status"] == "dry_run"
    assert session["prompt_preview"].startswith("Read the runtime contract")
    assert session["tool_policy"]["allowed_tools"] == ["Read", "Glob", "Grep", "Bash"]
    assert session["tool_policy"]["permission_mode"] == "dontAsk"
    assert session["artifacts"]["session_path"] == str(tmp_path / "session.json")


def test_claude_agent_sdk_runtime_passes_dontask_and_hook_policy(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime

    captured = {}

    class FakeOptions:
        def __init__(self, **kwargs):
            captured["options"] = kwargs

    class FakeHookMatcher:
        def __init__(self, **kwargs):
            captured["hook_matcher"] = kwargs

    class FakeSdk:
        ClaudeAgentOptions = FakeOptions
        HookMatcher = FakeHookMatcher

        @staticmethod
        async def query(*, prompt, options):
            captured["prompt"] = prompt
            captured["options_object"] = options
            yield "done"

    session = run_claude_agent_sdk_runtime(
        prompt="run",
        output_path=tmp_path / "session.json",
        event_log_path=tmp_path / "events.jsonl",
        workspace_dir=tmp_path,
        allowed_tools=["Read", "Bash"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=False,
        force=False,
        sdk_module=FakeSdk,
    )

    assert session["status"] == "completed"
    assert session["sdk_message_count"] == 1
    assert captured["options"]["permission_mode"] == "dontAsk"
    assert captured["options"]["allowed_tools"] == ["Read", "Bash"]
    assert captured["options"]["disallowed_tools"] == ["Edit"]
    assert "PreToolUse" in captured["options"]["hooks"]
    assert captured["hook_matcher"]["matcher"] == "Bash"
    assert inspect.iscoroutinefunction(captured["hook_matcher"]["hooks"][0])


def test_claude_agent_sdk_runtime_persists_failed_session_artifact(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime

    class FakeOptions:
        def __init__(self, **kwargs):
            pass

    class FakeSdk:
        ClaudeAgentOptions = FakeOptions
        HookMatcher = None

        @staticmethod
        async def query(*, prompt, options):
            _ = prompt, options
            raise RuntimeError("sdk transport failed")
            yield "unreachable"

    session = run_claude_agent_sdk_runtime(
        prompt="run",
        output_path=tmp_path / "failed_session.json",
        workspace_dir=tmp_path,
        allowed_tools=["Read", "Bash"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=False,
        force=False,
        sdk_module=FakeSdk,
    )

    assert session["status"] == "failed"
    assert session["sdk_message_count"] == 0
    assert session["errors"][0].startswith("RuntimeError: sdk transport failed")
    persisted = json.loads((tmp_path / "failed_session.json").read_text())
    assert persisted == session


def test_claude_agent_sdk_runtime_records_sdk_stderr_on_failure(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime

    class FakeOptions:
        def __init__(self, **kwargs):
            pass

    class FakeProcessError(Exception):
        stderr = "permission denied detail"
        exit_code = 2

    class FakeSdk:
        ClaudeAgentOptions = FakeOptions
        HookMatcher = None

        @staticmethod
        async def query(*, prompt, options):
            _ = prompt, options
            raise FakeProcessError("Command failed")
            yield "unreachable"

    session = run_claude_agent_sdk_runtime(
        prompt="run",
        output_path=tmp_path / "failed_session.json",
        workspace_dir=tmp_path,
        allowed_tools=["Read", "Bash"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=False,
        force=False,
        sdk_module=FakeSdk,
    )

    assert "permission denied detail" in session["errors"][0]
    assert "exit_code: 2" in session["errors"][0]


def test_claude_agent_sdk_runtime_streams_structured_readable_events(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime

    class FakeOptions:
        def __init__(self, **kwargs):
            pass

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name, input):
            self.name = name
            self.input = input

    class FakeToolResultBlock:
        def __init__(self, content, is_error=False):
            self.content = content
            self.is_error = is_error

    class FakeResultMessage:
        subtype = "success"
        is_error = False
        num_turns = 2
        duration_ms = 123
        result = "done"
        errors = []

    FakeAssistantMessage.__name__ = "AssistantMessage"
    FakeTextBlock.__name__ = "TextBlock"
    FakeToolUseBlock.__name__ = "ToolUseBlock"
    FakeToolResultBlock.__name__ = "ToolResultBlock"
    FakeResultMessage.__name__ = "ResultMessage"

    class FakeSdk:
        ClaudeAgentOptions = FakeOptions
        HookMatcher = None

        @staticmethod
        async def query(*, prompt, options):
            _ = prompt, options
            yield FakeAssistantMessage(
                [
                    FakeTextBlock("I will inspect evidence."),
                    FakeToolUseBlock(
                        "Bash",
                        {"command": "uv run --frozen pde-agent cockpit-snapshot"},
                    ),
                    FakeToolResultBlock("snapshot built\n" * 100),
                ]
            )
            yield FakeResultMessage()

    streamed = []
    session = run_claude_agent_sdk_runtime(
        prompt="run",
        output_path=tmp_path / "session.json",
        event_log_path=tmp_path / "events.jsonl",
        workspace_dir=tmp_path,
        allowed_tools=["Read", "Bash"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=False,
        force=False,
        sdk_module=FakeSdk,
        on_message=streamed.append,
    )

    assert session["status"] == "completed"
    assert session["sdk_message_count"] == 4
    assert streamed == session["sdk_messages"]
    assert session["artifacts"]["event_log_path"] == str(tmp_path / "events.jsonl")
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines() == (
        session["sdk_messages"]
    )
    assert '"type": "assistant"' in streamed[0]
    assert '"type": "tool_use"' in streamed[1]
    assert '"type": "tool_result"' in streamed[2]
    assert '"type": "result"' in streamed[3]


def test_claude_agent_sdk_runtime_times_out_and_writes_session(
    tmp_path: Path,
) -> None:
    import asyncio

    from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime

    class FakeOptions:
        def __init__(self, **kwargs):
            pass

    class FakeSdk:
        ClaudeAgentOptions = FakeOptions
        HookMatcher = None

        @staticmethod
        async def query(*, prompt, options):
            _ = prompt, options
            await asyncio.sleep(10)
            yield "unreachable"

    session = run_claude_agent_sdk_runtime(
        prompt="run",
        output_path=tmp_path / "session.json",
        event_log_path=tmp_path / "events.jsonl",
        workspace_dir=tmp_path,
        allowed_tools=["Read", "Bash"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=False,
        force=False,
        sdk_module=FakeSdk,
        timeout_seconds=0.01,
    )

    assert session["status"] == "timeout"
    assert session["sdk_message_count"] == 0
    assert "timed out after 0.01 seconds" in session["errors"][0]
    assert session["tool_policy"]["timeout_seconds"] == 0.01
    assert (tmp_path / "session.json").exists()
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines() == (
        session["sdk_messages"]
    )


def test_claude_cli_runtime_dry_run_records_command_session(tmp_path: Path) -> None:
    from pde_agent.core.agent_runtime import run_claude_cli_runtime

    session = run_claude_cli_runtime(
        prompt="hello",
        output_path=tmp_path / "claude_cli_session.json",
        workspace_dir=tmp_path,
        allowed_tools=["Read"],
        disallowed_tools=["Edit"],
        permission_mode="dontAsk",
        output_format="stream-json",
        dry_run=True,
        force=False,
    )

    assert session["state_hint"] == "CLAUDE_CLI_RUNTIME_RAN"
    assert session["status"] == "dry_run"
    assert session["runtime"]["name"] == "claude_cli"
    assert session["process"]["exit_code"] is None
    assert session["sdk_messages"] == []
    assert (tmp_path / "claude_cli_session.json").exists()


def test_claude_cli_runtime_converts_text_stdout_to_session_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import agent_runtime

    completed = Mock(returncode=0, stdout="pde-agent is current\n", stderr="")
    monkeypatch.setattr(agent_runtime.subprocess, "run", Mock(return_value=completed))

    session = agent_runtime.run_claude_cli_runtime(
        prompt="hello",
        output_path=tmp_path / "claude_cli_session.json",
        workspace_dir=tmp_path,
        allowed_tools=["Read"],
        disallowed_tools=["Edit"],
        permission_mode="dontAsk",
        output_format="text",
        dry_run=False,
        force=False,
    )

    assert session["status"] == "completed"
    assert session["sdk_message_count"] == 1
    assert "pde-agent is current" in session["sdk_messages"][0]
    called = agent_runtime.subprocess.run.call_args.kwargs
    assert called["cwd"] == tmp_path.resolve()
    assert called["capture_output"] is True


def test_claude_cli_runtime_cli_returns_payload(tmp_path: Path, monkeypatch) -> None:
    from pde_agent.core import agent_runtime
    from typer.testing import CliRunner

    from pde_agent.cli import app

    completed = Mock(returncode=0, stdout="ok\n", stderr="")
    monkeypatch.setattr(agent_runtime.subprocess, "run", Mock(return_value=completed))

    result = CliRunner().invoke(
        app,
        [
            "run-claude-cli-runtime",
            "--prompt",
            "hello",
            "--output-path",
            str(tmp_path / "session.json"),
            "--workspace-dir",
            str(tmp_path),
            "--output-format",
            "text",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state_hint"] == "CLAUDE_CLI_RUNTIME_RAN"
    assert payload["metrics"]["sdk_message_count"] == 1


def test_export_competition_llm_log_writes_jsonl_from_claude_session(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import export_competition_llm_log

    session_path = tmp_path / "claude_session.json"
    session_path.write_text(
        json.dumps(
            {
                "state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN",
                "sdk_messages": [
                    json.dumps({"type": "assistant", "text": "I will inspect evidence."}),
                    json.dumps(
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {
                                "command": "uv run --frozen pde-agent cockpit-snapshot"
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": False,
                            "result": "Iteration complete.",
                        }
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    report = export_competition_llm_log(
        session_path=session_path,
        output_path=tmp_path / "task1_logs.log",
        force=False,
    )

    assert report["state_hint"] == "COMPETITION_LLM_LOG_EXPORTED"
    rows = [
        json.loads(line)
        for line in (tmp_path / "task1_logs.log").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    assert all("timestamp" in row for row in rows)
    assert all("elapsed_seconds" in row for row in rows)
    assert all("response" in row or "tool_calls" in row for row in rows)
    assert rows[0]["response"] == "I will inspect evidence."
    assert rows[1]["tool_calls"].startswith("Bash(")
    assert rows[2]["response"] == "Iteration complete."


def test_export_competition_llm_log_cli_returns_payload(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    session_path = tmp_path / "claude_session.json"
    session_path.write_text(
        json.dumps(
            {
                "state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN",
                "sdk_messages": [
                    json.dumps({"type": "assistant", "text": "ready"}),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "export-competition-llm-log",
            "--session-path",
            str(session_path),
            "--output-path",
            str(tmp_path / "task1_logs.log"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state_hint"] == "COMPETITION_LLM_LOG_EXPORTED"
    assert payload["metrics"]["line_count"] == 1
    assert (tmp_path / "task1_logs.log").exists()


def test_bash_prefix_guard_hook_denies_unapproved_commands() -> None:
    from pde_agent.core.agent_runtime import build_bash_prefix_guard_hook

    hook = build_bash_prefix_guard_hook(
        ["uv run --frozen pde-agent"],
        allowed_pde_agent_commands=["build-scientific-tool-catalog"],
    )
    allowed = asyncio.run(
        hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_input": {"command": "uv run --frozen pde-agent build-scientific-tool-catalog"},
        },
        "tool_001",
        {},
        )
    )
    denied = asyncio.run(
        hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_input": {"command": "mkdir -p runs/x"},
        },
        "tool_002",
        {},
        )
    )

    assert allowed == {}
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "outside allowed pde-agent prefixes" in denied["hookSpecificOutput"][
        "permissionDecisionReason"
    ]

    denied_unlisted = asyncio.run(
        hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_input": {"command": "uv run --frozen pde-agent train-baseline"},
        },
        "tool_003",
        {},
        )
    )
    denied_shell = asyncio.run(
        hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_input": {
                "command": (
                    "uv run --frozen pde-agent build-scientific-tool-catalog; "
                    "rm -rf runs/x"
                )
            },
        },
        "tool_004",
        {},
        )
    )
    assert denied_unlisted["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "not in the allowed command gateway" in denied_unlisted[
        "hookSpecificOutput"
    ]["permissionDecisionReason"]
    assert denied_shell["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "shell control syntax" in denied_shell["hookSpecificOutput"][
        "permissionDecisionReason"
    ]


def test_build_claude_fno_training_smoke_task_maps_full_agent_loop(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import build_claude_fno_training_smoke_task

    task = build_claude_fno_training_smoke_task(
        output_dir=tmp_path / "task",
        workspace_dir=Path("/repo"),
        runtime_contract_path=Path("/repo/runs/runtime_contract.json"),
        tool_catalog_path=Path("/repo/runs/tool_catalog.json"),
        source_path=Path("/repo/data/burgers.hdf5"),
        previous_evidence_paths=[Path("/repo/runs/previous_evidence.json")],
        force=False,
    )

    assert task["state_hint"] == "CLAUDE_FNO_TRAINING_SMOKE_TASK_BUILT"
    assert task["runtime"]["name"] == "claude_agent_sdk"
    assert task["artifacts"]["prompt_path"].startswith(str(tmp_path))
    assert task["artifacts"]["prompt_path"].endswith("claude_agent_prompt.md")
    assert task["artifacts"]["checklist_path"].startswith(str(tmp_path))
    assert task["artifacts"]["checklist_path"].endswith("expected_artifacts.json")
    assert task["required_commands"] == [
        "build-fno-training-harness-contract",
        "run-fno-training-optimization-loop",
        "record-fno-training-optimization-evidence",
        "build-hypothesis-algorithm-design",
        "validate-hypothesis-algorithm-design",
        "record-promoted-algorithm-strategy-proposal",
        "convert-candidate-promotion-to-strategy-evidence",
        "audit-agent-runtime-availability",
    ]
    prompt = Path(task["artifacts"]["prompt_path"]).read_text(encoding="utf-8")
    assert "You are the Claude Agent SDK runtime for pde-agent" in prompt
    assert "Do not patch source files" in prompt
    assert "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent" in prompt
    assert "record-promoted-algorithm-strategy-proposal" in prompt
    assert str(tmp_path / "task" / "fno_training_harness_contract.json") in prompt
    assert "writeback_failure_report.json" in prompt
    assert "do not keep retrying or wait indefinitely" in prompt

    checklist = json.loads(
        Path(task["artifacts"]["checklist_path"]).read_text(encoding="utf-8")
    )
    assert checklist["failure_artifacts"] == {
        "writeback_failure_report_path": str(
            tmp_path / "task" / "writeback_failure_report.json"
        )
    }
    assert checklist["required_state_hints"] == [
        "FNO_TRAINING_HARNESS_CONTRACT_BUILT",
        "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
        "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
        "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
        "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED",
        "STRATEGY_PROPOSAL_RECORDED",
        "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "AGENT_RUNTIME_AVAILABILITY_AUDITED",
    ]


def test_build_claude_promotion_smoke_task_maps_promotion_branch(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import build_claude_promotion_smoke_task

    task = build_claude_promotion_smoke_task(
        output_dir=tmp_path / "promotion_task",
        workspace_dir=Path("/repo"),
        runtime_contract_path=Path("/repo/runs/runtime_contract.json"),
        tool_catalog_path=Path("/repo/runs/tool_catalog.json"),
        force=False,
    )

    assert task["state_hint"] == "CLAUDE_PROMOTION_SMOKE_TASK_BUILT"
    assert task["runtime"]["name"] == "claude_agent_sdk"
    assert task["required_commands"] == [
        "record-promoted-algorithm-strategy-proposal",
        "convert-candidate-promotion-to-strategy-evidence",
        "audit-agent-runtime-availability",
    ]
    prompt = Path(task["artifacts"]["prompt_path"]).read_text(encoding="utf-8")
    assert "Claude Agent SDK PDEAgent Promotion Smoke" in prompt
    assert "record-promoted-algorithm-strategy-proposal" in prompt
    assert "convert-candidate-promotion-to-strategy-evidence" in prompt
    assert "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent" in prompt
    assert "writeback_failure_report.json" in prompt
    assert "schema gap" in prompt

    checklist = json.loads(
        Path(task["artifacts"]["checklist_path"]).read_text(encoding="utf-8")
    )
    assert checklist["failure_artifacts"] == {
        "writeback_failure_report_path": str(
            tmp_path / "promotion_task" / "writeback_failure_report.json"
        )
    }
    assert checklist["required_state_hints"] == [
        "STRATEGY_PROPOSAL_RECORDED",
        "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "AGENT_RUNTIME_AVAILABILITY_AUDITED",
    ]
    promotion = json.loads(
        Path(checklist["fixture_artifacts"]["candidate_promotion_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert promotion["state_hint"] == "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED"
    assert promotion["decision"] == "promote"
    assert promotion["promote_to_strategy_growth"] is True


def test_build_claude_hypothesis_lab_closure_task_maps_agent_loop(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import build_claude_hypothesis_lab_closure_task

    task = build_claude_hypothesis_lab_closure_task(
        output_dir=tmp_path / "closure_task",
        workspace_dir=Path("/repo"),
        runtime_contract_path=Path("/repo/runs/runtime_contract.json"),
        tool_catalog_path=Path("/repo/runs/tool_catalog.json"),
        candidate_family_sandbox_path=Path("/repo/runs/family_sandbox.json"),
        context_path=Path("/repo/runs/context.hdf5"),
        target_path=Path("/repo/runs/target.hdf5"),
        reference_path=Path("/repo/data/reference.hdf5"),
        force=False,
    )

    assert task["state_hint"] == "CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_BUILT"
    assert task["runtime"]["name"] == "claude_agent_sdk"
    assert task["required_commands"] == [
        "materialize-candidate-family-item",
        "run-hypothesis-lab",
        "build-hypothesis-lab-next-policy",
        "plan-hypothesis-candidates",
        "audit-agent-runtime-availability",
        "record-agent-runtime-issue",
    ]
    prompt = Path(task["artifacts"]["prompt_path"]).read_text(encoding="utf-8")
    assert "Claude Agent SDK Hypothesis Lab Closure" in prompt
    assert "materialize-candidate-family-item" in prompt
    assert "run-hypothesis-lab" in prompt
    assert "build-hypothesis-lab-next-policy" in prompt
    assert "plan-hypothesis-candidates" in prompt
    assert "Do not patch source files" in prompt
    assert "Do not run discovery or probing commands" in prompt
    assert "record-agent-runtime-issue" in prompt
    assert "do not use Write" in prompt
    assert "Stop after building the next policy" in prompt
    assert "UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent" in prompt

    checklist = json.loads(
        Path(task["artifacts"]["checklist_path"]).read_text(encoding="utf-8")
    )
    assert checklist["required_state_hints"] == [
        "ALGORITHM_CANDIDATES_COMPILED",
        "HYPOTHESIS_LAB_RAN",
        "HYPOTHESIS_LAB_NEXT_POLICY_BUILT",
        "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
        "AGENT_RUNTIME_AVAILABILITY_AUDITED",
    ]
    assert checklist["expected_artifacts"]["failure_memory_path"].endswith(
        "failure_memory.json"
    )
    assert checklist["expected_artifacts"]["candidate_plan_path"].endswith(
        "hypothesis_candidate_plan.json"
    )
    assert checklist["failure_artifacts"]["execution_issue_path"].endswith(
        "closure_execution_issue.json"
    )


def test_build_claude_literature_candidate_closure_task_maps_agent_loop(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import (
        build_claude_literature_candidate_closure_task,
    )

    task = build_claude_literature_candidate_closure_task(
        output_dir=tmp_path / "literature_task",
        workspace_dir=Path("/repo"),
        runtime_contract_path=Path("/repo/runs/runtime_contract.json"),
        tool_catalog_path=Path("/repo/runs/tool_catalog.json"),
        paper_parse_path=Path("/repo/runs/paper_parse.json"),
        operator_family="fno",
        target_problem="PDE_Burgers task1 early/middle dynamics",
        registry_path=tmp_path / "literature_registry.json",
        current_best_score=120.0,
        force=False,
    )

    assert task["state_hint"] == "CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_BUILT"
    assert task["runtime"]["name"] == "claude_agent_sdk"
    assert task["required_commands"] == [
        "prepare-literature-extraction-context",
        "prepare-pde-literature-extraction-skill",
        "prepare-fno-literature-adaptation-task",
        "record-agent-algorithm-cards",
        "validate-algorithm-cards-v2",
        "compile-algorithm-candidates",
        "run-literature-algorithm-v2-pipeline",
        "audit-agent-runtime-availability",
        "record-agent-runtime-issue",
    ]
    prompt = Path(task["artifacts"]["prompt_path"]).read_text(encoding="utf-8")
    assert "Claude Agent SDK Literature-to-Candidate Closure" in prompt
    assert "prepare-literature-extraction-context" in prompt
    assert "prepare-pde-literature-extraction-skill" in prompt
    assert "prepare-fno-literature-adaptation-task" in prompt
    assert "agent_algorithm_cards.json" in prompt
    assert "record-agent-algorithm-cards" in prompt
    assert "Do not use Write." in prompt
    assert "validate-algorithm-cards-v2" in prompt
    assert "compile-algorithm-candidates" in prompt
    assert "run-literature-algorithm-v2-pipeline" in prompt
    assert "extract-algorithm-cards" not in prompt
    assert "Do not patch source files" in prompt
    assert "persistent agent workspace" in prompt
    assert "do not create sibling run directories" in prompt
    assert "expected runtime session artifact" in prompt

    checklist = json.loads(
        Path(task["artifacts"]["checklist_path"]).read_text(encoding="utf-8")
    )
    assert checklist["operator_family"] == "fno"
    assert checklist["workspace_policy"]["persistence"] == (
        "reuse_named_task_workspace"
    )
    assert checklist["workspace_policy"]["expected_session_path"].endswith(
        "claude_agent_sdk_session.json"
    )
    assert checklist["required_state_hints"] == [
        "LITERATURE_EXTRACTION_CONTEXT_PREPARED",
        "PDE_LITERATURE_EXTRACTION_SKILL_PREPARED",
        "FNO_LITERATURE_ADAPTATION_TASK_PREPARED",
        "ALGORITHM_CARDS_EXTRACTED",
        "ALGORITHM_CARDS_V2_VALIDATED",
        "ALGORITHM_CANDIDATES_COMPILED",
        "LITERATURE_ALGORITHM_V2_PIPELINE_RAN",
        "AGENT_RUNTIME_AVAILABILITY_AUDITED",
    ]
    assert checklist["expected_artifacts"]["claude_agent_sdk_session_path"].endswith(
        "claude_agent_sdk_session.json"
    )
    assert checklist["expected_artifacts"]["agent_algorithm_cards_path"].endswith(
        "agent_algorithm_cards.json"
    )
    assert checklist["expected_artifacts"]["algorithm_candidates_path"].endswith(
        "algorithm_candidates.json"
    )
    assert checklist["failure_artifacts"]["execution_issue_path"].endswith(
        "literature_closure_execution_issue.json"
    )


def test_claude_agent_sdk_runtime_reports_missing_optional_dependency(
    tmp_path: Path,
) -> None:
    from pde_agent.core.agent_runtime import AgentRuntimeError
    from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime

    try:
        run_claude_agent_sdk_runtime(
            prompt="run",
            output_path=tmp_path / "session.json",
            workspace_dir=tmp_path,
            allowed_tools=["Read"],
            disallowed_tools=[],
            allowed_bash_prefixes=["uv run --frozen pde-agent"],
            dry_run=False,
            force=False,
            sdk_module=None,
        )
    except AgentRuntimeError as exc:
        assert "claude-agent-sdk" in str(exc)
        assert "claude_agent_sdk" in str(exc)
    else:
        raise AssertionError("expected missing SDK dependency error")


def test_agent_runtime_commands_return_json_contracts(tmp_path: Path) -> None:
    from pde_agent.commands.agent_runtime import build_agent_runtime_contract_command
    from pde_agent.commands.agent_runtime import run_claude_agent_sdk_runtime_command

    contract_payload = build_agent_runtime_contract_command(
        output_path=str(tmp_path / "runtime_contract.json"),
        runtime_name="claude_agent_sdk",
        workspace_dir=str(tmp_path),
        tool_catalog_path=str(tmp_path / "tool_catalog.json"),
        required_commands=["run-fno-training-optimization-loop"],
        force=False,
    )
    assert contract_payload["command"] == "build-agent-runtime-contract"
    assert contract_payload["state_hint"] == "AGENT_RUNTIME_CONTRACT_BUILT"
    assert contract_payload["metrics"]["required_command_count"] == 1

    runtime_payload = run_claude_agent_sdk_runtime_command(
        prompt="Use pde-agent only.",
        prompt_file=None,
        output_path=str(tmp_path / "session.json"),
        event_log_path=None,
        workspace_dir=str(tmp_path),
        allowed_tools=["Read", "Bash"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        allowed_pde_agent_commands=[],
        dry_run=True,
        force=False,
    )
    assert runtime_payload["command"] == "run-claude-agent-sdk-runtime"
    assert runtime_payload["state_hint"] == "CLAUDE_AGENT_SDK_RUNTIME_RAN"
    assert runtime_payload["metrics"]["sdk_message_count"] == 0
    assert runtime_payload["runtime_status"] == "dry_run"
    assert runtime_payload["errors"] == []


def test_claude_runtime_command_accepts_prompt_file(tmp_path: Path) -> None:
    from pde_agent.commands.agent_runtime import run_claude_agent_sdk_runtime_command

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Read the contract from file.", encoding="utf-8")

    payload = run_claude_agent_sdk_runtime_command(
        prompt=None,
        prompt_file=str(prompt_path),
        output_path=str(tmp_path / "session.json"),
        event_log_path=None,
        workspace_dir=str(tmp_path),
        allowed_tools=["Read"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        allowed_pde_agent_commands=[],
        dry_run=True,
        force=False,
    )

    session = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert payload["state_hint"] == "CLAUDE_AGENT_SDK_RUNTIME_RAN"
    assert session["prompt_preview"] == "Read the contract from file."


def test_claude_runtime_command_propagates_failed_session_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.commands import agent_runtime

    def fake_runtime(**kwargs):
        output_path = Path(kwargs["output_path"])
        session = {
            "artifacts": {"session_path": str(output_path), "event_log_path": None},
            "sdk_message_count": 0,
            "status": "failed",
            "tool_policy": {"allowed_tools": ["Read"]},
            "bash_policy": {"allowed_pde_agent_commands": []},
            "errors": ["sdk failed"],
        }
        output_path.write_text(json.dumps(session), encoding="utf-8")
        return session

    monkeypatch.setattr(agent_runtime, "run_claude_agent_sdk_runtime", fake_runtime)

    payload = agent_runtime.run_claude_agent_sdk_runtime_command(
        prompt="run",
        prompt_file=None,
        output_path=str(tmp_path / "session.json"),
        event_log_path=None,
        workspace_dir=str(tmp_path),
        allowed_tools=["Read"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["UV_CACHE_DIR=.uv-cache uv run --frozen pde-agent"],
        allowed_pde_agent_commands=[],
        dry_run=False,
        force=False,
    )

    assert payload["status"] == "failed"
    assert payload["runtime_status"] == "failed"
    assert payload["errors"] == ["sdk failed"]


def test_audit_agent_runtime_availability_flags_unexpected_tool_use(
    tmp_path: Path,
) -> None:
    from pde_agent.core.availability_audit import audit_agent_runtime_availability

    session_path = tmp_path / "claude_session.json"
    session_path.write_text(
        json.dumps(
            {
                "state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN",
                "status": "completed",
                "sdk_message_count": 2,
                "tool_policy": {
                    "allowed_tools": ["Read", "Bash"],
                    "disallowed_tools": ["Edit"],
                },
                "sdk_messages": [
                    json.dumps(
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "report.json"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {"file_path": "foo.py"},
                        }
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    report = audit_agent_runtime_availability(
        runtime_name="claude_agent_sdk",
        output_path=tmp_path / "availability.json",
        tool_catalog_path=tmp_path / "tool_catalog.json",
        canary_report_paths=[session_path],
        cli_commands=["build-agent-runtime-contract"],
        required_commands=["build-agent-runtime-contract"],
        force=False,
    )

    assert report["decision"] == "needs_work"
    assert report["metrics"]["tool_policy_violation_count"] == 1
    assert report["gaps"]["unexpected_tool_uses"] == ["Edit"]
    assert report["next_command"] == "run-claude-agent-sdk-runtime"
    canary = report["canary_evidence"][0]
    assert canary["tool_use_names"] == ["Read", "Edit"]
    assert canary["unexpected_tool_uses"] == ["Edit"]


def test_claude_fno_training_smoke_task_command_returns_json_contract(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.agent_runtime import (
        build_claude_fno_training_smoke_task_command,
    )

    payload = build_claude_fno_training_smoke_task_command(
        output_dir=str(tmp_path / "task"),
        workspace_dir="/repo",
        runtime_contract_path="/repo/runtime_contract.json",
        tool_catalog_path="/repo/tool_catalog.json",
        source_path="/repo/data.hdf5",
        previous_evidence_paths=[],
        force=False,
    )

    assert payload["command"] == "build-claude-fno-training-smoke-task"
    assert payload["state_hint"] == "CLAUDE_FNO_TRAINING_SMOKE_TASK_BUILT"
    assert payload["metrics"]["required_command_count"] == 8
    assert payload["next_hint"] == "run-claude-agent-sdk-runtime"


def test_claude_promotion_smoke_task_command_returns_json_contract(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.agent_runtime import (
        build_claude_promotion_smoke_task_command,
    )

    payload = build_claude_promotion_smoke_task_command(
        output_dir=str(tmp_path / "promotion_task"),
        workspace_dir=str(tmp_path),
        runtime_contract_path=str(tmp_path / "runtime_contract.json"),
        tool_catalog_path=str(tmp_path / "tool_catalog.json"),
        force=False,
    )

    assert payload["command"] == "build-claude-promotion-smoke-task"
    assert payload["state_hint"] == "CLAUDE_PROMOTION_SMOKE_TASK_BUILT"
    assert payload["metrics"]["required_command_count"] == 3
    assert payload["next_hint"] == "run-claude-agent-sdk-runtime"


def test_claude_hypothesis_lab_closure_task_command_returns_json_contract(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.agent_runtime import (
        build_claude_hypothesis_lab_closure_task_command,
    )

    payload = build_claude_hypothesis_lab_closure_task_command(
        output_dir=str(tmp_path / "closure_task"),
        workspace_dir=str(tmp_path),
        runtime_contract_path=str(tmp_path / "runtime_contract.json"),
        tool_catalog_path=str(tmp_path / "tool_catalog.json"),
        candidate_family_sandbox_path=str(tmp_path / "family_sandbox.json"),
        context_path=str(tmp_path / "context.hdf5"),
        target_path=str(tmp_path / "target.hdf5"),
        reference_path=None,
        force=False,
    )

    assert payload["command"] == "build-claude-hypothesis-lab-closure-task"
    assert payload["state_hint"] == "CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_BUILT"
    assert payload["metrics"]["required_command_count"] == 6
    assert payload["next_hint"] == "run-claude-agent-sdk-runtime"


def test_claude_literature_candidate_closure_task_command_returns_json_contract(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.agent_runtime import (
        build_claude_literature_candidate_closure_task_command,
    )

    payload = build_claude_literature_candidate_closure_task_command(
        output_dir=str(tmp_path / "literature_task"),
        workspace_dir=str(tmp_path),
        runtime_contract_path=str(tmp_path / "runtime_contract.json"),
        tool_catalog_path=str(tmp_path / "tool_catalog.json"),
        paper_parse_path=str(tmp_path / "paper_parse.json"),
        operator_family="FNO",
        target_problem="PDE_Burgers task1",
        registry_path=str(tmp_path / "literature_registry.json"),
        current_best_score=120.0,
        force=False,
    )

    assert payload["command"] == "build-claude-literature-candidate-closure-task"
    assert payload["state_hint"] == "CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_BUILT"
    assert payload["metrics"]["required_command_count"] == 9
    assert payload["next_hint"] == "run-claude-agent-sdk-runtime"


def test_record_agent_runtime_issue_command_writes_structured_issue(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.agent_runtime import record_agent_runtime_issue_command

    payload = record_agent_runtime_issue_command(
        output_path=str(tmp_path / "issue.json"),
        failed_command="audit-agent-runtime-availability",
        artifact_path=str(tmp_path / "runtime_availability.json"),
        stderr_summary="missing canary session",
        likely_schema_gap=True,
        next_action="stop_after_issue_recorded",
        force=False,
    )

    assert payload["command"] == "record-agent-runtime-issue"
    assert payload["state_hint"] == "AGENT_RUNTIME_ISSUE_RECORDED"
    issue = json.loads((tmp_path / "issue.json").read_text(encoding="utf-8"))
    assert issue == {
        "artifact_path": str((tmp_path / "runtime_availability.json").resolve()),
        "failed_command": "audit-agent-runtime-availability",
        "likely_schema_gap": True,
        "next_action": "stop_after_issue_recorded",
        "schema_version": 1,
        "state_hint": "AGENT_RUNTIME_ISSUE_RECORDED",
        "stderr_summary": "missing canary session",
    }
