from __future__ import annotations

import inspect
import asyncio
import json
from pathlib import Path


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
    assert contract["runtime"]["sdk_package"] == "claude-agent-sdk"
    assert contract["runtime"]["sdk_import"] == "claude_agent_sdk"
    assert contract["tool_policy"]["allowed_tools"] == ["Read", "Glob", "Grep", "Bash"]
    assert contract["tool_policy"]["permission_mode"] == "dontAsk"
    assert contract["tool_policy"]["bash_guard_hook"] == "enabled"
    assert "Edit" in contract["tool_policy"]["disallowed_tools"]
    assert contract["bash_policy"]["allowed_prefixes"] == [
        "uv run --frozen pde-agent"
    ]
    assert contract["bash_policy"]["forbidden_patterns"] == [
        "mkdir ",
        "rm ",
        "python -c",
        "python3 -c",
        "pip install",
        "uv add",
    ]
    assert contract["required_commands"] == [
        "build-fno-training-harness-contract",
        "run-fno-training-optimization-loop",
    ]

    persisted = json.loads((tmp_path / "runtime_contract.json").read_text())
    assert persisted == contract


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


def test_bash_prefix_guard_hook_denies_unapproved_commands() -> None:
    from pde_agent.core.agent_runtime import build_bash_prefix_guard_hook

    hook = build_bash_prefix_guard_hook(["uv run --frozen pde-agent"])
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
    assert "uv run --frozen pde-agent" in prompt
    assert "record-promoted-algorithm-strategy-proposal" in prompt
    assert str(tmp_path / "task" / "fno_training_harness_contract.json") in prompt

    checklist = json.loads(
        Path(task["artifacts"]["checklist_path"]).read_text(encoding="utf-8")
    )
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
    assert "uv run --frozen pde-agent" in prompt

    checklist = json.loads(
        Path(task["artifacts"]["checklist_path"]).read_text(encoding="utf-8")
    )
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
        workspace_dir=str(tmp_path),
        allowed_tools=["Read", "Bash"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=True,
        force=False,
    )
    assert runtime_payload["command"] == "run-claude-agent-sdk-runtime"
    assert runtime_payload["state_hint"] == "CLAUDE_AGENT_SDK_RUNTIME_RAN"
    assert runtime_payload["metrics"]["sdk_message_count"] == 0


def test_claude_runtime_command_accepts_prompt_file(tmp_path: Path) -> None:
    from pde_agent.commands.agent_runtime import run_claude_agent_sdk_runtime_command

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Read the contract from file.", encoding="utf-8")

    payload = run_claude_agent_sdk_runtime_command(
        prompt=None,
        prompt_file=str(prompt_path),
        output_path=str(tmp_path / "session.json"),
        workspace_dir=str(tmp_path),
        allowed_tools=["Read"],
        disallowed_tools=["Edit"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=True,
        force=False,
    )

    session = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert payload["state_hint"] == "CLAUDE_AGENT_SDK_RUNTIME_RAN"
    assert session["prompt_preview"] == "Read the contract from file."


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
