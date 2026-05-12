import json
import asyncio
from pathlib import Path


def test_build_cockpit_snapshot_combines_runs_and_active_run(tmp_path: Path) -> None:
    from pde_agent.core.interactive_cockpit import build_cockpit_snapshot

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "fno_selection_score": 75.75,
                    "config": {"forecast_steps": 95},
                    "early_rel_mse": 0.63,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 1.92},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "claude_sdk_session.json").write_text(
        json.dumps({"state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN", "errors": []}),
        encoding="utf-8",
    )

    snapshot = build_cockpit_snapshot(runs_root=tmp_path / "runs", active_run_dir=run_dir)

    assert snapshot["state_hint"] == "INTERACTIVE_COCKPIT_SNAPSHOT_BUILT"
    assert snapshot["read_only_observation"] is True
    assert snapshot["active_run"]["run_id"] == "run_a"
    assert snapshot["active_run"]["fno_evidence"]["selected_candidate_id"] == (
        "fno_train_candidate_002"
    )
    assert "run_a" in snapshot["runs_text"]
    assert "FNO Evidence" in snapshot["active_run_text"]


def test_build_agent_dialogue_request_preserves_freeform_user_intent(tmp_path: Path) -> None:
    from pde_agent.core.interactive_cockpit import build_agent_dialogue_request

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "early_rel_mse": 0.63,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 1.92},
                    "late_guardrail": {"evaluated": False},
                    "selection_reason_codes": ["early_middle_objective_selected"],
                },
            }
        ),
        encoding="utf-8",
    )

    request = build_agent_dialogue_request(
        user_message="继续优化 early/middle，但不要把 late 当主目标",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        output_dir=tmp_path / "dialogue",
        force=False,
    )

    assert request["state_hint"] == "AGENT_DIALOGUE_REQUEST_BUILT"
    assert request["freeform_user_message"] == "继续优化 early/middle，但不要把 late 当主目标"
    assert request["not_finite_state_machine"] is True
    assert request["supervisor_constraints"]["allows_agent_algorithm_design"] is True
    assert request["supervisor_constraints"]["requires_artifact_audit"] is True
    assert Path(request["artifacts"]["prompt_path"]).exists()
    assert Path(request["artifacts"]["request_path"]).exists()
    prompt = Path(request["artifacts"]["prompt_path"]).read_text(encoding="utf-8")
    assert "继续优化 early/middle" in prompt
    assert "Do not reduce the request to a finite state-machine transition" in prompt
    assert "You may propose new algorithms, code changes, or training experiments" in prompt


def test_interactive_cockpit_cli_builds_snapshot_and_dialogue_request(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "early_rel_mse": 0.63,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 1.92},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    snapshot_result = runner.invoke(
        app,
        [
            "cockpit-snapshot",
            "--runs-root",
            str(tmp_path / "runs"),
            "--active-run-dir",
            str(run_dir),
        ],
    )
    assert snapshot_result.exit_code == 0
    snapshot = json.loads(snapshot_result.stdout)
    assert snapshot["state_hint"] == "INTERACTIVE_COCKPIT_SNAPSHOT_BUILT"
    assert snapshot["active_run"]["run_id"] == "run_a"

    request_result = runner.invoke(
        app,
        [
            "cockpit-dialogue-request",
            "--user-message",
            "自由设计下一轮训练",
            "--runs-root",
            str(tmp_path / "runs"),
            "--active-run-dir",
            str(run_dir),
            "--output-dir",
            str(tmp_path / "dialogue"),
        ],
    )
    assert request_result.exit_code == 0
    request = json.loads(request_result.stdout)
    assert request["not_finite_state_machine"] is True
    assert Path(request["artifacts"]["prompt_path"]).exists()

    chat_result = runner.invoke(
        app,
        [
            "cockpit-chat",
            "自由设计下一轮训练",
            "--runs-root",
            str(tmp_path / "runs"),
            "--active-run-dir",
            str(run_dir),
            "--dialogue-output-dir",
            str(tmp_path / "chat_dialogue"),
        ],
    )
    assert chat_result.exit_code == 0
    chat = json.loads(chat_result.stdout)
    assert chat["kind"] == "natural_language"
    assert chat["request"]["not_finite_state_machine"] is True


def test_textual_cockpit_app_can_be_imported() -> None:
    from pde_agent.core.interactive_cockpit_app import PDEAgentCockpitApp

    app = PDEAgentCockpitApp(runs_root="runs")

    assert app.runs_root.name == "runs"


def test_textual_cockpit_focuses_chat_input_on_mount(tmp_path: Path) -> None:
    from pde_agent.core.interactive_cockpit_app import PDEAgentCockpitApp

    app = PDEAgentCockpitApp(runs_root=tmp_path / "runs")

    async def run_probe() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.focused is app.query_one("#agent_input")

    asyncio.run(run_probe())


def test_textual_cockpit_natural_language_starts_visible_agent_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import interactive_cockpit_app
    from pde_agent.core.interactive_cockpit_app import PDEAgentCockpitApp

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "early_rel_mse": 0.63,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 1.92},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )
    app = PDEAgentCockpitApp(
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        dialogue_output_dir=tmp_path / "dialogue",
    )
    calls = []

    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        kwargs["on_message"](
            json.dumps(
                {
                    "type": "assistant",
                    "content": [{"type": "text", "text": "I will design the next step."}],
                }
            )
        )
        output_path = Path(kwargs["output_path"])
        output_path.write_text(
            json.dumps({"state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN"}),
            encoding="utf-8",
        )
        return {"artifacts": {"session_path": str(output_path)}}

    monkeypatch.setattr(
        interactive_cockpit_app,
        "dispatch_agent_dialogue_request",
        fake_dispatch,
    )

    async def run_probe() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("/", "h", "e", "l", "p", "enter")
            await pilot.pause()
            await pilot.press("d", "e", "s", "i", "g", "n", "enter")
            await pilot.pause(0.2)

    asyncio.run(run_probe())

    assert "> /help" in app.transcript
    assert any("PDEAgent shell" in entry for entry in app.transcript)
    assert "> design" in app.transcript
    assert any("request accepted" in entry for entry in app.transcript)
    assert any("agent: executing" in entry for entry in app.transcript)
    assert any(entry == "assistant: I will design the next step." for entry in app.transcript)
    assert calls and calls[0]["dry_run"] is False
    assert (tmp_path / "dialogue" / "agent_dialogue_request.json").exists()


def test_textual_cockpit_execute_dispatch_streams_agent_messages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import interactive_cockpit_app
    from pde_agent.core.interactive_cockpit import build_agent_dialogue_request
    from pde_agent.core.interactive_cockpit_app import PDEAgentCockpitApp

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "early_rel_mse": 0.63,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 1.92},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )
    build_agent_dialogue_request(
        user_message="design next training step",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        output_dir=tmp_path / "dialogue",
        force=False,
    )

    def fake_dispatch(**kwargs):
        kwargs["on_message"](
            json.dumps(
                {
                    "type": "assistant",
                    "content": [{"type": "text", "text": "thinking about evidence"}],
                }
            )
        )
        kwargs["on_message"](
            json.dumps(
                {
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": "uv run --frozen pde-agent cockpit-snapshot"},
                }
            )
        )
        kwargs["on_message"](
            json.dumps(
                {
                    "type": "tool_result",
                    "content": "snapshot built\n" * 100,
                }
            )
        )
        output_path = Path(kwargs["output_path"])
        output_path.write_text(
            json.dumps({"state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN"}),
            encoding="utf-8",
        )
        return {"artifacts": {"session_path": str(output_path)}}

    monkeypatch.setattr(
        interactive_cockpit_app,
        "dispatch_agent_dialogue_request",
        fake_dispatch,
    )
    app = PDEAgentCockpitApp(
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        dialogue_output_dir=tmp_path / "dialogue",
    )

    async def run_probe() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            for key in "/dispatch --execute":
                await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause(0.2)

    asyncio.run(run_probe())

    assert "> /dispatch --execute" in app.transcript
    assert any("agent: executing" in entry for entry in app.transcript)
    assert any(entry == "assistant: thinking about evidence" for entry in app.transcript)
    assert any(
        entry == "tool: Bash: uv run --frozen pde-agent cockpit-snapshot"
        for entry in app.transcript
    )
    assert any(entry.startswith("tool result (ok):") for entry in app.transcript)
    assert not any(entry.strip().startswith("{") for entry in app.transcript)
    assert any("completed, session artifact" in entry for entry in app.transcript)


def test_textual_cockpit_dry_run_slash_command_does_not_auto_execute(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import interactive_cockpit_app
    from pde_agent.core.interactive_cockpit import build_agent_dialogue_request
    from pde_agent.core.interactive_cockpit_app import PDEAgentCockpitApp

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "middle_guardrail": {"evaluated": False},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )
    build_agent_dialogue_request(
        user_message="design next training step",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        output_dir=tmp_path / "dialogue",
        force=False,
    )
    execute_calls = []

    def fake_execute(**kwargs):
        execute_calls.append(kwargs)
        return {"artifacts": {"session_path": str(Path(kwargs["output_path"]))}}

    monkeypatch.setattr(
        interactive_cockpit_app,
        "dispatch_agent_dialogue_request",
        fake_execute,
    )
    app = PDEAgentCockpitApp(
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        dialogue_output_dir=tmp_path / "dialogue",
    )

    async def run_probe() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            for key in "/dispatch --dry-run":
                await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause(0.2)

    asyncio.run(run_probe())

    assert not execute_calls
    assert any("Dispatch artifact:" in entry for entry in app.transcript)


def test_cockpit_formats_agent_events_as_compact_transcript_entries() -> None:
    from pde_agent.core.interactive_cockpit_app import _format_agent_event

    tool = json.dumps(
        {
            "type": "tool_use",
            "name": "Bash",
            "input": {
                "command": "uv run --frozen pde-agent run-fno-training-optimization-loop"
            },
        }
    )
    result = json.dumps(
        {
            "type": "tool_result",
            "content": "line\n" * 300,
        }
    )
    assistant = json.dumps(
        {
            "type": "assistant",
            "content": [{"type": "text", "text": "I will inspect the evidence first."}],
        }
    )

    assert _format_agent_event(tool) == (
        "tool: Bash: uv run --frozen pde-agent run-fno-training-optimization-loop"
    )
    formatted_result = _format_agent_event(result)
    assert formatted_result.startswith("tool result (ok):")
    assert len(formatted_result) < 250
    assert _format_agent_event(assistant) == "assistant: I will inspect the evidence first."
    assert _format_agent_event(
        json.dumps(
            {
                "type": "result",
                "is_error": False,
                "num_turns": 2,
                "duration_ms": 123,
                "result": "done",
            }
        )
    ) == "agent done: turns=2 duration_ms=123 done"


def test_dispatch_agent_dialogue_request_dry_run_uses_prompt_artifact(tmp_path: Path) -> None:
    from pde_agent.core.interactive_cockpit import build_agent_dialogue_request
    from pde_agent.core.interactive_cockpit import dispatch_agent_dialogue_request

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "middle_guardrail": {"evaluated": False},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )
    request = build_agent_dialogue_request(
        user_message="请自主设计下一轮实验",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        output_dir=tmp_path / "dialogue",
        force=False,
    )

    dispatch = dispatch_agent_dialogue_request(
        request_path=request["artifacts"]["request_path"],
        output_path=tmp_path / "dialogue" / "agent_session.json",
        workspace_dir=tmp_path,
        dry_run=True,
        force=False,
    )

    assert dispatch["state_hint"] == "AGENT_DIALOGUE_DISPATCHED"
    assert dispatch["dry_run"] is True
    assert dispatch["runtime_state_hint"] == "CLAUDE_AGENT_SDK_RUNTIME_RAN"
    assert Path(dispatch["artifacts"]["session_path"]).exists()


def test_dispatch_agent_dialogue_request_execute_uses_runtime_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import interactive_cockpit
    from pde_agent.core.interactive_cockpit import build_agent_dialogue_request
    from pde_agent.core.interactive_cockpit import dispatch_agent_dialogue_request

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "middle_guardrail": {"evaluated": False},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )
    request = build_agent_dialogue_request(
        user_message="请执行下一轮自主实验",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        output_dir=tmp_path / "dialogue",
        force=False,
    )
    captured = {}

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN",
            "status": "completed",
            "sdk_message_count": 1,
            "artifacts": {"session_path": str(output_path)},
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(interactive_cockpit, "run_claude_agent_sdk_runtime", fake_runtime)

    dispatch = dispatch_agent_dialogue_request(
        request_path=request["artifacts"]["request_path"],
        output_path=tmp_path / "dialogue" / "agent_session.json",
        workspace_dir=tmp_path,
        dry_run=False,
        force=False,
    )

    assert dispatch["state_hint"] == "AGENT_DIALOGUE_DISPATCHED"
    assert dispatch["dry_run"] is False
    assert captured["dry_run"] is False
    assert captured["allowed_tools"] == ["Read", "Glob", "Grep", "Bash"]
    assert "Edit" in captured["disallowed_tools"]
    assert captured["allowed_bash_prefixes"] == ["uv run --frozen pde-agent"]
    assert Path(dispatch["artifacts"]["session_path"]).exists()


def test_handle_chat_input_supports_natural_language_and_slash_commands(
    tmp_path: Path,
) -> None:
    from pde_agent.core.interactive_cockpit import handle_chat_input

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "early_rel_mse": 0.63,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 1.92},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )

    natural = handle_chat_input(
        text="继续优化 early/middle",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        dialogue_output_dir=tmp_path / "dialogue",
        workspace_dir=tmp_path,
        force=True,
    )
    assert natural["kind"] == "natural_language"
    assert natural["request"]["freeform_user_message"] == "继续优化 early/middle"
    assert natural["active_run_dir"] == str(run_dir)

    runs = handle_chat_input(
        text="/runs",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        dialogue_output_dir=tmp_path / "dialogue",
        workspace_dir=tmp_path,
        force=True,
    )
    assert runs["kind"] == "slash_command"
    assert runs["command"] == "runs"
    assert "run_a" in runs["text"]

    context = handle_chat_input(
        text="/context",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        dialogue_output_dir=tmp_path / "dialogue",
        workspace_dir=tmp_path,
        force=True,
    )
    assert context["kind"] == "slash_command"
    assert context["command"] == "context"
    assert "FNO Evidence" in context["text"]

    switch = handle_chat_input(
        text="/run run_a",
        runs_root=tmp_path / "runs",
        active_run_dir=None,
        dialogue_output_dir=tmp_path / "dialogue",
        workspace_dir=tmp_path,
        force=True,
    )
    assert switch["kind"] == "slash_command"
    assert switch["command"] == "run"
    assert switch["active_run_dir"] == str(run_dir)

    dispatch = handle_chat_input(
        text="/dispatch --dry-run",
        runs_root=tmp_path / "runs",
        active_run_dir=run_dir,
        dialogue_output_dir=tmp_path / "dialogue",
        workspace_dir=tmp_path,
        force=True,
    )
    assert dispatch["kind"] == "slash_command"
    assert dispatch["command"] == "dispatch"
    assert dispatch["dispatch"]["dry_run"] is True


def test_handle_chat_input_rejects_unknown_slash_command(tmp_path: Path) -> None:
    from pde_agent.core.interactive_cockpit import handle_chat_input

    result = handle_chat_input(
        text="/finite-state-machine",
        runs_root=tmp_path / "runs",
        active_run_dir=None,
        dialogue_output_dir=tmp_path / "dialogue",
        workspace_dir=tmp_path,
        force=True,
    )

    assert result["kind"] == "error"
    assert "Unknown slash command" in result["text"]


def test_chat_help_frames_slash_commands_as_shell_helpers(tmp_path: Path) -> None:
    from pde_agent.core.interactive_cockpit import handle_chat_input

    result = handle_chat_input(
        text="/help",
        runs_root=tmp_path / "runs",
        active_run_dir=None,
        dialogue_output_dir=tmp_path / "dialogue",
        workspace_dir=tmp_path,
        force=True,
    )

    assert result["kind"] == "slash_command"
    assert "PDEAgent shell" in result["text"]
    assert "/context" in result["text"]
    assert "not scientific workflow states" in result["text"]
