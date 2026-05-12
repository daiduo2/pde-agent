
import json
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from pde_agent.core.interactive_cockpit import build_cockpit_snapshot
from pde_agent.core.interactive_cockpit import dispatch_agent_dialogue_request
from pde_agent.core.interactive_cockpit import handle_chat_input


class PDEAgentCockpitApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #context {
        height: 5;
        border: solid $accent;
    }
    #main {
        height: 1fr;
    }
    #transcript {
        height: 1fr;
        border: solid $accent;
    }
    #agent_input {
        height: 3;
    }
    """
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        runs_root: str | Path = "runs",
        active_run_dir: str | Path | None = None,
        dialogue_output_dir: str | Path = "runs/cockpit_dialogue",
    ) -> None:
        super().__init__()
        self.runs_root = Path(runs_root)
        self.active_run_dir = Path(active_run_dir) if active_run_dir else None
        self.dialogue_output_dir = Path(dialogue_output_dir)
        self.snapshot: dict[str, object] = {}
        self.transcript: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="context")
        with Vertical(id="main"):
            yield RichLog(id="transcript", wrap=True, auto_scroll=True)
        yield Input(
            placeholder=(
                "Ask the agent in natural language. Slash helpers: /runs /run <id> /context /evidence "
                "/prompt /dispatch --dry-run /help"
            ),
            id="agent_input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_snapshot()
        self.query_one("#agent_input", Input).focus()

    def action_refresh(self) -> None:
        self.refresh_snapshot()

    def refresh_snapshot(self) -> None:
        if self.active_run_dir is None:
            runs_view = build_cockpit_snapshot(runs_root=self.runs_root)
            runs = runs_view.get("runs", [])
            if runs:
                first = runs[0]
                if isinstance(first, dict) and first.get("run_dir"):
                    self.active_run_dir = Path(str(first["run_dir"]))
        self.snapshot = build_cockpit_snapshot(
            runs_root=self.runs_root,
            active_run_dir=self.active_run_dir,
        )
        self._render_context()
        if not self.transcript:
            self._append_transcript(
                "system: natural language starts an audited agent session. /help lists shell helpers."
            )
            self._append_transcript(
                f"system: active run is {self.active_run_dir}. Use /context for full run details."
            )

    def _render_context(self) -> None:
        active = self.snapshot.get("active_run", {})
        fno = active.get("fno_evidence", {}) if isinstance(active, dict) else {}
        self.query_one("#context", Static).update(
            "\n".join(
                [
                    "PDEAgent Cockpit",
                    f"active: {self.active_run_dir}",
                    (
                        "selected: "
                        f"{fno.get('selected_candidate_id')} | "
                        f"early_rel={fno.get('early_rel_mse')} | "
                        f"middle_rel={fno.get('middle_rel_mse')} | "
                        f"late={fno.get('late_evaluated')}"
                    ),
                ]
            )
        )

    def _render_transcript(self) -> None:
        log = self.query_one("#transcript", RichLog)
        log.clear()
        for entry in self.transcript[-100:]:
            log.write(entry)
        log.scroll_end(animate=False)

    def _append_transcript(self, text: str) -> None:
        self.transcript.append(text)
        try:
            log = self.query_one("#transcript", RichLog)
        except Exception:
            return
        log.write(text)
        log.scroll_end(animate=False)

    def _append_agent_event(self, message: str) -> None:
        event = _format_agent_event(message)
        if event:
            self._append_transcript(event)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "agent_input":
            return
        value = event.value.strip()
        if not value:
            return
        if value == "/dispatch --execute":
            self._append_transcript(f"> {value}")
            self._start_execute_dispatch()
            event.input.value = ""
            return
        try:
            result = handle_chat_input(
                text=value,
                runs_root=self.runs_root,
                active_run_dir=self.active_run_dir,
                dialogue_output_dir=self.dialogue_output_dir,
                workspace_dir=Path.cwd(),
                force=True,
            )
        except Exception as exc:
            result = {"kind": "error", "text": f"{type(exc).__name__}: {exc}"}
        if result.get("active_run_dir"):
            self.active_run_dir = Path(str(result["active_run_dir"]))
        self._append_transcript(f"> {value}")
        if result.get("kind") == "natural_language":
            self._append_transcript("agent: request accepted; starting session")
        else:
            self._append_transcript(str(result.get("text", "")))
        event.input.value = ""
        self.refresh_snapshot()
        if result.get("kind") == "natural_language":
            self._start_execute_dispatch()

    def _start_execute_dispatch(self) -> None:
        request_path = self.dialogue_output_dir / "agent_dialogue_request.json"
        if not request_path.exists():
            self._append_transcript("No dialogue request exists yet.")
            return
        output_path = self.dialogue_output_dir / "agent_session.json"
        self._append_transcript(f"agent: executing -> {output_path}")

        def worker() -> None:
            try:
                dispatch = dispatch_agent_dialogue_request(
                    request_path=request_path,
                    output_path=output_path,
                    workspace_dir=Path.cwd(),
                    dry_run=False,
                    force=True,
                    on_message=lambda message: self.call_from_thread(
                        self._append_agent_event,
                        message,
                    ),
                )
            except Exception as exc:
                self.call_from_thread(
                    self._append_transcript,
                    f"agent error: {type(exc).__name__}: {exc}",
                )
                return
            self.call_from_thread(
                self._append_transcript,
                f"agent: completed, session artifact: {dispatch['artifacts']['session_path']}",
            )
            self.call_from_thread(self.refresh_snapshot)

        self.run_worker(worker, thread=True, name="agent-dispatch", exit_on_error=False)


def _format_agent_event(raw_message: str) -> str:
    payload = _parse_message_payload(raw_message)
    if payload is None:
        return f"agent: {_compact_text(raw_message)}"
    event_type = str(payload.get("type") or payload.get("event") or payload.get("role") or "")
    if event_type:
        lowered = event_type.lower()
        if "tool" in lowered and "result" not in lowered:
            tool_name = payload.get("name") or payload.get("tool_name") or payload.get("tool")
            command = _nested_lookup(payload, ["input", "command"])
            suffix = f" {tool_name}" if tool_name else ""
            detail = f": {command}" if command else ""
            return f"tool:{suffix}{detail}"
        if lowered == "result":
            if payload.get("is_error") is True:
                text = payload.get("result") or payload.get("errors") or raw_message
                return f"agent error: {_compact_text(str(text))}"
            turns = payload.get("num_turns")
            duration = payload.get("duration_ms")
            result = payload.get("result")
            parts = ["agent done"]
            if turns is not None:
                parts.append(f"turns={turns}")
            if duration is not None:
                parts.append(f"duration_ms={duration}")
            if result:
                parts.append(_compact_text(str(result), limit=120))
            return ": ".join([parts[0], " ".join(parts[1:])]) if len(parts) > 1 else parts[0]
        if "result" in lowered:
            ok = payload.get("is_error") is not True
            label = "ok" if ok else "error"
            text = payload.get("content") or payload.get("output") or payload.get("text")
            return f"tool result ({label}): {_compact_text(str(text or ''))}"
        if lowered in {"assistant", "assistantmessage", "message"}:
            text = _extract_text(payload)
            return f"assistant: {_compact_text(text)}" if text else "assistant: ..."
        if "error" in lowered:
            text = payload.get("message") or payload.get("error") or raw_message
            return f"error: {_compact_text(str(text))}"
        text = _extract_text(payload)
        if text:
            return f"{event_type}: {_compact_text(text)}"
        return f"{event_type}: {_compact_text(raw_message)}"
    text = _extract_text(payload)
    return f"agent: {_compact_text(text)}" if text else f"agent: {_compact_text(raw_message)}"


def _parse_message_payload(raw_message: str) -> dict[str, Any] | None:
    stripped = raw_message.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_text(payload: dict[str, Any]) -> str:
    for key in ["text", "message", "content", "result"]:
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
    return ""


def _nested_lookup(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _compact_text(text: str, *, limit: int = 180) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."
