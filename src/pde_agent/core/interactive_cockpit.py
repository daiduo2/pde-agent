
import json
import shlex
from pathlib import Path
from typing import Any

from pde_agent.core.agent_runtime import run_claude_agent_sdk_runtime
from pde_agent.core.supervisor_tui import build_supervisor_run_view
from pde_agent.core.supervisor_tui import build_supervisor_runs_view


def build_cockpit_snapshot(
    *,
    runs_root: str | Path,
    active_run_dir: str | Path | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    runs = build_supervisor_runs_view(runs_root=runs_root, limit=limit, wide=True)
    active = (
        build_supervisor_run_view(run_dir=active_run_dir)
        if active_run_dir is not None
        else _empty_active_run()
    )
    return {
        "schema_version": 1,
        "state_hint": "INTERACTIVE_COCKPIT_SNAPSHOT_BUILT",
        "read_only_observation": True,
        "runs_root": str(Path(runs_root)),
        "active_run_dir": str(Path(active_run_dir)) if active_run_dir is not None else None,
        "runs": runs["runs"],
        "runs_text": runs["text"],
        "active_run": active,
        "active_run_text": active.get("text", ""),
    }


FNO_TRAINING_LOOP_PANEL_STATE_HINT = "FNO_TRAINING_LOOP_PANEL_BUILT"


def build_fno_training_loop_snapshot(
    *,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_dir).expanduser().resolve()

    # Find all memory files
    memory_files = sorted(workspace.glob("fno_training_memory_*.json"))
    convergence_files = sorted(workspace.glob("fno_training_convergence_*.json"))

    score_history: list[dict[str, Any]] = []
    blocked_families: list[str] = []
    current_decision = "unknown"
    last_checkpoint: str | None = None
    best_candidate: dict[str, Any] | None = None

    for mem_path in memory_files:
        mem = _load_json(mem_path)
        if mem is None:
            continue
        iteration_index = mem.get("iteration_index", 0)
        best_so_far = mem.get("best_so_far", {})
        if best_so_far and isinstance(best_so_far, dict) and best_so_far.get("score") is not None:
            score_history.append({
                "iteration": iteration_index,
                "score": best_so_far.get("score"),
                "candidate_id": best_so_far.get("candidate_id"),
            })
            if best_candidate is None or (best_so_far.get("score") or 0) > (best_candidate.get("score") or 0):
                best_candidate = dict(best_so_far)

        family_outcomes = mem.get("family_outcomes", {})
        if isinstance(family_outcomes, dict):
            for family, outcome in family_outcomes.items():
                if isinstance(outcome, dict) and outcome.get("status") == "blocked" and family not in blocked_families:
                    blocked_families.append(family)

    # Read latest convergence
    if convergence_files:
        latest_conv = _load_json(convergence_files[-1])
        if latest_conv is not None:
            current_decision = str(latest_conv.get("decision", "unknown"))

    # Check for checkpoints
    checkpoint_files = sorted(workspace.glob("checkpoint_*.json"))
    if checkpoint_files:
        last_checkpoint = str(checkpoint_files[-1])

    # Check for final report
    final_report_path = workspace / "fno_training_final_report.json"
    has_final_report = final_report_path.is_file()

    panel_text = _build_training_loop_panel_text(
        iteration_count=len(memory_files),
        score_history=score_history,
        current_decision=current_decision,
        blocked_families=blocked_families,
        last_checkpoint=last_checkpoint,
        has_final_report=has_final_report,
        best_candidate=best_candidate,
    )

    return {
        "schema_version": 1,
        "state_hint": FNO_TRAINING_LOOP_PANEL_STATE_HINT,
        "workspace_dir": str(workspace),
        "iteration_count": len(memory_files),
        "score_history": score_history,
        "current_decision": current_decision,
        "blocked_families": blocked_families,
        "last_checkpoint": last_checkpoint,
        "has_final_report": has_final_report,
        "best_candidate": best_candidate,
        "panel_text": panel_text,
    }


def _build_training_loop_panel_text(
    *,
    iteration_count: int,
    score_history: list[dict[str, Any]],
    current_decision: str,
    blocked_families: list[str],
    last_checkpoint: str | None,
    has_final_report: bool,
    best_candidate: dict[str, Any] | None,
) -> str:
    decision_color = {
        "converged": "green",
        "continue": "blue",
        "escalate": "yellow",
        "stagnant": "yellow",
        "doom_loop": "red",
        "error": "red",
    }.get(current_decision, "white")

    lines = [
        "=== FNO Training Loop Monitor ===",
        "",
        f"Iterations: {iteration_count}",
        f"Decision:   {current_decision} ({decision_color})",
        f"Final report: {'yes' if has_final_report else 'no'}",
        "",
        "--- Score History ---",
    ]
    for entry in score_history:
        lines.append(f"  iter {entry['iteration']}: score={entry['score']}, candidate={entry.get('candidate_id', 'N/A')}")
    if not score_history:
        lines.append("  (no scores yet)")

    lines.append("")
    lines.append("--- Best Candidate ---")
    if best_candidate:
        lines.append(f"  id:    {best_candidate.get('candidate_id', 'N/A')}")
        lines.append(f"  score: {best_candidate.get('score', 'N/A')}")
        lines.append(f"  family: {best_candidate.get('model_family', 'N/A')}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("--- Blocked Families ---")
    if blocked_families:
        for family in blocked_families:
            lines.append(f"  - {family}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("--- Last Checkpoint ---")
    lines.append(f"  {last_checkpoint or '(none)'}")

    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def build_agent_dialogue_request(
    *,
    user_message: str,
    runs_root: str | Path,
    active_run_dir: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    message = user_message.strip()
    if not message:
        raise ValueError("user_message must be non-empty")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "agent_dialogue_request.json"
    prompt_path = root / "agent_dialogue_prompt.md"
    if not force:
        existing = [path for path in [request_path, prompt_path] if path.exists()]
        if existing:
            raise FileExistsError(f"dialogue artifacts already exist: {existing[0]}")
    snapshot = build_cockpit_snapshot(
        runs_root=runs_root,
        active_run_dir=active_run_dir,
        limit=20,
    )
    prompt = _build_dialogue_prompt(user_message=message, snapshot=snapshot)
    request = {
        "schema_version": 1,
        "state_hint": "AGENT_DIALOGUE_REQUEST_BUILT",
        "freeform_user_message": message,
        "not_finite_state_machine": True,
        "runs_root": str(Path(runs_root)),
        "active_run_dir": str(Path(active_run_dir)),
        "supervisor_constraints": {
            "allows_agent_algorithm_design": True,
            "allows_agent_training_experiments": True,
            "allows_agent_code_change_proposals": True,
            "requires_artifact_audit": True,
            "requires_failure_report": True,
            "requires_budget_statement": True,
        },
        "artifacts": {
            "request_path": str(request_path),
            "prompt_path": str(prompt_path),
        },
        "next_command": "run-claude-agent-sdk-runtime",
    }
    prompt_path.write_text(prompt, encoding="utf-8")
    request_path.write_text(
        json.dumps(request, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return request


def dispatch_agent_dialogue_request(
    *,
    request_path: str | Path,
    output_path: str | Path,
    workspace_dir: str | Path,
    dry_run: bool,
    force: bool,
    on_message: Any | None = None,
) -> dict[str, Any]:
    request_file = Path(request_path)
    request = _read_request(request_file)
    prompt_path = Path(str(request["artifacts"]["prompt_path"]))
    session = run_claude_agent_sdk_runtime(
        prompt=prompt_path.read_text(encoding="utf-8"),
        output_path=Path(output_path),
        workspace_dir=Path(workspace_dir),
        allowed_tools=["Read", "Glob", "Grep", "Bash"],
        disallowed_tools=["Edit", "Write", "NotebookEdit", "WebFetch"],
        allowed_bash_prefixes=["uv run --frozen pde-agent"],
        dry_run=dry_run,
        force=force,
        on_message=on_message,
    )
    return {
        "schema_version": 1,
        "state_hint": "AGENT_DIALOGUE_DISPATCHED",
        "request_path": str(request_file),
        "prompt_path": str(prompt_path),
        "dry_run": bool(dry_run),
        "runtime_state_hint": session.get("state_hint"),
        "artifacts": {
            "session_path": session["artifacts"]["session_path"],
        },
        "metrics": {
            "sdk_message_count": session.get("sdk_message_count"),
        },
    }


def handle_chat_input(
    *,
    text: str,
    runs_root: str | Path,
    active_run_dir: str | Path | None,
    dialogue_output_dir: str | Path,
    workspace_dir: str | Path,
    force: bool,
) -> dict[str, Any]:
    message = text.strip()
    if not message:
        return {"kind": "empty", "text": ""}
    if not message.startswith("/"):
        if active_run_dir is None:
            return {"kind": "error", "text": "Select an active run before sending agent dialogue."}
        request = build_agent_dialogue_request(
            user_message=message,
            runs_root=runs_root,
            active_run_dir=active_run_dir,
            output_dir=dialogue_output_dir,
            force=force,
        )
        return {
            "kind": "natural_language",
            "active_run_dir": str(Path(active_run_dir)),
            "request": request,
            "text": "\n".join(
                [
                    "Agent request prepared.",
                    f"prompt: {request['artifacts']['prompt_path']}",
                    "TUI: executing now. CLI: use /dispatch --dry-run or /dispatch --execute.",
                ]
            ),
        }
    return _handle_slash_command(
        message=message,
        runs_root=Path(runs_root),
        active_run_dir=Path(active_run_dir) if active_run_dir is not None else None,
        dialogue_output_dir=Path(dialogue_output_dir),
        workspace_dir=Path(workspace_dir),
        force=force,
    )


def _empty_active_run() -> dict[str, Any]:
    return {
        "state_hint": "SUPERVISOR_TUI_RUN_VIEW_EMPTY",
        "read_only": True,
        "text": "No active run selected.",
    }


def _handle_slash_command(
    *,
    message: str,
    runs_root: Path,
    active_run_dir: Path | None,
    dialogue_output_dir: Path,
    workspace_dir: Path,
    force: bool,
) -> dict[str, Any]:
    try:
        parts = shlex.split(message)
    except ValueError as exc:
        return {"kind": "error", "text": f"Invalid slash command: {exc}"}
    if not parts:
        return {"kind": "empty", "text": ""}
    command = parts[0][1:]
    args = parts[1:]
    if command == "help":
        return {"kind": "slash_command", "command": command, "text": _slash_help()}
    if command == "runs":
        snapshot = build_cockpit_snapshot(runs_root=runs_root, active_run_dir=active_run_dir)
        return {
            "kind": "slash_command",
            "command": command,
            "active_run_dir": str(active_run_dir) if active_run_dir is not None else None,
            "text": snapshot["runs_text"],
        }
    if command == "run":
        if not args:
            return {"kind": "error", "text": "Usage: /run <run_id-or-path>"}
        target = Path(args[0])
        selected = target if target.exists() else runs_root / args[0]
        if not selected.exists():
            return {"kind": "error", "text": f"Run not found: {args[0]}"}
        snapshot = build_cockpit_snapshot(runs_root=runs_root, active_run_dir=selected)
        return {
            "kind": "slash_command",
            "command": command,
            "active_run_dir": str(selected),
            "text": snapshot["active_run_text"],
        }
    if command in {"context", "evidence"}:
        if active_run_dir is None:
            return {"kind": "error", "text": "No active run selected."}
        snapshot = build_cockpit_snapshot(runs_root=runs_root, active_run_dir=active_run_dir)
        if command == "context":
            return {
                "kind": "slash_command",
                "command": command,
                "active_run_dir": str(active_run_dir),
                "text": snapshot["active_run_text"],
            }
        fno = snapshot["active_run"].get("fno_evidence", {})
        return {
            "kind": "slash_command",
            "command": command,
            "active_run_dir": str(active_run_dir),
            "text": json.dumps(fno, indent=2, sort_keys=True),
        }
    if command == "prompt":
        request_path = dialogue_output_dir / "agent_dialogue_request.json"
        if not request_path.exists():
            return {"kind": "error", "text": "No dialogue request exists yet."}
        request = _read_request(request_path)
        prompt_path = request["artifacts"]["prompt_path"]
        return {
            "kind": "slash_command",
            "command": command,
            "active_run_dir": str(active_run_dir) if active_run_dir is not None else None,
            "text": Path(str(prompt_path)).read_text(encoding="utf-8"),
        }
    if command == "dispatch":
        request_path = dialogue_output_dir / "agent_dialogue_request.json"
        if not request_path.exists():
            return {"kind": "error", "text": "No dialogue request exists yet."}
        dry_run = "--execute" not in args
        dispatch = dispatch_agent_dialogue_request(
            request_path=request_path,
            output_path=dialogue_output_dir / (
                "agent_session_dry_run.json" if dry_run else "agent_session.json"
            ),
            workspace_dir=workspace_dir,
            dry_run=dry_run,
            force=force,
        )
        return {
            "kind": "slash_command",
            "command": command,
            "active_run_dir": str(active_run_dir) if active_run_dir is not None else None,
            "dispatch": dispatch,
            "text": f"Dispatch artifact: {dispatch['artifacts']['session_path']}",
        }
    return {"kind": "error", "text": f"Unknown slash command: /{command}\n{_slash_help()}"}


def _slash_help() -> str:
    return "\n".join(
        [
            "PDEAgent shell",
            "",
            "Type natural language in the TUI to start a free-form agent session.",
            "",
            "Slash commands:",
            "/runs                         list recent runs",
            "/run <run_id-or-path>         select and inspect a run",
            "/context                      show active run context",
            "/evidence                     show active FNO evidence JSON",
            "/prompt                       print latest generated agent prompt",
            "/dispatch --dry-run           audit latest request without executing",
            "/dispatch --execute           manually replay latest prompt in the agent runtime",
            "/help                         show this help",
            "",
            "Slash commands are shell helpers, not scientific workflow states.",
        ]
    )


def _read_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dialogue request must be a JSON object")
    if payload.get("state_hint") != "AGENT_DIALOGUE_REQUEST_BUILT":
        raise ValueError("dialogue request has invalid state_hint")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts.get("prompt_path"):
        raise ValueError("dialogue request is missing prompt artifact")
    return payload


def _build_dialogue_prompt(*, user_message: str, snapshot: dict[str, Any]) -> str:
    active = snapshot["active_run"]
    fno = active.get("fno_evidence", {}) if isinstance(active, dict) else {}
    session = active.get("agent_session", {}) if isinstance(active, dict) else {}
    return "\n".join(
        [
            "# PDEAgent Interactive Cockpit Dialogue",
            "",
            "You are an autonomous PDE research agent supervised through a chat-style shell.",
            "Do not reduce the request to a finite state-machine transition.",
            "You may propose new algorithms, code changes, or training experiments when justified by evidence.",
            "Use artifacts and command outputs as evidence. Record failures explicitly.",
            "",
            "## User Message",
            user_message,
            "",
            "## Active Run Context",
            f"- run_dir: {snapshot.get('active_run_dir')}",
            f"- selected_candidate: {fno.get('selected_candidate_id')}",
            f"- early_rel_mse: {fno.get('early_rel_mse')}",
            f"- middle_rel_mse: {fno.get('middle_rel_mse')}",
            f"- late_evaluated: {fno.get('late_evaluated')}",
            f"- agent_session_state: {session.get('state_hint')}",
            "",
            "## Current Run View",
            snapshot.get("active_run_text", ""),
            "",
            "## Supervisor Contract",
            "- Preserve free-form reasoning and algorithm design freedom.",
            "- Prefer proposing executable experiments over choosing from a fixed enum.",
            "- Before execution, state budget, artifact targets, and verification plan.",
            "- After execution, report produced artifacts, metrics, failures, and next constraints.",
        ]
    )
