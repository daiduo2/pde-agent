
import json
import time
from pathlib import Path
from typing import Any


PIPELINE_ARTIFACTS = [
    "fno_training_plan.json",
    "optimization/fno_training_optimization_report.json",
    "fno_training_optimization_evidence.json",
    "hypothesis_algorithm_design.json",
    "hypothesis_validation_result.json",
    "algorithm_candidate_promotion.json",
    "strategy_proposal.json",
    "strategy_evidence_summary.json",
]


def build_supervisor_runs_view(
    *,
    runs_root: str | Path,
    limit: int = 20,
    wide: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    root = Path(runs_root)
    timestamp = time.time() if now is None else float(now)
    run_dirs = [
        item
        for item in root.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ] if root.exists() else []
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    rows = [_summarize_run(path, now=timestamp) for path in run_dirs[: max(limit, 0)]]
    report = {
        "schema_version": 1,
        "state_hint": "SUPERVISOR_TUI_RUNS_VIEW_BUILT",
        "read_only": True,
        "runs_root": str(root),
        "run_count": len(rows),
        "wide": bool(wide),
        "runs": rows,
    }
    report["text"] = _render_runs_table(rows, wide=wide)
    return report


def build_supervisor_run_view(*, run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    fno = _extract_fno_evidence(root)
    pipeline = [_artifact_status(root, name) for name in PIPELINE_ARTIFACTS]
    session = _extract_agent_session(root)
    report = {
        "schema_version": 1,
        "state_hint": "SUPERVISOR_TUI_RUN_VIEW_BUILT",
        "read_only": True,
        "run_id": root.name,
        "run_dir": str(root),
        "pipeline": pipeline,
        "fno_evidence": fno,
        "agent_session": session,
    }
    report["text"] = _render_run_detail(report)
    return report


def _summarize_run(run_dir: Path, *, now: float) -> dict[str, Any]:
    fno = _extract_fno_evidence(run_dir)
    session = _extract_agent_session(run_dir)
    updated_at = run_dir.stat().st_mtime
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "updated_at": updated_at,
        "age_seconds": max(0.0, now - updated_at),
        "age": _format_age(max(0.0, now - updated_at)),
        "pipeline_state": _latest_pipeline_state(run_dir),
        "agent_state": session.get("state_hint"),
        "selected_candidate_id": fno.get("selected_candidate_id"),
        "fno_selection_score": fno.get("fno_selection_score"),
        "early_rel_mse": fno.get("early_rel_mse"),
        "early_score": fno.get("early_score"),
        "middle_rel_mse": fno.get("middle_rel_mse"),
        "middle_score": fno.get("middle_score"),
        "late_rel_mse": fno.get("late_rel_mse"),
        "late_score": fno.get("late_score"),
        "late_evaluated": fno.get("late_evaluated"),
        "selection_reasons": fno.get("selection_reasons", []),
    }


def _latest_pipeline_state(run_dir: Path) -> str | None:
    for artifact in reversed(PIPELINE_ARTIFACTS):
        payload = _read_json(run_dir / artifact)
        if payload:
            state = payload.get("state_hint")
            return str(state) if state else artifact
    return None


def _artifact_status(run_dir: Path, artifact: str) -> dict[str, Any]:
    path = run_dir / artifact
    payload = _read_json(path)
    return {
        "artifact": artifact,
        "path": str(path),
        "exists": path.exists(),
        "state_hint": payload.get("state_hint") if payload else None,
    }


def _extract_fno_evidence(run_dir: Path) -> dict[str, Any]:
    evidence_path = _select_fno_evidence_path(run_dir)
    payload = _read_json(evidence_path) if evidence_path is not None else {}
    if not payload:
        return {}
    selected = payload.get("selected_candidate")
    if not isinstance(selected, dict):
        return {}
    middle = selected.get("middle_guardrail")
    if not isinstance(middle, dict):
        middle = {}
    late = selected.get("late_guardrail")
    if not isinstance(late, dict):
        late = {}
    return {
        "artifact": evidence_path.name if evidence_path is not None else None,
        "path": str(evidence_path) if evidence_path is not None else None,
        "state_hint": payload.get("state_hint"),
        "selected_candidate_id": selected.get("candidate_id"),
        "fno_selection_score": selected.get("fno_selection_score"),
        "config": selected.get("config") if isinstance(selected.get("config"), dict) else {},
        "early_rel_mse": selected.get("early_rel_mse"),
        "early_score": selected.get("early_score"),
        "middle_evaluated": bool(middle.get("evaluated")),
        "middle_rel_mse": middle.get("rel_mse"),
        "middle_score": middle.get("score"),
        "late_evaluated": bool(late.get("evaluated")),
        "late_rel_mse": late.get("rel_mse"),
        "late_score": late.get("score"),
        "selection_reasons": selected.get("selection_reason_codes", []),
    }


def _select_fno_evidence_path(run_dir: Path) -> Path | None:
    candidates = [
        path
        for path in run_dir.glob("fno_training_optimization_evidence*.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_agent_session(run_dir: Path) -> dict[str, Any]:
    sessions = sorted(run_dir.glob("*session*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in sessions:
        payload = _read_json(path)
        if not payload:
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
        return {
            "path": str(path),
            "state_hint": payload.get("state_hint"),
            "sdk_message_count": metrics.get("sdk_message_count"),
            "dry_run": metrics.get("dry_run"),
            "error_count": len(errors),
        }
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _render_runs_table(rows: list[dict[str, Any]], *, wide: bool) -> str:
    headers = ["run", "age", "state", "agent", "candidate", "early_rel", "middle_rel", "late"]
    widths = [56, 8, 42, 34, 32, 10, 10, 8] if wide else [24, 8, 34, 28, 24, 10, 10, 8]
    lines = [_format_row(headers, widths), _format_row(["-" * width for width in widths], widths)]
    for row in rows:
        lines.append(
            _format_row(
                [
                    row.get("run_id"),
                    row.get("age"),
                    row.get("pipeline_state"),
                    row.get("agent_state"),
                    row.get("selected_candidate_id"),
                    _fmt(row.get("early_rel_mse")),
                    _fmt(row.get("middle_rel_mse")),
                    "yes" if row.get("late_evaluated") else "no",
                ],
                widths,
            )
        )
    return "\n".join(lines)


def _render_run_detail(report: dict[str, Any]) -> str:
    fno = report["fno_evidence"]
    session = report["agent_session"]
    lines = [
        f"Run: {report['run_id']}",
        "",
        "Pipeline",
    ]
    for item in report["pipeline"]:
        mark = "ok" if item["exists"] else "missing"
        lines.append(f"- {mark:7} {item['artifact']} {item.get('state_hint') or ''}")
    lines.extend(
        [
            "",
            "FNO Evidence",
            f"- artifact: {fno.get('artifact')}",
            f"- selected: {fno.get('selected_candidate_id')}",
            f"- score: {_fmt(fno.get('fno_selection_score'))}",
            f"- early_rel_mse: {_fmt(fno.get('early_rel_mse'))}",
            f"- middle_rel_mse: {_fmt(fno.get('middle_rel_mse'))}",
            f"- late_evaluated: {fno.get('late_evaluated')}",
            "",
            "Agent Session",
            f"- state: {session.get('state_hint')}",
            f"- sdk_message_count: {session.get('sdk_message_count')}",
            f"- error_count: {session.get('error_count')}",
        ]
    )
    return "\n".join(lines)


def _format_row(values: list[Any], widths: list[int]) -> str:
    return "  ".join(_truncate(value, width).ljust(width) for value, width in zip(values, widths))


def _truncate(value: Any, width: int) -> str:
    text = str(value or "")
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return str(value)
    return "" if value is None else str(value)


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60.0
    if hours < 48:
        return f"{int(hours)}h"
    return f"{int(hours / 24.0)}d"
