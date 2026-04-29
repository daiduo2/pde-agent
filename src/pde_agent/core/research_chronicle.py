"""Build a compact auditable research chronicle from Hermes artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "RESEARCH_CHRONICLE_BUILT"


class ResearchChronicleError(RuntimeError):
    pass


def build_research_chronicle(
    *,
    artifact_paths: list[str | Path],
    output_path: str | Path,
    submission_dir: str | Path | None = None,
    task_ids: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not artifact_paths:
        raise ResearchChronicleError("at least one artifact_path is required")
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise ResearchChronicleError(f"chronicle already exists: {output_file}")
    records = [_artifact_record(Path(path)) for path in artifact_paths]
    chronicle = _render_chronicle(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(chronicle, encoding="utf-8")
    updated_logs: list[str] = []
    if submission_dir is not None:
        for task_id in task_ids or [1, 2]:
            log_path = Path(submission_dir) / f"task{task_id}_logs.log"
            if log_path.exists():
                _append_chronicle(log_path=log_path, chronicle=chronicle)
                updated_logs.append(str(log_path))
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "artifacts": {
            "chronicle_path": str(output_file),
            "updated_log_paths": updated_logs,
        },
        "metrics": {
            "artifact_count": len(records),
            "updated_log_count": len(updated_logs),
        },
        "next_command": "audit-competition-submission",
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ResearchChronicleError(f"artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResearchChronicleError(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResearchChronicleError(f"artifact must be a JSON object: {path}")
    return {
        "path": str(path),
        "payload": payload,
        "kind": _artifact_kind(payload),
    }


def _render_chronicle(records: list[dict[str, Any]]) -> str:
    lines = [
        "Hermes Research Chronicle",
        "",
        "Evidence Sources:",
    ]
    for index, record in enumerate(records, start=1):
        payload = record["payload"]
        lines.append(
            f"- [{index}] kind={record['kind']} state={_state(payload)} path={record['path']}"
        )
    lines.extend(["", "Agent Research Trajectory:"])
    for record in records:
        lines.extend(_trajectory_lines(record))
    lines.extend(["", "Failed / Rejected Experiments:"])
    failure_lines = _failure_lines(records)
    lines.extend(failure_lines or ["- No failed experiment was detected in the provided artifacts."])
    lines.extend(["", "Strategy Decisions:"])
    decision_lines = _decision_lines(records)
    lines.extend(decision_lines or ["- No explicit strategy decision artifact was provided."])
    lines.extend(["", "Local Strategy Loop:"])
    loop_lines = _local_strategy_loop_lines(records)
    lines.extend(loop_lines or ["- No local strategy loop artifact was provided."])
    lines.extend(["", "Submission Decision:"])
    submission_lines = _submission_lines(records)
    lines.extend(submission_lines or ["- No competition submission pipeline artifact was provided."])
    lines.extend(["", "Arxiv Retrieval Policy:"])
    lines.extend(
        [
            "- Literature retrieval is useful as a source of candidate interventions, not as direct evidence of improvement.",
            "- Retrieved methods should be converted into declarative claims, implementation plans, expected observables, and known risks.",
            "- A strategy should only be internalized after local execution produces audited evidence with non-regressing score, runtime, and physics diagnostics.",
            "- For the current submission package, any selected operator candidate must pass shape, context-copy, runtime, and packaging audits before submission.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _trajectory_lines(record: dict[str, Any]) -> list[str]:
    payload = record["payload"]
    kind = record["kind"]
    if kind == "hermes_audit":
        return [
            "- Hermes audit session="
            + str(payload.get("session_id", "unknown"))
            + " arm="
            + str(payload.get("arm", "unknown"))
            + " status="
            + str(payload.get("status", "unknown"))
            + " tool_calls="
            + json.dumps(payload.get("tool_call_summary", {}), sort_keys=True),
        ]
    if kind == "strategy_benchmark":
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        return [
            "- Strategy benchmark completed with best_strategy_id="
            + str(metrics.get("best_strategy_id", "unknown"))
            + ", mean_score_delta="
            + str(metrics.get("mean_score_delta", "unknown"))
            + ", failure_rate="
            + str(metrics.get("failure_rate", "unknown"))
        ]
    if kind == "competition_pipeline":
        return [
            "- Competition pipeline task="
            + str(payload.get("task_id", "unknown"))
            + " model_family="
            + str(payload.get("model_family", "unknown"))
            + " audit="
                + str(_nested(payload, ["audit", "decision"], "unknown"))
        ]
    if kind == "local_strategy_loop":
        return [
            "- Hermes local strategy loop ran iteration_count="
            + str(payload.get("iteration_count", "unknown"))
            + " decision="
            + str(_nested(payload, ["decision", "decision"], "unknown"))
            + " next="
            + str(_nested(payload, ["decision", "next_command"], "unknown"))
        ]
    if kind == "local_strategy_leaderboard":
        return [
            "- Local strategy leaderboard result_count="
            + str(payload.get("result_count", "unknown"))
            + " best_strategy="
            + str(_nested(payload, ["best_result", "strategy_id"], "unknown"))
            + " score_delta="
            + str(_nested(payload, ["best_result", "score_delta"], "unknown"))
        ]
    return ["- Recorded artifact state=" + _state(payload)]


def _failure_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        payload = record["payload"]
        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            mean_delta = metrics.get("mean_score_delta")
            failure_rate = metrics.get("failure_rate")
            if isinstance(mean_delta, int | float) and mean_delta < 0:
                lines.append(
                    "- Negative score delta observed: mean_score_delta="
                    + str(mean_delta)
                    + " in "
                    + record["path"]
                )
            if isinstance(failure_rate, int | float) and failure_rate > 0:
                lines.append(
                    "- Nonzero failure rate observed: failure_rate="
                    + str(failure_rate)
                    + " in "
                    + record["path"]
                )
        decision = payload.get("decision")
        if isinstance(decision, dict):
            reasons = decision.get("reason_codes")
            if isinstance(reasons, list) and reasons:
                lines.append("- Decision reason_codes: " + ", ".join(map(str, reasons)))
    return lines


def _decision_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        payload = record["payload"]
        decision = payload.get("decision")
        if isinstance(decision, dict):
            lines.append(
                "- decision="
                + str(decision.get("decision", "unknown"))
                + " reason_codes="
                + ", ".join(map(str, decision.get("reason_codes", [])))
            )
        terminal = payload.get("terminal_artifact")
        if isinstance(terminal, dict):
            lines.append(
                "- terminal_strategy="
                + str(terminal.get("strategy_id", "unknown"))
                + " state="
                + str(terminal.get("state_hint", "unknown"))
            )
    return lines


def _submission_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        payload = record["payload"]
        if record["kind"] != "competition_pipeline":
            continue
        metrics = _nested(payload, ["materialized", "submission", "metrics"], {})
        if not isinstance(metrics, dict):
            metrics = {}
        lines.append(
            "- task="
            + str(payload.get("task_id", "unknown"))
            + " model_family="
            + str(payload.get("model_family", "unknown"))
            + " audit_decision="
            + str(_nested(payload, ["audit", "decision"], "unknown"))
            + " issue_count="
            + str(_nested(payload, ["audit", "issue_count"], "unknown"))
            + " prediction_shape="
            + str(metrics.get("prediction_shape", "unknown"))
            + " inference_time="
            + str(metrics.get("inference_time", "unknown"))
        )
    return lines


def _local_strategy_loop_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        payload = record["payload"]
        if record["kind"] == "local_strategy_loop":
            lines.append(
                "- loop path="
                + record["path"]
                + " iteration_count="
                + str(payload.get("iteration_count", "unknown"))
                + " decision="
                + str(_nested(payload, ["decision", "decision"], "unknown"))
                + " next="
                + str(_nested(payload, ["decision", "next_command"], "unknown"))
            )
            iterations = payload.get("iterations")
            if isinstance(iterations, list):
                for item in iterations:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        "- iteration="
                        + str(item.get("iteration", "unknown"))
                        + " strategy="
                        + str(item.get("strategy_id", "unknown"))
                        + " leaderboard_decision="
                        + str(item.get("leaderboard_decision", "unknown"))
                        + " route_command="
                        + str(item.get("route_command", "unknown"))
                    )
        if record["kind"] == "local_strategy_leaderboard":
            lines.append(
                "- leaderboard path="
                + record["path"]
                + " result_count="
                + str(payload.get("result_count", "unknown"))
                + " best_strategy="
                + str(_nested(payload, ["best_result", "strategy_id"], "unknown"))
                + " score_delta="
                + str(_nested(payload, ["best_result", "score_delta"], "unknown"))
                + " decision="
                + str(_nested(payload, ["decision", "decision"], "unknown"))
            )
    return lines


def _artifact_kind(payload: dict[str, Any]) -> str:
    state = _state(payload)
    if "session_id" in payload and "tool_call_summary" in payload:
        return "hermes_audit"
    if state == "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED":
        return "strategy_benchmark"
    if state == "COMPETITION_TASK_PIPELINE_COMPLETED":
        return "competition_pipeline"
    if state == "HERMES_LOCAL_STRATEGY_LOOP_RAN":
        return "local_strategy_loop"
    if state == "LOCAL_STRATEGY_LEADERBOARD_UPDATED":
        return "local_strategy_leaderboard"
    return "generic"


def _state(payload: dict[str, Any]) -> str:
    return str(payload.get("state_hint", payload.get("status", "unknown")))


def _nested(payload: dict[str, Any], keys: list[str], default: Any) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _append_chronicle(*, log_path: Path, chronicle: str) -> None:
    original = log_path.read_text(encoding="utf-8")
    marker = "\nHermes Research Chronicle\n"
    if marker in original:
        original = original.split(marker, maxsplit=1)[0].rstrip() + "\n"
    log_path.write_text(original.rstrip() + "\n\n" + chronicle, encoding="utf-8")
