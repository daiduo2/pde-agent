"""Build a compact auditable research chronicle from Hermes artifacts."""


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
    lines.extend(["", "Operator Runtime Tuning:"])
    tuning_lines = _operator_tuning_lines(records)
    lines.extend(tuning_lines or ["- No operator runtime tuning artifact was provided."])
    lines.extend(["", "UNet Competition Prediction:"])
    unet_lines = _unet_prediction_lines(records)
    lines.extend(unet_lines or ["- No UNet competition prediction artifact was provided."])
    lines.extend(["", "Hermes Submission Candidate:"])
    candidate_lines = _submission_candidate_lines(records)
    lines.extend(candidate_lines or ["- No Hermes submission candidate artifact was provided."])
    lines.extend(["", "Repair Policy Memory:"])
    repair_policy_lines = _repair_policy_memory_lines(records)
    lines.extend(
        repair_policy_lines
        or ["- No repair policy memory artifact was provided."]
    )
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
    if kind == "runtime_audit":
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
    if kind == "operator_tuning":
        selected = payload.get("selected_config")
        if not isinstance(selected, dict):
            selected = {}
        return [
            "- Hermes tuned operator="
            + str(payload.get("operator_family", "unknown"))
            + " selected inference_batch_size="
            + str(selected.get("inference_batch_size", "unknown"))
            + " torch_num_threads="
            + str(selected.get("torch_num_threads", "unknown"))
            + " projected_full_inference_seconds="
            + str(payload.get("best_projected_full_inference_seconds", "unknown"))
        ]
    if kind == "unet_prediction":
        return [
            "- Hermes executed UNet competition prediction samples="
            + str(_nested(payload, ["inputs", "sample_count"], "unknown"))
            + " inference_time="
            + str(_nested(payload, ["inference", "elapsed_seconds"], "unknown"))
            + " rollout_seconds="
            + str(_nested(payload, ["inference", "phase_timings", "rollout_seconds"], "unknown"))
        ]
    if kind == "submission_candidate":
        return [
            "- Hermes materialized submission candidate audit="
            + str(_nested(payload, ["audit", "decision"], "unknown"))
            + " issue_count="
            + str(_nested(payload, ["audit", "issue_count"], "unknown"))
            + " next="
            + str(payload.get("next_command", "unknown"))
        ]
    if kind == "runtime_closed_loop":
        return [
            "- Hermes runtime closed loop best_score="
            + str(_nested(payload, ["heartbeat", "best_score"], "unknown"))
            + " decision="
            + str(_nested(payload, ["heartbeat", "decision"], "unknown"))
            + " repair_policy_action_count="
            + str(
                _nested(
                    payload,
                    ["metrics", "repair_policy_recommended_action_count"],
                    "unknown",
                )
            )
        ]
    if kind == "runtime_audit_chain":
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        return [
            "- Hermes runtime audit chain decision="
            + str(payload.get("decision", "unknown"))
            + " auto_repair_decision="
            + str(metrics.get("auto_repair_decision", "unknown"))
            + " memory_bridge_passed="
            + str(metrics.get("auto_repair_memory_bridge_passed", "unknown"))
        ]
    if kind == "adaptive_heartbeat":
        return [
            "- Hermes heartbeat best_candidate="
            + str(_nested(payload, ["best_candidate", "candidate_id"], "unknown"))
            + " best_score="
            + str(payload.get("best_score", "unknown"))
            + " decision="
            + str(payload.get("decision", "unknown"))
            + " repair_policy_action_count="
            + str(
                _nested(
                    payload,
                    ["parameters", "repair_policy_recommended_action_count"],
                    "unknown",
                )
            )
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


def _operator_tuning_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        if record["kind"] != "operator_tuning":
            continue
        payload = record["payload"]
        selected = payload.get("selected_config")
        if not isinstance(selected, dict):
            selected = {}
        lines.append(
            "- operator="
            + str(payload.get("operator_family", "unknown"))
            + " inference_batch_size="
            + str(selected.get("inference_batch_size", "unknown"))
            + " torch_num_threads="
            + str(selected.get("torch_num_threads", "unknown"))
            + " projected_full_inference_seconds="
            + str(payload.get("best_projected_full_inference_seconds", "unknown"))
            + " decision="
            + str(_nested(payload, ["decision", "action"], "unknown"))
        )
    return lines


def _unet_prediction_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        if record["kind"] != "unet_prediction":
            continue
        payload = record["payload"]
        phase = _nested(payload, ["inference", "phase_timings"], {})
        if not isinstance(phase, dict):
            phase = {}
        lines.append(
            "- task="
            + str(_nested(payload, ["candidate", "task_id"], "unknown"))
            + " samples="
            + str(_nested(payload, ["inputs", "sample_count"], "unknown"))
            + " forecast_steps="
            + str(_nested(payload, ["inputs", "forecast_steps"], "unknown"))
            + " inference_time="
            + str(_nested(payload, ["inference", "elapsed_seconds"], "unknown"))
            + " rollout_seconds="
            + str(phase.get("rollout_seconds", "unknown"))
            + " inference_batch_size="
            + str(_nested(payload, ["inference", "inference_batch_size"], "unknown"))
            + " torch_num_threads="
            + str(_nested(payload, ["inference", "torch_num_threads"], "unknown"))
        )
    return lines


def _submission_candidate_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        if record["kind"] != "submission_candidate":
            continue
        payload = record["payload"]
        metrics = _nested(payload, ["materialized", "submission", "metrics"], {})
        if not isinstance(metrics, dict):
            metrics = {}
        lines.append(
            "- operator="
            + str(payload.get("operator_family", "unknown"))
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


def _repair_policy_memory_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        payload = record["payload"]
        if record["kind"] == "runtime_closed_loop":
            candidate_plan = payload.get("candidate_plan")
            if not isinstance(candidate_plan, dict):
                candidate_plan = {}
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            nonconverged = _string_list(
                candidate_plan.get("repair_nonconvergence_errors")
            )
            converged = _string_list(candidate_plan.get("repair_convergence_errors"))
            lines.append(
                "- runtime path="
                + record["path"]
                + " runtime_action_count="
                + str(metrics.get("repair_policy_recommended_action_count", 0))
                + " nonconverged_repair_errors="
                + _join_or_none(nonconverged)
                + " converged_repair_errors="
                + _join_or_none(converged)
            )
            lines.extend(
                _repair_action_lines(
                    source="runtime",
                    actions=candidate_plan.get("repair_policy_recommended_actions"),
                )
            )
        if record["kind"] == "runtime_audit_chain":
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            repair_convergence = payload.get("repair_convergence")
            if not isinstance(repair_convergence, dict):
                repair_convergence = {}
            converged = _string_list(repair_convergence.get("input_repair_errors"))
            if repair_convergence.get("decision") != "converged":
                converged = []
            nonconverged = _string_list(
                repair_convergence.get("repeated_audit_errors")
            )
            lines.append(
                "- auto_repair path="
                + record["path"]
                + " auto_repair_decision="
                + str(metrics.get("auto_repair_decision", "unknown"))
                + " memory_bridge_passed="
                + str(metrics.get("auto_repair_memory_bridge_passed", "unknown"))
                + " auto_repair_action_count="
                + str(
                    metrics.get(
                        "auto_repair_repair_policy_recommended_action_count",
                        0,
                    )
                )
                + " nonconverged_repair_errors="
                + _join_or_none(nonconverged)
                + " converged_repair_errors="
                + _join_or_none(converged)
            )
            lines.extend(
                _repair_action_lines(
                    source="auto_repair",
                    actions=_auto_repair_actions(repair_convergence),
                )
            )
        if record["kind"] == "adaptive_heartbeat":
            parameters = payload.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            lines.append(
                "- heartbeat path="
                + record["path"]
                + " heartbeat_action_count="
                + str(parameters.get("repair_policy_recommended_action_count", 0))
                + " repair_policy_summary_path="
                + str(parameters.get("repair_policy_summary_path", "unknown"))
            )
            lines.extend(
                _repair_action_lines(
                    source="heartbeat",
                    actions=payload.get("repair_policy_recommended_actions"),
                )
            )
            batches = payload.get("batches")
            if isinstance(batches, list):
                for batch in batches:
                    if not isinstance(batch, dict):
                        continue
                    lines.append(
                        "- heartbeat_batch="
                        + str(batch.get("batch_index", "unknown"))
                        + " candidate_generator="
                        + str(batch.get("candidate_generator", "unknown"))
                        + " repair_policy_action_count="
                        + str(batch.get("repair_policy_recommended_action_count", 0))
                    )
    return lines


def _repair_action_lines(*, source: str, actions: Any) -> list[str]:
    lines: list[str] = []
    if not isinstance(actions, list):
        return lines
    seen: set[tuple[str, str, str]] = set()
    for action in actions:
        if not isinstance(action, dict):
            continue
        key = (
            str(action.get("audit_error", "unknown")),
            str(action.get("action", "unknown")),
            str(action.get("reason", "unknown")),
        )
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            "- repair_action source="
            + source
            + " audit_error="
            + key[0]
            + " action="
            + key[1]
            + " reason="
            + key[2]
        )
    return lines


def _auto_repair_actions(repair_convergence: dict[str, Any]) -> list[dict[str, str]]:
    if repair_convergence.get("decision") != "converged":
        return []
    return [
        {
            "audit_error": error,
            "action": "prefer_previously_converged_repair_path",
            "reason": "repair_converged",
        }
        for error in _string_list(repair_convergence.get("input_repair_errors"))
    ]


def _artifact_kind(payload: dict[str, Any]) -> str:
    state = _state(payload)
    if "session_id" in payload and "tool_call_summary" in payload:
        return "runtime_audit"
    if state == "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED":
        return "strategy_benchmark"
    if state == "COMPETITION_TASK_PIPELINE_COMPLETED":
        return "competition_pipeline"
    if state == "LOCAL_STRATEGY_LOOP_RAN":
        return "local_strategy_loop"
    if state == "LOCAL_STRATEGY_LEADERBOARD_UPDATED":
        return "local_strategy_leaderboard"
    if state == "OPERATOR_TUNING_LOOP_RAN":
        return "operator_tuning"
    if state == "UNET_COMPETITION_PREDICTION_READY":
        return "unet_prediction"
    if state == "SUBMISSION_CANDIDATE_READY":
        return "submission_candidate"
    if state == "RUNTIME_CLOSED_LOOP_RAN":
        return "runtime_closed_loop"
    if state == "RUNTIME_AUDIT_CHAIN_RAN":
        return "runtime_audit_chain"
    if state == "HEARTBEAT_RAN":
        return "adaptive_heartbeat"
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _join_or_none(values: list[str]) -> str:
    return ",".join(values) if values else "none"


def _append_chronicle(*, log_path: Path, chronicle: str) -> None:
    original = log_path.read_text(encoding="utf-8")
    marker = "\nHermes Research Chronicle\n"
    if marker in original:
        original = original.split(marker, maxsplit=1)[0].rstrip() + "\n"
    log_path.write_text(original.rstrip() + "\n\n" + chronicle, encoding="utf-8")
