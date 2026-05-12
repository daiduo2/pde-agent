"""Scientific repetition guard for Hypothesis Lab strategy loops."""


import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_trace import (
    HypothesisLabTraceError,
    read_hypothesis_lab_trace,
)


STATE_HINT = "HYPOTHESIS_DOOM_LOOP_CHECKED"


class HypothesisDoomLoopError(RuntimeError):
    pass


def check_hypothesis_doom_loop(
    *,
    trace_path: str | Path,
    output_path: str | Path | None,
    recent_window: int = 20,
    repeated_family_threshold: int = 3,
    failed_family_threshold: int = 2,
    min_score_delta: float = 0.1,
    force: bool,
) -> dict[str, Any]:
    if recent_window < 1:
        raise HypothesisDoomLoopError("recent_window must be positive")
    if repeated_family_threshold < 2:
        raise HypothesisDoomLoopError("repeated_family_threshold must be >= 2")
    if failed_family_threshold < 1:
        raise HypothesisDoomLoopError("failed_family_threshold must be positive")
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    try:
        events = read_hypothesis_lab_trace(trace_path)
    except HypothesisLabTraceError as exc:
        raise HypothesisDoomLoopError(str(exc)) from exc
    recent = events[-recent_window:]
    observations = [_observation(event) for event in recent]
    observations = [item for item in observations if item.get("candidate_family")]

    triggers: list[dict[str, Any]] = []
    repeated_family = _repeated_non_improving_family(
        observations,
        threshold=repeated_family_threshold,
        min_score_delta=min_score_delta,
    )
    if repeated_family is not None:
        triggers.append(repeated_family)
    failed_families = _failed_family_counts(
        observations,
        failed_family_threshold=failed_family_threshold,
    )
    for family, count in failed_families.items():
        if repeated_family is not None and family == repeated_family["candidate_family"]:
            continue
        triggers.append(
            {
                "trigger": "repeated_failed_family",
                "candidate_family": family,
                "count": count,
            }
        )

    blocked = sorted({str(trigger["candidate_family"]) for trigger in triggers})
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "trace_path": str(trace_path),
        "recent_event_count": len(recent),
        "guard_active": bool(triggers),
        "decision": "inject_corrective_prompt" if triggers else "continue",
        "blocked_candidate_families": blocked,
        "triggers": triggers,
        "corrective_prompt": _corrective_prompt(blocked) if triggers else "",
        "next_command": "build-scientific-tool-catalog",
    }
    _write_optional_json(output_file, report)
    return report


def _observation(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    score_delta = payload.get("score_delta")
    return {
        "event_type": event.get("event_type"),
        "candidate_family": payload.get("candidate_family"),
        "algorithm_candidate_id": payload.get("algorithm_candidate_id"),
        "candidate_key": payload.get("candidate_key"),
        "outcome": payload.get("outcome"),
        "score_delta": float(score_delta)
        if isinstance(score_delta, int | float) and not isinstance(score_delta, bool)
        else None,
    }


def _repeated_non_improving_family(
    observations: list[dict[str, Any]],
    *,
    threshold: int,
    min_score_delta: float,
) -> dict[str, Any] | None:
    run: list[dict[str, Any]] = []
    for obs in observations:
        if run and obs["candidate_family"] != run[-1]["candidate_family"]:
            run = []
        run.append(obs)
        if len(run) >= threshold and _all_non_improving(run[-threshold:], min_score_delta):
            return {
                "trigger": "repeated_non_improving_family",
                "candidate_family": obs["candidate_family"],
                "count": len(run),
                "min_score_delta": min_score_delta,
            }
    return None


def _all_non_improving(
    observations: list[dict[str, Any]],
    min_score_delta: float,
) -> bool:
    for obs in observations:
        score_delta = obs.get("score_delta")
        if isinstance(score_delta, float) and score_delta >= min_score_delta:
            return False
    return True


def _failed_family_counts(
    observations: list[dict[str, Any]],
    *,
    failed_family_threshold: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in observations:
        if obs.get("outcome") not in {"failed", "rejected", "blocked"}:
            continue
        family = obs.get("candidate_family")
        if not isinstance(family, str) or not family:
            continue
        counts[family] = counts.get(family, 0) + 1
    return {
        family: count
        for family, count in counts.items()
        if count >= failed_family_threshold
    }


def _corrective_prompt(blocked_families: list[str]) -> str:
    families = ", ".join(blocked_families)
    return (
        "[SYSTEM: SCIENTIFIC REPETITION GUARD] Recent Hypothesis Lab trace "
        f"shows repeated non-improving or failed work in: {families}. "
        "Stop expanding these candidate families for the next iteration. "
        "Choose a fundamentally different candidate family, explain why it "
        "targets a different error mode, and record the new hypothesis, "
        "blueprint, sandbox result, and promotion decision in the trace."
    )


def _guard_output(path: Path | None, *, force: bool) -> None:
    if path is not None and path.exists() and not force:
        raise HypothesisDoomLoopError(f"output already exists: {path}")


def _write_optional_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
