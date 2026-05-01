from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from pde_agent.core.hypothesis_lab import run_hypothesis_lab


STATE_HINT = "HYPOTHESIS_LAB_MEMORY_AB_COMPARED"

HypothesisLabRunner = Callable[..., dict[str, Any]]


def compare_hypothesis_lab_memory(
    *,
    manual_report_paths: list[str | Path],
    context_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    submission_id: str,
    force: bool,
    adaptive_report_paths: list[str | Path] | None = None,
    log_paths: list[str | Path] | None = None,
    checkpoint_paths: list[str | Path] | None = None,
    literature_paths: list[str | Path] | None = None,
    algorithm_candidate_paths: list[str | Path] | None = None,
    failure_memory_paths: list[str | Path] | None = None,
    runner: HypothesisLabRunner = run_hypothesis_lab,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_path = root / "memory_ab_report.json"
    _guard_output(report_path, force=force)
    root.mkdir(parents=True, exist_ok=True)

    common_inputs = {
        "manual_report_paths": _paths(manual_report_paths),
        "adaptive_report_paths": _paths(adaptive_report_paths),
        "log_paths": _paths(log_paths),
        "checkpoint_paths": _paths(checkpoint_paths),
        "literature_paths": _paths(literature_paths),
        "algorithm_candidate_paths": _paths(algorithm_candidate_paths),
        "context_path": Path(context_path),
        "target_path": Path(target_path),
        "force": force,
    }

    no_memory_report = runner(
        **common_inputs,
        failure_memory_paths=[],
        failure_memory_output_path=None,
        output_dir=root / "no_memory",
        submission_id=f"{submission_id}_no_memory",
    )
    memory_report = runner(
        **common_inputs,
        failure_memory_paths=_paths(failure_memory_paths),
        failure_memory_output_path=None,
        output_dir=root / "memory",
        submission_id=f"{submission_id}_memory",
    )

    no_memory = _summarize_arm(no_memory_report)
    memory = _summarize_arm(memory_report)
    score_delta_diff = _round_metric(
        memory["score_delta_vs_baseline"] - no_memory["score_delta_vs_baseline"]
    )
    best_segment_score_diff = _round_metric(
        memory["best_segment_score"] - no_memory["best_segment_score"]
    )
    candidate_budget_diff = (
        memory["candidate_budget"] - no_memory["candidate_budget"]
    )
    best_candidate_changed = (
        memory["best_candidate_id"] != no_memory["best_candidate_id"]
    )

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "decision": _decision(
            score_delta_diff=score_delta_diff,
            best_segment_score_diff=best_segment_score_diff,
            candidate_budget_diff=candidate_budget_diff,
            best_candidate_changed=best_candidate_changed,
            no_memory=no_memory,
            memory=memory,
        ),
        "score_delta_diff": score_delta_diff,
        "best_segment_score_diff": best_segment_score_diff,
        "candidate_budget_diff": candidate_budget_diff,
        "best_candidate_changed": best_candidate_changed,
        "arms": {
            "no_memory": no_memory,
            "memory": memory,
        },
        "artifacts": {
            "comparison_report_path": str(report_path),
            "no_memory_report_path": _artifact_path(no_memory_report),
            "memory_report_path": _artifact_path(memory_report),
        },
    }
    report["next_command"] = _next_command(report["decision"])
    _write_json(report_path, report)
    return report


def _paths(paths: list[str | Path] | None) -> list[Path]:
    return [Path(path) for path in (paths or [])]


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass force=True to overwrite")


def _summarize_arm(report: dict[str, Any]) -> dict[str, Any]:
    sandbox = _dict(report.get("sandbox"))
    verifier = _dict(sandbox.get("verifier"))
    best_candidate = _dict(sandbox.get("best_candidate"))
    pareto_gate = _dict(verifier.get("pareto_promotion_gate"))
    axes = _dict(pareto_gate.get("objective_axes"))
    candidate_budget_axis = _dict(axes.get("candidate_budget"))
    return {
        "candidate_family": sandbox.get("candidate_family"),
        "best_candidate_id": best_candidate.get("candidate_id"),
        "score_delta_vs_baseline": _float(
            verifier.get("score_delta_vs_baseline")
        ),
        "best_segment_score": _float(verifier.get("best_segment_score")),
        "decision": verifier.get("decision"),
        "pareto_status": pareto_gate.get("pareto_status"),
        "candidate_budget": _int(candidate_budget_axis.get("value")),
        "anchor_histogram": _anchor_histogram(sandbox),
        "artifacts": _dict(report.get("artifacts")),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _anchor_histogram(sandbox: dict[str, Any]) -> dict[str, int]:
    anchors: Counter[str] = Counter()
    top_candidates = sandbox.get("top_candidates")
    if isinstance(top_candidates, list):
        for candidate in top_candidates:
            anchor = _dict(_dict(candidate).get("params")).get("early_anchor")
            if isinstance(anchor, str) and anchor:
                anchors[anchor] += 1
    if not anchors:
        anchor = _dict(_dict(sandbox.get("best_candidate")).get("params")).get(
            "early_anchor"
        )
        if isinstance(anchor, str) and anchor:
            anchors[anchor] += 1
    return dict(sorted(anchors.items()))


def _decision(
    *,
    score_delta_diff: float,
    best_segment_score_diff: float,
    candidate_budget_diff: int,
    best_candidate_changed: bool,
    no_memory: dict[str, Any],
    memory: dict[str, Any],
) -> str:
    _ = score_delta_diff
    if best_segment_score_diff > 0:
        return "memory_helped"
    if best_segment_score_diff < 0:
        return "memory_regressed"
    if (
        best_candidate_changed
        or candidate_budget_diff != 0
        or memory["anchor_histogram"] != no_memory["anchor_histogram"]
    ):
        return "memory_changed_search"
    return "memory_neutral"


def _next_command(decision: str) -> str:
    if decision == "memory_regressed":
        return "inspect-memory-policy"
    if decision in {"memory_helped", "memory_changed_search"}:
        return "run-hypothesis-lab"
    return "continue-hypothesis-lab"


def _round_metric(value: float) -> float:
    return round(value, 12)


def _artifact_path(report: dict[str, Any]) -> str | None:
    artifacts = _dict(report.get("artifacts"))
    path = artifacts.get("hypothesis_lab_report_path")
    return str(path) if path is not None else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
