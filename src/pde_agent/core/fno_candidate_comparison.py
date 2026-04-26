"""Compare evaluated FNO candidates and select the next iteration winner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FNOCandidateComparisonError(RuntimeError):
    pass


def compare_fno_candidates(
    *,
    evaluation_report_paths: list[str | Path],
    output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    if not evaluation_report_paths:
        raise FNOCandidateComparisonError(
            "at least one candidate evaluation report is required"
        )
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise FNOCandidateComparisonError(
            f"comparison output already exists: {output_file}"
        )

    summaries = [
        _load_candidate_summary(Path(path)) for path in evaluation_report_paths
    ]
    _reject_duplicate_experiment_ids(summaries)
    ranked = sorted(
        summaries,
        key=lambda candidate: (
            -candidate["score"],
            candidate["rel_mse"],
            candidate["rmse"],
            candidate["experiment_id"],
        ),
    )
    winner = ranked[0]
    candidates = [
        _ranked_candidate_summary(
            candidate=candidate,
            winner=winner,
            rank=index + 1,
        )
        for index, candidate in enumerate(ranked)
    ]
    report = {
        "schema_version": 1,
        "state_hint": "FNO_CANDIDATES_COMPARED",
        "evaluation_report_paths": [str(Path(path)) for path in evaluation_report_paths],
        "candidate_count": len(candidates),
        "winner": candidates[0],
        "rejected_candidates": candidates[1:],
        "candidates": candidates,
        "next_command": "prepare-next-fno-iteration",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_candidate_summary(path: Path) -> dict[str, Any]:
    report_file = _require_file(path, "candidate evaluation report")
    report = _load_json_object(report_file, "candidate evaluation report")
    if report.get("state_hint") != "FNO_CANDIDATE_EVALUATED":
        raise FNOCandidateComparisonError(
            "candidate evaluation report must include state_hint "
            "FNO_CANDIDATE_EVALUATED"
        )
    candidate = _require_object(report, "candidate")
    score = _require_object(report, "score")
    error = _require_object(report, "error_diagnosis")
    physics = _require_object(report, "physics")
    hypothesis = _require_object(report, "selected_hypothesis")
    experiment_id = _require_string(candidate, "experiment_id", "candidate")
    return {
        "experiment_id": experiment_id,
        "intervention": _require_string(candidate, "intervention", "candidate"),
        "score": _require_number(
            score, "competition_like_score_short", "score"
        ),
        "rel_mse": _require_number(score, "rel_mse", "score"),
        "rmse": _require_number(score, "rmse", "score"),
        "baseline_score": _optional_baseline_value(
            report.get("baseline_comparison"), "baseline_score"
        ),
        "score_delta": _optional_baseline_value(
            report.get("baseline_comparison"), "score_delta"
        ),
        "improved": _optional_improved(report.get("baseline_comparison")),
        "dominant_error_pattern": _require_string(
            error, "dominant_error_pattern", "error_diagnosis"
        ),
        "physics_issue": _require_string(physics, "physics_issue", "physics"),
        "selected_hypothesis": _require_string(
            hypothesis, "selected_hypothesis", "selected_hypothesis"
        ),
        "evaluation_report_path": str(report_file),
    }


def _ranked_candidate_summary(
    *,
    candidate: dict[str, Any],
    winner: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "experiment_id": candidate["experiment_id"],
        "intervention": candidate["intervention"],
        "score": candidate["score"],
        "rel_mse": candidate["rel_mse"],
        "rmse": candidate["rmse"],
        "baseline_score": candidate["baseline_score"],
        "score_delta": candidate["score_delta"],
        "improved": candidate["improved"],
        "dominant_error_pattern": candidate["dominant_error_pattern"],
        "physics_issue": candidate["physics_issue"],
        "selected_hypothesis": candidate["selected_hypothesis"],
        "evaluation_report_path": candidate["evaluation_report_path"],
        "rejection_reason": None
        if rank == 1
        else _rejection_reason(candidate=candidate, winner=winner),
    }


def _rejection_reason(
    *,
    candidate: dict[str, Any],
    winner: dict[str, Any],
) -> str:
    if candidate["score"] < winner["score"]:
        return "lower_score"
    if candidate["rel_mse"] > winner["rel_mse"]:
        return "tie_broken_by_rel_mse"
    if candidate["rmse"] > winner["rmse"]:
        return "tie_broken_by_rmse"
    return "tie_broken_by_experiment_id"


def _reject_duplicate_experiment_ids(candidates: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        experiment_id = candidate["experiment_id"]
        if experiment_id in seen:
            raise FNOCandidateComparisonError(
                f"duplicate experiment_id in candidate reports: {experiment_id}"
            )
        seen.add(experiment_id)


def _optional_baseline_value(value: object, key: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise FNOCandidateComparisonError("baseline_comparison must be null or object")
    metric = value.get(key)
    if metric is None:
        return None
    if not isinstance(metric, int | float):
        raise FNOCandidateComparisonError(f"baseline_comparison.{key} must be numeric")
    return float(metric)


def _optional_improved(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise FNOCandidateComparisonError("baseline_comparison must be null or object")
    improved = value.get("improved")
    if improved is None:
        return None
    if not isinstance(improved, bool):
        raise FNOCandidateComparisonError("baseline_comparison.improved must be bool")
    return improved


def _require_object(report: dict[str, Any], key: str) -> dict[str, Any]:
    value = report.get(key)
    if not isinstance(value, dict):
        raise FNOCandidateComparisonError(f"candidate report must include {key}")
    return value


def _require_string(report: dict[str, Any], key: str, label: str) -> str:
    value = report.get(key)
    if not isinstance(value, str):
        raise FNOCandidateComparisonError(f"{label}.{key} must be a string")
    return value


def _require_number(report: dict[str, Any], key: str, label: str) -> float:
    value = report.get(key)
    if not isinstance(value, int | float):
        raise FNOCandidateComparisonError(f"{label}.{key} must be numeric")
    return float(value)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FNOCandidateComparisonError(
            f"failed to parse {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise FNOCandidateComparisonError(f"{label} JSON must be an object")
    return data


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FNOCandidateComparisonError(f"missing {label} path: {path}")
    if not path.is_file():
        raise FNOCandidateComparisonError(f"{label} path is not a file: {path}")
    return path
