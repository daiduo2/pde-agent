"""Build grouped spectral residual evidence from an audited stratification grid."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


STATE_HINT = "SPECTRAL_RESIDUAL_WINDOW_STRATIFICATION_GRID_COMPLETED"
EVIDENCE_STATE_HINT = "STRATEGY_EVIDENCE_SUMMARY_BUILT"
SOURCE_STATE_HINT = "SPECTRAL_RESIDUAL_WINDOW_STRATIFICATION_COMPLETED"
INTERVENTION = "stratify_spectral_residual_window_evidence"


class SpectralResidualWindowStratificationError(RuntimeError):
    pass


def run_spectral_residual_window_stratification_grid(
    *,
    audit_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    audit_file = Path(audit_path)
    audit = _read_json(audit_file, "experiment config audit")
    _validate_audit(audit)
    config_file = Path(audit["config_path"])
    config = _read_json(config_file, "experiment config")
    _validate_config(config)
    plan_file = Path(audit.get("plan_path", config.get("source_plan_path")))
    plan = _read_json(plan_file, "next experiment plan")
    grid = _read_json(Path(plan["source_grid_summary_path"]), "convergence grid")
    source_evidence = _read_json(
        Path(grid["best_candidate"]["evidence_summary_path"]), "source evidence"
    )
    _validate_source_evidence(source_evidence)

    output_root = Path(output_dir)
    summary_path = output_root / "spectral_residual_window_stratification_summary.json"
    if summary_path.exists() and not force:
        raise SpectralResidualWindowStratificationError(
            f"window stratification summary already exists: {summary_path}"
        )

    candidate_reports = []
    for candidate in config["candidate_experiments"]:
        candidate_reports.append(
            _execute_candidate(
                candidate=candidate,
                source_evidence=source_evidence,
                output_root=output_root,
                force=force,
            )
        )
    ranked = sorted(candidate_reports, key=_ranking_key)
    summary = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "audit_path": str(audit_file),
        "config_path": str(config_file),
        "source_plan_path": str(plan_file),
        "source_grid_summary_path": str(plan["source_grid_summary_path"]),
        "source_evidence_summary_path": str(
            grid["best_candidate"]["evidence_summary_path"]
        ),
        "candidate_count": len(candidate_reports),
        "candidate_reports": candidate_reports,
        "ranking_policy": [
            "failure_rate == 0",
            "score_delta_std <= 2.0",
            "mean_score_delta desc",
        ],
        "best_candidate": ranked[0] if ranked else None,
        "window_stratification_summary_path": str(summary_path),
        "next_command": "recommend-strategy-next-action",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _execute_candidate(
    *,
    candidate: dict[str, Any],
    source_evidence: dict[str, Any],
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    experiment_id = _required_string(candidate, "experiment_id", "candidate")
    parameters = _required_object(candidate, "parameters", "candidate")
    fixed = _required_object(candidate, "fixed_parameters", "candidate")
    grouping = _required_string(parameters, "window_grouping", "candidate parameters")
    aggregation = _required_string(
        parameters, "aggregation_policy", "candidate parameters"
    )
    start_indices = _integer_list(fixed, "start_indices")
    records = _records_by_start(source_evidence, start_indices)
    groups = _window_groups(start_indices=start_indices, grouping=grouping)
    group_records = [
        _group_record(
            group_index=index,
            start_group=group,
            records=[records[start] for start in group],
            strategy_id=source_evidence["best_strategy_id"],
            aggregation_policy=aggregation,
        )
        for index, group in enumerate(groups, start=1)
    ]
    score_deltas = [float(record["score_delta"]) for record in group_records]
    mean_score_delta = _aggregate_score(score_deltas, aggregation)
    evidence_dir = output_root / experiment_id
    evidence_path = evidence_dir / "window_stratified_evidence_summary.json"
    if evidence_path.exists() and not force:
        raise SpectralResidualWindowStratificationError(
            f"window stratified evidence already exists: {evidence_path}"
        )
    evidence = {
        "schema_version": 1,
        "state_hint": EVIDENCE_STATE_HINT,
        "model_family": "SpectralResidual",
        "source_state_hint": SOURCE_STATE_HINT,
        "source_evidence_summary_path": source_evidence["evidence_summary_path"],
        "evidence_summary_path": str(evidence_path),
        "evaluation_window_count": len(group_records),
        "mean_score_delta": mean_score_delta,
        "score_delta_std": float(pstdev(score_deltas)) if len(score_deltas) > 1 else 0.0,
        "long_window_score_delta": mean_score_delta,
        "physics_residual_delta": source_evidence["physics_residual_delta"],
        "inference_time_regression": source_evidence["inference_time_regression"],
        "failure_rate": sum(delta < 0.0 for delta in score_deltas)
        / len(score_deltas),
        "candidate_records": group_records,
        "strategy_ranking": [
            {
                "rank": 1,
                "strategy_id": source_evidence["best_strategy_id"],
                "training_strategy": "spectral_residual_window_stratified",
                "record_count": len(group_records),
                "win_count": sum(delta > 0.0 for delta in score_deltas),
                "win_rate": sum(delta > 0.0 for delta in score_deltas)
                / len(score_deltas),
                "mean_score_delta": mean_score_delta,
                "failure_rate": sum(delta < 0.0 for delta in score_deltas)
                / len(score_deltas),
            }
        ],
        "best_strategy_id": source_evidence["best_strategy_id"],
        "source_metrics": {
            "start_indices": start_indices,
            "window_grouping": grouping,
            "aggregation_policy": aggregation,
            "spectral_modes": fixed["modes"],
            "spectral_ridge": fixed["ridge"],
            "record_count": len(group_records),
        },
        "provenance_notes": [
            "window_stratified_evidence_reuses_existing_per_start_records"
        ],
        "next_command": "recommend-strategy-next-action",
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "experiment_id": experiment_id,
        "parameters": {
            "window_grouping": grouping,
            "aggregation_policy": aggregation,
        },
        "fixed_parameters": {
            "modes": fixed["modes"],
            "ridge": fixed["ridge"],
            "start_indices": start_indices,
        },
        "evidence_summary_path": str(evidence_path),
        "best_strategy_id": evidence["best_strategy_id"],
        "evaluation_window_count": evidence["evaluation_window_count"],
        "mean_score_delta": evidence["mean_score_delta"],
        "score_delta_std": evidence["score_delta_std"],
        "long_window_score_delta": evidence["long_window_score_delta"],
        "failure_rate": evidence["failure_rate"],
        "candidate_record_count": len(group_records),
        "strategy_ranking_count": 1,
    }


def _group_record(
    *,
    group_index: int,
    start_group: list[int],
    records: list[dict[str, Any]],
    strategy_id: str,
    aggregation_policy: str,
) -> dict[str, Any]:
    score_deltas = [float(record["score_delta"]) for record in records]
    scores = [float(record["score"]) for record in records]
    baseline_scores = [float(record["baseline_score"]) for record in records]
    return {
        "model_family": "SpectralResidual",
        "strategy_id": strategy_id,
        "training_strategy": "spectral_residual_window_stratified",
        "experiment_id": f"{strategy_id}_group_{group_index:03d}",
        "start_indices": start_group,
        "score": float(mean(scores)),
        "score_delta": _aggregate_score(score_deltas, aggregation_policy),
        "baseline_score": float(mean(baseline_scores)),
        "dominant_error_patterns": sorted(
            {str(record.get("dominant_error_pattern")) for record in records}
        ),
        "physics_issues": sorted({str(record.get("physics_issue")) for record in records}),
        "source_record_count": len(records),
    }


def _aggregate_score(values: list[float], policy: str) -> float:
    if policy == "median_group_score_delta":
        return float(median(values))
    if policy == "min_group_score_delta":
        return float(min(values))
    raise SpectralResidualWindowStratificationError(
        f"unsupported aggregation_policy: {policy}"
    )


def _window_groups(*, start_indices: list[int], grouping: str) -> list[list[int]]:
    starts = sorted(start_indices)
    if grouping == "contiguous_pairs":
        return [starts[index : index + 2] for index in range(0, len(starts), 2)]
    if grouping == "interleaved_even_odd":
        return [starts[::2], starts[1::2]]
    if grouping == "leave_one_start_out":
        return [
            [start for start in starts if start != held_out] for held_out in starts
        ]
    raise SpectralResidualWindowStratificationError(
        f"unsupported window_grouping: {grouping}"
    )


def _records_by_start(
    source_evidence: dict[str, Any], start_indices: list[int]
) -> dict[int, dict[str, Any]]:
    records = source_evidence.get("candidate_records")
    if not isinstance(records, list) or not records:
        raise SpectralResidualWindowStratificationError(
            "source evidence must include non-empty candidate_records"
        )
    by_start = {}
    for record in records:
        if not isinstance(record, dict):
            raise SpectralResidualWindowStratificationError(
                "candidate_records must contain objects"
            )
        start = record.get("start_index")
        if isinstance(start, bool) or not isinstance(start, int):
            raise SpectralResidualWindowStratificationError(
                "candidate record must include integer start_index"
            )
        by_start[start] = record
    missing = [start for start in start_indices if start not in by_start]
    if missing:
        raise SpectralResidualWindowStratificationError(
            f"source evidence missing start records: {missing}"
        )
    return by_start


def _ranking_key(report: dict[str, Any]) -> tuple[bool, float, bool, float, float]:
    failure_rate = float(report["failure_rate"])
    score_delta_std = float(report["score_delta_std"])
    mean_score_delta = float(report["mean_score_delta"])
    return (
        failure_rate != 0.0,
        failure_rate,
        score_delta_std > 2.0,
        score_delta_std,
        -mean_score_delta,
    )


def _validate_audit(audit: dict[str, Any]) -> None:
    if audit.get("state_hint") != "EXPERIMENT_CONFIG_AUDITED":
        raise SpectralResidualWindowStratificationError(
            "audit must include state_hint EXPERIMENT_CONFIG_AUDITED"
        )
    if audit.get("decision") != "approved_for_execution":
        raise SpectralResidualWindowStratificationError(
            "audit decision must be approved_for_execution"
        )
    if not isinstance(audit.get("config_path"), str):
        raise SpectralResidualWindowStratificationError(
            "audit must include string config_path"
        )


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("state_hint") != "EXPERIMENT_CONFIG_GENERATED":
        raise SpectralResidualWindowStratificationError(
            "config must include state_hint EXPERIMENT_CONFIG_GENERATED"
        )
    if config.get("selected_intervention") != INTERVENTION:
        raise SpectralResidualWindowStratificationError(
            "config selected_intervention must be stratify_spectral_residual_window_evidence"
        )
    candidates = config.get("candidate_experiments")
    if not isinstance(candidates, list) or not candidates:
        raise SpectralResidualWindowStratificationError(
            "config must include non-empty candidate_experiments"
        )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise SpectralResidualWindowStratificationError(
                "candidate_experiments must contain objects"
            )


def _validate_source_evidence(source_evidence: dict[str, Any]) -> None:
    if source_evidence.get("state_hint") != EVIDENCE_STATE_HINT:
        raise SpectralResidualWindowStratificationError(
            "source evidence has invalid state_hint"
        )
    if source_evidence.get("model_family") != "SpectralResidual":
        raise SpectralResidualWindowStratificationError(
            "source evidence must be SpectralResidual"
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SpectralResidualWindowStratificationError(
            f"missing {label} path: {path}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SpectralResidualWindowStratificationError(
            f"{label} must be a JSON object"
        )
    return data


def _required_string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SpectralResidualWindowStratificationError(
            f"{label} must include non-empty string {key}"
        )
    return value


def _required_object(mapping: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise SpectralResidualWindowStratificationError(
            f"{label} must include object {key}"
        )
    return value


def _integer_list(mapping: dict[str, Any], key: str) -> list[int]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise SpectralResidualWindowStratificationError(
            f"{key} must be a non-empty list"
        )
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SpectralResidualWindowStratificationError(
                f"{key} must contain integers"
            )
        result.append(item)
    return result
