"""Plan the next intervention from spectral residual grid variance."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


STATE_HINT = "NEXT_EXPERIMENT_PLANNED"
SELECTED_INTERVENTION = "stratify_spectral_residual_window_evidence"
PRIMARY_FAILURE_MODE = "start_window_variance_dominates_parameter_sweep"


class SpectralResidualVarianceInterventionError(RuntimeError):
    pass


def plan_spectral_residual_variance_intervention(
    *,
    grid_summary_path: str | Path,
    promotion_report_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    grid_file = Path(grid_summary_path)
    promotion_file = Path(promotion_report_path)
    grid = _read_json(grid_file, "convergence grid summary")
    promotion = _read_json(promotion_file, "promotion report")
    _validate_inputs(grid=grid, promotion=promotion)

    reports = _candidate_reports(grid)
    best = _required_object(grid, "best_candidate", "convergence grid summary")
    parameters = _required_object(best, "parameters", "best candidate")
    fixed_parameters = _required_object(best, "fixed_parameters", "best candidate")
    start_indices = _integer_list(fixed_parameters, "start_indices")
    candidate_parameters = {
        "aggregation_policy": ["median_group_score_delta", "min_group_score_delta"],
        "window_grouping": [
            "contiguous_pairs",
            "interleaved_even_odd",
            "leave_one_start_out",
        ],
    }
    candidate_count = len(candidate_parameters["aggregation_policy"]) * len(
        candidate_parameters["window_grouping"]
    )
    attribution = {
        "primary_failure_mode": PRIMARY_FAILURE_MODE,
        "best_candidate_still_violates_std_gate": float(
            best["score_delta_std"]
        )
        > 2.0,
        "promotion_reason_codes": _string_list(promotion, "reason_codes"),
        "best_candidate": {
            "experiment_id": best["experiment_id"],
            "best_strategy_id": best["best_strategy_id"],
            "mean_score_delta": best["mean_score_delta"],
            "score_delta_std": best["score_delta_std"],
            "failure_rate": best["failure_rate"],
        },
        "parameter_sensitivity": {
            "modes": _parameter_sensitivity(reports, "modes"),
            "ridge": _parameter_sensitivity(reports, "ridge"),
        },
    }
    plan = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "selected_intervention": SELECTED_INTERVENTION,
        "source_grid_summary_path": str(grid_file),
        "source_promotion_report_path": str(promotion_file),
        "source_strategy_id": best["best_strategy_id"],
        "failure_mode": {
            "promotion_decision": promotion["decision"],
            "reason_codes": _string_list(promotion, "reason_codes"),
            "metric_results": promotion["metric_results"],
        },
        "variance_attribution": attribution,
        "change_scope": "evidence_window_stratification",
        "intervention_plan": {
            "description": (
                "Keep the best spectral residual hyperparameters fixed and "
                "stratify evidence by start-window groups before promotion."
            ),
            "allowed_actions": [
                "reuse audited spectral residual convergence grid results",
                "group start windows before aggregating evidence",
                "rank candidates by zero failure rate and low group variance",
                "preserve the best modes/ridge pair from the previous grid",
            ],
            "forbidden_actions": [
                "relax promotion rubric thresholds",
                "promote while score_delta_std remains above threshold",
                "change hidden evaluation assumptions",
                "mix official FNO or UNet failures into spectral residual evidence",
            ],
            "candidate_parameters": candidate_parameters,
            "fixed_parameters": {
                "modes": int(parameters["modes"]),
                "ridge": float(parameters["ridge"]),
                "start_indices": start_indices,
            },
        },
        "budget": {
            "max_experiment_count": candidate_count,
            "max_train_time_seconds": 0,
            "max_inference_time_seconds": 600,
            "requires_training": False,
        },
        "controls": {
            "baseline_required": True,
            "baseline_artifacts": [
                "previous spectral residual convergence grid",
                "promotion report reason codes",
                "best candidate evidence summary",
            ],
        },
        "ranking_policy": [
            "failure_rate == 0",
            "group_score_delta_std <= 2.0",
            "median_group_score_delta desc",
        ],
        "acceptance_criteria": [
            "all retained groups have failure_rate 0.0",
            "group_score_delta_std is <= 2.0",
            "median_group_score_delta remains positive",
        ],
        "rejection_criteria": [
            "any retained group has negative mean_score_delta",
            "all grouping policies keep group_score_delta_std > 2.0",
            "failure_rate becomes nonzero",
        ],
        "required_next_artifacts": [
            "window_stratified_evidence_summary.json",
            "strategy_decision.json",
            "promotion_report.json",
            "strategy_ledger.jsonl",
        ],
        "automation_policy": {
            "may_execute_without_human_review": True,
            "stop_condition": (
                "promotion decision is promote or all grouping policies revise "
                "for score_delta_variance_too_high"
            ),
            "next_decision_command": "recommend-strategy-next-action",
        },
        "next_command": "generate-experiment-config",
    }
    if output_path is not None:
        output_file = Path(output_path)
        plan["next_experiment_plan_path"] = str(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(plan, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return plan


def _validate_inputs(*, grid: dict[str, Any], promotion: dict[str, Any]) -> None:
    if grid.get("state_hint") != "SPECTRAL_RESIDUAL_CONVERGENCE_GRID_COMPLETED":
        raise SpectralResidualVarianceInterventionError(
            "grid summary has invalid state_hint"
        )
    if promotion.get("state_hint") != "STRATEGY_PROMOTION_EVALUATED":
        raise SpectralResidualVarianceInterventionError(
            "promotion report has invalid state_hint"
        )
    if promotion.get("decision") not in {"revise", "reject"}:
        raise SpectralResidualVarianceInterventionError(
            "promotion decision must be revise or reject"
        )
    reason_codes = _string_list(promotion, "reason_codes")
    if "score_delta_variance_too_high" not in reason_codes:
        raise SpectralResidualVarianceInterventionError(
            "promotion must include score_delta_variance_too_high"
        )
    if not isinstance(promotion.get("metric_results"), dict):
        raise SpectralResidualVarianceInterventionError(
            "promotion metric_results must be an object"
        )


def _candidate_reports(grid: dict[str, Any]) -> list[dict[str, Any]]:
    value = grid.get("candidate_reports")
    if not isinstance(value, list) or not value:
        raise SpectralResidualVarianceInterventionError(
            "grid summary must include non-empty candidate_reports"
        )
    reports = []
    for item in value:
        if not isinstance(item, dict):
            raise SpectralResidualVarianceInterventionError(
                "candidate_reports must contain objects"
            )
        reports.append(item)
    return reports


def _parameter_sensitivity(
    reports: list[dict[str, Any]], parameter_name: str
) -> dict[str, Any]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        parameters = _required_object(report, "parameters", "candidate report")
        grouped[parameters[parameter_name]].append(report)
    rows = []
    for value, items in sorted(grouped.items(), key=lambda entry: entry[0]):
        rows.append(
            {
                "value": value,
                "candidate_count": len(items),
                "mean_score_delta": float(
                    mean(float(item["mean_score_delta"]) for item in items)
                ),
                "mean_score_delta_std": float(
                    mean(float(item["score_delta_std"]) for item in items)
                ),
            }
        )
    score_range = _range(row["mean_score_delta"] for row in rows)
    std_range = _range(row["mean_score_delta_std"] for row in rows)
    return {
        "groups": rows,
        "mean_score_delta_range": score_range,
        "score_delta_std_range": std_range,
    }


def _range(values: Any) -> float:
    numbers = [float(value) for value in values]
    return round(max(numbers) - min(numbers), 12) if numbers else 0.0


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SpectralResidualVarianceInterventionError(
            f"missing {label} path: {path}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SpectralResidualVarianceInterventionError(
            f"{label} must be a JSON object"
        )
    return data


def _required_object(mapping: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise SpectralResidualVarianceInterventionError(
            f"{label} must include object {key}"
        )
    return value


def _integer_list(mapping: dict[str, Any], key: str) -> list[int]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise SpectralResidualVarianceInterventionError(
            f"{key} must be a non-empty list"
        )
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SpectralResidualVarianceInterventionError(
                f"{key} must contain integers"
            )
        result.append(item)
    return result


def _string_list(mapping: dict[str, Any], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise SpectralResidualVarianceInterventionError(
            f"{key} must be a non-empty list"
        )
    result = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise SpectralResidualVarianceInterventionError(
                f"{key} must contain strings"
            )
        result.append(item)
    return result
