"""Plan the next spectral residual sweep from promotion feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "NEXT_EXPERIMENT_PLANNED"


class SpectralResidualNextExperimentError(RuntimeError):
    pass


def plan_next_spectral_residual_experiment(
    *,
    evidence_summary_path: str | Path,
    promotion_report_path: str | Path,
    ledger_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    evidence_file = Path(evidence_summary_path)
    promotion_file = Path(promotion_report_path)
    ledger_file = Path(ledger_path)
    evidence = _read_json(evidence_file, "evidence summary")
    promotion = _read_json(promotion_file, "promotion report")
    ledger_row = _read_last_jsonl(ledger_file)
    _validate_inputs(evidence=evidence, promotion=promotion, ledger_row=ledger_row)

    source_metrics = evidence.get("source_metrics")
    if not isinstance(source_metrics, dict):
        raise SpectralResidualNextExperimentError(
            "evidence source_metrics must be an object"
        )
    previous_starts = _integer_list(source_metrics, "start_indices")
    expanded_starts = _expand_start_indices(previous_starts)
    modes = _next_modes(source_metrics.get("spectral_modes"))
    ridges = _next_ridges(source_metrics.get("spectral_ridge"))
    candidate_parameters = {
        "modes": modes,
        "ridge": ridges,
    }
    candidate_count = len(modes) * len(ridges)
    reason_codes = _string_list(promotion, "reason_codes")
    metric_results = promotion.get("metric_results")
    if not isinstance(metric_results, dict):
        raise SpectralResidualNextExperimentError(
            "promotion metric_results must be an object"
        )

    plan = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "selected_intervention": "stabilize_spectral_residual_evidence",
        "source_strategy_id": evidence["best_strategy_id"],
        "source_evidence_summary_path": str(evidence_file),
        "source_promotion_report_path": str(promotion_file),
        "source_ledger_path": str(ledger_file),
        "failure_mode": {
            "promotion_decision": promotion["decision"],
            "reason_codes": reason_codes,
            "metric_results": {
                "evaluation_window_count": metric_results.get(
                    "evaluation_window_count"
                ),
                "mean_score_delta": metric_results.get("mean_score_delta"),
                "score_delta_std": metric_results.get("score_delta_std"),
                "failure_rate": metric_results.get("failure_rate"),
            },
        },
        "change_scope": "evidence_sweep_parameter_grid",
        "intervention_plan": {
            "description": (
                "Reduce spectral residual evidence variance by expanding starts "
                "and sweeping low-mode/ridge settings before promotion."
            ),
            "allowed_actions": [
                "run spectral residual evidence sweeps",
                "expand real-HDF5 start indices",
                "sweep modes and ridge values",
                "rank by stability before mean score",
            ],
            "forbidden_actions": [
                "relax promotion rubric thresholds",
                "create skill candidate before promotion passes",
                "change hidden evaluation assumptions",
                "mix official FNO or UNet failures into spectral residual evidence",
            ],
            "candidate_parameters": candidate_parameters,
            "fixed_parameters": {
                "start_indices": expanded_starts,
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
                "persistence score per start",
                "previous spectral residual evidence summary",
                "promotion report reason codes",
            ],
        },
        "ranking_policy": [
            "failure_rate == 0",
            "score_delta_std <= 2.0",
            "mean_score_delta desc",
        ],
        "acceptance_criteria": [
            "failure_rate remains 0.0",
            "score_delta_std is <= 2.0",
            "mean_score_delta remains positive",
        ],
        "rejection_criteria": [
            "any candidate has negative mean_score_delta",
            "all candidates keep score_delta_std > 2.0",
            "failure_rate becomes nonzero",
        ],
        "required_next_artifacts": [
            "spectral_residual_evidence_summary.json",
            "strategy_decision.json",
            "promotion_report.json",
            "strategy_ledger.jsonl",
        ],
        "recommended_command": {
            "command": _recommended_command(
                evidence=evidence,
                start_indices=expanded_starts,
                modes=modes[0],
                ridge=ridges[0],
            ),
            "note": (
                "Execute once per candidate grid entry, replacing --spectral-modes "
                "and --spectral-ridge with each generated candidate."
            ),
        },
        "automation_policy": {
            "may_execute_without_human_review": True,
            "stop_condition": "promotion decision is promote or all candidate grids revise for the same reason",
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


def _validate_inputs(
    *,
    evidence: dict[str, Any],
    promotion: dict[str, Any],
    ledger_row: dict[str, Any],
) -> None:
    if evidence.get("state_hint") != "STRATEGY_EVIDENCE_SUMMARY_BUILT":
        raise SpectralResidualNextExperimentError(
            "evidence summary has invalid state_hint"
        )
    if evidence.get("model_family") != "SpectralResidual":
        raise SpectralResidualNextExperimentError(
            "evidence summary must be for SpectralResidual"
        )
    if promotion.get("state_hint") != "STRATEGY_PROMOTION_EVALUATED":
        raise SpectralResidualNextExperimentError(
            "promotion report has invalid state_hint"
        )
    if promotion.get("decision") not in {"revise", "reject"}:
        raise SpectralResidualNextExperimentError(
            "promotion decision must be revise or reject"
        )
    if ledger_row.get("state_hint") != "STRATEGY_LEDGER_APPENDED":
        raise SpectralResidualNextExperimentError("ledger row has invalid state_hint")
    if ledger_row.get("strategy_id") != promotion.get("strategy_id"):
        raise SpectralResidualNextExperimentError(
            "ledger and promotion strategy_id mismatch"
        )
    if evidence.get("best_strategy_id") != promotion.get("strategy_id"):
        raise SpectralResidualNextExperimentError(
            "evidence best_strategy_id must match promotion strategy_id"
        )


def _expand_start_indices(previous: list[int]) -> list[int]:
    if not previous:
        return [0, 4, 8, 12]
    sorted_starts = sorted(set(previous))
    if len(sorted_starts) == 1:
        step = 4
    else:
        step = max(1, min(b - a for a, b in zip(sorted_starts, sorted_starts[1:])))
        step = max(1, step // 2)
    start = sorted_starts[0]
    stop = sorted_starts[-1] + step
    expanded = list(range(start, stop + 1, step))
    return expanded[: max(8, len(sorted_starts))]


def _next_modes(value: Any) -> list[int]:
    current = int(value) if isinstance(value, int) and not isinstance(value, bool) else 16
    candidates = sorted({max(1, current // 2), max(1, current - 4), current})
    return candidates


def _next_ridges(value: Any) -> list[float]:
    current = float(value) if isinstance(value, (int, float)) else 1e-3
    return [current, current * 10.0, current * 100.0]


def _recommended_command(
    *,
    evidence: dict[str, Any],
    start_indices: list[int],
    modes: int,
    ridge: float,
) -> str:
    source_metrics = evidence.get("source_metrics")
    if not isinstance(source_metrics, dict):
        source_metrics = {}
    start_args = " ".join(f"--start-index {index}" for index in start_indices)
    return (
        "uv run --frozen --extra data pde-agent run-spectral-residual-evidence-sweep "
        "--source-path data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5 "
        "--output-dir runs/spectral_residual_evidence_sweep_next "
        "--dataset-key tensor "
        f"{start_args} "
        f"--sample-limit {int(source_metrics.get('sample_limit', 8))} "
        "--context-steps 10 "
        f"--forecast-steps {int(source_metrics.get('forecast_steps', 31))} "
        "--time-reduction 5 "
        "--space-reduction 4 "
        f"--spectral-modes {modes} "
        f"--spectral-ridge {ridge:g} "
        "--nu 0.001 --dt 1.0 --dx 1.0 --force"
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SpectralResidualNextExperimentError(f"missing {label} path: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SpectralResidualNextExperimentError(f"{label} must be a JSON object")
    return data


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SpectralResidualNextExperimentError(f"missing ledger path: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise SpectralResidualNextExperimentError("ledger must contain at least one row")
    data = json.loads(lines[-1])
    if not isinstance(data, dict):
        raise SpectralResidualNextExperimentError("ledger row must be a JSON object")
    return data


def _integer_list(mapping: dict[str, Any], key: str) -> list[int]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise SpectralResidualNextExperimentError(f"{key} must be a non-empty list")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SpectralResidualNextExperimentError(f"{key} must contain integers")
        result.append(item)
    return result


def _string_list(mapping: dict[str, Any], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise SpectralResidualNextExperimentError(f"{key} must be a non-empty list")
    result = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise SpectralResidualNextExperimentError(f"{key} must contain strings")
        result.append(item)
    return result
