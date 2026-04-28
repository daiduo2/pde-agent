"""Execute an audited spectral residual convergence grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.spectral_residual_evidence_sweep import (
    run_spectral_residual_evidence_sweep,
)


STATE_HINT = "SPECTRAL_RESIDUAL_CONVERGENCE_GRID_COMPLETED"
INTERVENTION = "stabilize_spectral_residual_evidence"


class SpectralResidualConvergenceGridError(RuntimeError):
    pass


def run_spectral_residual_convergence_grid(
    *,
    audit_path: str | Path,
    output_dir: str | Path,
    source_path: str | Path,
    dataset_key: str,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    nu: float,
    dt: float,
    dx: float,
    force: bool = False,
) -> dict[str, Any]:
    audit_file = Path(audit_path)
    audit = _read_json(audit_file, "experiment config audit")
    _validate_audit(audit)
    config_file = Path(audit["config_path"])
    config = _read_json(config_file, "experiment config")
    _validate_config(config)

    output_root = Path(output_dir)
    summary_path = output_root / "spectral_residual_convergence_grid_summary.json"
    if summary_path.exists() and not force:
        raise SpectralResidualConvergenceGridError(
            f"convergence grid summary already exists: {summary_path}"
        )

    candidate_reports = []
    for candidate in config["candidate_experiments"]:
        candidate_report = _execute_candidate(
            candidate=candidate,
            output_root=output_root,
            source_path=source_path,
            dataset_key=dataset_key,
            sample_limit=sample_limit,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            time_reduction=time_reduction,
            space_reduction=space_reduction,
            nu=nu,
            dt=dt,
            dx=dx,
            force=force,
        )
        candidate_reports.append(candidate_report)

    ranked = sorted(candidate_reports, key=_ranking_key)
    summary = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "audit_path": str(audit_file),
        "config_path": str(config_file),
        "source_path": str(source_path),
        "candidate_count": len(candidate_reports),
        "candidate_reports": candidate_reports,
        "ranking_policy": [
            "failure_rate == 0",
            "score_delta_std <= 2.0",
            "mean_score_delta desc",
        ],
        "best_candidate": ranked[0] if ranked else None,
        "convergence_grid_summary_path": str(summary_path),
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
    output_root: Path,
    source_path: str | Path,
    dataset_key: str,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, Any]:
    experiment_id = _required_string(candidate, "experiment_id", "candidate")
    parameters = _required_object(candidate, "parameters", "candidate")
    fixed_parameters = _required_object(candidate, "fixed_parameters", "candidate")
    modes = _required_int(parameters, "modes", "candidate parameters")
    ridge = _required_number(parameters, "ridge", "candidate parameters")
    start_indices = _integer_list(fixed_parameters, "start_indices")
    evidence = run_spectral_residual_evidence_sweep(
        source_path=source_path,
        output_dir=output_root / experiment_id,
        dataset_key=dataset_key,
        start_indices=start_indices,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        spectral_modes=modes,
        spectral_ridge=ridge,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    return {
        "experiment_id": experiment_id,
        "parameters": {
            "modes": modes,
            "ridge": ridge,
        },
        "fixed_parameters": {
            "start_indices": start_indices,
        },
        "evidence_summary_path": evidence["evidence_summary_path"],
        "best_strategy_id": evidence["best_strategy_id"],
        "evaluation_window_count": evidence["evaluation_window_count"],
        "mean_score_delta": evidence["mean_score_delta"],
        "score_delta_std": evidence["score_delta_std"],
        "long_window_score_delta": evidence["long_window_score_delta"],
        "failure_rate": evidence["failure_rate"],
        "candidate_record_count": len(evidence["candidate_records"]),
        "strategy_ranking_count": len(evidence["strategy_ranking"]),
    }


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
        raise SpectralResidualConvergenceGridError(
            "audit must include state_hint EXPERIMENT_CONFIG_AUDITED"
        )
    if audit.get("decision") != "approved_for_execution":
        raise SpectralResidualConvergenceGridError(
            "audit decision must be approved_for_execution"
        )
    if not isinstance(audit.get("config_path"), str):
        raise SpectralResidualConvergenceGridError(
            "audit must include string config_path"
        )


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("state_hint") != "EXPERIMENT_CONFIG_GENERATED":
        raise SpectralResidualConvergenceGridError(
            "config must include state_hint EXPERIMENT_CONFIG_GENERATED"
        )
    if config.get("selected_intervention") != INTERVENTION:
        raise SpectralResidualConvergenceGridError(
            "config selected_intervention must be stabilize_spectral_residual_evidence"
        )
    candidates = config.get("candidate_experiments")
    if not isinstance(candidates, list) or not candidates:
        raise SpectralResidualConvergenceGridError(
            "config must include non-empty candidate_experiments"
        )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise SpectralResidualConvergenceGridError(
                "candidate_experiments must contain objects"
            )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SpectralResidualConvergenceGridError(f"missing {label} path: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SpectralResidualConvergenceGridError(f"{label} must be a JSON object")
    return data


def _required_string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SpectralResidualConvergenceGridError(
            f"{label} must include non-empty string {key}"
        )
    return value


def _required_object(mapping: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise SpectralResidualConvergenceGridError(
            f"{label} must include object {key}"
        )
    return value


def _required_int(mapping: dict[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpectralResidualConvergenceGridError(
            f"{label} must include integer {key}"
        )
    return value


def _required_number(mapping: dict[str, Any], key: str, label: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpectralResidualConvergenceGridError(f"{label} must include number {key}")
    return float(value)


def _integer_list(mapping: dict[str, Any], key: str) -> list[int]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise SpectralResidualConvergenceGridError(f"{key} must be a non-empty list")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SpectralResidualConvergenceGridError(f"{key} must contain integers")
        result.append(item)
    return result
