from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.deeponet_strategy_sweep import run_deeponet_strategy_sweep
from pde_agent.core.operator_strategy_evidence import convert_operator_evidence_summary
from pde_agent.core.real_hdf5_baseline_smoke import run_real_hdf5_baseline_smoke
from pde_agent.core.strategy_decision import recommend_strategy_next_action


STATE_HINT = "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
DEFAULT_PROPOSAL_ID = "burgers_spectral_viscosity_real_hdf5_v0"
DEFAULT_TRAINING_STRATEGIES = [
    "branch_trunk_supervised",
    "branch_trunk_physics_residual",
    "branch_trunk_spectral",
]


class BurgersSpectralViscosityBenchmarkError(RuntimeError):
    pass


def run_burgers_spectral_viscosity_benchmark(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    sample_limit: int,
    min_evaluation_windows: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: list[int],
    hidden_widths: list[int],
    latent_dims: list[int],
    learning_rates: list[float],
    seeds: list[int],
    spectral_weights: list[float],
    physics_weights: list[float],
    proposal_id: str = DEFAULT_PROPOSAL_ID,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    source_file = Path(source_path)
    output_root = Path(output_dir)
    _validate_inputs(
        source_file=source_file,
        output_root=output_root,
        dataset_key=dataset_key,
        sample_limit=sample_limit,
        min_evaluation_windows=min_evaluation_windows,
        proposal_id=proposal_id,
    )
    paths = _paths(output_root, forecast_steps)
    if paths["benchmark_report_path"].exists() and not force:
        raise BurgersSpectralViscosityBenchmarkError(
            f"benchmark report already exists: {paths['benchmark_report_path']}"
        )

    baseline = run_real_hdf5_baseline_smoke(
        source_path=source_file,
        output_dir=paths["baseline_dir"],
        dataset_key=dataset_key,
        start_index=0,
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
    sweep = run_deeponet_strategy_sweep(
        windows_path=paths["baseline_windows_path"],
        output_dir=paths["strategy_sweep_dir"],
        baseline_score_path=paths["baseline_score_path"],
        start_indices=list(range(sample_limit)),
        training_strategies=DEFAULT_TRAINING_STRATEGIES,
        train_steps=train_steps,
        hidden_widths=hidden_widths,
        latent_dims=latent_dims,
        learning_rates=learning_rates,
        seeds=seeds,
        spectral_weights=spectral_weights,
        physics_weights=physics_weights,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    evidence = convert_operator_evidence_summary(
        sweep_summary_path=paths["deeponet_evidence_summary_path"],
        output_path=paths["strategy_evidence_summary_path"],
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=forecast_steps,
        force=force,
    )
    if force and paths["strategy_decision_path"].exists():
        paths["strategy_decision_path"].unlink()
    decision = recommend_strategy_next_action(
        evidence_summary_path=paths["strategy_evidence_summary_path"],
        output_path=paths["strategy_decision_path"],
        min_evaluation_windows=min_evaluation_windows,
    )
    report = _report(
        source_file=source_file,
        output_root=output_root,
        dataset_key=dataset_key,
        sample_limit=sample_limit,
        min_evaluation_windows=min_evaluation_windows,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        train_steps=train_steps,
        hidden_widths=hidden_widths,
        latent_dims=latent_dims,
        learning_rates=learning_rates,
        seeds=seeds,
        spectral_weights=spectral_weights,
        physics_weights=physics_weights,
        proposal_id=proposal_id,
        nu=nu,
        dt=dt,
        dx=dx,
        baseline=baseline,
        sweep=sweep,
        evidence=evidence,
        decision=decision,
        paths=paths,
    )
    paths["benchmark_report_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["benchmark_report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _validate_inputs(
    *,
    source_file: Path,
    output_root: Path,
    dataset_key: str,
    sample_limit: int,
    min_evaluation_windows: int,
    proposal_id: str,
) -> None:
    if not source_file.exists():
        raise BurgersSpectralViscosityBenchmarkError(
            f"source HDF5 file does not exist: {source_file}"
        )
    if not dataset_key:
        raise BurgersSpectralViscosityBenchmarkError("dataset_key must be non-empty")
    if sample_limit < 2:
        raise BurgersSpectralViscosityBenchmarkError("sample_limit must be at least 2")
    if min_evaluation_windows < 2:
        raise BurgersSpectralViscosityBenchmarkError(
            "min_evaluation_windows must be at least 2"
        )
    if sample_limit < min_evaluation_windows:
        raise BurgersSpectralViscosityBenchmarkError(
            "sample_limit must be at least min_evaluation_windows"
        )
    if not proposal_id:
        raise BurgersSpectralViscosityBenchmarkError("proposal_id must be non-empty")
    if output_root.exists() and not output_root.is_dir():
        raise BurgersSpectralViscosityBenchmarkError(
            f"output_dir exists and is not a directory: {output_root}"
        )


def _paths(output_root: Path, forecast_steps: int) -> dict[str, Path]:
    baseline_dir = output_root / "baseline"
    strategy_sweep_dir = output_root / "strategy_sweep"
    return {
        "baseline_dir": baseline_dir,
        "strategy_sweep_dir": strategy_sweep_dir,
        "baseline_summary_path": baseline_dir / "real_hdf5_baseline_smoke_summary.json",
        "baseline_windows_path": baseline_dir / f"windows_f{forecast_steps:03d}.npz",
        "baseline_score_path": baseline_dir / "persistence_score.json",
        "deeponet_strategy_sweep_summary_path": (
            strategy_sweep_dir / "deeponet_strategy_sweep_summary.json"
        ),
        "deeponet_evidence_summary_path": (
            strategy_sweep_dir / "deeponet_evidence" / "evidence_summary.json"
        ),
        "strategy_evidence_summary_path": output_root / "strategy_evidence_summary.json",
        "strategy_decision_path": output_root / "strategy_decision.json",
        "benchmark_report_path": (
            output_root / "burgers_spectral_viscosity_benchmark_report.json"
        ),
    }


def _report(
    *,
    source_file: Path,
    output_root: Path,
    dataset_key: str,
    sample_limit: int,
    min_evaluation_windows: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: list[int],
    hidden_widths: list[int],
    latent_dims: list[int],
    learning_rates: list[float],
    seeds: list[int],
    spectral_weights: list[float],
    physics_weights: list[float],
    proposal_id: str,
    nu: float,
    dt: float,
    dx: float,
    baseline: dict[str, Any],
    sweep: dict[str, Any],
    evidence: dict[str, Any],
    decision: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    best_strategy_id = evidence.get("best_strategy_id")
    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "proposal_id": proposal_id,
        "output_dir": str(output_root),
        "inputs": {
            "source_path": str(source_file),
            "dataset_key": dataset_key,
            "sample_limit": sample_limit,
            "min_evaluation_windows": min_evaluation_windows,
            "context_steps": context_steps,
            "forecast_steps": forecast_steps,
            "time_reduction": time_reduction,
            "space_reduction": space_reduction,
            "training_strategies": DEFAULT_TRAINING_STRATEGIES,
            "train_steps": train_steps,
            "hidden_widths": hidden_widths,
            "latent_dims": latent_dims,
            "learning_rates": learning_rates,
            "seeds": seeds,
            "spectral_weights": spectral_weights,
            "physics_weights": physics_weights,
            "nu": nu,
            "dt": dt,
            "dx": dx,
        },
        "evaluation_window_count": evidence["evaluation_window_count"],
        "metrics": {
            "baseline_score": baseline["metrics"]["competition_like_score_short"],
            "best_strategy_id": best_strategy_id,
            "mean_score_delta": evidence["mean_score_delta"],
            "long_window_score_delta": evidence["long_window_score_delta"],
            "failure_rate": evidence["failure_rate"],
            "score_delta_std": evidence.get("score_delta_std"),
            "physics_residual_delta": evidence.get("physics_residual_delta"),
            "inference_time_regression": evidence.get("inference_time_regression"),
            "best_high_frequency_energy_ratio": _ranking_metric(
                evidence, best_strategy_id, "max_high_frequency_energy_ratio"
            ),
            "best_mean_residual_rmse": _ranking_metric(
                evidence, best_strategy_id, "mean_residual_rmse"
            ),
            "candidate_record_count": len(evidence.get("candidate_records", [])),
            "strategy_ranking_count": len(evidence.get("strategy_ranking", [])),
            "deeponet_record_count": sweep["record_count"],
        },
        "decision": {
            "state_hint": decision["state_hint"],
            "decision": decision["decision"],
            "reason_codes": decision["reason_codes"],
            "next_command": decision["next_command"],
            "recommended_source_type": decision["recommended_source_type"],
        },
        "artifacts": {
            name: str(path)
            for name, path in paths.items()
            if name not in {"baseline_dir", "strategy_sweep_dir"}
        },
        "strategy_ranking": evidence.get("strategy_ranking", []),
        "next_command": decision["next_command"],
    }


def _ranking_metric(
    evidence: dict[str, Any],
    strategy_id: Any,
    metric_key: str,
) -> float | None:
    if not isinstance(strategy_id, str):
        return None
    ranking = evidence.get("strategy_ranking", [])
    if not isinstance(ranking, list):
        return None
    for row in ranking:
        if isinstance(row, dict) and row.get("strategy_id") == strategy_id:
            value = row.get(metric_key)
            if isinstance(value, (int, float)):
                return float(value)
    return None
