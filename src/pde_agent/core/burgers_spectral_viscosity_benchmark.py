from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from pde_agent.core.deeponet_strategy_sweep import run_deeponet_strategy_sweep
from pde_agent.core.operator_strategy_evidence import convert_operator_evidence_summary
from pde_agent.core.real_hdf5_baseline_smoke import run_real_hdf5_baseline_smoke
from pde_agent.core.strategy_decision import recommend_strategy_next_action
from pde_agent.core.unet_strategy_sweep import run_unet_strategy_sweep


STATE_HINT = "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
DEFAULT_PROPOSAL_ID = "burgers_spectral_viscosity_real_hdf5_v0"
DEFAULT_TRAINING_STRATEGIES = [
    "branch_trunk_supervised",
    "branch_trunk_physics_residual",
    "branch_trunk_spectral",
]
DEFAULT_UNET_TRAINING_STRATEGIES = [
    "unet_supervised",
    "unet_physics_residual",
    "unet_spectral",
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
    unet_base_channels: list[int] | None = None,
    unet_depths: list[int] | None = None,
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
        output_dir=paths["deeponet_strategy_sweep_dir"],
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
    unet_sweep = run_unet_strategy_sweep(
        windows_path=paths["baseline_windows_path"],
        output_dir=paths["unet_strategy_sweep_dir"],
        baseline_score_path=paths["baseline_score_path"],
        start_indices=list(range(sample_limit)),
        training_strategies=DEFAULT_UNET_TRAINING_STRATEGIES,
        train_steps=train_steps,
        base_channels=unet_base_channels or [8],
        depths=unet_depths or [2],
        learning_rates=learning_rates,
        seeds=seeds,
        spectral_weights=spectral_weights,
        physics_weights=physics_weights,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    deeponet_evidence = convert_operator_evidence_summary(
        sweep_summary_path=paths["deeponet_evidence_summary_path"],
        output_path=paths["deeponet_strategy_evidence_summary_path"],
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=forecast_steps,
        force=force,
    )
    unet_evidence = convert_operator_evidence_summary(
        sweep_summary_path=paths["unet_evidence_summary_path"],
        output_path=paths["unet_strategy_evidence_summary_path"],
        model_family="UNet",
        source_state_hint="UNET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=forecast_steps,
        force=force,
    )
    evidence = _write_combined_evidence(
        family_evidences=[deeponet_evidence, unet_evidence],
        output_path=paths["strategy_evidence_summary_path"],
        forecast_steps=forecast_steps,
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
        unet_base_channels=unet_base_channels or [8],
        unet_depths=unet_depths or [2],
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
        unet_sweep=unet_sweep,
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
    deeponet_strategy_sweep_dir = output_root / "strategy_sweep"
    unet_strategy_sweep_dir = output_root / "unet_strategy_sweep"
    return {
        "baseline_dir": baseline_dir,
        "deeponet_strategy_sweep_dir": deeponet_strategy_sweep_dir,
        "unet_strategy_sweep_dir": unet_strategy_sweep_dir,
        "baseline_summary_path": baseline_dir / "real_hdf5_baseline_smoke_summary.json",
        "baseline_windows_path": baseline_dir / f"windows_f{forecast_steps:03d}.npz",
        "baseline_score_path": baseline_dir / "persistence_score.json",
        "deeponet_strategy_sweep_summary_path": (
            deeponet_strategy_sweep_dir / "deeponet_strategy_sweep_summary.json"
        ),
        "deeponet_evidence_summary_path": (
            deeponet_strategy_sweep_dir / "deeponet_evidence" / "evidence_summary.json"
        ),
        "deeponet_strategy_evidence_summary_path": (
            output_root / "deeponet_strategy_evidence_summary.json"
        ),
        "unet_strategy_sweep_summary_path": (
            unet_strategy_sweep_dir / "unet_strategy_sweep_summary.json"
        ),
        "unet_evidence_summary_path": (
            unet_strategy_sweep_dir / "unet_evidence" / "evidence_summary.json"
        ),
        "unet_strategy_evidence_summary_path": (
            output_root / "unet_strategy_evidence_summary.json"
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
    unet_base_channels: list[int],
    unet_depths: list[int],
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
    unet_sweep: dict[str, Any],
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
            "unet_training_strategies": DEFAULT_UNET_TRAINING_STRATEGIES,
            "train_steps": train_steps,
            "hidden_widths": hidden_widths,
            "latent_dims": latent_dims,
            "unet_base_channels": unet_base_channels,
            "unet_depths": unet_depths,
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
            "best_model_family": _best_model_family(evidence, best_strategy_id),
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
            "unet_record_count": unet_sweep["record_count"],
            "combined_family_count": len(
                {
                    record.get("model_family")
                    for record in evidence.get("candidate_records", [])
                    if isinstance(record, dict)
                }
            ),
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
            if name
            not in {
                "baseline_dir",
                "deeponet_strategy_sweep_dir",
                "unet_strategy_sweep_dir",
            }
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


def _write_combined_evidence(
    *,
    family_evidences: list[dict[str, Any]],
    output_path: Path,
    forecast_steps: int,
    force: bool,
) -> dict[str, Any]:
    if output_path.exists() and not force:
        raise BurgersSpectralViscosityBenchmarkError(
            f"combined evidence already exists: {output_path}"
        )
    records = [
        _combined_candidate_record(record)
        for evidence in family_evidences
        for record in evidence.get("candidate_records", [])
        if isinstance(record, dict)
    ]
    if not records:
        raise BurgersSpectralViscosityBenchmarkError(
            "combined operator evidence produced no candidate records"
        )
    winners = _combined_winners(records)
    deltas = [winner["winner_score_delta"] for winner in winners]
    ranking = _combined_strategy_ranking(records=records, winners=winners)
    evidence = {
        "schema_version": 1,
        "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "model_family": "OperatorEnsemble",
        "source_state_hint": "COMBINED_OPERATOR_EVIDENCE_SWEEP_COMPLETED",
        "source_evidence_summary_path": str(output_path),
        "source_family_evidence_summary_paths": [
            evidence["evidence_summary_path"] for evidence in family_evidences
        ],
        "evidence_summary_path": str(output_path),
        "evaluation_window_count": len({record["start_index"] for record in records}),
        "mean_score_delta": float(mean(deltas)),
        "score_delta_std": float(pstdev(deltas)) if len(deltas) > 1 else 0.0,
        "long_window_score_delta": float(mean(deltas)) if forecast_steps > 0 else 0.0,
        "physics_residual_delta": 0.0,
        "inference_time_regression": 0.0,
        "failure_rate": sum(delta < 0 for delta in deltas) / len(deltas),
        "candidate_records": records,
        "strategy_ranking": ranking,
        "physics_metric_summary": _physics_metric_summary(records),
        "best_strategy_id": ranking[0]["strategy_id"] if ranking else None,
        "provenance_notes": ["combined_deeponet_unet_operator_evidence"],
        "source_metrics": {
            "forecast_steps": forecast_steps,
            "comparison_count": len(winners),
            "record_count": len(records),
            "family_count": len(family_evidences),
        },
        "next_command": "evaluate-strategy-promotion",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _combined_candidate_record(record: dict[str, Any]) -> dict[str, Any]:
    model_family = str(record["model_family"])
    source_strategy_id = str(record["strategy_id"])
    combined = dict(record)
    combined["source_strategy_id"] = source_strategy_id
    combined["strategy_id"] = f"{_model_slug(model_family)}:{source_strategy_id}"
    return combined


def _combined_winners(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["start_index"], record["train_steps"]), []).append(
            record
        )
    winners = []
    for (start_index, train_steps), group in sorted(grouped.items()):
        winner = sorted(
            group,
            key=lambda item: (
                -item["score"],
                item["rel_mse"],
                item["rmse"],
                item["experiment_id"],
            ),
        )[0]
        winners.append(
            {
                "start_index": start_index,
                "train_steps": train_steps,
                "comparison_group_key": (
                    f"start_{start_index:04d}_steps_{train_steps:04d}"
                ),
                "winner_experiment_id": winner["experiment_id"],
                "winner_training_strategy": winner["training_strategy"],
                "winner_strategy_id": winner["strategy_id"],
                "winner_model_family": winner["model_family"],
                "winner_score": winner["score"],
                "winner_score_delta": winner["score_delta"],
            }
        )
    return winners


def _combined_strategy_ranking(
    *,
    records: list[dict[str, Any]],
    winners: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    winner_ids = {winner["winner_experiment_id"] for winner in winners}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["strategy_id"], []).append(record)

    ranking = []
    for strategy_id, items in grouped.items():
        training_strategies = sorted({item["training_strategy"] for item in items})
        model_families = sorted({item["model_family"] for item in items})
        metric_summary = _physics_metric_summary(items)
        row = {
            "strategy_id": strategy_id,
            "source_strategy_id": items[0].get("source_strategy_id"),
            "model_family": model_families[0] if len(model_families) == 1 else "mixed",
            "training_strategy": (
                training_strategies[0]
                if len(training_strategies) == 1
                else "mixed"
            ),
            "record_count": len(items),
            "win_count": sum(item["experiment_id"] in winner_ids for item in items),
            "mean_score": float(mean(item["score"] for item in items)),
            "mean_score_delta": float(mean(item["score_delta"] for item in items)),
            "best_score": max(item["score"] for item in items),
            "failure_rate": sum(item["score_delta"] < 0 for item in items)
            / len(items),
            "dominant_error_patterns": sorted(
                {item["dominant_error_pattern"] for item in items}
            ),
            "physics_issues": sorted({item["physics_issue"] for item in items}),
        }
        row["win_rate"] = row["win_count"] / len(items)
        if metric_summary["record_count"]:
            row["mean_residual_rmse"] = metric_summary["mean_residual_rmse"]
            row["max_high_frequency_energy_ratio"] = metric_summary[
                "max_high_frequency_energy_ratio"
            ]
        ranking.append(row)

    ranking.sort(
        key=lambda item: (
            -item["win_count"],
            -item["mean_score_delta"],
            -item["mean_score"],
            item["strategy_id"],
        )
    )
    return [{"rank": rank, **item} for rank, item in enumerate(ranking, start=1)]


def _physics_metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        record["physics_metrics"]
        for record in records
        if isinstance(record.get("physics_metrics"), dict)
    ]
    if not metrics:
        return {
            "record_count": 0,
            "mean_residual_rmse": None,
            "max_high_frequency_energy_ratio": None,
            "max_mass_drift": None,
            "max_energy_growth_ratio": None,
        }
    residual_rmse = _metric_values(metrics, "residual_rmse")
    high_frequency = _metric_values(metrics, "high_frequency_energy_ratio")
    mass_drift = _metric_values(metrics, "mass_drift")
    energy_growth = _metric_values(metrics, "energy_growth_ratio")
    return {
        "record_count": len(metrics),
        "mean_residual_rmse": float(mean(residual_rmse)) if residual_rmse else None,
        "max_high_frequency_energy_ratio": max(high_frequency)
        if high_frequency
        else None,
        "max_mass_drift": max(mass_drift) if mass_drift else None,
        "max_energy_growth_ratio": max(energy_growth) if energy_growth else None,
    }


def _metric_values(metrics: list[dict[str, float]], key: str) -> list[float]:
    return [item[key] for item in metrics if key in item]


def _best_model_family(evidence: dict[str, Any], strategy_id: Any) -> str | None:
    if not isinstance(strategy_id, str):
        return None
    for row in evidence.get("strategy_ranking", []):
        if isinstance(row, dict) and row.get("strategy_id") == strategy_id:
            value = row.get("model_family")
            return value if isinstance(value, str) else None
    return None


def _model_slug(model_family: str) -> str:
    return "".join(char.lower() for char in model_family if char.isalnum()) or "model"
