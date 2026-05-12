"""Run bounded DeepONet strategy sweeps on an existing local window file."""


import csv
import json
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

from pde_agent.core.deeponet_candidate_evaluation import evaluate_deeponet_candidate
from pde_agent.core.deeponet_candidate_execution import execute_deeponet_candidate
from pde_agent.core.deeponet_evidence_sweep import build_deeponet_evidence_sweep


class DeepONetStrategySweepError(RuntimeError):
    pass


_CSV_FIELDS = [
    "start_index",
    "train_steps",
    "experiment_id",
    "training_strategy",
    "strategy_label",
    "hidden_width",
    "latent_dim",
    "learning_rate",
    "seed",
    "spectral_weight",
    "physics_weight",
    "baseline_score",
    "score",
    "score_delta",
    "rel_mse",
    "rmse",
    "dominant_error_pattern",
    "physics_issue",
    "initial_loss",
    "final_loss",
]


def run_deeponet_strategy_sweep(
    *,
    windows_path: str | Path,
    output_dir: str | Path,
    baseline_score_path: str | Path,
    start_index: int | None = None,
    start_indices: list[int] | None = None,
    training_strategies: list[str] | None = None,
    train_steps: list[int],
    hidden_widths: list[int],
    latent_dims: list[int],
    learning_rates: list[float],
    seeds: list[int],
    spectral_weights: list[float] | None = None,
    physics_weights: list[float] | None = None,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    windows_file = _require_file(Path(windows_path), "windows")
    baseline_file = _require_file(Path(baseline_score_path), "baseline score")
    output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise DeepONetStrategySweepError(
            f"output_dir already contains files: {output_root}"
        )

    start_values = _start_indices(start_index=start_index, start_indices=start_indices)
    strategy_values = _training_strategies(training_strategies)
    step_values = _unique_positive_ints(train_steps, label="train_steps")
    width_values = _unique_positive_ints(hidden_widths, label="hidden_widths")
    latent_values = _unique_positive_ints(latent_dims, label="latent_dims")
    lr_values = _unique_positive_floats(learning_rates, label="learning_rates")
    seed_values = _unique_non_negative_ints(seeds, label="seeds")
    spectral_weight_values = _loss_weights(
        spectral_weights,
        default=0.1,
        label="spectral_weights",
    )
    physics_weight_values = _loss_weights(
        physics_weights,
        default=0.01,
        label="physics_weights",
    )

    records: list[dict[str, Any]] = []
    evaluation_paths: list[Path] = []
    for start, training_strategy, step_count, width, latent, lr, seed in product(
        start_values,
        strategy_values,
        step_values,
        width_values,
        latent_values,
        lr_values,
        seed_values,
    ):
        for spectral_weight, physics_weight, loss_suffix in _loss_weight_variants(
            training_strategy=training_strategy,
            spectral_weights=spectral_weight_values,
            physics_weights=physics_weight_values,
        ):
            label = _strategy_label(
                training_strategy=training_strategy,
                hidden_width=width,
                latent_dim=latent,
                learning_rate=lr,
                seed=seed,
                include_strategy_prefix=len(strategy_values) > 1,
                loss_weight_suffix=loss_suffix,
            )
            candidate_root = (
                output_root
                / "candidates"
                / label
                / f"start_{start:04d}"
                / f"steps_{step_count:04d}"
            )
            execution = execute_deeponet_candidate(
                windows_path=windows_file,
                output_dir=candidate_root,
                start_index=start,
                train_steps=step_count,
                training_strategy=training_strategy,
                hidden_width=width,
                latent_dim=latent,
                learning_rate=lr,
                seed=seed,
                strategy_label=label,
                nu=nu,
                dt=dt,
                dx=dx,
                spectral_weight=spectral_weight,
                physics_weight=physics_weight,
                force=force,
            )
            evaluation_root = candidate_root / "evaluation"
            evaluation = evaluate_deeponet_candidate(
                candidate_report_path=execution["artifacts"][
                    "candidate_execution_report_path"
                ],
                target_path=windows_file,
                output_dir=evaluation_root,
                baseline_score_path=baseline_file,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
            evaluation_path = evaluation_root / "candidate_evaluation_report.json"
            evaluation_paths.append(evaluation_path)
            records.append(
                _record(
                    execution=execution,
                    evaluation=evaluation,
                    evaluation_report_path=evaluation_path,
                    hidden_width=width,
                    latent_dim=latent,
                    learning_rate=lr,
                    seed=seed,
                )
            )

    evidence = build_deeponet_evidence_sweep(
        evaluation_report_paths=evaluation_paths,
        output_dir=output_root / "deeponet_evidence",
        force=force,
    )
    winner = _winner(records)
    recommendation = _recommendation(records=records, winner=winner)

    summary_path = output_root / "deeponet_strategy_sweep_summary.json"
    csv_path = output_root / "deeponet_strategy_sweep_summary.csv"
    summary = {
        "schema_version": 1,
        "state_hint": "DEEPONET_STRATEGY_SWEEP_COMPLETED",
        "inputs": {
            "windows_path": str(windows_file),
            "baseline_score_path": str(baseline_file),
            "start_index": start_values[0],
            "start_indices": start_values,
            "training_strategies": strategy_values,
            "train_steps": step_values,
            "hidden_widths": width_values,
            "latent_dims": latent_values,
            "learning_rates": lr_values,
            "seeds": seed_values,
            "spectral_weights": spectral_weight_values,
            "physics_weights": physics_weight_values,
            "nu": nu,
            "dt": dt,
            "dx": dx,
        },
        "artifacts": {
            "summary_json_path": str(summary_path),
            "summary_csv_path": str(csv_path),
            "evidence_summary_json_path": evidence["artifacts"][
                "summary_json_path"
            ],
            "evidence_summary_csv_path": evidence["artifacts"]["summary_csv_path"],
        },
        "records": records,
        "record_count": len(records),
        "winner": winner,
        "strategy_statistics": _strategy_statistics(records),
        "evidence_sweep": {
            "state_hint": evidence["state_hint"],
            "record_count": evidence["record_count"],
            "comparison_count": evidence["comparison_count"],
            "comparison_winners": evidence["comparison_winners"],
        },
        "evaluation_window_count": len(
            {record["start_index"] for record in records}
        ),
        "recommendation": recommendation,
        "next_command": recommendation["next_command"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, records)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _record(
    *,
    execution: dict[str, Any],
    evaluation: dict[str, Any],
    evaluation_report_path: Path,
    hidden_width: int,
    latent_dim: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    loss_trace = execution["train"]["loss_trace"]
    candidate = execution["candidate"]
    return {
        "start_index": candidate["start_index"],
        "train_steps": candidate["train_steps"],
        "experiment_id": candidate["experiment_id"],
        "training_strategy": candidate["training_strategy"],
        "strategy_label": candidate["strategy_label"],
        "hidden_width": hidden_width,
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
        "seed": seed,
        "spectral_weight": execution["train"]["loss_weights"]["spectral"],
        "physics_weight": execution["train"]["loss_weights"]["burgers_residual"],
        "baseline_score": evaluation["baseline_comparison"]["baseline_score"],
        "score": evaluation["score"]["competition_like_score_short"],
        "score_delta": evaluation["baseline_comparison"]["score_delta"],
        "rel_mse": evaluation["score"]["rel_mse"],
        "rmse": evaluation["score"]["rmse"],
        "dominant_error_pattern": evaluation["error_diagnosis"][
            "dominant_error_pattern"
        ],
        "physics_issue": evaluation["physics"]["physics_issue"],
        "initial_loss": loss_trace[0] if loss_trace else None,
        "final_loss": loss_trace[-1] if loss_trace else None,
        "candidate_execution_report_path": execution["artifacts"][
            "candidate_execution_report_path"
        ],
        "evaluation_report_path": str(evaluation_report_path),
    }


def _winner(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise DeepONetStrategySweepError("strategy sweep produced no records")
    winner = sorted(
        records,
        key=lambda record: (
            -float(record["score"]),
            float(record["rel_mse"]),
            float(record["rmse"]),
            str(record["experiment_id"]),
        ),
    )[0]
    return {
        "experiment_id": winner["experiment_id"],
        "strategy_label": winner["strategy_label"],
        "score": winner["score"],
        "score_delta": winner["score_delta"],
        "hidden_width": winner["hidden_width"],
        "latent_dim": winner["latent_dim"],
        "learning_rate": winner["learning_rate"],
        "seed": winner["seed"],
    }


def _recommendation(
    *,
    records: list[dict[str, Any]],
    winner: dict[str, Any],
) -> dict[str, Any]:
    all_improved = all(record["score_delta"] > 0 for record in records)
    stable_winner = len({record["strategy_label"] for record in records}) == 1
    add_more = not (stable_winner and all_improved)
    return {
        "add_more_training_strategies": add_more,
        "recommended_strategy_label": None if add_more else winner["strategy_label"],
        "all_candidates_improved_over_baseline": all_improved,
        "rationale": (
            "single strategy improved over baseline"
            if not add_more
            else "multiple strategies were tested or at least one candidate underperformed"
        ),
        "next_command": "run-deeponet-strategy-sweep",
    }


def _strategy_statistics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["strategy_label"]), []).append(record)
    return {
        strategy_label: {
            "count": len(items),
            "mean_score": mean(item["score"] for item in items),
            "mean_score_delta": mean(item["score_delta"] for item in items),
            "best_score": max(item["score"] for item in items),
        }
        for strategy_label, items in sorted(grouped.items())
    }


def _strategy_label(
    *,
    training_strategy: str,
    hidden_width: int,
    latent_dim: int,
    learning_rate: float,
    seed: int,
    include_strategy_prefix: bool,
    loss_weight_suffix: str,
) -> str:
    prefix = f"{_strategy_prefix(training_strategy)}_" if include_strategy_prefix else ""
    weight_part = f"{loss_weight_suffix}_" if loss_weight_suffix else ""
    return (
        f"{prefix}{weight_part}h{hidden_width}_l{latent_dim}_"
        f"lr{_learning_rate_label(learning_rate)}"
        f"_seed{seed}"
    )


def _strategy_prefix(training_strategy: str) -> str:
    return training_strategy.removeprefix("branch_trunk_")


def _loss_weight_variants(
    *,
    training_strategy: str,
    spectral_weights: list[float],
    physics_weights: list[float],
) -> list[tuple[float, float, str]]:
    default_spectral = 0.1
    default_physics = 0.01
    if training_strategy == "branch_trunk_spectral":
        return [
            (
                value,
                default_physics,
                f"sw{_learning_rate_label(value)}" if len(spectral_weights) > 1 else "",
            )
            for value in spectral_weights
        ]
    if training_strategy == "branch_trunk_physics_residual":
        return [
            (
                default_spectral,
                value,
                f"pw{_learning_rate_label(value)}" if len(physics_weights) > 1 else "",
            )
            for value in physics_weights
        ]
    return [(default_spectral, default_physics, "")]


def _learning_rate_label(value: float) -> str:
    text = f"{value:.0e}"
    mantissa, exponent = text.split("e")
    magnitude = exponent.lstrip("+-0") or "0"
    return f"{mantissa}e{magnitude}"


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in _CSV_FIELDS})


def _unique_positive_ints(values: list[int], *, label: str) -> list[int]:
    if not values:
        raise DeepONetStrategySweepError(f"{label} must be non-empty")
    seen: set[int] = set()
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DeepONetStrategySweepError(f"{label} must contain positive integers")
        if value in seen:
            raise DeepONetStrategySweepError(f"{label} contains duplicate value: {value}")
        seen.add(value)
        normalized.append(value)
    return normalized


def _unique_non_negative_ints(values: list[int], *, label: str) -> list[int]:
    if not values:
        raise DeepONetStrategySweepError(f"{label} must be non-empty")
    seen: set[int] = set()
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DeepONetStrategySweepError(
                f"{label} must contain non-negative integers"
            )
        if value in seen:
            raise DeepONetStrategySweepError(f"{label} contains duplicate value: {value}")
        seen.add(value)
        normalized.append(value)
    return normalized


def _start_indices(
    *,
    start_index: int | None,
    start_indices: list[int] | None,
) -> list[int]:
    if start_indices is not None:
        if start_index is not None:
            raise DeepONetStrategySweepError(
                "use either start_index or start_indices, not both"
            )
        return _unique_non_negative_ints(start_indices, label="start_indices")
    if start_index is None:
        raise DeepONetStrategySweepError("start_index or start_indices is required")
    return [_non_negative_int(start_index, "start_index")]


def _training_strategies(values: list[str] | None) -> list[str]:
    raw_values = values if values is not None else ["branch_trunk_supervised"]
    if not raw_values:
        raise DeepONetStrategySweepError("training_strategies must be non-empty")
    allowed = {
        "branch_trunk_supervised",
        "branch_trunk_spectral",
        "branch_trunk_physics_residual",
    }
    seen: set[str] = set()
    normalized = []
    for value in raw_values:
        if value not in allowed:
            raise DeepONetStrategySweepError(
                "training_strategies must contain only: " + ", ".join(sorted(allowed))
            )
        if value in seen:
            raise DeepONetStrategySweepError(
                f"training_strategies contains duplicate value: {value}"
            )
        seen.add(value)
        normalized.append(value)
    return normalized


def _loss_weights(
    values: list[float] | None,
    *,
    default: float,
    label: str,
) -> list[float]:
    return _unique_positive_floats([default] if values is None else values, label=label)


def _unique_positive_floats(values: list[float], *, label: str) -> list[float]:
    if not values:
        raise DeepONetStrategySweepError(f"{label} must be non-empty")
    seen: set[float] = set()
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise DeepONetStrategySweepError(f"{label} must contain positive numbers")
        number = float(value)
        if number in seen:
            raise DeepONetStrategySweepError(f"{label} contains duplicate value: {value}")
        seen.add(number)
        normalized.append(number)
    return normalized


def _non_negative_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeepONetStrategySweepError(f"{label} must be a non-negative integer")
    return value


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise DeepONetStrategySweepError(f"missing {label} path: {path}")
    if not path.is_file():
        raise DeepONetStrategySweepError(f"{label} path is not a file: {path}")
    return path
