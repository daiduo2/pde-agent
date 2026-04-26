from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.fno_evidence_sweep import run_fno_evidence_sweep
from pde_agent.core.json_output import success_payload


def run_fno_evidence_sweep_command(
    *,
    source_path: str,
    checkpoint_path: str,
    output_dir: str,
    dataset_key: str,
    start_indices: list[int],
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: list[int],
    learning_rate: float,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    summary = run_fno_evidence_sweep(
        source_path=Path(source_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        start_indices=start_indices,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        train_steps=train_steps,
        learning_rate=learning_rate,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    recommendation = summary["recommendation"]
    return success_payload(
        run_id="local",
        command="run-fno-evidence-sweep",
        state_hint="FNO_EVIDENCE_SWEEP_COMPLETED",
        artifacts={
            "summary_json_path": summary["artifacts"]["summary_json_path"],
            "summary_csv_path": summary["artifacts"]["summary_csv_path"],
        },
        metrics={
            "record_count": summary["record_count"],
            "comparison_count": summary["comparison_count"],
            "best_score": _best_score(summary),
            "recommended_training_strategy": recommendation[
                "recommended_training_strategy"
            ],
            "add_more_training_strategies": recommendation[
                "add_more_training_strategies"
            ],
        },
        next_hint=recommendation["next_command"],
    )


def _best_score(summary: dict[str, Any]) -> float | None:
    best: float | None = None
    for stats in summary["strategy_statistics"].values():
        score = stats.get("best_score")
        if isinstance(score, int | float):
            best = float(score) if best is None else max(best, float(score))
    return best
