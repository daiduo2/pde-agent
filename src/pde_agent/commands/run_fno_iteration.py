
from pathlib import Path

from pde_agent.core.fno_iteration_runner import run_fno_iteration
from pde_agent.core.json_output import success_payload


def run_fno_iteration_command(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_dir: str,
    max_train_steps: int,
    learning_rate: float,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    report = run_fno_iteration(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    winner = report["winner"]
    return success_payload(
        run_id="local",
        command="run-fno-iteration",
        state_hint="FNO_ITERATION_COMPLETED",
        artifacts={
            "iteration_report_path": report["artifacts"]["iteration_report_path"],
            "comparison_report_path": report["artifacts"]["comparison_report_path"],
        },
        metrics={
            "baseline_score": report["baseline"]["competition_like_score_short"],
            "selected_intervention": report["baseline"]["selected_intervention"],
            "candidate_count": report["candidate_count"],
            "winner_experiment_id": winner["experiment_id"],
            "winner_score": winner["score"],
            "winner_score_delta": winner["score_delta"],
        },
        next_hint=report["next_command"],
    )
