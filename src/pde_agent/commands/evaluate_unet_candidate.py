
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.unet_candidate_evaluation import evaluate_unet_candidate


def evaluate_unet_candidate_command(
    *,
    candidate_report_path: str,
    target_path: str,
    output_dir: str,
    baseline_score_path: str | None,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    report = evaluate_unet_candidate(
        candidate_report_path=Path(candidate_report_path),
        target_path=Path(target_path),
        output_dir=Path(output_dir),
        baseline_score_path=Path(baseline_score_path)
        if baseline_score_path is not None
        else None,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    comparison = report["baseline_comparison"]
    return success_payload(
        run_id="local",
        command="evaluate-unet-candidate",
        state_hint="UNET_CANDIDATE_EVALUATED",
        artifacts={
            "candidate_score_path": report["artifacts"]["score_path"],
            "candidate_rollout_error_path": report["artifacts"]["rollout_error_path"],
            "candidate_burgers_physics_path": report["artifacts"][
                "burgers_physics_path"
            ],
            "candidate_burgers_residual_path": report["artifacts"][
                "burgers_residual_path"
            ],
            "candidate_selected_hypothesis_path": report["artifacts"][
                "selected_hypothesis_path"
            ],
            "evaluation_report_path": report["artifacts"]["evaluation_report_path"],
        },
        metrics={
            "experiment_id": report["candidate"]["experiment_id"],
            "start_index": report["candidate"]["start_index"],
            "train_steps": report["candidate"]["train_steps"],
            "training_strategy": report["candidate"]["training_strategy"],
            "score": report["score"]["competition_like_score_short"],
            "rel_mse": report["score"]["rel_mse"],
            "rmse": report["score"]["rmse"],
            "dominant_error_pattern": report["error_diagnosis"][
                "dominant_error_pattern"
            ],
            "physics_issue": report["physics"]["physics_issue"],
            "residual_rmse": report["physics_residual"]["residual_rmse"],
            "high_frequency_energy_ratio": report["physics_residual"][
                "high_frequency_energy_ratio"
            ],
            "selected_hypothesis": report["selected_hypothesis"][
                "selected_hypothesis"
            ],
            "score_delta": comparison["score_delta"]
            if isinstance(comparison, dict)
            else None,
            "improved": comparison["improved"]
            if isinstance(comparison, dict)
            else None,
        },
        next_hint="build-unet-evidence-sweep",
    )
