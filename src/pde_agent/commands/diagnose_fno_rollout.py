
from pathlib import Path

from pde_agent.core.fno_rollout_diagnosis import diagnose_fno_rollout as diagnose_core
from pde_agent.core.json_output import success_payload


def diagnose_fno_rollout(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_dir: str,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    summary = diagnose_core(
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="diagnose-fno-rollout",
        state_hint="FNO_ROLLOUT_DIAGNOSED",
        artifacts={
            "fno_prediction_npz_path": summary["artifacts"]["prediction_npz_path"],
            "fno_score_json_path": summary["artifacts"]["score_json_path"],
            "fno_rollout_error_json_path": summary["artifacts"][
                "rollout_error_json_path"
            ],
            "fno_burgers_physics_json_path": summary["artifacts"][
                "burgers_physics_json_path"
            ],
            "fno_rollout_summary_json_path": summary["summary_path"],
        },
        metrics={
            "adapter_type": summary["model"]["adapter_type"],
            "competition_like_score_short": summary["score"][
                "competition_like_score_short"
            ],
            "rel_mse": summary["score"]["rel_mse"],
            "rmse": summary["score"]["rmse"],
            "dominant_error_pattern": summary["error_diagnosis"][
                "dominant_error_pattern"
            ],
            "physics_issue": summary["physics"]["physics_issue"],
            "residual_ratio": summary["physics"]["residual_ratio"],
            "total_variation_ratio": summary["physics"]["total_variation_ratio"],
        },
        next_hint="select-pde-hypothesis",
    )
