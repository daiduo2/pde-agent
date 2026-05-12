
from pathlib import Path

from pde_agent.core.fno_iteration_dryrun import run_fno_iteration_dryrun
from pde_agent.core.json_output import success_payload


def run_fno_iteration_dryrun_command(
    *,
    windows_path: str,
    checkpoint_path: str,
    output_dir: str,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    manifest = run_fno_iteration_dryrun(
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
        command="run-fno-iteration-dryrun",
        state_hint="FNO_ITERATION_DRYRUN_COMPLETED",
        artifacts={
            "iteration_manifest_path": manifest["artifacts"][
                "iteration_manifest_path"
            ],
            "experiment_config_path": manifest["artifacts"]["experiment_config_path"],
        },
        metrics={
            "adapter_type": manifest["metrics"]["adapter_type"],
            "competition_like_score_short": manifest["metrics"][
                "competition_like_score_short"
            ],
            "selected_hypothesis": manifest["metrics"]["selected_hypothesis"],
            "selected_intervention": manifest["metrics"]["selected_intervention"],
            "candidate_count": manifest["metrics"]["candidate_count"],
            "requires_training": manifest["metrics"]["requires_training"],
        },
        next_hint="execute-candidate-experiment",
    )
