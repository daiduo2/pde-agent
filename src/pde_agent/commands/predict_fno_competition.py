
from pathlib import Path

from pde_agent.core.fno_competition_prediction import (
    STATE_HINT,
    predict_fno_competition,
)
from pde_agent.core.json_output import success_payload


def predict_fno_competition_command(
    *,
    task_id: int,
    test_path: str,
    checkpoint_path: str,
    output_dir: str,
    dataset_key: str,
    force: bool,
) -> dict[str, object]:
    report = predict_fno_competition(
        task_id=task_id,
        test_path=Path(test_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="predict-fno-competition",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "task_id": report["candidate"]["task_id"],
            "sample_count": report["inputs"]["sample_count"],
            "forecast_steps": report["inputs"]["forecast_steps"],
            "space_points": report["inputs"]["space_points"],
            "inference_time": report["inference"]["elapsed_seconds"],
            "backend": report["inference"]["backend"],
        },
        next_hint=str(report["next_command"]),
    )
