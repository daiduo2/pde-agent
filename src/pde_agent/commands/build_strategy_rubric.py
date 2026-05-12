
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import RUBRIC_STATE_HINT, build_strategy_rubric


def build_strategy_rubric_command(
    *,
    proposal_path: str,
    output_path: str,
    min_evaluation_windows: int,
    max_inference_time_regression: float,
) -> dict[str, object]:
    rubric = build_strategy_rubric(
        proposal_path=Path(proposal_path),
        output_path=Path(output_path),
        min_evaluation_windows=min_evaluation_windows,
        max_inference_time_regression=max_inference_time_regression,
    )
    return success_payload(
        run_id="local",
        command="build-strategy-rubric",
        state_hint=RUBRIC_STATE_HINT,
        artifacts={
            "experiment_rubric_path": rubric["experiment_rubric_path"],
            "strategy_proposal_path": rubric["strategy_proposal_path"],
        },
        metrics={
            "strategy_id": rubric["strategy_id"],
            "min_evaluation_windows": rubric["promotion_thresholds"][
                "min_evaluation_windows"
            ],
            "required_metric_count": len(rubric["required_metrics"]),
        },
        next_hint=str(rubric["next_command"]),
    )
