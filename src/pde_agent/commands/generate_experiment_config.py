
from pathlib import Path

from pde_agent.core.experiment_config import generate_experiment_config_from_file
from pde_agent.core.json_output import success_payload


def generate_experiment_config(
    *,
    plan_path: str,
    output_path: str,
) -> dict[str, object]:
    config = generate_experiment_config_from_file(
        plan_path=Path(plan_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="generate-experiment-config",
        state_hint="EXPERIMENT_CONFIG_GENERATED",
        artifacts={"experiment_config_path": config["experiment_config_path"]},
        metrics={
            "selected_intervention": config["selected_intervention"],
            "candidate_count": len(config["candidate_experiments"]),
            "requires_training": config["requires_training"],
            "change_scope": config["change_scope"],
        },
        next_hint="execute-candidate-experiment",
    )
