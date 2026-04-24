from pathlib import Path

from pde_agent.core.diagnostic_synthesis import hypothesis_payload, read_json, write_json
from pde_agent.core.json_output import success_payload


def select_hypothesis(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    interpretation = read_json(root / "diagnostics" / "error_interpretation.json")
    payload = hypothesis_payload(run_id=run_id, interpretation=interpretation)
    hypothesis_path = root / "diagnostics" / "selected_hypothesis.json"
    write_json(hypothesis_path, payload)
    return success_payload(
        run_id=run_id,
        command="select-hypothesis",
        state_hint="HYPOTHESIS_SELECTED",
        artifacts={"selected_hypothesis_path": str(hypothesis_path)},
        metrics={
            "selected_hypothesis": payload["selected_hypothesis"],
            "next_experiment_scope": payload["next_experiment_scope"],
        },
    )
