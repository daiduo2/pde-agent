from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.run_layout import create_run_tree


def init_run(run_id: str, runs_root: str) -> dict[str, object]:
    run_root = create_run_tree(Path(runs_root), run_id)
    return success_payload(
        run_id=run_id,
        command="init-run",
        state_hint="TASK_PARSED",
        artifacts={
            "run_root": str(run_root),
            "request_config_path": str(run_root / "config" / "request.json"),
        },
        metrics={},
    )
