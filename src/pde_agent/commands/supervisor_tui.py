
from pathlib import Path
from typing import Any

from pde_agent.core.supervisor_tui import build_supervisor_run_view
from pde_agent.core.supervisor_tui import build_supervisor_runs_view


def supervisor_tui_runs_command(
    *,
    runs_root: str,
    limit: int,
    wide: bool = False,
) -> dict[str, Any]:
    return build_supervisor_runs_view(runs_root=Path(runs_root), limit=limit, wide=wide)


def supervisor_tui_run_command(*, run_dir: str) -> dict[str, Any]:
    return build_supervisor_run_view(run_dir=Path(run_dir))
