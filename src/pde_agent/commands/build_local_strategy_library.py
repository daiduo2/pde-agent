
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_strategy_library import (
    LIBRARY_STATE_HINT,
    LOOKUP_STATE_HINT,
    build_local_strategy_library,
    lookup_local_strategies,
)


def build_local_strategy_library_command(
    *,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    result = build_local_strategy_library(
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-local-strategy-library",
        state_hint=LIBRARY_STATE_HINT,
        artifacts={"local_strategy_library_path": result["library_path"]},
        metrics={"entry_count": result["entry_count"]},
        next_hint=str(result["next_command"]),
    )


def lookup_local_strategy_command(
    *,
    library_path: str,
    applicable_pde: str,
    model_family: str,
    failure_modes: list[str],
    max_results: int,
    output_path: str,
) -> dict[str, object]:
    result = lookup_local_strategies(
        library_path=Path(library_path),
        applicable_pde=applicable_pde,
        model_family=model_family,
        failure_modes=failure_modes,
        max_results=max_results,
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="lookup-local-strategy",
        state_hint=LOOKUP_STATE_HINT,
        artifacts={"lookup_report_path": result["lookup_report_path"]},
        metrics={
            "match_count": result["match_count"],
            "top_strategy_id": result["matches"][0]["strategy_id"]
            if result["matches"]
            else None,
        },
        next_hint=str(result["next_command"]),
    )
