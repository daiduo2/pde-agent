from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_skill_registry import (
    LOOKUP_STATE_HINT,
    lookup_strategy_skills,
)


def lookup_strategy_skill_command(
    *,
    registry_path: str,
    applicable_pde: str,
    model_family: str,
    max_results: int,
    output_path: str,
) -> dict[str, object]:
    result = lookup_strategy_skills(
        registry_path=Path(registry_path),
        applicable_pde=applicable_pde,
        model_family=model_family,
        max_results=max_results,
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="lookup-strategy-skill",
        state_hint=LOOKUP_STATE_HINT,
        artifacts={"lookup_report_path": result["lookup_report_path"]},
        metrics={
            "match_count": result["match_count"],
            "top_skill_name": result["matches"][0]["skill_name"]
            if result["matches"]
            else None,
        },
        next_hint=str(result["next_command"]),
    )
