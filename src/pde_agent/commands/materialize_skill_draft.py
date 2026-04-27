from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import (
    SKILL_DRAFT_STATE_HINT,
    materialize_skill_draft,
)


def materialize_skill_draft_command(
    *,
    skill_candidate_path: str,
    skill_review_path: str,
    draft_root: str,
) -> dict[str, object]:
    manifest = materialize_skill_draft(
        skill_candidate_path=Path(skill_candidate_path),
        skill_review_path=Path(skill_review_path),
        draft_root=Path(draft_root),
    )
    return success_payload(
        run_id="local",
        command="materialize-skill-draft",
        state_hint=SKILL_DRAFT_STATE_HINT,
        artifacts={
            "skill_md_path": manifest["skill_md_path"],
            "skill_draft_manifest_path": manifest["skill_draft_manifest_path"],
        },
        metrics={
            "strategy_id": manifest["strategy_id"],
            "skill_name": manifest["skill_name"],
            "review_decision": manifest["review_decision"],
        },
        next_hint=str(manifest["next_command"]),
    )
