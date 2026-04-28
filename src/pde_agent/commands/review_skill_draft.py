from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import (
    SKILL_DRAFT_REVIEW_STATE_HINT,
    review_skill_draft,
)


def review_skill_draft_command(
    *,
    skill_draft_manifest_path: str,
    output_path: str,
    reviewer_id: str,
    decision: str,
    review_notes: str,
    reason_codes: list[str],
) -> dict[str, object]:
    review = review_skill_draft(
        skill_draft_manifest_path=Path(skill_draft_manifest_path),
        output_path=Path(output_path),
        reviewer_id=reviewer_id,
        decision=decision,
        review_notes=review_notes,
        reason_codes=reason_codes,
    )
    return success_payload(
        run_id="local",
        command="review-skill-draft",
        state_hint=SKILL_DRAFT_REVIEW_STATE_HINT,
        artifacts={"skill_draft_review_path": review["skill_draft_review_path"]},
        metrics={
            "strategy_id": review["strategy_id"],
            "skill_name": review["skill_name"],
            "decision": review["decision"],
            "approved_for_install": review["approved_for_install"],
            "reason_count": len(review["reason_codes"]),
        },
        next_hint=str(review["next_command"]),
    )
