from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import (
    SKILL_REVIEW_STATE_HINT,
    review_skill_candidate,
)


def review_skill_candidate_command(
    *,
    skill_candidate_path: str,
    output_path: str,
    reviewer_id: str,
) -> dict[str, object]:
    review = review_skill_candidate(
        skill_candidate_path=Path(skill_candidate_path),
        output_path=Path(output_path),
        reviewer_id=reviewer_id,
    )
    return success_payload(
        run_id="local",
        command="review-skill-candidate",
        state_hint=SKILL_REVIEW_STATE_HINT,
        artifacts={"skill_review_path": review["skill_review_path"]},
        metrics={
            "strategy_id": review["strategy_id"],
            "skill_name": review["skill_name"],
            "decision": review["decision"],
            "approved_for_draft": review["approved_for_draft"],
            "reason_count": len(review["reason_codes"]),
        },
        next_hint=str(review["next_command"]),
    )
