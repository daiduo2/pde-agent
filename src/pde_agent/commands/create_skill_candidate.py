
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import (
    SKILL_CANDIDATE_STATE_HINT,
    create_skill_candidate,
)


def create_skill_candidate_command(
    *,
    proposal_path: str,
    rubric_path: str,
    evidence_summary_path: str,
    promotion_report_path: str,
    output_path: str,
    skill_name: str,
    skill_dir: str,
) -> dict[str, object]:
    candidate = create_skill_candidate(
        proposal_path=Path(proposal_path),
        rubric_path=Path(rubric_path),
        evidence_summary_path=Path(evidence_summary_path),
        promotion_report_path=Path(promotion_report_path),
        output_path=Path(output_path),
        skill_name=skill_name,
        skill_dir=skill_dir,
    )
    return success_payload(
        run_id="local",
        command="create-skill-candidate",
        state_hint=SKILL_CANDIDATE_STATE_HINT,
        artifacts={"skill_candidate_path": candidate["skill_candidate_path"]},
        metrics={
            "strategy_id": candidate["strategy_id"],
            "skill_name": candidate["skill_name"],
            "procedure_step_count": len(candidate["procedure_steps"]),
            "failure_boundary_count": len(candidate["failure_boundaries"]),
        },
        next_hint=str(candidate["next_command"]),
    )
