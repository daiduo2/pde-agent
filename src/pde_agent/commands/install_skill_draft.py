
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.strategy_growth import (
    SKILL_INSTALL_STATE_HINT,
    install_skill_draft,
)


def install_skill_draft_command(
    *,
    skill_draft_manifest_path: str,
    skill_draft_review_path: str,
    install_root: str,
) -> dict[str, object]:
    install_manifest = install_skill_draft(
        skill_draft_manifest_path=Path(skill_draft_manifest_path),
        skill_draft_review_path=Path(skill_draft_review_path),
        install_root=Path(install_root),
    )
    return success_payload(
        run_id="local",
        command="install-skill-draft",
        state_hint=SKILL_INSTALL_STATE_HINT,
        artifacts={
            "installed_skill_md_path": install_manifest["installed_skill_md_path"],
            "skill_install_manifest_path": install_manifest[
                "skill_install_manifest_path"
            ],
        },
        metrics={
            "strategy_id": install_manifest["strategy_id"],
            "skill_name": install_manifest["skill_name"],
            "review_decision": install_manifest["review_decision"],
        },
        next_hint=str(install_manifest["next_command"]),
    )
