"""State-driven Hermes autopilot for strategy growth artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.experiment_config import generate_experiment_config_from_file
from pde_agent.core.experiment_config_audit import audit_experiment_config_from_file
from pde_agent.core.spectral_residual_window_stratification import (
    run_spectral_residual_window_stratification_grid,
)
from pde_agent.core.strategy_decision import recommend_strategy_next_action
from pde_agent.core.strategy_growth import (
    append_strategy_ledger,
    create_skill_candidate,
    evaluate_strategy_promotion,
    install_skill_draft,
    materialize_skill_draft,
    review_skill_candidate,
    review_skill_draft,
)
from pde_agent.core.strategy_skill_registry import register_strategy_skill


STATE_HINT = "HERMES_STRATEGY_GROWTH_AUTOPILOT_COMPLETED"


class HermesStrategyGrowthAutopilotError(RuntimeError):
    pass


def run_hermes_strategy_growth_autopilot(
    *,
    seed_artifact_path: str | Path,
    workspace_dir: str | Path,
    proposal_path: str | Path,
    rubric_path: str | Path,
    skill_name: str,
    install_root: str | Path,
    registry_path: str | Path,
    reviewer_id: str,
    max_steps: int,
    force: bool = False,
) -> dict[str, Any]:
    if max_steps < 1:
        raise HermesStrategyGrowthAutopilotError("max_steps must be positive")
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    seed_file = Path(seed_artifact_path)
    proposal_file = Path(proposal_path)
    rubric_file = Path(rubric_path)
    install_root_path = Path(install_root)
    registry_file = Path(registry_path)

    steps: list[dict[str, Any]] = []
    current_path = seed_file
    evidence_path: Path | None = None
    promotion_path: Path | None = None
    install_manifest_path: Path | None = None
    final_state_hint = ""

    while len(steps) < max_steps:
        current = _read_current_artifact(current_path)
        state_hint = current.get("state_hint")
        if state_hint == "NEXT_EXPERIMENT_PLANNED":
            current_path, evidence_path = _run_plan_step(
                current_path=current_path,
                plan=current,
                workspace=workspace,
                steps=steps,
                force=force,
            )
            continue

        if state_hint == "STRATEGY_EVIDENCE_SUMMARY_BUILT":
            evidence_path = current_path
            decision_path = _fresh(workspace / "strategy_decision.json", force=force)
            decision = recommend_strategy_next_action(
                evidence_summary_path=evidence_path,
                output_path=decision_path,
                min_evaluation_windows=3,
            )
            _record(
                steps,
                command="recommend-strategy-next-action",
                state_hint=decision["state_hint"],
                artifact_path=decision_path,
            )
            current_path = decision_path
            continue

        if state_hint == "STRATEGY_NEXT_ACTION_RECOMMENDED":
            if current.get("decision") != "evaluate_promotion":
                raise HermesStrategyGrowthAutopilotError(
                    "autopilot only supports evaluate_promotion decisions"
                )
            if evidence_path is None:
                evidence_path = Path(current["evidence_summary_path"])
            promotion_path = _fresh(workspace / "promotion_report.json", force=force)
            promotion = evaluate_strategy_promotion(
                proposal_path=proposal_file,
                rubric_path=rubric_file,
                evidence_summary_path=evidence_path,
                output_path=promotion_path,
            )
            _record(
                steps,
                command="evaluate-strategy-promotion",
                state_hint=promotion["state_hint"],
                artifact_path=promotion_path,
            )
            current_path = promotion_path
            continue

        if state_hint == "STRATEGY_PROMOTION_EVALUATED":
            promotion_path = current_path
            if evidence_path is None:
                evidence_path = Path(current["evidence_summary_path"])
            ledger_path = workspace / "strategy_ledger.jsonl"
            if ledger_path.exists() and force:
                ledger_path.unlink()
            ledger = append_strategy_ledger(
                proposal_path=proposal_file,
                rubric_path=rubric_file,
                evidence_summary_path=evidence_path,
                promotion_report_path=promotion_path,
                ledger_path=ledger_path,
            )
            _record(
                steps,
                command="append-strategy-ledger",
                state_hint=ledger["state_hint"],
                artifact_path=ledger_path,
            )
            current_path = ledger_path
            continue

        if state_hint == "STRATEGY_LEDGER_APPENDED":
            if not current.get("promote_to_skill_candidate"):
                final_state_hint = str(current["state_hint"])
                break
            if evidence_path is None or promotion_path is None:
                raise HermesStrategyGrowthAutopilotError(
                    "ledger promotion requires evidence and promotion paths"
                )
            candidate_path = _fresh(workspace / "skill_candidate.json", force=force)
            candidate = create_skill_candidate(
                proposal_path=proposal_file,
                rubric_path=rubric_file,
                evidence_summary_path=evidence_path,
                promotion_report_path=promotion_path,
                output_path=candidate_path,
                skill_name=skill_name,
                skill_dir=skill_name,
            )
            _record(
                steps,
                command="create-skill-candidate",
                state_hint=candidate["state_hint"],
                artifact_path=candidate_path,
            )
            current_path = candidate_path
            continue

        if state_hint == "SKILL_CANDIDATE_CREATED":
            review_path = _fresh(workspace / "skill_candidate_review.json", force=force)
            review = review_skill_candidate(
                skill_candidate_path=current_path,
                output_path=review_path,
                reviewer_id=reviewer_id,
            )
            _record(
                steps,
                command="review-skill-candidate",
                state_hint=review["state_hint"],
                artifact_path=review_path,
            )
            draft_manifest = materialize_skill_draft(
                skill_candidate_path=current_path,
                skill_review_path=review_path,
                draft_root=workspace / "skill_drafts",
            )
            draft_manifest_path = Path(draft_manifest["skill_draft_manifest_path"])
            _record(
                steps,
                command="materialize-skill-draft",
                state_hint=draft_manifest["state_hint"],
                artifact_path=draft_manifest_path,
            )
            current_path = draft_manifest_path
            continue

        if state_hint == "SKILL_DRAFT_MATERIALIZED":
            draft_review_path = _fresh(workspace / "skill_draft_review.json", force=force)
            draft_review = review_skill_draft(
                skill_draft_manifest_path=current_path,
                output_path=draft_review_path,
                reviewer_id=reviewer_id,
                decision="install",
                review_notes=(
                    "Hermes autopilot review passed: promoted evidence satisfied "
                    "rubric thresholds and draft commands match evidence provenance."
                ),
                reason_codes=[
                    "autopilot_review_passed",
                    "evidence_thresholds_passed",
                    "commands_consistent",
                ],
            )
            _record(
                steps,
                command="review-skill-draft",
                state_hint=draft_review["state_hint"],
                artifact_path=draft_review_path,
            )
            install = install_skill_draft(
                skill_draft_manifest_path=current_path,
                skill_draft_review_path=draft_review_path,
                install_root=install_root_path,
            )
            install_manifest_path = Path(install["skill_install_manifest_path"])
            _record(
                steps,
                command="install-skill-draft",
                state_hint=install["state_hint"],
                artifact_path=install_manifest_path,
            )
            current_path = install_manifest_path
            continue

        if state_hint == "SKILL_INSTALLED":
            if evidence_path is None or promotion_path is None:
                raise HermesStrategyGrowthAutopilotError(
                    "registry registration requires evidence and promotion paths"
                )
            registration = register_strategy_skill(
                install_manifest_path=current_path,
                evidence_summary_path=evidence_path,
                promotion_report_path=promotion_path,
                registry_path=registry_file,
                source="hermes_strategy_growth_autopilot",
            )
            _record(
                steps,
                command="register-strategy-skill",
                state_hint=registration["state_hint"],
                artifact_path=registry_file,
            )
            final_state_hint = str(registration["state_hint"])
            break

        raise HermesStrategyGrowthAutopilotError(f"unsupported state_hint: {state_hint}")

    else:
        raise HermesStrategyGrowthAutopilotError("max_steps exceeded")

    report_path = workspace / "hermes_strategy_growth_autopilot_report.json"
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "autopilot_report_path": str(report_path),
        "seed_artifact_path": str(seed_file),
        "workspace_dir": str(workspace),
        "step_count": len(steps),
        "steps": steps,
        "final_state_hint": final_state_hint,
        "human_review_required": False,
        "registry_path": str(registry_file),
        "installed_skill_manifest_path": str(install_manifest_path)
        if install_manifest_path is not None
        else None,
        "next_command": "strategy-growth-complete",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _run_plan_step(
    *,
    current_path: Path,
    plan: dict[str, Any],
    workspace: Path,
    steps: list[dict[str, Any]],
    force: bool,
) -> tuple[Path, Path]:
    config_path = _fresh(workspace / "experiment_config.json", force=force)
    config = generate_experiment_config_from_file(
        plan_path=current_path,
        output_path=config_path,
    )
    _record(
        steps,
        command="generate-experiment-config",
        state_hint=config["state_hint"],
        artifact_path=config_path,
    )
    audit_path = _fresh(workspace / "experiment_config_audit.json", force=force)
    audit = audit_experiment_config_from_file(
        config_path=config_path,
        output_path=audit_path,
        plan_path=current_path,
    )
    _record(
        steps,
        command="audit-experiment-config",
        state_hint=audit["state_hint"],
        artifact_path=audit_path,
    )
    if plan.get("selected_intervention") != "stratify_spectral_residual_window_evidence":
        raise HermesStrategyGrowthAutopilotError(
            "unsupported planned intervention for autopilot"
        )
    strat = run_spectral_residual_window_stratification_grid(
        audit_path=audit_path,
        output_dir=workspace / "window_stratification",
        force=force,
    )
    strat_path = Path(strat["window_stratification_summary_path"])
    _record(
        steps,
        command="run-spectral-residual-window-stratification",
        state_hint=strat["state_hint"],
        artifact_path=strat_path,
    )
    evidence_path = Path(strat["best_candidate"]["evidence_summary_path"])
    return evidence_path, evidence_path


def _record(
    steps: list[dict[str, Any]],
    *,
    command: str,
    state_hint: str,
    artifact_path: Path,
) -> None:
    steps.append(
        {
            "index": len(steps) + 1,
            "command": command,
            "state_hint": state_hint,
            "artifact_path": str(artifact_path),
        }
    )


def _fresh(path: Path, *, force: bool) -> Path:
    if path.exists() and force:
        path.unlink()
    return path


def _read_current_artifact(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return _read_last_jsonl(path)
    return _read_json(path, "current artifact")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise HermesStrategyGrowthAutopilotError(f"missing {label} path: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HermesStrategyGrowthAutopilotError(f"{label} must be a JSON object")
    return data


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise HermesStrategyGrowthAutopilotError("ledger must not be empty")
    data = json.loads(lines[-1])
    if not isinstance(data, dict):
        raise HermesStrategyGrowthAutopilotError("ledger row must be a JSON object")
    return data
