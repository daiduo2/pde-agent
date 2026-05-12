"""CLI commands for strategy domain."""

import json
import time

import typer

from pde_agent.commands.append_strategy_ledger import append_strategy_ledger_command
from pde_agent.commands.build_research_chronicle import (
    build_research_chronicle_command,
)
from pde_agent.commands.build_strategy_rubric import build_strategy_rubric_command
from pde_agent.commands.create_skill_candidate import create_skill_candidate_command
from pde_agent.commands.formalize_external_strategy_candidate import (
    formalize_external_strategy_candidate_command,
)
from pde_agent.commands.install_skill_draft import install_skill_draft_command
from pde_agent.commands.materialize_skill_draft import materialize_skill_draft_command
from pde_agent.commands.record_autonomy_decision import (
    record_autonomy_decision_command,
)
from pde_agent.commands.record_strategy_proposal import (
    record_strategy_proposal_command,
)
from pde_agent.commands.recommend_strategy_next_action import (
    recommend_strategy_next_action_command,
)
from pde_agent.commands.review_skill_candidate import review_skill_candidate_command
from pde_agent.commands.review_skill_draft import review_skill_draft_command

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("record-autonomy-decision")
    def record_autonomy_decision_cmd(
        decision: str = typer.Option(...),
        reason_code: str = typer.Option(...),
        evidence_source_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_autonomy_decision_command(
                    decision=decision,
                    reason_code=reason_code,
                    evidence_source_path=evidence_source_path,
                    output_path=output_path,
                )
            )
        )

    @app.command("record-strategy-proposal")
    def record_strategy_proposal_cmd(
        strategy_id: str = typer.Option(...),
        title: str = typer.Option(...),
        source_type: str = typer.Option(...),
        problem_context: str = typer.Option(...),
        hypothesis: str = typer.Option(...),
        mechanism: str = typer.Option(...),
        implementation_plan_json: str = typer.Option(...),
        expected_observables: list[str] = typer.Option(..., "--expected-observable"),
        known_risks: list[str] = typer.Option(..., "--known-risk"),
        output_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_strategy_proposal_command(
                    strategy_id=strategy_id,
                    title=title,
                    source_type=source_type,
                    problem_context=problem_context,
                    hypothesis=hypothesis,
                    mechanism=mechanism,
                    implementation_plan=json.loads(implementation_plan_json),
                    expected_observables=expected_observables,
                    known_risks=known_risks,
                    output_path=output_path,
                )
            )
        )

    @app.command("formalize-external-strategy-candidate")
    def formalize_external_strategy_candidate_cmd(
        strategy_decision_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        strategy_id: str = typer.Option(...),
        title: str = typer.Option(...),
        source_title: str = typer.Option(...),
        source_url: str = typer.Option(...),
        scientific_claim: str = typer.Option(...),
        mechanism: str = typer.Option(...),
        applicable_pde: str = typer.Option(...),
        targeted_failure_modes: list[str] = typer.Option(
            ...,
            "--targeted-failure-mode",
        ),
        implementation_plan_json: str = typer.Option(...),
        expected_observables: list[str] = typer.Option(..., "--expected-observable"),
        known_risks: list[str] = typer.Option(..., "--known-risk"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                formalize_external_strategy_candidate_command(
                    strategy_decision_path=strategy_decision_path,
                    output_path=output_path,
                    strategy_id=strategy_id,
                    title=title,
                    source_title=source_title,
                    source_url=source_url,
                    scientific_claim=scientific_claim,
                    mechanism=mechanism,
                    applicable_pde=applicable_pde,
                    targeted_failure_modes=targeted_failure_modes,
                    implementation_plan=json.loads(implementation_plan_json),
                    expected_observables=expected_observables,
                    known_risks=known_risks,
                )
            )
        )

    @app.command("build-strategy-rubric")
    def build_strategy_rubric_cmd(
        proposal_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        min_evaluation_windows: int = typer.Option(3),
        max_inference_time_regression: float = typer.Option(0.10),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_strategy_rubric_command(
                    proposal_path=proposal_path,
                    output_path=output_path,
                    min_evaluation_windows=min_evaluation_windows,
                    max_inference_time_regression=max_inference_time_regression,
                )
            )
        )

    @app.command("recommend-strategy-next-action")
    def recommend_strategy_next_action_cmd(
        evidence_summary_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        min_evaluation_windows: int = typer.Option(3),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                recommend_strategy_next_action_command(
                    evidence_summary_path=evidence_summary_path,
                    output_path=output_path,
                    min_evaluation_windows=min_evaluation_windows,
                )
            )
        )

    @app.command("append-strategy-ledger")
    def append_strategy_ledger_cmd(
        proposal_path: str = typer.Option(...),
        rubric_path: str = typer.Option(...),
        evidence_summary_path: str = typer.Option(...),
        promotion_report_path: str = typer.Option(...),
        ledger_path: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                append_strategy_ledger_command(
                    proposal_path=proposal_path,
                    rubric_path=rubric_path,
                    evidence_summary_path=evidence_summary_path,
                    promotion_report_path=promotion_report_path,
                    ledger_path=ledger_path,
                )
            )
        )

    @app.command("create-skill-candidate")
    def create_skill_candidate_cmd(
        proposal_path: str = typer.Option(...),
        rubric_path: str = typer.Option(...),
        evidence_summary_path: str = typer.Option(...),
        promotion_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        skill_name: str = typer.Option(...),
        skill_dir: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                create_skill_candidate_command(
                    proposal_path=proposal_path,
                    rubric_path=rubric_path,
                    evidence_summary_path=evidence_summary_path,
                    promotion_report_path=promotion_report_path,
                    output_path=output_path,
                    skill_name=skill_name,
                    skill_dir=skill_dir,
                )
            )
        )

    @app.command("review-skill-candidate")
    def review_skill_candidate_cmd(
        skill_candidate_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        reviewer_id: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                review_skill_candidate_command(
                    skill_candidate_path=skill_candidate_path,
                    output_path=output_path,
                    reviewer_id=reviewer_id,
                )
            )
        )

    @app.command("materialize-skill-draft")
    def materialize_skill_draft_cmd(
        skill_candidate_path: str = typer.Option(...),
        skill_review_path: str = typer.Option(...),
        draft_root: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                materialize_skill_draft_command(
                    skill_candidate_path=skill_candidate_path,
                    skill_review_path=skill_review_path,
                    draft_root=draft_root,
                )
            )
        )

    @app.command("review-skill-draft")
    def review_skill_draft_cmd(
        skill_draft_manifest_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        reviewer_id: str = typer.Option(...),
        decision: str = typer.Option(...),
        review_notes: str = typer.Option(...),
        reason_codes: list[str] = typer.Option(..., "--reason-code"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                review_skill_draft_command(
                    skill_draft_manifest_path=skill_draft_manifest_path,
                    output_path=output_path,
                    reviewer_id=reviewer_id,
                    decision=decision,
                    review_notes=review_notes,
                    reason_codes=reason_codes,
                )
            )
        )

    @app.command("install-skill-draft")
    def install_skill_draft_cmd(
        skill_draft_manifest_path: str = typer.Option(...),
        skill_draft_review_path: str = typer.Option(...),
        install_root: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                install_skill_draft_command(
                    skill_draft_manifest_path=skill_draft_manifest_path,
                    skill_draft_review_path=skill_draft_review_path,
                    install_root=install_root,
                )
            )
        )

    @app.command("build-research-chronicle")
    def build_research_chronicle_cmd(
        artifact_paths: list[str] = typer.Option(..., "--artifact-path"),
        output_path: str = typer.Option(...),
        submission_dir: str | None = typer.Option(None),
        task_ids: list[int] | None = typer.Option(None, "--task-id"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_research_chronicle_command(
                    artifact_paths=artifact_paths,
                    output_path=output_path,
                    submission_dir=submission_dir,
                    task_ids=task_ids,
                    force=force,
                )
            )
        )
