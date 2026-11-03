"""Agent infrastructure CLI commands for hypothesis-lab."""

from __future__ import annotations

import json

import typer

from pde_agent.commands.hypothesis_lab_autonomous_loop import (
    agent_hypothesis_run_autonomous_loop_command,
)
from pde_agent.commands.hypothesis_lab_memory import (
    agent_hypothesis_get_family_history_command,
    agent_hypothesis_get_global_pareto_front_command,
    agent_hypothesis_get_lessons_learned_command,
)
from pde_agent.commands.hypothesis_lab_primitives import (
    agent_hypothesis_build_cards_command,
    agent_hypothesis_create_session_command,
    agent_hypothesis_design_blueprints_command,
    agent_hypothesis_generate_hypotheses_command,
    agent_hypothesis_get_best_evidence_command,
    agent_hypothesis_get_failure_pattern_command,
    agent_hypothesis_get_pareto_front_command,
    agent_hypothesis_list_candidates_command,
    agent_hypothesis_list_pending_blueprints_command,
    agent_hypothesis_load_session_command,
    agent_hypothesis_query_state_command,
    agent_hypothesis_record_outcome_command,
    agent_hypothesis_run_pipeline_command,
    agent_hypothesis_run_sandbox_command,
    agent_hypothesis_summarize_events_command,
)
from pde_agent.commands.hypothesis_lab_tool_catalog import (
    build_hypothesis_lab_tool_catalog_command,
)


def register_agent_commands(app: typer.Typer) -> None:
    """Register agent infrastructure commands."""

    @app.command("agent-hypothesis-create-session")
    def agent_hypothesis_create_session_cmd(
        workspace: str = typer.Option(...),
        session_id: str | None = typer.Option(None),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_create_session_command(
                    workspace=workspace,
                    session_id=session_id,
                )
            )
        )

    @app.command("agent-hypothesis-load-session")
    def agent_hypothesis_load_session_cmd(
        workspace: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_load_session_command(
                    workspace=workspace,
                )
            )
        )

    @app.command("agent-hypothesis-build-cards")
    def agent_hypothesis_build_cards_cmd(
        workspace: str = typer.Option(...),
        manual_report_paths: list[str] = typer.Option([], "--manual-report-path"),
        adaptive_report_paths: list[str] = typer.Option([], "--adaptive-report-path"),
        log_paths: list[str] = typer.Option([], "--log-path"),
        checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
        literature_paths: list[str] = typer.Option([], "--literature-path"),
        algorithm_candidate_paths: list[str] = typer.Option(
            [], "--algorithm-candidates-path"
        ),
        failure_memory_paths: list[str] = typer.Option([], "--failure-memory-path"),
        artifact_id: str = typer.Option("knowledge_cards"),
        force: bool = typer.Option(True, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_build_cards_command(
                    workspace=workspace,
                    manual_report_paths=manual_report_paths,
                    adaptive_report_paths=adaptive_report_paths or None,
                    log_paths=log_paths or None,
                    checkpoint_paths=checkpoint_paths or None,
                    literature_paths=literature_paths or None,
                    algorithm_candidate_paths=algorithm_candidate_paths or None,
                    failure_memory_paths=failure_memory_paths or None,
                    artifact_id=artifact_id,
                    force=force,
                )
            )
        )

    @app.command("agent-hypothesis-generate-hypotheses")
    def agent_hypothesis_generate_hypotheses_cmd(
        workspace: str = typer.Option(...),
        cards_artifact_id: str = typer.Option("knowledge_cards"),
        artifact_id: str = typer.Option("hypotheses"),
        force: bool = typer.Option(True, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_generate_hypotheses_command(
                    workspace=workspace,
                    cards_artifact_id=cards_artifact_id,
                    artifact_id=artifact_id,
                    force=force,
                )
            )
        )

    @app.command("agent-hypothesis-design-blueprints")
    def agent_hypothesis_design_blueprints_cmd(
        workspace: str = typer.Option(...),
        hypotheses_artifact_id: str = typer.Option("hypotheses"),
        artifact_id: str = typer.Option("blueprints"),
        force: bool = typer.Option(True, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_design_blueprints_command(
                    workspace=workspace,
                    hypotheses_artifact_id=hypotheses_artifact_id,
                    artifact_id=artifact_id,
                    force=force,
                )
            )
        )

    @app.command("agent-hypothesis-run-sandbox")
    def agent_hypothesis_run_sandbox_cmd(
        workspace: str = typer.Option(...),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        reference_path: str | None = typer.Option(None),
        reference_sample_limit: int | None = typer.Option(None),
        submission_id: str = typer.Option("session_pipeline"),
        blueprints_artifact_id: str = typer.Option("blueprints"),
        blueprint_index: int = typer.Option(0),
        artifact_id: str = typer.Option("sandbox"),
        limit: int | None = typer.Option(None),
        force: bool = typer.Option(True, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_run_sandbox_command(
                    workspace=workspace,
                    context_path=context_path,
                    target_path=target_path,
                    reference_path=reference_path,
                    reference_sample_limit=reference_sample_limit,
                    submission_id=submission_id,
                    blueprints_artifact_id=blueprints_artifact_id,
                    blueprint_index=blueprint_index,
                    artifact_id=artifact_id,
                    limit=limit,
                    force=force,
                )
            )
        )

    @app.command("agent-hypothesis-record-outcome")
    def agent_hypothesis_record_outcome_cmd(
        workspace: str = typer.Option(...),
        cards_artifact_id: str = typer.Option("knowledge_cards"),
        hypotheses_artifact_id: str = typer.Option("hypotheses"),
        blueprints_artifact_id: str = typer.Option("blueprints"),
        sandbox_artifact_id: str = typer.Option("sandbox"),
        failure_memory_output_path: str | None = typer.Option(None),
        artifact_id: str = typer.Option("outcome"),
        force: bool = typer.Option(True, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_record_outcome_command(
                    workspace=workspace,
                    cards_artifact_id=cards_artifact_id,
                    hypotheses_artifact_id=hypotheses_artifact_id,
                    blueprints_artifact_id=blueprints_artifact_id,
                    sandbox_artifact_id=sandbox_artifact_id,
                    failure_memory_output_path=failure_memory_output_path,
                    artifact_id=artifact_id,
                    force=force,
                )
            )
        )

    @app.command("agent-hypothesis-run-pipeline")
    def agent_hypothesis_run_pipeline_cmd(
        workspace: str = typer.Option(...),
        manual_report_paths: list[str] = typer.Option([], "--manual-report-path"),
        adaptive_report_paths: list[str] = typer.Option([], "--adaptive-report-path"),
        log_paths: list[str] = typer.Option([], "--log-path"),
        checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
        literature_paths: list[str] = typer.Option([], "--literature-path"),
        algorithm_candidate_paths: list[str] = typer.Option(
            [], "--algorithm-candidates-path"
        ),
        failure_memory_paths: list[str] = typer.Option([], "--failure-memory-path"),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        reference_path: str | None = typer.Option(None),
        reference_sample_limit: int | None = typer.Option(None),
        submission_id: str = typer.Option("session_pipeline"),
        failure_memory_output_path: str | None = typer.Option(None),
        force: bool = typer.Option(True, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_run_pipeline_command(
                    workspace=workspace,
                    manual_report_paths=manual_report_paths,
                    adaptive_report_paths=adaptive_report_paths or None,
                    log_paths=log_paths or None,
                    checkpoint_paths=checkpoint_paths or None,
                    literature_paths=literature_paths or None,
                    algorithm_candidate_paths=algorithm_candidate_paths or None,
                    failure_memory_paths=failure_memory_paths or None,
                    context_path=context_path,
                    target_path=target_path,
                    reference_path=reference_path,
                    reference_sample_limit=reference_sample_limit,
                    submission_id=submission_id,
                    failure_memory_output_path=failure_memory_output_path,
                    force=force,
                )
            )
        )

    @app.command("agent-hypothesis-query-state")
    def agent_hypothesis_query_state_cmd(
        workspace: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_query_state_command(
                    workspace=workspace,
                )
            )
        )

    @app.command("agent-hypothesis-list-candidates")
    def agent_hypothesis_list_candidates_cmd(
        workspace: str = typer.Option(...),
        family: str | None = typer.Option(None),
        status: str | None = typer.Option(None),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_list_candidates_command(
                    workspace=workspace,
                    family=family,
                    status=status,
                )
            )
        )

    @app.command("agent-hypothesis-get-best-evidence")
    def agent_hypothesis_get_best_evidence_cmd(
        workspace: str = typer.Option(...),
        metric: str = typer.Option("score"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_get_best_evidence_command(
                    workspace=workspace,
                    metric=metric,
                )
            )
        )

    @app.command("agent-hypothesis-get-failure-pattern")
    def agent_hypothesis_get_failure_pattern_cmd(
        workspace: str = typer.Option(...),
        family: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_get_failure_pattern_command(
                    workspace=workspace,
                    family=family,
                )
            )
        )

    @app.command("agent-hypothesis-list-pending-blueprints")
    def agent_hypothesis_list_pending_blueprints_cmd(
        workspace: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_list_pending_blueprints_command(
                    workspace=workspace,
                )
            )
        )

    @app.command("agent-hypothesis-get-pareto-front")
    def agent_hypothesis_get_pareto_front_cmd(
        workspace: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_get_pareto_front_command(
                    workspace=workspace,
                )
            )
        )

    @app.command("agent-hypothesis-summarize-events")
    def agent_hypothesis_summarize_events_cmd(
        workspace: str = typer.Option(...),
        n: int = typer.Option(20),
        event_type: str | None = typer.Option(None),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_summarize_events_command(
                    workspace=workspace,
                    n=n,
                    event_type=event_type,
                )
            )
        )

    @app.command("agent-hypothesis-get-global-pareto-front")
    def agent_hypothesis_get_global_pareto_front_cmd(
        workspace: str = typer.Option(...),
        memory_path: str | None = typer.Option(None, "--memory-path"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_get_global_pareto_front_command(
                    workspace=workspace,
                    memory_path=memory_path,
                )
            )
        )

    @app.command("agent-hypothesis-get-family-history")
    def agent_hypothesis_get_family_history_cmd(
        family: str = typer.Option(...),
        memory_path: str | None = typer.Option(None, "--memory-path"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_get_family_history_command(
                    family=family,
                    memory_path=memory_path,
                )
            )
        )

    @app.command("agent-hypothesis-get-lessons-learned")
    def agent_hypothesis_get_lessons_learned_cmd(
        n: int = typer.Option(10),
        memory_path: str | None = typer.Option(None, "--memory-path"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_get_lessons_learned_command(
                    n=n,
                    memory_path=memory_path,
                )
            )
        )

    @app.command("agent-hypothesis-run-autonomous-loop")
    def agent_hypothesis_run_autonomous_loop_cmd(
        workspace: str = typer.Option(...),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        reference_path: str | None = typer.Option(None),
        reference_sample_limit: int | None = typer.Option(None),
        manual_report_paths: list[str] = typer.Option([], "--manual-report-path"),
        max_iterations: int = typer.Option(10),
        autonomy_level: str = typer.Option("deterministic"),
        submission_id: str = typer.Option("autonomous_loop"),
        force: bool = typer.Option(True, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                agent_hypothesis_run_autonomous_loop_command(
                    workspace=workspace,
                    context_path=context_path,
                    target_path=target_path,
                    reference_path=reference_path,
                    reference_sample_limit=reference_sample_limit,
                    manual_report_paths=manual_report_paths or None,
                    max_iterations=max_iterations,
                    autonomy_level=autonomy_level,
                    submission_id=submission_id,
                    force=force,
                )
            )
        )

    @app.command("build-hypothesis-lab-tool-catalog")
    def build_hypothesis_lab_tool_catalog_cmd(
        output_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_hypothesis_lab_tool_catalog_command(
                    output_path=output_path,
                    force=force,
                )
            )
        )
