"""CLI commands for hypothesis domain."""

import json
import time

import typer

from pde_agent.commands.hypothesis_lab_policy import (
    build_hypothesis_lab_next_policy_command,
)
from pde_agent.commands.hypothesis_lab_runtime import (
    adapt_hypothesis_algorithm_sandbox_item_command,
    audit_agent_runtime_availability_command,
    build_fno_training_harness_contract_command,
    build_hypothesis_algorithm_design_command,
    build_hypothesis_kpi_report_command,
    build_scientific_tool_catalog_command,
    detect_hypothesis_doom_loop_command,
    plan_fno_training_candidates_command,
    plan_hypothesis_algorithm_sandbox_command,
    plan_hypothesis_candidates_command,
    prepare_pde_literature_extraction_skill_command,
    record_fno_training_optimization_evidence_command,
    record_hypothesis_algorithm_evidence_command,
    record_hypothesis_lab_trace_event_command,
    run_fno_training_optimization_loop_command,
    run_hypothesis_algorithm_sandbox_command,
    run_hypothesis_algorithm_designer_stability_batch_command,
    run_task1_short_window_family_selection_loop_command,
    run_task1_short_window_training_loop_command,
    summarize_hypothesis_lab_trace_command,
    validate_hypothesis_algorithm_design_command,
)
from pde_agent.commands.run_hypothesis_lab import run_hypothesis_lab_command

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("run-hypothesis-lab")
    def run_hypothesis_lab_cmd(
        manual_report_paths: list[str] = typer.Option([], "--manual-report-path"),
        adaptive_report_paths: list[str] = typer.Option([], "--adaptive-report-path"),
        log_paths: list[str] = typer.Option([], "--log-path"),
        checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
        literature_paths: list[str] = typer.Option([], "--literature-path"),
        algorithm_candidate_paths: list[str] = typer.Option(
            [],
            "--algorithm-candidates-path",
        ),
        failure_memory_paths: list[str] = typer.Option([], "--failure-memory-path"),
        failure_memory_output_path: str | None = typer.Option(
            None,
            "--failure-memory-output-path",
        ),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        reference_path: str | None = typer.Option(None),
        reference_sample_limit: int | None = typer.Option(None),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option("team_default"),
        timing_profile: str | None = typer.Option(None),
        timing_factor: float | None = typer.Option(None),
        trace_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_hypothesis_lab_command(
                    manual_report_paths=manual_report_paths,
                    adaptive_report_paths=adaptive_report_paths,
                    log_paths=log_paths,
                    checkpoint_paths=checkpoint_paths,
                    literature_paths=literature_paths,
                    algorithm_candidate_paths=algorithm_candidate_paths,
                    failure_memory_paths=failure_memory_paths,
                    failure_memory_output_path=failure_memory_output_path,
                    context_path=context_path,
                    target_path=target_path,
                    reference_path=reference_path,
                    reference_sample_limit=reference_sample_limit,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    timing_profile=timing_profile,
                    timing_factor=timing_factor,
                    trace_path=trace_path,
                    force=force,
                )
            )
        )

    @app.command("record-hypothesis-lab-trace-event")
    def record_hypothesis_lab_trace_event_cmd(
        trace_path: str = typer.Option(...),
        event_type: str = typer.Option(...),
        payload_json: str = typer.Option(...),
        run_id: str | None = typer.Option(None),
        source_artifact_path: str | None = typer.Option(None),
        tags: list[str] = typer.Option([], "--tag"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_hypothesis_lab_trace_event_command(
                    trace_path=trace_path,
                    event_type=event_type,
                    payload_json=payload_json,
                    run_id=run_id,
                    source_artifact_path=source_artifact_path,
                    tags=tags,
                )
            )
        )

    @app.command("summarize-hypothesis-lab-trace")
    def summarize_hypothesis_lab_trace_cmd(
        trace_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                summarize_hypothesis_lab_trace_command(
                    trace_path=trace_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("detect-hypothesis-doom-loop")
    def detect_hypothesis_doom_loop_cmd(
        trace_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        recent_window: int = typer.Option(20),
        repeated_family_threshold: int = typer.Option(3),
        failed_family_threshold: int = typer.Option(2),
        min_score_delta: float = typer.Option(0.1),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                detect_hypothesis_doom_loop_command(
                    trace_path=trace_path,
                    output_path=output_path,
                    recent_window=recent_window,
                    repeated_family_threshold=repeated_family_threshold,
                    failed_family_threshold=failed_family_threshold,
                    min_score_delta=min_score_delta,
                    force=force,
                )
            )
        )

    @app.command("build-scientific-tool-catalog")
    def build_scientific_tool_catalog_cmd(
        output_path: str = typer.Option(...),
        include_compatibility_tools: bool = typer.Option(
            False,
            "--include-compatibility-tools",
        ),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_scientific_tool_catalog_command(
                    output_path=output_path,
                    include_compatibility_tools=include_compatibility_tools,
                    force=force,
                )
            )
        )

    @app.command("build-hypothesis-kpi-report")
    def build_hypothesis_kpi_report_cmd(
        trace_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        recent_window: int = typer.Option(50),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_hypothesis_kpi_report_command(
                    trace_path=trace_path,
                    output_path=output_path,
                    recent_window=recent_window,
                    force=force,
                )
            )
        )

    @app.command("plan-hypothesis-candidates")
    def plan_hypothesis_candidates_cmd(
        output_path: str = typer.Option(...),
        candidate_generator: str = typer.Option("postprocess"),
        hypothesis_lab_next_policy_path: str | None = typer.Option(None),
        doom_loop_report_path: str | None = typer.Option(None),
        hypothesis_kpi_report_path: str | None = typer.Option(None),
        repair_policy_summary_path: str | None = typer.Option(None),
        research_chronicle_path: str | None = typer.Option(None),
        expansion_handoff_report_path: str | None = typer.Option(None),
        expansion_report_path: str | None = typer.Option(None),
        audit_repair_tasks_path: str | None = typer.Option(None),
        preferred_error_modes: list[str] = typer.Option([], "--preferred-error-mode"),
        max_schedule_length: int = typer.Option(5),
        min_viable_candidate_count: int = typer.Option(1),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_hypothesis_candidates_command(
                    output_path=output_path,
                    candidate_generator=candidate_generator,
                    hypothesis_lab_next_policy_path=hypothesis_lab_next_policy_path,
                    doom_loop_report_path=doom_loop_report_path,
                    hypothesis_kpi_report_path=hypothesis_kpi_report_path,
                    repair_policy_summary_path=repair_policy_summary_path,
                    research_chronicle_path=research_chronicle_path,
                    expansion_handoff_report_path=expansion_handoff_report_path,
                    expansion_report_path=expansion_report_path,
                    audit_repair_tasks_path=audit_repair_tasks_path,
                    preferred_error_modes=preferred_error_modes,
                    max_schedule_length=max_schedule_length,
                    min_viable_candidate_count=min_viable_candidate_count,
                    force=force,
                )
            )
        )

    @app.command("build-hypothesis-algorithm-design")
    def build_hypothesis_algorithm_design_cmd(
        output_path: str = typer.Option(...),
        failure_memory_paths: list[str] = typer.Option([], "--failure-memory-path"),
        evidence_paths: list[str] = typer.Option([], "--evidence-path"),
        log_paths: list[str] = typer.Option([], "--log-path"),
        strategy_library_paths: list[str] = typer.Option([], "--strategy-library-path"),
        literature_card_paths: list[str] = typer.Option([], "--literature-card-path"),
        min_hypothesis_count: int = typer.Option(5),
        min_mechanism_family_count: int = typer.Option(4),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_hypothesis_algorithm_design_command(
                    output_path=output_path,
                    failure_memory_paths=failure_memory_paths,
                    evidence_paths=evidence_paths,
                    log_paths=log_paths,
                    strategy_library_paths=strategy_library_paths,
                    literature_card_paths=literature_card_paths,
                    min_hypothesis_count=min_hypothesis_count,
                    min_mechanism_family_count=min_mechanism_family_count,
                    force=force,
                )
            )
        )

    @app.command("validate-hypothesis-algorithm-design")
    def validate_hypothesis_algorithm_design_cmd(
        design_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        min_hypothesis_count: int = typer.Option(5),
        min_mechanism_family_count: int = typer.Option(4),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                validate_hypothesis_algorithm_design_command(
                    design_path=design_path,
                    output_path=output_path,
                    min_hypothesis_count=min_hypothesis_count,
                    min_mechanism_family_count=min_mechanism_family_count,
                    force=force,
                )
            )
        )

    @app.command("record-hypothesis-algorithm-evidence")
    def record_hypothesis_algorithm_evidence_cmd(
        design_path: str = typer.Option(...),
        sandbox_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_hypothesis_algorithm_evidence_command(
                    design_path=design_path,
                    sandbox_report_path=sandbox_report_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("plan-hypothesis-algorithm-sandbox")
    def plan_hypothesis_algorithm_sandbox_cmd(
        design_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        max_ready_items: int = typer.Option(5),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_hypothesis_algorithm_sandbox_command(
                    design_path=design_path,
                    output_path=output_path,
                    max_ready_items=max_ready_items,
                    force=force,
                )
            )
        )

    @app.command("adapt-hypothesis-algorithm-sandbox-item")
    def adapt_hypothesis_algorithm_sandbox_item_cmd(
        sandbox_plan_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        blueprint_id: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                adapt_hypothesis_algorithm_sandbox_item_command(
                    sandbox_plan_path=sandbox_plan_path,
                    output_path=output_path,
                    blueprint_id=blueprint_id,
                    force=force,
                )
            )
        )

    @app.command("run-hypothesis-algorithm-sandbox")
    def run_hypothesis_algorithm_sandbox_cmd(
        adapted_blueprint_path: str = typer.Option(...),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option("team_default"),
        reference_path: str | None = typer.Option(None),
        reference_sample_limit: int | None = typer.Option(None),
        timing_profile: str | None = typer.Option(None),
        timing_factor: float | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_hypothesis_algorithm_sandbox_command(
                    adapted_blueprint_path=adapted_blueprint_path,
                    context_path=context_path,
                    target_path=target_path,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    reference_path=reference_path,
                    reference_sample_limit=reference_sample_limit,
                    timing_profile=timing_profile,
                    timing_factor=timing_factor,
                    force=force,
                )
            )
        )

    @app.command("run-hypothesis-algorithm-designer-stability-batch")
    def run_hypothesis_algorithm_designer_stability_batch_cmd(
        memory_set_json: str = typer.Option(...),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option("team_default"),
        reference_path: str | None = typer.Option(None),
        reference_sample_limit: int | None = typer.Option(None),
        initial_submission_dir: str | None = typer.Option(None),
        max_sandbox_items_per_set: int = typer.Option(1),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_hypothesis_algorithm_designer_stability_batch_command(
                    memory_set_json=memory_set_json,
                    context_path=context_path,
                    target_path=target_path,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    reference_path=reference_path,
                    reference_sample_limit=reference_sample_limit,
                    initial_submission_dir=initial_submission_dir,
                    max_sandbox_items_per_set=max_sandbox_items_per_set,
                    force=force,
                )
            )
        )

    @app.command("build-hypothesis-lab-next-policy")
    def build_hypothesis_lab_next_policy_cmd(
        output_path: str = typer.Option(...),
        failure_memory_path: str | None = typer.Option(None),
        hypothesis_lab_report_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_hypothesis_lab_next_policy_command(
                    output_path=output_path,
                    failure_memory_path=failure_memory_path,
                    hypothesis_lab_report_path=hypothesis_lab_report_path,
                    force=force,
                )
            )
        )
