"""CLI commands for literature domain."""

import json
import time

import typer

from pde_agent.commands.run_literature_strategy_expansion import (
    build_local_asset_algorithm_candidates_command,
    convert_candidate_promotion_to_strategy_evidence_command,
    evaluate_expansion_sandbox_queue_command,
    plan_candidate_evidence_accumulation_command,
    prepare_expansion_blocked_executor_authoring_tasks_command,
    record_promoted_algorithm_strategy_proposal_command,
    refine_algorithm_candidate_command,
    run_candidate_expansion_handoff_command,
    run_expansion_sandbox_queue_command,
    run_literature_strategy_expansion_command,
    summarize_literature_registry_command,
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
from pde_agent.commands.task1_candidate_factory import (
    evaluate_candidate_factory_capability_command,
    generate_task1_candidate_families_command,
    materialize_candidate_family_item_command,
    run_candidate_family_sandbox_command,
)
from pde_agent.commands.task1_fno_graft_evaluator import (
    run_task1_fno_graft_from_optimization_report_command,
    run_task1_fno_graft_evaluator_command,
)
from pde_agent.commands.literature_algorithm_lab import (
    adapt_fno_architecture_candidate_command,
    audit_bounded_executor_authoring_task_command,
    build_blocked_candidate_handoff_command,
    build_executor_taxonomy_command,
    build_literature_source_records_command,
    complete_bounded_executor_authoring_task_command,
    compile_algorithm_candidates_command,
    create_bounded_executor_design_spec_command,
    create_executor_design_request_command,
    download_mineru_result_command,
    download_source_document_command,
    evaluate_algorithm_candidate_promotion_command,
    extract_algorithm_cards_command,
    ingest_mineru_parse_command,
    materialize_bounded_executor_template_command,
    normalize_arxiv_sources_command,
    normalize_github_sources_command,
    normalize_openalex_sources_command,
    plan_candidate_sandbox_command,
    poll_mineru_parse_command,
    prepare_bounded_executor_authoring_task_command,
    prepare_deeponet_literature_adaptation_task_command,
    prepare_fno_literature_adaptation_task_command,
    prepare_literature_extraction_context_command,
    prepare_unet_literature_adaptation_task_command,
    record_agent_algorithm_cards_command,
    record_bounded_executor_sandbox_evidence_command,
    record_bounded_executor_sandbox_event_command,
    record_literature_registry_event_command,
    run_arxiv_mineru_parse_pipeline_command,
    run_bounded_executor_sandbox_command,
    run_literature_algorithm_v2_pipeline_command,
    submit_mineru_parse_command,
    update_literature_registry_command,
    validate_algorithm_cards_v2_command,
    validate_bounded_executor_implementation_command,
    validate_executable_blueprints_command,
)
from pde_agent.commands.run_checkpoint_family_sandbox import (
    run_checkpoint_family_sandbox_command,
)

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

    @app.command("prepare-pde-literature-extraction-skill")
    def prepare_pde_literature_extraction_skill_cmd(
        literature_extraction_context_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        operator_family: str = typer.Option(...),
        target_problem: str = typer.Option("PDE_Burgers"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                prepare_pde_literature_extraction_skill_command(
                    literature_extraction_context_path=literature_extraction_context_path,
                    output_path=output_path,
                    operator_family=operator_family,
                    target_problem=target_problem,
                    force=force,
                )
            )
        )

    @app.command("generate-task1-candidate-families")
    def generate_task1_candidate_families_cmd(
        benchmark_path: str = typer.Option(...),
        chain_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        prior_registry_path: str | None = typer.Option(None),
        max_families: int = typer.Option(5),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                generate_task1_candidate_families_command(
                    benchmark_path=benchmark_path,
                    chain_report_path=chain_report_path,
                    output_path=output_path,
                    prior_registry_path=prior_registry_path,
                    max_families=max_families,
                    force=force,
                )
            )
        )

    @app.command("run-candidate-family-sandbox")
    def run_candidate_family_sandbox_cmd(
        family_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_candidate_family_sandbox_command(
                    family_report_path=family_report_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("evaluate-candidate-factory-capability")
    def evaluate_candidate_factory_capability_cmd(
        family_report_path: str = typer.Option(...),
        sandbox_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        min_family_count: int = typer.Option(3),
        min_executable_count: int = typer.Option(1),
        min_mechanism_count: int = typer.Option(3),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                evaluate_candidate_factory_capability_command(
                    family_report_path=family_report_path,
                    sandbox_report_path=sandbox_report_path,
                    output_path=output_path,
                    min_family_count=min_family_count,
                    min_executable_count=min_executable_count,
                    min_mechanism_count=min_mechanism_count,
                    force=force,
                )
            )
        )

    @app.command("materialize-candidate-family-item")
    def materialize_candidate_family_item_cmd(
        sandbox_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        candidate_family_id: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                materialize_candidate_family_item_command(
                    sandbox_report_path=sandbox_report_path,
                    output_path=output_path,
                    candidate_family_id=candidate_family_id,
                    force=force,
                )
            )
        )

    @app.command("normalize-openalex-sources")
    def normalize_openalex_sources_cmd(
        search_results_path: str = typer.Option(...),
        query: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                normalize_openalex_sources_command(
                    search_results_path=search_results_path,
                    query=query,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("normalize-arxiv-sources")
    def normalize_arxiv_sources_cmd(
        search_results_path: str = typer.Option(...),
        query: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                normalize_arxiv_sources_command(
                    search_results_path=search_results_path,
                    query=query,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("normalize-github-sources")
    def normalize_github_sources_cmd(
        search_results_path: str = typer.Option(...),
        query: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                normalize_github_sources_command(
                    search_results_path=search_results_path,
                    query=query,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("run-arxiv-mineru-parse-pipeline")
    def run_arxiv_mineru_parse_pipeline_cmd(
        query: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        mode: str = typer.Option("precise_url"),
        model_version: str = typer.Option("vlm"),
        max_results: int = typer.Option(5),
        selected_index: int = typer.Option(-1),
        max_polls: int = typer.Option(3),
        poll_interval_seconds: float = typer.Option(5.0),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_arxiv_mineru_parse_pipeline_command(
                    query=query,
                    output_dir=output_dir,
                    mode=mode,
                    model_version=model_version,
                    max_results=max_results,
                    selected_index=selected_index,
                    max_polls=max_polls,
                    poll_interval_seconds=poll_interval_seconds,
                    force=force,
                )
            )
        )

    @app.command("download-source-document")
    def download_source_document_cmd(
        source_candidates_path: str = typer.Option(...),
        source_id: str = typer.Option(...),
        output_path: str = typer.Option(...),
        cache_dir: str = typer.Option("data/literature_lab/sources"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                download_source_document_command(
                    source_candidates_path=source_candidates_path,
                    source_id=source_id,
                    output_path=output_path,
                    cache_dir=cache_dir,
                    force=force,
                )
            )
        )

    @app.command("build-literature-source-records")
    def build_literature_source_records_cmd(
        source_fixture_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_literature_source_records_command(
                    source_fixture_path=source_fixture_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("prepare-literature-extraction-context")
    def prepare_literature_extraction_context_cmd(
        paper_parse_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                prepare_literature_extraction_context_command(
                    paper_parse_path=paper_parse_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("prepare-fno-literature-adaptation-task")
    def prepare_fno_literature_adaptation_task_cmd(
        literature_extraction_context_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                prepare_fno_literature_adaptation_task_command(
                    literature_extraction_context_path=literature_extraction_context_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("prepare-deeponet-literature-adaptation-task")
    def prepare_deeponet_literature_adaptation_task_cmd(
        literature_extraction_context_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                prepare_deeponet_literature_adaptation_task_command(
                    literature_extraction_context_path=literature_extraction_context_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("prepare-unet-literature-adaptation-task")
    def prepare_unet_literature_adaptation_task_cmd(
        literature_extraction_context_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                prepare_unet_literature_adaptation_task_command(
                    literature_extraction_context_path=literature_extraction_context_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("update-literature-registry")
    def update_literature_registry_cmd(
        registry_path: str = typer.Option(...),
        downloaded_source_paths: list[str] = typer.Option(
            [],
            "--downloaded-source-path",
        ),
        paper_parse_paths: list[str] = typer.Option([], "--paper-parse-path"),
        algorithm_cards_paths: list[str] = typer.Option([], "--algorithm-cards-path"),
        algorithm_candidates_paths: list[str] = typer.Option(
            [],
            "--algorithm-candidates-path",
        ),
        event_paths: list[str] = typer.Option([], "--event-path"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                update_literature_registry_command(
                    registry_path=registry_path,
                    downloaded_source_paths=downloaded_source_paths,
                    paper_parse_paths=paper_parse_paths,
                    algorithm_cards_paths=algorithm_cards_paths,
                    algorithm_candidates_paths=algorithm_candidates_paths,
                    event_paths=event_paths,
                    force=force,
                )
            )
        )

    @app.command("record-literature-registry-event")
    def record_literature_registry_event_cmd(
        event_source_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_literature_registry_event_command(
                    event_source_path=event_source_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("submit-mineru-parse")
    def submit_mineru_parse_cmd(
        downloaded_source_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        mode: str = typer.Option("precise_url"),
        model_version: str = typer.Option("vlm"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                submit_mineru_parse_command(
                    downloaded_source_path=downloaded_source_path,
                    output_path=output_path,
                    mode=mode,
                    model_version=model_version,
                    force=force,
                )
            )
        )

    @app.command("poll-mineru-parse")
    def poll_mineru_parse_cmd(
        mineru_task_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                poll_mineru_parse_command(
                    mineru_task_path=mineru_task_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("download-mineru-result")
    def download_mineru_result_cmd(
        mineru_poll_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                download_mineru_result_command(
                    mineru_poll_path=mineru_poll_path,
                    output_path=output_path,
                    output_dir=output_dir,
                    force=force,
                )
            )
        )

    @app.command("ingest-mineru-parse")
    def ingest_mineru_parse_cmd(
        source_records_path: str = typer.Option(...),
        source_id: str = typer.Option(...),
        markdown_path: str = typer.Option(...),
        asset_dir: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                ingest_mineru_parse_command(
                    source_records_path=source_records_path,
                    source_id=source_id,
                    markdown_path=markdown_path,
                    asset_dir=asset_dir,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("extract-algorithm-cards")
    def extract_algorithm_cards_cmd(
        paper_parse_paths: list[str] = typer.Option(..., "--paper-parse-path"),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                extract_algorithm_cards_command(
                    paper_parse_paths=paper_parse_paths,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("record-agent-algorithm-cards")
    def record_agent_algorithm_cards_cmd(
        cards_json: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_agent_algorithm_cards_command(
                    cards_json=cards_json,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("compile-algorithm-candidates")
    def compile_algorithm_candidates_cmd(
        algorithm_cards_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                compile_algorithm_candidates_command(
                    algorithm_cards_path=algorithm_cards_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("validate-algorithm-cards-v2")
    def validate_algorithm_cards_v2_cmd(
        algorithm_cards_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                validate_algorithm_cards_v2_command(
                    algorithm_cards_path=algorithm_cards_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("validate-executable-blueprints")
    def validate_executable_blueprints_cmd(
        blueprints_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                validate_executable_blueprints_command(
                    blueprints_path=blueprints_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("run-literature-algorithm-v2-pipeline")
    def run_literature_algorithm_v2_pipeline_cmd(
        algorithm_cards_path: str = typer.Option(...),
        workspace_dir: str = typer.Option(...),
        registry_path: str = typer.Option(...),
        current_best_score: float = typer.Option(...),
        algorithm_candidate_id: str | None = typer.Option(None),
        sandbox_evidence_paths: list[str] = typer.Option([], "--sandbox-evidence-path"),
        paper_parse_paths: list[str] = typer.Option([], "--paper-parse-path"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_literature_algorithm_v2_pipeline_command(
                    algorithm_cards_path=algorithm_cards_path,
                    workspace_dir=workspace_dir,
                    registry_path=registry_path,
                    current_best_score=current_best_score,
                    algorithm_candidate_id=algorithm_candidate_id,
                    sandbox_evidence_paths=sandbox_evidence_paths,
                    paper_parse_paths=paper_parse_paths,
                    force=force,
                )
            )
        )

    @app.command("plan-algorithm-candidate-sandbox")
    def plan_algorithm_candidate_sandbox_cmd(
        algorithm_candidates_path: str = typer.Option(...),
        algorithm_candidate_id: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_candidate_sandbox_command(
                    algorithm_candidates_path=algorithm_candidates_path,
                    algorithm_candidate_id=algorithm_candidate_id,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("evaluate-algorithm-candidate-promotion")
    def evaluate_algorithm_candidate_promotion_cmd(
        algorithm_candidates_path: str = typer.Option(...),
        algorithm_candidate_id: str = typer.Option(...),
        sandbox_evidence_paths: list[str] = typer.Option(..., "--sandbox-evidence-path"),
        output_path: str = typer.Option(...),
        current_best_score: float = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                evaluate_algorithm_candidate_promotion_command(
                    algorithm_candidates_path=algorithm_candidates_path,
                    algorithm_candidate_id=algorithm_candidate_id,
                    sandbox_evidence_paths=sandbox_evidence_paths,
                    output_path=output_path,
                    current_best_score=current_best_score,
                    force=force,
                )
            )
        )

    @app.command("build-executor-taxonomy")
    def build_executor_taxonomy_cmd(
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_executor_taxonomy_command(
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("adapt-fno-architecture-candidate")
    def adapt_fno_architecture_candidate_cmd(
        algorithm_candidates_path: str = typer.Option(...),
        algorithm_candidate_id: str = typer.Option(...),
        checkpoint_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                adapt_fno_architecture_candidate_command(
                    algorithm_candidates_path=algorithm_candidates_path,
                    algorithm_candidate_id=algorithm_candidate_id,
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("create-executor-design-request")
    def create_executor_design_request_cmd(
        algorithm_candidates_path: str = typer.Option(...),
        algorithm_candidate_id: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                create_executor_design_request_command(
                    algorithm_candidates_path=algorithm_candidates_path,
                    algorithm_candidate_id=algorithm_candidate_id,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-blocked-candidate-handoff")
    def build_blocked_candidate_handoff_cmd(
        executor_design_request_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_blocked_candidate_handoff_command(
                    executor_design_request_path=executor_design_request_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("create-bounded-executor-design-spec")
    def create_bounded_executor_design_spec_cmd(
        executor_design_request_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                create_bounded_executor_design_spec_command(
                    executor_design_request_path=executor_design_request_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("materialize-bounded-executor-template")
    def materialize_bounded_executor_template_cmd(
        design_spec_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                materialize_bounded_executor_template_command(
                    design_spec_path=design_spec_path,
                    output_dir=output_dir,
                    force=force,
                )
            )
        )

    @app.command("validate-bounded-executor-implementation")
    def validate_bounded_executor_implementation_cmd(
        design_spec_path: str = typer.Option(...),
        manifest_path: str = typer.Option(...),
        executor_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                validate_bounded_executor_implementation_command(
                    design_spec_path=design_spec_path,
                    manifest_path=manifest_path,
                    executor_path=executor_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("run-bounded-executor-sandbox")
    def run_bounded_executor_sandbox_cmd(
        design_spec_path: str = typer.Option(...),
        manifest_path: str = typer.Option(...),
        executor_path: str = typer.Option(...),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option("team_default"),
        trace_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_bounded_executor_sandbox_command(
                    design_spec_path=design_spec_path,
                    manifest_path=manifest_path,
                    executor_path=executor_path,
                    context_path=context_path,
                    target_path=target_path,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    trace_path=trace_path,
                    force=force,
                )
            )
        )

    @app.command("record-bounded-executor-sandbox-event")
    def record_bounded_executor_sandbox_event_cmd(
        sandbox_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_bounded_executor_sandbox_event_command(
                    sandbox_report_path=sandbox_report_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("record-bounded-executor-sandbox-evidence")
    def record_bounded_executor_sandbox_evidence_cmd(
        sandbox_report_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        current_best_score: float = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_bounded_executor_sandbox_evidence_command(
                    sandbox_report_path=sandbox_report_path,
                    output_path=output_path,
                    current_best_score=current_best_score,
                    force=force,
                )
            )
        )

    @app.command("prepare-bounded-executor-authoring-task")
    def prepare_bounded_executor_authoring_task_cmd(
        design_spec_path: str = typer.Option(...),
        manifest_path: str = typer.Option(...),
        executor_path: str = typer.Option(...),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        sandbox_output_dir: str = typer.Option(...),
        audit_output_path: str = typer.Option(...),
        event_output_path: str = typer.Option(...),
        submission_id: str = typer.Option("team_default"),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                prepare_bounded_executor_authoring_task_command(
                    design_spec_path=design_spec_path,
                    manifest_path=manifest_path,
                    executor_path=executor_path,
                    context_path=context_path,
                    target_path=target_path,
                    output_path=output_path,
                    sandbox_output_dir=sandbox_output_dir,
                    audit_output_path=audit_output_path,
                    event_output_path=event_output_path,
                    submission_id=submission_id,
                    force=force,
                )
            )
        )

    @app.command("audit-bounded-executor-authoring-task")
    def audit_bounded_executor_authoring_task_cmd(
        authoring_task_path: str = typer.Option(...),
        skill_root: str = typer.Option("skills"),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                audit_bounded_executor_authoring_task_command(
                    authoring_task_path=authoring_task_path,
                    skill_root=skill_root,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("complete-bounded-executor-authoring-task")
    def complete_bounded_executor_authoring_task_cmd(
        authoring_task_path: str = typer.Option(...),
        algorithm_candidates_path: str = typer.Option(...),
        registry_path: str = typer.Option(...),
        current_best_score: float = typer.Option(...),
        skill_root: str = typer.Option("skills"),
        trace_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                complete_bounded_executor_authoring_task_command(
                    authoring_task_path=authoring_task_path,
                    algorithm_candidates_path=algorithm_candidates_path,
                    registry_path=registry_path,
                    current_best_score=current_best_score,
                    skill_root=skill_root,
                    force=force,
                    trace_path=trace_path,
                )
            )
        )

    @app.command("run-checkpoint-family-sandbox")
    def run_checkpoint_family_sandbox_cmd(
        checkpoint_paths: list[str] = typer.Option(..., "--checkpoint-path"),
        task_id: int = typer.Option(...),
        test_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option("team_default"),
        dataset_key: str = typer.Option("tensor"),
        prediction_dataset_key: str = typer.Option("tensor"),
        calibration_weights: list[float] = typer.Option([1.0], "--calibration-weight"),
        fno_finetune_steps: int = typer.Option(0),
        timing_profile: str | None = typer.Option(None),
        timing_factor: float | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_checkpoint_family_sandbox_command(
                    checkpoint_paths=checkpoint_paths,
                    task_id=task_id,
                    test_path=test_path,
                    target_path=target_path,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    dataset_key=dataset_key,
                    prediction_dataset_key=prediction_dataset_key,
                    calibration_weights=calibration_weights,
                    fno_finetune_steps=fno_finetune_steps,
                    timing_profile=timing_profile,
                    timing_factor=timing_factor,
                    force=force,
                )
            )
        )

    @app.command("run-candidate-expansion-handoff")
    def run_candidate_expansion_handoff_cmd(
        candidate_plan_path: str = typer.Option(...),
        workspace_dir: str = typer.Option(...),
        local_strategy_library_path: str | None = typer.Option(None),
        checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
        max_strategy_candidates: int | None = typer.Option(None),
        failure_memory_path: str | None = typer.Option(None),
        registry_summary_path: str | None = typer.Option(None),
        trace_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_candidate_expansion_handoff_command(
                    candidate_plan_path=candidate_plan_path,
                    workspace_dir=workspace_dir,
                    local_strategy_library_path=local_strategy_library_path,
                    checkpoint_paths=checkpoint_paths,
                    max_strategy_candidates=max_strategy_candidates,
                    failure_memory_path=failure_memory_path,
                    registry_summary_path=registry_summary_path,
                    trace_path=trace_path,
                    force=force,
                )
            )
        )

    @app.command("run-literature-strategy-expansion")
    def run_literature_strategy_expansion_cmd(
        algorithm_candidates_path: str = typer.Option(...),
        workspace_dir: str = typer.Option(...),
        checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
        failure_memory_path: str | None = typer.Option(None),
        registry_summary_path: str | None = typer.Option(None),
        trace_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_literature_strategy_expansion_command(
                    algorithm_candidates_path=algorithm_candidates_path,
                    workspace_dir=workspace_dir,
                    checkpoint_paths=checkpoint_paths,
                    failure_memory_path=failure_memory_path,
                    registry_summary_path=registry_summary_path,
                    trace_path=trace_path,
                    force=force,
                )
            )
        )

    @app.command("run-expansion-sandbox-queue")
    def run_expansion_sandbox_queue_cmd(
        expansion_report_path: str = typer.Option(...),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        initial_submission_dir: str | None = typer.Option(None),
        output_dir: str = typer.Option(...),
        submission_id: str = typer.Option("team_default"),
        max_items: int | None = typer.Option(None),
        failure_memory_path: str | None = typer.Option(None),
        reference_path: str | None = typer.Option(None),
        reference_sample_limit: int | None = typer.Option(None),
        timing_profile: str | None = typer.Option(None),
        timing_factor: float | None = typer.Option(None),
        selected_algorithm_candidate_id: str | None = typer.Option(None),
        trace_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                run_expansion_sandbox_queue_command(
                    expansion_report_path=expansion_report_path,
                    context_path=context_path,
                    target_path=target_path,
                    initial_submission_dir=initial_submission_dir,
                    output_dir=output_dir,
                    submission_id=submission_id,
                    max_items=max_items,
                    failure_memory_path=failure_memory_path,
                    reference_path=reference_path,
                    reference_sample_limit=reference_sample_limit,
                    timing_profile=timing_profile,
                    timing_factor=timing_factor,
                    selected_algorithm_candidate_id=selected_algorithm_candidate_id,
                    trace_path=trace_path,
                    force=force,
                )
            )
        )

    @app.command("evaluate-expansion-sandbox-queue")
    def evaluate_expansion_sandbox_queue_cmd(
        queue_report_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        registry_path: str = typer.Option(...),
        current_best_score: float = typer.Option(...),
        trace_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                evaluate_expansion_sandbox_queue_command(
                    queue_report_path=queue_report_path,
                    output_dir=output_dir,
                    registry_path=registry_path,
                    current_best_score=current_best_score,
                    trace_path=trace_path,
                    force=force,
                )
            )
        )

    @app.command("summarize-literature-registry")
    def summarize_literature_registry_cmd(
        registry_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        current_best_score: float = typer.Option(...),
        trace_path: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                summarize_literature_registry_command(
                    registry_path=registry_path,
                    output_path=output_path,
                    current_best_score=current_best_score,
                    trace_path=trace_path,
                    force=force,
                )
            )
        )

    @app.command("prepare-expansion-blocked-executor-authoring-tasks")
    def prepare_expansion_blocked_executor_authoring_tasks_cmd(
        queue_report_path: str = typer.Option(...),
        output_dir: str = typer.Option(...),
        context_path: str = typer.Option(...),
        target_path: str = typer.Option(...),
        submission_id: str = typer.Option("team_default"),
        max_tasks: int | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                prepare_expansion_blocked_executor_authoring_tasks_command(
                    queue_report_path=queue_report_path,
                    output_dir=output_dir,
                    context_path=context_path,
                    target_path=target_path,
                    submission_id=submission_id,
                    max_tasks=max_tasks,
                    force=force,
                )
            )
        )

    @app.command("refine-algorithm-candidate")
    def refine_algorithm_candidate_cmd(
        registry_summary_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        algorithm_candidate_id: str | None = typer.Option(None),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                refine_algorithm_candidate_command(
                    registry_summary_path=registry_summary_path,
                    output_path=output_path,
                    algorithm_candidate_id=algorithm_candidate_id,
                    force=force,
                )
            )
        )

    @app.command("record-promoted-algorithm-strategy-proposal")
    def record_promoted_algorithm_strategy_proposal_cmd(
        candidate_promotion_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                record_promoted_algorithm_strategy_proposal_command(
                    candidate_promotion_path=candidate_promotion_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("convert-candidate-promotion-to-strategy-evidence")
    def convert_candidate_promotion_to_strategy_evidence_cmd(
        candidate_promotion_path: str = typer.Option(...),
        extra_candidate_promotion_paths: list[str] = typer.Option(
            [],
            "--extra-candidate-promotion-path",
        ),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                convert_candidate_promotion_to_strategy_evidence_command(
                    candidate_promotion_path=candidate_promotion_path,
                    extra_candidate_promotion_paths=extra_candidate_promotion_paths,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("plan-candidate-evidence-accumulation")
    def plan_candidate_evidence_accumulation_cmd(
        strategy_promotion_report_path: str = typer.Option(...),
        strategy_evidence_summary_path: str | None = typer.Option(None),
        output_path: str = typer.Option(...),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                plan_candidate_evidence_accumulation_command(
                    strategy_promotion_report_path=strategy_promotion_report_path,
                    strategy_evidence_summary_path=strategy_evidence_summary_path,
                    output_path=output_path,
                    force=force,
                )
            )
        )

    @app.command("build-local-asset-algorithm-candidates")
    def build_local_asset_algorithm_candidates_cmd(
        local_strategy_library_path: str = typer.Option(...),
        output_path: str = typer.Option(...),
        checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
        max_strategy_candidates: int = typer.Option(5),
        force: bool = typer.Option(False, "--force"),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(
            json.dumps(
                build_local_asset_algorithm_candidates_command(
                    local_strategy_library_path=local_strategy_library_path,
                    checkpoint_paths=checkpoint_paths,
                    output_path=output_path,
                    max_strategy_candidates=max_strategy_candidates,
                    force=force,
                )
            )
        )
