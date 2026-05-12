import json
import time

import typer

from pde_agent.commands.agent_runtime import build_agent_runtime_contract_command
from pde_agent.commands.agent_runtime import (
    build_claude_fno_training_smoke_task_command,
    build_claude_hypothesis_lab_closure_task_command,
    build_claude_literature_candidate_closure_task_command,
    build_claude_promotion_smoke_task_command,
)
from pde_agent.commands.agent_runtime import export_competition_llm_log_command
from pde_agent.commands.agent_runtime import record_agent_runtime_issue_command
from pde_agent.commands.agent_runtime import run_claude_cli_runtime_command
from pde_agent.commands.agent_runtime import run_claude_agent_sdk_runtime_command
from pde_agent.commands.fno_training_closed_loop import (
    build_fno_training_closed_loop_contract_command,
    build_fno_training_final_report_command,
    build_fno_training_health_report_command,
    build_fno_training_memory_command,
    checkpoint_fno_training_loop_command,
    check_fno_training_convergence_command,
    escalate_fno_training_to_hypothesis_command,
    integrate_hypothesis_training_feedback_command,
    resume_fno_training_loop_command,
    run_fno_training_iteration_command,
)
from pde_agent.commands.fno_training_agent_loop import (
    build_claude_fno_training_closed_loop_task_command,
    build_fno_training_tool_catalog_command,
    run_claude_fno_training_closed_loop_command,
)
from pde_agent.commands.meta_research_directive import (
    build_agent_meta_research_directive_command,
)
from pde_agent.commands.research_directive import (
    build_agent_research_directive_command,
    evaluate_research_directive_followthrough_command,
)
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
from pde_agent.commands.hypothesis_lab_policy import (
    build_hypothesis_lab_next_policy_command,
)
from pde_agent.commands.plan_next_spectral_residual_experiment import (
    plan_next_spectral_residual_experiment_command,
)
from pde_agent.commands.lookup_strategy_skill import lookup_strategy_skill_command
from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.append_strategy_ledger import append_strategy_ledger_command
from pde_agent.commands.audit_experiment_config import audit_experiment_config
from pde_agent.commands.audit_competition_submission import (
    audit_competition_submission_command,
)
from pde_agent.commands.audit_task2_source_policy import (
    audit_task2_source_policy_command,
)
from pde_agent.commands.audit_local_strategy_loop import (
    audit_local_strategy_loop_command,
)
from pde_agent.commands.build_local_windows import build_local_windows
from pde_agent.commands.build_local_window_suite import build_local_window_suite
from pde_agent.commands.build_local_strategy_library import (
    build_local_strategy_library_command,
    lookup_local_strategy_command,
)
from pde_agent.commands.build_deeponet_evidence_sweep import (
    build_deeponet_evidence_sweep_command,
)
from pde_agent.commands.build_competition_submission import (
    build_competition_submission_command,
)
from pde_agent.commands.build_research_chronicle import (
    build_research_chronicle_command,
)
from pde_agent.commands.build_submission import build_submission
from pde_agent.commands.build_strategy_rubric import build_strategy_rubric_command
from pde_agent.commands.build_unet_evidence_sweep import (
    build_unet_evidence_sweep_command,
)
from pde_agent.commands.check_burgers_physics_local import check_burgers_physics_local
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.compare_fno_candidates import compare_fno_candidates_command
from pde_agent.commands.convert_fno_evidence_summary import (
    convert_fno_evidence_summary_command,
)
from pde_agent.commands.convert_operator_evidence_summary import (
    convert_operator_evidence_summary_command,
)
from pde_agent.commands.convert_official_comparison_evidence import (
    convert_official_comparison_evidence_command,
)
from pde_agent.commands.create_skill_candidate import create_skill_candidate_command
from pde_agent.commands.diagnose_fno_rollout import diagnose_fno_rollout
from pde_agent.commands.diagnose_unet_rollout import diagnose_unet_rollout
from pde_agent.commands.diagnose_rollout_error import diagnose_rollout_error
from pde_agent.commands.diagnose_task2_nu_shift import (
    diagnose_task2_nu_shift_command,
)
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.evaluate_burgers_residual import (
    evaluate_burgers_residual_command,
)
from pde_agent.commands.evaluate_strategy_promotion import (
    evaluate_strategy_promotion_command,
)
from pde_agent.commands.evaluate_deeponet_candidate import (
    evaluate_deeponet_candidate_command,
)
from pde_agent.commands.evaluate_fno_candidate import evaluate_fno_candidate_command
from pde_agent.commands.evaluate_unet_candidate import evaluate_unet_candidate_command
from pde_agent.commands.execute_deeponet_candidate import (
    execute_deeponet_candidate_command,
)
from pde_agent.commands.execute_deeponet_candidate_dryrun import (
    execute_deeponet_candidate_dryrun_command,
)
from pde_agent.commands.execute_fno_finetune_candidate import (
    execute_fno_finetune_candidate_command,
)
from pde_agent.commands.execute_unet_candidate import execute_unet_candidate_command
from pde_agent.commands.formalize_external_strategy_candidate import (
    formalize_external_strategy_candidate_command,
)
from pde_agent.commands.generate_baseline_prediction import generate_baseline_prediction
from pde_agent.commands.generate_experiment_config import generate_experiment_config
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
from pde_agent.commands.init_run import init_run
from pde_agent.commands.ingest_local_strategy_result import (
    ingest_local_strategy_result_command,
)
from pde_agent.commands.interactive_cockpit import (
    build_agent_dialogue_request_command,
    build_cockpit_snapshot_command,
    build_fno_training_loop_snapshot_command,
    cockpit_chat_command,
    dispatch_agent_dialogue_request_command,
    run_interactive_cockpit_command,
)
from pde_agent.commands.inspect_data import inspect_data
from pde_agent.commands.inspect_fno_checkpoint import inspect_fno_checkpoint
from pde_agent.commands.inspect_unet_checkpoint import inspect_unet_checkpoint
from pde_agent.commands.install_skill_draft import install_skill_draft_command
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.local_strategy_scheduler import (
    claim_next_scheduled_experiment_command,
    schedule_local_strategy_evaluation_command,
)
from pde_agent.commands.local_strategy_leaderboard import (
    record_local_strategy_result_command,
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
from pde_agent.commands.materialize_skill_draft import materialize_skill_draft_command
from pde_agent.commands.materialize_competition_candidate import (
    materialize_competition_candidate_command,
)
from pde_agent.commands.merge_competition_submissions import (
    merge_competition_submissions_command,
)
from pde_agent.commands.predict_fno_local import predict_fno_local
from pde_agent.commands.predict_fno_competition import (
    predict_fno_competition_command,
)
from pde_agent.commands.predict_task1_retrieval_prior import (
    predict_task1_retrieval_prior_command,
)
from pde_agent.commands.predict_unet_local import predict_unet_local
from pde_agent.commands.predict_unet_competition import (
    predict_unet_competition_command,
)
from pde_agent.commands.profile_unet_runtime import profile_unet_runtime_command
from pde_agent.commands.predict_local_model import predict_local_model
from pde_agent.commands.plan_next_experiment import plan_next_experiment
from pde_agent.commands.plan_next_experiment_from_fno_summary import (
    plan_next_experiment_from_fno_summary_command,
)
from pde_agent.commands.plan_spectral_residual_variance_intervention import (
    plan_spectral_residual_variance_intervention_command,
)
from pde_agent.commands.prepare_candidate_execution import (
    prepare_candidate_execution,
)
from pde_agent.commands.prepare_competition_windows import (
    prepare_competition_windows_command,
)
from pde_agent.commands.record_autonomy_decision import (
    record_autonomy_decision_command,
)
from pde_agent.commands.record_unet_timeout_evidence import (
    record_unet_timeout_evidence_command,
)
from pde_agent.commands.route_local_strategy_next_action import (
    route_local_strategy_next_action_command,
)
from pde_agent.commands.record_strategy_proposal import (
    record_strategy_proposal_command,
)
from pde_agent.commands.recommend_strategy_next_action import (
    recommend_strategy_next_action_command,
)
from pde_agent.commands.review_skill_candidate import review_skill_candidate_command
from pde_agent.commands.review_skill_draft import review_skill_draft_command
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.resolve_config import resolve_config
from pde_agent.commands.run_real_hdf5_baseline_smoke import (
    run_real_hdf5_baseline_smoke_command,
)
from pde_agent.commands.run_fno_iteration_dryrun import (
    run_fno_iteration_dryrun_command,
)
from pde_agent.commands.run_fno_iteration import run_fno_iteration_command
from pde_agent.commands.run_fno_evidence_sweep import (
    run_fno_evidence_sweep_command,
)
from pde_agent.commands.run_checkpoint_family_sandbox import (
    run_checkpoint_family_sandbox_command,
)
from pde_agent.commands.compare_hypothesis_lab_memory import (
    compare_hypothesis_lab_memory_command,
)
from pde_agent.commands.run_hypothesis_lab import run_hypothesis_lab_command
from pde_agent.commands.run_official_operator_comparison import (
    run_official_operator_comparison_command,
)
from pde_agent.commands.run_spectral_residual_evidence_sweep import (
    run_spectral_residual_evidence_sweep_command,
)
from pde_agent.commands.run_spectral_residual_convergence_grid import (
    run_spectral_residual_convergence_grid_command,
)
from pde_agent.commands.run_spectral_residual_window_stratification import (
    run_spectral_residual_window_stratification_command,
)
from pde_agent.commands.run_deeponet_strategy_sweep import (
    run_deeponet_strategy_sweep_command,
)
from pde_agent.commands.run_burgers_spectral_viscosity_benchmark import (
    run_burgers_spectral_viscosity_benchmark_command,
)
from pde_agent.commands.run_competition_task_pipeline import (
    run_competition_task_pipeline_command,
)
from pde_agent.commands.run_unet_strategy_sweep import run_unet_strategy_sweep_command
from pde_agent.commands.score_local_rollout import score_local_rollout
from pde_agent.commands.score_competition_submission import (
    score_competition_submission_command,
)
from pde_agent.commands.select_hypothesis import select_hypothesis
from pde_agent.commands.select_pde_hypothesis import select_pde_hypothesis
from pde_agent.commands.select_pde_hypothesis_from_fno_summary import (
    select_pde_hypothesis_from_fno_summary_command,
)
from pde_agent.commands.supervisor_tui import supervisor_tui_run_command
from pde_agent.commands.supervisor_tui import supervisor_tui_runs_command
from pde_agent.commands.train_baseline import train_baseline
from pde_agent.commands.train_spectral_residual_baseline import (
    train_spectral_residual_baseline,
)


app = typer.Typer(help="CLI-first diagnostic harness for PDE agent experiments.")


@app.command("supervisor-tui-runs")
def supervisor_tui_runs_cmd(
    runs_root: str = typer.Option("runs"),
    limit: int = typer.Option(20),
    wide: bool = typer.Option(False, "--wide"),
    watch: float = typer.Option(0.0, "--watch"),
    json_flag: bool = typer.Option(False, "--json"),
) -> None:
    while True:
        view = supervisor_tui_runs_command(runs_root=runs_root, limit=limit, wide=wide)
        if json_flag:
            typer.echo(json.dumps(view))
        else:
            typer.echo(view["text"])
        if watch <= 0:
            break
        time.sleep(watch)


@app.command("supervisor-tui-run")
def supervisor_tui_run_cmd(
    run_dir: str = typer.Option(...),
    json_flag: bool = typer.Option(False, "--json"),
) -> None:
    view = supervisor_tui_run_command(run_dir=run_dir)
    if json_flag:
        typer.echo(json.dumps(view))
    else:
        typer.echo(view["text"])


@app.command("cockpit")
def cockpit_cmd(
    runs_root: str = typer.Option("runs"),
    active_run_dir: str | None = typer.Option(None),
    dialogue_output_dir: str = typer.Option("runs/cockpit_dialogue"),
) -> None:
    run_interactive_cockpit_command(
        runs_root=runs_root,
        active_run_dir=active_run_dir,
        dialogue_output_dir=dialogue_output_dir,
    )


@app.command("cockpit-snapshot")
def cockpit_snapshot_cmd(
    runs_root: str = typer.Option("runs"),
    active_run_dir: str | None = typer.Option(None),
    limit: int = typer.Option(20),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_cockpit_snapshot_command(
                runs_root=runs_root,
                active_run_dir=active_run_dir,
                limit=limit,
            )
        )
    )


@app.command("cockpit-dialogue-request")
def cockpit_dialogue_request_cmd(
    user_message: str = typer.Option(...),
    runs_root: str = typer.Option("runs"),
    active_run_dir: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_agent_dialogue_request_command(
                user_message=user_message,
                runs_root=runs_root,
                active_run_dir=active_run_dir,
                output_dir=output_dir,
                force=force,
            )
        )
    )


@app.command("cockpit-dispatch-dialogue")
def cockpit_dispatch_dialogue_cmd(
    request_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    workspace_dir: str = typer.Option("."),
    dry_run: bool = typer.Option(True, "--dry-run/--execute"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            dispatch_agent_dialogue_request_command(
                request_path=request_path,
                output_path=output_path,
                workspace_dir=workspace_dir,
                dry_run=dry_run,
                force=force,
            )
        )
    )


@app.command("cockpit-chat")
def cockpit_chat_cmd(
    text: str = typer.Argument(...),
    runs_root: str = typer.Option("runs"),
    active_run_dir: str | None = typer.Option(None),
    dialogue_output_dir: str = typer.Option("runs/cockpit_dialogue"),
    workspace_dir: str = typer.Option("."),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    result = cockpit_chat_command(
        text=text,
        runs_root=runs_root,
        active_run_dir=active_run_dir,
        dialogue_output_dir=dialogue_output_dir,
        workspace_dir=workspace_dir,
        force=force,
    )
    if json_flag:
        typer.echo(json.dumps(result))
    else:
        typer.echo(result.get("text", ""))


@app.command("monitor-fno-training-loop")
def monitor_fno_training_loop_cmd(
    workspace_dir: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    result = build_fno_training_loop_snapshot_command(
        workspace_dir=workspace_dir,
    )
    if json_flag:
        typer.echo(json.dumps(result))
    else:
        typer.echo(result.get("panel_text", ""))


@app.command("init-run")
def init_run_cmd(
    run_id: str = typer.Option(...),
    runs_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(init_run(run_id=run_id, runs_root=runs_root)))



@app.command("lookup-strategy-skill")
def lookup_strategy_skill_cmd(
    registry_path: str = typer.Option(...),
    applicable_pde: str = typer.Option(...),
    model_family: str = typer.Option(...),
    output_path: str = typer.Option(...),
    max_results: int = typer.Option(3),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            lookup_strategy_skill_command(
                registry_path=registry_path,
                applicable_pde=applicable_pde,
                model_family=model_family,
                max_results=max_results,
                output_path=output_path,
            )
        )
    )


@app.command("build-local-strategy-library")
def build_local_strategy_library_cmd(
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_local_strategy_library_command(
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("lookup-local-strategy")
def lookup_local_strategy_cmd(
    library_path: str = typer.Option(...),
    applicable_pde: str = typer.Option(...),
    model_family: str = typer.Option(...),
    failure_modes: list[str] = typer.Option(..., "--failure-mode"),
    output_path: str = typer.Option(...),
    max_results: int = typer.Option(3),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            lookup_local_strategy_command(
                library_path=library_path,
                applicable_pde=applicable_pde,
                model_family=model_family,
                failure_modes=failure_modes,
                max_results=max_results,
                output_path=output_path,
            )
        )
    )


@app.command("schedule-local-strategy-evaluation")
def schedule_local_strategy_evaluation_cmd(
    lookup_report_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    workspace_root: str = typer.Option(...),
    max_experiments: int = typer.Option(3),
    max_train_seconds: float = typer.Option(3600.0),
    max_inference_seconds: float = typer.Option(120.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            schedule_local_strategy_evaluation_command(
                lookup_report_path=lookup_report_path,
                output_path=output_path,
                workspace_root=workspace_root,
                max_experiments=max_experiments,
                max_train_seconds=max_train_seconds,
                max_inference_seconds=max_inference_seconds,
                force=force,
            )
        )
    )


@app.command("claim-next-scheduled-experiment")
def claim_next_scheduled_experiment_cmd(
    schedule_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    completed_experiment_ids: list[str] | None = typer.Option(
        None,
        "--completed-experiment-id",
    ),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            claim_next_scheduled_experiment_command(
                schedule_path=schedule_path,
                completed_experiment_ids=completed_experiment_ids or [],
                output_path=output_path,
            )
        )
    )


@app.command("record-local-strategy-result")
def record_local_strategy_result_cmd(
    schedule_path: str = typer.Option(...),
    experiment_id: str = typer.Option(...),
    result_summary_path: str = typer.Option(...),
    leaderboard_path: str = typer.Option(...),
    result_status: str = typer.Option("completed"),
    min_score_delta: float = typer.Option(1.0),
    max_inference_seconds: float = typer.Option(120.0),
    failure_reason: str | None = typer.Option(None),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            record_local_strategy_result_command(
                schedule_path=schedule_path,
                experiment_id=experiment_id,
                result_summary_path=result_summary_path,
                leaderboard_path=leaderboard_path,
                result_status=result_status,
                min_score_delta=min_score_delta,
                max_inference_seconds=max_inference_seconds,
                failure_reason=failure_reason,
            )
        )
    )


@app.command("audit-local-strategy-loop")
def audit_local_strategy_loop_cmd(
    output_dir: str = typer.Option(...),
    report_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            audit_local_strategy_loop_command(
                output_dir=output_dir,
                report_path=report_path,
                force=force,
            )
        )
    )


@app.command("route-local-strategy-next-action")
def route_local_strategy_next_action_cmd(
    artifact_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            route_local_strategy_next_action_command(
                artifact_path=artifact_path,
                output_path=output_path,
            )
        )
    )


@app.command("ingest-local-strategy-result")
def ingest_local_strategy_result_cmd(
    source_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    baseline_score: float | None = typer.Option(None),
    result_status: str = typer.Option("completed"),
    failure_reason: str | None = typer.Option(None),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            ingest_local_strategy_result_command(
                source_path=source_path,
                output_path=output_path,
                baseline_score=baseline_score,
                result_status=result_status,
                failure_reason=failure_reason,
            )
        )
    )



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



@app.command("audit-agent-runtime-availability")
def audit_agent_runtime_availability_cmd(
    output_path: str = typer.Option(...),
    runtime_name: str = typer.Option("claude_agent_sdk"),
    tool_catalog_path: str = typer.Option(...),
    canary_report_paths: list[str] = typer.Option([], "--canary-report-path"),
    cli_commands: list[str] = typer.Option([], "--cli-command"),
    required_commands: list[str] = typer.Option([], "--required-command"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            audit_agent_runtime_availability_command(
                output_path=output_path,
                runtime_name=runtime_name,
                tool_catalog_path=tool_catalog_path,
                canary_report_paths=canary_report_paths,
                cli_commands=cli_commands,
                required_commands=required_commands,
                force=force,
            )
        )
    )


@app.command("build-agent-runtime-contract")
def build_agent_runtime_contract_cmd(
    output_path: str = typer.Option(...),
    runtime_name: str = typer.Option("claude_agent_sdk"),
    workspace_dir: str = typer.Option("."),
    tool_catalog_path: str = typer.Option(...),
    required_commands: list[str] = typer.Option([], "--required-command"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_agent_runtime_contract_command(
                output_path=output_path,
                runtime_name=runtime_name,
                workspace_dir=workspace_dir,
                tool_catalog_path=tool_catalog_path,
                required_commands=required_commands,
                force=force,
            )
        )
    )


@app.command("run-claude-agent-sdk-runtime")
def run_claude_agent_sdk_runtime_cmd(
    prompt: str | None = typer.Option(None),
    prompt_file: str | None = typer.Option(None),
    output_path: str = typer.Option(...),
    event_log_path: str | None = typer.Option(None),
    workspace_dir: str = typer.Option("."),
    allowed_tools: list[str] = typer.Option(
        ["Read", "Glob", "Grep", "Bash"],
        "--allowed-tool",
    ),
    disallowed_tools: list[str] = typer.Option(
        ["Edit", "Write", "NotebookEdit", "WebFetch"],
        "--disallowed-tool",
    ),
    allowed_bash_prefixes: list[str] = typer.Option(
        ["uv run --frozen pde-agent"],
        "--allowed-bash-prefix",
    ),
    allowed_pde_agent_commands: list[str] = typer.Option(
        [],
        "--allowed-pde-agent-command",
    ),
    timeout_seconds: int = typer.Option(3600),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_claude_agent_sdk_runtime_command(
                prompt=prompt,
                prompt_file=prompt_file,
                output_path=output_path,
                event_log_path=event_log_path,
                workspace_dir=workspace_dir,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                allowed_bash_prefixes=allowed_bash_prefixes,
                allowed_pde_agent_commands=allowed_pde_agent_commands,
                dry_run=dry_run,
                force=force,
                timeout_seconds=timeout_seconds,
            )
        )
    )


@app.command("run-claude-cli-runtime")
def run_claude_cli_runtime_cmd(
    prompt: str | None = typer.Option(None),
    prompt_file: str | None = typer.Option(None),
    output_path: str = typer.Option(...),
    workspace_dir: str = typer.Option("."),
    allowed_tools: list[str] = typer.Option(["Read", "Bash"], "--allowed-tool"),
    disallowed_tools: list[str] = typer.Option(["Edit", "Write"], "--disallowed-tool"),
    permission_mode: str = typer.Option("dontAsk"),
    output_format: str = typer.Option("stream-json"),
    timeout_seconds: int = typer.Option(3600),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_claude_cli_runtime_command(
                prompt=prompt,
                prompt_file=prompt_file,
                output_path=output_path,
                workspace_dir=workspace_dir,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                permission_mode=permission_mode,
                output_format=output_format,
                timeout_seconds=timeout_seconds,
                dry_run=dry_run,
                force=force,
            )
        )
    )


@app.command("export-competition-llm-log")
def export_competition_llm_log_cmd(
    session_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            export_competition_llm_log_command(
                session_path=session_path,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("build-claude-fno-training-smoke-task")
def build_claude_fno_training_smoke_task_cmd(
    output_dir: str = typer.Option(...),
    workspace_dir: str = typer.Option("."),
    runtime_contract_path: str = typer.Option(...),
    tool_catalog_path: str = typer.Option(...),
    source_path: str = typer.Option(...),
    previous_evidence_paths: list[str] = typer.Option([], "--previous-evidence-path"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_claude_fno_training_smoke_task_command(
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                runtime_contract_path=runtime_contract_path,
                tool_catalog_path=tool_catalog_path,
                source_path=source_path,
                previous_evidence_paths=previous_evidence_paths,
                force=force,
            )
        )
    )


@app.command("build-claude-promotion-smoke-task")
def build_claude_promotion_smoke_task_cmd(
    output_dir: str = typer.Option(...),
    workspace_dir: str = typer.Option("."),
    runtime_contract_path: str = typer.Option(...),
    tool_catalog_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_claude_promotion_smoke_task_command(
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                runtime_contract_path=runtime_contract_path,
                tool_catalog_path=tool_catalog_path,
                force=force,
            )
        )
    )


@app.command("build-claude-hypothesis-lab-closure-task")
def build_claude_hypothesis_lab_closure_task_cmd(
    output_dir: str = typer.Option(...),
    workspace_dir: str = typer.Option("."),
    runtime_contract_path: str = typer.Option(...),
    tool_catalog_path: str = typer.Option(...),
    candidate_family_sandbox_path: str = typer.Option(...),
    context_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    reference_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_claude_hypothesis_lab_closure_task_command(
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                runtime_contract_path=runtime_contract_path,
                tool_catalog_path=tool_catalog_path,
                candidate_family_sandbox_path=candidate_family_sandbox_path,
                context_path=context_path,
                target_path=target_path,
                reference_path=reference_path,
                force=force,
            )
        )
    )


@app.command("build-claude-literature-candidate-closure-task")
def build_claude_literature_candidate_closure_task_cmd(
    output_dir: str = typer.Option(...),
    workspace_dir: str = typer.Option("."),
    runtime_contract_path: str = typer.Option(...),
    tool_catalog_path: str = typer.Option(...),
    paper_parse_path: str = typer.Option(...),
    operator_family: str = typer.Option("fno"),
    target_problem: str = typer.Option("PDE_Burgers task1 early/middle dynamics"),
    registry_path: str = typer.Option(...),
    current_best_score: float = typer.Option(0.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_claude_literature_candidate_closure_task_command(
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                runtime_contract_path=runtime_contract_path,
                tool_catalog_path=tool_catalog_path,
                paper_parse_path=paper_parse_path,
                operator_family=operator_family,
                target_problem=target_problem,
                registry_path=registry_path,
                current_best_score=current_best_score,
                force=force,
            )
        )
    )


@app.command("record-agent-runtime-issue")
def record_agent_runtime_issue_cmd(
    output_path: str = typer.Option(...),
    failed_command: str = typer.Option(...),
    artifact_path: str | None = typer.Option(None),
    stderr_summary: str = typer.Option(...),
    likely_schema_gap: bool = typer.Option(False, "--likely-schema-gap/--not-schema-gap"),
    next_action: str = typer.Option("stop_after_issue_recorded"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            record_agent_runtime_issue_command(
                output_path=output_path,
                failed_command=failed_command,
                artifact_path=artifact_path,
                stderr_summary=stderr_summary,
                likely_schema_gap=likely_schema_gap,
                next_action=next_action,
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


@app.command("run-task1-short-window-training-loop")
def run_task1_short_window_training_loop_cmd(
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    train_sample_limit: int = typer.Option(64),
    val_sample_limit: int = typer.Option(32),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(47),
    time_reduction: int = typer.Option(1),
    space_reduction: int = typer.Option(4),
    model_family: str = typer.Option("conv1d_residual"),
    normalization: str = typer.Option("none"),
    train_steps: int = typer.Option(8),
    batch_size: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_task1_short_window_training_loop_command(
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                train_sample_limit=train_sample_limit,
                val_sample_limit=val_sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                model_family=model_family,
                normalization=normalization,
                train_steps=train_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                force=force,
            )
        )
    )


@app.command("run-task1-short-window-family-selection-loop")
def run_task1_short_window_family_selection_loop_cmd(
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    model_families: list[str] = typer.Option(
        ["conv1d_residual", "fno_lite"],
        "--model-family",
    ),
    train_sample_limit: int = typer.Option(64),
    val_sample_limit: int = typer.Option(32),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(47),
    time_reduction: int = typer.Option(1),
    space_reduction: int = typer.Option(4),
    train_steps: int = typer.Option(8),
    batch_size: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_task1_short_window_family_selection_loop_command(
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                model_families=model_families,
                train_sample_limit=train_sample_limit,
                val_sample_limit=val_sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                train_steps=train_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                force=force,
            )
        )
    )


@app.command("build-fno-training-harness-contract")
def build_fno_training_harness_contract_cmd(
    output_path: str = typer.Option(...),
    workspace_dir: str = typer.Option(...),
    source_path: str = typer.Option(...),
    baseline_prediction_path: str | None = typer.Option(None),
    task1_test_path: str | None = typer.Option(None),
    task1_target_path: str | None = typer.Option(None),
    previous_evidence_paths: list[str] = typer.Option([], "--previous-evidence-path"),
    dataset_key: str = typer.Option("tensor"),
    train_sample_limit: int = typer.Option(16),
    val_sample_limit: int = typer.Option(8),
    context_steps: int = typer.Option(10),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    batch_size: int = typer.Option(4),
    learning_rate: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    max_candidates: int = typer.Option(3),
    modes: list[int] = typer.Option([4, 8], "--mode"),
    widths: list[int] = typer.Option([16, 32], "--width"),
    depths: list[int] = typer.Option([1], "--depth"),
    loss_types: list[str] = typer.Option(
        ["mse", "early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        "--loss-type",
    ),
    normalizations: list[str] = typer.Option(["none"], "--normalization"),
    forecast_steps_values: list[int] = typer.Option(
        [12, 20],
        "--forecast-steps-value",
    ),
    train_steps_values: list[int] = typer.Option([1, 2], "--train-steps-value"),
    search_guidance: list[str] = typer.Option([], "--search-guidance"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_fno_training_harness_contract_command(
                output_path=output_path,
                workspace_dir=workspace_dir,
                source_path=source_path,
                baseline_prediction_path=baseline_prediction_path,
                task1_test_path=task1_test_path,
                task1_target_path=task1_target_path,
                previous_evidence_paths=previous_evidence_paths,
                dataset_key=dataset_key,
                train_sample_limit=train_sample_limit,
                val_sample_limit=val_sample_limit,
                context_steps=context_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                max_candidates=max_candidates,
                modes=modes,
                widths=widths,
                depths=depths,
                loss_types=loss_types,
                normalizations=normalizations,
                forecast_steps_values=forecast_steps_values,
                train_steps_values=train_steps_values,
                search_guidance=search_guidance,
                force=force,
            )
        )
    )




@app.command("plan-fno-training-candidates")
def plan_fno_training_candidates_cmd(
    evidence_paths: list[str] = typer.Option([], "--evidence-path"),
    output_path: str = typer.Option(...),
    max_candidates: int = typer.Option(4),
    modes: list[int] = typer.Option([4, 8], "--mode"),
    widths: list[int] = typer.Option([16, 32], "--width"),
    depths: list[int] = typer.Option([1], "--depth"),
    loss_types: list[str] = typer.Option(
        ["mse", "early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        "--loss-type",
    ),
    normalizations: list[str] = typer.Option(["none"], "--normalization"),
    forecast_steps_values: list[int] = typer.Option(
        [12, 20],
        "--forecast-steps-value",
    ),
    train_steps_values: list[int] = typer.Option([1, 2], "--train-steps-value"),
    search_guidance: list[str] = typer.Option([], "--search-guidance"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            plan_fno_training_candidates_command(
                evidence_paths=evidence_paths,
                output_path=output_path,
                max_candidates=max_candidates,
                modes=modes,
                widths=widths,
                depths=depths,
                loss_types=loss_types,
                normalizations=normalizations,
                forecast_steps_values=forecast_steps_values,
                train_steps_values=train_steps_values,
                search_guidance=search_guidance,
                force=force,
            )
        )
    )


@app.command("run-fno-training-optimization-loop")
def run_fno_training_optimization_loop_cmd(
    candidate_plan_path: str = typer.Option(...),
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    train_sample_limit: int = typer.Option(64),
    val_sample_limit: int = typer.Option(32),
    context_steps: int = typer.Option(10),
    time_reduction: int = typer.Option(1),
    space_reduction: int = typer.Option(4),
    batch_size: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_fno_training_optimization_loop_command(
                candidate_plan_path=candidate_plan_path,
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                train_sample_limit=train_sample_limit,
                val_sample_limit=val_sample_limit,
                context_steps=context_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                force=force,
            )
        )
    )


@app.command("run-task1-fno-graft-evaluator")
def run_task1_fno_graft_evaluator_cmd(
    baseline_prediction_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    test_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    target_dataset_key: str = typer.Option("tensor"),
    windows: list[str] = typer.Option([], "--window"),
    blend_weights: list[float] = typer.Option([], "--blend-weight"),
    late_non_regression_tolerance: float = typer.Option(0.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_task1_fno_graft_evaluator_command(
                baseline_prediction_path=baseline_prediction_path,
                checkpoint_path=checkpoint_path,
                test_path=test_path,
                target_path=target_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                target_dataset_key=target_dataset_key,
                windows=windows,
                blend_weights=blend_weights,
                late_non_regression_tolerance=late_non_regression_tolerance,
                force=force,
            )
        )
    )


@app.command("run-task1-fno-graft-from-optimization-report")
def run_task1_fno_graft_from_optimization_report_cmd(
    optimization_report_path: str = typer.Option(...),
    baseline_prediction_path: str = typer.Option(...),
    test_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    target_dataset_key: str = typer.Option("tensor"),
    windows: list[str] = typer.Option([], "--window"),
    blend_weights: list[float] = typer.Option([], "--blend-weight"),
    late_non_regression_tolerance: float = typer.Option(0.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_task1_fno_graft_from_optimization_report_command(
                optimization_report_path=optimization_report_path,
                baseline_prediction_path=baseline_prediction_path,
                test_path=test_path,
                target_path=target_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                target_dataset_key=target_dataset_key,
                windows=windows,
                blend_weights=blend_weights,
                late_non_regression_tolerance=late_non_regression_tolerance,
                force=force,
            )
        )
    )


@app.command("record-fno-training-optimization-evidence")
def record_fno_training_optimization_evidence_cmd(
    optimization_report_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    graft_report_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            record_fno_training_optimization_evidence_command(
                optimization_report_path=optimization_report_path,
                output_path=output_path,
                graft_report_path=graft_report_path,
                force=force,
            )
        )
    )


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



@app.command("build-agent-research-directive")
def build_agent_research_directive_cmd(
    task1_benchmark_path: str = typer.Option(...),
    candidate_factory_benchmark_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    recent_judge_report_paths: list[str] = typer.Option([], "--recent-judge-report-path"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_agent_research_directive_command(
                task1_benchmark_path=task1_benchmark_path,
                candidate_factory_benchmark_path=candidate_factory_benchmark_path,
                output_path=output_path,
                recent_judge_report_paths=recent_judge_report_paths,
                force=force,
            )
        )
    )


@app.command("evaluate-research-directive-followthrough")
def evaluate_research_directive_followthrough_cmd(
    directive_path: str = typer.Option(...),
    chain_report_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            evaluate_research_directive_followthrough_command(
                directive_path=directive_path,
                chain_report_path=chain_report_path,
                output_path=output_path,
                force=force,
            )
        )
    )



@app.command("build-agent-meta-research-directive")
def build_agent_meta_research_directive_cmd(
    followthrough_report_path: str = typer.Option(...),
    chain_report_path: str = typer.Option(...),
    task1_benchmark_path: str = typer.Option(...),
    candidate_factory_benchmark_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_agent_meta_research_directive_command(
                followthrough_report_path=followthrough_report_path,
                chain_report_path=chain_report_path,
                task1_benchmark_path=task1_benchmark_path,
                candidate_factory_benchmark_path=candidate_factory_benchmark_path,
                output_path=output_path,
                force=force,
            )
        )
    )



@app.command("compare-hypothesis-lab-memory")
def compare_hypothesis_lab_memory_cmd(
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
    context_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    submission_id: str = typer.Option("team_default"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            compare_hypothesis_lab_memory_command(
                manual_report_paths=manual_report_paths,
                adaptive_report_paths=adaptive_report_paths,
                log_paths=log_paths,
                checkpoint_paths=checkpoint_paths,
                literature_paths=literature_paths,
                algorithm_candidate_paths=algorithm_candidate_paths,
                failure_memory_paths=failure_memory_paths,
                context_path=context_path,
                target_path=target_path,
                output_dir=output_dir,
                submission_id=submission_id,
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



@app.command("diagnose-task2-nu-shift")
def diagnose_task2_nu_shift_cmd(
    target_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    reference_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    nu_key: str = typer.Option("nu"),
    context_steps: int = typer.Option(10),
    sample_limit: int | None = typer.Option(None),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            diagnose_task2_nu_shift_command(
                target_path=target_path,
                output_path=output_path,
                reference_path=reference_path,
                dataset_key=dataset_key,
                nu_key=nu_key,
                context_steps=context_steps,
                sample_limit=sample_limit,
            )
        )
    )


@app.command("record-unet-timeout-evidence")
def record_unet_timeout_evidence_cmd(
    output_path: str = typer.Option(...),
    elapsed_seconds: float = typer.Option(...),
    timeout_seconds: float = typer.Option(120.0),
    attempted_command: str = typer.Option("predict-unet-competition"),
    checkpoint_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            record_unet_timeout_evidence_command(
                output_path=output_path,
                elapsed_seconds=elapsed_seconds,
                timeout_seconds=timeout_seconds,
                attempted_command=attempted_command,
                checkpoint_path=checkpoint_path,
            )
        )
    )



@app.command("resolve-config")
def resolve_config_cmd(
    run_id: str = typer.Option(...),
    base_config_file: str = typer.Option(...),
    override_file: str = typer.Option(...),
    task_spec_file: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            resolve_config(
                run_id=run_id,
                base_config_file=base_config_file,
                override_file=override_file,
                task_spec_file=task_spec_file,
                run_root=run_root,
            )
        )
    )


@app.command("build-local-windows")
def build_local_windows_cmd(
    source_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    start_index: int = typer.Option(0),
    sample_limit: int | None = typer.Option(None),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(31),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_local_windows(
                source_path=source_path,
                output_path=output_path,
                dataset_key=dataset_key,
                start_index=start_index,
                sample_limit=sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                force=force,
            )
        )
    )


@app.command("build-local-window-suite")
def build_local_window_suite_cmd(
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    start_index: int = typer.Option(0),
    sample_limit: int | None = typer.Option(None),
    context_steps: int = typer.Option(10),
    forecast_steps: list[int] = typer.Option(..., "--forecast-steps"),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_local_window_suite(
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                start_index=start_index,
                sample_limit=sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                force=force,
            )
        )
    )


@app.command("build-deeponet-evidence-sweep")
def build_deeponet_evidence_sweep_cmd(
    evaluation_report_paths: list[str] = typer.Option(
        ...,
        "--evaluation-report-path",
    ),
    output_dir: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_deeponet_evidence_sweep_command(
                evaluation_report_paths=evaluation_report_paths,
                output_dir=output_dir,
                force=force,
            )
        )
    )


@app.command("build-unet-evidence-sweep")
def build_unet_evidence_sweep_cmd(
    evaluation_report_paths: list[str] = typer.Option(
        ...,
        "--evaluation-report-path",
    ),
    output_dir: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_unet_evidence_sweep_command(
                evaluation_report_paths=evaluation_report_paths,
                output_dir=output_dir,
                force=force,
            )
        )
    )


@app.command("score-local-rollout")
def score_local_rollout_cmd(
    prediction_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            score_local_rollout(
                prediction_path=prediction_path,
                target_path=target_path,
                output_path=output_path,
            )
        )
    )


@app.command("generate-baseline-prediction")
def generate_baseline_prediction_cmd(
    windows_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    method: str = typer.Option("persistence"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            generate_baseline_prediction(
                windows_path=windows_path,
                output_path=output_path,
                method=method,
                force=force,
            )
        )
    )


@app.command("train-spectral-residual-baseline")
def train_spectral_residual_baseline_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    modes: int = typer.Option(16),
    ridge: float = typer.Option(1e-3),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            train_spectral_residual_baseline(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                modes=modes,
                ridge=ridge,
                force=force,
            )
        )
    )


@app.command("predict-local-model")
def predict_local_model_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            predict_local_model(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("inspect-fno-checkpoint")
def inspect_fno_checkpoint_cmd(
    checkpoint_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            inspect_fno_checkpoint(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
            )
        )
    )


@app.command("predict-fno-local")
def predict_fno_local_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    metadata_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            predict_fno_local(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                metadata_path=metadata_path,
                force=force,
            )
        )
    )


@app.command("inspect-unet-checkpoint")
def inspect_unet_checkpoint_cmd(
    checkpoint_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            inspect_unet_checkpoint(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
            )
        )
    )


@app.command("predict-unet-local")
def predict_unet_local_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    metadata_path: str | None = typer.Option(None),
    inference_batch_size: int | None = typer.Option(None),
    torch_num_threads: int | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            predict_unet_local(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                metadata_path=metadata_path,
                inference_batch_size=inference_batch_size,
                torch_num_threads=torch_num_threads,
                force=force,
            )
        )
    )


@app.command("diagnose-fno-rollout")
def diagnose_fno_rollout_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            diagnose_fno_rollout(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("diagnose-unet-rollout")
def diagnose_unet_rollout_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            diagnose_unet_rollout(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-fno-iteration-dryrun")
def run_fno_iteration_dryrun_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_fno_iteration_dryrun_command(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-fno-iteration")
def run_fno_iteration_cmd(
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    max_train_steps: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-4),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_fno_iteration_command(
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                max_train_steps=max_train_steps,
                learning_rate=learning_rate,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-fno-evidence-sweep")
def run_fno_evidence_sweep_cmd(
    source_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    start_indices: list[int] = typer.Option(..., "--start-index"),
    sample_limit: int = typer.Option(...),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(...),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    train_steps: list[int] = typer.Option(..., "--train-steps"),
    learning_rate: float = typer.Option(1e-4),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_fno_evidence_sweep_command(
                source_path=source_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                start_indices=start_indices,
                sample_limit=sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                train_steps=train_steps,
                learning_rate=learning_rate,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-official-operator-comparison")
def run_official_operator_comparison_cmd(
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    fno_checkpoint_path: str = typer.Option(...),
    unet_checkpoint_path: str = typer.Option(...),
    sample_limit: int = typer.Option(8),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(31),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    spectral_modes: int = typer.Option(16),
    spectral_ridge: float = typer.Option(1e-3),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_official_operator_comparison_command(
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                fno_checkpoint_path=fno_checkpoint_path,
                unet_checkpoint_path=unet_checkpoint_path,
                sample_limit=sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                spectral_modes=spectral_modes,
                spectral_ridge=spectral_ridge,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-spectral-residual-evidence-sweep")
def run_spectral_residual_evidence_sweep_cmd(
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    start_indices: list[int] = typer.Option(..., "--start-index"),
    sample_limit: int = typer.Option(8),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(31),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    spectral_modes: int = typer.Option(16),
    spectral_ridge: float = typer.Option(1e-3),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_spectral_residual_evidence_sweep_command(
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                start_indices=start_indices,
                sample_limit=sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                spectral_modes=spectral_modes,
                spectral_ridge=spectral_ridge,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-spectral-residual-convergence-grid")
def run_spectral_residual_convergence_grid_cmd(
    audit_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    source_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    sample_limit: int = typer.Option(8),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(31),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_spectral_residual_convergence_grid_command(
                audit_path=audit_path,
                output_dir=output_dir,
                source_path=source_path,
                dataset_key=dataset_key,
                sample_limit=sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-spectral-residual-window-stratification")
def run_spectral_residual_window_stratification_cmd(
    audit_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_spectral_residual_window_stratification_command(
                audit_path=audit_path,
                output_dir=output_dir,
                force=force,
            )
        )
    )


@app.command("run-deeponet-strategy-sweep")
def run_deeponet_strategy_sweep_cmd(
    windows_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    baseline_score_path: str = typer.Option(...),
    start_indices: list[int] = typer.Option(..., "--start-index"),
    training_strategies: list[str] = typer.Option(
        ["branch_trunk_supervised"],
        "--training-strategy",
    ),
    train_steps: list[int] = typer.Option(..., "--train-steps"),
    hidden_widths: list[int] = typer.Option(..., "--hidden-width"),
    latent_dims: list[int] = typer.Option(..., "--latent-dim"),
    learning_rates: list[float] = typer.Option(..., "--learning-rate"),
    seeds: list[int] = typer.Option(..., "--seed"),
    spectral_weights: list[float] = typer.Option([0.1], "--spectral-weight"),
    physics_weights: list[float] = typer.Option([0.01], "--physics-weight"),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_deeponet_strategy_sweep_command(
                windows_path=windows_path,
                output_dir=output_dir,
                baseline_score_path=baseline_score_path,
                start_indices=start_indices,
                training_strategies=training_strategies,
                train_steps=train_steps,
                hidden_widths=hidden_widths,
                latent_dims=latent_dims,
                learning_rates=learning_rates,
                seeds=seeds,
                spectral_weights=spectral_weights,
                physics_weights=physics_weights,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-unet-strategy-sweep")
def run_unet_strategy_sweep_cmd(
    windows_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    baseline_score_path: str = typer.Option(...),
    start_indices: list[int] = typer.Option(..., "--start-index"),
    training_strategies: list[str] = typer.Option(
        ["unet_supervised"],
        "--training-strategy",
    ),
    train_steps: list[int] = typer.Option(..., "--train-steps"),
    base_channels: list[int] = typer.Option(..., "--base-channels"),
    depths: list[int] = typer.Option(..., "--depth"),
    learning_rates: list[float] = typer.Option(..., "--learning-rate"),
    seeds: list[int] = typer.Option(..., "--seed"),
    spectral_weights: list[float] = typer.Option([0.1], "--spectral-weight"),
    physics_weights: list[float] = typer.Option([0.01], "--physics-weight"),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_unet_strategy_sweep_command(
                windows_path=windows_path,
                output_dir=output_dir,
                baseline_score_path=baseline_score_path,
                start_indices=start_indices,
                training_strategies=training_strategies,
                train_steps=train_steps,
                base_channels=base_channels,
                depths=depths,
                learning_rates=learning_rates,
                seeds=seeds,
                spectral_weights=spectral_weights,
                physics_weights=physics_weights,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("run-burgers-spectral-viscosity-benchmark")
def run_burgers_spectral_viscosity_benchmark_cmd(
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    sample_limit: int = typer.Option(8),
    min_evaluation_windows: int = typer.Option(4),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(31),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    train_steps: list[int] = typer.Option([4], "--train-steps"),
    hidden_widths: list[int] = typer.Option([8], "--hidden-width"),
    latent_dims: list[int] = typer.Option([4], "--latent-dim"),
    unet_base_channels: list[int] = typer.Option([8], "--unet-base-channels"),
    unet_depths: list[int] = typer.Option([2], "--unet-depth"),
    learning_rates: list[float] = typer.Option([0.01], "--learning-rate"),
    seeds: list[int] = typer.Option([5], "--seed"),
    spectral_weights: list[float] = typer.Option([0.05, 0.1], "--spectral-weight"),
    physics_weights: list[float] = typer.Option([0.005], "--physics-weight"),
    proposal_id: str = typer.Option("burgers_spectral_viscosity_real_hdf5_v0"),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_burgers_spectral_viscosity_benchmark_command(
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                sample_limit=sample_limit,
                min_evaluation_windows=min_evaluation_windows,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                train_steps=train_steps,
                hidden_widths=hidden_widths,
                latent_dims=latent_dims,
                unet_base_channels=unet_base_channels,
                unet_depths=unet_depths,
                learning_rates=learning_rates,
                seeds=seeds,
                spectral_weights=spectral_weights,
                physics_weights=physics_weights,
                proposal_id=proposal_id,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("convert-fno-evidence-summary")
def convert_fno_evidence_summary_cmd(
    sweep_summary_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    long_window_threshold: int = typer.Option(31),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            convert_fno_evidence_summary_command(
                sweep_summary_path=sweep_summary_path,
                output_path=output_path,
                long_window_threshold=long_window_threshold,
                force=force,
            )
        )
    )


@app.command("convert-operator-evidence-summary")
def convert_operator_evidence_summary_cmd(
    sweep_summary_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    model_family: str = typer.Option(...),
    source_state_hint: str = typer.Option(...),
    long_window_threshold: int = typer.Option(31),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            convert_operator_evidence_summary_command(
                sweep_summary_path=sweep_summary_path,
                output_path=output_path,
                model_family=model_family,
                source_state_hint=source_state_hint,
                long_window_threshold=long_window_threshold,
                force=force,
            )
        )
    )


@app.command("convert-official-comparison-evidence")
def convert_official_comparison_evidence_cmd(
    comparison_report_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    long_window_threshold: int = typer.Option(31),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            convert_official_comparison_evidence_command(
                comparison_report_path=comparison_report_path,
                output_path=output_path,
                long_window_threshold=long_window_threshold,
                force=force,
            )
        )
    )


@app.command("diagnose-rollout-error")
def diagnose_rollout_error_cmd(
    prediction_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    score_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            diagnose_rollout_error(
                prediction_path=prediction_path,
                target_path=target_path,
                score_path=score_path,
                output_path=output_path,
            )
        )
    )


@app.command("check-burgers-physics-local")
def check_burgers_physics_local_cmd(
    prediction_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            check_burgers_physics_local(
                prediction_path=prediction_path,
                target_path=target_path,
                output_path=output_path,
                nu=nu,
                dt=dt,
                dx=dx,
            )
        )
    )


@app.command("evaluate-burgers-residual")
def evaluate_burgers_residual_cmd(
    trajectory_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    array_key: str = typer.Option("trajectory"),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            evaluate_burgers_residual_command(
                trajectory_path=trajectory_path,
                output_path=output_path,
                array_key=array_key,
                nu=nu,
                dt=dt,
                dx=dx,
            )
        )
    )


@app.command("run-real-hdf5-baseline-smoke")
def run_real_hdf5_baseline_smoke_cmd(
    source_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    start_index: int = typer.Option(0),
    sample_limit: int = typer.Option(1),
    context_steps: int = typer.Option(10),
    forecast_steps: int = typer.Option(31),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_real_hdf5_baseline_smoke_command(
                source_path=source_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                start_index=start_index,
                sample_limit=sample_limit,
                context_steps=context_steps,
                forecast_steps=forecast_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


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


@app.command("evaluate-strategy-promotion")
def evaluate_strategy_promotion_cmd(
    proposal_path: str = typer.Option(...),
    rubric_path: str = typer.Option(...),
    evidence_summary_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            evaluate_strategy_promotion_command(
                proposal_path=proposal_path,
                rubric_path=rubric_path,
                evidence_summary_path=evidence_summary_path,
                output_path=output_path,
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


@app.command("eval-rollout")
def eval_rollout_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(eval_rollout(run_id=run_id, run_root=run_root)))


@app.command("analyze-spectrum")
def analyze_spectrum_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(analyze_spectrum(run_id=run_id, run_root=run_root)))


@app.command("check-physics")
def check_physics_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(check_physics(run_id=run_id, run_root=run_root)))


@app.command("inspect-data")
def inspect_data_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(inspect_data(run_id=run_id, run_root=run_root)))


@app.command("interpret-error")
def interpret_error_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(interpret_error(run_id=run_id, run_root=run_root)))


@app.command("select-hypothesis")
def select_hypothesis_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(select_hypothesis(run_id=run_id, run_root=run_root)))


@app.command("select-pde-hypothesis")
def select_pde_hypothesis_cmd(
    score_path: str = typer.Option(...),
    error_diagnosis_path: str = typer.Option(...),
    physics_check_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            select_pde_hypothesis(
                score_path=score_path,
                error_diagnosis_path=error_diagnosis_path,
                physics_check_path=physics_check_path,
                output_path=output_path,
            )
        )
    )


@app.command("select-pde-hypothesis-from-fno-summary")
def select_pde_hypothesis_from_fno_summary_cmd(
    summary_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            select_pde_hypothesis_from_fno_summary_command(
                summary_path=summary_path,
                output_path=output_path,
            )
        )
    )


@app.command("plan-next-experiment")
def plan_next_experiment_cmd(
    score_path: str = typer.Option(...),
    error_diagnosis_path: str = typer.Option(...),
    physics_check_path: str = typer.Option(...),
    hypothesis_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            plan_next_experiment(
                score_path=score_path,
                error_diagnosis_path=error_diagnosis_path,
                physics_check_path=physics_check_path,
                hypothesis_path=hypothesis_path,
                output_path=output_path,
            )
        )
    )


@app.command("plan-next-experiment-from-fno-summary")
def plan_next_experiment_from_fno_summary_cmd(
    summary_path: str = typer.Option(...),
    hypothesis_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            plan_next_experiment_from_fno_summary_command(
                summary_path=summary_path,
                hypothesis_path=hypothesis_path,
                output_path=output_path,
            )
        )
    )


@app.command("generate-experiment-config")
def generate_experiment_config_cmd(
    plan_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            generate_experiment_config(
                plan_path=plan_path,
                output_path=output_path,
            )
        )
    )


@app.command("audit-experiment-config")
def audit_experiment_config_cmd(
    config_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    plan_path: str | None = typer.Option(None),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            audit_experiment_config(
                config_path=config_path,
                output_path=output_path,
                plan_path=plan_path,
            )
        )
    )


@app.command("plan-next-spectral-residual-experiment")
def plan_next_spectral_residual_experiment_cmd(
    evidence_summary_path: str = typer.Option(...),
    promotion_report_path: str = typer.Option(...),
    ledger_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            plan_next_spectral_residual_experiment_command(
                evidence_summary_path=evidence_summary_path,
                promotion_report_path=promotion_report_path,
                ledger_path=ledger_path,
                output_path=output_path,
            )
        )
    )


@app.command("plan-spectral-residual-variance-intervention")
def plan_spectral_residual_variance_intervention_cmd(
    grid_summary_path: str = typer.Option(...),
    promotion_report_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            plan_spectral_residual_variance_intervention_command(
                grid_summary_path=grid_summary_path,
                promotion_report_path=promotion_report_path,
                output_path=output_path,
            )
        )
    )


@app.command("prepare-candidate-execution")
def prepare_candidate_execution_cmd(
    audit_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    candidate_index: int = typer.Option(0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            prepare_candidate_execution(
                audit_path=audit_path,
                output_dir=output_dir,
                candidate_index=candidate_index,
                force=force,
            )
        )
    )


@app.command("execute-fno-finetune-candidate")
def execute_fno_finetune_candidate_cmd(
    candidate_plan_path: str = typer.Option(...),
    windows_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    max_train_steps: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-4),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            execute_fno_finetune_candidate_command(
                candidate_plan_path=candidate_plan_path,
                windows_path=windows_path,
                checkpoint_path=checkpoint_path,
                max_train_steps=max_train_steps,
                learning_rate=learning_rate,
                force=force,
            )
        )
    )


@app.command("execute-deeponet-candidate-dryrun")
def execute_deeponet_candidate_dryrun_cmd(
    windows_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    start_index: int = typer.Option(0),
    train_steps: int = typer.Option(1),
    training_strategy: str = typer.Option("branch_trunk_rollout"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            execute_deeponet_candidate_dryrun_command(
                windows_path=windows_path,
                output_dir=output_dir,
                start_index=start_index,
                train_steps=train_steps,
                training_strategy=training_strategy,
                force=force,
            )
        )
    )


@app.command("execute-deeponet-candidate")
def execute_deeponet_candidate_cmd(
    windows_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    start_index: int = typer.Option(0),
    train_steps: int = typer.Option(8),
    training_strategy: str = typer.Option("branch_trunk_supervised"),
    hidden_width: int = typer.Option(32),
    latent_dim: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    strategy_label: str | None = typer.Option(None),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    spectral_weight: float = typer.Option(0.1),
    physics_weight: float = typer.Option(0.01),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            execute_deeponet_candidate_command(
                windows_path=windows_path,
                output_dir=output_dir,
                start_index=start_index,
                train_steps=train_steps,
                training_strategy=training_strategy,
                hidden_width=hidden_width,
                latent_dim=latent_dim,
                learning_rate=learning_rate,
                seed=seed,
                strategy_label=strategy_label,
                nu=nu,
                dt=dt,
                dx=dx,
                spectral_weight=spectral_weight,
                physics_weight=physics_weight,
                force=force,
            )
        )
    )


@app.command("execute-unet-candidate")
def execute_unet_candidate_cmd(
    windows_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    start_index: int = typer.Option(0),
    train_steps: int = typer.Option(8),
    training_strategy: str = typer.Option("unet_supervised"),
    base_channels: int = typer.Option(16),
    depth: int = typer.Option(2),
    learning_rate: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    strategy_label: str | None = typer.Option(None),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    spectral_weight: float = typer.Option(0.1),
    physics_weight: float = typer.Option(0.01),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            execute_unet_candidate_command(
                windows_path=windows_path,
                output_dir=output_dir,
                start_index=start_index,
                train_steps=train_steps,
                training_strategy=training_strategy,
                base_channels=base_channels,
                depth=depth,
                learning_rate=learning_rate,
                seed=seed,
                strategy_label=strategy_label,
                nu=nu,
                dt=dt,
                dx=dx,
                spectral_weight=spectral_weight,
                physics_weight=physics_weight,
                force=force,
            )
        )
    )


@app.command("evaluate-fno-candidate")
def evaluate_fno_candidate_cmd(
    candidate_report_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    baseline_score_path: str | None = typer.Option(None),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            evaluate_fno_candidate_command(
                candidate_report_path=candidate_report_path,
                target_path=target_path,
                output_dir=output_dir,
                baseline_score_path=baseline_score_path,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("evaluate-deeponet-candidate")
def evaluate_deeponet_candidate_cmd(
    candidate_report_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    baseline_score_path: str | None = typer.Option(None),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            evaluate_deeponet_candidate_command(
                candidate_report_path=candidate_report_path,
                target_path=target_path,
                output_dir=output_dir,
                baseline_score_path=baseline_score_path,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("evaluate-unet-candidate")
def evaluate_unet_candidate_cmd(
    candidate_report_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    baseline_score_path: str | None = typer.Option(None),
    nu: float = typer.Option(0.001),
    dt: float = typer.Option(1.0),
    dx: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            evaluate_unet_candidate_command(
                candidate_report_path=candidate_report_path,
                target_path=target_path,
                output_dir=output_dir,
                baseline_score_path=baseline_score_path,
                nu=nu,
                dt=dt,
                dx=dx,
                force=force,
            )
        )
    )


@app.command("compare-fno-candidates")
def compare_fno_candidates_cmd(
    evaluation_report_paths: list[str] = typer.Option(
        ...,
        "--evaluation-report-path",
    ),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            compare_fno_candidates_command(
                evaluation_report_paths=evaluation_report_paths,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("record-timing")
def record_timing_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(record_timing(run_id=run_id, run_root=run_root)))


@app.command("collect-artifacts")
def collect_artifacts_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    config_hash: str = typer.Option("unknown"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            collect_artifacts(
                run_id=run_id,
                run_root=run_root,
                config_hash=config_hash,
            )
        )
    )


@app.command("train-baseline")
def train_baseline_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(train_baseline(run_id=run_id, run_root=run_root)))


@app.command("build-submission")
def build_submission_cmd(
    run_id: str = typer.Option(...),
    run_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(build_submission(run_id=run_id, run_root=run_root)))


@app.command("build-competition-submission")
def build_competition_submission_cmd(
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    submission_id: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    method: str = typer.Option("persistence"),
    source_prediction_path: str | None = typer.Option(None),
    source_prediction_key: str = typer.Option("prediction"),
    train_time: float = typer.Option(0.0),
    inference_time: float | None = typer.Option(None),
    timing_profile: str | None = typer.Option(None),
    timing_factor: float | None = typer.Option(None),
    local_loop_chronicle_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_competition_submission_command(
                task_id=task_id,
                test_path=test_path,
                output_dir=output_dir,
                submission_id=submission_id,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                method=method,
                source_prediction_path=source_prediction_path,
                source_prediction_key=source_prediction_key,
                train_time=train_time,
                inference_time=inference_time,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                local_loop_chronicle_path=local_loop_chronicle_path,
                force=force,
            )
        )
    )


@app.command("score-competition-submission")
def score_competition_submission_cmd(
    task_id: int = typer.Option(...),
    prediction_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    prediction_dataset_key: str = typer.Option("tensor"),
    target_dataset_key: str = typer.Option("tensor"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            score_competition_submission_command(
                task_id=task_id,
                prediction_path=prediction_path,
                target_path=target_path,
                output_path=output_path,
                prediction_dataset_key=prediction_dataset_key,
                target_dataset_key=target_dataset_key,
            )
        )
    )


@app.command("materialize-competition-candidate")
def materialize_competition_candidate_cmd(
    task_id: int = typer.Option(...),
    candidate_report_path: str = typer.Option(...),
    test_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    submission_id: str = typer.Option(...),
    target_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    source_prediction_key: str = typer.Option("prediction"),
    train_time: float | None = typer.Option(None),
    timing_profile: str | None = typer.Option(None),
    timing_factor: float | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            materialize_competition_candidate_command(
                task_id=task_id,
                candidate_report_path=candidate_report_path,
                test_path=test_path,
                output_dir=output_dir,
                submission_id=submission_id,
                target_path=target_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                source_prediction_key=source_prediction_key,
                train_time=train_time,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                force=force,
            )
        )
    )


@app.command("merge-competition-submissions")
def merge_competition_submissions_cmd(
    output_dir: str = typer.Option(...),
    submission_id: str = typer.Option(...),
    task_submission_dirs: list[str] = typer.Option(..., "--task-submission-dir"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            merge_competition_submissions_command(
                output_dir=output_dir,
                submission_id=submission_id,
                task_submission_dirs=task_submission_dirs,
                force=force,
            )
        )
    )


@app.command("audit-competition-submission")
def audit_competition_submission_cmd(
    task_id: int = typer.Option(...),
    submission_dir: str = typer.Option(...),
    test_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            audit_competition_submission_command(
                task_id=task_id,
                submission_dir=submission_dir,
                test_path=test_path,
                output_path=output_path,
                dataset_key=dataset_key,
            )
        )
    )


@app.command("audit-task2-source-policy")
def audit_task2_source_policy_cmd(
    train_data_paths: list[str] = typer.Option([], "--train-data-path"),
    test_data_paths: list[str] = typer.Option([], "--test-data-path"),
    checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
    candidate_source_paths: list[str] = typer.Option([], "--candidate-source-path"),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            audit_task2_source_policy_command(
                train_data_paths=train_data_paths,
                test_data_paths=test_data_paths,
                checkpoint_paths=checkpoint_paths,
                candidate_source_paths=candidate_source_paths,
                output_path=output_path,
            )
        )
    )


@app.command("prepare-competition-windows")
def prepare_competition_windows_cmd(
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            prepare_competition_windows_command(
                task_id=task_id,
                test_path=test_path,
                output_path=output_path,
                dataset_key=dataset_key,
                force=force,
            )
        )
    )


@app.command("predict-unet-competition")
def predict_unet_competition_cmd(
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    inference_batch_size: int | None = typer.Option(None),
    torch_num_threads: int | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            predict_unet_competition_command(
                task_id=task_id,
                test_path=test_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                inference_batch_size=inference_batch_size,
                torch_num_threads=torch_num_threads,
                force=force,
            )
        )
    )


@app.command("profile-unet-runtime")
def profile_unet_runtime_cmd(
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    shard_sizes: list[int] = typer.Option(..., "--shard-size"),
    dataset_key: str = typer.Option("tensor"),
    timeout_seconds: float = typer.Option(120.0),
    full_sample_count: int = typer.Option(1000),
    inference_batch_size: int | None = typer.Option(None),
    torch_num_threads: int | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            profile_unet_runtime_command(
                task_id=task_id,
                test_path=test_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                shard_sizes=shard_sizes,
                dataset_key=dataset_key,
                timeout_seconds=timeout_seconds,
                full_sample_count=full_sample_count,
                inference_batch_size=inference_batch_size,
                torch_num_threads=torch_num_threads,
                force=force,
            )
        )
    )


@app.command("predict-fno-competition")
def predict_fno_competition_cmd(
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            predict_fno_competition_command(
                task_id=task_id,
                test_path=test_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                force=force,
            )
        )
    )


@app.command("predict-task1-retrieval-prior")
def predict_task1_retrieval_prior_cmd(
    reference_path: str = typer.Option(...),
    test_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    k: int = typer.Option(20),
    mode: str = typer.Option("affine"),
    leave_one_out: bool = typer.Option(False, "--leave-one-out"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            predict_task1_retrieval_prior_command(
                reference_path=reference_path,
                test_path=test_path,
                output_dir=output_dir,
                dataset_key=dataset_key,
                k=k,
                mode=mode,
                leave_one_out=leave_one_out,
                force=force,
            )
        )
    )


@app.command("run-competition-task-pipeline")
def run_competition_task_pipeline_cmd(
    task_id: int = typer.Option(...),
    model_family: str = typer.Option(...),
    test_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    submission_id: str = typer.Option(...),
    target_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    timing_profile: str | None = typer.Option(None),
    timing_factor: float | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    local_strategy_schedule_path: str | None = typer.Option(None),
    local_strategy_experiment_id: str | None = typer.Option(None),
    local_strategy_leaderboard_path: str | None = typer.Option(None),
    local_strategy_baseline_score: float | None = typer.Option(None),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_competition_task_pipeline_command(
                task_id=task_id,
                model_family=model_family,
                test_path=test_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                submission_id=submission_id,
                target_path=target_path,
                dataset_key=dataset_key,
                timing_profile=timing_profile,
                timing_factor=timing_factor,
                force=force,
                local_strategy_schedule_path=local_strategy_schedule_path,
                local_strategy_experiment_id=local_strategy_experiment_id,
                local_strategy_leaderboard_path=local_strategy_leaderboard_path,
                local_strategy_baseline_score=local_strategy_baseline_score,
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


@app.command("run-fno-training-iteration")
def run_fno_training_iteration_cmd(
    workspace_dir: str = typer.Option(...),
    source_path: str = typer.Option(...),
    memory_path: str | None = typer.Option(None),
    evidence_path: list[str] | None = typer.Option(None, "--evidence-path"),
    dataset_key: str = typer.Option("tensor"),
    train_sample_limit: int = typer.Option(64),
    val_sample_limit: int = typer.Option(32),
    context_steps: int = typer.Option(10),
    time_reduction: int = typer.Option(5),
    space_reduction: int = typer.Option(4),
    batch_size: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    max_candidates: int = typer.Option(3),
    mode: list[int] | None = typer.Option(None, "--mode"),
    width: list[int] | None = typer.Option(None, "--width"),
    depth: list[int] | None = typer.Option(None, "--depth"),
    loss_type: list[str] | None = typer.Option(None, "--loss-type"),
    forecast_steps_value: list[int] | None = typer.Option(None, "--forecast-steps-value"),
    train_steps_value: list[int] | None = typer.Option(None, "--train-steps-value"),
    normalization: list[str] | None = typer.Option(None, "--normalization"),
    target_score: float | None = typer.Option(None),
    patience: int = typer.Option(3),
    checkpoint_every: int = typer.Option(1),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_fno_training_iteration_command(
                workspace_dir=workspace_dir,
                source_path=source_path,
                memory_path=memory_path,
                evidence_paths=evidence_path,
                dataset_key=dataset_key,
                train_sample_limit=train_sample_limit,
                val_sample_limit=val_sample_limit,
                context_steps=context_steps,
                time_reduction=time_reduction,
                space_reduction=space_reduction,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                max_candidates=max_candidates,
                modes=mode,
                widths=width,
                depths=depth,
                loss_types=loss_type,
                forecast_steps_values=forecast_steps_value,
                train_steps_values=train_steps_value,
                normalizations=normalization,
                target_score=target_score,
                patience=patience,
                checkpoint_every=checkpoint_every,
                force=force,
            )
        )
    )


@app.command("build-fno-training-closed-loop-contract")
def build_fno_training_closed_loop_contract_cmd(
    workspace_dir: str = typer.Option(...),
    source_path: str = typer.Option(...),
    output_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    baseline_prediction_path: str | None = typer.Option(None),
    task1_test_path: str | None = typer.Option(None),
    task1_target_path: str | None = typer.Option(None),
    required_command: list[str] | None = typer.Option(None, "--required-command"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_fno_training_closed_loop_contract_command(
                workspace_dir=workspace_dir,
                source_path=source_path,
                output_path=output_path,
                dataset_key=dataset_key,
                baseline_prediction_path=baseline_prediction_path,
                task1_test_path=task1_test_path,
                task1_target_path=task1_target_path,
                required_commands=required_command,
                force=force,
            )
        )
    )


@app.command("check-fno-training-convergence")
def check_fno_training_convergence_cmd(
    memory_path: str = typer.Option(...),
    target_score: float | None = typer.Option(None),
    patience: int = typer.Option(3),
    output_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            check_fno_training_convergence_command(
                memory_path=memory_path,
                target_score=target_score,
                patience=patience,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("build-fno-training-memory")
def build_fno_training_memory_cmd(
    optimization_report_path: str = typer.Option(...),
    iteration_index: int = typer.Option(...),
    previous_memory_path: str | None = typer.Option(None),
    output_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_fno_training_memory_command(
                optimization_report_path=optimization_report_path,
                iteration_index=iteration_index,
                previous_memory_path=previous_memory_path,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("checkpoint-fno-training-loop")
def checkpoint_fno_training_loop_cmd(
    workspace_dir: str = typer.Option(...),
    iteration_index: int = typer.Option(...),
    memory_path: str = typer.Option(...),
    convergence_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            checkpoint_fno_training_loop_command(
                workspace_dir=workspace_dir,
                iteration_index=iteration_index,
                memory_path=memory_path,
                convergence_path=convergence_path,
                force=force,
            )
        )
    )


@app.command("resume-fno-training-loop")
def resume_fno_training_loop_cmd(
    checkpoint_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            resume_fno_training_loop_command(
                checkpoint_path=checkpoint_path,
            )
        )
    )


@app.command("escalate-fno-training-to-hypothesis")
def escalate_fno_training_to_hypothesis_cmd(
    workspace_dir: str = typer.Option(...),
    memory_path: str = typer.Option(...),
    output_path: str | None = typer.Option(None),
    max_strategy_candidates: int = typer.Option(5, "--max-strategy-candidates"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            escalate_fno_training_to_hypothesis_command(
                workspace_dir=workspace_dir,
                memory_path=memory_path,
                output_path=output_path,
                max_strategy_candidates=max_strategy_candidates,
                force=force,
            )
        )
    )


@app.command("integrate-hypothesis-training-feedback")
def integrate_hypothesis_training_feedback_cmd(
    hypothesis_report_path: str = typer.Option(...),
    memory_path: str = typer.Option(...),
    output_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            integrate_hypothesis_training_feedback_command(
                hypothesis_report_path=hypothesis_report_path,
                memory_path=memory_path,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("build-fno-training-final-report")
def build_fno_training_final_report_cmd(
    workspace_dir: str = typer.Option(...),
    memory_path: str = typer.Option(...),
    output_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_fno_training_final_report_command(
                workspace_dir=workspace_dir,
                memory_path=memory_path,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("build-fno-training-health-report")
def build_fno_training_health_report_cmd(
    workspace_dir: str = typer.Option(...),
    output_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_fno_training_health_report_command(
                workspace_dir=workspace_dir,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("run-claude-fno-training-closed-loop")
def run_claude_fno_training_closed_loop_cmd(
    workspace_dir: str = typer.Option(...),
    source_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    max_iterations: int = typer.Option(10),
    target_score: float | None = typer.Option(None),
    timeout_seconds: int = typer.Option(3600),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_claude_fno_training_closed_loop_command(
                workspace_dir=workspace_dir,
                source_path=source_path,
                dataset_key=dataset_key,
                max_iterations=max_iterations,
                target_score=target_score,
                timeout_seconds=timeout_seconds,
                dry_run=dry_run,
                force=force,
            )
        )
    )


@app.command("build-claude-fno-training-closed-loop-task")
def build_claude_fno_training_closed_loop_task_cmd(
    workspace_dir: str = typer.Option(...),
    source_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    max_iterations: int = typer.Option(10),
    target_score: float | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_claude_fno_training_closed_loop_task_command(
                workspace_dir=workspace_dir,
                source_path=source_path,
                dataset_key=dataset_key,
                max_iterations=max_iterations,
                target_score=target_score,
                force=force,
            )
        )
    )


@app.command("build-fno-training-tool-catalog")
def build_fno_training_tool_catalog_cmd(
    output_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_fno_training_tool_catalog_command(
                output_path=output_path,
                force=force,
            )
        )
    )
