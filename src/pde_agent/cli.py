import json

import typer

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.append_strategy_ledger import append_strategy_ledger_command
from pde_agent.commands.audit_hermes_strategy_growth_autopilot import (
    audit_hermes_strategy_growth_autopilot_command,
)
from pde_agent.commands.audit_experiment_config import audit_experiment_config
from pde_agent.commands.audit_competition_submission import (
    audit_competition_submission_command,
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
from pde_agent.commands.build_hermes_iteration_memory import (
    build_hermes_iteration_memory_command,
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
from pde_agent.commands.diagnose_hermes_result import diagnose_hermes_result_command
from pde_agent.commands.diagnose_hermes_model_quality import (
    diagnose_hermes_model_quality_command,
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
from pde_agent.commands.generate_hermes_postprocess_candidate import (
    generate_hermes_postprocess_candidate_command,
)
from pde_agent.commands.hermes_strategy_growth_autopilot import (
    hermes_strategy_growth_autopilot_command,
)
from pde_agent.commands.init_run import init_run
from pde_agent.commands.ingest_local_strategy_result import (
    ingest_local_strategy_result_command,
)
from pde_agent.commands.inspect_data import inspect_data
from pde_agent.commands.inspect_fno_checkpoint import inspect_fno_checkpoint
from pde_agent.commands.inspect_unet_checkpoint import inspect_unet_checkpoint
from pde_agent.commands.install_skill_draft import install_skill_draft_command
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.judge_hermes_submission import judge_hermes_submission_command
from pde_agent.commands.lookup_strategy_skill import lookup_strategy_skill_command
from pde_agent.commands.local_strategy_scheduler import (
    claim_next_scheduled_experiment_command,
    schedule_local_strategy_evaluation_command,
)
from pde_agent.commands.local_strategy_leaderboard import (
    record_local_strategy_result_command,
)
from pde_agent.commands.literature_algorithm_lab import (
    adapt_fno_architecture_candidate_command,
    build_blocked_candidate_handoff_command,
    build_executor_taxonomy_command,
    build_literature_source_records_command,
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
from pde_agent.commands.plan_hermes_next_action import plan_hermes_next_action_command
from pde_agent.commands.plan_next_spectral_residual_experiment import (
    plan_next_spectral_residual_experiment_command,
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
from pde_agent.commands.run_hermes_batch_optimization_loop import (
    run_hermes_batch_optimization_loop_command,
)
from pde_agent.commands.run_checkpoint_family_sandbox import (
    run_checkpoint_family_sandbox_command,
)
from pde_agent.commands.compare_hypothesis_lab_memory import (
    compare_hypothesis_lab_memory_command,
)
from pde_agent.commands.run_hypothesis_lab import run_hypothesis_lab_command
from pde_agent.commands.run_hermes_adaptive_heartbeat import (
    run_hermes_adaptive_heartbeat_command,
)
from pde_agent.commands.run_hermes_local_strategy_loop import (
    run_hermes_local_strategy_loop_command,
)
from pde_agent.commands.run_hermes_operator_tuning_loop import (
    run_hermes_operator_tuning_loop_command,
)
from pde_agent.commands.run_hermes_validation_scoring import (
    run_hermes_validation_scoring_command,
)
from pde_agent.commands.run_hermes_model_iteration_loop import (
    run_hermes_model_iteration_loop_command,
)
from pde_agent.commands.run_hermes_convergence_loop import (
    run_hermes_convergence_loop_command,
)
from pde_agent.commands.run_hermes_heartbeat import run_hermes_heartbeat_command
from pde_agent.commands.run_hermes_literature_strategy_expansion import (
    build_local_asset_algorithm_candidates_command,
    record_hermes_literature_candidate_outcome_command,
    run_hermes_literature_strategy_expansion_command,
    summarize_literature_registry_for_hermes_command,
)
from pde_agent.commands.run_hermes_three_stage_policy import (
    run_hermes_three_stage_policy_command,
)
from pde_agent.commands.run_hermes_submission_candidate import (
    run_hermes_submission_candidate_command,
)
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
from pde_agent.commands.train_baseline import train_baseline
from pde_agent.commands.train_spectral_residual_baseline import (
    train_spectral_residual_baseline,
)


app = typer.Typer(help="CLI-first diagnostic harness for PDE agent experiments.")


@app.command("init-run")
def init_run_cmd(
    run_id: str = typer.Option(...),
    runs_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(init_run(run_id=run_id, runs_root=runs_root)))


@app.command("hermes-strategy-growth-autopilot")
def hermes_strategy_growth_autopilot_cmd(
    seed_artifact_path: str = typer.Option(...),
    workspace_dir: str = typer.Option(...),
    proposal_path: str = typer.Option(...),
    rubric_path: str = typer.Option(...),
    skill_name: str = typer.Option(...),
    install_root: str = typer.Option(...),
    registry_path: str = typer.Option(...),
    reviewer_id: str = typer.Option("hermes-autopilot"),
    max_steps: int = typer.Option(32),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            hermes_strategy_growth_autopilot_command(
                seed_artifact_path=seed_artifact_path,
                workspace_dir=workspace_dir,
                proposal_path=proposal_path,
                rubric_path=rubric_path,
                skill_name=skill_name,
                install_root=install_root,
                registry_path=registry_path,
                reviewer_id=reviewer_id,
                max_steps=max_steps,
                force=force,
            )
        )
    )


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


@app.command("run-hermes-local-strategy-loop")
def run_hermes_local_strategy_loop_cmd(
    output_dir: str = typer.Option(...),
    applicable_pde: str = typer.Option(...),
    model_family: str = typer.Option(...),
    failure_modes: list[str] = typer.Option(..., "--failure-mode"),
    max_experiments: int = typer.Option(3),
    min_score_delta: float = typer.Option(1.0),
    max_inference_seconds: float = typer.Option(120.0),
    executor_name: str = typer.Option("dry-run", "--executor"),
    task_id: int | None = typer.Option(None),
    test_path: str | None = typer.Option(None),
    checkpoint_path: str | None = typer.Option(None),
    target_path: str | None = typer.Option(None),
    submission_id: str = typer.Option("team_hermes"),
    dataset_key: str = typer.Option("tensor"),
    baseline_score: float | None = typer.Option(None),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_local_strategy_loop_command(
                output_dir=output_dir,
                applicable_pde=applicable_pde,
                model_family=model_family,
                failure_modes=failure_modes,
                max_experiments=max_experiments,
                min_score_delta=min_score_delta,
                max_inference_seconds=max_inference_seconds,
                executor_name=executor_name,
                task_id=task_id,
                test_path=test_path,
                checkpoint_path=checkpoint_path,
                target_path=target_path,
                submission_id=submission_id,
                dataset_key=dataset_key,
                baseline_score=baseline_score,
                dry_run=dry_run,
                force=force,
            )
        )
    )


@app.command("run-hermes-operator-tuning-loop")
def run_hermes_operator_tuning_loop_cmd(
    output_dir: str = typer.Option(...),
    operator_family: str = typer.Option(...),
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    shard_sizes: list[int] = typer.Option(..., "--shard-size"),
    inference_batch_sizes: list[int] = typer.Option(..., "--inference-batch-size"),
    torch_num_threads_values: list[int] = typer.Option(..., "--torch-num-threads"),
    dataset_key: str = typer.Option("tensor"),
    timeout_seconds: float = typer.Option(120.0),
    full_sample_count: int = typer.Option(1000),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_operator_tuning_loop_command(
                output_dir=output_dir,
                operator_family=operator_family,
                task_id=task_id,
                test_path=test_path,
                checkpoint_path=checkpoint_path,
                shard_sizes=shard_sizes,
                inference_batch_sizes=inference_batch_sizes,
                torch_num_threads_values=torch_num_threads_values,
                dataset_key=dataset_key,
                timeout_seconds=timeout_seconds,
                full_sample_count=full_sample_count,
                force=force,
            )
        )
    )


@app.command("run-hermes-validation-scoring")
def run_hermes_validation_scoring_cmd(
    output_dir: str = typer.Option(...),
    operator_family: str = typer.Option(...),
    task_id: int = typer.Option(...),
    validation_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    submission_id: str = typer.Option("team_hermes_validation"),
    inference_batch_size: int | None = typer.Option(None),
    torch_num_threads: int | None = typer.Option(None),
    sample_limit: int | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_validation_scoring_command(
                output_dir=output_dir,
                operator_family=operator_family,
                task_id=task_id,
                validation_path=validation_path,
                checkpoint_path=checkpoint_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                submission_id=submission_id,
                inference_batch_size=inference_batch_size,
                torch_num_threads=torch_num_threads,
                sample_limit=sample_limit,
                force=force,
            )
        )
    )


@app.command("diagnose-hermes-model-quality")
def diagnose_hermes_model_quality_cmd(
    validation_report_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    min_segment_score: float = typer.Option(60.0),
    min_task_prediction_score: float = typer.Option(45.0),
    max_late_rmse: float = typer.Option(0.25),
    max_inference_seconds: float = typer.Option(120.0),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            diagnose_hermes_model_quality_command(
                validation_report_path=validation_report_path,
                output_path=output_path,
                min_segment_score=min_segment_score,
                min_task_prediction_score=min_task_prediction_score,
                max_late_rmse=max_late_rmse,
                max_inference_seconds=max_inference_seconds,
            )
        )
    )


@app.command("run-hermes-model-iteration-loop")
def run_hermes_model_iteration_loop_cmd(
    output_dir: str = typer.Option(...),
    operator_family: str = typer.Option(...),
    task_id: int = typer.Option(...),
    validation_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    submission_id: str = typer.Option("team_hermes_validation"),
    inference_batch_size: int | None = typer.Option(None),
    torch_num_threads: int | None = typer.Option(None),
    sample_limit: int | None = typer.Option(None),
    min_segment_score: float = typer.Option(60.0),
    min_task_prediction_score: float = typer.Option(45.0),
    max_late_rmse: float = typer.Option(0.25),
    max_inference_seconds: float = typer.Option(120.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_model_iteration_loop_command(
                output_dir=output_dir,
                operator_family=operator_family,
                task_id=task_id,
                validation_path=validation_path,
                checkpoint_path=checkpoint_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                submission_id=submission_id,
                inference_batch_size=inference_batch_size,
                torch_num_threads=torch_num_threads,
                sample_limit=sample_limit,
                min_segment_score=min_segment_score,
                min_task_prediction_score=min_task_prediction_score,
                max_late_rmse=max_late_rmse,
                max_inference_seconds=max_inference_seconds,
                force=force,
            )
        )
    )


@app.command("judge-hermes-submission")
def judge_hermes_submission_cmd(
    submission_dir: str = typer.Option(...),
    output_path: str = typer.Option(...),
    task1_test_path: str | None = typer.Option(None),
    task1_target_path: str | None = typer.Option(None),
    task2_test_path: str | None = typer.Option(None),
    task2_target_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            judge_hermes_submission_command(
                submission_dir=submission_dir,
                output_path=output_path,
                task1_test_path=task1_test_path,
                task1_target_path=task1_target_path,
                task2_test_path=task2_test_path,
                task2_target_path=task2_target_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
            )
        )
    )


@app.command("run-hermes-batch-optimization-loop")
def run_hermes_batch_optimization_loop_cmd(
    output_dir: str = typer.Option(...),
    initial_submission_dir: str = typer.Option(...),
    max_rounds: int = typer.Option(3),
    patience: int = typer.Option(1),
    min_score_improvement: float = typer.Option(1.0),
    submit_score_threshold: float = typer.Option(120.0),
    learning_objective: str = typer.Option("score"),
    task1_test_path: str | None = typer.Option(None),
    task1_target_path: str | None = typer.Option(None),
    task2_test_path: str | None = typer.Option(None),
    task2_target_path: str | None = typer.Option(None),
    unet_checkpoint_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    candidate_submission_dirs: list[str] = typer.Option(
        [],
        "--candidate-submission-dir",
    ),
    candidate_generator: str = typer.Option("none"),
    hypothesis_blueprint_path: str | None = typer.Option(None),
    postprocess_search_mode: str = typer.Option("paired"),
    postprocess_strategies: list[str] = typer.Option(
        [],
        "--postprocess-strategy",
    ),
    postprocess_blend_weights: list[float] = typer.Option(
        [],
        "--postprocess-blend-weight",
    ),
    postprocess_adaptive_step: float = typer.Option(0.125),
    submission_id: str = typer.Option("team_hermes"),
    deadline_seconds: float | None = typer.Option(None),
    checkpoint_every: int = typer.Option(0),
    memory_json_path: str | None = typer.Option(None),
    avoid_repeating: bool = typer.Option(False, "--avoid-repeating"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_batch_optimization_loop_command(
                output_dir=output_dir,
                initial_submission_dir=initial_submission_dir,
                max_rounds=max_rounds,
                patience=patience,
                min_score_improvement=min_score_improvement,
                submit_score_threshold=submit_score_threshold,
                learning_objective=learning_objective,
                task1_test_path=task1_test_path,
                task1_target_path=task1_target_path,
                task2_test_path=task2_test_path,
                task2_target_path=task2_target_path,
                unet_checkpoint_path=unet_checkpoint_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                candidate_submission_dirs=candidate_submission_dirs,
                candidate_generator=candidate_generator,
                hypothesis_blueprint_path=hypothesis_blueprint_path,
                postprocess_search_mode=postprocess_search_mode,
                postprocess_strategies=postprocess_strategies,
                postprocess_blend_weights=postprocess_blend_weights,
                postprocess_adaptive_step=postprocess_adaptive_step,
                submission_id=submission_id,
                deadline_seconds=deadline_seconds,
                checkpoint_every=checkpoint_every,
                memory_json_path=memory_json_path,
                avoid_repeating=avoid_repeating,
                force=force,
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
    output_dir: str = typer.Option(...),
    submission_id: str = typer.Option("team_hermes"),
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
                output_dir=output_dir,
                submission_id=submission_id,
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
    submission_id: str = typer.Option("team_hermes"),
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
    selected_index: int = typer.Option(0),
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
    submission_id: str = typer.Option("team_hermes"),
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
    submission_id: str = typer.Option("team_hermes"),
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


@app.command("run-checkpoint-family-sandbox")
def run_checkpoint_family_sandbox_cmd(
    checkpoint_paths: list[str] = typer.Option(..., "--checkpoint-path"),
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    target_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    submission_id: str = typer.Option("team_hermes"),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    calibration_weights: list[float] = typer.Option([1.0], "--calibration-weight"),
    fno_finetune_steps: int = typer.Option(0),
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
                force=force,
            )
        )
    )


@app.command("generate-hermes-postprocess-candidate")
def generate_hermes_postprocess_candidate_cmd(
    base_submission_dir: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    strategy: str = typer.Option(...),
    blend_weight: float = typer.Option(0.25),
    submission_id: str = typer.Option("team_hermes"),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            generate_hermes_postprocess_candidate_command(
                base_submission_dir=base_submission_dir,
                output_dir=output_dir,
                task_id=task_id,
                test_path=test_path,
                strategy=strategy,
                blend_weight=blend_weight,
                submission_id=submission_id,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                force=force,
            )
        )
    )


@app.command("build-hermes-iteration-memory")
def build_hermes_iteration_memory_cmd(
    batch_loop_report_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    memory_json_path: str = typer.Option(...),
    submission_dir: str | None = typer.Option(None),
    task_ids: list[int] = typer.Option([], "--task-id"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            build_hermes_iteration_memory_command(
                batch_loop_report_path=batch_loop_report_path,
                output_path=output_path,
                memory_json_path=memory_json_path,
                submission_dir=submission_dir,
                task_ids=task_ids,
                force=force,
            )
        )
    )


@app.command("run-hermes-adaptive-heartbeat")
def run_hermes_adaptive_heartbeat_cmd(
    output_dir: str = typer.Option(...),
    initial_submission_dir: str = typer.Option(...),
    verified_blueprint_path: str | None = typer.Option(None),
    max_policy_stages: int = typer.Option(2),
    max_batches: int = typer.Option(1),
    batch_deadline_seconds: float = typer.Option(16800.0),
    max_rounds_per_batch: int = typer.Option(50),
    checkpoint_every: int = typer.Option(5),
    max_stagnant_batches: int = typer.Option(1),
    min_batch_score_improvement: float = typer.Option(1.0),
    min_score_improvement: float = typer.Option(1.0),
    submit_score_threshold: float = typer.Option(120.0),
    learning_objective: str = typer.Option("score"),
    patience: int = typer.Option(10),
    task1_test_path: str | None = typer.Option(None),
    task1_target_path: str | None = typer.Option(None),
    task2_test_path: str | None = typer.Option(None),
    task2_target_path: str | None = typer.Option(None),
    unet_checkpoint_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    initial_candidate_generator: str = typer.Option("postprocess"),
    postprocess_search_mode: str = typer.Option("adaptive"),
    postprocess_strategies: list[str] = typer.Option([], "--postprocess-strategy"),
    postprocess_blend_weights: list[float] = typer.Option(
        [],
        "--postprocess-blend-weight",
    ),
    postprocess_adaptive_step: float = typer.Option(0.125),
    submission_id: str = typer.Option("team_hermes"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_adaptive_heartbeat_command(
                output_dir=output_dir,
                initial_submission_dir=initial_submission_dir,
                verified_blueprint_path=verified_blueprint_path,
                max_policy_stages=max_policy_stages,
                max_batches=max_batches,
                batch_deadline_seconds=batch_deadline_seconds,
                max_rounds_per_batch=max_rounds_per_batch,
                checkpoint_every=checkpoint_every,
                max_stagnant_batches=max_stagnant_batches,
                min_batch_score_improvement=min_batch_score_improvement,
                min_score_improvement=min_score_improvement,
                submit_score_threshold=submit_score_threshold,
                learning_objective=learning_objective,
                patience=patience,
                task1_test_path=task1_test_path,
                task1_target_path=task1_target_path,
                task2_test_path=task2_test_path,
                task2_target_path=task2_target_path,
                unet_checkpoint_path=unet_checkpoint_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                initial_candidate_generator=initial_candidate_generator,
                postprocess_search_mode=postprocess_search_mode,
                postprocess_strategies=postprocess_strategies,
                postprocess_blend_weights=postprocess_blend_weights,
                postprocess_adaptive_step=postprocess_adaptive_step,
                submission_id=submission_id,
                force=force,
            )
        )
    )


@app.command("run-hermes-heartbeat")
def run_hermes_heartbeat_cmd(
    output_dir: str = typer.Option(...),
    initial_submission_dir: str = typer.Option(...),
    max_batches: int = typer.Option(1),
    batch_deadline_seconds: float = typer.Option(16800.0),
    max_rounds_per_batch: int = typer.Option(50),
    checkpoint_every: int = typer.Option(5),
    max_stagnant_batches: int = typer.Option(1),
    min_batch_score_improvement: float = typer.Option(1.0),
    min_score_improvement: float = typer.Option(1.0),
    submit_score_threshold: float = typer.Option(120.0),
    learning_objective: str = typer.Option("score"),
    patience: int = typer.Option(10),
    task1_test_path: str | None = typer.Option(None),
    task1_target_path: str | None = typer.Option(None),
    task2_test_path: str | None = typer.Option(None),
    task2_target_path: str | None = typer.Option(None),
    unet_checkpoint_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    candidate_generator: str = typer.Option("postprocess"),
    hypothesis_blueprint_path: str | None = typer.Option(None),
    postprocess_search_mode: str = typer.Option("adaptive"),
    postprocess_strategies: list[str] = typer.Option(
        [],
        "--postprocess-strategy",
    ),
    postprocess_blend_weights: list[float] = typer.Option(
        [],
        "--postprocess-blend-weight",
    ),
    postprocess_adaptive_step: float = typer.Option(0.125),
    submission_id: str = typer.Option("team_hermes"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_heartbeat_command(
                output_dir=output_dir,
                initial_submission_dir=initial_submission_dir,
                max_batches=max_batches,
                batch_deadline_seconds=batch_deadline_seconds,
                max_rounds_per_batch=max_rounds_per_batch,
                checkpoint_every=checkpoint_every,
                max_stagnant_batches=max_stagnant_batches,
                min_batch_score_improvement=min_batch_score_improvement,
                min_score_improvement=min_score_improvement,
                submit_score_threshold=submit_score_threshold,
                learning_objective=learning_objective,
                patience=patience,
                task1_test_path=task1_test_path,
                task1_target_path=task1_target_path,
                task2_test_path=task2_test_path,
                task2_target_path=task2_target_path,
                unet_checkpoint_path=unet_checkpoint_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                candidate_generator=candidate_generator,
                hypothesis_blueprint_path=hypothesis_blueprint_path,
                postprocess_search_mode=postprocess_search_mode,
                postprocess_strategies=postprocess_strategies,
                postprocess_blend_weights=postprocess_blend_weights,
                postprocess_adaptive_step=postprocess_adaptive_step,
                submission_id=submission_id,
                force=force,
            )
        )
    )


@app.command("run-hermes-literature-strategy-expansion")
def run_hermes_literature_strategy_expansion_cmd(
    algorithm_candidates_path: str = typer.Option(...),
    workspace_dir: str = typer.Option(...),
    checkpoint_paths: list[str] = typer.Option([], "--checkpoint-path"),
    failure_memory_path: str | None = typer.Option(None),
    registry_summary_path: str | None = typer.Option(None),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_literature_strategy_expansion_command(
                algorithm_candidates_path=algorithm_candidates_path,
                workspace_dir=workspace_dir,
                checkpoint_paths=checkpoint_paths,
                failure_memory_path=failure_memory_path,
                registry_summary_path=registry_summary_path,
                force=force,
            )
        )
    )


@app.command("record-hermes-literature-candidate-outcome")
def record_hermes_literature_candidate_outcome_cmd(
    algorithm_candidate_id: str = typer.Option(...),
    candidate_key: str = typer.Option(...),
    outcome: str = typer.Option(...),
    reason_codes: list[str] = typer.Option([], "--reason-code"),
    output_path: str = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            record_hermes_literature_candidate_outcome_command(
                algorithm_candidate_id=algorithm_candidate_id,
                candidate_key=candidate_key,
                outcome=outcome,
                reason_codes=reason_codes,
                output_path=output_path,
                force=force,
            )
        )
    )


@app.command("summarize-literature-registry-for-hermes")
def summarize_literature_registry_for_hermes_cmd(
    registry_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    current_best_score: float = typer.Option(...),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            summarize_literature_registry_for_hermes_command(
                registry_path=registry_path,
                output_path=output_path,
                current_best_score=current_best_score,
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


@app.command("run-hermes-three-stage-policy")
def run_hermes_three_stage_policy_cmd(
    output_dir: str = typer.Option(...),
    initial_submission_dir: str = typer.Option(...),
    task1_test_path: str | None = typer.Option(None),
    task1_target_path: str | None = typer.Option(None),
    task2_test_path: str | None = typer.Option(None),
    task2_target_path: str | None = typer.Option(None),
    unet_checkpoint_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    submission_id: str = typer.Option("team_hermes"),
    stage_deadline_seconds: float = typer.Option(600.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_three_stage_policy_command(
                output_dir=output_dir,
                initial_submission_dir=initial_submission_dir,
                task1_test_path=task1_test_path,
                task1_target_path=task1_target_path,
                task2_test_path=task2_test_path,
                task2_target_path=task2_target_path,
                unet_checkpoint_path=unet_checkpoint_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
                submission_id=submission_id,
                stage_deadline_seconds=stage_deadline_seconds,
                force=force,
            )
        )
    )


@app.command("diagnose-hermes-result")
def diagnose_hermes_result_cmd(
    artifact_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    max_inference_seconds: float = typer.Option(120.0),
    min_score_delta: float = typer.Option(1.0),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            diagnose_hermes_result_command(
                artifact_path=artifact_path,
                output_path=output_path,
                max_inference_seconds=max_inference_seconds,
                min_score_delta=min_score_delta,
            )
        )
    )


@app.command("plan-hermes-next-action")
def plan_hermes_next_action_cmd(
    diagnosis_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            plan_hermes_next_action_command(
                diagnosis_path=diagnosis_path,
                output_path=output_path,
            )
        )
    )


@app.command("run-hermes-convergence-loop")
def run_hermes_convergence_loop_cmd(
    initial_artifact_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    max_steps: int = typer.Option(3),
    max_inference_seconds: float = typer.Option(120.0),
    min_score_delta: float = typer.Option(1.0),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_convergence_loop_command(
                initial_artifact_path=initial_artifact_path,
                output_dir=output_dir,
                max_steps=max_steps,
                max_inference_seconds=max_inference_seconds,
                min_score_delta=min_score_delta,
                force=force,
            )
        )
    )


@app.command("run-hermes-submission-candidate")
def run_hermes_submission_candidate_cmd(
    tuning_report_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    task_id: int = typer.Option(...),
    test_path: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    submission_id: str = typer.Option(...),
    target_path: str | None = typer.Option(None),
    dataset_key: str = typer.Option("tensor"),
    prediction_dataset_key: str = typer.Option("tensor"),
    force: bool = typer.Option(False, "--force"),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            run_hermes_submission_candidate_command(
                tuning_report_path=tuning_report_path,
                output_dir=output_dir,
                task_id=task_id,
                test_path=test_path,
                checkpoint_path=checkpoint_path,
                submission_id=submission_id,
                target_path=target_path,
                dataset_key=dataset_key,
                prediction_dataset_key=prediction_dataset_key,
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


@app.command("audit-hermes-strategy-growth-autopilot")
def audit_hermes_strategy_growth_autopilot_cmd(
    autopilot_report_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(
        json.dumps(
            audit_hermes_strategy_growth_autopilot_command(
                autopilot_report_path=autopilot_report_path,
                output_path=output_path,
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
