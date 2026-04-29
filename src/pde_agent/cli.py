import json

import typer

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.append_strategy_ledger import append_strategy_ledger_command
from pde_agent.commands.audit_hermes_strategy_growth_autopilot import (
    audit_hermes_strategy_growth_autopilot_command,
)
from pde_agent.commands.audit_experiment_config import audit_experiment_config
from pde_agent.commands.build_local_windows import build_local_windows
from pde_agent.commands.build_local_window_suite import build_local_window_suite
from pde_agent.commands.build_deeponet_evidence_sweep import (
    build_deeponet_evidence_sweep_command,
)
from pde_agent.commands.build_competition_submission import (
    build_competition_submission_command,
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
from pde_agent.commands.hermes_strategy_growth_autopilot import (
    hermes_strategy_growth_autopilot_command,
)
from pde_agent.commands.init_run import init_run
from pde_agent.commands.inspect_data import inspect_data
from pde_agent.commands.inspect_fno_checkpoint import inspect_fno_checkpoint
from pde_agent.commands.inspect_unet_checkpoint import inspect_unet_checkpoint
from pde_agent.commands.install_skill_draft import install_skill_draft_command
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.lookup_strategy_skill import lookup_strategy_skill_command
from pde_agent.commands.materialize_skill_draft import materialize_skill_draft_command
from pde_agent.commands.predict_fno_local import predict_fno_local
from pde_agent.commands.predict_unet_local import predict_unet_local
from pde_agent.commands.predict_local_model import predict_local_model
from pde_agent.commands.plan_next_experiment import plan_next_experiment
from pde_agent.commands.plan_next_experiment_from_fno_summary import (
    plan_next_experiment_from_fno_summary_command,
)
from pde_agent.commands.plan_next_spectral_residual_experiment import (
    plan_next_spectral_residual_experiment_command,
)
from pde_agent.commands.plan_spectral_residual_variance_intervention import (
    plan_spectral_residual_variance_intervention_command,
)
from pde_agent.commands.prepare_candidate_execution import (
    prepare_candidate_execution,
)
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
from pde_agent.commands.run_unet_strategy_sweep import run_unet_strategy_sweep_command
from pde_agent.commands.score_local_rollout import score_local_rollout
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
                force=force,
            )
        )
    )
