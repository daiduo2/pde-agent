import json

import typer

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.audit_experiment_config import audit_experiment_config
from pde_agent.commands.build_local_windows import build_local_windows
from pde_agent.commands.build_local_window_suite import build_local_window_suite
from pde_agent.commands.build_submission import build_submission
from pde_agent.commands.check_burgers_physics_local import check_burgers_physics_local
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.compare_fno_candidates import compare_fno_candidates_command
from pde_agent.commands.diagnose_fno_rollout import diagnose_fno_rollout
from pde_agent.commands.diagnose_rollout_error import diagnose_rollout_error
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.evaluate_fno_candidate import evaluate_fno_candidate_command
from pde_agent.commands.execute_fno_finetune_candidate import (
    execute_fno_finetune_candidate_command,
)
from pde_agent.commands.generate_baseline_prediction import generate_baseline_prediction
from pde_agent.commands.generate_experiment_config import generate_experiment_config
from pde_agent.commands.init_run import init_run
from pde_agent.commands.inspect_data import inspect_data
from pde_agent.commands.inspect_fno_checkpoint import inspect_fno_checkpoint
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.predict_fno_local import predict_fno_local
from pde_agent.commands.predict_local_model import predict_local_model
from pde_agent.commands.plan_next_experiment import plan_next_experiment
from pde_agent.commands.plan_next_experiment_from_fno_summary import (
    plan_next_experiment_from_fno_summary_command,
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
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.resolve_config import resolve_config
from pde_agent.commands.run_fno_iteration_dryrun import (
    run_fno_iteration_dryrun_command,
)
from pde_agent.commands.run_fno_iteration import run_fno_iteration_command
from pde_agent.commands.run_fno_evidence_sweep import (
    run_fno_evidence_sweep_command,
)
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
