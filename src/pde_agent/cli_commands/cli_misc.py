"""CLI commands for misc domain."""

import json
import time

import typer

from pde_agent.commands.meta_research_directive import (
    build_agent_meta_research_directive_command,
)
from pde_agent.commands.research_directive import (
    build_agent_research_directive_command,
    evaluate_research_directive_followthrough_command,
)
from pde_agent.commands.build_local_windows import build_local_windows
from pde_agent.commands.build_local_window_suite import build_local_window_suite
from pde_agent.commands.build_deeponet_evidence_sweep import (
    build_deeponet_evidence_sweep_command,
)
from pde_agent.commands.build_unet_evidence_sweep import (
    build_unet_evidence_sweep_command,
)
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.init_run import init_run
from pde_agent.commands.interactive_cockpit import (
    build_agent_dialogue_request_command,
    build_cockpit_snapshot_command,
    build_fno_training_loop_snapshot_command,
    cockpit_chat_command,
    dispatch_agent_dialogue_request_command,
    run_interactive_cockpit_command,
)
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.profile_unet_runtime import profile_unet_runtime_command
from pde_agent.commands.record_unet_timeout_evidence import (
    record_unet_timeout_evidence_command,
)
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.resolve_config import resolve_config
from pde_agent.commands.run_real_hdf5_baseline_smoke import (
    run_real_hdf5_baseline_smoke_command,
)
from pde_agent.commands.select_hypothesis import select_hypothesis
from pde_agent.commands.select_pde_hypothesis import select_pde_hypothesis
from pde_agent.commands.select_pde_hypothesis_from_fno_summary import (
    select_pde_hypothesis_from_fno_summary_command,
)
from pde_agent.commands.train_baseline import train_baseline

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

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

    @app.command("init-run")
    def init_run_cmd(
        run_id: str = typer.Option(...),
        runs_root: str = typer.Option(...),
        json_flag: bool = typer.Option(True, "--json"),
    ) -> None:
        _ = json_flag
        typer.echo(json.dumps(init_run(run_id=run_id, runs_root=runs_root)))

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
