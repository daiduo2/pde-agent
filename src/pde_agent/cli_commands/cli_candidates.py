"""CLI commands for candidates domain."""

import json
import time

import typer

from pde_agent.commands.research_directive import (
    build_agent_research_directive_command,
    evaluate_research_directive_followthrough_command,
)
from pde_agent.commands.compare_fno_candidates import compare_fno_candidates_command
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
from pde_agent.commands.prepare_candidate_execution import (
    prepare_candidate_execution,
)
from pde_agent.commands.compare_hypothesis_lab_memory import (
    compare_hypothesis_lab_memory_command,
)

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

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
