import json

import typer

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.build_submission import build_submission
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.init_run import init_run
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.resolve_config import resolve_config
from pde_agent.commands.select_hypothesis import select_hypothesis
from pde_agent.commands.train_baseline import train_baseline


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
