import json

import typer

from pde_agent.commands.init_run import init_run


app = typer.Typer(help="CLI-first diagnostic harness for PDE agent experiments.")


@app.command("init-run")
def init_run_cmd(
    run_id: str = typer.Option(...),
    runs_root: str = typer.Option(...),
    json_flag: bool = typer.Option(True, "--json"),
) -> None:
    _ = json_flag
    typer.echo(json.dumps(init_run(run_id=run_id, runs_root=runs_root)))
