"""PDE Agent CLI entry point."""

import json
import time

import typer

from pde_agent.cli_commands import (
    cli_agent_runtime,
    cli_burgers,
    cli_candidates,
    cli_cockpit,
    cli_experiments,
    cli_fno_training,
    cli_hypothesis,
    cli_literature,
    cli_local_strategy,
    cli_misc,
    cli_prediction,
    cli_spectral,
    cli_strategy,
    cli_submission,
    cli_task1_training,
)

app = typer.Typer()

# Register all domain commands
cli_agent_runtime.register_commands(app)
cli_burgers.register_commands(app)
cli_candidates.register_commands(app)
cli_cockpit.register_commands(app)
cli_experiments.register_commands(app)
cli_fno_training.register_commands(app)
cli_hypothesis.register_commands(app)
cli_literature.register_commands(app)
cli_local_strategy.register_commands(app)
cli_misc.register_commands(app)
cli_prediction.register_commands(app)
cli_spectral.register_commands(app)
cli_strategy.register_commands(app)
cli_submission.register_commands(app)
cli_task1_training.register_commands(app)

if __name__ == "__main__":
    app()
