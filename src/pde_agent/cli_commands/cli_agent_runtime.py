"""CLI commands for agent_runtime domain."""

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
from pde_agent.commands.fno_training_agent_loop import (
    build_claude_fno_training_closed_loop_task_command,
    build_fno_training_tool_catalog_command,
    run_claude_fno_training_closed_loop_command,
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

def register_commands(app: typer.Typer) -> None:
    """Register all commands in this domain."""

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
