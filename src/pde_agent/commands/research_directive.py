
from pathlib import Path

from pde_agent.core.research_directive import (
    DIRECTIVE_STATE_HINT,
    FOLLOWTHROUGH_STATE_HINT,
    build_research_directive,
    evaluate_research_directive_followthrough,
)
from pde_agent.core.json_output import success_payload


def build_agent_research_directive_command(
    *,
    task1_benchmark_path: str,
    candidate_factory_benchmark_path: str,
    output_path: str,
    recent_judge_report_paths: list[str] | None,
    force: bool,
) -> dict[str, object]:
    report = build_research_directive(
        task1_benchmark_path=Path(task1_benchmark_path),
        candidate_factory_benchmark_path=Path(candidate_factory_benchmark_path),
        output_path=Path(output_path),
        recent_judge_report_paths=[
            Path(path) for path in recent_judge_report_paths or []
        ],
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-agent-research-directive",
        state_hint=DIRECTIVE_STATE_HINT,
        artifacts={
            "research_directive_json": output_path,
            "research_directive_path": output_path,
        },
        metrics={
            "decision": report["decision"],
            "research_focus": report["directive"]["research_focus"],
            "candidate_generator": report["directive"]["candidate_generator"],
        },
        next_hint=report["next_command"],
    )


def evaluate_research_directive_followthrough_command(
    *,
    directive_path: str,
    chain_report_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = evaluate_research_directive_followthrough(
        directive_path=Path(directive_path),
        chain_report_path=Path(chain_report_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="evaluate-research-directive-followthrough",
        state_hint=FOLLOWTHROUGH_STATE_HINT,
        artifacts={
            "research_directive_followthrough_json": output_path,
            "followthrough_report_path": output_path,
        },
        metrics={
            "decision": report["decision"],
            "candidate_generator_followed": report["checks"][
                "candidate_generator_followed"
            ]["passed"],
            "audit_clean": report["checks"]["audit_clean"]["passed"],
        },
        next_hint=report["next_command"],
    )
